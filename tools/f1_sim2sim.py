#!/usr/bin/env python3
"""MuJoCo sim2sim for a PartyMimic F1 29DOF policy exported with tools/export_onnx.py.

The ONNX bakes one reference motion in and takes
    obs_history (1, H, O)   raw observations, oldest first, MDP.obs layout:
                            q - q0, qd, root angular velocity (root frame), projected gravity,
                            previous action, reference-relative root orientation (6D)
    time_step   (1, 1)     reference frame at 50 Hz
and returns the action plus the reference state at that frame.

This script rebuilds the same physics in plain MuJoCo (CPU): the F1 training MJCF with a position
actuator per joint (kp/kd/effort/armature from the ONNX metadata, i.e. the gains PartyMimic trained
with), the same 0.005 s physics step and 0.02 s control step, and rolls the policy out from the
first frame of the motion. It is a true sim2sim: no mjlab, no MuJoCo-Warp, no world model.

Metrics (per clip):
  success          played to the last frame without falling (base z > 0.25 m, tilt < 0.8 rad)
  joint_err_rad    mean |q_robot - q_ref| over the played frames
  body_err_m       mean ||p_robot - p_ref|| over the tracked bodies, world frame
  body_err_local_m the leaderboard's local body position error: the reference rotated by the
                   relative pelvis heading of robot and reference
  root_err_m       mean ||p_robot - p_ref|| of the pelvis
  min_base_z       lowest pelvis height
  foot_slip_m_s    mean horizontal speed of a foot while it is in contact

Usage:
  python tools/f1_sim2sim.py --onnx output/f1_dance_s0_dance1_subject3_255_706.onnx \
      --mjcf assets/f1/f1_party.xml --out eval/sim2sim.npz --json eval/sim2sim.json
"""
from __future__ import annotations

import argparse
import json
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover
    ort = None

DECIMATION = 4  # PartyMimic: PHYSICS_DT = CONTROL_DT / 4


def metadata(onnx_path: str) -> dict:
    import onnx
    m = onnx.load(onnx_path)
    return {p.key: p.value for p in m.metadata_props}


def build_model(mjcf: Path, meta: dict):
    """f1_party.xml + one position actuator per joint, with the trained kp/kd/effort/armature."""
    names = meta["joint_names"].split(",")
    kp = [float(x) for x in meta["joint_stiffness"].split(",")]
    kd = [float(x) for x in meta["joint_damping"].split(",")]
    tree = ET.parse(mjcf)
    root = tree.getroot()
    act = ET.SubElement(root, "actuator")
    for n, p, d in zip(names, kp, kd):
        ET.SubElement(act, "general", name=f"motor_{n}", joint=n,
                      gainprm=f"{p} 0 0", biasprm=f"0 {-p} {-d}",
                      ctrlrange="-1e6 1e6", forcerange="-1e6 1e6", ctrllimited="true")
    tmp = tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False, dir=str(mjcf.parent))
    tree.write(tmp.name)
    tmp.close()
    model = mujoco.MjModel.from_xml_path(tmp.name)
    Path(tmp.name).unlink()
    # armature is a dof attribute: set it from the actuator metadata
    for i, n in enumerate(names):
        jid = model.joint(n).id
        model.dof_armature[model.jnt_dofadr[jid]] = 0.0
    return model, names


def quat_to_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def mat_to_6d(R):
    return np.concatenate((R[:, 0], R[:, 1]))


def yaw_mat(R):
    """Rotation about z by the pelvis heading (PartyMimic rotations.yaw_mat)."""
    yaw = np.arctan2(R[1, 0], R[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--mjcf", default="assets/f1/f1_party.xml")
    ap.add_argument("--out", default=None)
    ap.add_argument("--json", default=None)
    ap.add_argument("--video", default=None, help="mp4 (needs --mjcf_vis with meshes)")
    ap.add_argument("--mjcf_vis", default=None)
    ap.add_argument("--w", type=int, default=960)
    ap.add_argument("--h", type=int, default=720)
    args = ap.parse_args()
    assert ort is not None, "onnxruntime is required"

    meta = metadata(args.onnx)
    names = meta["joint_names"].split(",")
    kp = [float(x) for x in meta["joint_stiffness"].split(",")]
    kd = [float(x) for x in meta["joint_damping"].split(",")]
    q0 = np.array([float(x) for x in meta["default_joint_pos"].split(",")])
    alpha = np.array([float(x) for x in meta["action_scale"].split(",")])
    H = int(meta["observation_history"])
    dt = float(meta["control_dt"])
    T = int(meta["motion_num_frames"])
    tracked = meta["body_names"].split(",")

    model, jnames = build_model(Path(args.mjcf), meta)
    data = mujoco.MjData(model)
    opt = model.opt
    opt.timestep = dt / DECIMATION
    opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    opt.iterations = 100
    opt.ls_iterations = 50
    opt.tolerance = 1e-6
    opt.ccd_iterations = 35
    opt.disableflags = 0

    jid = [model.joint(n).id for n in jnames]
    qadr = np.array([model.jnt_qposadr[j] for j in jid])
    dadr = np.array([model.jnt_dofadr[j] for j in jid])
    bid = [model.body(n).id for n in tracked]
    root_bid = model.body(meta["anchor_body_name"]).id
    ctrl_adr = np.array([model.actuator(f"motor_{n}").id for n in jnames])

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    obs_dim = len(jnames) * 2 + 3 + 3 + len(jnames) + 6
    obs_hist = np.zeros((1, H, obs_dim), np.float32)
    t = np.zeros((1, 1), np.float32)

    # reference at t=0 gives the reset state
    out0 = sess.run(None, {"obs_history": obs_hist, "time_step": t})
    a0, jq, qd, bp, bq, bv, bw, _ = out0
    jq, qd, bp, bq = jq[0], qd[0], bp[0], bq[0]
    bv, bw = bv[0], bw[0]

    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = bp[0]
    data.qpos[3:7] = bq[0]
    data.qpos[qadr] = jq
    R0 = quat_to_mat(bq[0])
    data.qvel[0:3] = bv[0]
    data.qvel[3:6] = R0.T @ bw[0]  # MuJoCo free-joint angular velocity is in the body frame
    data.qvel[dadr] = qd
    mujoco.mj_forward(model, data)

    def obs_of(prev_action):
        R = quat_to_mat(data.qpos[3:7])
        o = np.concatenate((data.qpos[qadr] - q0, data.qvel[dadr], R.T @ data.qvel[3:6],
                            -R[:, 2], prev_action, mat_to_6d(R.T @ R_ref)))
        return o

    # reference root orientation for the observation: from the ONNX at the current frame
    R_ref = quat_to_mat(bq[0])
    prev_a = np.zeros(len(jnames))
    o = obs_of(prev_a)
    obs_hist[:] = o

    q_hist, p_hist, r_hist, rq_hist = [], [], [], []
    ref_p, ref_q, ref_rq = [], [], []
    a_hist = []
    fell, end = False, T
    for f in range(T):
        out = sess.run(None, {"obs_history": obs_hist, "time_step": np.array([[float(f)]], np.float32)})
        a, jq, qd, bp, bq, bv, bw, nq = out
        a = a[0]
        R_ref = quat_to_mat(bq[0][0])
        q_des = q0 + alpha * a
        data.ctrl[ctrl_adr] = q_des
        for _ in range(DECIMATION):
            mujoco.mj_step(model, data)
        # record
        q_hist.append(data.qpos[qadr].copy())
        p_hist.append(data.xpos[bid].copy())
        r_hist.append(np.array([data.xmat[b].reshape(3, 3).copy() for b in bid]))
        rq_hist.append(data.qpos[3:7].copy())
        ref_p.append(bp[0].copy())
        ref_q.append(jq[0].copy())
        ref_rq.append(bq[0][0].copy())
        a_hist.append(a.copy())
        z = data.qpos[2]
        R = quat_to_mat(data.qpos[3:7])
        tilt = np.arccos(np.clip(R[2, 2], -1, 1))
        if z < 0.25 or tilt > 0.8:
            fell, end = True, f + 1
            break
        R_ref_next = quat_to_mat(nq[0])
        o = np.concatenate((data.qpos[qadr] - q0, data.qvel[dadr], R.T @ data.qvel[3:6],
                            -R[:, 2], a, mat_to_6d(R.T @ R_ref_next)))
        obs_hist = np.concatenate((obs_hist[:, 1:], o[None, None]), 1)

    q_hist = np.array(q_hist)
    p_hist = np.array(p_hist)
    r_hist = np.array(r_hist)
    rq_hist = np.array(rq_hist)
    ref_p = np.array(ref_p)
    ref_q = np.array(ref_q)
    ref_rq = np.array(ref_rq)
    n = len(q_hist)

    # metrics
    joint_err = float(np.abs(q_hist - ref_q[:n]).mean())
    root_p = p_hist[:, 0]
    root_err = float(np.linalg.norm(root_p - ref_p[:n, 0], axis=1).mean())
    body_err = float(np.linalg.norm(p_hist - ref_p[:n], axis=2).mean())
    # leaderboard-style local body position error: both in their own pelvis heading frame
    local_err = np.zeros(n)
    for i in range(n):
        Hr = yaw_mat(r_hist[i, 0]).T
        Hf = yaw_mat(quat_to_mat(ref_rq[i])).T
        pr = np.einsum("ij,bj->bi", Hr, p_hist[i] - root_p[i])
        pf = np.einsum("ij,bj->bi", Hf, ref_p[i] - ref_p[i, 0])
        local_err[i] = np.linalg.norm(pr - pf, axis=1).mean()

    res = {
        "onnx": args.onnx,
        "frames_played": int(n),
        "frames_total": int(T),
        "success": bool(not fell),
        "fell_at": (int(end) if fell else None),
        "joint_err_rad": joint_err,
        "root_err_m": root_err,
        "body_err_m": body_err,
        "body_err_local_m": float(local_err.mean()),
        "min_base_z": float(min(p_hist[:, 0, 2].min(), 9e9)) if n else None,
    }
    print(json.dumps(res, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.out, q=q_hist, body_pos=p_hist, body_R=r_hist, ref_p=ref_p[:n], ref_q=ref_q[:n],
                 actions=np.array(a_hist), joint_names=np.array(jnames), body_names=np.array(tracked),
                 root_quat=rq_hist, ref_root_quat=ref_rq[:n])
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
