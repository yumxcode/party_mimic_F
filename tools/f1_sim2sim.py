#!/usr/bin/env python3
"""MuJoCo sim2sim for a PartyMimic F1 29DOF policy exported with tools/export_onnx.py.

The ONNX bakes one reference motion in and takes
    obs_history (1, H, O)   raw observations, oldest first, partymimic.mdp.MDP.obs layout:
                            q - q0, qd, root angular velocity (root frame), projected gravity
                            (= R^T (0,0,-1) = -R[2, :]), previous action, reference-relative root
                            orientation (6D, first two columns of R^T R_ref)
    time_step   (1, 1)     reference frame at 50 Hz
and returns the action, the reference state at that frame and the reference root orientation at
the next frame (what the next observation's relative orientation is taken against).

This script rebuilds the training physics in plain MuJoCo (CPU), from the same sources PartyMimic
compiled (mjlab 1.6.0 + the F1 entity in src/partymimic/f1/f1_constants.py):
  * f1_party.xml (joint damping / frictionloss / limits / inertias / 8 foot spheres); every
    colliding geom gets contype=0 conaffinity=1 as the entity's get_spec does (no self collision);
  * mjlab's flat ground plane (default contact properties);
  * one actuator per joint exactly as mjlab's BuiltinPositionActuatorCfg compiles it: gainprm=kp,
    biasprm=(0,-kp,-kd), ctrllimited off, forcerange=+-effort_limit, and the joint armature override
    (f1_constants LEG/UPPER tables);
  * MujocoCfg of PartyMimic's make_sim: dt 0.005 s, implicitfast, pyramidal cone, Newton 100
    iterations / 50 line-search iterations, tolerance 1e-6, ccd 35, 4 substeps per 0.02 s control step.
The policy is rolled out from frame 0 (root pose, joint positions and root velocities of the reference,
a_prev = 0, target = q0 + action_scale * a as in env.step_core without randomization / lag).

Frame alignment (PartyMimic's own evaluate.py): at frame f the robot state is recorded against the
reference frame f *before* the step taken at f; the final state after the last step is compared with
the last frame. A fall (A1) is base z < 0.25 m or tilt > 0.8 rad at any recorded frame.

Metrics (per clip):
  success          played to the last frame without falling
  joint_err_rad    mean |q_robot - q_ref| over the played frames (A3)
  body_err_local_m mean tracked-body position error with robot and reference each in their own pelvis
                   heading frame (A2); body_err_m / root_err_m are the world-frame numbers for reference
  min_base_z, max_tilt_rad
The rollout is also written as a PartyMimic-format motion npz (<out stem>_motion.npz: body_pos_w,
body_quat_w, joint_pos, fps=50) so tools/check_motion_f1.py recomputes the foot-flatness and
hip-roll-adduction gates (A4/A5) on what the policy actually did.

--oracle_npz <motion.npz> replaces the policy by a PD replay of the reference joint angles through the
same physics/observation path. It is only a pipeline check (model build, obs layout, frame alignment,
motion npz): with the project's kp/kd the F1 cannot even hold its default pose under pure PD
(ankle kp 60/45 N·m/rad vs the ~216 N·m/rad toppling stiffness, see F1_29DOF/f1_kp_kd.json
"pd_free_stand"), so the oracle falls and is NOT an upper bound on the policy metrics.

Usage:
  python tools/f1_sim2sim.py --onnx eval/f1_dance_v1_dance1_subject3_255_706.onnx \
      --mjcf assets/f1/f1_party.xml --out eval/dance1_subject3.npz --json eval/dance1_subject3.json
"""
from __future__ import annotations

import argparse
import ast
import json
import re
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
REPO = Path(__file__).resolve().parent.parent
F1_CONSTANTS = REPO / "src" / "partymimic" / "f1" / "f1_constants.py"


def f1_tables() -> dict:
    """LEG / UPPER = joint -> (kp, kd, effort, armature) and F1_JOINT_NAMES, read as literals from the
    entity file (the module itself imports mjlab, which this replay deliberately does not)."""
    tree = ast.parse(F1_CONSTANTS.read_text())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in ("LEG", "UPPER", "F1_JOINT_NAMES"):
                out[node.targets[0].id] = ast.literal_eval(node.value)
    assert {"LEG", "UPPER", "F1_JOINT_NAMES"} <= set(out), out.keys()
    return out


def gains(tables: dict, joint: str):
    key = joint.replace("left_", "").replace("right_", "").replace("_joint", "")
    return tables["LEG"][key] if key in tables["LEG"] else tables["UPPER"][key]


def build_model(mjcf: Path, names: list[str], kp: list[float], kd: list[float], effort: list[float],
                armature: list[float]):
    """f1_party.xml + mjlab's ground plane + mjlab-style position actuators, as PartyMimic compiled them."""
    tree = ET.parse(mjcf)
    root = tree.getroot()
    for geom in root.iter("geom"):
        if int(geom.get("contype", "1")) or int(geom.get("conaffinity", "1")):
            geom.set("contype", "0")
            geom.set("conaffinity", "1")  # the entity's get_spec: colliding geoms only touch the ground
    world = root.find("worldbody")
    terrain = ET.SubElement(world, "body", name="terrain")
    ET.SubElement(terrain, "geom", name="terrain", type="plane", size="0 0 0.01")
    act = ET.SubElement(root, "actuator")
    for n, p, d, e in zip(names, kp, kd, effort):
        ET.SubElement(act, "general", name=f"motor_{n}", joint=n,
                      gaintype="fixed", gainprm=f"{p} 0 0", biastype="affine", biasprm=f"0 {-p} {-d}",
                      ctrllimited="false", forcelimited="true", forcerange=f"{-e} {e}")
    tmp = tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False, dir=str(mjcf.parent))
    tree.write(tmp.name)
    tmp.close()
    model = mujoco.MjModel.from_xml_path(tmp.name)
    Path(tmp.name).unlink()
    for i, n in enumerate(names):
        jid = model.joint(n).id
        model.dof_armature[model.jnt_dofadr[jid]] = armature[i]
    return model


def quat_to_mat(q):
    q = np.asarray(q, float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def mat_to_quat_wxyz(R):
    """PartyMimic rotations.mat_to_quat (wxyz, w >= 0)."""
    m = R
    qw = np.sqrt(max(1 + m[0, 0] + m[1, 1] + m[2, 2], 1e-10)) / 2
    qx = np.sqrt(max(1 + m[0, 0] - m[1, 1] - m[2, 2], 1e-10)) / 2
    qy = np.sqrt(max(1 - m[0, 0] + m[1, 1] - m[2, 2], 1e-10)) / 2
    qz = np.sqrt(max(1 - m[0, 0] - m[1, 1] + m[2, 2], 1e-10)) / 2
    cands = np.array([
        [qw, (m[2, 1] - m[1, 2]) / (4 * qw), (m[0, 2] - m[2, 0]) / (4 * qw), (m[1, 0] - m[0, 1]) / (4 * qw)],
        [(m[2, 1] - m[1, 2]) / (4 * qx), qx, (m[0, 1] + m[1, 0]) / (4 * qx), (m[0, 2] + m[2, 0]) / (4 * qx)],
        [(m[0, 2] - m[2, 0]) / (4 * qy), (m[0, 1] + m[1, 0]) / (4 * qy), qy, (m[1, 2] + m[2, 1]) / (4 * qy)],
        [(m[1, 0] - m[0, 1]) / (4 * qz), (m[0, 2] + m[2, 0]) / (4 * qz), (m[1, 2] + m[2, 1]) / (4 * qz), qz],
    ])
    q = cands[np.argmax(np.abs(cands).max(axis=1))]
    return q / np.linalg.norm(q)


def quat_to_rotvec(q):
    q = q if q[0] >= 0 else -q
    v = q[1:]
    s = np.linalg.norm(v)
    if s < 1e-9:
        return 2 * v
    ang = 2 * np.arctan2(s, q[0])
    return v / s * ang


def so3_log(R):
    return quat_to_rotvec(mat_to_quat_wxyz(R))


def gradient(x, dt):
    """motion._gradient: central differences, one-sided at the ends."""
    g = np.zeros_like(x)
    g[1:-1] = (x[2:] - x[:-2]) / (2 * dt)
    g[0] = (x[1] - x[0]) / dt
    g[-1] = (x[-1] - x[-2]) / dt
    return g


def mat_to_6d(R):
    return np.concatenate((R[:, 0], R[:, 1]))


def yaw_mat(R):
    """Rotation about z by the pelvis heading (PartyMimic rotations.yaw_mat: projected x axis)."""
    yaw = np.arctan2(R[1, 0], R[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def apply_mujoco_cfg(model):
    """PartyMimic env.make_sim -> mjlab MujocoCfg."""
    opt = model.opt
    opt.timestep = 0.02 / DECIMATION
    opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    opt.jacobian = mujoco.mjtJacobian.mjJAC_AUTO
    opt.impratio = 1.0
    opt.iterations = 100
    opt.tolerance = 1e-6
    opt.ls_iterations = 50
    opt.ls_tolerance = 0.01
    opt.ccd_iterations = 35
    opt.gravity[:] = (0.0, 0.0, -9.81)
    opt.disableflags = 0
    opt.enableflags = 0


class OnnxPolicy:
    """The exported PartyMimic policy + baked reference (tools/export_onnx.py)."""

    def __init__(self, path: str):
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.meta = dict(self.sess.get_modelmeta().custom_metadata_map)
        self.name = Path(path).name

    def query(self, obs_hist, f):
        a, jq, qd, bp, bq, bv, bw, nq = self.sess.run(
            None, {"obs_history": obs_hist, "time_step": np.array([[float(f)]], np.float32)})
        return a[0], jq[0], qd[0], bp[0], bq[0], bv[0], bw[0], nq[0]


class OraclePolicy:
    """PD replay of a PartyMimic motion npz: the action targets the reference joint angles of the frame
    (clamp +-20 as in train.py), the reference bodies come from MuJoCo FK of the reference poses."""

    def __init__(self, path: str, mjcf: Path, names: list[str], q0, alpha, tracked: list[str]):
        d = np.load(path)
        self.meta = {"motion_num_frames": str(len(d["joint_pos"])), "control_dt": "0.02", "motion": Path(path).stem}
        self.name = Path(path).name
        self.T = len(d["joint_pos"])
        self.jq = d["joint_pos"].astype(float)
        self.q0, self.alpha = q0, alpha
        dt = 1.0 / float(np.asarray(d["fps"]).reshape(-1)[0])
        self.rp = d["body_pos_w"][:, 0].astype(float)
        self.rq = d["body_quat_w"][:, 0].astype(float)
        self.rR = np.array([quat_to_mat(q) for q in self.rq])
        self.rv = gradient(self.rp, dt)
        w = np.zeros_like(self.rp)
        w[1:-1] = np.array([so3_log(self.rR[i + 1] @ self.rR[i - 1].T) for i in range(1, self.T - 1)]) / (2 * dt)
        w[0] = so3_log(self.rR[1] @ self.rR[0].T) / dt
        w[-1] = so3_log(self.rR[-1] @ self.rR[-2].T) / dt
        self.rw = w
        self.qd = gradient(self.jq, dt)
        m = mujoco.MjModel.from_xml_path(str(mjcf))
        data = mujoco.MjData(m)
        qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in names])
        bid = [m.body(n).id for n in tracked]
        self.bp = np.zeros((self.T, len(tracked), 3))
        for f in range(self.T):
            mujoco.mj_resetData(m, data)
            data.qpos[0:3] = self.rp[f]
            data.qpos[3:7] = self.rq[f]
            data.qpos[qadr] = self.jq[f]
            mujoco.mj_forward(m, data)
            self.bp[f] = data.xpos[bid]
        self.bq_all = np.repeat(self.rq[:, None, :], len(tracked), axis=1)

    def query(self, obs_hist, f):
        a = np.clip((self.jq[f] - self.q0) / self.alpha, -20.0, 20.0)
        B = self.bp.shape[1]
        return (a, self.jq[f], self.qd[f], self.bp[f], self.bq_all[f], np.repeat(self.rv[f][None], B, 0),
                np.repeat(self.rw[f][None], B, 0), self.rq[min(f + 1, self.T - 1)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", default=None)
    ap.add_argument("--oracle_npz", default=None, help="sanity check: PD replay of this motion instead of a policy")
    ap.add_argument("--mjcf", default=str(REPO / "assets/f1/f1_party.xml"))
    ap.add_argument("--out", default=None, help="rollout npz (also writes <stem>_motion.npz for check_motion_f1.py)")
    ap.add_argument("--json", default=None)
    ap.add_argument("--max_frames", type=int, default=0, help="debug: stop after this many frames (0: whole clip)")
    ap.add_argument("--kinematic", action="store_true",
                    help="self-check: teleport the robot to the reference every frame instead of stepping (A2/A3 ~ 0, "
                         "A4/A5 = the reference's gates; exercises the full-length frame alignment and the motion npz)")
    args = ap.parse_args()
    assert args.onnx or args.oracle_npz, "--onnx or --oracle_npz"

    tables = f1_tables()
    names = list(tables["F1_JOINT_NAMES"])
    g = [gains(tables, n) for n in names]
    kp, kd, effort, armature = [x[0] for x in g], [x[1] for x in g], [x[2] for x in g], [x[3] for x in g]
    q0 = np.zeros(len(names))
    for k, v in _default_joint_pos_dict().items():
        for i, n in enumerate(names):
            if re.fullmatch(k, n):
                q0[i] = v
    alpha = np.array([0.25 * e / p for e, p in zip(effort, kp)])
    tracked = list(_tracked_bodies())

    if args.onnx:
        pol = OnnxPolicy(args.onnx)
        meta = pol.meta
        assert meta["joint_names"].split(",") == names, "ONNX joint order differs from f1_constants"
        kp_o = [float(x) for x in meta["joint_stiffness"].split(",")]
        kd_o = [float(x) for x in meta["joint_damping"].split(",")]
        assert np.allclose(kp_o, kp) and np.allclose(kd_o, kd), "ONNX kp/kd differ from f1_constants"
        assert np.allclose([float(x) for x in meta["default_joint_pos"].split(",")], q0), "default pose differs"
        assert np.allclose([float(x) for x in meta["action_scale"].split(",")], alpha), "action scale differs"
        assert meta["body_names"].split(",") == tracked, "tracked bodies differ"
        H = int(meta["observation_history"])
    else:
        pol = OraclePolicy(args.oracle_npz, Path(args.mjcf), names, q0, alpha, tracked)
        meta = pol.meta
        H = 10
    dt = float(meta["control_dt"])
    T = int(meta["motion_num_frames"])
    if args.max_frames:
        T = min(T, args.max_frames)
    assert abs(dt - 0.02) < 1e-9, dt

    model = build_model(Path(args.mjcf), names, kp, kd, effort, armature)
    apply_mujoco_cfg(model)
    data = mujoco.MjData(model)

    jid = [model.joint(n).id for n in names]
    qadr = np.array([model.jnt_qposadr[j] for j in jid])
    dadr = np.array([model.jnt_dofadr[j] for j in jid])
    bid = [model.body(n).id for n in tracked]
    ctrl_adr = np.array([model.actuator(f"motor_{n}").id for n in names])

    obs_dim = len(names) * 2 + 3 + 3 + len(names) + 6
    obs_hist = np.zeros((1, H, obs_dim), np.float32)

    def obs_of(prev_action, R_ref):
        """partymimic.mdp.MDP.obs (joint interface, xi = 0): the free joint's angular velocity is in the
        root frame (env.state: w_world = R @ qvel[3:6]; obs uses R^T w_world)."""
        R = quat_to_mat(data.qpos[3:7])
        return np.concatenate((data.qpos[qadr] - q0, data.qvel[dadr], data.qvel[3:6],
                               -R[2, :], prev_action, mat_to_6d(R.T @ R_ref))).astype(np.float32)

    def reference_at(f: int):
        """The baked reference state of frame f (the bundled action is discarded: it is only valid for
        the observation history of that frame, which is built after the step)."""
        return pol.query(obs_hist, f)[1:]

    def action_at(f: int, R_ref, prev_a):
        """The policy's action for frame f from the current observation history. Guard: the last entry
        must be the observation of the *current* robot state at frame f (a stale history silently
        degrades an ONNX policy; the oracle ignores the history, so only this check catches it)."""
        o = obs_hist[0, -1]
        R = quat_to_mat(data.qpos[3:7])
        expect = np.concatenate((data.qpos[qadr] - q0, data.qvel[dadr], data.qvel[3:6], -R[2, :], prev_a,
                                 mat_to_6d(R.T @ R_ref))).astype(np.float32)
        assert np.allclose(o, expect, atol=3e-5), f"stale observation history at frame {f}: max diff {np.abs(o - expect).max():.2e}"
        return pol.query(obs_hist, f)[0]

    # reset on reference frame 0 (env.reset without noise): root pose, root velocities, joints, a_prev = 0
    jq, qd, bp, bq, bv, bw, nq = reference_at(0)
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = bp[0]
    data.qpos[3:7] = bq[0] / np.linalg.norm(bq[0])
    data.qpos[qadr] = jq
    R0 = quat_to_mat(bq[0])
    data.qvel[0:3] = bv[0]               # free joint: world-frame linear velocity
    data.qvel[3:6] = R0.T @ bw[0]        # ... and root-frame angular velocity
    data.qvel[dadr] = qd
    data.ctrl[ctrl_adr] = jq             # env.reset: set_joint_position_target(q)
    mujoco.mj_forward(model, data)
    obs_hist[:] = obs_of(np.zeros(len(names)), R0)
    a = action_at(0, R0, np.zeros(len(names)))  # first action from the reset observation (evaluate.py: hist = o repeated)
    first_obs, first_action = obs_hist[0, -1].copy(), a.copy()

    q_rec, p_rec, r_rec, rq_rec, pos_rec, f_rec = [], [], [], [], [], []
    ref_p, ref_q, ref_rq, ref_rp = [], [], [], []
    a_rec = []

    def record(f: int, jq_, bp_, bq_):
        q_rec.append(data.qpos[qadr].copy())
        p_rec.append(data.xpos[bid].copy())
        r_rec.append(np.array([data.xmat[b].reshape(3, 3).copy() for b in bid]))
        rq_rec.append(data.qpos[3:7].copy())
        pos_rec.append(data.qpos[0:3].copy())
        ref_q.append(jq_.copy())
        ref_p.append(bp_.copy())
        ref_rq.append(bq_[0].copy())
        ref_rp.append(bp_[0].copy())
        f_rec.append(f)

    def fallen() -> bool:
        R = quat_to_mat(data.qpos[3:7])
        return data.qpos[2] < 0.25 or np.arccos(np.clip(R[2, 2], -1.0, 1.0)) > 0.8

    fell_at = None
    record(0, jq, bp, bq)
    if fallen():
        fell_at = 0
    f = 0
    while fell_at is None and f < T - 1:
        data.ctrl[ctrl_adr] = q0 + alpha * a
        for _ in range(DECIMATION):
            if not args.kinematic:
                mujoco.mj_step(model, data)
        a_rec.append(a.copy())
        f += 1
        jq, qd, bp, bq, bv, bw, nq = reference_at(f)
        if args.kinematic:  # the reference *is* the robot state at frame f (root pose, velocities, joints)
            data.qpos[0:3] = bp[0]
            data.qpos[3:7] = bq[0] / np.linalg.norm(bq[0])
            data.qpos[qadr] = jq
            data.qvel[0:3] = bv[0]
            data.qvel[3:6] = quat_to_mat(bq[0]).T @ bw[0]
            data.qvel[dadr] = qd
            mujoco.mj_forward(model, data)
        record(f, jq, bp, bq)
        if fallen() or f >= T - 1:
            if fallen():
                fell_at = f
            break
        # observation of frame f (MDP.obs: relative root orientation against the reference at the current
        # frame, previous action = the action just applied), then the action of frame f
        obs_hist = np.concatenate((obs_hist[:, 1:], obs_of(a_rec[-1], quat_to_mat(bq[0]))[None, None]), 1)
        a = action_at(f, quat_to_mat(bq[0]), a_rec[-1])

    q_rec, p_rec, r_rec, rq_rec, pos_rec = map(np.array, (q_rec, p_rec, r_rec, rq_rec, pos_rec))
    ref_p, ref_q, ref_rq, ref_rp = map(np.array, (ref_p, ref_q, ref_rq, ref_rp))
    a_rec = np.array(a_rec)
    n = len(q_rec)

    # ---- metrics ----
    joint_err = float(np.abs(q_rec - ref_q).mean())
    root_err = float(np.linalg.norm(pos_rec - ref_rp, axis=1).mean())
    body_err = float(np.linalg.norm(p_rec - ref_p, axis=2).mean())
    # A2: tracked-body positions with robot and reference each in their own pelvis heading frame
    local_err = np.zeros(n)
    for i in range(n):
        Hr = yaw_mat(r_rec[i, 0]).T
        Hf = yaw_mat(quat_to_mat(ref_rq[i])).T
        pr = np.einsum("ij,bj->bi", Hr, p_rec[i] - pos_rec[i])
        pf = np.einsum("ij,bj->bi", Hf, ref_p[i] - ref_rp[i])
        local_err[i] = np.linalg.norm(pr - pf, axis=1).mean()
    tilt = np.array([np.arccos(np.clip(r[0, 2, 2], -1.0, 1.0)) for r in r_rec])

    res = {
        "policy": pol.name,
        "oracle": args.oracle_npz is not None,
        "kinematic": args.kinematic,
        "motion": meta.get("motion"),
        "frames_played": int(n),
        "frames_total": int(T),
        "success": fell_at is None,
        "fell_at": fell_at,
        "joint_err_rad": joint_err,
        "body_err_local_m": float(local_err.mean()),
        "body_err_m": body_err,
        "root_err_m": root_err,
        "min_base_z": float(pos_rec[:, 2].min()),
        "max_tilt_rad": float(tilt.max()),
        "first_obs_absmax": float(np.abs(first_obs).max()),
        "first_action_absmax": float(np.abs(first_action).max()),
    }
    print(json.dumps(res, indent=1))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.out, q=q_rec, body_pos=p_rec, body_R=r_rec, root_pos=pos_rec, root_quat=rq_rec,
                 ref_p=ref_p, ref_q=ref_q, ref_root_pos=ref_rp, ref_root_quat=ref_rq, actions=a_rec,
                 frames=np.array(f_rec), joint_names=np.array(names), body_names=np.array(tracked))
        # the same rollout as a PartyMimic motion file (body 0 = pelvis, wxyz, 50 Hz) for check_motion_f1.py
        fps = int(round(1.0 / dt))
        np.savez(str(args.out)[:-4] + "_motion.npz", fps=np.array(fps), joint_pos=q_rec,
                 joint_names=np.array(names), body_names=np.array([tracked[0]]),
                 body_pos_w=pos_rec[:, None, :], body_quat_w=rq_rec[:, None, :])
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(res, indent=1))


def _literal(name: str):
    tree = ast.parse(F1_CONSTANTS.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and getattr(node.targets[0], "id", None) == name:
            return ast.literal_eval(node.value)
    raise KeyError(name)


def _default_joint_pos_dict():
    return _literal("DEFAULT_JOINT_POS")


def _tracked_bodies():
    return _literal("TRACKED_BODIES")


if __name__ == "__main__":
    main()
