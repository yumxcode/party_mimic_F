#!/usr/bin/env python3
"""Compare PartyMimic's own (mjlab / MuJoCo-Warp) replay of a policy with the CPU-MuJoCo sim2sim.

Input: the records of tools/f1_native_replay.py (``hist`` (T,H,99), ``goal``, ``action`` (torch policy),
``root_p``, ``root_quat`` wxyz, ``q``, ``f``) and the exported ONNX used by tools/f1_sim2sim.py.
  (a) ONNX action on every recorded observation history vs the torch action (export check);
  (b) the sim2sim's own observation at the reset state vs the native first observation (layout check,
      incl. the 6D relative-orientation element order);
  (c) the native trajectory as a motion npz for tools/check_motion_f1.py (A4/A5 natively) and its
      A2/A3 against the reference, next to the CPU-MuJoCo rollout of tools/f1_sim2sim.py.
Usage: python tools/f1_native_compare.py --native eval/native/<clip>.npz --onnx eval/f1_dance_v1/f1_dance_v1_<clip>.onnx \
           --sim eval/f1_dance_v1/acceptance/<clip>.npz --out eval/native/<clip>_compare.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parent))
import f1_sim2sim as S  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--native", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--sim", default=None, help="CPU-MuJoCo rollout npz of tools/f1_sim2sim.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--stance_min_frames", type=int, default=8)
    args = ap.parse_args()
    d = np.load(args.native, allow_pickle=True)
    hist, fs, ta, q, rp, rq, f_rec = d["hist"], d["f"], d["action"], d["q"], d["root_p"], d["root_quat"], d["f"]
    T = len(ta)
    res = {"native": args.native, "frames": int(T), "terminated": bool(d["term"][-1]) if len(d["term"]) else None}

    # (a) export check: ONNX on the recorded histories vs the torch policy (torch ran under bf16 autocast
    #     when the run used --amp bf16, so differences of order 1e-2 are expected)
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    meta = dict(sess.get_modelmeta().custom_metadata_map)
    acts = np.stack([sess.run(None, {"obs_history": hist[i:i + 1].astype(np.float32),
                                     "time_step": np.array([[float(f_rec[i])]], np.float32)})[0][0] for i in range(T)])
    diff = np.abs(acts - ta)
    res["onnx_vs_torch"] = {"max_abs": float(diff.max()), "mean_abs": float(diff.mean()),
                            "torch_action_absmax": float(np.abs(ta).max()), "note": "torch under the run's --amp autocast"}

    # (b) the sim2sim observation at the reset state vs the native first observation
    tables = S.f1_tables()
    names = list(tables["F1_JOINT_NAMES"])
    g = [S.gains(tables, n) for n in names]
    kp, kd, ef, ar = [x[0] for x in g], [x[1] for x in g], [x[2] for x in g], [x[3] for x in g]
    q0 = np.zeros(len(names))
    for k, v in S._default_joint_pos_dict().items():
        q0[[i for i, n in enumerate(names) if re.fullmatch(k, n)]] = v
    model = S.build_model(REPO / "assets/f1/f1_party.xml", names, kp, kd, ef, ar)
    S.apply_mujoco_cfg(model)
    data = mujoco.MjData(model)
    jid = [model.joint(n).id for n in names]
    qadr = np.array([model.jnt_qposadr[j] for j in jid])
    dadr = np.array([model.jnt_dofadr[j] for j in jid])
    out0 = sess.run(None, {"obs_history": np.zeros((1, 10, hist.shape[-1]), np.float32), "time_step": np.zeros((1, 1), np.float32)})
    a0, jq, qd, bp, bq, bv, bw, nq = out0
    jq, qd, bp, bq, bv, bw = jq[0], qd[0], bp[0], bq[0], bv[0], bw[0]
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = bp[0]
    data.qpos[3:7] = bq[0] / np.linalg.norm(bq[0])
    data.qpos[qadr] = jq
    R0 = S.quat_to_mat(bq[0])
    data.qvel[0:3] = bv[0]
    data.qvel[3:6] = R0.T @ bw[0]
    data.qvel[dadr] = qd
    mujoco.mj_forward(model, data)
    R = S.quat_to_mat(data.qpos[3:7])
    o_sim = np.concatenate((data.qpos[qadr] - q0, data.qvel[dadr], data.qvel[3:6], -R[2, :], np.zeros(len(names)),
                            S.mat_to_6d(R.T @ R0))).astype(np.float32)
    o_nat = hist[0, -1]
    res["first_obs"] = {"max_abs_diff": float(np.abs(o_sim - o_nat).max()),
                        "per_block": {name: float(np.abs(o_sim[a:b] - o_nat[a:b]).max()) for name, a, b in
                                      [("q-q0", 0, 29), ("qd", 29, 58), ("w_b", 58, 61), ("g_b", 61, 64), ("a_prev", 64, 93), ("rel6d", 93, 99)]},
                        "native_rel6d": [float(x) for x in o_nat[93:99]], "sim_rel6d": [float(x) for x in o_sim[93:99]],
                        "history_all_equal_first": bool(np.allclose(hist[0], hist[0, -1:], atol=1e-6))}
    res["first_action"] = {"onnx": [float(x) for x in acts[0][:6]], "torch": [float(x) for x in ta[0][:6]]}

    # (c) native trajectory as a motion file -> gates; native A2/A3 vs reference; next to the CPU rollout
    clip = Path(args.native).name.replace("f1_dance_v1_native_", "").replace(".npz", "")
    motion = str(args.out)[:-5] + "_motion.npz"
    np.savez(motion, fps=np.array(50), joint_pos=q, joint_names=np.array(names), body_names=np.array(["base_link"]),
             body_pos_w=rp[:, None, :], body_quat_w=rq[:, None, :])
    gates = {}
    for n in sorted({args.stance_min_frames, 5}):
        gj = str(args.out)[:-5] + f"_gates_smf{n}.json"
        p = subprocess.run([sys.executable, str(REPO / "tools/check_motion_f1.py"), "--npz", motion, "--fps", "50",
                            "--stance_min_frames", str(n), "--json_out", gj], capture_output=True, text=True)
        gates[n] = json.load(open(gj))
    g = gates[args.stance_min_frames]
    res["native_gates"] = {"flat_pct_L": g["foot_flatness"]["left"]["flat_pct"], "flat_pct_R": g["foot_flatness"]["right"]["flat_pct"],
                           "pitch_p95_L": g["foot_flatness"]["left"]["pitch_p95_deg"], "pitch_p95_R": g["foot_flatness"]["right"]["pitch_p95_deg"],
                           "stance_frames": [g["foot_flatness"]["left"]["frames"], g["foot_flatness"]["right"]["frames"]],
                           "hiproll_adduct_pct": g["stance_openness"]["both_hiproll_adducted"]["pct"],
                           "flat_pct_smf5_LR": [gates[5]["foot_flatness"]["left"]["flat_pct"], gates[5]["foot_flatness"]["right"]["flat_pct"]]}
    # native A2/A3 against the reference (from the ONNX at each frame)
    ref_q, ref_p, ref_rq = [], [], []
    for i in range(T):
        _, jq, qd, bp, bq, bv, bw, nq = sess.run(None, {"obs_history": hist[i:i + 1].astype(np.float32),
                                                         "time_step": np.array([[float(f_rec[i])]], np.float32)})
        ref_q.append(jq[0]), ref_p.append(bp[0]), ref_rq.append(bq[0][0])
    ref_q, ref_p, ref_rq = map(np.array, (ref_q, ref_p, ref_rq))
    # native tracked-body positions come from FK of the recorded state (same kinematics)
    bid = [model.body(n).id for n in meta["body_names"].split(",")]
    p_nat = np.zeros((T, len(bid), 3))
    for i in range(T):
        mujoco.mj_resetData(model, data)
        data.qpos[0:3] = rp[i]
        data.qpos[3:7] = rq[i]
        data.qpos[qadr] = q[i]
        mujoco.mj_forward(model, data)
        p_nat[i] = data.xpos[bid]
    rR = np.array([S.quat_to_mat(x) for x in rq])
    le = []
    for i in range(T):
        Hr = S.yaw_mat(rR[i]).T
        Hf = S.yaw_mat(S.quat_to_mat(ref_rq[i])).T
        pr = np.einsum("ij,bj->bi", Hr, p_nat[i] - rp[i])
        pf = np.einsum("ij,bj->bi", Hf, ref_p[i] - ref_p[i, 0])
        le.append(np.linalg.norm(pr - pf, axis=1).mean())
    tilt = np.array([np.arccos(np.clip(r[2, 2], -1, 1)) for r in rR])
    res["native_metrics"] = {"joint_err_rad": float(np.abs(q - ref_q).mean()), "body_err_local_m": float(np.mean(le)),
                             "body_err_local_p95_m": float(np.percentile(le, 95)), "min_base_z": float(rp[:, 2].min()),
                             "max_tilt_rad": float(tilt.max())}
    if args.sim and Path(args.sim).exists():
        s = np.load(args.sim, allow_pickle=True)
        n = min(len(s["q"]), T)
        res["cpu_vs_native"] = {"joint_abs_diff_rad_mean": float(np.abs(s["q"][:n] - q[:n]).mean()),
                                "root_pos_diff_m_mean": float(np.linalg.norm(s["root_pos"][:n] - rp[:n], axis=1).mean()),
                                "right_ankle_pitch_cpu_minus_native_rad": float(np.mean(s["q"][:n, 10] - q[:n, 10]))}
        sj = json.loads(Path(str(args.sim)[:-4] + ".json").read_text())
        res["cpu_metrics"] = {k: sj[k] for k in ("joint_err_rad", "body_err_local_m", "min_base_z", "max_tilt_rad")}
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
