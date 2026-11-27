#!/usr/bin/env python3
"""Build a *test* ONNX with the interface of PartyMimic's tools/export_onnx.py from a motion npz:
the "policy" is an oracle whose action targets the reference joint angles of the frame (clamp +-20),
the reference outputs are the motion (tracked bodies by MuJoCo FK, root velocities by finite differences).

Only for checking the sim2sim plumbing (metadata, session I/O, frame handling) without a trained
policy; the F1 cannot hold a pose under pure PD with the project's gains, so the oracle rollout falls.

Usage: python tools/f1_make_test_onnx.py --motion data/motions/f1/dance1_subject3_255_706.npz --out work/test.onnx
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parent))
import f1_sim2sim as S  # noqa: E402

REPO = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--motion", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mjcf", default=str(REPO / "assets/f1/f1_party.xml"))
    args = ap.parse_args()
    tables = S.f1_tables()
    names = list(tables["F1_JOINT_NAMES"])
    g = [S.gains(tables, n) for n in names]
    kp, kd, effort = [x[0] for x in g], [x[1] for x in g], [x[2] for x in g]
    alpha = np.array([0.25 * e / p for e, p in zip(effort, kp)])
    import re
    q0 = np.zeros(len(names))
    for k, v in S._default_joint_pos_dict().items():
        q0[[i for i, n in enumerate(names) if re.fullmatch(k, n)]] = v
    tracked = list(S._tracked_bodies())
    pol = S.OraclePolicy(args.motion, Path(args.mjcf), names, q0, alpha, tracked)
    T, B = pol.T, len(tracked)
    actions = np.clip((pol.jq - q0) / alpha, -20, 20).astype(np.float32)
    bv = np.repeat(pol.rv[:, None, :], B, 1).astype(np.float32)
    bw = np.repeat(pol.rw[:, None, :], B, 1).astype(np.float32)
    f32 = lambda x: np.ascontiguousarray(x, np.float32)  # noqa: E731

    inits = [numpy_helper.from_array(actions, "act_c"), numpy_helper.from_array(f32(pol.jq), "jq"),
             numpy_helper.from_array(f32(pol.qd), "qd"), numpy_helper.from_array(f32(pol.bp), "bp"),
             numpy_helper.from_array(f32(pol.bq_all), "bq"), numpy_helper.from_array(bv, "bv"),
             numpy_helper.from_array(bw, "bw"), numpy_helper.from_array(np.array([T - 1], np.int64), "last"),
             numpy_helper.from_array(np.array([1], np.int64), "shape1"), numpy_helper.from_array(np.array(1, np.int64), "one"),
             numpy_helper.from_array(f32(pol.rq), "bq_root")]
    outs = ["actions", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"]
    nodes = [helper.make_node("Reshape", ["time_step", "shape1"], ["t1"]), helper.make_node("Round", ["t1"], ["tr"]),
             helper.make_node("Cast", ["tr"], ["idx"], to=TensorProto.INT64),
             helper.make_node("Add", ["idx", "one"], ["idx1"]), helper.make_node("Min", ["idx1", "last"], ["idx1c"])]
    srcs = ["act_c", "jq", "qd", "bp", "bq", "bv", "bw"]
    for o, s in zip(outs, srcs):
        nodes.append(helper.make_node("Gather", [s, "idx"], [o], axis=0))
    nodes.append(helper.make_node("Gather", ["bq_root", "idx1c"], ["next_anchor_quat_w"], axis=0))
    graph = helper.make_graph(
        nodes, "oracle_policy_with_motion",
        [helper.make_tensor_value_info("obs_history", TensorProto.FLOAT, [1, 10, len(names) * 2 + 6 + len(names) + 6]),
         helper.make_tensor_value_info("time_step", TensorProto.FLOAT, [1, 1])],
        [helper.make_tensor_value_info(n, TensorProto.FLOAT, None) for n in outs + ["next_anchor_quat_w"]], inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    fmt = lambda xs: ",".join(f"{float(x):.6g}" for x in xs)  # noqa: E731
    meta = {
        "run_path": "oracle", "motion": Path(args.motion).stem, "motion_num_frames": str(T),
        "joint_names": ",".join(names), "joint_stiffness": fmt(kp), "joint_damping": fmt(kd),
        "default_joint_pos": fmt(q0), "action_scale": fmt(alpha), "command_names": "motion",
        "observation_names": "joint_pos,joint_vel,base_ang_vel,projected_gravity,actions,motion_anchor_ori_b",
        "observation_history": "10", "goal_frames": "10", "control_dt": "0.02",
        "anchor_body_name": tracked[0], "body_names": ",".join(tracked),
    }
    for k, v in meta.items():
        model.metadata_props.add(key=k, value=v)
    onnx.save(model, args.out)
    print(f"wrote {args.out}: {T} frames, J={len(names)}, B={B}")


if __name__ == "__main__":
    main()
