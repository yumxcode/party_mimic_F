#!/usr/bin/env python3
"""Convert an F1 retargeted CSV (36 cols, URDF joint order, 30 fps) to a PartyMimic motion npz.

PartyMimic motion format (README "Motion format"):
    fps           scalar (50)
    joint_pos     [T, J]  joint angles in the robot's joint order (MJCF order for us)
    body_pos_w    [T, B, 3]   world body positions, first body = pelvis
    body_quat_w   [T, B, 4]   world body quaternions (wxyz), first body = pelvis
PartyMimic only reads body index 0 and recomputes the rest by FK, so we store B=1.

Input CSV layout (tools_bvh_to_csv_f1.py): root_pos xyz, root_quat xyzw, 29 joints in URDF order
(lumbar -> arms -> legs). We permute to the MJCF order of F1_29DOF (legs -> lumbar -> arms) and
verify the permutation by FK: the soles must sit on z=0.

Usage:
    python tools/f1_csv_to_pm_npz.py --csv work/retarget_csv/dance1_subject1_f1.csv \
        --out data/motions/f1/dance1_subject1.npz
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parent.parent
MJCF = REPO / "F1_29DOF/mjcf/f1_flat.xml"

# URDF revolute order (== UMR / Beyondminic CSV order)
URDF_ORDER = [
    "lumbar_yaw_joint", "lumbar_roll_joint", "lumbar_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint", "left_elbow_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint", "right_elbow_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint",
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_pitch_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]


def mjcf_joint_order(model) -> list[str]:
    return [model.joint(i).name for i in range(1, model.njnt)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--input_fps", type=float, default=30.0)
    ap.add_argument("--output_fps", type=float, default=50.0)
    ap.add_argument("--sole_check", action="store_true", default=True)
    args = ap.parse_args()

    raw = np.loadtxt(args.csv, delimiter=",")
    assert raw.shape[1] == 36, raw.shape
    n_in = raw.shape[0]
    in_dt, out_dt = 1.0 / args.input_fps, 1.0 / args.output_fps
    duration = (n_in - 1) * in_dt
    times = np.arange(0, duration, out_dt)
    phase = times / duration
    i0 = np.floor(phase * (n_in - 1)).astype(int)
    i1 = np.minimum(i0 + 1, n_in - 1)
    a = phase * (n_in - 1) - i0

    pos = raw[:, 0:3]
    quat_wxyz = raw[:, [6, 3, 4, 5]]  # xyzw -> wxyz
    j_urdf = raw[:, 7:36]
    assert j_urdf.shape[1] == 29

    p = pos[i0] * (1 - a[:, None]) + pos[i1] * a[:, None]
    j = j_urdf[i0] * (1 - a[:, None]) + j_urdf[i1] * a[:, None]
    # slerp for the root quaternion
    ra = R.from_quat(quat_wxyz[i0][:, [1, 2, 3, 0]])
    rb = R.from_quat(quat_wxyz[i1][:, [1, 2, 3, 0]])
    # slerp: r(a) = r0 * (r1 * r0^-1)^a
    q_wxyz = np.array([(ra[k] * (rb[k] * ra[k].inv()) ** float(a[k])).as_quat()[[3, 0, 1, 2]]
                       for k in range(len(times))])
    T = len(times)

    model = mujoco.MjModel.from_xml_path(str(MJCF))
    data = mujoco.MjData(model)
    mj_order = mjcf_joint_order(model)
    assert set(mj_order) == set(URDF_ORDER), (set(mj_order) ^ set(URDF_ORDER))
    perm = [URDF_ORDER.index(n) for n in mj_order]
    j_mj = j[:, perm]
    qadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in mj_order])
    bid0 = model.body("base_link").id

    body_pos = np.zeros((T, 1, 3), dtype=np.float32)
    body_quat = np.zeros((T, 1, 4), dtype=np.float32)
    for t in range(T):
        mujoco.mj_resetData(model, data)
        data.qpos[0:3] = p[t]
        data.qpos[3:7] = q_wxyz[t]
        data.qpos[qadr] = j_mj[t]
        mujoco.mj_forward(model, data)
        body_pos[t, 0] = data.xpos[bid0]
        body_quat[t, 0] = data.xquat[bid0]

    # FK sanity: soles must sit on the floor if the joint order/convention is right
    stats = {}
    if args.sole_check:
        corners = {"left": (-0.0967, 0.1071, -0.0395, 0.0365), "right": (-0.0967, 0.1071, -0.0365, 0.0395)}
        bid = {s: model.body(f"{s}_ankle_roll").id for s in corners}
        zs = []
        for t in range(0, T, 5):
            mujoco.mj_resetData(model, data)
            data.qpos[0:3] = p[t]
            data.qpos[3:7] = q_wxyz[t]
            data.qpos[qadr] = j_mj[t]
            mujoco.mj_forward(model, data)
            for s, (x0, x1, y0, y1) in corners.items():
                Rm = data.xmat[bid[s]].reshape(3, 3)
                for x in (x0, x1):
                    for y in (y0, y1):
                        zs.append((data.xpos[bid[s]] + Rm @ np.array([x, y, -0.0337]))[2])
        zs = np.array(zs)
        stats = {"sole_z_mean": float(zs.mean()), "sole_z_p5": float(np.percentile(zs, 5)),
                 "sole_z_min": float(zs.min()), "frames_checked": int(len(range(0, T, 5)))}
        print("[fk] sole z: mean %.4f p5 %.4f min %.4f" % (zs.mean(), np.percentile(zs, 5), zs.min()))

    lim = model.jnt_range[1:]
    viol = ((j_mj < lim[:, 0] - 1e-6) | (j_mj > lim[:, 1] + 1e-6))
    stats["joint_limit_violation_frames"] = int(viol.any(1).sum())

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, fps=np.float32(args.output_fps), joint_pos=j_mj.astype(np.float32),
             body_pos_w=body_pos, body_quat_w=body_quat,
             joint_names=np.array(mj_order), source=args.csv.name)
    meta = {"source": str(args.csv), "frames_in": int(n_in), "frames_out": int(T),
            "duration_s": float(duration), "fps_out": args.output_fps,
            "perm_urdf_to_mjcf": perm, "fk": stats}
    Path(str(args.out) + ".json").write_text(json.dumps(meta, indent=2))
    print(f"[convert] {n_in}f@{args.input_fps} -> {T}f@{args.output_fps} ({duration:.2f}s) -> {args.out}")
    print("[convert] joint limit violations: %d frames" % stats["joint_limit_violation_frames"])


if __name__ == "__main__":
    main()
