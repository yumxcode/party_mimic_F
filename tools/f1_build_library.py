#!/usr/bin/env python3
"""Build the PartyMimic F1 motion library from selected retargeted CSV windows.

Per window: slice the CSV, lift ground-penetrating frames (the Beyondminic fix: only root z moves,
Gaussian-smoothed, joints untouched), resample 30 -> 50 Hz, permute the joints from URDF order to
the MJCF order, FK the pelvis, and write the PartyMimic npz (body 0 = pelvis).

Plan file: {"clips": [{"csv": ..., "start": int, "end": int, "name": str, "lift": bool}]}
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parent.parent
MJCF = REPO / "assets/f1/f1_party.xml"
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
CORNERS = {"left": (-0.0967, 0.1071, -0.0395, 0.0365), "right": (-0.0967, 0.1071, -0.0365, 0.0395)}


class Fk:
    def __init__(self):
        self.m = mujoco.MjModel.from_xml_path(str(MJCF))
        self.d = mujoco.MjData(self.m)
        self.mj_order = [self.m.joint(i).name for i in range(1, self.m.njnt)]
        assert set(self.mj_order) == set(URDF_ORDER)
        self.perm = [URDF_ORDER.index(n) for n in self.mj_order]
        self.qadr = np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in self.mj_order])
        self.bid = {s: self.m.body(f"{s}_ankle_roll").id for s in CORNERS}

    def forward(self, pos, quat_wxyz, j_mj):
        d, m = self.d, self.m
        mujoco.mj_resetData(m, d)
        d.qpos[0:3] = pos
        d.qpos[3:7] = quat_wxyz
        d.qpos[self.qadr] = j_mj
        mujoco.mj_forward(m, d)
        return d

    def sole_min(self, pos, quat_wxyz, j_mj):
        d = self.forward(pos, quat_wxyz, j_mj)
        z = np.inf
        for s, (x0, x1, y0, y1) in CORNERS.items():
            Rm = d.xmat[self.bid[s]].reshape(3, 3)
            for x in (x0, x1):
                for y in (y0, y1):
                    z = min(z, (d.xpos[self.bid[s]] + Rm @ np.array([x, y, -0.0337]))[2] - 0.002)
        return z


def convert(raw: np.ndarray, out: Path, fk: Fk, in_fps=30.0, out_fps=50.0, lift=True,
            target_depth=-0.013, sigma=3.0):
    pos = raw[:, 0:3].copy()
    quat_wxyz = raw[:, [6, 3, 4, 5]]
    j_urdf = raw[:, 7:36]
    j_mj = j_urdf[:, fk.perm]
    T = len(raw)

    # resample
    duration = (T - 1) / in_fps
    times = np.arange(0, duration, 1.0 / out_fps)
    phase = times / duration
    i0 = np.floor(phase * (T - 1)).astype(int)
    i1 = np.minimum(i0 + 1, T - 1)
    a = phase * (T - 1) - i0
    p = pos[i0] * (1 - a[:, None]) + pos[i1] * a[:, None]
    j = j_mj[i0] * (1 - a[:, None]) + j_mj[i1] * a[:, None]
    ra = R.from_quat(quat_wxyz[i0][:, [1, 2, 3, 0]])
    rb = R.from_quat(quat_wxyz[i1][:, [1, 2, 3, 0]])
    q_wxyz = np.array([(ra[k] * (rb[k] * ra[k].inv()) ** float(a[k])).as_quat()[[3, 0, 1, 2]]
                       for k in range(len(times))])

    lift_report = None
    if lift:
        zs = np.array([fk.sole_min(p[t], q_wxyz[t], j[t]) for t in range(len(times))])
        z0 = zs.copy()
        total_bump = np.zeros(len(times))
        for _ in range(8):  # iterate: the Gaussian bump spreads, so re-lift what is still deep
            need = np.maximum(0.0, -(zs - target_depth))
            if need.max() <= 0:
                break
            bump = gaussian_filter1d(need, sigma, mode="nearest")
            total_bump += bump
            p[:, 2] += bump
            zs = np.array([fk.sole_min(p[t], q_wxyz[t], j[t]) for t in range(len(times))])
        # residual per-frame lift (<= a few mm) so no frame stays below the gate
        res = np.maximum(0.0, -(zs - target_depth))
        if res.max() > 0:
            p[:, 2] += res
            total_bump += res
            zs = np.array([fk.sole_min(p[t], q_wxyz[t], j[t]) for t in range(len(times))])
        lift_report = {"frames_penetrating": int((z0 < target_depth).sum()),
                       "max_penetration_m": float(min(0.0, z0.min())),
                       "max_lift_m": float(total_bump.max()),
                       "after_min_m": float(zs.min())}

    bp = np.zeros((len(times), 1, 3), np.float32)
    bq = np.zeros((len(times), 1, 4), np.float32)
    bid0 = fk.m.body("base_link").id
    for t in range(len(times)):
        d = fk.forward(p[t], q_wxyz[t], j[t])
        bp[t, 0] = d.xpos[bid0]
        bq[t, 0] = d.xquat[bid0]

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, fps=np.float32(out_fps), joint_pos=j.astype(np.float32),
             body_pos_w=bp, body_quat_w=bq, joint_names=np.array(fk.mj_order))
    return {"frames_out": int(len(times)), "duration_s": float(duration), "lift": lift_report,
            "sole_z_after_mean": float(bp[:, 0, 2].mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, default=REPO / "data/motions/f1")
    ap.add_argument("--no_lift", action="store_true")
    args = ap.parse_args()
    plan = json.loads(args.plan.read_text())
    fk = Fk()
    manifest = []
    for c in plan["clips"]:
        raw = np.loadtxt(c["csv"], delimiter=",")
        a, b = int(c["start"]), int(c["end"])
        seg = raw[a:b]
        name = c["name"]
        out = args.out_dir / f"{name}.npz"
        rep = convert(seg, out, fk, lift=(not args.no_lift) and c.get("lift", True))
        rep.update(name=name, csv=str(c["csv"]), start=a, end=b, source=c.get("source", ""),
                   gates=c.get("gates"))
        manifest.append(rep)
        print(f"{name}: {b-a}f -> {rep['frames_out']}f {rep['duration_s']:.1f}s  "
              f"lift={rep['lift']}", flush=True)
    (args.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False))
    print(f"total {sum(m['duration_s'] for m in manifest):.1f}s in {len(manifest)} clips -> {args.out_dir}")


if __name__ == "__main__":
    main()
