#!/usr/bin/env python3
"""Foot-flatness compression for retargeted F1 motion CSVs (a retargeting-recipe stage).

The GMR retarget follows the LAFAN1 dancer's foot orientation, so tiptoe phases come out as tiptoe
(sole pitch up to ~18 deg). This stage compresses the sole pitch/roll of BOTH feet toward flat while a
foot is in STANCE (gate-7 definition: a contact run of >= stance_min_frames; swing feet keep the dance):
    |sole| > thr  ->  thr + factor * (|sole| - thr)
(e.g. thr 4 deg, factor 0.2: a 10 deg tiptoe becomes 5.2 deg, 18 deg -> 6.9 deg), by changing only the
ankle pitch / ankle roll joints (gain measured numerically from the FK; each joint change capped), and
shifting the root height up so no sole corner ends up lower than before. The stance weight fades over
`ramp` frames, the corrections and root shift are smoothed (centered moving average). Everything else
(hips, knees, torso, arms, root xy / orientation) is untouched. The nine gates (tools/check_motion_f1.py)
are re-run on the output; nothing is relaxed.

CSV layout (tools_bvh_to_csv_f1.py): root_pos xyz, root_quat xyzw, 29 joints in URDF order
(check_motion_f1.F1_JOINT_NAMES). FK uses assets/f1/f1_party.xml (same kinematics / sole spheres).

Usage:
  python tools/f1_flatten_feet.py --csv work/retarget_csv/dance1_subject3_f1.csv \
      --out work/retarget_csv_flat/dance1_subject3_f1.csv --pitch_thr_deg 4 --roll_thr_deg 3 --factor 0.2
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_motion_f1 as C  # noqa: E402  (FOOT_CORNERS, FOOT_LEN, FOOT_WID, F1_JOINT_NAMES (URDF order), B)

REPO = Path(__file__).resolve().parent.parent


class FK:
    def __init__(self, mjcf: Path):
        self.m = mujoco.MjModel.from_xml_path(str(mjcf))
        self.d = mujoco.MjData(self.m)
        self.qadr = {n: int(self.m.jnt_qposadr[self.m.joint(n).id]) for n in C.F1_MJCF_ORDER}
        self.lo = {n: self.m.jnt_range[self.m.joint(n).id][0] for n in C.F1_MJCF_ORDER}
        self.hi = {n: self.m.jnt_range[self.m.joint(n).id][1] for n in C.F1_MJCF_ORDER}
        self.bid = {s: self.m.body(C.B(f"{s}_ankle_roll_link")).id for s in ("left", "right")}
        self.perm = [C.F1_MJCF_ORDER.index(n) for n in C.F1_JOINT_NAMES]  # URDF col -> MJCF index

    def soles(self, pos, quat_xyzw, j_urdf):
        q = np.zeros(self.m.nq)
        q[0:3] = pos
        q[3:7] = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]  # wxyz
        j_mj = np.empty(len(self.perm))
        j_mj[self.perm] = np.asarray(j_urdf)  # perm[i] = MJCF index of URDF joint i
        for i, n in enumerate(C.F1_MJCF_ORDER):
            q[self.qadr[n]] = j_mj[i]
        self.d.qpos[:] = q
        mujoco.mj_forward(self.m, self.d)
        out = {}
        for s in ("left", "right"):
            R = self.d.xmat[self.bid[s]].reshape(3, 3)
            out[s] = np.array([self.d.xpos[self.bid[s]] + R @ np.array(c) for c in C.FOOT_CORNERS[s]])
        return out


def pr_angles(cz):
    heel, toe = cz[[0, 2]].mean(0), cz[[1, 3]].mean(0)
    medial, lateral = cz[[2, 3]].mean(0), cz[[0, 1]].mean(0)
    return (np.degrees(np.arctan2(heel[2] - toe[2], C.FOOT_LEN)),
            np.degrees(np.arctan2(medial[2] - lateral[2], C.FOOT_WID)))


def smooth(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    k = np.ones(window) / window
    pad = window // 2
    xp = np.concatenate([x[:pad][::-1], x, x[-pad:][::-1]], axis=0)
    out = np.stack([np.convolve(xp[:, c], k, mode="same")[pad:pad + len(x)] for c in range(x.shape[1])], 1)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pitch_thr_deg", type=float, default=4.0)
    ap.add_argument("--roll_thr_deg", type=float, default=3.0)
    ap.add_argument("--factor", type=float, default=0.2, help="slope beyond the threshold (0: hard cap)")
    ap.add_argument("--smooth", type=int, default=5, help="moving-average window for the corrections (frames)")
    ap.add_argument("--contact_h", type=float, default=0.03, help="gate 7: lowest sole corner below this = contact (m)")
    ap.add_argument("--stance_min_frames", type=int, default=5, help="gate 7: contact run length for stance (frames @30 fps)")
    ap.add_argument("--ramp", type=int, default=3, help="fade width of the stance weight around stance frames")
    ap.add_argument("--max_corr", type=float, default=0.35, help="cap on any ankle-joint change (rad)")
    ap.add_argument("--passes", type=int, default=2)
    ap.add_argument("--json", default=None)
    ap.add_argument("--mjcf", default=str(REPO / "assets/f1/f1_party.xml"))
    args = ap.parse_args()

    raw = np.loadtxt(args.csv, delimiter=",")
    pos, quat, joints = raw[:, :3].copy(), raw[:, 3:7].copy(), raw[:, 7:].copy()
    T = len(raw)
    fk = FK(Path(args.mjcf))
    thr_p, thr_r, fac = args.pitch_thr_deg, args.roll_thr_deg, args.factor

    # numerical gain of sole pitch / roll w.r.t. the ankle joints (frame 0, both feet)
    gain = {}
    for s in ("left", "right"):
        for axis, jn in (("pitch", f"{s}_ankle_pitch_joint"), ("roll", f"{s}_ankle_roll_joint")):
            jp, jm = joints[0].copy(), joints[0].copy()
            col = C.F1_JOINT_NAMES.index(jn)
            jp[col] += 0.01
            jm[col] -= 0.01
            ap_ = pr_angles(fk.soles(pos[0], quat[0], jp)[s])[0 if axis == "pitch" else 1]
            am_ = pr_angles(fk.soles(pos[0], quat[0], jm)[s])[0 if axis == "pitch" else 1]
            gain[(s, axis)] = (ap_ - am_) / 0.02  # deg per rad
    print("sole-angle gain (deg/rad):", {f"{s}_{a}": round(g, 3) for (s, a), g in gain.items()})

    def compress(a, thr):
        return np.sign(a) * np.where(np.abs(a) > thr, thr + fac * (np.abs(a) - thr), np.abs(a))

    # gate-7 definitions: contact = lowest sole corner < contact_h; stance = contact runs >= stance_min frames
    before = {"pitch": {s: np.zeros(T) for s in ("left", "right")}, "roll": {s: np.zeros(T) for s in ("left", "right")}}
    minz_before = {s: np.zeros(T) for s in ("left", "right")}
    stance = {s: np.zeros(T, bool) for s in ("left", "right")}
    for t in range(T):
        cz = fk.soles(pos[t], quat[t], joints[t])
        for s in ("left", "right"):
            before["pitch"][s][t], before["roll"][s][t] = pr_angles(cz[s])
            minz_before[s][t] = cz[s][:, 2].min()
    for s in ("left", "right"):
        contact = minz_before[s] - C.FOOT_SPHERE_R < args.contact_h
        i = 0
        while i < T:
            if contact[i]:
                j = i
                while j < T and contact[j]:
                    j += 1
                if j - i >= args.stance_min_frames:
                    stance[s][i:j] = True
                i = j
            else:
                i += 1
    # soft weight: 1 on stance frames, fading over `ramp` frames around them
    w = {s: smooth(stance[s].astype(float)[:, None], 2 * args.ramp + 1)[:, 0] for s in ("left", "right")}

    def compress(a, thr):
        return np.sign(a) * np.where(np.abs(a) > thr, thr + fac * (np.abs(a) - thr), np.abs(a))

    cols = {("left", "pitch"): C.F1_JOINT_NAMES.index("left_ankle_pitch_joint"),
            ("left", "roll"): C.F1_JOINT_NAMES.index("left_ankle_roll_joint"),
            ("right", "pitch"): C.F1_JOINT_NAMES.index("right_ankle_pitch_joint"),
            ("right", "roll"): C.F1_JOINT_NAMES.index("right_ankle_roll_joint")}
    jn = {("left", "pitch"): "left_ankle_pitch_joint", ("left", "roll"): "left_ankle_roll_joint",
          ("right", "pitch"): "right_ankle_pitch_joint", ("right", "roll"): "right_ankle_roll_joint"}
    corr = np.zeros((T, 4))  # L pitch, L roll, R pitch, R roll (rad)
    dz = np.zeros(T)
    work = joints.copy()
    for it in range(args.passes):
        dc = np.zeros((T, 4))
        for t in range(T):
            cz = fk.soles(pos[t], quat[t], work[t])
            for s in ("left", "right"):
                p, r = pr_angles(cz[s])
                for axis, a in (("pitch", p), ("roll", r)):
                    target = compress(a, thr_p if axis == "pitch" else thr_r)
                    dc[t, list(cols).index((s, axis))] = w[s][t] * (target - a) / gain[(s, axis)]
        dc = smooth(dc, args.smooth)
        for k, (s, axis) in enumerate(cols):
            c = cols[(s, axis)]
            lo_ = np.maximum(fk.lo[jn[(s, axis)]] + 0.01, joints[:, c] - args.max_corr)
            hi_ = np.minimum(fk.hi[jn[(s, axis)]] - 0.01, joints[:, c] + args.max_corr)
            work[:, c] = np.clip(work[:, c] + dc[:, k], lo_, hi_)
            corr[:, k] = work[:, c] - joints[:, c]
        for t in range(T):
            cz = fk.soles(pos[t], quat[t], work[t])
            dz[t] = max(minz_before[s][t] - cz[s][:, 2].min() for s in ("left", "right"))
        dz = np.maximum(np.maximum(smooth(dz[:, None], args.smooth)[:, 0], dz), 0.0)  # never under-compensate
    pos[:, 2] += dz

    after = {"pitch": {s: np.zeros(T) for s in ("left", "right")}, "roll": {s: np.zeros(T) for s in ("left", "right")}}
    contact = {s: np.zeros(T, bool) for s in ("left", "right")}
    minz_after = {s: np.zeros(T) for s in ("left", "right")}
    for t in range(T):
        cz = fk.soles(pos[t], quat[t], work[t])
        for s in ("left", "right"):
            after["pitch"][s][t], after["roll"][s][t] = pr_angles(cz[s])
            minz_after[s][t] = cz[s][:, 2].min()
            contact[s][t] = stance[s][t]

    out = np.concatenate((pos, quat, work), axis=1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.out, out, delimiter=",", fmt="%.8f")
    rep = {"csv": args.csv, "out": args.out, "frames": T, "params": {"pitch_thr_deg": thr_p, "roll_thr_deg": thr_r, "factor": fac, "smooth": args.smooth,
                                                                      "contact_h": args.contact_h, "stance_min_frames": args.stance_min_frames, "ramp": args.ramp, "max_corr": args.max_corr},
           "gain_deg_per_rad": {f"{s}_{a}": round(g, 4) for (s, a), g in gain.items()}, "feet": {}}
    for s in ("left", "right"):
        c = contact[s]
        flat_b = (np.abs(before["pitch"][s][c]) <= 10) & (np.abs(before["roll"][s][c]) <= 8)
        flat_a = (np.abs(after["pitch"][s][c]) <= 10) & (np.abs(after["roll"][s][c]) <= 8)
        rep["feet"][s] = {
            "stance_frames": int(c.sum()),
            "flat_pct_stance_before": round(100 * flat_b.mean(), 2) if c.any() else None,
            "flat_pct_stance_after": round(100 * flat_a.mean(), 2) if c.any() else None,
            "pitch_p95_before": round(float(np.percentile(np.abs(before["pitch"][s][c]), 95)), 2) if c.any() else None,
            "pitch_p95_after": round(float(np.percentile(np.abs(after["pitch"][s][c]), 95)), 2) if c.any() else None,
            "pitch_max_after": round(float(np.abs(after["pitch"][s]).max()), 2),
            "roll_p95_before": round(float(np.percentile(np.abs(before["roll"][s][c]), 95)), 2) if c.any() else None,
            "roll_p95_after": round(float(np.percentile(np.abs(after["roll"][s][c]), 95)), 2) if c.any() else None,
            "ankle_pitch_correction_rad": [round(float(corr[:, 0 if s == "left" else 2].min()), 4), round(float(corr[:, 0 if s == "left" else 2].max()), 4)],
            "ankle_roll_correction_rad": [round(float(corr[:, 1 if s == "left" else 3].min()), 4), round(float(corr[:, 1 if s == "left" else 3].max()), 4)],
            "min_sole_z_after": round(float(minz_after[s].min()), 4),
        }
    rep["root_z_shift_m"] = {"mean": round(float(dz.mean()), 4), "max": round(float(dz.max()), 4)}
    print(json.dumps(rep, indent=1))
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
