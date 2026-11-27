#!/usr/bin/env python3
"""Per-frame metrics + nine-gate window selection for F1 retargeted CSVs.

The nine gates and all thresholds are those of Beyondminic-Weilai-X1/tools_check_motion_f1.py
(the validated F1 gate; the F1 asset is md5-identical). This script computes the expensive
per-frame quantities once per clip (FK, self-collision distances) and then evaluates every
candidate window, so window selection is exact and cheap.

Gate thresholds (all inherited, none relaxed):
  1 ground        min sole-corner z - 0.002 > -0.015 m
  2 self          min pair distance > 0.005 m
  3 limits        no joint outside URDF range (+1e-4)
  4 velocities    |dq/dt| <= 1.5x URDF limit (wrist 2.0x)
  5 root height   0.35 .. 1.0 m
  6 continuity    finite, consecutive root quats dot > 0.1
  7 foot flatness >= 90 % of stance frames |sole pitch| <= 10 deg and |sole roll| <= 8 deg
  8 arm quality   <= 10 % frames with an arm joint within 0.05 rad of a limit,
                  every wrist ROM >= 2 % of its range
  9 stance        double-support ankle gap / hip width: mean >= 1.15, p5 >= 0.70,
                  both hip rolls adducted (>3 deg) <= 3 % frames, >= 10 double-support frames

Usage:
  python tools/f1_gate_windows.py --csv work/retarget_csv/dance1_subject1_f1.csv \
      --out work/gate/dance1_subject1.json --win 450 --stride 150
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))
import check_motion_f1 as G  # noqa: E402

F1_JOINT_NAMES = G.F1_JOINT_NAMES


def build_model():
    tree = ET.parse(G.F1_XML)
    bodies = {b.get("name"): b for b in tree.getroot().iter("body")}
    for i, off in enumerate(G.TORSO_OFFSETS):
        g = ET.SubElement(bodies[G.TORSO_BODY], "geom")
        g.set("type", "sphere"); g.set("size", str(G.TORSO_R))
        g.set("pos", " ".join(str(v) for v in off))
        g.set("contype", "0"); g.set("conaffinity", "0")
        g.set("rgba", "0.2 0.6 0.6 0.3"); g.set("name", f"torso_{i}")
    for body_name, r in G.CAPSULE_R.items():
        el = bodies[G.B(body_name)]
        g = ET.SubElement(el, "geom")
        g.set("type", "sphere"); g.set("size", str(r))
        g.set("contype", "0"); g.set("conaffinity", "0")
        g.set("rgba", "0.2 0.6 0.2 0.3"); g.set("name", f"cap_{body_name}")
    tmp = tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False, dir=str(G.TMP_DIR))
    tree.write(tmp.name); tmp.close()
    model = mujoco.MjModel.from_xml_path(tmp.name)
    Path(tmp.name).unlink()
    return model


def per_frame_metrics(csv: Path, fps: float, model, data):
    raw = np.loadtxt(csv, delimiter=",")
    assert raw.shape[1] == 36, raw.shape
    root_pos, quat_xyzw, joints = raw[:, :3], raw[:, 3:7], raw[:, 7:]
    T = len(joints)
    limits = G.urdf_limits()
    jid = {n: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in F1_JOINT_NAMES}
    qadr = np.array([model.jnt_qposadr[jid[n]] for n in F1_JOINT_NAMES])
    jlo = np.array([limits[n][0] for n in F1_JOINT_NAMES])
    jup = np.array([limits[n][1] for n in F1_JOINT_NAMES])
    bid = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, G.B(f"{s}_ankle_roll_link")) for s in ("left", "right")}
    assert all(v >= 0 for v in bid.values()), bid
    hip_id = {s: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{s}_hip_yaw") for s in ("left", "right")}
    geom_names = {n: f"cap_{n}" for n in G.CAPSULE_R}
    geom_names.update({f"torso_{i}": f"torso_{i}" for i in range(len(G.TORSO_OFFSETS))})
    gid = {}
    for a, b in G.CHECK_PAIRS:
        for k in (a, b):
            if k not in gid:
                gid[k] = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_names[k])

    sole_z = {s: np.full((T, 4), np.nan) for s in ("left", "right")}
    hip_pos = {s: np.full((T, 3), np.nan) for s in ("left", "right")}
    ank_pos = {s: np.full((T, 3), np.nan) for s in ("left", "right")}
    self_min = np.full(T, np.inf)
    self_pair = [None] * T
    lim_bad = np.zeros(T, bool)
    q = np.zeros(model.nq)
    for f in range(T):
        mujoco.mj_resetData(model, data)
        q[:] = data.qpos
        q[0:3] = root_pos[f]
        qq = quat_xyzw[f] / np.linalg.norm(quat_xyzw[f])
        q[3:7] = [qq[3], qq[0], qq[1], qq[2]]
        q[qadr] = joints[f]
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        for side in ("left", "right"):
            Rm = data.xmat[bid[side]].reshape(3, 3)
            hip_pos[side][f] = data.xpos[hip_id[side]]
            ank_pos[side][f] = data.xpos[bid[side]]
            for k, s in enumerate(G.FOOT_CORNERS[side]):
                sole_z[side][f, k] = (data.xpos[bid[side]] + Rm @ np.array(s))[2]
        for a, b in G.CHECK_PAIRS:
            d = mujoco.mj_geomDistance(model, data, gid[a], gid[b], 5.0, None)
            if d < self_min[f]:
                self_min[f], self_pair[f] = d, [a, b]
        lim_bad[f] = bool(((joints[f] < jlo - 1e-4) | (joints[f] > jup + 1e-4)).any())

    dt = 1.0 / fps
    dq = np.abs(np.diff(joints, axis=0)) / dt
    def _vmargin(n):
        return 2.0 if "wrist" in n else 1.5
    vlim = np.array([_vmargin(n) * limits[n][2] for n in F1_JOINT_NAMES])
    vel_bad = (dq > vlim).any(axis=1)  # length T-1, aligned to frame f (between f and f+1)

    return dict(root_pos=root_pos, quat_xyzw=quat_xyzw, joints=joints, sole_z=sole_z,
                hip_pos=hip_pos, ank_pos=ank_pos, self_min=self_min, self_pair=self_pair,
                lim_bad=lim_bad, vel_bad=vel_bad, dt=dt, limits=limits, jlo=jlo, jup=jup)


def flatness(m, a, b, args):
    out = {}
    for side in ("left", "right"):
        cz = m["sole_z"][side][a:b]
        contact = cz.min(axis=1) < args.contact_h
        stance = np.zeros(b - a, bool)
        i = 0
        while i < b - a:
            if contact[i]:
                j = i
                while j < b - a and contact[j]:
                    j += 1
                if j - i >= args.stance_min_frames:
                    stance[i:j] = True
                i = j
            else:
                i += 1
        use = stance if stance.sum() >= 10 else contact
        heel = cz[:, [0, 2]].mean(axis=1)
        toe = cz[:, [1, 3]].mean(axis=1)
        medial = cz[:, [2, 3]].mean(axis=1)
        lateral = cz[:, [0, 1]].mean(axis=1)
        pitch = np.degrees(np.arctan2(heel - toe, G.FOOT_LEN))
        roll = np.degrees(np.arctan2(medial - lateral, G.FOOT_WID))
        flat = (np.abs(pitch[use]) <= args.pitch_lim) & (np.abs(roll[use]) <= args.roll_lim)
        flat_pct = float(flat.mean() * 100) if use.any() else 0.0
        out[side] = dict(phase="stance" if stance.sum() >= 10 else "contact", frames=int(use.sum()),
                         flat_pct=flat_pct)
        out[side]["pass"] = bool(flat_pct >= args.flat_min_pct and use.sum() >= 10)
    return out


def gate_window(m, a, b, args):
    j = m["joints"][a:b]
    rep = {}
    rep["ground"] = dict(min_z=float(m["sole_z"]["left"][a:b].min() - G.FOOT_SPHERE_R),
                         min_z_r=float(m["sole_z"]["right"][a:b].min() - G.FOOT_SPHERE_R))
    rep["ground"]["pass"] = rep["ground"]["min_z"] > -args.penetration_tol and \
        rep["ground"]["min_z_r"] > -args.penetration_tol
    sm = m["self_min"][a:b]
    k = int(np.argmin(sm))
    rep["self"] = dict(min_dist=float(sm[k]), pair=m["self_pair"][a + k],
                       pass_=bool(sm[k] > args.self_clearance))
    rep["self"]["pass"] = rep["self"].pop("pass_")
    rep["limits"] = dict(violations=int(m["lim_bad"][a:b].sum()))
    rep["limits"]["pass"] = rep["limits"]["violations"] == 0
    rep["vel"] = dict(bad=int(m["vel_bad"][a:max(b - 1, a + 1)].sum()))
    rep["vel"]["pass"] = rep["vel"]["bad"] == 0
    z = m["root_pos"][a:b, 2]
    rep["height"] = dict(min=float(z.min()), max=float(z.max()),
                         pass_=bool((z >= args.height_range[0]).all() and (z <= args.height_range[1]).all()))
    rep["height"]["pass"] = rep["height"].pop("pass_")
    qn = m["quat_xyzw"][a:b] / np.linalg.norm(m["quat_xyzw"][a:b], axis=1, keepdims=True)
    dots = np.minimum((qn[1:] * qn[:-1]).sum(1).min(), 1.0) if b - a > 1 else 1.0
    rep["continuity"] = dict(min_dot=float(dots), pass_=bool(np.isfinite(j).all() and dots > 0.1))
    rep["continuity"]["pass"] = rep["continuity"].pop("pass_")
    fl = flatness(m, a, b, args)
    rep["flatness"] = dict(left=fl["left"], right=fl["right"], pass_=fl["left"]["pass"] and fl["right"]["pass"])
    rep["flatness"]["pass"] = rep["flatness"].pop("pass_")
    # arm quality
    aj = j[:, G.ARM_JOINT_IDX]
    an = [F1_JOINT_NAMES[i] for i in G.ARM_JOINT_IDX]
    alo = np.array([m["limits"][n][0] for n in an])
    aup = np.array([m["limits"][n][1] for n in an])
    sat = (aj <= alo[None, :] + args.arm_sat_margin) | (aj >= aup[None, :] - args.arm_sat_margin)
    any_sat = float(sat.any(axis=1).mean() * 100)
    roms = {}
    for k, n in enumerate(an):
        if "wrist" in n:
            roms[n] = float((np.percentile(aj[:, k], 95) - np.percentile(aj[:, k], 5)) / (aup[k] - alo[k]) * 100)
    rep["arm"] = dict(any_sat_pct=any_sat, wrist_rom=roms,
                      pass_=bool(any_sat <= args.arm_sat_max_pct and all(v >= args.arm_dead_rom_pct for v in roms.values())))
    rep["arm"]["pass"] = rep["arm"].pop("pass_")
    # stance openness
    hipL, hipR = m["hip_pos"]["left"][a:b], m["hip_pos"]["right"][a:b]
    ankL, ankR = m["ank_pos"]["left"][a:b], m["ank_pos"]["right"][a:b]
    hv = hipL - hipR
    hw = np.linalg.norm(hv, axis=1)
    lat = hv / hw[:, None]
    gap = np.sum((ankL - ankR) * lat, axis=1)
    ratio = gap / hw
    cl = m["sole_z"]["left"][a:b].min(axis=1) < args.contact_h
    cr = m["sole_z"]["right"][a:b].min(axis=1) < args.contact_h
    ds = cl & cr
    dsr = ratio[ds] if ds.sum() >= 10 else ratio
    li = F1_JOINT_NAMES.index("left_hip_roll_joint")
    ri = F1_JOINT_NAMES.index("right_hip_roll_joint")
    thr = np.radians(args.adduct_deg)
    both = (j[:, li] < -thr) & (j[:, ri] > thr)
    rep["stance"] = dict(ds_frames=int(ds.sum()), mean=float(dsr.mean()), p5=float(np.percentile(dsr, 5)),
                         adduct_pct=float(both.mean() * 100),
                         pass_=bool(dsr.mean() >= args.stance_gap_min and np.percentile(dsr, 5) >= args.stance_gap_p5_min
                                    and both.mean() * 100 <= args.adduct_max_pct and ds.sum() >= 10))
    rep["stance"]["pass"] = rep["stance"].pop("pass_")
    ig = set(x for x in args.ignore_gates.split(",") if x)
    rep["pass"] = all(rep[k]["pass"] for k in
                      ("ground", "self", "limits", "vel", "height", "continuity", "flatness", "arm", "stance")
                      if k not in ig)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--win", type=int, default=450, help="window length (frames @ fps)")
    ap.add_argument("--stride", type=int, default=150)
    ap.add_argument("--penetration_tol", type=float, default=0.015)
    ap.add_argument("--self_clearance", type=float, default=0.005)
    ap.add_argument("--vel_margin", type=float, default=1.5)
    ap.add_argument("--height_range", type=float, nargs=2, default=(0.35, 1.0))
    ap.add_argument("--contact_h", type=float, default=0.03)
    ap.add_argument("--stance_min_frames", type=int, default=5)
    ap.add_argument("--pitch_lim", type=float, default=10.0)
    ap.add_argument("--roll_lim", type=float, default=8.0)
    ap.add_argument("--flat_min_pct", type=float, default=90.0)
    ap.add_argument("--arm_sat_margin", type=float, default=0.05)
    ap.add_argument("--arm_sat_max_pct", type=float, default=10.0)
    ap.add_argument("--arm_dead_rom_pct", type=float, default=2.0)
    ap.add_argument("--stance_gap_min", type=float, default=1.15)
    ap.add_argument("--stance_gap_p5_min", type=float, default=0.70)
    ap.add_argument("--adduct_deg", type=float, default=3.0)
    ap.add_argument("--adduct_max_pct", type=float, default=3.0)
    ap.add_argument("--ignore_gates", default="", help="comma list of gates not enforced (diagnostics)")
    args = ap.parse_args()

    model = build_model()
    data = mujoco.MjData(model)
    m = per_frame_metrics(args.csv, args.fps, model, data)
    T = len(m["joints"])
    wins = []
    for a in range(0, max(T - args.win, 0) + 1, args.stride):
        b = min(a + args.win, T)
        if b - a < args.win * 0.75:
            continue
        r = gate_window(m, a, b, args)
        r["start"], r["end"] = a, b
        wins.append(r)
    passed = [w for w in wins if w["pass"]]
    # greedy non-overlapping selection, longest first
    passed.sort(key=lambda w: (w["end"] - w["start"]), reverse=True)
    used = np.zeros(T, bool)
    sel = []
    for w in passed:
        a, b = w["start"], w["end"]
        if not used[a:b].any():
            used[a:b] = True
            sel.append(w)
    sel.sort(key=lambda w: w["start"])
    out = dict(csv=str(args.csv), frames=T, fps=args.fps, win=args.win, stride=args.stride,
               ignore_gates=args.ignore_gates,
               n_windows=len(wins), n_pass=len(passed), selected=[{k: w[k] for k in ("start", "end")} for w in sel],
               selected_s=float(sum(w["end"] - w["start"] for w in sel) / args.fps),
               windows=wins)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    print(f"{args.csv.name}: {T}f, {len(wins)} windows, {len(passed)} pass, "
          f"selected {len(sel)} = {out['selected_s']:.1f}s")
    for w in sel:
        print(f"  [{w['start']},{w['end']}) {w['end']-w['start']}f")


if __name__ == "__main__":
    main()
