#!/usr/bin/env python3
"""Dance acceptance (docs/acceptance_plan.md, pre-registered A1-A5) on MuJoCo sim2sim rollouts.

For each clip of the plan:
  1. tools/f1_sim2sim.py replays the exported ONNX policy in plain MuJoCo -> metrics json + rollout npz
     (+ <stem>_motion.npz: the rollout as a PartyMimic motion file, 50 Hz);
  2. tools/check_motion_f1.py (the nine retargeting gates' formulas) runs on that motion file and its
     gate 7 (foot flatness over stance frames) and gate 9 (both hip rolls adducted, % of frames) become
     A4 / A5. Stance = contact runs of >= --stance_min_frames frames; the retargeting gates used 5 frames
     at 30 fps (0.167 s), the rollout is 50 Hz, so the same physical definition is 8 frames (0.16 s).
     The 5-frame number is reported next to it as a sensitivity row only.
  3. A1 fall: base z > 0.25 m and tilt < 0.8 rad at every recorded frame (0/4 clips may fall);
     A2 local body position error (each side in its own pelvis heading frame): trained <= 50 mm,
     held-out <= 80 mm; A3 joint tracking error <= 0.20 rad; A4 both feet >= 90 %; A5 <= 3 %.
Thresholds are read from docs/acceptance_plan.md's table and never adjusted here.

Usage:
  python tools/f1_acceptance.py --onnx_dir eval/f1_dance_v1 --tag f1_dance_v1 --out eval/f1_dance_v1/acceptance
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PLAN = REPO / "docs" / "acceptance_plan.md"

# docs/acceptance_plan.md section 1 (clip, kind, trained)
CLIPS = [
    ("dance1_subject3_255_706", "trained"),
    ("dance2_subject1_352_795", "trained"),
    ("dance2_subject4_2835_2925", "trained"),
    ("dance2_subject3_300_450", "held-out"),
]
THRESH = {  # docs/acceptance_plan.md section 2
    "A2_trained_mm": 50.0, "A2_heldout_mm": 80.0, "A3_rad": 0.20, "A4_pct": 90.0, "A5_pct": 3.0,
}


def parse_plan_thresholds() -> dict:
    """Re-read the numbers of the plan so a silent edit of the plan shows up as a mismatch."""
    txt = PLAN.read_text()
    got = {}
    import re
    m = re.search(r"训练段 ≤ \*\*(\d+) mm\*\*；held-out ≤ \*\*(\d+) mm\*\*", txt)
    if m:
        got["A2_trained_mm"], got["A2_heldout_mm"] = float(m.group(1)), float(m.group(2))
    m = re.search(r"\| A3 \| 关节跟踪误差 \| ≤ \*\*([\d.]+) rad\*\*", txt)
    if m:
        got["A3_rad"] = float(m.group(1))
    m = re.search(r"\| A4 \|[^|]*\| 左右脚均 ≥ \*\*(\d+)%\*\*", txt)
    if m:
        got["A4_pct"] = float(m.group(1))
    m = re.search(r"\| A5 \|[^|]*\| ≤ \*\*(\d+)%\*\*", txt)
    if m:
        got["A5_pct"] = float(m.group(1))
    for k, v in got.items():
        if THRESH[k] != v:
            raise SystemExit(f"acceptance_plan.md threshold {k}={v} differs from this script's {THRESH[k]}")
    return got


def run(cmd: list[str]) -> subprocess.CompletedProcess:
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, text=True, capture_output=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx_dir", required=True, help="directory with <tag>_<clip>.onnx files")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True, help="output directory (rollouts, json, acceptance.md)")
    ap.add_argument("--stance_min_frames", type=int, default=8, help="gate-7 stance run length at 50 Hz (= 5 frames @ 30 fps)")
    ap.add_argument("--python", default=sys.executable)
    args = ap.parse_args()
    th = parse_plan_thresholds()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    py = args.python

    rows, details = [], {}
    all_pass = True
    for clip, kind in CLIPS:
        onnx = Path(args.onnx_dir) / f"{args.tag}_{clip}.onnx"
        if not onnx.exists():
            rows.append({"clip": clip, "kind": kind, "error": f"missing {onnx}"})
            all_pass = False
            continue
        stem = out / clip
        p = run([py, str(REPO / "tools" / "f1_sim2sim.py"), "--onnx", str(onnx), "--out", f"{stem}.npz",
                 "--json", f"{stem}.json"])
        if p.returncode != 0 or not Path(f"{stem}.json").exists():
            rows.append({"clip": clip, "kind": kind, "error": (p.stderr or p.stdout)[-800:]})
            all_pass = False
            continue
        sim = json.loads(Path(f"{stem}.json").read_text())
        gates = {}
        for n in sorted({args.stance_min_frames, 5}):
            g = run([py, str(REPO / "tools" / "check_motion_f1.py"), "--npz", f"{stem}_motion.npz", "--fps", "50",
                     "--stance_min_frames", str(n), "--json_out", f"{stem}_gates_smf{n}.json"])
            if g.returncode not in (0, 1, 2) or not Path(f"{stem}_gates_smf{n}.json").exists():
                rows.append({"clip": clip, "kind": kind, "error": "check_motion_f1 failed: " + (g.stderr or g.stdout)[-800:]})
                break
            gates[n] = json.loads(Path(f"{stem}_gates_smf{n}.json").read_text())
        if args.stance_min_frames not in gates:
            all_pass = False
            continue
        g = gates[args.stance_min_frames]
        flat_l = g["foot_flatness"]["left"]["flat_pct"]
        flat_r = g["foot_flatness"]["right"]["flat_pct"]
        stance_frames = [g["foot_flatness"]["left"]["frames"], g["foot_flatness"]["right"]["frames"]]
        a4_valid = min(stance_frames) >= 10  # gate 7's own minimum sample: fewer stance frames = no verdict
        adduct = g["stance_openness"]["both_hiproll_adducted"]["pct"]
        g5 = gates.get(5, {})
        a2_mm = sim["body_err_local_m"] * 1000.0
        a2_lim = th["A2_heldout_mm"] if kind == "held-out" else th["A2_trained_mm"]
        checks = {
            "A1_no_fall": sim["success"],
            "A2_local_body_err": a2_mm <= a2_lim,
            "A3_joint_err": sim["joint_err_rad"] <= th["A3_rad"],
            "A4_foot_flat": bool(a4_valid and min(flat_l, flat_r) >= th["A4_pct"]),
            "A5_hiproll_adduct": adduct <= th["A5_pct"],
        }
        ok = all(checks.values())
        all_pass &= ok
        rows.append({
            "clip": clip, "kind": kind, "frames_played": sim["frames_played"], "frames_total": sim["frames_total"],
            "A1_fall": not sim["success"], "fell_at": sim["fell_at"], "min_base_z": sim["min_base_z"],
            "max_tilt_rad": sim["max_tilt_rad"],
            "A2_local_mm": a2_mm, "A2_limit_mm": a2_lim, "A3_joint_rad": sim["joint_err_rad"], "A3_limit": th["A3_rad"],
            "A4_flat_l": flat_l, "A4_flat_r": flat_r, "A4_limit": th["A4_pct"], "A4_stance_frames": stance_frames,
            "A4_valid": a4_valid,
            "A5_adduct_pct": adduct, "A5_limit": th["A5_pct"],
            "A4_sensitivity_smf5": [g5.get("foot_flatness", {}).get("left", {}).get("flat_pct"),
                                    g5.get("foot_flatness", {}).get("right", {}).get("flat_pct")],
            "world_body_err_m": sim["body_err_m"], "root_err_m": sim["root_err_m"],
            "checks": checks, "pass": ok,
        })
        details[clip] = {"sim": sim, "gates": g}
    res = {"tag": args.tag, "onnx_dir": args.onnx_dir, "stance_min_frames": args.stance_min_frames,
           "thresholds": th, "clips": rows, "pass": all_pass}
    (out / "acceptance.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))

    # markdown
    L = [f"# Sim2sim acceptance — {args.tag} (MuJoCo CPU, A1–A5 of docs/acceptance_plan.md)", "",
         f"Stance run length for A4: {args.stance_min_frames} frames @ 50 Hz (= 5 frames @ 30 fps). "
         f"Overall: **{'PASS' if all_pass else 'FAIL'}**", "",
         "| clip | kind | frames | A1 fall | A2 local err mm (limit) | A3 joint err rad (limit) | A4 flat % L/R (limit) | A5 adduct % (limit) | pass |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if "error" in r:
            L.append(f"| {r['clip']} | {r['kind']} | — | error: {r['error'][:120]} | | | | | ✗ |")
            continue
        L.append(f"| {r['clip']} | {r['kind']} | {r['frames_played']}/{r['frames_total']} | "
                 f"{'FELL@' + str(r['fell_at']) if r['A1_fall'] else 'no'} | "
                 f"{r['A2_local_mm']:.1f} ({r['A2_limit_mm']:.0f}) | {r['A3_joint_rad']:.4f} ({r['A3_limit']}) | "
                 f"{r['A4_flat_l']:.1f}/{r['A4_flat_r']:.1f} ({r['A4_limit']:.0f}){'' if r['A4_valid'] else ' INVALID(<10 stance frames)'} | "
                 f"{r['A5_adduct_pct']:.2f} ({r['A5_limit']}) | "
                 f"{'✓' if r['pass'] else '✗'} |")
    L += ["", "A4 stance frames per clip (gate 7 sample size, minimum 10): " +
          "; ".join(f"{r['clip']}: {r['A4_stance_frames'][0]}/{r['A4_stance_frames'][1]}" for r in rows if "error" not in r), "",
          "A4 with the checker's default stance run (5 frames @ 50 Hz, sensitivity only): " +
          "; ".join(f"{r['clip']}: {r['A4_sensitivity_smf5'][0]:.1f}/{r['A4_sensitivity_smf5'][1]:.1f}" for r in rows if "error" not in r)]
    (out / "acceptance.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
