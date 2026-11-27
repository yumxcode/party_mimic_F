#!/usr/bin/env python
"""Bootstrap PartyMimic + the F1 29DOF additions on gradmotion, then train.

gradmotion clones this repository into /workspace/party_mimic_F and runs
``gm-run party_mimic_F/scripts/f1_train.py [args]``. This script

  1. clones upstream PartyMimic (pinned) into /workspace/PartyMimic,
  2. copies in src/partymimic/f1, assets/f1 and data/motions/f1,
  3. applies the two backward-compatible edits of tools/patch_partymimic.py,
  4. ``uv sync`` (the authors' lock: torch 2.9.0+cu128, mjlab 1.6, mujoco-warp 3.11),
  5. runs ``python -m partymimic.train --robot f1 ...``,
  6. copies the checkpoints to /workspace/output/<name>.pt, the directory the gradmotion SDK
     uploads as model files.

Usage (inside the task):
    gm-run party_mimic_F/scripts/f1_train.py --iterations 12000 --tag f1_dance_s0
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PM_REPO = "https://github.com/Roboparty/PartyMimic.git"
PM_REF = "main"
WS = Path("/workspace")


def find_repo() -> Path:
    """Locate this repository inside the task container (gradmotion may clone it under a different
    name, and gm-run may resolve the script through a symlink)."""
    marker = Path("src/partymimic/f1/f1_constants.py")
    cands = []
    if os.environ.get("PM_F1_REPO"):
        cands.append(Path(os.environ["PM_F1_REPO"]))
    try:
        cands.append(Path(__file__).resolve().parent.parent)
    except NameError:
        pass
    cands += [Path.cwd(), WS / "party_mimic_F", WS]
    for c in cands:
        for d in (c, *c.parents):
            if (d / marker).exists():
                return d
    for d in sorted(WS.glob("*")):
        if (d / marker).exists():
            return d
    print("[boot] /workspace listing:", [p.name for p in WS.glob("*")], flush=True)
    raise SystemExit(f"cannot find the repository (marker {marker}); __file__={__file__} cwd={Path.cwd()}")


HERE = find_repo()


def run(cmd, cwd=None, env=None, check=True, tail=None):
    print(f"[boot] $ {' '.join(map(str, cmd))}  (cwd={cwd})", flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True)
    out = (p.stdout or "") + (p.stderr or "")
    if tail:
        out = "\n".join(out.splitlines()[-tail:])
    print(out, flush=True)
    print(f"[boot] rc={p.returncode} in {time.time()-t0:.1f}s", flush=True)
    if check and p.returncode != 0:
        raise SystemExit(f"command failed: {cmd}")
    return p


def disable_mjlab_cuda_graph(pm: Path) -> None:
    """The 4090D hosts run a CUDA driver that warp reports as >= 12.4 but that rejects conditional
    graph nodes ("Conditional graph nodes require CUDA driver 12.4+", raised from mjlab's
    create_graph -> mjwarp.step -> wp.capture_while). Make mjlab's own driver check fail so no graph
    is captured: identical physics, no capture."""
    hits = list(pm.glob(".venv/lib/python*/site-packages/mjlab/sim/sim.py"))
    if not hits:
        print("[boot] mjlab sim.py not found; skipping the cuda-graph patch", flush=True)
        return
    for path in hits:
        s = path.read_text()
        if "_GRAPH_CAPTURE_MIN_DRIVER = (99, 0)" in s:
            print(f"[boot] cuda-graph patch already applied: {path}", flush=True)
            continue
        assert "_GRAPH_CAPTURE_MIN_DRIVER = (12, 4)" in s, path
        path.write_text(s.replace("_GRAPH_CAPTURE_MIN_DRIVER = (12, 4)", "_GRAPH_CAPTURE_MIN_DRIVER = (99, 0)"))
        print(f"[boot] patched {path}: CUDA graph capture disabled (driver check)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="f1_dance_s0")
    ap.add_argument("--iterations", type=int, default=12000)
    ap.add_argument("--num_envs", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--motion_dir", default=None)
    ap.add_argument("--extra", default="", help="extra train.py flags, space separated")
    ap.add_argument("--skip_sync", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="2 iterations, 64 envs, no checkpoint copy")
    args = ap.parse_args()

    pm = WS / "PartyMimic"
    if not (pm / ".git").exists():
        run(["git", "clone", "--depth", "1", "--branch", PM_REF, PM_REPO, str(pm)], cwd=WS)
    print("[boot] PartyMimic HEAD:", run(["git", "rev-parse", "HEAD"], cwd=pm, check=False).stdout.strip(), flush=True)

    # 2. copy the F1 additions
    for rel in ("src/partymimic/f1", "assets/f1", "data/motions/f1"):
        src, dst = HERE / rel, pm / rel
        if not src.exists():
            raise SystemExit(f"missing {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"[boot] copied {rel}", flush=True)

    # 3. patch PartyMimic (feet bodies, cuda-graph switch)
    run([sys.executable, str(HERE / "tools/patch_partymimic.py"), str(pm)], cwd=WS)
    run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv"], cwd=WS, check=False, tail=5)

    # 4. environment
    if not args.skip_sync:
        run(["uv", "sync", "--frozen"], cwd=pm, tail=40)
    disable_mjlab_cuda_graph(pm)

    # 5. train
    extra = args.extra.split()
    cmd = ["uv", "run", "--frozen", "python", "-u", "-m", "partymimic.train",
           "--robot", "f1", "--log_dir", f"logs/{args.tag}", "--seed", str(args.seed),
           "--iterations", str(args.iterations), "--num_envs", str(args.num_envs),
           "--motion_dir", args.motion_dir or "data/motions/f1", *extra]
    log_dir = pm / "logs" / args.tag
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "bootstrap_cmd.json").write_text(json.dumps({"cmd": cmd, "pm_ref": PM_REF}, indent=2))
    run(cmd, cwd=pm, check=False, tail=60)

    # 6. publish checkpoints where the gradmotion SDK looks
    if args.smoke:
        print("[boot] smoke run: no checkpoint copy", flush=True)
        return
    out = WS / "output"
    out.mkdir(exist_ok=True)
    for p in sorted(log_dir.glob("*.pt")):
        dst = out / f"{args.tag}_{p.stem}.pt"
        shutil.copy2(p, dst)
        print(f"[boot] published {dst} ({dst.stat().st_size/1e6:.1f} MB)", flush=True)
    print("[boot] DONE", flush=True)


if __name__ == "__main__":
    main()
