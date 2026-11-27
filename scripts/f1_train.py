#!/usr/bin/env python3
"""Bootstrap PartyMimic + the F1 29DOF additions on gradmotion, then train.

gradmotion clones this repository into /workspace/party_mimic_F and runs
``gm-run party_mimic_F/scripts/f1_train.py [args]``. This script

  1. clones upstream PartyMimic (main) into /workspace/PartyMimic,
  2. copies in src/partymimic/f1, assets/f1, data/motions/f1 and data/testsets/f1_holdout,
  3. applies the backward-compatible edits of tools/patch_partymimic.py (feet bodies, eager-collect
     switch, SimulationCfg graph switch),
  4. ``uv sync --frozen`` (the authors' lock: torch 2.9.0+cu128, mjlab 1.6, mujoco-warp 3.11),
  5. turns the CUDA-graph captures that need conditional graph nodes off (the 4090D hosts run a CUDA
     driver < 12.4): mjlab's own capture (driver check) and MuJoCo-Warp's conditional solver loop
     (``Model.opt.graph_conditional = False`` = mujoco_warp's own recommendation for driver < 12.4;
     the solver then runs a fixed-iteration loop that PartyMimic's collection graph can capture),
  6. runs ``python -m partymimic.train --robot f1 ...`` (``--smoke_then_full``: a 2-iteration / 64-env
     smoke first; if its collection-graph capture fails, once more with ``PM_F1_EAGER_COLLECT=1``),
  7. exports one ONNX per clip (policy + that reference motion baked in),
  8. publishes checkpoints / ONNX / the training log to /workspace/output (the directory the
     gradmotion SDK uploads); a background thread republishes new checkpoints every 60 s.

Usage (inside the task):
    gm-run party_mimic_F/scripts/f1_train.py --smoke_then_full --iterations 12000 --num_envs 4096 --tag f1_dance_v1
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

PM_REPO = "https://github.com/Roboparty/PartyMimic.git"
PM_REF = "main"
WS = Path("/workspace")
COND_GRAPH_ERR = "Conditional graph nodes require CUDA driver"

BOOT: list[str] = []  # every bootstrap line, re-printed at the end (the task log keeps only its last 200 KB)


def blog(msg: str) -> None:
    line = f"[boot] {msg}"
    BOOT.append(line)
    print(line, flush=True)


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
    blog(f"/workspace listing: {[p.name for p in WS.glob('*')]}")
    raise SystemExit(f"cannot find the repository (marker {marker}); __file__={__file__} cwd={Path.cwd()}")


HERE = find_repo()


def run(cmd, cwd=None, env=None, check=True, tail=None, timeout=None):
    blog(f"$ {' '.join(map(str, cmd))}  (cwd={cwd})")
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        class _TimedOut:
            returncode, stdout, stderr = 124, (e.stdout or "") if isinstance(e.stdout, str) else "", f"timed out after {timeout} s"
        p = _TimedOut()
    out = (p.stdout or "") + (p.stderr or "")
    shown = "\n".join(out.splitlines()[-tail:]) if tail else out
    print(shown, flush=True)
    blog(f"rc={p.returncode} in {time.time() - t0:.1f}s")
    if check and p.returncode != 0:
        raise SystemExit(f"command failed: {cmd}")
    return p


def clone_partyMimic(pm: Path, tries: int = 4) -> None:
    """git clone of upstream PartyMimic with retries: the gradmotion hosts see transient
    'GnuTLS recv error (-110)' against github.com (TASK_20261127_681 died on its first attempt)."""
    for k in range(1, tries + 1):
        if (pm / ".git").exists():
            return
        p = run(["git", "clone", "--depth", "1", "--branch", PM_REF, PM_REPO, str(pm)], cwd=WS, check=False)
        if p.returncode == 0 and (pm / ".git").exists():
            return
        blog(f"git clone attempt {k}/{tries} failed (rc={p.returncode}); retrying in 30 s")
        shutil.rmtree(pm, ignore_errors=True)
        time.sleep(30)
    raise SystemExit(f"could not clone {PM_REPO} after {tries} attempts")


TORCH_MIRROR = "https://mirrors.aliyun.com/pytorch-wheels/cu128/"
TORCH_WHEEL_MIN_BYTES = 8e8  # the cu128 torch 2.9.0 wheel is 900 MB; a smaller file is a failed download


def uv_sync(pm: Path, attempts: int = 1, timeout: int = 720) -> str:
    """``uv sync --frozen`` (the authors' lock). The gradmotion hosts' route to download.pytorch.org
    stalled for hours on 2026-11-27 (8 MB range: 0 B in 60 s; PyPI was fine), so after ``attempts``
    bounded tries everything but torch is installed from the lock and the torch wheel (the very file
    the lock names, sha256-checked against the lock) comes from the Aliyun mirror of the PyTorch
    wheels. Callers must then use ``uv run --no-sync`` (a sync would re-download torch)."""
    import hashlib
    import re
    env = dict(os.environ, UV_HTTP_TIMEOUT="300")
    for k in range(1, attempts + 1):
        p = run(["uv", "sync", "--frozen"], cwd=pm, env=env, tail=30, check=False, timeout=timeout)
        if p.returncode == 0:
            return "direct"
        blog(f"uv sync attempt {k}/{attempts} failed (rc={p.returncode})")
    lock = (pm / "uv.lock").read_text()
    m = re.search(r'url = "https://download-r2\.pytorch\.org/whl/cu128/(torch-2\.9\.0%2Bcu128-cp312-cp312-manylinux_2_28_x86_64\.whl)", hash = "sha256:([0-9a-f]{64})"', lock)
    assert m, "torch wheel entry not found in uv.lock"
    wheel_name, wheel_sha = m.group(1), m.group(2)
    # the killed sync leaves .venv/.lock behind, after which uv "audits" the empty venv as up to date
    # (0.3 s, rc=0, nothing installed: TASK_20261127_762/763): start from a clean venv
    shutil.rmtree(pm / ".venv", ignore_errors=True)
    p = run(["uv", "sync", "--frozen", "--no-install-package", "torch"], cwd=pm, env=env, tail=30, check=False, timeout=timeout)
    if p.returncode != 0:
        raise SystemExit("uv sync (without torch) failed")
    whl = WS / "torch_wheel" / wheel_name.replace("%2B", "+")  # wheel filenames carry a literal '+', not %2B
    whl.parent.mkdir(exist_ok=True)
    for k in range(1, 4):
        p = run(["curl", "-L", "--retry", "3", "-C", "-", "-s", "-S", "-o", str(whl), TORCH_MIRROR + wheel_name], cwd=WS,
                check=False, timeout=3600)
        if p.returncode == 0 and whl.exists() and whl.stat().st_size > TORCH_WHEEL_MIN_BYTES:
            break
        blog(f"torch wheel download attempt {k}/3 failed (rc={p.returncode}, {whl.stat().st_size if whl.exists() else 0} B)")
        time.sleep(30)
    h = hashlib.sha256(whl.read_bytes()).hexdigest()
    if h != wheel_sha:
        raise SystemExit(f"torch wheel sha256 {h} != lock {wheel_sha}")
    blog(f"torch wheel from {TORCH_MIRROR}: {whl.stat().st_size / 1e6:.0f} MB, sha256 ok")
    # --no-deps: the sync above installed torch's dependencies (nvidia-* wheels from PyPI)
    p = run(["uv", "pip", "install", "--python", str(pm / ".venv/bin/python"), "--no-deps", str(whl)], cwd=pm, tail=8)
    run(["uv", "run", "--no-sync", "python", "-c", "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"], cwd=pm, tail=3)
    return "aliyun-mirror"


def disable_mjlab_cuda_graph(pm: Path) -> None:
    """The 4090D hosts run a CUDA driver < 12.4 (no conditional graph nodes). mjlab's own graph capture
    (create_graph -> wp.ScopedCapture -> mjwarp.step) would trip on it; make mjlab's driver check fail so
    its step() runs ``mjwarp.step`` directly (identical physics; PartyMimic captures that call itself)."""
    hits = sorted(pm.glob(".venv/lib/python*/site-packages/mjlab/sim/sim.py"))
    if not hits:
        blog("mjlab sim.py not found; skipping the cuda-graph patch")
        return
    for path in hits:
        s = path.read_text()
        if "_GRAPH_CAPTURE_MIN_DRIVER = (99, 0)" in s:
            blog(f"cuda-graph patch already applied: {path}")
            continue
        assert "_GRAPH_CAPTURE_MIN_DRIVER = (12, 4)" in s, path
        path.write_text(s.replace("_GRAPH_CAPTURE_MIN_DRIVER = (12, 4)", "_GRAPH_CAPTURE_MIN_DRIVER = (99, 0)"))
        blog(f"patched {path}: mjlab CUDA graph capture disabled (driver check)")


def disable_mjwarp_conditional_graph(pm: Path) -> None:
    """mujoco_warp's solver runs its Newton iterations as ``wp.capture_while`` (a conditional graph
    node, CUDA driver >= 12.4) whenever ``Model.opt.graph_conditional`` is set (its put_model default).
    Its own driver check says: "conditional graph nodes are not available for < 12.4 - Model.opt.
    graph_conditional should be set to False": the solver then runs a fixed ``iterations`` loop whose
    kernels early-return for converged worlds, and PartyMimic's collection CUDA graph captures it as
    plain nodes."""
    hits = sorted(pm.glob(".venv/lib/python*/site-packages/mujoco_warp/_src/io.py"))
    assert hits, "mujoco_warp io.py not found"
    for path in hits:
        s = path.read_text()
        old = "  opt.graph_conditional = True\n"
        new = "  opt.graph_conditional = False  # F1 on gradmotion: driver < 12.4 has no conditional graph nodes\n"
        if new in s:
            blog(f"graph_conditional patch already applied: {path}")
            continue
        assert old in s, f"graph_conditional anchor not found in {path}"
        path.write_text(s.replace(old, new, 1))
        blog(f"patched {path}: opt.graph_conditional = False (fixed-iteration solver loop, capturable)")


class Publisher:
    """Copies checkpoints of the run to /workspace/output as ``<tag>_<stem>.pt``; while the run is
    going, a daemon thread re-checks every ``period`` s (PartyMimic writes them atomically), so a run
    that is cut short still leaves usable model files."""

    def __init__(self, log_dir: Path, out: Path, prefix: str, period: float = 60.0):
        self.log_dir, self.out, self.prefix, self.period = log_dir, out, prefix, period
        self.seen: dict[str, tuple] = {}
        self.stop = threading.Event()
        self.out.mkdir(parents=True, exist_ok=True)

    def publish_ckpts(self) -> list[Path]:
        done = []
        for p in sorted(self.log_dir.glob("*.pt")):
            st = p.stat()
            key = (st.st_size, int(st.st_mtime))
            if self.seen.get(p.name) == key:
                continue
            dst = self.out / f"{self.prefix}_{p.stem}.pt"
            shutil.copy2(p, dst)
            self.seen[p.name] = key
            blog(f"published {dst.name} ({dst.stat().st_size / 1e6:.1f} MB)")
            done.append(dst)
        return done

    def _loop(self) -> None:
        while not self.stop.wait(self.period):
            try:
                self.publish_ckpts()
            except Exception as e:  # never kill the training from the publisher
                blog(f"publisher error: {e!r}")

    def start(self) -> None:
        threading.Thread(target=self._loop, daemon=True).start()

    def publish_as_pt(self, src: Path, name: str) -> None:
        """A non-.pt artifact (training log, ONNX, boot summary) travels with a .pt suffix so the SDK uploads it."""
        dst = self.out / f"{self.prefix}_{name}.pt"
        shutil.copy2(src, dst)
        blog(f"published {dst.name} ({dst.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="f1_dance_s0")
    ap.add_argument("--iterations", type=int, default=12000)
    ap.add_argument("--num_envs", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--motion_dir", default=None)
    ap.add_argument("--extra", default="", help="extra train.py flags, space separated")
    ap.add_argument("--ckpt_minutes", type=float, default=20.0,
                    help="PartyMimic --ckpt_minutes: also checkpoint every n minutes (0: only every --ckpt_every)")
    ap.add_argument("--skip_sync", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="smoke run only (2 iterations, 64 envs), no export/publish")
    ap.add_argument("--eager_collect", action="store_true",
                    help="force PM_F1_EAGER_COLLECT=1 (no collection CUDA graph) for all phases")
    ap.add_argument("--smoke_then_full", action="store_true",
                    help="run a 2-iteration smoke first; if it produces a checkpoint, continue with the full run")
    args = ap.parse_args()
    summary: dict = {"tag": args.tag}

    def finish() -> None:
        summary["boot"] = BOOT
        (WS / "boot_summary.json").write_text(json.dumps(summary, indent=1))
        print("\n[boot] ===== bootstrap summary (re-printed: the task log keeps only its tail) =====", flush=True)
        for line in BOOT:
            print(line, flush=True)
        print("[boot] ===== end of bootstrap summary =====", flush=True)

    t_boot = time.time()
    try:
        _main(args, summary)
    finally:
        summary["boot_seconds"] = round(time.time() - t_boot, 1)
        finish()


def _main(args, summary: dict) -> None:
    pm = WS / "PartyMimic"
    clone_partyMimic(pm)
    head = run(["git", "rev-parse", "HEAD"], cwd=pm, check=False).stdout.strip()
    blog(f"PartyMimic HEAD: {head}")
    summary["pm_head"] = head

    # 2. copy the F1 additions
    copy_dirs = ["src/partymimic/f1", "assets/f1", "data/motions/f1", "data/testsets/f1_holdout"]
    if args.motion_dir and args.motion_dir not in copy_dirs:
        copy_dirs.append(args.motion_dir)  # a variant library (e.g. data/motions/f1_flat85)
    for rel in copy_dirs:
        src, dst = HERE / rel, pm / rel
        if not src.exists():
            raise SystemExit(f"missing {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        blog(f"copied {rel} ({len(list(dst.glob('*')))} files)")

    # 3. patch PartyMimic (feet bodies, eager-collect switch, SimulationCfg graph switch)
    run([sys.executable, str(HERE / "tools/patch_partymimic.py"), str(pm)], cwd=WS)
    run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"], cwd=WS, check=False, tail=5)

    # 4. environment (uv's downloads stalled for >70 min on the gradmotion hosts on 2026-11-27 evening:
    #    bound each attempt and retry)
    if not args.skip_sync:
        blog(f"uv sync: {uv_sync(pm)}")
    disable_mjlab_cuda_graph(pm)
    disable_mjwarp_conditional_graph(pm)

    # 5. train
    extra = args.extra.split() + (["--ckpt_minutes", str(args.ckpt_minutes)] if args.ckpt_minutes > 0 else [])
    base_env = dict(os.environ)

    def run_train(tag: str, iterations: int, num_envs: int, check: bool, eager: bool, require_ckpt: bool):
        env = dict(base_env, PM_F1_EAGER_COLLECT="1" if eager else "0")
        cmd = ["uv", "run", "--no-sync", "python", "-u", "-m", "partymimic.train",
               "--robot", "f1", "--log_dir", f"logs/{tag}", "--seed", str(args.seed),
               "--iterations", str(iterations), "--num_envs", str(num_envs),
               "--motion_dir", args.motion_dir or "data/motions/f1", *extra]
        ld = pm / "logs" / tag
        ld.mkdir(parents=True, exist_ok=True)
        (ld / "bootstrap_cmd.json").write_text(json.dumps({"cmd": cmd, "pm_ref": PM_REF, "pm_head": head,
                                                           "PM_F1_EAGER_COLLECT": env["PM_F1_EAGER_COLLECT"]}, indent=2))
        blog(f"=== train {tag}: {iterations} iterations, {num_envs} envs, PM_F1_EAGER_COLLECT={env['PM_F1_EAGER_COLLECT']} ===")
        p = run(cmd, cwd=pm, env=env, check=check, tail=60)
        if require_ckpt and not (ld / "last.pt").exists():
            raise SystemExit(f"no checkpoint produced in {ld}")
        return p

    eager = bool(args.eager_collect)
    if args.smoke_then_full:
        blog("=== smoke phase (2 iterations, 64 envs) ===")
        if eager:
            run_train("_smoke", 2, 64, check=True, eager=True, require_ckpt=True)
        else:
            p = run_train("_smoke", 2, 64, check=False, eager=False, require_ckpt=False)
            if p.returncode != 0 or not (pm / "logs" / "_smoke" / "last.pt").exists():
                out = (p.stdout or "") + (p.stderr or "")
                if COND_GRAPH_ERR not in out:
                    raise SystemExit("smoke failed for a reason other than the conditional graph capture")
                blog(f"smoke failed with '{COND_GRAPH_ERR}': retrying with PM_F1_EAGER_COLLECT=1 (no collection graph)")
                eager = True
                shutil.rmtree(pm / "logs" / "_smoke", ignore_errors=True)
                run_train("_smoke", 2, 64, check=True, eager=True, require_ckpt=True)
        summary["eager_collect"] = eager
        blog(f"=== smoke ok (eager_collect={eager}), full run ===")

    log_dir = pm / "logs" / args.tag
    pub = Publisher(log_dir, WS / "output", args.tag)
    pub.start()
    if args.smoke:
        run_train(args.tag, 2, 64, check=False, eager=eager, require_ckpt=False)
        pub.stop.set()
        blog("smoke run: no export / checkpoint copy")
        return
    if args.smoke_then_full:
        p = run_train(args.tag, args.iterations, args.num_envs, check=False, eager=eager, require_ckpt=False)
    else:
        p = run_train(args.tag, args.iterations, args.num_envs, check=False, eager=eager, require_ckpt=False)
    summary["train_rc"] = p.returncode
    pub.stop.set()
    pub.publish_ckpts()

    # 6. export one ONNX per clip (policy + that reference motion baked in)
    ckpts = sorted(log_dir.glob("ckpt_*.pt"))
    ckpt = log_dir / "last.pt"
    if not ckpt.exists() and ckpts:
        ckpt = ckpts[-1]
        blog(f"no last.pt (train rc={p.returncode}): exporting the latest periodic checkpoint {ckpt.name}")
    if ckpt.exists():
        clips = sorted((pm / "data" / "motions" / "f1").glob("*.npz"))
        clips += sorted((pm / "data" / "testsets" / "f1_holdout").glob("*.npz"))
        out_dir = pm / "output_onnx"
        out_dir.mkdir(exist_ok=True)
        summary["export_ckpt"] = ckpt.name
        for c in clips:
            dst = out_dir / f"{args.tag}_{c.stem}.onnx"
            run(["uv", "run", "--no-sync", "python", "tools/export_onnx.py", "--robot", "f1",
                 "--ckpt", str(ckpt), "--motion", str(c), "--out", str(dst)], cwd=pm, check=False, tail=6)
        # ONNX files travel as .pt so the SDK uploads them (rename back to .onnx after download)
        for q in sorted(out_dir.glob("*.onnx")):
            pub.publish_as_pt(q, q.name)
    else:
        blog("no checkpoint at all: ONNX export skipped")

    # 7. publish the training log / boot summary next to the checkpoints
    for name in ("train_log.csv", "bootstrap_cmd.json"):
        if (log_dir / name).exists():
            pub.publish_as_pt(log_dir / name, name.replace(".", "_"))
    blog("DONE")


if __name__ == "__main__":
    main()
