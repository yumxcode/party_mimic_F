#!/usr/bin/env python3
"""gradmotion bootstrap: evaluate the periodic checkpoints of a finished F1 run in PartyMimic's own
simulator (partymimic.evaluate: one pass per clip from frame 0, no randomization) and publish, for the
checkpoint with the best normalized return (training-side selection, decided before any acceptance
replay), one ONNX per acceptance clip.

Same environment as scripts/f1_train.py. Checkpoints come from signed gradmotion OSS urls, passed
base64-encoded as a JSON {file_name: url} (the platform drops start scripts with % & = characters).

Usage:
    gm-run party_mimic_F/scripts/f1_ckpt_export.py --urls_b64 <base64 json> --tag f1_dance_v2_36k
"""
from __future__ import annotations

import argparse
import base64
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import f1_train as boot  # noqa: E402

WS, HERE = boot.WS, boot.HERE
CLIPS = [("data/motions/f1", "dance1_subject3_255_706"), ("data/motions/f1", "dance2_subject1_352_795"),
         ("data/motions/f1", "dance2_subject4_2835_2925"), ("data/testsets/f1_holdout", "dance2_subject3_300_450")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--urls_b64", required=True)
    ap.add_argument("--tag", default="f1_dance_v2_36k")
    ap.add_argument("--skip_sync", action="store_true")
    args = ap.parse_args()
    summary = {"tag": args.tag, "boot": boot.BOOT}
    try:
        _main(args, summary)
    finally:
        (WS / "ckpt_export_summary.json").write_text(json.dumps(summary, indent=1))
        print("\n[boot] ===== bootstrap summary =====", flush=True)
        for line in boot.BOOT:
            print(line, flush=True)


def _main(args, summary: dict) -> None:
    pm = WS / "PartyMimic"
    boot.clone_partyMimic(pm)
    boot.blog(f"PartyMimic HEAD: {boot.run(['git', 'rev-parse', 'HEAD'], cwd=pm, check=False).stdout.strip()}")
    for rel in ("src/partymimic/f1", "assets/f1", "data/motions/f1", "data/testsets/f1_holdout"):
        src, dst = HERE / rel, pm / rel
        if not src.exists():
            raise SystemExit(f"missing {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    boot.run([sys.executable, str(HERE / "tools/patch_partymimic.py"), str(pm)], cwd=WS)
    if not args.skip_sync:
        boot.blog(f"uv sync: {boot.uv_sync(pm)}")
    boot.disable_mjlab_cuda_graph(pm)
    boot.disable_mjwarp_conditional_graph(pm)

    # checkpoints
    urls = json.loads(base64.b64decode(args.urls_b64.strip()).decode())
    log_dir = pm / "logs" / args.tag
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpts = {}
    for name, url in sorted(urls.items()):
        dst = log_dir / name
        for k in range(1, 4):
            if dst.exists() and dst.stat().st_size > 1e6:
                break
            try:
                boot.blog(f"downloading {name} (attempt {k})")
                urllib.request.urlretrieve(url, dst)
            except Exception as e:  # noqa: BLE001
                boot.blog(f"download failed: {e!r}")
                dst.unlink(missing_ok=True)
                time.sleep(20)
        if not dst.exists():
            raise SystemExit(f"could not download {name}")
        ckpts[name] = dst
    # evaluate.py reads args.json next to the checkpoint (for --interface); take it from the first checkpoint
    first = next(iter(ckpts.values()))
    boot.run(["uv", "run", "--no-sync", "python", "-c",
              "import torch,sys,json; ck=torch.load(sys.argv[1],map_location='cpu'); json.dump(ck['args'], open(sys.argv[2],'w')); print('iteration', ck['iteration'])",
              str(first), str(log_dir / "args.json")], cwd=pm, tail=2)

    # native evaluation of every checkpoint (training-side selection criterion)
    results = {}
    for name, ck in ckpts.items():
        out = log_dir / f"eval_{Path(name).stem}.json"
        p = boot.run(["uv", "run", "--no-sync", "python", "-m", "partymimic.evaluate", "--robot", "f1", "--run_dir", str(log_dir),
                      "--ckpt", str(ck), "--motion_dir", "data/motions/f1", "--out", str(out), "--repeats", "1"], cwd=pm, tail=6, check=False)
        if out.exists():
            r = json.loads(out.read_text())
            results[name] = {k: r[k] for k in ("norm_return", "norm_return_framew", "n_failed", "fail_rate")}
            results[name]["progress_mean"] = float(sum(r["per_clip_progress"].values()) / max(len(r["per_clip_progress"]), 1))
            boot.blog(f"native eval {name}: {json.dumps(results[name])}")
    if not results:
        raise SystemExit("no checkpoint evaluated")
    best = max(results, key=lambda n: (results[n]["norm_return_framew"], -results[n]["n_failed"]))
    summary["native_eval"] = results
    summary["best"] = best
    boot.blog(f"best checkpoint by native frame-weighted normalized return: {best}")

    # export one ONNX per acceptance clip from the best checkpoint (and run the export's own torch check)
    boot.run(["uv", "pip", "install", "--python", str(pm / ".venv/bin/python"), "onnxruntime"], cwd=pm, tail=3)
    out_dir = pm / "output_onnx"
    out_dir.mkdir(exist_ok=True)
    exported = []
    for rel_dir, clip in CLIPS:
        dst = out_dir / f"{args.tag}_{Path(best).stem.replace('ckpt_', 'it')}_{clip}.onnx"
        boot.run(["uv", "run", "--no-sync", "python", "tools/export_onnx.py", "--robot", "f1", "--ckpt", str(ckpts[best]),
                  "--motion", str(pm / rel_dir / f"{clip}.npz"), "--out", str(dst)], cwd=pm, tail=16, check=False)
        if dst.exists():
            exported.append(dst)
    out = WS / "output"
    out.mkdir(exist_ok=True)
    for dst in exported:
        shutil.copy2(dst, out / (dst.name + ".pt"))
        boot.blog(f"published {dst.name}.pt ({dst.stat().st_size / 1e6:.1f} MB)")
    for name in ckpts:
        ej = log_dir / f"eval_{Path(name).stem}.json"
        if ej.exists():
            shutil.copy2(ej, out / f"{args.tag}_{ej.name}.pt")
    boot.blog("DONE")


if __name__ == "__main__":
    main()
