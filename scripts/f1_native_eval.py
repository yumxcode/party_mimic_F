#!/usr/bin/env python3
"""gradmotion bootstrap: replay the trained F1 policy in PartyMimic's OWN simulator (mjlab / MuJoCo-Warp).

Same environment as scripts/f1_train.py (clone upstream PartyMimic, copy the F1 additions, patch, uv
sync, disable the two CUDA-graph captures), then
  1. download the checkpoint (a signed gradmotion OSS url from `gm task model list`),
  2. re-export one ONNX per acceptance clip (tools/export_onnx.py) and install onnxruntime so the
     export's own ONNX-vs-torch check runs (the training image has no onnxruntime: it was skipped),
  3. tools/f1_native_replay.py: play each clip from frame 0 exactly as partymimic.evaluate does and
     record observation history / goal / action / state per step, plus the ONNX action on the same
     observations,
  4. publish the records to /workspace/output as .pt (the SDK uploads every file there).

Usage:
    gm-run party_mimic_F/scripts/f1_native_eval.py --ckpt_url_b64 <base64 of the oss url> --tag f1_dance_v1
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import f1_train as boot  # noqa: E402  (find_repo, run, blog, the two cuda-graph patches)

WS = boot.WS
HERE = boot.HERE
CLIPS = ["dance1_subject3_255_706", "dance2_subject1_352_795", "dance2_subject4_2835_2925"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt_url_b64", required=True,
                    help="base64 of the signed gradmotion OSS url (the platform's start-script handling drops "
                         "strings with % & = characters: TASK_20261127_676 ran with an empty START_SCRIPT)")
    ap.add_argument("--tag", default="f1_dance_v1")
    ap.add_argument("--clips", default=",".join(CLIPS), help="training clips (data/motions/f1/<clip>.npz)")
    ap.add_argument("--holdout", default="dance2_subject3_300_450")
    ap.add_argument("--skip_sync", action="store_true")
    args = ap.parse_args()
    summary = {"tag": args.tag, "boot": boot.BOOT}
    try:
        _main(args, summary)
    finally:
        (WS / "native_eval_summary.json").write_text(json.dumps(summary, indent=1))
        print("\n[boot] ===== bootstrap summary =====", flush=True)
        for line in boot.BOOT:
            print(line, flush=True)


def _main(args, summary: dict) -> None:
    pm = WS / "PartyMimic"
    boot.clone_partyMimic(pm)
    head = boot.run(["git", "rev-parse", "HEAD"], cwd=pm, check=False).stdout.strip()
    boot.blog(f"PartyMimic HEAD: {head}")
    summary["pm_head"] = head
    for rel in ("src/partymimic/f1", "assets/f1", "data/motions/f1", "data/testsets/f1_holdout"):
        src, dst = HERE / rel, pm / rel
        if not src.exists():
            raise SystemExit(f"missing {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        boot.blog(f"copied {rel}")
    boot.run([sys.executable, str(HERE / "tools/patch_partymimic.py"), str(pm)], cwd=WS)
    boot.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"], cwd=WS, check=False, tail=5)
    if not args.skip_sync:
        boot.blog(f"uv sync: {boot.uv_sync(pm)}")
    boot.disable_mjlab_cuda_graph(pm)
    boot.disable_mjwarp_conditional_graph(pm)
    boot.run(["uv", "pip", "install", "--python", str(pm / ".venv/bin/python"), "onnxruntime"], cwd=pm, tail=5)

    # 1. checkpoint
    log_dir = pm / "logs" / args.tag
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt = log_dir / "last.pt"
    if not ckpt.exists():
        import base64
        url = base64.b64decode(args.ckpt_url_b64.strip()).decode()
        for k in range(1, 4):
            try:
                boot.blog(f"downloading checkpoint ({url.split('?')[0].rsplit('/', 1)[-1]}), attempt {k}")
                urllib.request.urlretrieve(url, ckpt)
                break
            except Exception as e:  # noqa: BLE001
                boot.blog(f"download failed: {e!r}")
                ckpt.unlink(missing_ok=True)
                time.sleep(20)
        boot.blog(f"checkpoint: {ckpt.stat().st_size / 1e6:.1f} MB")
    # evaluate.py reads args.json next to the checkpoint for --interface; the checkpoint carries args too
    ck_meta = boot.run(["uv", "run", "--no-sync", "python", "-c",
                        "import torch,sys; ck=torch.load(sys.argv[1],map_location='cpu'); print(ck['iteration']); "
                        "import json; json.dump(ck['args'], open(sys.argv[2],'w'))", str(ckpt), str(log_dir / "args.json")],
                       cwd=pm, tail=3)
    summary["ckpt_iteration"] = (ck_meta.stdout or "").strip().splitlines()[-1] if ck_meta.stdout else None

    # 2. one ONNX per clip (+ the export's own ONNX-vs-torch check now that onnxruntime is there)
    clips = [c for c in args.clips.split(",") if c]
    all_clips = [("data/motions/f1", c) for c in clips] + [("data/testsets/f1_holdout", args.holdout)] if args.holdout else []
    out_dir = pm / "output_onnx"
    out_dir.mkdir(exist_ok=True)
    onnx = {}
    for rel_dir, clip in all_clips:
        src = pm / rel_dir / f"{clip}.npz"
        dst = out_dir / f"{args.tag}_{clip}.onnx"
        boot.run(["uv", "run", "--no-sync", "python", "tools/export_onnx.py", "--robot", "f1", "--ckpt", str(ckpt),
                  "--motion", str(src), "--out", str(dst)], cwd=pm, tail=14)
        onnx[clip] = dst

    # 3. native replay per clip
    shutil.copy2(HERE / "tools/f1_native_replay.py", pm / "tools/f1_native_replay.py")
    out = WS / "output"
    out.mkdir(exist_ok=True)
    results = {}
    for rel_dir, clip in all_clips:
        rec = pm / "logs" / f"native_{clip}.npz"
        p = boot.run(["uv", "run", "--no-sync", "python", "tools/f1_native_replay.py", "--robot", "f1",
                      "--motion", str(pm / rel_dir / f"{clip}.npz"), "--ckpt", str(ckpt), "--out", str(rec),
                      "--onnx", str(onnx[clip])], cwd=pm, tail=8, check=False)
        for line in (p.stdout or "").splitlines():
            if line.startswith("{"):
                try:
                    results[clip] = {**results.get(clip, {}), **json.loads(line)}
                except json.JSONDecodeError:
                    pass
        if rec.exists():
            shutil.copy2(rec, out / f"{args.tag}_native_{clip}.npz.pt")
            boot.blog(f"published {args.tag}_native_{clip}.npz.pt ({rec.stat().st_size / 1e6:.1f} MB)")
    summary["native"] = results
    boot.blog("native results: " + json.dumps(results))
    boot.blog("DONE")


if __name__ == "__main__":
    main()
