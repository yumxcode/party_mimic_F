# party_mimic_F — PartyMimic on the F1 29DOF humanoid (dance motion imitation + MuJoCo sim2sim)

PartyMimic (Roboparty, MIT) adapted to the NIO F1 29DOF humanoid, trained on a retargeted LAFAN1 dance
library on the gradmotion platform, and replayed in plain MuJoCo.

```
src/partymimic/f1/f1_constants.py   RobotSpec for F1: MJCF joint order, tracked bodies, PD gains, action scale
assets/f1/f1_party.xml               physics-only training MJCF (8 foot contact spheres, no meshes)
assets/f1/f1_party_vis.xml           same + vendor visual meshes (rendering; needs F1_29DOF/mjcf/meshes)
data/motions/f1/*.npz                13 PartyMimic motion clips (76.4 s) + manifest.json (tiers, gates, lifts)
data/testsets/f1_holdout/            held-out Tier A clip (never trained on)
scripts/f1_train.py                  gradmotion bootstrap: clone PartyMimic, add F1, uv sync, train, export ONNX
tools/build_f1_party_mjcf.py         derive the two MJCFs from the vendor f1_serial.xml
tools/check_motion_f1.py             the nine F1 retargeting gates (from Beyondminic-Weilai-X1)
tools/f1_gate_windows.py             per-frame metrics + nine-gate window selection
tools/f1_build_library.py            CSV window -> lift -> resample 50 Hz -> PartyMimic npz
tools/f1_csv_to_pm_npz.py            single CSV -> PartyMimic npz (with an FK sole check)
tools/f1_sim2sim.py                  MuJoCo (CPU) sim2sim of an exported ONNX policy
tools/f1_render_compare.py           reference | policy comparison video
docs/                                data requirements, gates, acceptance plan, reports
```

## Reproduce

### 1. Retarget (LAFAN1 BVH -> F1 CSV)
Needs the Beyondminic-Weilai-X1 checkout (GMR + its F1 IK config):
```bash
python tools_bvh_to_csv_f1.py --bvh_file downloads/lafan1/dance2_subject1.bvh \
    --output_file work/retarget_csv/dance2_subject1_f1.csv \
    --arm_relock c3.0_r0.92 --clamp_margin 0.06 --stance_widen 0.075
python tools/f1_gate_windows.py --csv work/retarget_csv/dance2_subject1_f1.csv \
    --out work/gate/dance2_subject1_w150.json --win 150 --stride 75
python tools/f1_build_library.py --plan work/library_plan_final.json --out_dir data/motions/f1
```

### 2. Train (gradmotion, image BJX00000396 / V000452, 1x RTX 4090D 24 GB)
Repo `https://github.com/yumxcode/party_mimic_F.git`, start script
```
gm-run party_mimic_F/scripts/f1_train.py --smoke_then_full --iterations 12000 --num_envs 4096 --tag f1_dance_v1
```
The bootstrap clones upstream PartyMimic at `main`, copies in `src/partymimic/f1`, `assets/f1`,
`data/motions/f1`, applies two backward-compatible patches (feet bodies, CUDA-graph off), runs
`uv sync --frozen`, trains, exports one ONNX per clip and publishes checkpoints to `/workspace/output`.

### 3. MuJoCo sim2sim
```bash
python tools/f1_sim2sim.py --onnx eval/f1_dance_v1_dance1_subject3_255_706.onnx \
    --mjcf assets/f1/f1_party.xml --out eval/rollout.npz --json eval/rollout.json
python tools/f1_render_compare.py --rollout eval/rollout.npz --out videos/dance_compare.mp4
```

## Status

See `docs/` for the data requirements, the nine retargeting gates with per-clip results, the
pre-registered dance acceptance plan, and the final report.
