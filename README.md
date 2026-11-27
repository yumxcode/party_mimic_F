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
`data/motions/f1`, `data/testsets/f1_holdout`, applies the backward-compatible patches of
`tools/patch_partymimic.py` (feet bodies, eager-collect switch), runs `uv sync --frozen`, disables the two
CUDA-graph captures that need conditional graph nodes (the 4090D hosts run a CUDA driver < 12.4: mjlab's
own capture and MuJoCo-Warp's conditional solver loop, the latter via `opt.graph_conditional=False` as
mujoco_warp's driver check recommends), runs a 2-iteration smoke (falling back to an eager collection step
if the collection-graph capture still fails), trains, exports one ONNX per clip and publishes checkpoints
(every 2000 iterations / 20 min), ONNX files and `train_log.csv` to `/workspace/output` as `.pt`.
Progress and the gradmotion task ids are kept in `PROGRESS.md`.

### 3. MuJoCo sim2sim and the pre-registered acceptance (A1-A5)
```bash
# downloaded ONNX files renamed to <tag>_<clip>.onnx in one directory (see PROGRESS.md)
python tools/f1_acceptance.py --onnx_dir eval/f1_dance_v1 --tag f1_dance_v1 --out eval/f1_dance_v1/acceptance
python tools/f1_sim2sim.py --onnx eval/f1_dance_v1/f1_dance_v1_dance1_subject3_255_706.onnx --out eval/x.npz --json eval/x.json
python tools/f1_render_compare.py --rollout eval/f1_dance_v1/acceptance/*.npz --out videos/f1_dance_v1_compare.mp4 --stride 2 --quality 7
```
`tools/f1_sim2sim.py` rebuilds the training physics (mjlab actuators with the F1 armature/effort limits,
ground plane, MujocoCfg) and the observation of `partymimic.mdp.MDP.obs`; a replay in PartyMimic's own
simulator (`scripts/f1_native_eval.py`, `tools/f1_native_compare.py`) agrees with it to <0.003 rad.

## Status (2026-11-28)

* Training runs on gradmotion (F1 29DOF, project kp/kd, 13-clip library): `f1_dance_v1` (12k it) and
  `f1_dance_v2_36k` (36k it; diverged after ~33k, its 26k checkpoint equals v1). PartyMimic's own
  evaluation: normalized return 0.916 on the 13 clips, no failures.
* **Dance acceptance (docs/acceptance_plan.md, A1-A5 on CPU MuJoCo sim2sim): 3 of 4 clips pass.**
  `dance1_subject3_255_706` fails A4 (right-foot flatness 86.4 % < 90 %; the reference itself is at
  93.7 %). All clips stay up (A1), local body error 7.4-9.4 mm trained / 44.5 mm held-out (A2), joint
  error 0.031-0.065 rad (A3), no hip-roll adduction (A5). Thresholds were not changed; the gap is the
  policy's foot-pitch tracking precision, not the sim2sim tool (native replay gives 86.1 %).
* Details, per-gate numbers, the training-task ids and all disclosures: `docs/sim2sim_report.md`.
  Comparison video: `videos/f1_dance_v1_compare.mp4`. Resumption notes: `PROGRESS.md`.
