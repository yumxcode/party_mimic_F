# party_mimic_F — PartyMimic on the F1 29DOF humanoid (dance motion imitation + MuJoCo sim2sim)

PartyMimic (Roboparty, MIT) adapted to the NIO F1 29DOF humanoid, trained on a retargeted LAFAN1 dance
library on the gradmotion platform, and replayed in plain MuJoCo.

```
src/partymimic/f1/f1_constants.py   RobotSpec for F1: MJCF joint order, tracked bodies, PD gains, action scale
assets/f1/f1_party.xml               physics-only training MJCF (8 foot contact spheres, no meshes)
assets/f1/f1_party_vis.xml           same + vendor visual meshes (rendering; needs F1_29DOF/mjcf/meshes)
data/motions/f1/*.npz                49 PartyMimic motion clips (232 s, v6 recipe, all nine gates pass) + manifest.json
data/testsets/f1_holdout/            held-out clip dance2_subject3_300_450 (never trained on)
data/motions/f1_v5recipe/, f1_v6_14clips/   archived earlier libraries (v5 GMR only; v6 14-clip set)
tools/f1_flatten_feet.py             v6 recipe: stance-sole flatness compression on a v5 CSV (ankle-only)
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
python tools/f1_flatten_feet.py --csv work/retarget_csv/dance2_subject1_f1.csv --out work/retarget_csv_flat/dance2_subject1_f1.csv  # v6
python tools/f1_gate_windows.py --csv work/retarget_csv_flat/dance2_subject1_f1.csv \
    --out work/gate_flat/dance2_subject1_w150.json --win 150 --stride 75
python tools/f1_build_library.py --plan work/library_plan_v6b.json --out_dir data/motions/f1   # plan: docs/library_plans/library_plan_v6b.json
```

### 2. Train (gradmotion, image BJX00000396 / V000452, 1x RTX 4090D 24 GB)
Repo `https://github.com/yumxcode/party_mimic_F.git`, start script
```
gm-run party_mimic_F/scripts/f1_train.py --smoke_then_full --iterations 12000 --num_envs 4096 --seed 0 --tag f1_dance_v6b_s0
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
python tools/f1_acceptance.py --onnx_dir eval/f1_dance_v6b_s0 --tag f1_dance_v6b_s0 --out eval/f1_dance_v6b_s0/acceptance
python tools/f1_sim2sim.py --onnx eval/f1_dance_v6b_s0/f1_dance_v6b_s0_dance1_subject3_255_706.onnx --out eval/x.npz --json eval/x.json
python tools/f1_render_compare.py --rollout eval/f1_dance_v6b_s0/acceptance/dance*.npz --out videos/f1_dance_v6b_compare.mp4 --stride 2 --quality 7
```
`tools/f1_sim2sim.py` rebuilds the training physics (mjlab actuators with the F1 armature/effort limits,
ground plane, MujocoCfg) and the observation of `partymimic.mdp.MDP.obs`; a replay in PartyMimic's own
simulator (`scripts/f1_native_eval.py`, `tools/f1_native_compare.py`) agrees with it to <0.003 rad.

## Status (2026-11-28)

* **Final: the pre-registered dance acceptance (docs/acceptance_plan.md, A1-A5 on CPU MuJoCo sim2sim) PASSES
  4/4 clips for `f1_dance_v6b_s0`** (gradmotion TASK_20261128_111: 12k it / 4096 envs, 49-clip / 232 s
  v6 library, seed 0; replicated from scratch with seed 2, TASK_20261128_134, also 4/4; the seed-0
  checkpoint gives the same gates in PartyMimic's own simulator, mjlab, joint diff < 0.002 rad). No fall in any clip (A1); local body error 9.0 / 8.9 / 8.3 mm trained, 37.2 mm
  held-out (limits 50 / 80); joint error 0.033-0.053 rad (limit 0.20); foot flatness 97.7/97.3,
  98.3/98.9, 100/100, held-out 100/100 % (limit 90); hip-roll adduction <= 0.54 % (limit 3).
  Thresholds were never changed. Video: `videos/f1_dance_v6b_compare.mp4`. Report: `docs/sim2sim_report.md` §11.
* Read with it: (1) the references of the 4 acceptance clips are the **v6 re-retargeted** motions
  (stance soles flattened; user-authorized option b after the v5-reference result below); under the
  original v5 references the same gates still give 3/4. (2) The identical configuration passes on
  **2 of 3 seeds**: seed 1 fails (falls at frame 145/149 of the 3 s clip, A1) although its other 3
  clips pass; the delivered policy is seed 0. (3) With the v6 references but only the 14-clip /
  85 s library the held-out clip failed A4 (71.9/81.0 %); enlarging the library to all 49 nine-gate
  windows closed it.
* v5-reference history (all kept in the report): `f1_dance_v1` (12k it) and `f1_dance_v2_36k` (diverged
  after ~33k; its 26k checkpoint equals v1) pass 3/4 clips; `dance1_subject3_255_706` fails A4 (right-foot
  flatness 86.4 % < 90 %, reference itself 93.7 %). Single-variable follow-ups, all 3/4: seed 1 -> 86.4 %,
  `--contact_reward 0.25` -> 84.7 %, training on the 8 clips with both feet >= 85 % flat -> 87.1 %. Native
  mjlab replay agrees with the CPU MuJoCo replay (86.1 %, <0.003 rad).
* Retargeting gates (nine gates, thresholds unchanged): all 49 library clips + the held-out clip pass
  (`docs/retarget_gate_table.md`); v5 tables in `docs/retarget_gate_table_v5.md`.
* Details, per-gate numbers, the training-task ids and all disclosures: `docs/sim2sim_report.md`.
  Resumption notes and task ids: `PROGRESS.md`.
