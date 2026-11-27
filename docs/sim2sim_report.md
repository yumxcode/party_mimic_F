# F1 29DOF PartyMimic 跳舞 — 训练与 MuJoCo sim2sim 验收报告（v1）

日期：2026-11-27 · 任务：A14 续跑 · 判定依据：`docs/acceptance_plan.md`（训练前预注册，阈值未改）

## 0. 结论（先看这里）

| 项 | 结果 |
|---|---|
| PartyMimic 训练（gradmotion，F1 29DOF，项目 kp/kd） | **跑通**：TASK_20261127_632，12000 it / 4096 envs，0.594 h（5.72 it/s），`sim_r` 0.86，训练期终止率 0 |
| 策略导出 + CPU MuJoCo sim2sim | **跑通**：14 个 ONNX（每段一个），`tools/f1_sim2sim.py` 重建训练物理与观测后回放 |
| 舞蹈验收门 A1–A5（4 段） | **未通过（3/4 段通过）**：`dance1_subject3_255_706` 的 **A4 右脚平贴率 86.4% < 90%**；其余 3 段（含 held-out）A1–A5 全过 |
| 对比视频 | `videos/f1_dance_v1_compare.mp4`（参考｜策略，18.7 MB，25 fps，37.8 s，4 段） |

A4 差距的性质：该段参考本身右脚平贴率只有 93.7%（脚掌 pitch p95 10.5°，门限 10°），策略回放的脚掌 pitch 比参考系统性高约 3°（p95 13.0°），把 57 个"参考 ≤10° 但回放 >10°"的帧推过门限。属策略在脚掌姿态上的跟踪精度（A3 整体关节误差 0.031 rad），不是跌倒或数据缺陷；详见 §4。

## 1. 训练

| 项 | 值 |
|---|---|
| 平台 | gradmotion，账号池 id=85，项目 PRO_20261127_084，镜像 BJX00000396 / V000452，1× RTX 4090 24 GB（驱动 535.183.01，CUDA 12.2） |
| 任务 | **TASK_20261127_632** `pm-f1-dance-v1`，18:28 启动，19:24 结束（状态 5） |
| 启动脚本 | `gm-run party_mimic_F/scripts/f1_train.py --smoke_then_full --iterations 12000 --num_envs 4096 --tag f1_dance_v1`（bootstrap 提交 7338c64） |
| 算法/代码 | PartyMimic `main@0c17a63`（uv.lock：torch 2.9.0+cu128、mjlab 1.6.0、mujoco-warp 3.11.0、warp 1.16.0）+ 本仓库 `src/partymimic/f1`、`assets/f1`、`data/motions/f1` |
| 配置 | PartyMimic 默认配方（MLP 编码器、joint 接口、`--pi_ref_skip`、`--wm_guard 5 --v_guard 5`），seed 0，12000 迭代，4096 环境，`--ckpt_minutes 20`；完整命令见 `eval/f1_dance_v1/bootstrap_cmd.json` |
| kp/kd | `src/partymimic/f1/f1_constants.py`：下肢取自项目《F1 下肢 kp_kd 选取.html》（hip 70/9.5、70/9.0、90/4.2，knee 90/6.9，ankle 60/4.6、45/2.4），**上肢取自 Beyondminic-Weilai-X1/F1_29DOF/f1_kp_kd.json 的 `upper_x1_planB`**（lumbar 90/8、70/8、100/8，shoulder 40/4、40/4、40/2，elbow 30/1.9、20/1.0，wrist 15/1.0、15/1.0）；armature 腿 0.0638/0.196/0.068、上肢 0.01 |
| 训练曲线 | `eval/f1_dance_v1/train_log.csv`：1000 it 后基本平台（sim_r 0.844→0.869@2k→0.86@12k，wm_err8 0.124→0.09→0.082），term_frac 0 |
| 产物 | `f1_dance_v1_ckpt_002000…012000.pt`、`f1_dance_v1_last.pt`（132 MB）、14 个 `…_<clip>.onnx.pt`（各 37 MB）、`train_log.csv`；`gm task model list --task-id TASK_20261127_632`。本仓库留存 4 段验收 ONNX 于 `eval/f1_dance_v1/`（不入 git） |

### 1.1 A14 失败根因与修复（为什么此前训练从未跑通）

A14 的三次 smoke（TASK_563/564/598）全部败于 `RuntimeError: Conditional graph nodes require CUDA driver 12.4+`（564 的"状态 5"是 `check=False` 掩盖的失败）。根因：gradmotion 4090D 主机的驱动为 535（CUDA 12.2），而 MuJoCo-Warp 3.11 的求解器把 Newton 迭代循环实现为条件图节点（`wp.capture_while`），mjlab 自带的图捕获和 PartyMimic `train.py` 自己的采集图捕获都会撞上它。修复（`scripts/f1_train.py`）：

1. mjlab `sim.py` 的 `_GRAPH_CAPTURE_MIN_DRIVER = (12, 4)` → `(99, 0)`：mjlab 自身不再捕获，`step()` 直接调 `mjwarp.step`（A14 已做，保留）。
2. mujoco_warp `io.py` 的 `opt.graph_conditional = True` → `False`：这是 mujoco_warp 自己的驱动检查给出的建议（"conditional graph nodes are not available for < 12.4 — Model.opt.graph_conditional should be set to False"），求解器改为固定 100 次迭代的普通循环（已收敛世界早退，物理等价），PartyMimic 的采集图就能把它当普通节点捕获。
3. 兜底：`PM_F1_EAGER_COLLECT=1` 让采集步保持 eager（`tools/patch_partymimic.py` 的开关），smoke 若再败于条件图节点则自动回退。本次 **未触发**：日志 `cuda_graph: captured one collection step in 0.2 s`，`eager_collect=False`。

其它 bootstrap 改动：checkpoint 看护线程每 60 s 把新 `.pt` 发布到 `/workspace/output`；训练日志、ONNX 以 `.pt` 后缀随产物上传；补上 held-out 片段的拷贝（A14 的导出循环因目录未拷贝而为空）；`[boot]` 摘要在结束前重打（gradmotion 日志只留最后 200 KB）。

## 2. MuJoCo sim2sim（`tools/f1_sim2sim.py`）

A14 留下的脚本未跑过，经逐项对拍 PartyMimic / mjlab 源码后发现并修正了以下与训练不一致之处（修正前的回放结果保留在 `eval/f1_dance_v1/acceptance_toolbug_6d/`，策略在 0.3–0.6 s 内倾倒，是工具错误而非策略表现）：

| 项 | A14 版本 | 现在（对拍依据） |
|---|---|---|
| 地面 | 无 | mjlab `TerrainEntityCfg(plane)` 的平面几何（默认接触参数） |
| 脚底球碰撞组 | 原始 2/4、4/2（脚与脚相撞、与地不撞） | 实体 `get_spec()`：contype=0 conaffinity=1 |
| 执行器 | `forcerange ±1e6`，armature=0 | 按 mjlab `BuiltinPositionActuatorCfg` 编译结果：gainprm=kp，biasprm=(0,−kp,−kd)，不限 ctrl，forcerange=±effort_limit，armature=0.0638/0.196/0.068/0.01；kp/kd/默认位姿/action_scale/tracked bodies 与 ONNX 元数据逐项断言一致 |
| 物理参数 | 部分 | PartyMimic `make_sim` → mjlab `MujocoCfg`：dt 0.005，implicitfast，pyramidal，Newton 100/50，tol 1e-6，ls_tol 0.01，ccd 35 |
| 观测 | 对 body 系 qvel 左乘 Rᵀ；投影重力取列 | `MDP.obs`/`env.state()`：根角速度 = qvel[3:6]（根系），投影重力 = −R[2,:] = Rᵀ(0,0,−1)，6D 相对姿态 = `RᵀR_ref` 前两列按行展开（`R[:, :2].reshape(6)`） |
| 帧对齐 | 回放后一帧 vs 参考当前帧 | 与 `evaluate.py` 一致：帧 f 的机器人状态在步进前记录，对比参考 f |
| 动作时序 | 用上一帧历史算动作 | 先追加帧 f 的观测再算帧 f 的动作；`action_at` 内断言历史末项等于当前观测（stale history 直接报错） |

自检：`--oracle_npz`（纯 PD 跟参考）与 `--kinematic`（逐帧置为参考）两种模式；kinematic 整段回放 A2=4e-8 m、A3=0，A4/A5 与参考门数值一致（dance1_subject3：97.2/93.7，0%）。注意 F1 的踝 kp 60/45 远小于 ~216 N·m/rad 的倾覆刚度，纯 PD 连默认站姿都保持不住（`f1_kp_kd.json` 的 `pd_free_stand` 记载），所以 oracle 在 1–2 s 内倾倒属预期，不是上界。

A4/A5 的计算：回放轨迹写成 PartyMimic 格式的 `<clip>_motion.npz`（50 Hz），用九门工具 `tools/check_motion_f1.py` 的 gate 7/9 公式重算；支撑期 = 触地连续 ≥ **8 帧@50 Hz**（等价于九门的 5 帧@30 fps = 0.167 s；在看到策略结果前决定），并记录支撑帧数（<10 帧判为无效）。A1 = 全程 base z > 0.25 m 且倾角 < 0.8 rad。

## 3. 验收结果（A1–A5，逐门实测值）

回放：4 段均从第 0 帧播到最后一帧，无跌倒。

| 片段 | 类型 | 帧数 | A1 跌倒 | A2 局部 body 误差 mm（限） | A3 关节误差 rad（限 0.20） | A4 平贴率 % L/R（限 90） | A5 内收 %（限 3） | 判定 |
|---|---|---|---|---|---|---|---|---|
| dance1_subject3_255_706 | 训练 | 750/750 | 否（min z 0.530，max 倾角 0.55 rad） | **8.2**（50） | **0.0310** | 94.8 / **86.4 ✗** | 0.00 | ✗（A4 右脚） |
| dance2_subject1_352_795 | 训练 | 737/737 | 否（0.559，0.47） | **9.4**（50） | **0.0354** | 93.0 / 94.3 | 0.00 | ✓ |
| dance2_subject4_2835_2925 | 训练 | 149/149 | 否（0.566，0.26） | **7.4**（50） | **0.0320** | 100 / 100 | 0.00 | ✓ |
| dance2_subject3_300_450 | held-out | 249/249 | 否（0.578，0.70） | **44.5**（80） | **0.0651** | 97.6 / 100 | 0.00 | ✓ |

支撑帧数（gate 7 样本）：750/750、732/732、149/149、249/249，全部有效。A4 用检查器默认 5 帧的敏感性数值与 8 帧完全相同。参考动作本身的 A4/A5：97.2/93.7（pitch p95 8.8/10.5°）、90.2/93.3、100/100、90.5/100，内收 ≤0.95%。

统计口径：A2/A3 按计划表取全程均值（计划未写统计量），补充分位数以免均值掩盖尖峰（`acceptance/error_percentiles.json`）：A3 的 p95（逐关节逐帧）0.102 / 0.115 / 0.097 / 0.189 rad，最大 0.84 rad（dance1_subject3 某一关节的瞬时尖峰）；A2 的 p95 16.8 / 14.6 / 15.3 / 65.9 mm，最大 43.3 / 24.1 / 22.9 / 72.4 mm。

原始数据：`eval/f1_dance_v1/acceptance/acceptance.json|md`，每段 `*.json`（指标）、`*.npz`（回放）、`*_motion.npz`、`*_gates_smf{5,8}.json`。

## 4. A4 差距分析（dance1_subject3_255_706 右脚）

- 参考右脚 750 帧中 47 帧平贴率不达标（pitch > 10°）；回放 102 帧，其中 **57 帧是"参考 ≤10° 而回放 >10°"**，集中在 f=315–340、362–366 等参考 pitch 已达 9.3–9.9° 的踮脚段，回放比参考高约 3°（12.5–13.0°）。
- 对应右踝 pitch 关节：回放 − 参考 均值 +0.020 rad（1.1°），最大 0.109 rad；整体 A3 0.031 rad。即策略把脚掌姿态跟踪到了约 1–3° 的精度，但参考在该段正处在 10° 门限边缘，这一点偏差足以把平贴率从 93.7% 拉到 86.4%。
- 训练曲线在 1000–2000 迭代后已平台（§1），单纯加迭代的预期收益有限；这是该配方 + 项目 kp/kd 下脚掌姿态的跟踪精度，不是数据不足导致的泛化失败（held-out 段 A2 44.5 mm 通过）。
- 未做的事：没有改动阈值、没有更换或裁剪验收片段、没有为了过门修改参考动作。

## 5. 披露

- 训练库 13 段 / 76.3 s（A14 文档写 76.4 s，为逐段取整差异），远小于 PartyMimic 论文通用配方（41 h / 10,745 段）；本项目目标是复现这批舞蹈而非零样本泛化。
- 13 段中仅 3 段九门全过（Tier A，均为验收段）；其余 10 段（Tier B）脚底平贴率未达 90%、只用于训练、不参与验收。核实 manifest 时发现 Tier B 中有两段右脚低于文档所述的 80% 下限：`dance2_subject3_2325_2475` 79.5%、`dance2_subject3_5400_5550` 76.7%（A14 文档写"flat 80–95%"）；训练按原设计使用了全部 13 段，未因此改动。
- 上肢 kp/kd 来源见 §1（项目页面只有下肢）；MJCF 惯性合计 32.22 kg 与页面页首的 35.03 kg 不一致：35.03 是页首标称整机质量，页面计算与 URDF/MJCF 均用 CAD 导出的 32.22 kg，仿真沿用 URDF 值，未改动。
- gradmotion 镜像内没有 onnxruntime，上游 `export_onnx.py` 自带的 ONNX-vs-torch 数值校验被跳过；为核实导出与跨栈一致性，另提交了 PartyMimic 原生仿真器（mjlab/MuJoCo-Warp）的回放任务 TASK_20261127_676（记录逐步 obs/goal/action 并做 ONNX-vs-torch 校验），结果见 §6。
- 账号：A14 的账号（id=79，项目 PRO_20261127_007）余额 20 且被其它任务占用，本次改用 id=85（余额 50，训练后 44.5）；训练前已查余额。

## 6. 原生栈回放（PartyMimic 自己的仿真器，跨栈对照；验收裁决仍以 §3 的 CPU MuJoCo 回放为准）

目的：把 A4 的 86.4% 归因到"工具/物理栈差异"还是"策略精度"。`scripts/f1_native_eval.py` + `tools/f1_native_replay.py` 在 mjlab/MuJoCo-Warp 里按 `evaluate.py` 的方式从第 0 帧回放 4 段，逐步记录 obs 历史、goal、动作、状态，并在真实观测上做 ONNX-vs-torch 校验（训练镜像没有 onnxruntime，上游导出脚本自带的这项校验被跳过了）。

判读规则（读结果前写下）：
1. ONNX-vs-torch 最大绝对差 ≪ 动作尺度（例如 < 0.05）→ 导出无误；否则只修导出，不重训。
2. 原生第 0 帧的观测历史与 `tools/f1_sim2sim.py` 在同一 reset 状态算出的观测逐元素对比（容差 1e-5）→ 数值验证 obs 布局（含 6D 相对姿态的元素顺序）。
3. 原生回放的 q/根位姿写成 `_motion.npz` 跑 `check_motion_f1.py`：若 mjlab 里右脚平贴率 ≥ 90% 而 CPU MuJoCo 为 86.4% → 两套物理栈的差异，需修 sim2sim 保真度；若两边都 ≈ 86% → 策略/数据侧，再比较右踝的目标、参考与实际，区分"策略给错目标"与"PD/接触受限"。

过程记录：首次提交 TASK_20261127_676 因平台把含 `% & =` 的启动脚本清空（容器日志 `START_SCRIPT:` 为空）而空跑 3 分钟；改为 base64 传 URL 后重提 TASK_20261127_685。

（结果待填。）

## 7. 复现

```bash
# 训练（gradmotion）
gm-run party_mimic_F/scripts/f1_train.py --smoke_then_full --iterations 12000 --num_envs 4096 --tag f1_dance_v1
# 下载（work/gm.sh / work/gm_fetch_models.sh 为本地辅助脚本）后：
python tools/f1_acceptance.py --onnx_dir eval/f1_dance_v1 --tag f1_dance_v1 --out eval/f1_dance_v1/acceptance
python tools/f1_render_compare.py --rollout eval/f1_dance_v1/acceptance/*.npz --out videos/f1_dance_v1_compare.mp4 --stride 2 --quality 7
```
