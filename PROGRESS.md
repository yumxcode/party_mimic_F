# PROGRESS — F1 29DOF PartyMimic 跳舞（A14 续跑）

> 可随时中断续接的进度文件。时间均为 CST。**2026-11-28 07:40 续：用户选择 (b) 改重定向配方并重新预注册。v6 配方（脚掌平贴压缩）已落地：14 段 Tier A 训练库 + held-out，九门全过，验收参考平贴率 98–100%；两个重训任务已提交（TASK_20261128_101 seed 0 / TASK_20261128_102 seed 1，账号 85），预计 ~09:00 出结果；完成后下载 4 段 ONNX 跑 A1–A5。见下方"v6 重新预注册与重训"。**

## 最终状态

| 阶段 | 状态 | 说明 |
|---|---|---|
| 数据/重定向/九门/预注册验收门 | 完成（A14） | `docs/retarget_gates.md`、`docs/acceptance_plan.md`、`data/motions/f1/`（13 段 76.3 s）+ `data/testsets/f1_holdout/`（1 段） |
| 接手核对 + bootstrap 修复 | 完成 | A14 三次 smoke 均败于驱动 535 的条件图节点；mujoco_warp `graph_conditional=False` + mjlab 图捕获关闭 + 克隆重试 + uv sync 镜像回退 |
| 训练 v1（12k it） | 完成 | TASK_20261127_632（账号 85）：0.594 h，5.72 it/s；`eval/f1_dance_v1/`（4 个验收 ONNX 不入 git、train_log.csv、bootstrap_cmd.json） |
| sim2sim 工具 | 完成 | `tools/f1_sim2sim.py` 重建训练物理/观测并对拍 mjlab（原生回放逐帧关节差 <0.003 rad）；A4/A5 由 `tools/check_motion_f1.py` 在回放轨迹上重算；`tools/f1_acceptance.py` 出 A1–A5 表 |
| **验收 A1–A5（v1）** | **未通过：3/4 段通过** | `dance1_subject3_255_706` A4 右脚 86.4% < 90%（参考自身 93.7%）；其余 3 段全过（A2 7.4–9.4 mm，held-out 44.5 mm，A3 ≤0.065，A5 0%）。`eval/f1_dance_v1/acceptance/`，报告 `docs/sim2sim_report.md` |
| 原生栈回放（归因） | 完成 | TASK_20261127_801：mjlab 里同一策略 A4 右脚 86.1%、A2 8.2 mm，ONNX-vs-torch 最大 0.03–0.06（bf16），reset 观测一致到 1e-7 → 差距属策略跟踪精度。`eval/native/` |
| 训练 v2（36k it，单变量） | 完成，假设证否 | TASK_20261127_800（账号 83）：1.72 h；33k–36k 发散，终态 A4 81.3%；发散前 26k–32k 原生归一化回报 0.916（持平），训练侧选 26k：A4 85.6%、3/4 段通过。`eval/f1_dance_v2_36k/`（含 `ckpts/acceptance/`） |
| 对比视频 | 完成 | `videos/f1_dance_v1_compare.mp4`（参考｜策略，18.7 MB，25 fps，37.8 s，4 段） |
| 报告/文档 | 完成 | `docs/sim2sim_report.md`（§0 结论、§1 训练、§2 工具、§3 结果、§4 差距分析、§5 披露、§6 原生回放、§8–9 v2）；README 状态已更新 |

## v6 重新预注册与重训（进行中，2026-11-28 07:40；lab_notebook exp_af737447）

| 步骤 | 状态 | 说明 |
|---|---|---|
| 重定向配方 v6 | 完成 | `tools/f1_flatten_feet.py`：支撑期 sole pitch >4°、roll >3° 的部分 ×0.2，只改踝关节；8 个源全部重做（`work/retarget_csv_flat/`），九门扫窗通过窗口 6→49 |
| 训练库 | 完成 | `data/motions/f1/`：14 段 / 85.3 s 全 Tier A（3 个验收训练段 + 11 个新窗口）；`data/testsets/f1_holdout/` 同一 held-out 窗口；v5 库归档于 `*_v5recipe/`；过门表 `docs/retarget_gate_table.md` |
| 重新预注册 | 完成 | `docs/acceptance_plan.md` 修订 v2：同 4 个窗口、同阈值、同脚本；v1 结果保留在 §5 |
| 重训 | **进行中** | TASK_20261128_101（seed 0，`f1_dance_v3flat_s0`）、TASK_20261128_102（seed 1，`f1_dance_v3flat_s1`），账号 85，12k it/4096 envs |
| 重判（14 段库） | 完成：3/4 | 两个 seed 的 3 个训练段全过（A4 98–100%，原未过片段已过）；**held-out 段 A4 未过**（seed 0 71.9/81.0，seed 1 95.6/88.8）：未见动作上脚掌 pitch/roll 偏 10–14°，属 85 s 库的泛化差距。`eval/f1_dance_v3flat_s0|s1/` |
| 扩库（契约第 4 条） | 完成 | v6 扫窗的全部 49 个九门窗口入库（232 s，允许重叠）：`data/motions/f1/`（14 段版归档 `f1_v6_14clips/`）；验收参考不变（npz 逐字节相同）；lab_notebook exp_c0d36761 |
| 重训 v6b | **进行中** | TASK_20261128_111（账号 85，seed 0，`f1_dance_v6b_s0`）、TASK_20261128_112（账号 82，seed 1，`f1_dance_v6b_s1`），08:50 提交；完成后同上流程重判 |

## A4 攻关实验（v5 参考；**已完成**，2026-11-28 05:29–06:28 运行；lab_notebook exp_f5125d37 / exp_a557f372 / exp_91338346，均 falsified）

| 实验 | 任务号 | 账号 | 单变量 | 产物 | 完成后 |
|---|---|---|---|---|---|
| seed 1 | TASK_20261128_062 | 85 | seed 0→1 | **A4 右脚 86.4%（= seed 0）**，`eval/f1_dance_s1/` | `./work/gm_fetch_models.sh 85 TASK_20261128_062 eval/f1_dance_s1 <clip>.onnx` ×4 + train_log；改名 `<tag>_f1_dance_s1_<clip>.onnx.pt`→`f1_dance_s1_<clip>.onnx`；`tools/f1_acceptance.py --tag f1_dance_s1` |
| contact_reward 0.25 | TASK_20261128_063 | 85 | PartyMimic `--contact_reward 0.25` | **A4 右脚 84.7%**（更差），`eval/f1_dance_contact/` | 同上，tag `f1_dance_contact` |
| flat85 数据子集 | TASK_20261128_064 | 83 | 训练库 13 段→8 段（双脚 ≥85%） | **A4 右脚 87.1%**（+0.7pp），`eval/f1_dance_flat85/` | 同上，tag `f1_dance_flat85` |

数据侧现状（无可补窗口的证据）：8 个 LAFAN 舞蹈源的全部扫窗 JSON（w90/w150/w300）里，过物理门（1–6、8、9）且双脚平贴 ≥80% 的窗口共 20 个，全部已在库中或与库重叠；LAFAN1 没有其它舞蹈文件；UMR 的 F1 数据对应 F1.5 模型（A14 核对，其舞蹈片段未过门）。

## gradmotion 任务（11-28 03:40 前）

| 任务号 | 账号 | 内容 | 结果 |
|---|---|---|---|
| TASK_20261127_598/564/563 | 79 | A14 的 smoke/full（旧 bootstrap） | 失败（条件图节点；564 的状态 5 是 check=False 掩盖） |
| **TASK_20261127_632** | 85 (PRO_20261127_084) | v1 训练 12k it，`f1_dance_v1` | 完成 11-27 19:24；产物见 `gm task model list` |
| TASK_20261127_676 | 85 | 原生回放（启动脚本含 %&= 被平台清空） | 空跑 |
| TASK_20261127_681/684/748/762/779/800 | 83 (PRO_20261127_085) | v2 36k 的各次提交 | 681 克隆 GnuTLS；684 uv sync 停滞；748/762/779 回退代码 bug；**800 成功**（11-28 01:39） |
| TASK_20261127_685/731/749/763/780/801 | 85 | 原生回放的各次提交 | 685 校验批维度；731/749/763/780 uv sync/回退 bug；**801 成功**（11-27 23:49） |
| TASK_20261127_736 | 85 | 网络探针 | pytorch.org 停滞、PyPI 正常（11-27 21:25） |
| TASK_20261128_031/038 | 83 | v2 周期 checkpoint 原生评估 + 最佳导出 | 031 缺 --run_dir；**038 成功**（11-28 02:48，选 26k） |

checkpoint 路径：`gm task model list --task-id TASK_20261127_632`（v1：`f1_dance_v1_ckpt_00xxxx.pt`、`f1_dance_v1_last.pt`、`f1_dance_v1_*_<clip>.onnx.pt`）；`--task-id TASK_20261127_800`（v2：`f1_dance_v2_36k_ckpt_002000…036000.pt`、`f1_dance_v2_36k_last.pt`、ONNX）；`--task-id TASK_20261128_038`（26k 的 4 个验收 ONNX、4 个原生评估 json）。本地下载脚本 `work/gm_fetch_models.sh <账号池id> <task> <dest> [子串]`；账号余额 11-28 03:30 约 85: 31、83: 10。

## 下一步（可选，未被要求且未执行）

- 若用户希望继续攻 A4：余量只有 3.6 个百分点、参考自身 93.7%，可选方向是数据侧（换/补脚掌平贴率余量更大的 Tier A 窗口需重新预注册验收片段，本次未做）或方法侧（如 `--contact_reward` 等改变奖励的 PartyMimic 选项），都不在本次范围内。
- v2 的 26k checkpoint 与 v1 等价，可任选其一作为交付策略；本仓库交付 v1。

## 注意事项

- gradmotion：`gm task logs` 只回最后 200 KB（容器完整日志在 `task info` 的 `argoLogDownloadUrl`）；启动脚本含 `% & =` 会被清空（用 base64）；download.pytorch.org 晚间可能停滞（bootstrap 已有阿里云镜像回退）；github.com 克隆偶发 GnuTLS 错误（已加重试）；余额接口连续调用会误报 401。
- 本机 raw.githubusercontent.com 偶发失败，用 GitHub API（blob base64）取文件。
