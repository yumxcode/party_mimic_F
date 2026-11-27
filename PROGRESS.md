# PROGRESS — F1 29DOF PartyMimic 跳舞（A14 续跑）

> 可随时中断续接的进度文件。时间均为 CST。**本任务已到终止状态（2026-11-28 03:30）：按预注册止损分支不再追加训练；若续接，从"下一步（可选）"开始。**

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

## gradmotion 任务（全部已结束）

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
