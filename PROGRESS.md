# PROGRESS — F1 29DOF PartyMimic 跳舞（A14 续跑）

> 可随时中断续接的进度文件。每完成一个阶段更新并提交。时间均为 CST。

## 当前状态（更新于 2026-11-27 19:45）

| 阶段 | 状态 | 说明 |
|---|---|---|
| 数据/重定向/九门/预注册验收门 | 完成（A14） | `docs/retarget_gates.md`、`docs/acceptance_plan.md`、`data/motions/f1/`（13 段 76.3 s）+ `data/testsets/f1_holdout/`（1 段） |
| 接手核对 + bootstrap 修复 | 完成 | 根因：驱动 535 无条件图节点；mujoco_warp `graph_conditional=False` + mjlab 图捕获关闭 |
| 远端训练 v1 | **完成** | TASK_20261127_632：12000 it / 0.594 h / 5.72 it/s；产物已下载（4 个 ONNX、train_log） |
| sim2sim 工具 | 完成 | 重写物理/观测对拍、A4/A5、kinematic 自检；修了 6D 相对姿态元素顺序 bug（修前回放 0.3 s 内倒，属工具错误） |
| 验收 A1–A5 | **3/4 段通过，总体未通过** | `dance1_subject3_255_706` A4 右脚 86.4% < 90%；其余全过（A2 7.4–9.4 mm，held-out 44.5 mm）。见 `docs/sim2sim_report.md` |
| 对比视频 | 完成 | `videos/f1_dance_v1_compare.mp4`（18.7 MB） |
| 原生栈回放核对 | 进行中 | **TASK_20261127_801**（23:31 提交，账号 85，bootstrap a3114d5：轮子按真实文件名保存）；下载 `./work/gm_fetch_models.sh 85 TASK_20261127_801 eval/native`。背景：685 已证实原生整段 749 步不终止（记录未发布）；731 卡在 uv sync 已停止。原计划：ONNX-vs-torch 校验 + 原生仿真器里的同一片段回放；判读规则见 docs/sim2sim_report.md §6；下载 `./work/gm_fetch_models.sh 85 TASK_20261127_731 eval/native`。（676 平台清空启动脚本而空跑；685 跑完原生回放 749 步——策略在 mjlab 里也全程未终止——但我的 ONNX 校验按整批喂入 fixed-batch 模型报错，产物未发布，已修为逐步校验） |
| 重训 v2（36k it，单变量） | 进行中 | **TASK_20261127_800**（23:29 启动，账号 83；uv 镜像回退生效，smoke 通过，训练中，预计 ~01:30 结束；748/749 因回退正则组号错误失败，762/763 因被杀死的 uv sync 留下 .venv/.lock 导致后续 sync 认为环境已同步而失败——镜像下载本身 25 s/901 MB 正常）：684 曾在 uv sync 卡住 74 min（download.pytorch.org 停滞，探针 736 实测），已改为 15 min 直连 → 阿里云 pytorch-wheels 镜像取同一 torch 轮子（sha256 对锁文件校验）→ `uv run --no-sync`。完成后 `./work/gm_fetch_models.sh 83 TASK_20261127_800 eval/f1_dance_v2_36k` + `tools/f1_acceptance.py --tag f1_dance_v2_36k` |
| 若 v2 仍未过 A4 | 待定 | 不再追加训练；如实报告精度极限与差距（lab_notebook exp_11393470 的止损分支） |

## gradmotion 任务

- 账号：account-pool id=85（giftBalance 50.0，5.76/h，最高）。A14 用的是 id=79（项目 PRO_20261127_007，余额 20，另有别的会话任务 579 在跑）。
- gm-cli 用法：`work/gm.sh <account-pool-id> <gm args>`（设置 HOME=.gmhome 与 GM_API_KEY，不回显密钥）。
- 任务：见下表（提交后填写）。

| 任务号 | 账号 | 启动脚本 | 状态 | 产物 / checkpoint |
|---|---|---|---|---|
| TASK_20261127_598 | 79 | `--smoke_then_full`（旧 bootstrap） | 失败 17:23 | 无 |
| **TASK_20261127_632** | 85 (PRO_20261127_084) | `gm-run party_mimic_F/scripts/f1_train.py --smoke_then_full --iterations 12000 --num_envs 4096 --tag f1_dance_v1`（bootstrap 7338c64） | **完成 19:24**（smoke 18:47 通过；12000 it，0.594 h） |
| TASK_20261127_676 | 85 | `f1_native_eval.py --ckpt_url <含 %&= 的 URL>` | 空跑（平台清空启动脚本） |
| TASK_20261127_681 | 83 | v2 36k（bootstrap e544c75） | 失败：github.com 克隆 GnuTLS 瞬时错误 |
| TASK_20261127_685 | 85 | `f1_native_eval.py --ckpt_url_b64 <base64>` | 失败：原生回放完成(749 步)但 ONNX 校验批维度报错，未发布产物 |
| TASK_20261127_731 | 85 | 同上，逐步校验版（bootstrap c081c08） | 排队/运行中（20:27 提交） |
| TASK_20261127_684 | 83 | v2 36k（bootstrap 1aa2c73） | uv sync 卡住，21:06 停止 |
| TASK_20261127_731 | 85 | 原生回放（c081c08） | uv sync 卡住，21:03 停止 |
| TASK_20261127_736 | 85 | `f1_net_probe.py` 下载速度探针 | 完成 21:25：pytorch.org torch 轮子 0 B/60 s（停滞），files.pythonhosted 4.9 MB/s，github 13 B/s |
| TASK_20261127_748 | 83 | v2 36k（f43d746） | 失败：直连 uv sync 超时后回退代码正则组号错误（IndexError） |
| TASK_20261127_749 | 85 | 原生回放（f43d746） | 同上失败 |
| TASK_20261127_762 | 83 | v2 36k（3f0e810） | 失败：镜像下载成功，但 uv sync --no-install-package 因残留 .venv/.lock 0.3 s 空过 |
| TASK_20261127_763 | 85 | 原生回放（3f0e810） | 同上失败 |
| TASK_20261127_779 | 83 | v2 36k（370f83c） | 失败：轮子文件名 %2B 不被 uv 接受 |
| TASK_20261127_780 | 85 | 原生回放（370f83c） | 同上失败 |
| **TASK_20261127_800** | 83 | v2 36k（a3114d5，镜像回退成功） | 完成 01:39（36000 it，1.72 h） |
| **TASK_20261128_031** | 83 | v2 checkpoint 评估+导出（`f1_ckpt_export.py`） | 01:53 提交 |
| **TASK_20261127_801** | 85 | 原生回放（a3114d5） | 23:31 提交 | 容器内 `/workspace/output/f1_dance_v1_*.pt`；日志 `/workspace/PartyMimic/logs/f1_dance_v1/train_log.csv`；下载：`./work/gm.sh 85 task model list --task-id TASK_20261127_632`，`./work/gm_fetch_models.sh 85 TASK_20261127_632 eval/f1_dance_v1` |

注意：建任务时不要填 `personalDataPath=/personal`（账号 85 无个人存储，run 会报 601 "没有文件"），已用 `task edit` 清空。

## 下一步

1. 训练 TASK_20261127_632：进度只看产物——`./work/gm.sh 85 task logs --task-id TASK_20261127_632 --raw --no-request-log | grep -a -o 'published [^ ]*'`（训练子进程输出被 bootstrap 收集，任务日志里没有 it/s）；SDK 文件检测时间戳给出 ≈5.7 it/s（2000 it ≈ 5.8 min），12000 it 预计 19:22 结束，随后导出 14 个 ONNX（`.onnx.pt`）。
2. 结束后：`./work/gm_fetch_models.sh 85 TASK_20261127_632 eval/f1_dance_v1`，把 `*.onnx.pt` 改回 `.onnx`，`train_log_csv.pt` 改回 csv；确认 train_log 最后一行 iteration=12000（任务状态 5 不等于训练完成）。
3. `python tools/f1_acceptance.py --onnx_dir eval/f1_dance_v1 --tag f1_dance_v1 --out eval/f1_dance_v1/acceptance` → A1–A5 表（stance 8 帧@50 Hz；A4 样本 <10 帧标 INVALID）。
4. `python tools/f1_render_compare.py --rollout eval/f1_dance_v1/acceptance/*.npz --out videos/f1_dance_v1_compare.mp4 --stride 2`（<25 MB）。
5. 报告 docs/sim2sim_report.md + README 状态；未过门则如实写差距，属数据/重定向问题的回数据层补库重训。

## 注意事项

- `gm task logs` 只保留最后 200 KB；uv sync 期间 SDK 噪声会淹没早期 `[boot]` 行，bootstrap 结束时会重打摘要。
- 本机 raw.githubusercontent.com 偶发失败，用 GitHub API（blob base64）取文件。
