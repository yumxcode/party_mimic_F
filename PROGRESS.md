# PROGRESS — F1 29DOF PartyMimic 跳舞（A14 续跑）

> 可随时中断续接的进度文件。每完成一个阶段更新并提交。时间均为 CST。

## 当前状态（更新于 2026-11-27 18:40）

| 阶段 | 状态 | 说明 |
|---|---|---|
| 数据/重定向/九门/预注册验收门 | 完成（A14） | `docs/retarget_gates.md`、`docs/acceptance_plan.md`、`data/motions/f1/`（13 段 76.4 s）+ `data/testsets/f1_holdout/`（1 段） |
| 接手核对 | 完成 | A14 的 gradmotion 任务 527 probe 成功；563/564/598 全部失败，**训练从未跑通、无 checkpoint**。根因：PartyMimic 自身采集 CUDA 图捕获 mjwarp.step 的条件图节点（驱动 < 12.4） |
| bootstrap 修复 | 完成（待远端验证） | `scripts/f1_train.py`：mujoco_warp `opt.graph_conditional=False`（官方对驱动<12.4 的建议）+ 采集图失败时自动回退 eager；checkpoint 看护线程；held-out 片段漏拷修复；训练日志随产物发布 |
| 远端训练 | 进行中 | 见下方"gradmotion 任务" |
| MuJoCo sim2sim + A1–A5 判定 | 未开始 | 依赖 checkpoint/ONNX |
| 对比视频 / 最终报告 | 未开始 | |

## gradmotion 任务

- 账号：account-pool id=85（giftBalance 50.0，5.76/h，最高）。A14 用的是 id=79（项目 PRO_20261127_007，余额 20，另有别的会话任务 579 在跑）。
- gm-cli 用法：`work/gm.sh <account-pool-id> <gm args>`（设置 HOME=.gmhome 与 GM_API_KEY，不回显密钥）。
- 任务：见下表（提交后填写）。

| 任务号 | 账号 | 启动脚本 | 状态 | 产物 / checkpoint |
|---|---|---|---|---|
| TASK_20261127_598 | 79 | `--smoke_then_full`（旧 bootstrap） | 失败 17:23 | 无 |
| **TASK_20261127_632** | 85 (PRO_20261127_084) | `gm-run party_mimic_F/scripts/f1_train.py --smoke_then_full --iterations 12000 --num_envs 4096 --tag f1_dance_v1`（bootstrap 7338c64） | **运行中，18:28 启动；smoke 18:47 通过（采集图捕获成功，eager 未启用），full run 18:48 开始** | 容器内 `/workspace/output/f1_dance_v1_*.pt`；日志 `/workspace/PartyMimic/logs/f1_dance_v1/train_log.csv`；下载：`./work/gm.sh 85 task model list --task-id TASK_20261127_632`，`./work/gm_fetch_models.sh 85 TASK_20261127_632 eval/f1_dance_v1` |

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
