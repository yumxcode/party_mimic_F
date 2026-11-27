# PartyMimic 数据需求结论（F1 29DOF 跳舞）

日期：2026-11-27 · 依据：PartyMimic `main@0c17a63`（README、`src/partymimic/motion.py`、`train.py`、`robot.py`、`env.py`）

## 1. 算法与训练入口

PartyMimic 是 SuperMimic 系的模型化强化学习跟踪器：在线拟合一个 MLP 世界模型，策略通过 32 步想象（BPTT）训练，critic 覆盖更长的 horizon；**一个策略覆盖整个动作库**，不是一段动作一个策略。

- 训练入口：`python -m partymimic.train --robot <pkg> --motion_dir <dir>`（`scripts/train.sh`）。
- 依赖：`mjlab>=1.6.0`、`torch`、`mujoco-warp`、`warp-lang`；`uv sync` 按 `uv.lock` 装出作者环境（torch 2.9.0+cu128）。Linux x86_64 + 单卡 24 GB。
- 机器人以 `src/partymimic/<robot>/<robot>_constants.py` 的 `RobotSpec` 描述：MJCF、关节顺序、tracked bodies、默认姿态、action scale、执行器（kp/kd/effort/armature）、动作加载器。
- 导出：`tools/export_onnx.py` 把策略 + 一段参考动作烘进 ONNX（输入 `obs_history (1,H,O)`、`time_step (1,1)`，输出 action + 参考状态），元数据里带 kp/kd/q0/action_scale。这就是 sim2sim 的部署格式。

## 2. 需要的运动数据形态（逐字依据）

README「Motion format」原文：

> `--motion_dir` takes a directory of `.npz` clips, each with `fps` (50), `joint_pos` `[T, J]` in the robot's joint order (`src/partymimic/<robot>/`), and the pelvis pose as the first body of `body_pos_w` `[T, B, 3]` and `body_quat_w` `[T, B, 4]` (wxyz). PartyMimic computes the other bodies by forward kinematics.

代码实现 `src/partymimic/motion.py::load_npz`：

```python
def load_npz(path):
    d = np.load(path)
    return d["body_pos_w"][:, 0], d["body_quat_w"][:, 0], d["joint_pos"], float(d["fps"].reshape(-1)[0])
```

即：**只读 body 0（骨盆）位姿与 `joint_pos`**，其余 body 由 `Kinematics`（torch FK，从同一 MJCF 建）算出。`B` 可以是 1。

结论：

| 项 | 要求 | 说明 |
|---|---|---|
| 文件 | 每 clip 一个 `.npz` | 目录由 `--motion_dir` 指定 |
| `fps` | 标量，50 | 非 50 会被 `MotionLib` 线性/slerp 重采样到 50 Hz |
| `joint_pos` | `[T, J]`，**机器人 MJCF 关节顺序** | F1 的 MJCF 顺序是 腿→腰→臂（与 URDF/UMR 的 腰→臂→腿 不同，必须显式置换） |
| `body_pos_w` | `[T, B, 3]`，body 0 = 骨盆 | 世界系 |
| `body_quat_w` | `[T, B, 4]`，wxyz | 世界系 |
| 根速度 | 不需要 | 由位姿数值差分得到 |
| AMP 数据 | **不需要** | PartyMimic 不是 AMP/对抗方法 |
| teacher / 先验策略 | **不需要** | 从零训练（`--init` 只用于微调） |
| 奖励标注 / 接触标签 | **不需要** | 奖励由参考状态与机器人 FK 解析计算 |

## 3. 数据量

- 论文/README 的通用配方：41 h、10,745 段（LAFAN1 40 + BONES-SEED 10k + KungfuAthlete 635 + XtremeMotion 14 + 自采跪爬 56），12k 迭代 ≈ 28 min（单张 RTX 4090）。
- 这不是算法的**下限**而是"通用跟踪"的数据量。README 明确：`--init <ckpt>` 可在"a few clips"上微调，2k 迭代把 3 段困难 ground 过渡的误差从 120–415 mm 降到 24–36 mm。
- 因此对"复现指定舞蹈"这一目标，数据量的决定因素是**覆盖的 1 s bin 数**（`MotionLib` 以 1 s bin 为重采样单位）：本项目的 F1 舞蹈库 13 段 / 76.4 s = 76 个 bin，另留 1 段 5 s 作 held-out。

## 4. 本项目的数据来源与对齐结论

| 项 | 结论 |
|---|---|
| 源 | LAFAN1 dance1_subject1..3、dance2_subject1..5（本地 `~/code/Beyondminic-Weilai-X1/downloads/lafan1`）+ UMR `umr_data/F1/lafan1_dance/dance1_subject2_smplx_weichai_f1_29.npz`（用于核对关节定义，未入库） |
| 重定向 | GMR（Beyondminic 线已验证的 F1 管线，`tools_bvh_to_csv_f1.py`，`--arm_relock c3.0_r0.92 --clamp_margin 0.06 --stance_widen 0.075`） |
| 资产一致性 | 本地 `F1_29DOF/mjcf/robot/f1/f1_serial.xml` 与 Beyondminic 的 MD5 相同（`a6c135b5…`）→ 同一套关节定义/限位/inertia |
| UMR 关节顺序 | = URDF 顺序（腰→臂→腿），与 MJCF 顺序（腿→腰→臂）不同，置换 `[17..28, 0..16]`；四元数 wxyz。FK 验证：置换后脚底 z 均值 0.011 m、零限位违反；不置换脚底 z 均值 0.41 m（错） |
| 帧率 | 源 30 fps → 输出 50 fps（线性插值 + 四元数 slerp） |

## 5. 未满足/风险

- 严格九门下可用的 LAFAN1 舞蹈窗口很少（见 `docs/retarget_gates.md`）：13 段 76.4 s，远小于论文的 41 h。这是数据源（LAFAN1 舞蹈大量踮脚/地板动作）与 F1 可达性共同决定的，不是流程缺陷。
- 通用化能力因此受限：策略预期是"复现这批舞蹈"而非"零样本跟任意舞蹈"。
