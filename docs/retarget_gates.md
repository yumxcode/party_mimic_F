# 重定向过门记录（F1 29DOF，LAFAN1 舞蹈）

日期：2026-11-27 · 工具：`tools/check_motion_f1.py`（Beyondminic-Weilai-X1 的 F1 九门工具，资产 MD5 一致）与 `tools/f1_gate_windows.py`（同一套公式，先算每帧指标再逐窗口判定）

## 1. 九门定义与阈值出处

阈值**全部沿用** Beyondminic-Weilai-X1 的 `tools_check_motion_f1.py` 默认值，本项目未放宽任何一项。该工具在 v1–v5 的 F1 舞蹈线上用过，其资产与本项目 `F1_29DOF` 字节一致。

| # | 门 | 判据 | 阈值 | 出处 |
|---|---|---|---|---|
| 1 | ground | 支撑期脚底 4 角最低点 − 球半径 | > −0.015 m | 沿用（F1 脚底 AABB 标定） |
| 2 | self | 胶囊球近似的最小肢体间距 | > 0.005 m | 沿用 |
| 3 | limits | 关节角超出 URDF 限位 | 0 帧 | 沿用（F1 非对称 hip_roll ±20°/90°） |
| 4 | velocities | \|dq/dt\| ≤ 1.5× URDF 速度限（腕 2.0×） | 0 帧 | 沿用 |
| 5 | root_height | 骨盆高度带 | [0.35, 1.0] m | 沿用（F1 站立 0.59） |
| 6 | continuity | 有限值 + 相邻根四元数点积 | > 0.1 | 沿用 |
| 7 | foot_flatness | 支撑期 \|sole pitch\| ≤ 10° 且 \|sole roll\| ≤ 8° 的帧占比 | ≥ 90% | 沿用（在"已验证干净"的 dance_f1_train 96–98% 与被否决的 cmu137 0% 之间标定） |
| 8 | arm_quality | 任一臂关节贴限位（0.05 rad 内）帧占比 ≤ 10%，且每个腕 ROM ≥ 量程 2% | 沿用 |
| 9 | stance_openness | 双支撑脚距/髋宽 mean ≥ 1.15、p5 ≥ 0.70；双腿 hip roll 同时内收 >3° 的帧 ≤ 3% | 沿用（v5 新增，源校准：LAFAN 源 p5=0.53，0.70=源×1.3） |

## 2. 扫描与选窗

- 源：LAFAN1 `dance1_subject1..3`、`dance2_subject1..5`（8 个 BVH，共 1523 s）。
- 重定向：GMR `tools_bvh_to_csv_f1.py`，`--arm_relock c3.0_r0.92 --clamp_margin 0.06 --stance_widen 0.075`（v5 配方）。
- 扫描：窗口 300/150/90 帧（10/5/3 s @30 fps），步长 150/75/45 帧，逐窗口跑九门。
- 结果：严格九门下通过窗口极少（多数窗口败在 flatness 与 ground）。原因不是流程错误，而是 LAFAN1 舞蹈本身大量踮脚/地板动作——`dance1_subject1` 全片脚底平贴率仅 ~60%，即使把脚掌旋转权重从 10 提到 60 也只到 87/83%，仍不过 90%。
- 处理：只取**物理可行性门（1–6、8、9）全过**的窗口，再按 flatness 分两档：
  - **Tier A（验收档）**：九门全过。
  - **Tier B（仅训练）**：其余门全过、flatness ≥ 80%。文档化理由：90% 阈值是在"干净舞蹈"数据上标定的，LAFAN1 舞蹈的踮脚是编排而非缺陷；Tier B 不参与验收。
- 穿地修正：`tools/f1_build_library.py` 对重采样后的 50 Hz 数据做"仅抬根高"的高斯平滑抬升（Beyondminic `tools_lift_ground_f1.py` 的做法，关节角不变），迭代至最低脚底 ≥ −0.013 m。逐段 lift 量见 manifest。

## 3. 逐段结果

见 `docs/retarget_gate_table.md`（由 `work/gate/final/*.json` 生成）。

要点：

- **Tier A（4 段）**：`dance1_subject3_255_706`（15.0 s，flat 97/94）、`dance2_subject1_352_795`（14.7 s，flat 90/93）、`dance2_subject3_300_450`（5.0 s，flat 91/100）、`dance2_subject4_2835_2925`（3.0 s，flat 100/100）。
- **Tier B（10 段，仅训练）**：flat 80–95%，其余门全过。
- **held-out**：`dance2_subject3_300_450`（Tier A）不参与训练，只用于泛化检查。
- 训练库：13 段 / 76.4 s。

## 4. 未过门清单（如实列出）

`work/gate/*.json` 里记录了所有窗口的逐门结果。被淘汰的典型原因：

| 源 | 主要淘汰原因 |
|---|---|
| dance1_subject1 | flatness 60–83%、ground −0.05、self −0.07（wrist–torso） |
| dance1_subject2 | flatness < 80% |
| dance2_subject2 | flatness < 80% |
| dance2_subject5 | 未通过任何窗口（flatness/stance） |
| UMR dance1_subject2 | ground/self 过，但 flat 50/54%、height 0.329、stance 内收 17.6% |

## 5. 复算方式

```bash
# 单段（CSV 或 npz）
python tools/check_motion_f1.py --csv work/retarget_csv/dance1_subject3_f1.csv --fps 30 --json_out out.json
# 扫窗
python tools/f1_gate_windows.py --csv work/retarget_csv/dance2_subject3_f1.csv --out work/gate/x.json --win 150 --stride 75
```

## 6. v6 配方：脚掌平贴压缩（2026-11-28，用户授权重新预注册验收片段）

背景：v5 配方下，预注册验收片段 `dance1_subject3_255_706` 的参考右脚平贴率只有 93.7%（踮脚段 sole pitch 9.3–9.9°，门限 10°），6 个策略在 CPU MuJoCo / mjlab 回放中的右脚平贴率都停在 81–87%（`docs/sim2sim_report.md` §3–§10）。种子、迭代数、接触奖励、训练片段筛选都不改变这一点；现有 8 个 LAFAN 舞蹈源里过物理门且双脚 ≥80% 平贴的窗口全部已入库，无可补窗口。用户在"接受 3/4 / 改重定向并重新预注册 / 放宽约束"中授权 **改重定向并重新预注册**。

做法（`tools/f1_flatten_feet.py`，v5 CSV → v6 CSV）：对每只脚按 gate 7 的定义找支撑期（触地连续 ≥5 帧@30fps），支撑期内把 sole pitch 超过 4° 的部分、sole roll 超过 3° 的部分各乘 0.2（10° 踮脚 → 5.2°，18° → 6.9°），只改踝 pitch/roll 关节（FK 数值测得的增益，单关节改动 ≤0.35 rad），根高只升不降；支撑权重在 ±3 帧内渐变，修正量做 5 帧滑动平均。髋/膝/躯干/手臂、根 xy 与朝向不变；摆动腿的脚掌姿态不变（保留舞蹈内容）。随后沿用原流程：抬根（仅根高）→ 50 Hz 重采样 → 九门。

结果（`docs/retarget_gate_table.md`，`work/gate_flat/final/*.json`）：
- 四个验收窗口（同一帧区间）参考平贴率：`dance1_subject3_255_706` 97.2/93.7 → **100/100**（pitch p95 8.8/10.5° → 4.4/4.6°）；`dance2_subject1_352_795` 90.2/93.3 → 98.3/99.4；`dance2_subject4_2835_2925` 100/100 → 100/100；held-out `dance2_subject3_300_450` 90.5/100 → 100/100。九门全过。
- 扫窗（90/150/300 帧）：九门通过窗口从 v5 的 ~6 个增至 49 个（dance2_subject3 为主），训练库从 13 段/76.3 s（3 A + 10 B）改为 **14 段/85.3 s 全 Tier A**（3 个验收训练段 + 11 个不重叠的新 Tier A 窗口），held-out 不变。v5 库保留在 `data/motions/f1_v5recipe/`。
- 代价：踮脚被压平，动作与 LAFAN1 原舞者的脚掌姿态不再一致（这是用户授权的取舍）；踝关节在支撑期的改动均值 0.02 rad、最大 0.35 rad（少数帧）。

验收片段的身份、A1–A5 阈值与判定脚本不变（`docs/acceptance_plan.md` 修订 v2）。
