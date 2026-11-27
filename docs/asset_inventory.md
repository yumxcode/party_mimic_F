# F1 29DOF 资产盘点（本地 `F1_29DOF/`）

## 模型入口
- `F1_29DOF/mjcf/f1_flat.xml` = `robot/f1/f1_serial.xml` + `environment/flat.xml`
- 主模型：`F1_29DOF/mjcf/robot/f1/f1_serial.xml`（compiler angle=radian, meshdir=meshes/, timestep=0.001）
- 备用：`urdf/SYS-SPEC-0002-V1.1-F1整机模型29DOF.urdf`、`mjcf/robot/f1/f1_isaac.urdf`、`mjcf/_f1_source.urdf`

## 关节（MJCF 遍历顺序，共 29 驱动 + 1 free）
| # | body | joint | range(rad) | axis | actuatorfrcrange | damping | frictionloss |
|---|---|---|---|---|---|---|---|
| 0 | base_link | floating_base (free) | - | - | - | - | - |
| 1 | left_hip_pitch | left_hip_pitch_joint | -1.5708..1.5708 | 0 1 0 | -130..130 | 0.3 | 0.5 |
| 2 | left_hip_roll | left_hip_roll_joint | -0.349066..1.5708 | 1 0 0 | -130..130 | 0.3 | 0.5 |
| 3 | left_hip_yaw | left_hip_yaw_joint | -1.5708..1.5708 | (default 0 0 1) | -130..130 | 0.3 | 0.5 |
| 4 | left_knee_pitch | left_knee_pitch_joint | 0..1.5708 | 0 1 0 | -130..130 | 0.3 | 0.5 |
| 5 | left_ankle_pitch | left_ankle_pitch_joint | -0.523599..0.523599 | 0 1 0 | -241.875..241.875 | 0.1 | 0.2 |
| 6 | left_ankle_roll | left_ankle_roll_joint | -0.523599..0.523599 | 1 0 0 | -218.4..218.4 | 0.1 | 0.2 |
| 7 | right_hip_pitch | right_hip_pitch_joint | -1.5708..1.5708 | 0 1 0 | -130..130 | 0.3 | 0.5 |
| 8 | right_hip_roll | right_hip_roll_joint | -1.5708..0.349066 | 1 0 0 | -130..130 | 0.3 | 0.5 |
| 9 | right_hip_yaw | right_hip_yaw_joint | -1.5708..1.5708 | (default 0 0 1) | -130..130 | 0.3 | 0.5 |
| 10 | right_knee_pitch | right_knee_pitch_joint | 0..1.5708 | 0 1 0 | -130..130 | 0.3 | 0.5 |
| 11 | right_ankle_pitch | right_ankle_pitch_joint | -0.523599..0.523599 | 0 1 0 | -241.875..241.875 | 0.1 | 0.2 |
| 12 | right_ankle_roll | right_ankle_roll_joint | -0.523599..0.523599 | 1 0 0 | -218.4..218.4 | 0.1 | 0.2 |
| 13 | lumbar_yaw_link | lumbar_yaw_joint | -1..1 | (default 0 0 1) | -150..150 | 1 | - |
| 14 | lumbar_roll_link | lumbar_roll_joint | -0.25..0.25 | (default 0 0 1) | -150..150 | 1 | - |
| 15 | lumbar_pitch_link | lumbar_pitch_joint | -0.15..0.45 | 0 0 -1 | -180..180 | 1 | - |
| 16 | left_shoulder_pitch_link | left_shoulder_pitch_joint | -2.5..1.5 | 0 0 -1 | -20..20 | 1 | - |
| 17 | left_shoulder_roll_link | left_shoulder_roll_joint | -2..0 | 0 0 -1 | -20..20 | 1 | - |
| 18 | left_shoulder_yaw_link | left_shoulder_yaw_joint | -1.8..1.8 | (default) | -20..20 | 1 | - |
| 19 | left_elbow_pitch_link | left_elbow_pitch_joint | 0..2 | (default) | -20..20 | 1 | - |
| 20 | left_elbow_yaw_link | left_elbow_yaw_joint | -1.8..1.8 | (default) | -20..20 | 1 | - |
| 21 | left_wrist_pitch_link | left_wrist_pitch_joint | -0.4..0.4 | 0 0 -1 | -10..10 | 1 | - |
| 22 | left_wrist_roll_link | left_wrist_roll_joint | -0.4..0.4 | 0 0 -1 | -10..10 | 1 | - |
| 23 | right_shoulder_pitch_link | right_shoulder_pitch_joint | -2.5..1.5 | (default) | -20..20 | 1 | - |
| 24 | right_shoulder_roll_link | right_shoulder_roll_joint | -2..0 | (default) | -20..20 | 1 | - |
| 25 | right_shoulder_yaw_link | right_shoulder_yaw_joint | -1.8..1.8 | (default) | -20..20 | 1 | - |
| 26 | right_elbow_pitch_link | right_elbow_pitch_joint | 0..2 | 0 0 -1 | -20..20 | 1 | - |
| 27 | right_elbow_yaw_link | right_elbow_yaw_joint | -1.8..1.8 | (default) | -20..20 | 1 | - |
| 28 | right_wrist_pitch_link | right_wrist_pitch_joint | -0.4..0.4 | 0 0 -1 | -10..10 | 1 | - |
| 29 | right_wrist_roll_link | right_wrist_roll_joint | -0.4..0.4 | 0 0 -1 | -10..10 | 1 | - |

关节顺序：左右腿（hip pitch/roll/yaw → knee → ankle pitch/roll）→ 腰 3 → 左臂 7 → 右臂 7。
`f1_stand.json` 另给出 X1 传感器顺序（legs→lumbar→arms）与 URDF 顺序（lumbar→arms→legs）两套排列，两套内容一致。

## 站立关键帧（keyframe "stand"）
base z = 0.587565（qpos[2]），base x=0.000804861；hip_pitch=-0.291529, hip_roll=0, hip_yaw=-6.7e-5, knee=0.45, ankle_pitch=-0.158471, ankle_roll≈0；腰、臂全 0。
`f1_stand.json`: base_z=0.585565（sole_z=-0.585565 → 脚底贴地时 base 高 0.5856 m）。

## 质量
MJCF inertial 求和 = **32.22 kg**；`F1 下肢 kp_kd 选取.html` 页头写 35.03 kg（差异待核实，可能是含电池/外覆件或另一版模型）。

## kp/kd（来自 `F1 下肢 kp_kd 选取.html`，关节侧，N·m/rad, N·m·s/rad）
| 关节 | kp | kd |
|---|---|---|
| hip_pitch | 70 | 9.5 |
| hip_roll | 70 | 9.0 |
| hip_yaw | 90 | 4.2 |
| knee_pitch | 90 | 6.9 |
| ankle_pitch | 60 | 4.6 |
| ankle_roll | 45 | 2.4 |

设计依据：ζ = kd / 2√(kp·I_eff) 落在 0.5–0.9；armature 按 EC-A8116-P1-18（197 kg·mm²×18²=0.0638）与 EC-A4315-P2-36（25 kg·mm²×36²=0.0324 ×2×r²）。
上肢 kp/kd 页面未列出（页面只做下肢），需另行设定。

## 尺寸（页面给出）
- base/骨盆高 0.586 m（与 f1_stand.json base_z 一致）
- 小腿 0.331 m，足长 0.204 m
