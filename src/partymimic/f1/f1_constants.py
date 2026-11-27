"""F1 29DOF (NIO F1, SYS-SPEC-0002-V1.1) entity for PartyMimic.

Kinematics, inertias and joint limits come from the vendor MJCF
``F1_29DOF/mjcf/robot/f1/f1_serial.xml`` (md5 a6c135b5…, byte-identical to the asset used by the
Beyondminic-Weilai-X1 F1 dance line, whose nine retargeting gates this project reuses).

Actuators: joint-side PD with the gains of ``F1_29DOF/F1 下肢 kp_kd 选取.html`` for the legs and
the X1 PlanB gains for the upper body (``Beyondminic-Weilai-X1/F1_29DOF/f1_kp_kd.json``):

    hip_pitch 70/9.5   hip_roll 70/9.0   hip_yaw 90/4.2   knee 90/6.9
    ankle_pitch 60/4.6 ankle_roll 45/2.4
    lumbar_yaw 90/8    lumbar_roll 70/8  lumbar_pitch 100/8
    shoulder_pitch/roll 40/4   shoulder_yaw 40/2   elbow_pitch 30/1.9
    elbow_yaw 20/1.0   wrist_pitch/roll 15/1.0

The leg damping ratios were designed for these gains on the reflected motor armature
(EC-A8116-P1-18: 197 kg·mm² × 18² = 0.0638 kg·m²; EC-A4315-P2-36 ×2 through the parallel-ankle
linkage: pitch 0.196, roll 0.068 kg·m²), giving zeta 0.5–0.9. The upper-body armature is not
documented for F1; 0.01 kg·m² is used (the legs are the only ones the gains were derived for).
"""

from __future__ import annotations

import os

import mujoco
import numpy as np
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from partymimic.robot import MotionArrays, RobotSpec

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MJCF_PATH = os.path.join(REPO, "assets", "f1", "f1_party.xml")
MOTION_DIR = os.path.join(REPO, "data", "motions", "f1")

# Joint order of the MJCF (legs -> lumbar -> arms). PartyMimic asserts that the compiled entity's
# joint order equals this tuple, and the motion files are stored in it.
F1_JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_pitch_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_pitch_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "lumbar_yaw_joint", "lumbar_roll_joint", "lumbar_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint", "left_elbow_yaw_joint",
    "left_wrist_pitch_joint", "left_wrist_roll_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint", "right_elbow_yaw_joint",
    "right_wrist_pitch_joint", "right_wrist_roll_joint",
]  # fmt: skip

# Bodies in the tracking reward; the first one is the root (anchor). Same set as BeyondMimic's G1,
# with F1's link names (leg links have no _link suffix).
TRACKED_BODIES = [
    "base_link", "left_hip_roll", "left_knee_pitch", "left_ankle_roll",
    "right_hip_roll", "right_knee_pitch", "right_ankle_roll", "lumbar_pitch_link",
    "left_shoulder_roll_link", "left_elbow_pitch_link", "left_wrist_roll_link",
    "right_shoulder_roll_link", "right_elbow_pitch_link", "right_wrist_roll_link",
]  # fmt: skip

# joint -> (kp, kd, effort limit, armature)
LEG = {
    "hip_pitch": (70.0, 9.5, 130.0, 0.0638),
    "hip_roll": (70.0, 9.0, 130.0, 0.0638),
    "hip_yaw": (90.0, 4.2, 130.0, 0.0638),
    "knee_pitch": (90.0, 6.9, 130.0, 0.0638),
    "ankle_pitch": (60.0, 4.6, 241.875, 0.196),
    "ankle_roll": (45.0, 2.4, 218.4, 0.068),
}
UPPER = {
    "lumbar_yaw": (90.0, 8.0, 150.0, 0.01),
    "lumbar_roll": (70.0, 8.0, 150.0, 0.01),
    "lumbar_pitch": (100.0, 8.0, 180.0, 0.01),
    "shoulder_pitch": (40.0, 4.0, 20.0, 0.01),
    "shoulder_roll": (40.0, 4.0, 20.0, 0.01),
    "shoulder_yaw": (40.0, 2.0, 20.0, 0.01),
    "elbow_pitch": (30.0, 1.9, 20.0, 0.01),
    "elbow_yaw": (20.0, 1.0, 20.0, 0.01),
    "wrist_pitch": (15.0, 1.0, 10.0, 0.01),
    "wrist_roll": (15.0, 1.0, 10.0, 0.01),
}


def gains(joint_name: str) -> tuple[float, float, float, float]:
    key = joint_name.replace("left_", "").replace("right_", "").replace("_joint", "")
    return LEG[key] if key in LEG else UPPER[key]


def action_scale(joint_name: str) -> float:
    """BeyondMimic's 0.25 * effort / stiffness (rad per unit action)."""
    kp, _, effort, _ = gains(joint_name)
    return 0.25 * effort / kp


DEFAULT_JOINT_POS = {
    ".*_hip_pitch_joint": -0.2915,
    ".*_knee_pitch_joint": 0.45,
    ".*_ankle_pitch_joint": -0.1585,
}


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(MJCF_PATH)
    # Self-collision off (as PartyMimic does for G1): colliding geoms only touch the ground.
    for geom in spec.geoms:
        if geom.contype or geom.conaffinity:
            geom.contype, geom.conaffinity = 0, 1
    return spec


def _actuators() -> tuple[BuiltinPositionActuatorCfg, ...]:
    acts = []
    for name in F1_JOINT_NAMES:
        kp, kd, effort, armature = gains(name)
        acts.append(BuiltinPositionActuatorCfg(
            target_names_expr=(name,), stiffness=kp, damping=kd,
            effort_limit=effort, armature=armature,
        ))
    return tuple(acts)


def get_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.586), joint_pos=DEFAULT_JOINT_POS,
                                             joint_vel={".*": 0.0}),
        spec_fn=get_spec,
        articulation=EntityArticulationInfoCfg(actuators=_actuators(), soft_joint_pos_limit_factor=0.9),
    )


def load_motion_npz(path: str) -> MotionArrays:
    """PartyMimic npz: body 0 = pelvis, joint_pos in F1_JOINT_NAMES order, fps."""
    d = np.load(path)
    fps = float(np.asarray(d["fps"]).reshape(-1)[0])
    return d["body_pos_w"][:, 0], d["body_quat_w"][:, 0], d["joint_pos"], fps


SPEC = RobotSpec(
    name="f1",
    mjcf_path=MJCF_PATH,
    joint_names=tuple(F1_JOINT_NAMES),
    tracked_bodies=tuple(TRACKED_BODIES),
    default_joint_pos=DEFAULT_JOINT_POS,
    action_scale={n: action_scale(n) for n in F1_JOINT_NAMES},
    com_body="lumbar_pitch_link",
    com_range=((-0.025, 0.025), (-0.05, 0.05), (-0.05, 0.05)),
    robot_cfg=get_robot_cfg,
    motion_dir=MOTION_DIR,
    motion_glob="*.npz",
    load_motion=load_motion_npz,
    feet_bodies=("left_ankle_roll", "right_ankle_roll"),
)
