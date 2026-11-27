#!/usr/bin/env python3
"""Apply the F1 additions to a fresh PartyMimic checkout (in place).

Edits (both backward compatible):
  1. RobotSpec gains ``feet_bodies: tuple[str, ...] = ()`` — the foot bodies for the contact
     reward / reference contact. F1's leg links are called ``left_ankle_roll`` (no _link suffix),
     so train.py's hardcoded ``("left_ankle_roll_link", "right_ankle_roll_link")`` would leave
     FEET empty and trip the ``--contact_reward`` assertion.
  2. train.py takes _FEET from ``robot.feet_bodies`` when the robot defines it.

Everything else (src/partymimic/f1, assets/f1, data/motions/f1) is copied in by the bootstrap.
"""
from __future__ import annotations

import sys
from pathlib import Path


def patch(path: Path, old: str, new: str) -> None:
    s = path.read_text()
    if new.strip().splitlines()[0].strip() in s and old not in s:
        print(f"[patch] already applied: {path.name}")
        return
    assert old in s, f"anchor not found in {path}:\n{old}"
    path.write_text(s.replace(old, new, 1))
    print(f"[patch] ok: {path.name}")


def main(repo: Path) -> None:
    src = repo / "src" / "partymimic"

    patch(src / "robot.py",
          '''    mirror_sign: tuple[float, ...] | None = None
    """Sign of each joint under the left-right mirror (joint_names order); None: flip roll and yaw joints (G1 axes)."""''',
          '''    mirror_sign: tuple[float, ...] | None = None
    """Sign of each joint under the left-right mirror (joint_names order); None: flip roll and yaw joints (G1 axes)."""
    feet_bodies: tuple[str, ...] = ()
    """Ankle-roll bodies for the contact reward and the reference foot contact; empty: G1 names."""''')

    patch(repo / "src" / "partymimic" / "train.py",
          '''_FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
FEET = [robot.tracked_bodies.index(b) for b in _FEET] if all(b in robot.tracked_bodies for b in _FEET) else []''',
          '''_FEET = tuple(robot.feet_bodies) or ("left_ankle_roll_link", "right_ankle_roll_link")
FEET = [robot.tracked_bodies.index(b) for b in _FEET] if all(b in robot.tracked_bodies for b in _FEET) else []''')


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
