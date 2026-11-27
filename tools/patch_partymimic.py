#!/usr/bin/env python3
"""Apply the F1 additions to a fresh PartyMimic checkout (in place).

Edits (all backward compatible):
  1. RobotSpec gains ``feet_bodies: tuple[str, ...] = ()`` — the foot bodies for the contact
     reward / reference contact. F1's leg links are called ``left_ankle_roll`` (no _link suffix),
     so train.py's hardcoded ``("left_ankle_roll_link", "right_ankle_roll_link")`` would leave
     FEET empty and trip the ``--contact_reward`` assertion.
  2. train.py takes _FEET from ``robot.feet_bodies`` when the robot defines it.
  3. env.py: the SimulationCfg's CUDA-graph switch (if the field exists) is turned off.
  4. train.py: ``PM_F1_EAGER_COLLECT=1`` keeps the collection step eager (no collection CUDA graph).
     Only used as a fallback: the gradmotion 4090D hosts have a CUDA driver < 12.4, where MuJoCo-Warp's
     conditional solver loop cannot be captured; the primary fix is ``graph_conditional=False`` in
     mujoco_warp (see scripts/f1_train.py), which makes the solver loop capturable as plain nodes.

Everything else (src/partymimic/f1, assets/f1, data/motions/f1) is copied in by the bootstrap.
"""
from __future__ import annotations

import sys
from pathlib import Path


def patch(path: Path, old: str, new: str, marker: str | None = None) -> None:
    """Replace ``old`` with ``new`` once; ``marker`` (default: first line of ``new``) identifies an
    already applied patch."""
    s = path.read_text()
    if marker is not None:
        if marker in s:  # an explicit marker is unique to the patched text
            print(f"[patch] already applied: {path.name}")
            return
    else:
        marker = new.strip().splitlines()[0].strip()
        if marker in s and old not in s:
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
    """Ankle-roll bodies for the contact reward and the reference foot contact; empty: G1 names."""''',
          marker="feet_bodies: tuple[str, ...] = ()")

    patch(repo / "src" / "partymimic" / "train.py",
          '''_FEET = ("left_ankle_roll_link", "right_ankle_roll_link")
FEET = [robot.tracked_bodies.index(b) for b in _FEET] if all(b in robot.tracked_bodies for b in _FEET) else []''',
          '''_FEET = tuple(robot.feet_bodies) or ("left_ankle_roll_link", "right_ankle_roll_link")
FEET = [robot.tracked_bodies.index(b) for b in _FEET] if all(b in robot.tracked_bodies for b in _FEET) else []''')

    patch_collect_eager_switch(repo / "src" / "partymimic" / "train.py")


def patch_collect_eager_switch(path: Path) -> None:
    """``PM_F1_EAGER_COLLECT=1``: never capture the collection step (the 3 warmup calls already run it
    eagerly, so this only keeps that path). Default 0: unchanged upstream behaviour."""
    s = path.read_text()
    if "_F1_EAGER_COLLECT" in s:
        print("[patch] already applied: train.py eager-collect switch")
        return
    old = 'CG = {"calls": 0}  # the collection step: static tensors and the graph'
    new = ('_F1_EAGER_COLLECT = os.environ.get("PM_F1_EAGER_COLLECT", "0") == "1"  # F1 fallback: no collection CUDA graph\n'
           'print(f"[f1] collection CUDA graph: {\'OFF (PM_F1_EAGER_COLLECT=1)\' if _F1_EAGER_COLLECT else \'on\'}", flush=True)\n'
           + old)
    assert old in s, "CG anchor not found"
    s = s.replace(old, new, 1)
    old2 = '''    if CG["calls"] <= 3:  # warmup on the default stream (Warp's own stream orders with it implicitly)
        out = _collect_core()'''
    new2 = '''    if CG["calls"] <= 3 or _F1_EAGER_COLLECT:  # warmup on the default stream; F1 fallback: always eager
        out = _collect_core()'''
    assert old2 in s, "collect_graphed_step warmup anchor not found"
    s = s.replace(old2, new2, 1)
    path.write_text(s)
    print("[patch] ok: train.py eager-collect switch")


def patch_env_cuda_graph(repo: Path) -> None:
    """The gradmotion 4090D hosts run a CUDA driver < 12.4, so MuJoCo-Warp's CUDA-graph capture with
    conditional nodes raises "Conditional graph nodes require CUDA driver 12.4+". Turn the graph capture
    off in make_sim (same physics, just no capture)."""
    path = repo / "src" / "partymimic" / "env.py"
    s = path.read_text()
    if "use_cuda_graph" in s and "_F1_NO_CUDA_GRAPH" in s:
        print("[patch] already applied: env.py cuda graph")
        return
    old = """def make_sim(robot: RobotSpec) -> SimulationCfg:
    return SimulationCfg("""
    new = """def make_sim(robot: RobotSpec) -> SimulationCfg:  # _F1_NO_CUDA_GRAPH
    import dataclasses as _dc
    cfg = SimulationCfg("""
    assert old in s, "make_sim anchor not found"
    s = s.replace(old, new, 1)
    # close the call and disable the graph capture
    old2 = """            ccd_iterations=35,
        ),
    )"""
    new2 = """            ccd_iterations=35,
        ),
    )
    fields = getattr(cfg, "__dataclass_fields__", {}) or {}
    print("[f1] SimulationCfg fields:", sorted(fields), flush=True)
    for name in ("use_cuda_graph", "cuda_graph", "capture_cuda_graph", "graph"):
        if name in fields:
            cfg = _dc.replace(cfg, **{name: False})
            print(f"[f1] disabled SimulationCfg.{name}", flush=True)
            break
    return cfg"""
    assert old2 in s, "SimulationCfg tail anchor not found"
    s = s.replace(old2, new2, 1)
    path.write_text(s)
    print("[patch] ok: env.py cuda graph")


if __name__ == "__main__":
    repo = Path(sys.argv[1]).resolve()
    main(repo)
    patch_env_cuda_graph(repo)
