#!/usr/bin/env python3
"""Render a reference | policy comparison video from a sim2sim rollout (MuJoCo, CPU).

Left: the reference motion (the ONNX's baked reference, i.e. the retargeted F1 dance).
Right: what the trained policy actually did in MuJoCo.
Both are played in the same F1 MJCF with the vendor visual meshes, same camera, same timestep.

Usage:
  python tools/f1_render_compare.py --rollout eval/sim2sim.npz \
      --mjcf_vis assets/f1/f1_party_vis.xml --out videos/dance_compare.mp4 --fps 50
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", required=True)
    ap.add_argument("--mjcf_vis", default="assets/f1/f1_party_vis.xml")
    ap.add_argument("--out", default="videos/compare.mp4")
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=640)
    ap.add_argument("--camera", default="-1.6 -2.2 1.3 / 0 0 0.75")
    args = ap.parse_args()

    d = np.load(args.rollout, allow_pickle=True)
    q = d["q"]                      # (T, J) policy joints
    body_pos = d["body_pos"]        # (T, B, 3) policy tracked bodies
    ref_p = d["ref_p"]              # (T, B, 3) reference tracked bodies
    ref_q = d["ref_q"]              # (T, J) reference joints
    rq = d["root_quat"]            # (T, 4) policy root quat (wxyz)
    ref_rq = d["ref_root_quat"]    # (T, 4) reference root quat (wxyz)
    jnames = [str(x) for x in d["joint_names"]]
    T = len(q)

    vis = mujoco.MjModel.from_xml_path(args.mjcf_vis)
    vdata = mujoco.MjData(vis)
    jid = [vis.joint(n).id for n in jnames]
    qadr = np.array([vis.jnt_qposadr[j] for j in jid])
    root_bid = vis.body("base_link").id
    # reference root pose comes from the tracked root body
    ref_root_bid = vis.body("base_link").id

    ren = mujoco.Renderer(vis, height=args.h, width=args.w)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = [0, 0, 0.75]
    cam.distance = 2.8
    cam.azimuth = 145
    cam.elevation = -12

    import imageio.v2 as imageio
    writer = imageio.get_writer(args.out, fps=args.fps, codec="libx264", quality=8)
    for t in range(T):
        # reference
        mujoco.mj_resetData(vis, vdata)
        vdata.qpos[0:3] = ref_p[t, 0]
        vdata.qpos[3:7] = ref_rq[t]
        vdata.qpos[qadr] = ref_q[t]
        mujoco.mj_forward(vis, vdata)
        ren.update_scene(vdata, camera=cam)
        left = ren.render()

        mujoco.mj_resetData(vis, vdata)
        vdata.qpos[0:3] = body_pos[t, 0]
        vdata.qpos[3:7] = rq[t]
        vdata.qpos[qadr] = q[t]
        mujoco.mj_forward(vis, vdata)
        ren.update_scene(vdata, camera=cam)
        right = ren.render()
        writer.append_data(np.concatenate((left, right), 1))
    writer.close()
    print(f"wrote {args.out}: {T} frames, {T/args.fps:.1f}s")


if __name__ == "__main__":
    main()
