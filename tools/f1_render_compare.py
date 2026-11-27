#!/usr/bin/env python3
"""Render a reference | policy comparison video from a sim2sim rollout (tools/f1_sim2sim.py --out).

Left: the reference motion (the ONNX's baked reference, i.e. the retargeted F1 dance).
Right: what the trained policy actually did in MuJoCo (CPU).
Both are played in the F1 MJCF with the vendor visual meshes (assets/f1/f1_party_vis.xml, meshes
from F1_29DOF/mjcf/meshes via the assets/f1/meshes symlink), same camera, same ground plane.

Usage:
  python tools/f1_render_compare.py --rollout eval/dance1_subject3.npz --out videos/dance1_subject3.mp4
  (several rollouts: --rollout a.npz b.npz ... --out videos/compare.mp4 concatenates them)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw

REPO = Path(__file__).resolve().parent.parent


def load_vis(mjcf_vis: Path, w: int, h: int) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(mjcf_vis))
    spec.visual.global_.offwidth = max(int(spec.visual.global_.offwidth), w)
    spec.visual.global_.offheight = max(int(spec.visual.global_.offheight), h)
    wb = spec.worldbody
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.01],
                rgba=[0.55, 0.57, 0.6, 1.0], contype=0, conaffinity=0)
    dsl = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    wb.add_light(name="key", pos=[1.5, -2.0, 3.0], dir=[-0.4, 0.5, -0.8], diffuse=[0.9, 0.9, 0.9],
                 specular=[0.2, 0.2, 0.2], type=dsl)
    wb.add_light(name="fill", pos=[-2.0, -1.0, 2.5], dir=[0.6, 0.3, -0.7], diffuse=[0.35, 0.35, 0.35],
                 specular=[0.0, 0.0, 0.0], type=dsl)
    return spec.compile()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rollout", nargs="+", required=True)
    ap.add_argument("--mjcf_vis", default=str(REPO / "assets/f1/f1_party_vis.xml"))
    ap.add_argument("--out", default="videos/compare.mp4")
    ap.add_argument("--fps", type=int, default=50, help="output frame rate = rollout frame rate")
    ap.add_argument("--stride", type=int, default=1, help="keep every k-th frame")
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=640)
    ap.add_argument("--quality", type=int, default=8)
    args = ap.parse_args()

    vis = load_vis(Path(args.mjcf_vis), args.w, args.h)
    vdata = mujoco.MjData(vis)
    ren = mujoco.Renderer(vis, height=args.h, width=args.w)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance = 3.0
    cam.azimuth = 150
    cam.elevation = -10

    import imageio.v2 as imageio
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(args.out, fps=args.fps // args.stride, codec="libx264", quality=args.quality,
                                macro_block_size=None)
    n_out = 0
    for rf in args.rollout:
        d = np.load(rf, allow_pickle=True)
        q, body_pos, ref_p, ref_q = d["q"], d["body_pos"], d["ref_p"], d["ref_q"]
        rq, ref_rq, jnames = d["root_quat"], d["ref_root_quat"], [str(x) for x in d["joint_names"]]
        jid = [vis.joint(n).id for n in jnames]
        qadr = np.array([vis.jnt_qposadr[j] for j in jid])
        # a fixed camera per clip: centred on the reference root, far enough for both robots
        centre = np.concatenate((ref_p[:, 0, :2].mean(0), [0.7]))
        cam.lookat[:] = centre
        cam.distance = max(2.8, 1.6 * float(np.ptp(ref_p[:, 0, 0]) + 0.5))
        for t in range(0, len(q), args.stride):
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
            img = np.concatenate((left, right), 1)
            im = Image.fromarray(img)
            dr = ImageDraw.Draw(im)
            dr.rectangle([0, 0, args.w, 26], fill=(0, 0, 0))
            dr.rectangle([args.w, 0, 2 * args.w, 26], fill=(0, 0, 0))
            dr.text((10, 6), f"reference  {Path(rf).stem}  t={t / args.fps:.2f}s", fill=(255, 255, 255))
            dr.text((args.w + 10, 6), f"policy (MuJoCo sim2sim)  frame {t}/{len(q)}", fill=(255, 255, 255))
            writer.append_data(np.asarray(im))
            n_out += 1
    writer.close()
    size = Path(args.out).stat().st_size / 1e6
    print(f"wrote {args.out}: {n_out} frames, {n_out / (args.fps / args.stride):.1f}s, {size:.1f} MB")


if __name__ == "__main__":
    main()
