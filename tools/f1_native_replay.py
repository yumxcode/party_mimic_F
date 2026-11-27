"""Replay one clip from its first frame in PartyMimic's own simulator (mjlab / MuJoCo-Warp), recording
the observation history, goal, action and robot state of every control step, exactly as
partymimic.evaluate does (eval_mode, no randomization, hist = first observation repeated).

With --onnx: run that exported ONNX on every recorded observation history and report the max abs
difference to the torch policy's action (the export check export_onnx.py skips without onnxruntime).

Run inside the PartyMimic checkout:
    uv run --no-sync python tools/f1_native_replay.py --robot f1 --motion data/motions/f1/<clip>.npz \
        --ckpt logs/f1_dance_v1/last.pt --out logs/native_<clip>.npz [--onnx <clip>.onnx]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

import partymimic.networks as networks
from partymimic.dr import XiLayout
from partymimic.env import TrackingSim, make_sim
from partymimic.kinematics import Kinematics
from partymimic.mdp import MDP
from partymimic.motion import MotionLib
from partymimic.networks import Policy, policy_kwargs
from partymimic.robot import get_robot
from partymimic.rotations import mat_to_quat

ap = argparse.ArgumentParser()
ap.add_argument("--robot", default="f1")
ap.add_argument("--motion", required=True)
ap.add_argument("--ckpt", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--onnx", default=None)
args = ap.parse_args()

dev = "cuda:0"
torch.manual_seed(0)
torch.backends.cuda.matmul.allow_tf32 = True
robot = get_robot(args.robot)
kin = Kinematics(robot, dev)
motion = MotionLib([args.motion], kin, dev, robot.load_motion)
ck = torch.load(args.ckpt, map_location=dev)
a_ = ck["args"]
motion.set_interface(a_.get("interface", "joint"), robot.tracked_bodies, a_.get("obs_err") or "none")
mdp = MDP(kin, motion, torch.tensor(robot.default_joint_pos_list(), device=dev), XiLayout(robot, dev).dim, robot.sensing)
env = TrackingSim(make_sim(robot), 1, robot, motion, mdp, dev, eval_mode=True, randomize=False)
networks.AMP_DTYPE = torch.bfloat16 if a_.get("amp", "none") == "bf16" else None
K, Hh = a_["goal_frames"], a_["hist"]
widths = lambda key, default: tuple(int(x) for x in a_.get(key, default).split(","))  # noqa: E731
pi = Policy(mdp.obs_dim, motion.goal_dim(K), robot.num_joints, hist=Hh,
            enc_hidden=widths("enc_hidden", "2048,1024,512,256"), dec_hidden=widths("dec_hidden", "2048,1024,512,256"),
            **policy_kwargs(a_, robot)).to(dev)  # fmt: skip
pi.load_state_dict(ck["pi"])
pi.eval()
print(f"checkpoint iteration {ck['iteration']}, clip {motion.names[0]} ({int(motion.clip_len[0])} frames), obs_dim {mdp.obs_dim}", flush=True)

env.done_eval[:] = False
env.failed_eval[:] = False
env.reset(torch.arange(1, device=dev), motion.clip_start.clone())
rec = {k: [] for k in ("hist", "goal", "action", "root_p", "root_quat", "q", "f", "alive", "term", "ref_root_p", "ref_q")}
with torch.no_grad():
    hist = None
    while not bool(env.done_eval.all()):
        S = env.state()
        o = env.observe()
        g = motion.goal(env.f, K)
        hist = o.unsqueeze(1).repeat(1, Hh, 1) if hist is None else torch.cat((hist[:, 1:], o.unsqueeze(1)), 1)
        a = pi(hist, g)
        alive = ~env.done_eval.clone()
        rec["hist"].append(hist[0].cpu()), rec["goal"].append(g[0].cpu()), rec["action"].append(a[0].cpu())
        rec["root_p"].append(S[0][0].cpu()), rec["root_quat"].append(mat_to_quat(S[1][0]).cpu()), rec["q"].append(S[4][0].cpu())
        rec["f"].append(int(env.f[0])), rec["alive"].append(bool(alive[0]))
        rec["ref_root_p"].append(motion.root_p[env.f][0].cpu()), rec["ref_q"].append(motion.q[env.f][0].cpu())
        out = env.step(a)
        rec["term"].append(bool(out["term"][0]))
        if not bool(alive[0]):
            break
res = {"clip": motion.names[0], "frames_played": int(sum(rec["alive"])), "frames_total": int(motion.clip_len[0]),
       "terminated": bool(rec["term"][-1]) if rec["term"] else None, "failed": bool(env.failed_eval[0])}
np.savez(args.out, **{k: np.array(v) for k, v in rec.items()})
print(json.dumps(res), flush=True)

if args.onnx:
    import onnxruntime as ort
    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    hs = np.stack(rec["hist"]).astype(np.float32)
    fs = np.array(rec["f"][: len(hs)], np.float32).reshape(-1, 1)
    acts = np.stack([sess.run(None, {"obs_history": hs[i:i + 1], "time_step": fs[i:i + 1]})[0][0] for i in range(len(hs))])
    ta = np.stack(rec["action"][: len(hs)])
    diff = np.abs(acts - ta)
    res["onnx_vs_torch_max_abs"] = float(diff.max())
    res["onnx_vs_torch_mean_abs"] = float(diff.mean())
    res["torch_amp"] = a_.get("amp", "none")
    print(json.dumps({k: res[k] for k in ("onnx_vs_torch_max_abs", "onnx_vs_torch_mean_abs", "torch_amp")}), flush=True)
    json.dump(res, open(os.path.splitext(args.out)[0] + "_onnxcheck.json", "w"), indent=1)
