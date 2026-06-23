#!/usr/bin/env python3
"""Render a Kimodo AMASS/SMPL-X npz as a CLEAR LIT human video FACING THE CAMERA — the source for
wan2gp VACE 'Transfer Human Motion' (wan2gp runs its own DWPose annotator on it). Fixed full-body
portrait frame so the whole standing figure is visible every frame.
  python vace_control_render.py <amass.npz> <out.mp4> [W H maxframes]"""
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import sys, subprocess
import numpy as np, torch, smplx, pyrender, trimesh
from PIL import Image
sys.path.insert(0, "/work"); sys.path.insert(0, "/lw/render")
from render_smplx import _aim

NPZ, OUT = sys.argv[1], sys.argv[2]
W = int(sys.argv[3]) if len(sys.argv) > 3 else 480
H = int(sys.argv[4]) if len(sys.argv) > 4 else 832
MAXF = int(sys.argv[5]) if len(sys.argv) > 5 else 0
d = np.load(NPZ, allow_pickle=True)
go = d["root_orient"].astype(np.float32); body = d["pose_body"].astype(np.float32)
trans = d["trans"].astype(np.float32); F = len(body)
fps = int(d["mocap_frame_rate"]) if "mocap_frame_rate" in d else 30
if MAXF and F > MAXF:
    go, body, trans, F = go[:MAXF], body[:MAXF], trans[:MAXF], MAXF


def aa2mat(a):
    t = np.linalg.norm(a)
    if t < 1e-8: return np.eye(3, dtype=np.float32)
    k = a / t; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return (np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)).astype(np.float32)


def mat2aa(R):
    tr = np.clip((R[0, 0] + R[1, 1] + R[2, 2] - 1) / 2, -1, 1); ang = np.arccos(tr)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]); n = np.linalg.norm(v)
    return (v / n * ang).astype(np.float32) if n > 1e-8 else np.zeros(3, np.float32)


th = np.radians(180.0)                                          # yaw 180 -> face camera
Rfix = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], np.float32)  # Z-up -> Y-up
Ry = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]], np.float32)
M = (Ry @ Rfix).astype(np.float32)
go = np.stack([mat2aa(M @ aa2mat(go[i])) for i in range(F)])
trans = trans @ M.T

model = smplx.create("/work/models", model_type="smplx", gender="neutral", num_betas=10,
                     use_pca=False, flat_hand_mean=True, batch_size=F)
with torch.no_grad():
    v = model(global_orient=torch.from_numpy(go), body_pose=torch.from_numpy(body),
              transl=torch.from_numpy(trans)).vertices.numpy().astype(np.float32)
faces = model.faces.astype(np.int64)

allv = v.reshape(-1, 3); mn = allv.min(0); mx = allv.max(0); ctr = (mn + mx) / 2
bodyH = (mx[1] - mn[1]); YFOV = 0.85
dist = bodyH / (2 * np.tan(0.5 * YFOV)) * 1.15
cam = _aim([ctr[0], ctr[1], ctr[2] + dist], ctr)
r = pyrender.OffscreenRenderer(W, H); os.makedirs("/tmp/vctl", exist_ok=True)
for fi in range(F):
    s = pyrender.Scene(bg_color=[0.5, 0.5, 0.5, 1], ambient_light=[0.6, 0.6, 0.6])  # mid-grey bg
    s.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(v[fi], faces, process=False), smooth=True))
    s.add(pyrender.PerspectiveCamera(yfov=YFOV, aspectRatio=W / H), pose=cam)
    s.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.5), pose=_aim([ctr[0] + 1, ctr[1] + 2, ctr[2] + 3], ctr))
    s.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.5), pose=_aim([ctr[0] - 2, ctr[1] + 1, ctr[2] + 2], ctr))
    Image.fromarray(r.render(s)[0]).save(f"/tmp/vctl/f_{fi:04d}.png")
r.delete()
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", "/tmp/vctl/f_%04d.png",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", OUT], check=True)
print("VACE_CONTROL_OK", OUT, "frames", F, "fps", fps, f"{W}x{H}")
