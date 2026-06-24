#!/usr/bin/env python3
"""Render an SMPL-X head TALKING + TURNING, driven by a FLAME source (LAM or FLOAT-retargeted),
to A/B the two lip sources on a head that actually rotates (the FLOAT-can't-turn point).
  python head_ab_render.py --arkit a.json --out o.mp4 --audio a.wav [--frames 125 --fps 25 --yaw 28]"""
import os; os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import sys, argparse, subprocess, numpy as np, torch, smplx, pyrender, trimesh
from PIL import Image
sys.path.insert(0, "/lw/render"); from face_flame import FlameDriver
from render_smplx import _aim
import json as _json

ap = argparse.ArgumentParser()
ap.add_argument("--arkit", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--audio", required=True); ap.add_argument("--frames", type=int, default=125)
ap.add_argument("--fps", type=int, default=25); ap.add_argument("--yaw", type=float, default=28.0)
ap.add_argument("--gender", default="male"); ap.add_argument("--mappings", default="/work/tools/mp2flame/mappings")
a = ap.parse_args()
F = a.frames
expr, jaw, leye, reye = FlameDriver(a.mappings).drive(_json.load(open(a.arkit)), F, a.fps, gain=0.4)
# head turn: yaw sweep (one left-right cycle) about vertical -> real 3D rotation while talking
th = np.radians(a.yaw) * np.sin(np.linspace(0, 2 * np.pi, F))
go = np.zeros((F, 3), np.float32); go[:, 1] = th               # global_orient yaw
model = smplx.create("/work/models", model_type="smplx", gender=a.gender, num_betas=10,
                     use_pca=False, flat_hand_mean=True, num_expression_coeffs=100, batch_size=F)
with torch.no_grad():
    v = model(global_orient=torch.from_numpy(go),
              expression=torch.from_numpy(expr), jaw_pose=torch.from_numpy(jaw),
              leye_pose=torch.from_numpy(leye), reye_pose=torch.from_numpy(reye)).vertices.numpy().astype(np.float32)
faces = model.faces.astype(np.int64)
# frame the head (top ~22% of the mesh height), fixed camera
ys = v[:, :, 1]; top = float(ys.max()); headc = top - 0.13
ctr = np.array([0.0, headc, 0.0], np.float32); YFOV = 0.6
dist = 0.34 / np.tan(0.5 * YFOV)
cam = _aim([0, headc, dist], ctr)
r = pyrender.OffscreenRenderer(420, 520); os.makedirs("/tmp/headab", exist_ok=True)
for fi in range(F):
    s = pyrender.Scene(bg_color=[0.1, 0.1, 0.12, 1], ambient_light=[0.5, 0.5, 0.5])
    s.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(v[fi], faces, process=False), smooth=True))
    s.add(pyrender.PerspectiveCamera(yfov=YFOV, aspectRatio=420 / 520), pose=cam)
    s.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.2), pose=_aim([1, headc + 2, dist + 1], ctr))
    s.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.3), pose=_aim([-2, headc + 1, dist], ctr))
    Image.fromarray(r.render(s)[0]).save(f"/tmp/headab/f_{fi:04d}.png")
r.delete()
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps), "-i", "/tmp/headab/f_%04d.png",
                "-i", a.audio, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-c:a", "aac", "-shortest", a.out], check=True)
print("HEAD_AB_OK", a.out, "F", F)
