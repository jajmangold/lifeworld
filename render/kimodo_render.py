#!/usr/bin/env python3
"""Render Kimodo's AMASS/SMPL-X npz (native params — no conversion). AMASS is Z-up; rotate to
our Y-up. python kimodo_render.py <amass.npz> <out.mp4>"""
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import sys, subprocess
import numpy as np, torch, smplx, pyrender, trimesh
from PIL import Image
sys.path.insert(0, "/work"); sys.path.insert(0, "/lw/render")
from render_smplx import _aim

NPZ, OUT = sys.argv[1], sys.argv[2]
d = np.load(NPZ, allow_pickle=True)
go = d["root_orient"].astype(np.float32); body = d["pose_body"].astype(np.float32)
trans = d["trans"].astype(np.float32); F = len(body)
fps = int(d["mocap_frame_rate"]) if "mocap_frame_rate" in d else 30


def aa2mat(a):
    t = np.linalg.norm(a);
    if t < 1e-8: return np.eye(3, dtype=np.float32)
    k = a / t; K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return (np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)).astype(np.float32)


def mat2aa(R):
    tr = np.clip((R[0, 0] + R[1, 1] + R[2, 2] - 1) / 2, -1, 1); ang = np.arccos(tr)
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]); n = np.linalg.norm(v)
    return (v / n * ang).astype(np.float32) if n > 1e-8 else np.zeros(3, np.float32)


Rx = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], np.float32)   # Z-up -> Y-up (-90 about X)
go = np.stack([mat2aa(Rx @ aa2mat(go[i])) for i in range(F)])
trans = trans @ Rx.T

model = smplx.create("/work/models", model_type="smplx", gender="neutral", num_betas=10,
                     use_pca=False, flat_hand_mean=True, batch_size=F)
with torch.no_grad():
    v = model(global_orient=torch.from_numpy(go), body_pose=torch.from_numpy(body),
              transl=torch.from_numpy(trans)).vertices.numpy().astype(np.float32)
faces = model.faces.astype(np.int64)
cen = v.mean(1)                                            # per-frame body center (follow-cam)
hgt = float(np.ptp(v.reshape(-1, 3)[:, 1]))               # standing height for distance
YFOV = 0.7; dist = hgt / (2 * np.tan(0.5 * YFOV)) * 1.15
r = pyrender.OffscreenRenderer(512, 512); os.makedirs("/tmp/kimo_f", exist_ok=True)
for fi in range(F):
    ctr = cen[fi]; cam = _aim([ctr[0], ctr[1], ctr[2] + dist], ctr)
    s = pyrender.Scene(bg_color=[0.08, 0.08, 0.09, 1], ambient_light=[0.5, 0.5, 0.5])
    s.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(v[fi], faces, process=False), smooth=True))
    s.add(pyrender.PerspectiveCamera(yfov=YFOV, aspectRatio=1.0), pose=cam)
    s.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=_aim([ctr[0] + 1, ctr[1] + 2, ctr[2] + 2], ctr))
    s.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.2), pose=_aim([ctr[0] - 2, ctr[1] + 1, ctr[2] + 1], ctr))
    Image.fromarray(r.render(s)[0]).save(f"/tmp/kimo_f/f_{fi:04d}.png")
r.delete()
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", "/tmp/kimo_f/f_%04d.png",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", OUT], check=True)
print("KIMODO_RENDER_OK", OUT, "frames", F, "fps", fps)
