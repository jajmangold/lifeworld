#!/usr/bin/env python3
"""SMPL-X bridge + render for MDM motion. Reads local rotations (mdm_extract_local.py output),
forwards SMPL-X (global_orient+body_pose+transl), renders a full-body clip + muxes nothing (no
audio). HumanML3D is Y-up facing +Z; SMPL-X neutral faces -Z, so we yaw the root 180.
  python mdm_render.py <local.npz> <out.mp4>"""
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import sys, subprocess
import numpy as np
import torch
import smplx
import pyrender
import trimesh
from PIL import Image
sys.path.insert(0, "/work"); sys.path.insert(0, "/lw/render")
from render_smplx import build_mesh, frame_camera, _aim

NPZ, OUT = sys.argv[1], sys.argv[2]
FPS = 20                                                   # HumanML3D native
d = np.load(NPZ); mats = d["mats"].astype(np.float32); root = d["root_pos"].astype(np.float32)
F = len(mats)


def mat2aa(R):                                             # [N,3,3] -> [N,3] axis-angle
    tr = np.clip((R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] - 1) / 2, -1, 1)
    ang = np.arccos(tr)
    ax = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], 1)
    n = np.linalg.norm(ax, axis=1, keepdims=True); n[n < 1e-8] = 1.0
    return (ax / n) * ang[:, None]


aa = mat2aa(mats.reshape(-1, 3, 3)).reshape(F, 22, 3)
go = aa[:, 0].copy(); body = aa[:, 1:22].reshape(F, 63)
# face the camera: rotate root 180 about vertical (compose yaw into global_orient)
import numpy as _np
def compose_yaw(go, deg):
    th = _np.radians(deg); Ry = _np.array([[_np.cos(th), 0, _np.sin(th)], [0, 1, 0], [-_np.sin(th), 0, _np.cos(th)]], _np.float32)
    out = _np.zeros_like(go)
    for i in range(len(go)):
        a = go[i]; t = _np.linalg.norm(a)
        if t < 1e-8:
            R = _np.eye(3)
        else:
            k = a / t; K = _np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            R = _np.eye(3) + _np.sin(t) * K + (1 - _np.cos(t)) * (K @ K)
        R2 = Ry @ R
        tr = _np.clip((R2[0, 0] + R2[1, 1] + R2[2, 2] - 1) / 2, -1, 1); ang = _np.arccos(tr)
        v = _np.array([R2[2, 1] - R2[1, 2], R2[0, 2] - R2[2, 0], R2[1, 0] - R2[0, 1]]); nv = _np.linalg.norm(v)
        out[i] = (v / nv * ang) if nv > 1e-8 else _np.zeros(3)
    return out.astype(_np.float32)
go = compose_yaw(go, 180)
transl = root.copy(); transl[:, 0] *= -1; transl[:, 2] *= -1   # match the 180 yaw

model = smplx.create("/work/models", model_type="smplx", gender="neutral", num_betas=10,
                     use_pca=False, flat_hand_mean=True, batch_size=F)
with torch.no_grad():
    v = model(global_orient=torch.from_numpy(go), body_pose=torch.from_numpy(body.astype(np.float32)),
              transl=torch.from_numpy(transl)).vertices.numpy().astype(np.float32)
faces = model.faces.astype(np.int64)
# fixed camera framing the whole trajectory
yfov, cam, ctr = frame_camera(v.reshape(-1, 3)[None], "full", 1.0) if False else (None, None, None)
allv = v.reshape(-1, 3); mn = allv.min(0); mx = allv.max(0)
ctr = (mn + mx) / 2.0; size = float((mx - mn).max())
YFOV = 0.7; dist = size / (2 * np.tan(0.5 * YFOV)) * 1.25
cam = _aim([ctr[0], ctr[1], ctr[2] + dist], ctr)
r = pyrender.OffscreenRenderer(512, 512); os.makedirs("/tmp/mdm_f", exist_ok=True)
for fi in range(F):
    s = pyrender.Scene(bg_color=[0.08, 0.08, 0.09, 1], ambient_light=[0.5, 0.5, 0.5])
    s.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(v[fi], faces, process=False), smooth=True))
    s.add(pyrender.PerspectiveCamera(yfov=YFOV, aspectRatio=1.0), pose=cam)
    s.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=_aim([1, 2, 2], ctr))
    s.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.2), pose=_aim([-2, 1, 1], ctr))
    Image.fromarray(r.render(s)[0]).save(f"/tmp/mdm_f/f_{fi:04d}.png")
r.delete()
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", "/tmp/mdm_f/f_%04d.png",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", OUT], check=True)
print("MDM_RENDER_OK", OUT, "frames", F)
