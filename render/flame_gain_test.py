#!/usr/bin/env python3
"""Render the peak-expression frame of theo_0 at several FLAME expression gains, side by side,
to pick a gain that's expressive without distorting."""
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import json, sys
import numpy as np
import torch
import smplx
import pyrender
import trimesh
from PIL import Image, ImageDraw
sys.path.insert(0, "/work"); sys.path.insert(0, "/lw/render")
from face_flame import FlameDriver
from face.talk import lip_region
from render_smplx import build_mesh, frame_camera, _aim
from mouth_parts import build_mouth

ark = json.load(open("/work/output/scene/theo_0.arkit.json"))
F = round(len(ark["weights"]) / ark["fps"] * 24)
TEX = Image.open("/work/assets/smplx_texture_m_alb_eyefix.png").convert("RGB")
uv = np.load("/work/assets/smplx_uv_2023.npz")["uv_coordinates"]
betas = np.zeros((1, 10), np.float32)
rest = np.zeros((1, 63), np.float32); rest[0, 15 * 3 + 2] = -1.0; rest[0, 16 * 3 + 2] = 1.0
model = smplx.create("/work/models", model_type="smplx", gender="male", num_betas=10,
                     use_pca=False, flat_hand_mean=True, num_expression_coeffs=100, batch_size=F)
faces = model.faces.astype(np.int64); lip = lip_region(model, betas[0])["idx"]
# find peak frame from full-gain energy
e1, jaw, le, re = FlameDriver("/work/tools/mp2flame/mappings").drive(ark, F, 24, gain=1.0)
pk = int(np.linalg.norm(e1, axis=1).argmax())
cols = []
for gain in [1.0, 0.55, 0.4, 0.0]:
    exp = e1 * gain
    out = model(betas=torch.from_numpy(np.tile(betas, (F, 1))), global_orient=torch.zeros((F, 3)),
                body_pose=torch.from_numpy(np.tile(rest, (F, 1))),
                expression=torch.from_numpy(exp.astype(np.float32)), jaw_pose=torch.from_numpy(jaw),
                leye_pose=torch.from_numpy(le), reye_pose=torch.from_numpy(re))
    v = out.vertices.detach().numpy().astype(np.float32)
    mv, mf, mcol = build_mouth(v, lip, jaw[:, 0], model=model)
    yfov, cam, ctr = frame_camera(v, "head", 1.0)
    r = pyrender.OffscreenRenderer(420, 480)
    s = pyrender.Scene(bg_color=[0.08, 0.08, 0.09, 1], ambient_light=[0.5, 0.5, 0.5])
    s.add(build_mesh(v[pk], faces, uv, TEX))
    if jaw[pk, 0] > 0.05:
        s.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(mv[pk], mf, vertex_colors=mcol, process=False), smooth=False))
    s.add(pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=420 / 480), pose=cam)
    s.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.2), pose=_aim([1, 1, 2], ctr))
    s.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.5), pose=_aim([-2, 1, 1], ctr))
    im = Image.fromarray(r.render(s)[0]); r.delete()
    ImageDraw.Draw(im).text((8, 8), f"gain {gain}", fill=(255, 255, 0))
    cols.append(im)
W = sum(c.width for c in cols); strip = Image.new("RGB", (W, cols[0].height))
x = 0
for c in cols:
    strip.paste(c, (x, 0)); x += c.width
strip.save("/work/output/scene/flame_gain.png"); print("GAIN_TEST_OK frame", pk)
