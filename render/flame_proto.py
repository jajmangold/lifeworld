#!/usr/bin/env python3
"""Prototype: LAM ARKit -> SMPL-X face, two ways, side by side.
 LEFT  = current (jaw + blink only, via face_drive.drive)
 RIGHT = NEW (full FLAME expression+jaw+eye via face_flame.FlameDriver / MP_2_FLAME)
Renders a head close-up clip with audio so the brows/smiles/squints gain is visible.
"""
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import json
import subprocess
import sys
import numpy as np
import torch
import smplx
import pyrender
import trimesh
from PIL import Image, ImageDraw

sys.path.insert(0, "/work"); sys.path.insert(0, "/lw/render")
from face_drive import drive
from face_flame import FlameDriver
from face.talk import eyelid_upper_indices, apply_blink, lip_region
from render_smplx import build_mesh, frame_camera, _aim
from mouth_parts import build_mouth

ARKIT = "/work/output/scene/theo_0.arkit.json"   # currently LAM
WAV = "/work/output/scene/theo_0.wav"
TEX = "/work/assets/smplx_texture_m_alb_eyefix.png"
UV = "/work/assets/smplx_uv_2023.npz"
MAP = "/work/tools/mp2flame/mappings"
FPS = 24
arkit = json.load(open(ARKIT))
F = max(2, round(len(arkit["weights"]) / float(arkit["fps"]) * FPS))
betas = np.zeros((1, 10), np.float32)
rest = np.zeros((1, 21, 3), np.float32); rest[0, 15] = [0, 0, -1.0]; rest[0, 16] = [0, 0, 1.0]
body = torch.from_numpy(np.tile(rest.reshape(1, 63), (F, 1)))
b10 = torch.from_numpy(np.tile(betas, (F, 1)))
model = smplx.create("/work/models", model_type="smplx", gender="male", num_betas=10,
                     use_pca=False, flat_hand_mean=True, num_expression_coeffs=100, batch_size=F)
faces = model.faces.astype(np.int64)
uv = np.load(UV)["uv_coordinates"]; tex = Image.open(TEX).convert("RGB")
lip = lip_region(model, betas[0])["idx"]


def bake_jaw():
    f = drive(arkit, F, FPS)
    out = model(betas=b10, global_orient=torch.zeros((F, 3)), body_pose=body,
                jaw_pose=torch.from_numpy(f["jaw"]), leye_pose=torch.from_numpy(f["leye"]),
                reye_pose=torch.from_numpy(f["reye"]))
    v = out.vertices.detach().numpy().astype(np.float32)
    li, ri = eyelid_upper_indices(model, betas[0]); apply_blink(v, li, ri, f["blink_l"], f["blink_r"])
    return v, f["jaw"][:, 0]


def bake_flame():
    exp, jaw, le, re = FlameDriver(MAP).drive(arkit, F, FPS)
    out = model(betas=b10, global_orient=torch.zeros((F, 3)), body_pose=body,
                expression=torch.from_numpy(exp), jaw_pose=torch.from_numpy(jaw),
                leye_pose=torch.from_numpy(le), reye_pose=torch.from_numpy(re))
    return out.vertices.detach().numpy().astype(np.float32), jaw[:, 0]


def render(v, jawx, tag):
    mv, mf, mcol = build_mouth(v, lip, jawx, model=model)
    yfov, cam, ctr = frame_camera(v, "head", 1.0)
    r = pyrender.OffscreenRenderer(540, 540)
    os.makedirs(f"/tmp/{tag}", exist_ok=True)
    for fi in range(F):
        s = pyrender.Scene(bg_color=[0.08, 0.08, 0.09, 1], ambient_light=[0.5, 0.5, 0.5])
        s.add(build_mesh(v[fi], faces, uv, tex))
        if jawx[fi] > 0.05:
            mt = trimesh.Trimesh(mv[fi], mf, vertex_colors=mcol, process=False)
            s.add(pyrender.Mesh.from_trimesh(mt, smooth=False))
        s.add(pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1.0), pose=cam)
        s.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.2), pose=_aim([1, 1, 2], ctr))
        s.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.5), pose=_aim([-2, 1, 1], ctr))
        im = Image.fromarray(r.render(s)[0]); ImageDraw.Draw(im).text((8, 8), tag, fill=(255, 255, 0))
        im.save(f"/tmp/{tag}/f_{fi:04d}.png")
    r.delete()


vj, jj = bake_jaw(); render(vj, jj, "jaw-only")
vf, jf = bake_flame(); render(vf, jf, "FLAME-expr")
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", "/tmp/jaw-only/f_%04d.png",
                "-framerate", str(FPS), "-i", "/tmp/FLAME-expr/f_%04d.png", "-i", WAV,
                "-filter_complex", "[0][1]hstack[v]", "-map", "[v]", "-map", "2:a",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-shortest",
                "/work/output/scene/flame_ab.mp4"], check=True)
print("FLAME_AB_OK")
