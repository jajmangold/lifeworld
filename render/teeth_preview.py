#!/usr/bin/env python3
"""Teeth tuner: render the open mouth (front + 3/4 view), cropped to the bite, using the
current render/mouth_params.json. Edit that JSON, re-run render/teeth.sh, look again (~15s).
"""
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import argparse
import json
import sys
import numpy as np
import torch
import smplx
import pyrender
import trimesh
from PIL import Image, ImageDraw

sys.path.insert(0, "/work")
sys.path.insert(0, "/lw/render")
from face_drive import drive
from face.talk import eyelid_upper_indices, apply_blink, lip_region, apply_lips
from render_smplx import frame_camera, _aim, build_mesh
from mouth_parts import build_mouth, load_params


def bake(arkit, model, betas, F, fps, yaw):
    rest = np.zeros((1, 21, 3), np.float32)
    rest[0, 15] = [0, 0, -1.0]; rest[0, 16] = [0, 0, 1.0]
    face = drive(arkit, F, fps)
    out = model(betas=torch.from_numpy(np.tile(betas, (F, 1))),
                global_orient=torch.from_numpy(np.tile([[0.0, yaw, 0.0]], (F, 1)).astype(np.float32)),
                body_pose=torch.from_numpy(np.tile(rest.reshape(1, 63), (F, 1))),
                jaw_pose=torch.from_numpy(face["jaw"]),
                leye_pose=torch.from_numpy(face["leye"]), reye_pose=torch.from_numpy(face["reye"]))
    v = out.vertices.detach().numpy().astype(np.float32)
    li, ri = eyelid_upper_indices(model, betas[0])
    apply_blink(v, li, ri, face["blink_l"], face["blink_r"])
    apply_lips(v, lip_region(model, betas[0]), face["mouth_close"], face["mouth_pucker"])
    return v, face


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arkit", default="/a2f/test.arkit.json")
    ap.add_argument("--gender", default="female")
    ap.add_argument("--uv", default="/work/assets/smplx_uv_2023.npz")
    ap.add_argument("--texture", default="/work/assets/smplx_texture_f_alb_eyefix.png")
    ap.add_argument("--res", type=int, default=512)
    ap.add_argument("--out", default="/a2f/teeth.png")
    a = ap.parse_args()
    fps = 24
    arkit = json.load(open(a.arkit))
    F = max(2, round(len(arkit["weights"]) / float(arkit["fps"]) * fps))
    model = smplx.create("/work/models", model_type="smplx", gender=a.gender, num_betas=10,
                         use_pca=False, flat_hand_mean=True, batch_size=F)
    betas = np.zeros((1, 10), np.float32)
    uv = np.load(a.uv)["uv_coordinates"]; tex = Image.open(a.texture).convert("RGB")
    gate = load_params()["gate"]
    r = pyrender.OffscreenRenderer(a.res, a.res)
    tiles = []
    for yaw_deg, label in [(0, "front"), (16, "3/4")]:
        yaw = np.radians(yaw_deg)
        v, face = bake(arkit, model, betas, F, fps, yaw)
        mv, mf, mcol = build_mouth(v, lip_region(model, betas[0])["idx"], face["jaw"][:, 0], yaw_rad=yaw, model=model)
        fi = int(face["jaw"][:, 0].argmax())                        # most-open frame
        yfov, cam_pose, center = frame_camera(v, "head", 1.0)
        s = pyrender.Scene(bg_color=[0.05, 0.05, 0.07, 1.0], ambient_light=[0.4, 0.4, 0.42])
        s.add(build_mesh(v[fi], model.faces.astype(np.int64), uv, tex))
        if face["jaw"][fi, 0] > gate:
            mt = trimesh.Trimesh(mv[fi], mf, vertex_colors=mcol, process=False)
            s.add(pyrender.Mesh.from_trimesh(mt, smooth=False))
        s.add(pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1.0), pose=cam_pose)
        s.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.2), pose=_aim([1, 1, 2], center))
        s.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.4), pose=_aim([-2, 1, 1], center))
        im = Image.fromarray(r.render(s)[0])
        W, H = im.size                                             # crop to the mouth
        crop = im.crop((int(W * 0.20), int(H * 0.52), int(W * 0.80), int(H * 0.92))).resize((420, 400))
        ImageDraw.Draw(crop).text((6, 6), f"{label} f{fi}", fill=(255, 255, 0))
        tiles.append(crop)
    sheet = Image.new("RGB", (420 * 2, 400))
    sheet.paste(tiles[0], (0, 0)); sheet.paste(tiles[1], (420, 0))
    sheet.save(a.out)
    print(f"TEETH_OK -> {a.out} (gate={gate})")
    r.delete()


if __name__ == "__main__":
    main()
