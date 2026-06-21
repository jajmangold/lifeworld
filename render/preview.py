#!/usr/bin/env python3
"""Fast lip-sync preview: cached ARKit json -> bake + head close-up -> contact sheet.

One container, no A2F re-run, single character. Outputs a tiled PNG of the key viseme
frames (peak jaw / max close / max pucker / rest / spaced) so you can judge lip-sync
from ONE image in seconds. Optional --mp4 for a full head-shot clip.

  python3 preview.py --arkit speech.arkit.json --texture .../f_alb_eyefix.png \
      --uv .../smplx_uv_2023.npz --out preview.png [--mp4 head.mp4 --wav speech.wav]
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
from PIL import Image

sys.path.insert(0, "/work")          # sampl: face.talk
sys.path.insert(0, "/lw/render")     # face_drive, render_smplx
from face_drive import drive
from face.talk import eyelid_upper_indices, apply_blink, lip_region, apply_lips
from render_smplx import build_mesh, frame_camera, _aim
from mouth_parts import build_mouth


def bake(arkit, model, betas_row, F, fps):
    rest = np.zeros((1, 21, 3), np.float32)
    rest[0, 15] = [0, 0, -1.0]; rest[0, 16] = [0, 0, 1.0]
    face = drive(arkit, F, fps)
    out = model(betas=torch.from_numpy(np.tile(betas_row, (F, 1))),
                global_orient=torch.zeros((F, 3)),
                body_pose=torch.from_numpy(np.tile(rest.reshape(1, 63), (F, 1))),
                jaw_pose=torch.from_numpy(face["jaw"]),
                leye_pose=torch.from_numpy(face["leye"]),
                reye_pose=torch.from_numpy(face["reye"]))
    v = out.vertices.detach().numpy().astype(np.float32)
    li, ri = eyelid_upper_indices(model, betas_row)
    apply_blink(v, li, ri, face["blink_l"], face["blink_r"])
    apply_lips(v, lip_region(model, betas_row), face["mouth_close"], face["mouth_pucker"])
    return v, face


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arkit", required=True)
    ap.add_argument("--model-dir", default="/work/models")
    ap.add_argument("--gender", default="female")
    ap.add_argument("--shape", type=float, default=0.0)
    ap.add_argument("--betas", default="")
    ap.add_argument("--uv", default="/work/assets/smplx_uv_2023.npz")
    ap.add_argument("--texture", default="/work/assets/smplx_texture_f_alb_eyefix.png")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--res", type=int, default=320)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mp4", default=None)
    ap.add_argument("--wav", default=None)
    a = ap.parse_args()

    arkit = json.load(open(a.arkit))
    dur = len(arkit["weights"]) / float(arkit["fps"])
    F = max(2, round(dur * a.fps))
    model = smplx.create(a.model_dir, model_type="smplx", gender=a.gender, num_betas=10,
                         use_pca=False, flat_hand_mean=True, batch_size=F)
    betas_row = np.full(10, float(a.shape), np.float32)
    if a.betas:
        bv = np.array([float(x) for x in a.betas.split(",")], np.float32)
        betas_row[:len(bv)] = bv[:10]
    verts, face = bake(arkit, model, betas_row, F, a.fps)
    faces = model.faces.astype(np.int64)
    # teeth + tongue + dark interior (SMPL-X has none -> open mouth is a black void)
    mv, mf, mcol = build_mouth(verts, lip_region(model, betas_row)["idx"], face["jaw"][:, 0])

    uv = np.load(a.uv)["uv_coordinates"] if os.path.exists(a.uv) else None
    tex = Image.open(a.texture).convert("RGB") if os.path.exists(a.texture) else None

    yfov, cam_pose, center = frame_camera(verts, "head", 1.0)
    cam = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1.0)
    lights = [(pyrender.DirectionalLight(color=np.ones(3), intensity=3.2), _aim([1, 1, 2], center)),
              (pyrender.DirectionalLight(color=np.ones(3), intensity=1.4), _aim([-2, 1, 1], center))]
    r = pyrender.OffscreenRenderer(a.res, a.res)

    def render(fi):
        s = pyrender.Scene(bg_color=[0.05, 0.05, 0.07, 1.0], ambient_light=[0.4, 0.4, 0.42])
        s.add(build_mesh(verts[fi], faces, uv, tex))
        if face["jaw"][fi, 0] > 0.05:        # only show teeth when the mouth is open
            mt = trimesh.Trimesh(mv[fi], mf, vertex_colors=mcol, process=False)
            s.add(pyrender.Mesh.from_trimesh(mt, smooth=False))
        s.add(cam, pose=cam_pose)
        for lt, p in lights:
            s.add(lt, pose=p)
        return r.render(s)[0]

    # contact sheet of the key viseme frames
    jaw = face["jaw"][:, 0]; mc = face["mouth_close"]; mp = face["mouth_pucker"]
    picks = {"rest": int(jaw.argmin()), "jawOpen": int(jaw.argmax()),
             "mouthClose": int(mc.argmax()), "pucker": int(mp.argmax()),
             "t1/3": F // 3, "t2/3": 2 * F // 3}
    tiles = []
    from PIL import ImageDraw
    for label, fi in picks.items():
        im = Image.fromarray(render(fi))
        d = ImageDraw.Draw(im); d.text((6, 6), f"{label} f{fi}", fill=(255, 255, 0))
        tiles.append(im)
    W = a.res; sheet = Image.new("RGB", (W * 3, W * 2))
    for i, im in enumerate(tiles):
        sheet.paste(im, ((i % 3) * W, (i // 3) * W))
    sheet.save(a.out)
    print(f"PREVIEW_OK frames={F} sheet={a.out} picks={ {k:v for k,v in picks.items()} }")

    if a.mp4 and a.wav:
        os.makedirs("/tmp/pf", exist_ok=True)
        for fi in range(F):
            Image.fromarray(render(fi)).save(f"/tmp/pf/frame_{fi:04d}.png")
        import subprocess
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(a.fps),
                        "-i", "/tmp/pf/frame_%04d.png", "-i", a.wav, "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-crf", "18", "-shortest", a.mp4])
        print(f"PREVIEW_MP4 {a.mp4}")
    r.delete()


if __name__ == "__main__":
    main()
