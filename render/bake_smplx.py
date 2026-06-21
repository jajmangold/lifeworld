#!/usr/bin/env python3
"""Bake an AMASS SMPL-X-native motion clip to per-frame vertices (CPU).

AMASS '..._stageii.npz' is already SMPL-X (separate global_orient/body_pose/
hand/jaw/eye poses + transl + betas) — so this is a straight SMPL-X forward
pass, zero retargeting. Output .npz (verts (F,V,3), faces) feeds render_smplx.py.

Runs CPU-only (no GPU container needed). Inside the sampl:dev image:
    $SAMPL_VENV/bin/python bake_smplx.py --amass clip_stageii.npz \
        --model-dir /work/models --gender female --fps 30 --max-frames 120 \
        --out output/clip_walk.npz
"""
import argparse
import numpy as np
import torch
import smplx


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--amass", required=True)
    ap.add_argument("--model-dir", default="/work/models")
    ap.add_argument("--gender", default="neutral")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main():
    a = parse_args()
    d = np.load(a.amass, allow_pickle=True)
    src_fps = float(d["mocap_framerate"]) if "mocap_framerate" in d.files else \
        (float(d["mocap_frame_rate"]) if "mocap_frame_rate" in d.files else 120.0)
    stride = max(1, round(src_fps / a.fps))

    def grab(key, width):
        if key in d.files:
            return d[key].astype(np.float32)[::stride]
        return None

    go = grab("global_orient", 3)
    bp = grab("body_pose", 63)
    F = len(bp)
    transl = grab("transl", 3)
    if transl is None:
        transl = np.zeros((F, 3), np.float32)
    transl = transl - transl[0]                      # start at origin

    def opt(key):
        v = grab(key, 0)
        return v if v is not None else None

    lh, rh = opt("left_hand_pose"), opt("right_hand_pose")
    jaw, le, re = opt("jaw_pose"), opt("leye_pose"), opt("reye_pose")

    if a.max_frames and F > a.max_frames:
        F = a.max_frames   # t() slices every component to [:F] below

    betas = np.zeros((1, 10), np.float32)
    if "betas" in d.files:
        b = d["betas"].astype(np.float32).reshape(-1)
        betas[0, :min(10, len(b))] = b[:10]

    use_pca = lh is not None and lh.shape[1] == 12   # PCA clips are 12-dim
    model = smplx.create(a.model_dir, model_type="smplx", gender=a.gender,
                         num_betas=10, use_pca=use_pca,
                         flat_hand_mean=True, batch_size=F)

    def t(x):
        return torch.from_numpy(np.ascontiguousarray(x[:F])) if x is not None else None

    kw = dict(
        global_orient=t(go), body_pose=t(bp), transl=t(transl),
        betas=torch.from_numpy(np.tile(betas, (F, 1))),
    )
    for k, v in (("left_hand_pose", lh), ("right_hand_pose", rh),
                 ("jaw_pose", jaw), ("leye_pose", le), ("reye_pose", re)):
        if v is not None:
            kw[k] = t(v)

    with torch.no_grad():
        verts = model(**kw).vertices.numpy().astype(np.float32)   # (F,V,3)
    faces = model.faces.astype(np.int64)
    np.savez_compressed(a.out, verts=verts, faces=faces, rot_x=0.0)
    print(f"[bake] {a.out}  verts={verts.shape} faces={faces.shape} "
          f"src_fps={src_fps} stride={stride} use_pca={use_pca}")


if __name__ == "__main__":
    main()
