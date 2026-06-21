#!/usr/bin/env python3
"""Bake an idle talking SMPL-X clip: standing rest pose + audio-driven face.

Uses sampl's current face.talk.audio_to_face (LAM Audio2Expression if the :8202
service is up, else a deterministic amplitude-jaw fallback) -> per-frame jaw /
expression / gaze, folded into the SMPL-X forward pass. Optional mesh-space
blinks. CPU-only. Output feeds render_smplx.py (render with --rot-x 0 --framing face).

Run inside sampl:dev with PYTHONPATH=/work (sampl repo, so face/engine resolve):
    $SAMPL_VENV/bin/python /lw/render/bake_talk.py --audio /work/output/speech.wav \
        --frames 110 --gender female --model-dir /work/models --out /work/output/clip_talk.npz
"""
import argparse
import json
import os
import numpy as np
import torch
import smplx

from face.talk import (audio_to_face, arkit_to_face, eyelid_upper_indices,
                       apply_blink, lip_region, apply_lips)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--arkit", default=None,
                    help="LAM ARKit-52 json (run lam on host first) -> learned visemes; "
                         "falls back to amplitude jaw if absent")
    ap.add_argument("--frames", type=int, default=110)
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--gender", default="female")
    ap.add_argument("--shape", type=float, default=0.0)
    ap.add_argument("--model-dir", default="/work/models")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main():
    a = parse_args()
    F = a.frames
    model = smplx.create(a.model_dir, model_type="smplx", gender=a.gender,
                         num_betas=10, use_pca=False, flat_hand_mean=True, batch_size=F)

    # relaxed A-pose: zero body except shoulders rotated down (T-pose -> arms down).
    # SMPL-X body_pose = 21 joints x 3 (axis-angle). Shoulders are joints 15/16.
    rest = np.zeros((1, 21, 3), np.float32)
    rest[0, 15] = [0.0, 0.0, -1.0]   # left shoulder down
    rest[0, 16] = [0.0, 0.0, 1.0]    # right shoulder down
    body_pose = torch.from_numpy(np.tile(rest.reshape(1, 63), (F, 1)))
    betas = torch.full((F, 10), float(a.shape))

    if a.arkit and os.path.exists(a.arkit):
        # gentle jaw (~13deg max) so it doesn't 'unhinge'; A2F jawOpen is clean/strong
        face = arkit_to_face(json.load(open(a.arkit)), F, a.fps, jaw_max=0.17, jaw_gain=1.0)
        src = "arkit"
    else:
        face = audio_to_face(a.audio, F, a.fps)                    # amplitude fallback
        src = "amplitude-fallback"
    kw = dict(
        global_orient=torch.zeros((F, 3)),
        body_pose=body_pose,
        jaw_pose=torch.from_numpy(face["jaw"]),
        expression=torch.from_numpy(face["expression"]),
        leye_pose=torch.from_numpy(face["leye"]),
        reye_pose=torch.from_numpy(face["reye"]),
    )
    with torch.no_grad():
        verts = model(betas=betas, **kw).vertices.numpy().astype(np.float32)
    faces = model.faces.astype(np.int64)

    # mesh-space blinks (SMPL-X has no eyelid joint): drop upper-lid verts per frame
    try:
        l_idx, r_idx = eyelid_upper_indices(model, np.full(10, float(a.shape), np.float32))
        apply_blink(verts, l_idx, r_idx, face["blink_l"], face["blink_r"])
        blinks = "on"
    except Exception as e:
        blinks = f"skipped ({e})"

    # mesh-space mouth shaping (bilabial close + pucker) for real visemes, not just jaw
    lips = "off"
    if "mouth_close" in face:
        try:
            lipr = lip_region(model, np.full(10, float(a.shape), np.float32))
            apply_lips(verts, lipr, face["mouth_close"], face["mouth_pucker"])
            lips = "on"
        except Exception as e:
            lips = f"skipped ({e})"

    np.savez_compressed(a.out, verts=verts, faces=faces, rot_x=0.0)
    print(f"[bake_talk] {a.out}  verts={verts.shape} frames={F} face={src} blinks={blinks} lips={lips}")


if __name__ == "__main__":
    main()
