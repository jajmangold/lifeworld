#!/usr/bin/env python3
"""Bake a multi-character talking SCENE to one (P,F,V,3) SMPL-X clip.

Reads a JSON spec: characters (gender/betas/position/facing) + beats (who speaks,
their TTS wav + LAM arkit). Builds a shared timeline: every character is present the
whole time in a standing A-pose at their spot; a character's face lip-syncs only
during the beats they speak (neutral + idle gaze otherwise). Output feeds
render_smplx.py (multi-person, per-body textures). CPU; run in sampl:dev with
PYTHONPATH=/work (sampl) so face.talk resolves.

  $SAMPL_VENV/bin/python /lw/render/bake_scene.py --config scene.json --out clip.npz
"""
import argparse
import json
import numpy as np
import torch
import smplx

from face.talk import (_gaze_channels, eyelid_upper_indices,
                       apply_blink, lip_region, apply_lips)
from face_drive import drive          # single coherent face driver
from mouth_parts import build_mouth


def apose():
    r = np.zeros((1, 21, 3), np.float32)
    r[0, 15] = [0.0, 0.0, -1.0]      # shoulders down out of T-pose
    r[0, 16] = [0.0, 0.0, 1.0]
    return r.reshape(1, 63)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--model-dir", default="/work/models")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    fps = cfg.get("fps", 24)
    chars = cfg["characters"]
    beats = cfg["beats"]

    # per-beat frame counts from each arkit clip's duration
    beat_arkit, beat_F = [], []
    for b in beats:
        ak = json.load(open(b["arkit"]))
        T = len(ak["weights"]); src = float(ak.get("fps", 30.0))
        Fb = max(1, round((T / src) * fps))
        beat_arkit.append(ak); beat_F.append(Fb)
    total_F = sum(beat_F)
    print(f"[bake_scene] {len(chars)} chars, {len(beats)} beats, {total_F} frames @ {fps}fps")

    # per-frame active speaker + character positions, for head look-at
    positions = [np.asarray(c.get("pos", [0.0, 0.0]), np.float32) for c in chars]
    spk = np.zeros(total_F, np.int64); _o = 0
    for bi, b in enumerate(beats):
        spk[_o:_o + beat_F[bi]] = b["speaker"]; _o += beat_F[bi]
    tline = np.arange(total_F, dtype=np.float32) / fps

    def _ma(v, k=9):
        return np.convolve(np.pad(v, k // 2, mode="edge"), np.ones(k) / k, "valid")[:len(v)]

    body0 = apose()
    all_verts = []; all_mouth = []; all_gate = []; mfaces = mcolors = None
    for ci, ch in enumerate(chars):
        jaw = np.zeros((total_F, 3), np.float32)
        leye, reye = _gaze_channels(total_F, seed=ci)      # ONE continuous gaze (never overwritten)
        mc = np.zeros(total_F, np.float32); mp = np.zeros(total_F, np.float32)
        bl = np.zeros(total_F, np.float32); br = np.zeros(total_F, np.float32)
        off = 0
        for bi, b in enumerate(beats):
            Fb = beat_F[bi]
            if b["speaker"] == ci:                          # overlay only mouth/jaw/blink
                f = drive(beat_arkit[bi], Fb, fps)
                jaw[off:off + Fb] = f["jaw"]
                mc[off:off + Fb] = f["mouth_close"]
                mp[off:off + Fb] = f["mouth_pucker"]
                bl[off:off + Fb] = f["blink_l"]
                br[off:off + Fb] = f["blink_r"]
            off += Fb

        g = ch.get("gender", "neutral")
        model = smplx.create(a.model_dir, model_type="smplx", gender=g, num_betas=10,
                             use_pca=False, flat_hand_mean=True, batch_size=total_F)
        betas = np.zeros((1, 10), np.float32)
        bv = np.asarray(ch.get("betas", []), np.float32)
        betas[0, :min(10, len(bv))] = bv[:10]
        yaw = np.radians(ch.get("yaw_deg", 0.0))
        x, z = ch.get("pos", [0.0, 0.0])
        # aliveness: turn head toward the active speaker (listeners -> speaker; the speaker
        # -> the others' midpoint), plus subtle breathing + weight sway.
        others = [k for k in range(len(chars)) if k != ci]
        center_o = np.mean([positions[k] for k in others], axis=0) if others else positions[ci]
        tgt = np.where((spk == ci)[:, None], center_o[None, :],
                       np.stack([positions[s] for s in spk]))
        ang = np.arctan2(tgt[:, 0] - x, tgt[:, 1] - z)      # world angle to target
        hy = (ang - yaw + np.pi) % (2 * np.pi) - np.pi
        hy = _ma(np.clip(hy, -0.7, 0.7))                    # smooth so it doesn't snap on cuts
        bp = np.tile(body0, (total_F, 1)).copy()
        bp[:, 11 * 3 + 1] += 0.45 * hy                      # neck yaw toward speaker
        bp[:, 14 * 3 + 1] += 0.55 * hy                      # head yaw toward speaker
        bp[:, 5 * 3 + 0] += 0.012 * np.sin(2 * np.pi * 0.22 * tline)   # breathing
        go = np.tile([[0.0, yaw, 0.0]], (total_F, 1)).astype(np.float32)
        go[:, 2] += 0.015 * np.sin(2 * np.pi * 0.13 * tline + ci)      # subtle weight sway
        kw = dict(
            betas=torch.from_numpy(np.tile(betas, (total_F, 1))),
            global_orient=torch.from_numpy(go),
            body_pose=torch.from_numpy(bp.astype(np.float32)),
            transl=torch.from_numpy(np.tile([[x, 0.0, z]], (total_F, 1)).astype(np.float32)),
            jaw_pose=torch.from_numpy(jaw),             # no `expression` (inert on SMPL-X)
            leye_pose=torch.from_numpy(leye), reye_pose=torch.from_numpy(reye),
        )
        with torch.no_grad():
            v = model(**kw).vertices.numpy().astype(np.float32)
        # mesh-space blinks + mouth shaping (real visemes, not just jaw)
        try:
            li, ri = eyelid_upper_indices(model, betas[0])
            apply_blink(v, li, ri, bl, br)
            apply_lips(v, lip_region(model, betas[0]), mc, mp)
        except Exception as e:
            print("  face-mesh apply skipped:", e)
        mvp, mfaces, mcolors = build_mouth(v, lip_region(model, betas[0])["idx"], jaw[:, 0],
                                           yaw_rad=yaw)
        all_mouth.append(mvp); all_gate.append(jaw[:, 0])
        all_verts.append(v)
        print(f"  {ch['name']}: gender={g} pos=({x},{z}) yaw={ch.get('yaw_deg',0)}")

    verts = np.stack(all_verts, 0)                          # (P, F, V, 3)
    faces = smplx.create(a.model_dir, model_type="smplx", gender="neutral").faces.astype(np.int64)
    np.savez_compressed(a.out, verts=verts, faces=faces, rot_x=0.0,
                        beat_frames=np.array(beat_F),
                        mouth_verts=np.stack(all_mouth, 0), mouth_faces=mfaces,
                        mouth_colors=mcolors, mouth_gate=np.stack(all_gate, 0))
    print(f"[bake_scene] wrote {a.out} verts={verts.shape}")


if __name__ == "__main__":
    main()
