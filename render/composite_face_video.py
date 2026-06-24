#!/usr/bin/env python3
"""Shoulder-anchored face composite (stable). Aligns the FLOAT head+shoulders portrait to the mesh
by a similarity fit on NOSE + L/R SHOULDER landmarks (big baseline -> stable scale, no eye-distance
twitch; anchors the head anatomically so it sits at the right size and the neck meets the body).
FLOAT's green bg is chroma-matted out. Optional depth occlusion.
  python composite_face_video.py --body b.mp4 --face f.mp4 --audio a.wav --out o.mp4
        --mesh-lmk body_lmk.npy --float-pose float_pose.json [--depth-dir d/ --body-fps 30 --face-fps 25 --smooth 9]"""
import argparse, os, json, numpy as np, cv2

ap = argparse.ArgumentParser()
ap.add_argument("--body", required=True); ap.add_argument("--face", required=True)
ap.add_argument("--audio", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--body-fps", type=float, default=30.0); ap.add_argument("--face-fps", type=float, default=25.0)
ap.add_argument("--mesh-lmk", required=True); ap.add_argument("--float-pose", required=True)
ap.add_argument("--depth-dir", default=None); ap.add_argument("--occ-delta", type=float, default=0.10)
ap.add_argument("--smooth", type=int, default=9)
a = ap.parse_args()
import glob
_nd = len(glob.glob(os.path.join(a.depth_dir, "*.npy"))) if a.depth_dir else 0
MESH = np.load(a.mesh_lmk)[:, [2, 3, 4], :].astype(np.float32)    # nose, shL, shR (mesh, stable)
FP = np.array(json.load(open(a.float_pose))["pts"], np.float32)   # (Ff,3,2) nose, shL, shR (FLOAT)
k = max(1, a.smooth | 1); pad = k // 2                            # temporal smooth FLOAT pose -> kill jitter
FP = np.stack([np.stack([np.convolve(np.pad(FP[:, p, c], pad, "edge"), np.ones(k)/k, "valid")
                         for c in range(2)], 1) for p in range(3)], 1)

faces = []
fcap = cv2.VideoCapture(a.face)
while True:
    ok, fr = fcap.read()
    if not ok: break
    faces.append(fr)
fcap.release()

def order(p):                                                     # keep nose[0]; sort shoulders[1,2] by image-x
    p = p.copy()
    if p[1, 0] > p[2, 0]: p[[1, 2]] = p[[2, 1]]
    return p

bcap = cv2.VideoCapture(a.body)
FRAMEDIR = os.path.join(os.path.dirname(a.out) or ".", "_vcompframes"); os.makedirs(FRAMEDIR, exist_ok=True)
fi = 0; prevM = None
while True:
    ok, body = bcap.read()
    if not ok: break
    H, W = body.shape[:2]
    j = min(int(round((fi / a.body_fps) * a.face_fps)), len(FP) - 1, len(faces) - 1)
    fp = order(FP[j]); mp = order(MESH[min(fi, len(MESH) - 1)])
    M, _ = cv2.estimateAffinePartial2D(fp, mp, method=cv2.LMEDS)
    M = M if M is not None else prevM
    if M is None: cv2.imwrite(f"{FRAMEDIR}/{fi:05d}.png", body); fi += 1; continue
    prevM = M
    warp = cv2.warpAffine(faces[j], M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 255, 0))
    # chroma-matte FLOAT green bg (the visible FLOAT head+neck pastes; green/off-frame drops out)
    b_, g_, r_ = warp[..., 0].astype(np.int16), warp[..., 1].astype(np.int16), warp[..., 2].astype(np.int16)
    greenness = g_ - np.maximum(b_, r_)
    m = np.clip((25 - greenness) / 25.0, 0, 1)
    if _nd:
        dep = np.load(os.path.join(a.depth_dir, f"{min(fi, _nd-1):04d}.npy")); valid = dep > 0
        sel = (m > 0.5) & valid
        if sel.any():
            hd = float(np.median(dep[sel])); m[valid & (dep < hd - a.occ_delta)] = 0.0
    m = m[..., None]
    out = (warp * m + body * (1 - m)).astype(np.uint8)
    cv2.imwrite(f"{FRAMEDIR}/{fi:05d}.png", out); fi += 1
bcap.release()
print("VCOMPOSITE_FRAMES_OK", FRAMEDIR, "frames", fi)
