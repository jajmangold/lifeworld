#!/usr/bin/env python3
"""Feature-locked face composite (stable). Fits a similarity (scale/roll/position) from the FLOAT
face's eye+nose landmarks to the mesh's projected eye+nose landmarks and warps the FLOAT face onto
the head so eyes/nose/ears lock. Stability: FLOAT landmarks are temporally smoothed (kills the
per-frame detector jitter); placement is driven by the stable mesh landmarks. FLOAT's green bg is
chroma-matted out (no halo). Optional depth occlusion (nearer geometry blocks the face).
  python composite_face_video.py --body b.mp4 --face f.mp4 --audio a.wav --out o.mp4
        --mesh-lmk body_lmk.npy [--depth-dir d/ --body-fps 30 --face-fps 25 --smooth 7]"""
import argparse, os, numpy as np, cv2, face_alignment

ap = argparse.ArgumentParser()
ap.add_argument("--body", required=True); ap.add_argument("--face", required=True)
ap.add_argument("--audio", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--body-fps", type=float, default=30.0); ap.add_argument("--face-fps", type=float, default=25.0)
ap.add_argument("--mesh-lmk", required=True); ap.add_argument("--depth-dir", default=None)
ap.add_argument("--occ-delta", type=float, default=0.10); ap.add_argument("--smooth", type=int, default=7)
ap.add_argument("--mask-rx", type=float, default=1.5); ap.add_argument("--mask-ry", type=float, default=1.85)
a = ap.parse_args()
import glob
_nd = len(glob.glob(os.path.join(a.depth_dir, "*.npy"))) if a.depth_dir else 0
MLMK = np.load(a.mesh_lmk)                                        # (Fb,3,2) eyeL,eyeR,nose (mesh, stable)
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device="cuda")
IBUG = [36, 45, 30]

# load FLOAT frames + detect landmarks, then SMOOTH over time (remove detector jitter)
faces = []; fl = []
fcap = cv2.VideoCapture(a.face)
while True:
    ok, fr = fcap.read()
    if not ok: break
    faces.append(fr)
    lm = fa.get_landmarks(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    fl.append(lm[0][IBUG].astype(np.float32) if lm else None)
fcap.release()
# fill gaps with nearest, then moving-average smooth each of the 3 pts (x,y)
good = [p for p in fl if p is not None]
ref = np.median(np.stack(good), 0) if good else np.float32([[200, 230], [310, 230], [256, 300]])
fl = [p if p is not None else ref for p in fl]
FL = np.stack(fl)                                                # (Ff,3,2)
k = max(1, a.smooth | 1); pad = k // 2
FLs = np.stack([np.stack([np.convolve(np.pad(FL[:, p, c], pad, "edge"), np.ones(k)/k, "valid")
                          for c in range(2)], 1) for p in range(3)], 1)  # (Ff,3,2) smoothed

bcap = cv2.VideoCapture(a.body)
FRAMEDIR = os.path.join(os.path.dirname(a.out) or ".", "_vcompframes"); os.makedirs(FRAMEDIR, exist_ok=True)
fi = 0; prevM = None
while True:
    ok, body = bcap.read()
    if not ok: break
    H, W = body.shape[:2]
    j = min(int(round((fi / a.body_fps) * a.face_fps)), len(faces) - 1)
    fp = FLs[j].copy(); mp = MLMK[min(fi, len(MLMK) - 1)].astype(np.float32).copy()
    if fp[0, 0] > fp[1, 0]: fp[[0, 1]] = fp[[1, 0]]              # order eyes L->R (image) both sides
    if mp[0, 0] > mp[1, 0]: mp[[0, 1]] = mp[[1, 0]]
    M, _ = cv2.estimateAffinePartial2D(fp, mp, method=cv2.LMEDS)
    M = M if M is not None else prevM
    if M is None: cv2.imwrite(f"{FRAMEDIR}/{fi:05d}.png", body); fi += 1; continue
    prevM = M
    warp = cv2.warpAffine(faces[j], M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    eyed = float(np.linalg.norm(mp[0] - mp[1])) + 1e-6
    cx, cy = mp[:2].mean(0); cy += 0.35 * eyed
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((xx - cx) / (a.mask_rx * eyed)) ** 2 + ((yy - cy) / (a.mask_ry * eyed)) ** 2)
    m = np.clip((1.0 - d) / 0.18, 0, 1)
    # chroma-matte FLOAT's green bg out of the warped face (kills the halo)
    b_, g_, r_ = warp[..., 0].astype(np.int16), warp[..., 1].astype(np.int16), warp[..., 2].astype(np.int16)
    greenness = g_ - np.maximum(b_, r_)
    m = m * np.clip((25 - greenness) / 25.0, 0, 1)
    if _nd:
        dep = np.load(os.path.join(a.depth_dir, f"{min(fi, _nd-1):04d}.npy")); valid = dep > 0
        sel = (m > 0.5) & valid
        if sel.any():
            hd = float(np.median(dep[sel]))
            m[valid & (dep < hd - a.occ_delta)] = 0.0
    m = m[..., None]
    out = (warp * m + body * (1 - m)).astype(np.uint8)
    cv2.imwrite(f"{FRAMEDIR}/{fi:05d}.png", out); fi += 1
bcap.release()
print("VCOMPOSITE_FRAMES_OK", FRAMEDIR, "frames", fi, "facefr", len(faces))
