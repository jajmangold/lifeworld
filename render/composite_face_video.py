#!/usr/bin/env python3
"""Composite a FLOAT talking-face video onto a moving body video.
Two modes:
  box   (default): track the face box, paste a scaled oval (loose).
  align (--mesh-lmk x_lmk.npy): FEATURE-LOCK — fit a similarity (scale/roll/translate) from the
        FLOAT face's eye+nose landmarks to the mesh's projected eye+nose landmarks, warp the FLOAT
        face onto the mesh head so eyes/nose/ears line up (no slip). Optional depth occlusion.
  python composite_face_video.py --body b.mp4 --face f.mp4 --audio a.wav --out o.mp4
        [--mesh-lmk body_lmk.npy --depth-dir d/ --body-fps 30 --face-fps 25 --mask-rx 1.7 --mask-ry 2.1]"""
import argparse, os, numpy as np, cv2, face_alignment

ap = argparse.ArgumentParser()
ap.add_argument("--body", required=True); ap.add_argument("--face", required=True)
ap.add_argument("--audio", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--body-fps", type=float, default=30.0); ap.add_argument("--face-fps", type=float, default=25.0)
ap.add_argument("--mesh-lmk", default=None); ap.add_argument("--depth-dir", default=None)
ap.add_argument("--occ-delta", type=float, default=0.08)
ap.add_argument("--mask-rx", type=float, default=1.7); ap.add_argument("--mask-ry", type=float, default=2.1)  # x eye-dist
a = ap.parse_args()
import glob
_nd = len(glob.glob(os.path.join(a.depth_dir, "*.npy"))) if a.depth_dir else 0
MLMK = np.load(a.mesh_lmk) if a.mesh_lmk else None                # (F,3,2): eyeL,eyeR,nose (mesh)

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device="cuda")
IBUG = [36, 45, 30]                                               # eye-outer L, eye-outer R, nose tip

def fpts(img):                                                    # FLOAT face landmarks (eyeL,eyeR,nose)
    lm = fa.get_landmarks(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return None if not lm else lm[0][IBUG].astype(np.float32)

faces = []
fcap = cv2.VideoCapture(a.face)
while True:
    ok, fr = fcap.read()
    if not ok: break
    faces.append(fr)
fcap.release()

bcap = cv2.VideoCapture(a.body)
FRAMEDIR = os.path.join(os.path.dirname(a.out) or ".", "_vcompframes"); os.makedirs(FRAMEDIR, exist_ok=True)
fi = 0; prevM = None
while True:
    ok, body = bcap.read()
    if not ok: break
    H, W = body.shape[:2]
    j = min(int(round((fi / a.body_fps) * a.face_fps)), len(faces) - 1)
    flo = faces[j]
    fp = fpts(flo)
    if fp is None or MLMK is None:                               # can't align -> passthrough this frame
        cv2.imwrite(f"{FRAMEDIR}/{fi:05d}.png", body); fi += 1; continue
    mp = MLMK[min(fi, len(MLMK) - 1)].astype(np.float32).copy()
    # order eyes left->right in image on BOTH so correspondence matches sides
    if fp[0, 0] > fp[1, 0]: fp[[0, 1]] = fp[[1, 0]]
    if mp[0, 0] > mp[1, 0]: mp[[0, 1]] = mp[[1, 0]]
    M, _ = cv2.estimateAffinePartial2D(fp, mp, method=cv2.LMEDS)  # similarity FLOAT->mesh
    if M is None: M = prevM
    if M is None: cv2.imwrite(f"{FRAMEDIR}/{fi:05d}.png", body); fi += 1; continue
    prevM = M
    warp = cv2.warpAffine(flo, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # elliptical mask in body coords, sized by mesh eye distance, centered on the face
    eyed = float(np.linalg.norm(mp[0] - mp[1])) + 1e-6
    cxy = mp[:2].mean(0); cx, cy = cxy[0], cxy[1] + 0.35 * eyed   # nudge to face center
    rx, ry = a.mask_rx * eyed, a.mask_ry * eyed
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2)
    m = np.clip((1.0 - d) / 0.18, 0, 1)
    if _nd:                                                       # depth occlusion: nearer geom (hands) blocks face
        dep = np.load(os.path.join(a.depth_dir, f"{min(fi, _nd-1):04d}.npy"))
        valid = dep > 0
        if valid.any():
            hd = float(np.median(dep[(m > 0.5) & valid])) if ((m > 0.5) & valid).any() else float(np.median(dep[valid]))
            m[valid & (dep < hd - a.occ_delta)] = 0.0
    m = m[..., None]
    out = (warp * m + body * (1 - m)).astype(np.uint8)
    cv2.imwrite(f"{FRAMEDIR}/{fi:05d}.png", out); fi += 1
bcap.release()
print("VCOMPOSITE_FRAMES_OK", FRAMEDIR, "frames", fi, "facefr", len(faces))
