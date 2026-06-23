#!/usr/bin/env python3
"""Composite a FLOAT talking-face video onto a MOVING body video (e.g. Wan-VACE gesturing pastor),
tracking the head per-frame. Run FLOAT on the body's OWN first-frame face first so identity matches.
Time-aligns the two by fps. Falls back to the previous box when a frame's face isn't detected.
  python composite_face_video.py --body vace.mp4 --face float.mp4 --audio a.wav --out out.mp4
                                 --body-fps 16 --face-fps 25"""
import argparse, subprocess, os
import numpy as np, cv2, face_alignment

ap = argparse.ArgumentParser()
ap.add_argument("--body", required=True); ap.add_argument("--face", required=True)
ap.add_argument("--audio", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--body-fps", type=float, default=16.0); ap.add_argument("--face-fps", type=float, default=25.0)
a = ap.parse_args()

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device="cuda")

def facebox(img):                                           # -> (mx,my,bs) or None, FLOAT's 1.6x box
    H = img.shape[0]; mult = 360. / H
    det = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (0, 0), fx=mult, fy=mult,
                     interpolation=cv2.INTER_AREA if mult < 1 else cv2.INTER_CUBIC)
    bx = fa.face_detector.detect_from_image(det)
    bx = [(x1/mult, y1/mult, x2/mult, y2/mult, s) for (x1, y1, x2, y2, s) in bx if s > 0.9]
    if not bx: return None
    x1, y1, x2, y2, _ = max(bx, key=lambda b: b[4])
    return (int((x1+x2)/2), int((y1+y2)/2), int(max((y2-y1)/2, (x2-x1)/2) * 1.6))

# load all FLOAT face frames
fcap = cv2.VideoCapture(a.face); faces = []
while True:
    ok, fr = fcap.read()
    if not ok: break
    faces.append(fr)
fcap.release()

bcap = cv2.VideoCapture(a.body)
FRAMEDIR = os.path.join(os.path.dirname(a.out) or ".", "_vcompframes")
os.makedirs(FRAMEDIR, exist_ok=True)
prev_box = None; fi = 0
while True:
    ok, body = bcap.read()
    if not ok: break
    H, W = body.shape[:2]
    box = facebox(body) or prev_box
    if box is None:                                         # no face yet -> passthrough
        cv2.imwrite(f"{FRAMEDIR}/{fi:05d}.png", body); fi += 1; continue
    prev_box = box; mx, my, bs = box
    j = min(int(round((fi / a.body_fps) * a.face_fps)), len(faces) - 1)   # time-align
    fr = cv2.resize(faces[j], (2*bs, 2*bs), interpolation=cv2.INTER_CUBIC)
    yy, xx = np.mgrid[0:2*bs, 0:2*bs].astype(np.float32)
    m = np.clip(np.minimum.reduce([xx, yy, 2*bs-1-xx, 2*bs-1-yy]) / (bs * 0.4), 0, 1)[..., None]
    out = body.copy(); y0, x0 = my - bs, mx - bs
    sy0, sx0 = max(0, -y0), max(0, -x0); dy0, dx0 = max(0, y0), max(0, x0)
    dy1, dx1 = min(H, y0 + 2*bs), min(W, x0 + 2*bs); sy1, sx1 = sy0 + (dy1 - dy0), sx0 + (dx1 - dx0)
    mm = m[sy0:sy1, sx0:sx1]
    out[dy0:dy1, dx0:dx1] = (fr[sy0:sy1, sx0:sx1] * mm + out[dy0:dy1, dx0:dx1] * (1 - mm)).astype(np.uint8)
    cv2.imwrite(f"{FRAMEDIR}/{fi:05d}.png", out); fi += 1
bcap.release()
print("VCOMPOSITE_FRAMES_OK", FRAMEDIR, "frames", fi, "facefr", len(faces))
