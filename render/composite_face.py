#!/usr/bin/env python3
"""Composite a FLOAT talking-face video back onto a wider reference image (so we get a zoomed-out
shot with a crisp animated face). Detects the same face crop box FLOAT uses, pastes each animated
frame there with a feathered edge, muxes the audio.
  python composite_face.py --bg wide.png --face float_face.mp4 --audio a.wav --out out.mp4 [--fps 25]"""
import argparse, subprocess, os
import numpy as np, cv2, face_alignment

ap = argparse.ArgumentParser()
ap.add_argument("--bg", required=True); ap.add_argument("--face", required=True)
ap.add_argument("--audio", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--fps", type=int, default=25)
a = ap.parse_args()

bg = cv2.imread(a.bg)                                    # BGR
H, W = bg.shape[:2]
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device="cuda")
rgb = cv2.cvtColor(bg, cv2.COLOR_BGR2RGB)
mult = 360. / H
det = cv2.resize(rgb, (0, 0), fx=mult, fy=mult, interpolation=cv2.INTER_AREA if mult < 1 else cv2.INTER_CUBIC)
bxs = fa.face_detector.detect_from_image(det)
bxs = [(x1/mult, y1/mult, x2/mult, y2/mult, s) for (x1, y1, x2, y2, s) in bxs if s > 0.95]
x1, y1, x2, y2, _ = bxs[0]
bsy, bsx = (y2-y1)/2, (x2-x1)/2
my, mx = int((y1+y2)/2), int((x1+x2)/2)
bs = int(max(bsy, bsx) * 1.6)                           # same 1.6x box as FLOAT
print(f"face box: center=({mx},{my}) bs={bs} -> region {2*bs}x{2*bs}")

# feather mask (2bs square): 1 in center, fades near the edges to hide the seam
yy, xx = np.mgrid[0:2*bs, 0:2*bs].astype(np.float32)
d = np.minimum.reduce([xx, yy, 2*bs-1-xx, 2*bs-1-yy]) / (bs * 0.45)
mask = np.clip(d, 0, 1)[..., None]                      # (2bs,2bs,1)

cap = cv2.VideoCapture(a.face)
FRAMEDIR = os.path.join(os.path.dirname(a.out) or ".", "_compframes")
os.makedirs(FRAMEDIR, exist_ok=True)
fi = 0
while True:
    ok, fr = cap.read()
    if not ok:
        break
    fr = cv2.resize(fr, (2*bs, 2*bs), interpolation=cv2.INTER_CUBIC)
    out = bg.copy()
    y0, x0 = my - bs, mx - bs                            # paste top-left in original coords
    # clip to image bounds
    sy0, sx0 = max(0, -y0), max(0, -x0)
    dy0, dx0 = max(0, y0), max(0, x0)
    dy1, dx1 = min(H, y0 + 2*bs), min(W, x0 + 2*bs)
    sy1, sx1 = sy0 + (dy1 - dy0), sx0 + (dx1 - dx0)
    m = mask[sy0:sy1, sx0:sx1]
    out[dy0:dy1, dx0:dx1] = (fr[sy0:sy1, sx0:sx1] * m + out[dy0:dy1, dx0:dx1] * (1 - m)).astype(np.uint8)
    cv2.imwrite(f"{FRAMEDIR}/{fi:05d}.png", out); fi += 1
cap.release()
print("COMPOSITE_FRAMES_OK", FRAMEDIR, "frames", fi)
