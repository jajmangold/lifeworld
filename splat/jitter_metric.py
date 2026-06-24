#!/usr/bin/env python3
"""Objective 'jumpiness' metric for a talking-head video — the yardstick the splat work must beat.
Optical flow (Farneback) in the head ROI -> per-frame mean motion; JITTER = mean frame-to-frame
ACCELERATION of the ROI's mean-flow vector (rapid back-and-forth), which isolates twitch from
smooth legit motion. Also reports mean flow + high-frequency energy.
  python jitter_metric.py video.mp4 [roi: x0 y0 x1 y1 as fractions]"""
import sys, numpy as np, cv2
V = sys.argv[1]
rx0, ry0, rx1, ry1 = (map(float, sys.argv[2:6]) if len(sys.argv) >= 6 else (0.30, 0.06, 0.70, 0.50))
cap = cv2.VideoCapture(V); ok, prev = cap.read()
if not ok: print("no frames"); sys.exit(1)
H, W = prev.shape[:2]
x0, y0, x1, y1 = int(rx0*W), int(ry0*H), int(rx1*W), int(ry1*H)
g0 = cv2.cvtColor(prev[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
vecs = []; mags = []
while True:
    ok, fr = cap.read()
    if not ok: break
    g1 = cv2.cvtColor(fr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    fl = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 3, 21, 3, 5, 1.2, 0)
    vecs.append([fl[..., 0].mean(), fl[..., 1].mean()])      # mean ROI translation this step
    mags.append(float(np.sqrt(fl[..., 0]**2 + fl[..., 1]**2).mean()))
    g0 = g1
cap.release()
v = np.array(vecs); m = np.array(mags)
accel = np.linalg.norm(np.diff(v, axis=0), axis=1)           # |Δ velocity| = jitter/twitch
k = 5; sm = np.convolve(m, np.ones(k)/k, "same"); hf = np.abs(m - sm)
print(f"{V.split('/')[-1]:28s}  JITTER(accel)={accel.mean():.3f}  meanflow={m.mean():.3f}  hf_energy={hf.mean():.3f}  frames={len(m)+1}")
