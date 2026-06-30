"""Stabilize the mouth during SILENCE so MuseTalk's idle lip-jitter goes still and NATURAL.
MuseTalk animates the mouth even on pure-zero audio. Research consensus for talking heads is "no
residual lip movement" during silence — STABILISE the mouth, don't snap it to a different pose
(forcing the avatar's render-neutral looked like a weird smile). So: within each silent run we pick
MuseTalk's OWN most-relaxed (most-closed, natural) mouth and HOLD it across the run, repositioned by
the per-frame mouth landmarks so it follows head motion. Temporal ramps at the edges keep it smooth.
  python3 mouth_freeze.py <muse.mp4> <swap.mp4|-> <faces.json> <gated_16k.wav> <out.mp4>
(swap arg kept for signature compatibility; unused.)
"""
import sys, json, wave, numpy as np, cv2

muse_p, _swap, faces_p, wav_p, out_p = sys.argv[1:6]
faces = json.load(open(faces_p))

cap = cv2.VideoCapture(muse_p)
fps = cap.get(cv2.CAP_PROP_FRAME_COUNT) and cap.get(cv2.CAP_PROP_FPS) or 25.0
frames = []
while True:
    ok, f = cap.read()
    if not ok: break
    frames.append(f)
cap.release()
N = len(frames); Hh, W = frames[0].shape[:2]

# per-frame silence from the gated audio (truly zero in pauses)
w = wave.open(wav_p); sr = w.getframerate(); a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32)
if w.getnchannels() > 1: a = a[::w.getnchannels()]
spf = sr / fps
sil = np.array([1.0 if (len(a[int(i*spf):int((i+1)*spf)]) and np.max(np.abs(a[int(i*spf):int((i+1)*spf)])) < 6.0) else 0.0
                for i in range(N)], np.float32)

def mouth_info(i):
    if i >= len(faces) or not faces[i]: return None
    k = np.array(faces[i][1], np.float32)
    mc = (k[3] + k[4]) / 2.0; mw = float(np.linalg.norm(k[4] - k[3]))
    return mc, mw

def openness(i):
    mi = mouth_info(i)
    if mi is None: return 1e9
    (mx, my), mw = mi
    r = int(mw * 0.5); x0, y0 = int(mx - r), int(my - int(mw*0.35))
    crop = frames[i][max(0,y0):y0+int(mw*0.7), max(0,x0):x0+2*r]
    if crop.size == 0: return 1e9
    g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float((g < 80).sum())          # open mouth -> dark interior -> high; closed -> low

# silent runs (>= 4 frames)
runs = []; i = 0
while i < N:
    if sil[i] > 0.5:
        j = i
        while j < N and sil[j] > 0.5: j += 1
        if j - i >= 4: runs.append((i, j))
        i = j
    else: i += 1

# temporal ramp weight per frame (0..1), 1 in the core of each run, ramped ~5 frames at the edges
wgt = np.zeros(N, np.float32)
for (s, e) in runs:
    L = e - s
    for t in range(L):
        ramp = min(1.0, (t + 1) / 5.0, (L - t) / 5.0)
        wgt[s + t] = ramp

out = cv2.VideoWriter(out_p, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, Hh))
res = [f.copy() for f in frames]
for (s, e) in runs:
    ref = min(range(s, e), key=openness)          # MuseTalk's OWN most-relaxed/closed mouth in this run
    mi = mouth_info(ref)
    if mi is None: continue
    (rx, ry), rmw = mi
    pw, ph = int(rmw * 1.6), int(rmw * 1.25)       # mouth patch around the ref mouth
    px0, py0 = int(rx - pw), int(ry - ph); px1, py1 = int(rx + pw), int(ry + ph)
    px0, py0 = max(0, px0), max(0, py0); px1, py1 = min(W, px1), min(Hh, py1)
    patch = frames[ref][py0:py1, px0:px1].astype(np.float32)
    pcx, pcy = rx - px0, ry - py0                  # ref mouth-centre within patch
    for fi in range(s, e):
        info = mouth_info(fi)
        if info is None: continue
        (mx, my), mw = info
        dx0 = int(mx - pcx); dy0 = int(my - pcy)    # place patch so its mouth-centre lands on this frame's mouth
        dx1, dy1 = dx0 + patch.shape[1], dy0 + patch.shape[0]
        sx0, sy0 = max(0, -dx0), max(0, -dy0)
        cx0, cy0 = max(0, dx0), max(0, dy0); cx1, cy1 = min(W, dx1), min(Hh, dy1)
        if cx1 <= cx0 or cy1 <= cy0: continue
        sub = patch[sy0:sy0 + (cy1 - cy0), sx0:sx0 + (cx1 - cx0)]
        mask = np.zeros((Hh, W), np.float32)
        cv2.ellipse(mask, (int(mx), int(my + mw*0.12)), (int(mw*1.0), int(mw*0.78)), 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), mw*0.2) * wgt[fi]
        m = mask[cy0:cy1, cx0:cx1, None]
        res[fi][cy0:cy1, cx0:cx1] = (sub * m + res[fi][cy0:cy1, cx0:cx1].astype(np.float32) * (1 - m)).astype(np.uint8)
for f in res: out.write(f)
out.release()
print(f"MOUTH_FREEZE_OK {int((wgt>0.02).sum())}/{N} frames stabilised, {len(runs)} silent runs -> {out_p}")
