"""Stabilize the mouth during SILENCE so MuseTalk's idle lip-jitter goes still and NATURAL.
MuseTalk animates the mouth even on pure-zero audio. Research consensus for talking heads is "no
residual lip movement" during silence — STABILISE the mouth, don't repose it. So within each silent
run we pick MuseTalk's OWN most-relaxed (most-closed, STABLE) mouth and HOLD it, warped to each frame
by a similarity transform from the 5 face landmarks (rotation+scale+translation — translation-only made
odd lip shapes when the head tilted). Temporal ramps at the edges keep it smooth.
  python3 mouth_freeze.py <muse.mp4> <swap.mp4|-> <faces.json> <gated_16k.wav> <out.mp4>
(swap arg unused; kept for signature compatibility.)
"""
import sys, json, wave, numpy as np, cv2

muse_p, _swap, faces_p, wav_p, out_p = sys.argv[1:6]
faces = json.load(open(faces_p))

cap = cv2.VideoCapture(muse_p)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
frames = []
while True:
    ok, f = cap.read()
    if not ok: break
    frames.append(f)
cap.release()
N = len(frames); Hh, W = frames[0].shape[:2]

w = wave.open(wav_p); sr = w.getframerate(); a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32)
if w.getnchannels() > 1: a = a[::w.getnchannels()]
spf = sr / fps
sil = np.array([1.0 if (len(a[int(i*spf):int((i+1)*spf)]) and np.max(np.abs(a[int(i*spf):int((i+1)*spf)])) < 6.0) else 0.0
                for i in range(N)], np.float32)

def kps(i):
    return np.array(faces[i][1], np.float32) if (i < len(faces) and faces[i]) else None
def mouth_cw(i):
    k = kps(i)
    if k is None: return None
    return (k[3] + k[4]) / 2.0, float(np.linalg.norm(k[4] - k[3]))
def openness(i):
    mi = mouth_cw(i)
    if mi is None: return 1e9
    (mx, my), mw = mi
    x0, y0 = int(mx - mw*0.5), int(my - mw*0.35)
    c = frames[i][max(0,y0):y0+int(mw*0.7), max(0,x0):x0+int(mw)]
    if c.size == 0: return 1e9
    return float((cv2.cvtColor(c, cv2.COLOR_BGR2GRAY) < 80).sum())   # open -> dark interior -> high

# silent runs (>= 5 frames)
runs = []; i = 0
while i < N:
    if sil[i] > 0.5:
        j = i
        while j < N and sil[j] > 0.5: j += 1
        if j - i >= 5: runs.append((i, j))
        i = j
    else: i += 1

# smooth ramp (smoothstep, ~7 frames) so the held mouth eases in/out without a sudden odd blend
RAMP = 7.0
wgt = np.zeros(N, np.float32)
for (s, e) in runs:
    L = e - s
    for t in range(L):
        u = min(1.0, (t + 1) / RAMP, (L - t) / RAMP)
        wgt[s + t] = u * u * (3 - 2 * u)

out = [f.copy() for f in frames]
for (s, e) in runs:
    # ref = MuseTalk's own most-closed mouth, chosen from the STABLE middle of the run (avoid transition frames)
    lo, hi = s + min(2, (e - s)//3), e - min(2, (e - s)//3)
    cand = [c for c in range(lo, hi) if kps(c) is not None] or [c for c in range(s, e) if kps(c) is not None]
    if not cand: continue
    ref = min(cand, key=openness)
    kref = kps(ref)
    for fi in range(s, e):
        kcur = kps(fi)
        mi = mouth_cw(fi)
        if kcur is None or mi is None: continue
        M, _ = cv2.estimateAffinePartial2D(kref, kcur, method=cv2.LMEDS)   # similarity: rot+scale+trans
        if M is None: continue
        warp = cv2.warpAffine(frames[ref], M, (W, Hh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        (mx, my), mw = mi
        mask = np.zeros((Hh, W), np.float32)
        cv2.ellipse(mask, (int(mx), int(my + mw*0.12)), (int(mw*1.0), int(mw*0.80)), 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), mw*0.22) * wgt[fi]
        m3 = mask[:, :, None]
        out[fi] = (warp.astype(np.float32) * m3 + out[fi].astype(np.float32) * (1 - m3)).astype(np.uint8)

vw = cv2.VideoWriter(out_p, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, Hh))
for f in out: vw.write(f)
vw.release()
print(f"MOUTH_FREEZE_OK {int((wgt>0.02).sum())}/{N} frames stabilised (affine), {len(runs)} runs -> {out_p}")
