"""Natural pause lips: instead of holding one frozen mouth, READ the dense lip shape per frame
(insightface 2d106) and during silence SMOOTH the lip-point trajectory (kills MuseTalk's jitter, keeps
the slow settle that interpolates the real shapes at the utterance boundaries), then WARP each pause
frame's mouth from its current lips to the smoothed targets via a piecewise-affine (Delaunay) morph.
Guard: clamp per-point displacement and ramp at the run edges so it eases in/out and never smears.
  python3 mouth_settle.py <muse.mp4> <lips.npy> <gated_16k.wav> <out.mp4>
"""
import sys, wave, numpy as np, cv2

muse_p, lips_p, wav_p, out_p = sys.argv[1:5]
lips = np.load(lips_p)                       # (N,20,2), NaN where no face
cap = cv2.VideoCapture(muse_p); fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
frames = []
while True:
    ok, f = cap.read()
    if not ok: break
    frames.append(f)
cap.release()
N = min(len(frames), len(lips)); Hh, W = frames[0].shape[:2]

w = wave.open(wav_p); sr = w.getframerate(); a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32)
if w.getnchannels() > 1: a = a[::w.getnchannels()]
spf = sr/fps
sil = np.array([1.0 if (len(a[int(i*spf):int((i+1)*spf)]) and np.max(np.abs(a[int(i*spf):int((i+1)*spf)])) < 6.0) else 0.0
                for i in range(N)], np.float32)
runs = []; i = 0
while i < N:
    if sil[i] > 0.5:
        j = i
        while j < N and sil[j] > 0.5: j += 1
        if j-i >= 5: runs.append((i, j))
        i = j
    else: i += 1

def warp_to(img, src_pts, dst_pts, mw, weight):
    """piecewise-affine warp of the mouth region: move src_pts -> dst_pts (lips + fixed box anchors)."""
    cx, cy = src_pts.mean(0)
    box = np.array([[cx-1.6*mw, cy-1.4*mw], [cx, cy-1.4*mw], [cx+1.6*mw, cy-1.4*mw],
                    [cx-1.6*mw, cy+1.4*mw], [cx, cy+1.4*mw], [cx+1.6*mw, cy+1.4*mw],
                    [cx-1.6*mw, cy], [cx+1.6*mw, cy]], np.float32)            # static surround anchors
    S = np.vstack([src_pts, box]).astype(np.float32)
    D = np.vstack([dst_pts, box]).astype(np.float32)
    rect = (0, 0, W, Hh)
    sub = cv2.Subdiv2D(rect)
    for p in D: sub.insert((float(np.clip(p[0],0,W-1)), float(np.clip(p[1],0,Hh-1))))
    out = img.copy()
    idx = {(round(p[0],1), round(p[1],1)): k for k, p in enumerate(D)}
    for t in sub.getTriangleList():
        pts = [(t[0],t[1]), (t[2],t[3]), (t[4],t[5])]
        tri = []
        for (x,y) in pts:
            k = idx.get((round(x,1), round(y,1)))
            if k is None: break
            tri.append(k)
        if len(tri) != 3: continue
        ds = D[tri].astype(np.float32); ss = S[tri].astype(np.float32)
        r = cv2.boundingRect(ds)
        if r[2] <= 0 or r[3] <= 0: continue
        dsl = ds - [r[0], r[1]]
        M = cv2.getAffineTransform(ss, dsl)
        patch = cv2.warpAffine(img, M, (r[2], r[3]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        mask = np.zeros((r[3], r[2]), np.float32); cv2.fillConvexPoly(mask, dsl.astype(np.int32), 1.0)
        mask = cv2.GaussianBlur(mask, (0,0), 1.5)[:, :, None] * weight
        y0, x0 = r[1], r[0]
        out[y0:y0+r[3], x0:x0+r[2]] = (patch*mask + out[y0:y0+r[3], x0:x0+r[2]]*(1-mask)).astype(np.uint8)
    return out

res = [f.copy() for f in frames[:N]]
for (s, e) in runs:
    seg = lips[s:e].copy()                      # (L,20,2)
    if np.isnan(seg).any():                     # fill gaps by nearest valid
        for p in range(seg.shape[1]):
            col = seg[:, p, :]
            good = ~np.isnan(col[:, 0])
            if good.sum() >= 2:
                for d in (0, 1):
                    col[:, d] = np.interp(np.arange(len(col)), np.where(good)[0], col[good, d])
            seg[:, p, :] = col
    L = len(seg)
    # temporal lowpass per lip point (gaussian over time) -> de-jittered settle trajectory
    sig = max(2.0, L/4.0)
    k = np.exp(-0.5*(np.arange(-int(3*sig), int(3*sig)+1)/sig)**2); k /= k.sum()
    smooth = seg.copy()
    for p in range(seg.shape[1]):
        for d in (0, 1):
            smooth[:, p, d] = np.convolve(np.pad(seg[:, p, d], int(3*sig), mode="edge"), k, "valid")
    for t in range(L):
        fi = s + t
        cur = seg[t]; tgt = smooth[t]
        if np.isnan(cur).any(): continue
        mw = float(np.linalg.norm(cur.max(0) - cur.min(0))) or 30.0
        disp = tgt - cur
        disp = np.clip(disp, -0.5*mw, 0.5*mw)   # guard: clamp warp magnitude (no smear)
        ramp = min(1.0, (t+1)/6.0, (L-t)/6.0); ramp = ramp*ramp*(3-2*ramp)
        res[fi] = warp_to(frames[fi], cur, cur + disp, mw, ramp)

vw = cv2.VideoWriter(out_p, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, Hh))
for f in res: vw.write(f)
vw.release()
print(f"MOUTH_SETTLE_OK {len(runs)} runs warped -> {out_p}")
