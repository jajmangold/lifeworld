"""Freeze the mouth to the neutral RENDER pose during SILENCE — kills MuseTalk's idle lip-jitter on
quiet/no-words sections (MuseTalk animates the mouth even on pure-zero audio). The pre-MuseTalk swap
render has the right neutral mouth at the SAME head pose (pixel-aligned), so we blend its mouth region
into the muse output for silent frames (temporal ramp so it closes naturally).
  python3 mouth_freeze.py <muse.mp4> <swap.mp4> <faces.json> <gated_16k.wav> <out.mp4>
"""
import sys, json, wave, numpy as np, cv2

muse_p, swap_p, faces_p, wav_p, out_p = sys.argv[1:6]
faces = json.load(open(faces_p))

# per-frame silence (audio is truly gated -> silent frames are ~zero), at the video fps
cap = cv2.VideoCapture(muse_p)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); Hh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
w = wave.open(wav_p); sr = w.getframerate(); a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32)
if w.getnchannels() > 1: a = a[::w.getnchannels()]
spf = sr / fps
sil = np.zeros(N, np.float32)
for i in range(N):
    seg = a[int(i*spf):int((i+1)*spf)]
    sil[i] = 1.0 if (len(seg) and np.max(np.abs(seg)) < 6.0) else 0.0   # ~silence
# temporal ramp: smooth the binary mask so the mouth closes/opens over ~5 frames (no pop)
k = np.array([.15,.35,.6,.85,1,.85,.6,.35,.15], np.float32); k/=k.sum()
silr = np.convolve(sil, k, mode="same")
# require a real silence core (avoid nibbling short gaps between words): only where raw silence held a bit
silr = silr * (np.convolve(sil, np.ones(5)/5, mode="same") > 0.4)
silr = np.clip(silr, 0, 1)

sc = cv2.VideoCapture(swap_p)
vw = cv2.VideoWriter(out_p, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, Hh))
frozen = 0
for i in range(N):
    okm, fm = cap.read(); oks, fs = sc.read()
    if not okm: break
    wgt = float(silr[i])
    if oks and wgt > 0.02 and i < len(faces) and faces[i]:
        kps = np.array(faces[i][1], np.float32)
        mc = (kps[3] + kps[4]) / 2.0
        mw = float(np.linalg.norm(kps[4] - kps[3]))
        ax = (int(max(18, mw*1.05)), int(max(14, mw*0.85)))   # ellipse over lips+a little chin/cheek
        cy = int(mc[1] + mw*0.12)
        mask = np.zeros((Hh, W), np.float32)
        cv2.ellipse(mask, (int(mc[0]), cy), ax, 0, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), mw*0.22) * wgt
        m3 = mask[:, :, None]
        fm = (fs.astype(np.float32)*m3 + fm.astype(np.float32)*(1-m3)).astype(np.uint8)
        frozen += 1
    vw.write(fm)
cap.release(); sc.release(); vw.release()
print(f"MOUTH_FREEZE_OK {frozen}/{N} frames mouth-neutralised -> {out_p}")
