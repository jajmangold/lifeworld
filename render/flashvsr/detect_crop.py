import sys, json
import cv2, numpy as np

inp  = sys.argv[1]                       # full path to source video
stem = sys.argv[2]                       # output name stem
cap = cv2.VideoCapture(inp)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); fps = cap.get(cv2.CAP_PROP_FPS)
casc = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ROBUST head box: take the LARGEST face per sampled frame (the real head), then aggregate by MEDIAN
# center + median size. Unioning every raw detection (old behaviour) let a single spurious background
# "face" blow the crop up to the whole frame -> full-frame FlashVSR -> CUDA OOM.
boxes = []   # per-frame largest (x,y,w,h)
i = 0
while True:
    ok, fr = cap.read()
    if not ok: break
    if i % 10 == 0:
        gray = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        dets = casc.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        if len(dets):
            x, y, w, h = max(dets, key=lambda d: d[2] * d[3])   # largest by area
            boxes.append((x, y, w, h))
    i += 1
cap.release()
print(f"[det] {W}x{H} {N} frames @ {fps:.2f}fps; per-frame faces={len(boxes)}", flush=True)
if not boxes:
    print("[det] NO FACES"); sys.exit(1)

arr = np.array(boxes, dtype=np.float32)
cx = float(np.median(arr[:, 0] + arr[:, 2] / 2))
cy = float(np.median(arr[:, 1] + arr[:, 3] / 2))
msz = float(np.median(np.maximum(arr[:, 2], arr[:, 3])))
side = msz * 2.2
import os as _os
side = min(side, H * 0.92, float(_os.environ.get("MAX_BOX", "464")))   # cap so 2x region fits 12GB
# (464 box -> 928 region fits; 512 -> 1024 OOMs FlashVSR on a 3060). MCU framing makes the head big.
half = side / 2
bx0 = max(0, int(cx - half));        by0 = max(0, int(cy - half * 1.05))
bx1 = min(W, int(cx + half));        by1 = min(H, int(cy + half * 1.15))
r16 = lambda v: (v // 16) * 16
bx0, by0 = r16(bx0), r16(by0)
bw, bh = r16(bx1 - bx0), r16(by1 - by0)
print(f"[det] head box x={bx0} y={by0} w={bw} h={bh} (median size {msz:.0f})", flush=True)
json.dump({"x": bx0, "y": by0, "w": bw, "h": bh, "W": W, "H": H, "fps": fps},
          open(f"/workspace/myinput/headbox_{stem}.json", "w"))

cap = cv2.VideoCapture(inp)
out = cv2.VideoWriter(f"/workspace/myinput/head_crop_{stem}.mp4",
                      cv2.VideoWriter_fourcc(*"mp4v"), fps, (bw, bh))
while True:
    ok, fr = cap.read()
    if not ok: break
    out.write(fr[by0:by0+bh, bx0:bx0+bw])
cap.release(); out.release()
print(f"[det] wrote head_crop_{stem}.mp4", flush=True)
