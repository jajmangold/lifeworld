"""Extract per-frame dense lip landmarks (insightface 2d106, indices 52-71 = outer+inner lip) from a
video -> npy of shape (N,20,2). Runs in swap-server (has insightface buffalo_l with landmark_2d_106).
NaN rows where no face. Used by mouth_settle.py to smooth+warp the lips during pauses.
  python3 lip_landmarks.py <video.mp4> <out.npy>
"""
import sys, cv2, numpy as np
from insightface.app import FaceAnalysis

LIP = list(range(52, 72))   # insightface 2d106 lip points (outer + inner contour)
app = FaceAnalysis(name="buffalo_l"); app.prepare(ctx_id=0, det_size=(640, 640))
cap = cv2.VideoCapture(sys.argv[1]); out = []
while True:
    ok, fr = cap.read()
    if not ok: break
    f = app.get(fr)
    if f:
        fa = max(f, key=lambda d: d.bbox[2] - d.bbox[0])
        out.append(fa.landmark_2d_106[LIP].astype(np.float32))
    else:
        out.append(np.full((len(LIP), 2), np.nan, np.float32))
cap.release()
np.save(sys.argv[2], np.stack(out))
print(f"LIP_OK {len(out)} frames -> {sys.argv[2]}")
