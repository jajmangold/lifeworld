"""Face swap that PRESERVES BLINKS/eye motion. Same inswapper_128 + gfpgan-1024 as local_swap.py,
but the GFPGAN enhance mask carves soft holes over the eyes (located via insightface 5-pt kps),
so eyes keep the inswapper output (which follows the target blink) instead of GFPGAN hallucinating
them open. Cheeks/mouth/forehead still get GFPGAN sharpening.
  python local_swap_keepeyes.py <source_face.png> <target.mp4> <out.mp4>
"""
import sys, cv2, numpy as np, onnxruntime as ort, insightface
from insightface.app import FaceAnalysis

SRC, TGT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
PROV = ['CUDAExecutionProvider', 'CPUExecutionProvider']
app = FaceAnalysis(name='buffalo_l', root='/s/checkpoints', providers=PROV)
app.prepare(ctx_id=0, det_size=(640, 640))
swapper = insightface.model_zoo.get_model('/s/models/inswapper_128.onnx', providers=PROV)
gfp = ort.InferenceSession('/s/models/gfpgan-1024.onnx', providers=PROV)

src = cv2.imread(SRC); sf = app.get(src)
assert sf, "no face in source image"
src_face = max(sf, key=lambda f: f.bbox[2] - f.bbox[0])


def enhance(frame, bbox, kps):
    x1, y1, x2, y2 = [int(v) for v in bbox]; mg = int(0.25 * (x2 - x1))
    x1, y1 = max(0, x1 - mg), max(0, y1 - mg)
    x2, y2 = min(frame.shape[1], x2 + mg), min(frame.shape[0], y2 + mg)
    crop = frame[y1:y2, x1:x2]; h, w = crop.shape[:2]
    if h < 16 or w < 16:
        return frame
    inp = cv2.cvtColor(cv2.resize(crop, (512, 512)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
    inp = ((inp - 0.5) / 0.5).transpose(2, 0, 1)[None]
    out = gfp.run(None, {'input': inp})[0][0]
    out = np.clip(out * 0.5 + 0.5, 0, 1).transpose(1, 2, 0)
    out = cv2.resize(cv2.cvtColor((out * 255).astype(np.uint8), cv2.COLOR_RGB2BGR), (w, h))
    mask = np.zeros((h, w), np.float32)
    cv2.ellipse(mask, (w // 2, h // 2), (int(w * 0.43), int(h * 0.47)), 0, 0, 360, 1, -1)
    # carve soft holes over the eyes so GFPGAN doesn't reopen blinks (keep inswapper eyes)
    eye_sp = float(np.linalg.norm(kps[1] - kps[0])) if kps is not None else (x2 - x1) * 0.33
    rx = max(8, int(eye_sp * 0.55)); ry = max(6, int(eye_sp * 0.40))
    holes = np.zeros((h, w), np.float32)
    for i in (0, 1):  # left_eye, right_eye keypoints
        ex, ey = int(kps[i][0] - x1), int(kps[i][1] - y1)
        cv2.ellipse(holes, (ex, ey), (rx, ry), 0, 0, 360, 1, -1)
    holes = cv2.GaussianBlur(holes, (41, 41), 0)
    mask = np.clip(mask - holes, 0, 1)
    mask = cv2.GaussianBlur(mask, (31, 31), 0)[..., None]
    frame[y1:y2, x1:x2] = (out * mask + crop * (1 - mask)).astype(np.uint8)
    return frame


cap = cv2.VideoCapture(TGT); fps = cap.get(cv2.CAP_PROP_FPS) or 25
out = None; n = 0; miss = 0
while True:
    ok, fr = cap.read()
    if not ok:
        break
    if out is None:
        out = cv2.VideoWriter(OUT, cv2.VideoWriter_fourcc(*'mp4v'), fps, (fr.shape[1], fr.shape[0]))
    faces = app.get(fr)
    if faces:
        tf = max(faces, key=lambda f: f.bbox[2] - f.bbox[0])
        fr = swapper.get(fr, tf, src_face, paste_back=True)
        fr = enhance(fr, tf.bbox, tf.kps)
    else:
        miss += 1
    out.write(fr); n += 1
out.release(); print(f"SWAP_OK frames {n} no-face {miss}")
