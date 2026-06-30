"""Resident face-swap server — loads insightface(buffalo_l) + inswapper_128 + gfpgan-1024 ONCE, then
polls a job queue. Eliminates the ~25s model-load paid per `docker run` of local_swap*. Same
eye-preserving logic as local_swap_keepeyes.py (carves GFPGAN eye holes so blinks survive).

Job: write /o/swap_jobs/<name>.json = {"src": "/o/face.png", "video": "/o/in.mp4", "out": "/o/out.mp4"}
  (optional "keepeyes": true|false, default true). Writes <name>.done / <name>.err. Output is video-only
  (mp4v) — mux audio downstream (MuseTalk does, or ffmpeg). Mounts: /s=bot/swap, /o=bot/output.
"""
import os, sys, time, json, glob, cv2, numpy as np, onnxruntime as ort, insightface
from insightface.app import FaceAnalysis

PROV = ['CUDAExecutionProvider', 'CPUExecutionProvider']
app = FaceAnalysis(name='buffalo_l', root='/s/checkpoints', providers=PROV)
app.prepare(ctx_id=0, det_size=(640, 640))
swapper = insightface.model_zoo.get_model('/s/models/inswapper_128.onnx', providers=PROV)
gfp = ort.InferenceSession('/s/models/gfpgan-1024.onnx', providers=PROV)


def enhance(frame, bbox, kps, keepeyes=True, strength=1.0):
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
    # partial-strength restore: blend GFPGAN toward the original crop so natural pores/skin survive
    # (full strength = waxy/plastic). strength<1 keeps (1-strength) of the real skin.
    if strength < 1.0:
        out = (out.astype(np.float32) * strength + crop.astype(np.float32) * (1.0 - strength)).astype(np.uint8)
    mask = np.zeros((h, w), np.float32)
    cv2.ellipse(mask, (w // 2, h // 2), (int(w * 0.43), int(h * 0.47)), 0, 0, 360, 1, -1)
    if keepeyes and kps is not None:
        eye_sp = float(np.linalg.norm(kps[1] - kps[0]))
        rx = max(8, int(eye_sp * 0.55)); ry = max(6, int(eye_sp * 0.40))
        holes = np.zeros((h, w), np.float32)
        for i in (0, 1):
            ex, ey = int(kps[i][0] - x1), int(kps[i][1] - y1)
            cv2.ellipse(holes, (ex, ey), (rx, ry), 0, 0, 360, 1, -1)
        holes = cv2.GaussianBlur(holes, (41, 41), 0)
        mask = np.clip(mask - holes, 0, 1)
    mask = cv2.GaussianBlur(mask, (31, 31), 0)[..., None]
    frame[y1:y2, x1:x2] = (out * mask + crop * (1 - mask)).astype(np.uint8)
    return frame


def enhance_known(frame, bbox, kps, keepeyes=True, strength=1.0):
    # GFPGAN restore using a KNOWN bbox/kps (no detection). bbox=[x1,y1,x2,y2], kps=2x2 eye points.
    return enhance(frame, bbox, np.asarray(kps, np.float32) if kps is not None else None, keepeyes=keepeyes, strength=strength)


def swap_video(src_path, tgt_path, out_path, keepeyes=True, do_enhance=True, save_faces=None):
    src = cv2.imread(src_path); sf = app.get(src)
    assert sf, f"no face in source {src_path}"
    src_face = max(sf, key=lambda f: f.bbox[2] - f.bbox[0])
    cap = cv2.VideoCapture(tgt_path); fps = cap.get(cv2.CAP_PROP_FPS) or 25
    out = None; n = 0; miss = 0; faces_log = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if out is None:
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (fr.shape[1], fr.shape[0]))
        faces = app.get(fr)
        if faces:
            tf = max(faces, key=lambda f: f.bbox[2] - f.bbox[0])
            fr = swapper.get(fr, tf, src_face, paste_back=True)
            if do_enhance:
                fr = enhance(fr, tf.bbox, tf.kps, keepeyes=keepeyes)
            faces_log.append([[float(v) for v in tf.bbox], tf.kps.tolist()])
        else:
            miss += 1; faces_log.append(None)
        out.write(fr); n += 1
    out.release()
    if save_faces:
        json.dump(faces_log, open(save_faces, "w"))
    return n, miss


def restore_video(tgt_path, out_path, faces_path, keepeyes=True, strength=1.0):
    # Final GFPGAN restore on a finished (post-MuseTalk) video. If faces_path is given, reuse the swap's
    # detected faces (no re-detection). Otherwise (e.g. baked-texture path, no swap) detect per frame.
    faces = json.load(open(faces_path)) if (faces_path and os.path.exists(faces_path)) else None
    cap = cv2.VideoCapture(tgt_path); fps = cap.get(cv2.CAP_PROP_FPS) or 25
    out = None; n = 0; miss = 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if out is None:
            out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (fr.shape[1], fr.shape[0]))
        if faces is not None:
            f = faces[n] if n < len(faces) else None
            if f is not None:
                bbox, kps = f
                fr = enhance_known(fr, bbox, kps, keepeyes=keepeyes, strength=strength)
            else:
                miss += 1
        else:
            dets = app.get(fr)
            if dets:
                tf = max(dets, key=lambda d: d.bbox[2] - d.bbox[0])
                fr = enhance(fr, tf.bbox, tf.kps, keepeyes=keepeyes, strength=strength)
            else:
                miss += 1
        out.write(fr); n += 1
    out.release()
    return n, miss


JOBS = os.environ.get("SWAP_JOBS", "/o/swap_jobs")
os.makedirs(JOBS, exist_ok=True)
print("SWAP_SERVER_READY", flush=True)
while True:
    for jf in sorted(glob.glob(os.path.join(JOBS, "*.json"))):
        name = os.path.basename(jf)[:-5]
        try:
            spec = json.load(open(jf)); os.remove(jf)
            t0 = time.time()
            if spec.get("mode") == "restore":
                n, miss = restore_video(spec["video"], spec["out"], spec.get("faces"), spec.get("keepeyes", True), spec.get("strength", 1.0))
                tag = "RESTORE"
            else:
                n, miss = swap_video(spec["src"], spec["video"], spec["out"], spec.get("keepeyes", True),
                                     spec.get("enhance", True), spec.get("save_faces"))
                tag = "SWAP"
            dt = round(time.time() - t0, 1)
            open(os.path.join(JOBS, name + ".done"), "w").write(f"ok {dt}s frames {n} no-face {miss} -> {spec['out']}")
            print(f"{tag} JOB DONE {name} {dt}s frames {n} no-face {miss}", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            open(os.path.join(JOBS, name + ".err"), "w").write(str(e))
    time.sleep(1)
