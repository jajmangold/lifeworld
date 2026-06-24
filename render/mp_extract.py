#!/usr/bin/env python3
"""Extract ARKit-52 blendshapes from a video via MediaPipe FaceLandmarker -> arkit-format json
(same shape FlameDriver expects: {arkit_names, weights[T,52], fps}). So a FLOAT talking video can
be retargeted to FLAME exactly like a LAM result.  python mp_extract.py in.mp4 out.json"""
import sys, json, cv2, numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision

IN, OUT = sys.argv[1], sys.argv[2]
import numpy as np
opts = vision.FaceLandmarkerOptions(
    base_options=mpp.BaseOptions(model_asset_path="/face_landmarker.task"),
    output_face_blendshapes=True, output_facial_transformation_matrixes=True, num_faces=1,
    running_mode=vision.RunningMode.VIDEO)
lm = vision.FaceLandmarker.create_from_options(opts)
cap = cv2.VideoCapture(IN); fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
names, weights, head_R = None, [], []
t = 0
while True:
    ok, fr = cap.read()
    if not ok: break
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    res = lm.detect_for_video(img, int(t * 1000 / fps)); t += 1
    if res.face_blendshapes:
        bs = res.face_blendshapes[0]
        if names is None: names = [c.category_name for c in bs]
        weights.append([c.score for c in bs])
    else:
        weights.append([0.0] * (len(names) if names else 52))
    # head pose: 3x3 rotation from the facial transformation matrix (identity if no face)
    if res.facial_transformation_matrixes:
        R = np.asarray(res.facial_transformation_matrixes[0])[:3, :3]
        head_R.append(R.flatten().tolist())
    else:
        head_R.append([1, 0, 0, 0, 1, 0, 0, 0, 1])
cap.release()
json.dump({"arkit_names": names, "weights": weights, "fps": fps, "num_frames": len(weights),
           "head_R": head_R}, open(OUT, "w"))
print("MP_EXTRACT_OK", OUT, "frames", len(weights), "names", len(names or []), "head_R", len(head_R))
