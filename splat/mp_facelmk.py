import sys, json, cv2, numpy as np, mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision
IN, OUT = sys.argv[1], sys.argv[2]
lm = vision.FaceLandmarker.create_from_options(vision.FaceLandmarkerOptions(
    base_options=mpp.BaseOptions(model_asset_path="/face_landmarker.task"), num_faces=1))
img = cv2.imread(IN); H, W = img.shape[:2]
r = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
L = r.face_landmarks[0]
pts = {k: [L[i].x*W, L[i].y*H] for k, i in [("eyeL",33), ("eyeR",263), ("nose",1), ("mouth",13), ("chin",152)]}
json.dump({"pts": pts, "w": W, "h": H}, open(OUT, "w")); print("FACELMK_OK", pts)
