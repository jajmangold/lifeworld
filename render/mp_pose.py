#!/usr/bin/env python3
"""MediaPipe Pose -> nose + L/R shoulder per frame (image px) for shoulder-based face alignment.
  python mp_pose.py in.mp4 out.json  -> {"pts":[[ [nx,ny],[lx,ly],[rx,ry] ],...], "fps":}"""
import sys, json, cv2, numpy as np, mediapipe as mp
from mediapipe.tasks import python as mpp
from mediapipe.tasks.python import vision
IN, OUT = sys.argv[1], sys.argv[2]
opts = vision.PoseLandmarkerOptions(base_options=mpp.BaseOptions(model_asset_path="/pose_landmarker.task"),
                                    running_mode=vision.RunningMode.VIDEO, num_poses=1)
lm = vision.PoseLandmarker.create_from_options(opts)
cap = cv2.VideoCapture(IN); fps = cap.get(cv2.CAP_PROP_FPS) or 25.0; W=int(cap.get(3)); H=int(cap.get(4))
pts=[]; prev=[[W/2,H*0.3],[W*0.35,H*0.6],[W*0.65,H*0.6]]; t=0
while True:
    ok,fr=cap.read()
    if not ok: break
    img=mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(fr,cv2.COLOR_BGR2RGB))
    r=lm.detect_for_video(img,int(t*1000/fps)); t+=1
    if r.pose_landmarks:
        L=r.pose_landmarks[0]
        prev=[[L[0].x*W,L[0].y*H],[L[11].x*W,L[11].y*H],[L[12].x*W,L[12].y*H]]  # nose, Lsh, Rsh
    pts.append(prev)
cap.release()
json.dump({"pts":pts,"fps":fps,"frames":len(pts)},open(OUT,"w"))
print("MP_POSE_OK",OUT,"frames",len(pts))
