#!/usr/bin/env python3
"""Head-pose estimator that works on BOTH the SMPL-X mesh render AND real faces (uses
face_alignment 68 landmarks + solvePnP) -> a consistent yardstick for head-track calibration.
  python fa_pose.py in.mp4 out_pose.json   ->  {"euler":[[pitch,yaw,roll]deg,...]}"""
import sys, json, cv2, numpy as np, face_alignment

IN, OUT = sys.argv[1], sys.argv[2]
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device="cuda")
# canonical 3D face model (mm) for 6 stable landmarks: nose tip, chin, L/R eye outer, L/R mouth
M = np.array([[0,0,0],[0,-63,-12],[-43,32,-26],[43,32,-26],[-28,-28,-24],[28,-28,-24]], np.float64)
IDX = [30, 8, 36, 45, 48, 54]                                   # 68-landmark indices

cap = cv2.VideoCapture(IN); W = int(cap.get(3)); H = int(cap.get(4))
K = np.array([[W, 0, W/2], [0, W, H/2], [0, 0, 1]], np.float64)  # approx intrinsics
eul = []
prev = [0.0, 0.0, 0.0]
while True:
    ok, fr = cap.read()
    if not ok: break
    lms = fa.get_landmarks(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    if not lms:
        eul.append(prev); continue
    p2 = lms[0][IDX].astype(np.float64)
    ok2, rvec, _ = cv2.solvePnP(M, p2, K, None, flags=cv2.SOLVEPNP_ITERATIVE)
    R, _ = cv2.Rodrigues(rvec)
    pitch = np.degrees(np.arctan2(-R[2, 1], R[2, 2]))
    yaw   = np.degrees(np.arctan2(R[2, 0], np.hypot(R[2, 1], R[2, 2])))
    roll  = np.degrees(np.arctan2(-R[1, 0], R[0, 0]))
    prev = [float(pitch), float(yaw), float(roll)]; eul.append(prev)
cap.release()
json.dump({"euler": eul, "frames": len(eul)}, open(OUT, "w"))
print("FA_POSE_OK", OUT, "frames", len(eul))
