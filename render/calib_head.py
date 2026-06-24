#!/usr/bin/env python3
"""Fast head-only render driven by a head_R sequence (same neck/head mapping as combine), for
closed-loop head-pose calibration: render -> MediaPipe-measure -> compare to source.
  python calib_head.py --arkit head_R.json --out mesh.mp4 [--signs 1,1,1 --gain 1 --fps 25]"""
import os; os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import sys, argparse, subprocess, json, numpy as np, torch, smplx, pyrender, trimesh
from PIL import Image
sys.path.insert(0, "/lw/render"); from render_smplx import _aim, build_mesh

ap = argparse.ArgumentParser()
ap.add_argument("--arkit", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--signs", default="1,1,1"); ap.add_argument("--gain", type=float, default=1.0)
ap.add_argument("--fps", type=int, default=25)
ap.add_argument("--tex", default=None); ap.add_argument("--uv", default=None)   # textured -> MediaPipe can detect the face
a = ap.parse_args()
_uv = np.load(a.uv)["uv_coordinates"] if a.uv else None
_tex = Image.open(a.tex).convert("RGB") if a.tex else None
sx, sy, sz = [float(x) for x in a.signs.split(",")]
AKJ = json.load(open(a.arkit)); HR = np.asarray(AKJ["head_R"], np.float32).reshape(-1, 3, 3); F = len(HR)


def mat2aa(R):
    tr = np.clip((R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] - 1) / 2, -1, 1); ang = np.arccos(tr)
    v = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], 1)
    n = np.linalg.norm(v, axis=1, keepdims=True); n[n < 1e-8] = 1
    return (v / n) * ang[:, None]


pitch = sx * a.gain * np.arctan2(-HR[:, 2, 1], HR[:, 2, 2])
yaw   = sy * a.gain * np.arctan2(HR[:, 2, 0], np.hypot(HR[:, 2, 1], HR[:, 2, 2]))
roll  = sz * a.gain * np.arctan2(-HR[:, 1, 0], HR[:, 0, 0])
def _Rx(t): o=np.zeros((F,3,3),np.float32);c,s=np.cos(t),np.sin(t);o[:,0,0]=1;o[:,1,1]=c;o[:,1,2]=-s;o[:,2,1]=s;o[:,2,2]=c;return o
def _Ry(t): o=np.zeros((F,3,3),np.float32);c,s=np.cos(t),np.sin(t);o[:,1,1]=1;o[:,0,0]=c;o[:,0,2]=s;o[:,2,0]=-s;o[:,2,2]=c;return o
def _Rz(t): o=np.zeros((F,3,3),np.float32);c,s=np.cos(t),np.sin(t);o[:,2,2]=1;o[:,0,0]=c;o[:,0,1]=-s;o[:,1,0]=s;o[:,1,1]=c;return o
aa = mat2aa(_Ry(yaw) @ _Rx(pitch) @ _Rz(roll))

bp = np.zeros((F, 21, 3), np.float32)
bp[:, 11] = aa * 0.6; bp[:, 14] = aa * 0.4                        # neck + head (same split as combine)
go = np.zeros((F, 3), np.float32)                                # frontal (Y-up SMPL-X faces -Z? use yaw180)
th = np.radians(180.0); go[:, 1] = th
model = smplx.create("/work/models", model_type="smplx", gender="male", num_betas=10,
                     use_pca=False, flat_hand_mean=True, num_expression_coeffs=100, batch_size=F)
with torch.no_grad():
    v = model(global_orient=torch.from_numpy(go), body_pose=torch.from_numpy(bp.reshape(F, -1))).vertices.numpy().astype(np.float32)
faces = model.faces.astype(np.int64)
ys = v[:, :, 1]; top = float(ys.max()); headc = top - 0.13
ctr = np.array([0, headc, 0], np.float32); YFOV = 0.6; dist = 0.34 / np.tan(0.5 * YFOV)
cam = _aim([0, headc, dist], ctr)
r = pyrender.OffscreenRenderer(288, 360); os.makedirs("/tmp/calib", exist_ok=True)
for fi in range(F):
    s = pyrender.Scene(bg_color=[0.5, 0.5, 0.5, 1], ambient_light=[0.85, 0.85, 0.85])
    s.add(build_mesh(v[fi], faces, _uv, _tex) if _tex is not None else
          pyrender.Mesh.from_trimesh(trimesh.Trimesh(v[fi], faces, process=False), smooth=True))
    s.add(pyrender.PerspectiveCamera(yfov=YFOV, aspectRatio=288/360), pose=cam)
    s.add(pyrender.DirectionalLight(color=np.ones(3), intensity=4.0), pose=_aim([0, headc, dist+1], ctr))
    Image.fromarray(r.render(s)[0]).save(f"/tmp/calib/f_{fi:04d}.png")
r.delete()
subprocess.run(["ffmpeg","-y","-loglevel","error","-framerate",str(a.fps),"-i","/tmp/calib/f_%04d.png",
                "-c:v","libx264","-pix_fmt","yuv420p","-crf","16",a.out], check=True)
print("CALIB_HEAD_OK", a.out, "F", F)
