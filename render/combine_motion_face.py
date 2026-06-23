#!/usr/bin/env python3
"""One SMPL-X character = Kimodo BODY (body_pose/orient/transl) + LAM/FLAME FACE (expression/jaw/
eyes) + bone-rigged teeth, textured, audio-synced. Body and face are orthogonal SMPL-X inputs.
  python combine_motion_face.py --kimodo amass.npz --arkit lam.json --audio wav --gender male
                                --tex tex.png --uv uv.npz --out out.mp4 [--yaw 0]
"""
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import sys, json, argparse, subprocess
import numpy as np, torch, smplx, pyrender, trimesh
from PIL import Image
sys.path.insert(0, "/work"); sys.path.insert(0, "/lw/render")
from face_flame import FlameDriver
from face.talk import lip_region
from render_smplx import build_mesh, _aim
from mouth_parts import build_mouth

ap = argparse.ArgumentParser()
ap.add_argument("--kimodo", required=True); ap.add_argument("--arkit", required=True)
ap.add_argument("--audio", required=True); ap.add_argument("--gender", default="male")
ap.add_argument("--tex", required=True); ap.add_argument("--uv", required=True)
ap.add_argument("--map-dir", default="/work/tools/kimodo/text_encoders/../../kimodo_dummy")
ap.add_argument("--mappings", default="/work/tools/mp2flame/mappings")
ap.add_argument("--yaw", type=float, default=0.0); ap.add_argument("--out", required=True)
ap.add_argument("--face-lead-ms", type=float, default=0.0)   # LAM jaw trails audio ~66ms; advance face
a = ap.parse_args()
# ---- Kimodo body at NATIVE frame rate (do NOT resample rotations: linear-interpolating
# axis-angle is invalid and flips the body sideways mid-clip). Render at Kimodo's fps. ----
d = np.load(a.kimodo, allow_pickle=True)
FPS = int(d["mocap_frame_rate"]) if "mocap_frame_rate" in d else 30
body = d["pose_body"].astype(np.float32)              # (F,63) axis-angle, used as-is
go0 = d["root_orient"].astype(np.float32); tr0 = d["trans"].astype(np.float32)
F = len(body)

def aa2mat(v):
    t = np.linalg.norm(v, axis=1, keepdims=True); t = np.clip(t, 1e-8, None); k = v / t
    K = np.zeros((len(v), 3, 3), np.float32); K[:, 0, 1] = -k[:, 2]; K[:, 0, 2] = k[:, 1]
    K[:, 1, 0] = k[:, 2]; K[:, 1, 2] = -k[:, 0]; K[:, 2, 0] = -k[:, 1]; K[:, 2, 1] = k[:, 0]
    t = t[:, :, None]; I = np.eye(3)[None]
    return I + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)
def mat2aa(R):
    tr = np.clip((R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] - 1) / 2, -1, 1); ang = np.arccos(tr)
    v = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], 1)
    n = np.linalg.norm(v, axis=1, keepdims=True); n[n < 1e-8] = 1
    return (v / n) * ang[:, None]
th = np.radians(a.yaw)
Rfix = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], np.float32)          # Z-up -> Y-up
Ry = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]], np.float32)
M = (Ry @ Rfix).astype(np.float32)
go = mat2aa(M[None] @ aa2mat(go0)).astype(np.float32)
transl = (tr0 @ M.T).astype(np.float32)

# ---- LAM/FLAME face ----
expr, jaw, leye, reye = FlameDriver(a.mappings).drive(json.load(open(a.arkit)), F, FPS, gain=0.4)
# LAM's frame 0 is already mouth-open, but TTS has leading silence -> mouth moves before sound.
# Align the face ONSET to the audio onset: neutral (closed) until speech starts, then LAM motion.
import wave as _wave
_w = _wave.open(a.audio); _sr = _w.getframerate()
_s = np.abs(np.frombuffer(_w.readframes(_w.getnframes()), np.int16).astype(np.float32))
onset_f = int(round(np.argmax(_s > _s.max() * 0.05) / _sr * FPS))
shift = onset_f - int(round(a.face_lead_ms / 1000.0 * FPS))     # delay to onset (minus optional lead)
if shift > 0:                                                   # delay: neutral-fill the silent head
    pad = lambda x: np.concatenate([np.zeros((shift,) + x.shape[1:], x.dtype), x[:-shift]], 0)
    expr, jaw, leye, reye = pad(expr), pad(jaw), pad(leye), pad(reye)
elif shift < 0:                                                 # advance: hold last frame at the tail
    k = -shift; adv = lambda x: np.concatenate([x[k:], np.repeat(x[-1:], k, 0)], 0)
    expr, jaw, leye, reye = adv(expr), adv(jaw), adv(leye), adv(reye)

betas = np.zeros((1, 10), np.float32)
model = smplx.create("/work/models", model_type="smplx", gender=a.gender, num_betas=10,
                     use_pca=False, flat_hand_mean=True, num_expression_coeffs=100, batch_size=F)
with torch.no_grad():
    v = model(betas=torch.from_numpy(np.tile(betas, (F, 1))), global_orient=torch.from_numpy(go),
              body_pose=torch.from_numpy(body), transl=torch.from_numpy(transl),
              expression=torch.from_numpy(expr), jaw_pose=torch.from_numpy(jaw),
              leye_pose=torch.from_numpy(leye), reye_pose=torch.from_numpy(reye)).vertices.numpy().astype(np.float32)
faces = model.faces.astype(np.int64)
uv = np.load(a.uv)["uv_coordinates"]; tex = Image.open(a.tex).convert("RGB")
lip = lip_region(model, betas[0])["idx"]
mv, mf, mcol = build_mouth(v, lip, jaw[:, 0], model=model)

# ---- medium shot: frame head down to ~mid-thigh, follow body, fixed distance ----
ys = v[:, :, 1]; head = float(ys.max()); pelvis = float(np.median(ys))
cen = v.mean(1); shot_h = (head - pelvis) * 2.2                          # head..~thigh
YFOV = 0.7; dist = shot_h / (2 * np.tan(0.5 * YFOV)) * 1.1
r = pyrender.OffscreenRenderer(540, 720); os.makedirs("/tmp/combo_f", exist_ok=True)
for fi in range(F):
    cy = head - shot_h * 0.42                                            # center a bit below head
    ctr = np.array([cen[fi, 0], cy, cen[fi, 2]], np.float32)
    cam = _aim([ctr[0], ctr[1], ctr[2] + dist], ctr)
    s = pyrender.Scene(bg_color=[0.07, 0.07, 0.08, 1], ambient_light=[0.5, 0.5, 0.5])
    s.add(build_mesh(v[fi], faces, uv, tex))
    if jaw[fi, 0] > 0.05:
        s.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(mv[fi], mf, vertex_colors=mcol, process=False), smooth=False))
    s.add(pyrender.PerspectiveCamera(yfov=YFOV, aspectRatio=540 / 720), pose=cam)
    s.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.0), pose=_aim([ctr[0] + 1, ctr[1] + 2, ctr[2] + 2], ctr))
    s.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.3), pose=_aim([ctr[0] - 2, ctr[1] + 1, ctr[2] + 1], ctr))
    Image.fromarray(r.render(s)[0]).save(f"/tmp/combo_f/f_{fi:04d}.png")
r.delete()
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS), "-i", "/tmp/combo_f/f_%04d.png",
                "-i", a.audio,
                # hold the last frame 0.4s so the video OUTLASTS the audio (AAC end-padding makes the
                # audio track ~50ms longer than the frames -> otherwise the last words play after the
                # picture ends). No -shortest, so the padded video is kept.
                "-vf", "tpad=stop_mode=clone:stop_duration=0.4",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18",
                "-c:a", "aac", "-movflags", "+faststart", a.out], check=True)
print("COMBO_OK", a.out, "F", F)
