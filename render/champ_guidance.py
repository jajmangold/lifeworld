#!/usr/bin/env python3
"""Render a Kimodo AMASS/SMPL-X motion into Champ guidance maps (depth + normal) so Champ can
animate a reference image with our generated motion. Full-body, portrait, facing camera.
  python champ_guidance.py <amass.npz> <out_dir> [W H]
Writes <out_dir>/depth/####.png and <out_dir>/normal/####.png"""
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import sys, numpy as np, torch, smplx, pyrender, trimesh
from PIL import Image
NPZ, OUT = sys.argv[1], sys.argv[2]
W = int(sys.argv[3]) if len(sys.argv) > 3 else 512
H = int(sys.argv[4]) if len(sys.argv) > 4 else 768
d = np.load(NPZ, allow_pickle=True)
go0 = d["root_orient"].astype(np.float32); body = d["pose_body"].astype(np.float32)
trans = d["trans"].astype(np.float32); F = len(body)


def aa2mat(v):
    t = np.linalg.norm(v, axis=1, keepdims=True); t = np.clip(t, 1e-8, None); k = v / t
    K = np.zeros((len(v), 3, 3), np.float32); K[:, 0, 1] = -k[:, 2]; K[:, 0, 2] = k[:, 1]
    K[:, 1, 0] = k[:, 2]; K[:, 1, 2] = -k[:, 0]; K[:, 2, 0] = -k[:, 1]; K[:, 2, 1] = k[:, 0]
    t = t[:, :, None]
    return np.eye(3)[None] + np.sin(t) * K + (1 - np.cos(t)) * (K @ K)


def mat2aa(R):
    tr = np.clip((R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2] - 1) / 2, -1, 1); ang = np.arccos(tr)
    v = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], 1)
    n = np.linalg.norm(v, axis=1, keepdims=True); n[n < 1e-8] = 1
    return (v / n) * ang[:, None]


Rfix = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], np.float32)          # Z-up -> Y-up
th = np.radians(180.0)                                                    # face camera
Ry = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]], np.float32)
M = (Ry @ Rfix).astype(np.float32)
go = mat2aa(M[None] @ aa2mat(go0)).astype(np.float32)
transl = (trans @ M.T).astype(np.float32)
model = smplx.create("/work/models", model_type="smplx", gender="male", num_betas=10,
                     use_pca=False, flat_hand_mean=True, batch_size=F)
with torch.no_grad():
    _out = model(global_orient=torch.from_numpy(go), body_pose=torch.from_numpy(body),
                 transl=torch.from_numpy(transl))
    v = _out.vertices.numpy().astype(np.float32)
    J = _out.joints.numpy().astype(np.float32)          # (F, n_joints, 3) for dwpose
faces = model.faces.astype(np.int64)
# fixed camera: frame the whole motion's body, portrait, small margin
allv = v.reshape(-1, 3); mn = allv.min(0); mx = allv.max(0); ctr = (mn + mx) / 2
bodyH = (mx[1] - mn[1]); YFOV = 0.85
dist = bodyH / (2 * np.tan(0.5 * YFOV)) * 1.12
cam_pose = np.eye(4); cam_pose[:3, 3] = [ctr[0], ctr[1], ctr[2] + dist]
Rcam = cam_pose[:3, :3]
os.makedirs(f"{OUT}/depth", exist_ok=True); os.makedirs(f"{OUT}/normal", exist_ok=True)
os.makedirs(f"{OUT}/lit", exist_ok=True)   # lit grey human; run real DWPose on this -> dwpose/ (proper _all format)
r = pyrender.OffscreenRenderer(W, H)
cam = pyrender.PerspectiveCamera(yfov=YFOV, aspectRatio=W / H)
litlight = pyrender.DirectionalLight(color=np.ones(3), intensity=3.5)
vn_all = None
for fi in range(F):
    tm = trimesh.Trimesh(v[fi], faces, process=False)
    vn = tm.vertex_normals.astype(np.float32)                            # world-space normals
    vn_view = vn @ Rcam                                                  # to view space (cam looks -Z)
    ncol = np.clip((vn_view * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
    # ---- normal pass: flat (full ambient, no lights) so vertex colors ARE the normal map ----
    tmn = trimesh.Trimesh(v[fi], faces, vertex_colors=ncol, process=False)
    s = pyrender.Scene(bg_color=[0, 0, 0, 1], ambient_light=[1.0, 1.0, 1.0])
    s.add(pyrender.Mesh.from_trimesh(tmn, smooth=True)); s.add(cam, pose=cam_pose)
    ncolor, depth = r.render(s, flags=pyrender.constants.RenderFlags.FLAT)
    Image.fromarray(ncolor).save(f"{OUT}/normal/{fi:04d}.png")
    # ---- depth pass: near=bright on black ----
    dd = depth.copy(); m = dd > 0
    if m.any():
        near, far = dd[m].min(), dd[m].max()
        g = np.zeros_like(dd); g[m] = 1.0 - (dd[m] - near) / (far - near + 1e-6)  # near bright
        gi = (g * 255).astype(np.uint8)
    else:
        gi = np.zeros(dd.shape, np.uint8)
    Image.fromarray(gi).convert("RGB").save(f"{OUT}/depth/{fi:04d}.png")
    # ---- lit pass: mid-grey human on grey bg (DWPose annotates this for proper dwpose) ----
    sl = pyrender.Scene(bg_color=[0.5, 0.5, 0.5, 1], ambient_light=[0.6, 0.6, 0.6])
    sl.add(pyrender.Mesh.from_trimesh(trimesh.Trimesh(v[fi], faces, process=False), smooth=True))
    sl.add(cam, pose=cam_pose); sl.add(litlight, pose=cam_pose)
    Image.fromarray(r.render(sl)[0]).save(f"{OUT}/lit/{fi:04d}.png")
r.delete()
print("GUIDANCE_OK", OUT, "frames", F, f"{W}x{H}")
