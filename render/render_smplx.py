#!/usr/bin/env python3
"""Blender-free SMPL-X renderer — GPU rasterization via pyrender/EGL.

Takes the baked-vertex clip from sampl's gen_motion.py (verts (F,V,3) or
(P,F,V,3), faces) and rasterizes each frame to PNG on the GPU, headless, with no
OpenGL display/context dependency (EGL offscreen). ~100s of fps vs Blender
EEVEE's CPU-software fallback in a headless container.

Usage (inside the pyrender image, GPU visible):
    python render_smplx.py --clip clip.npz --out-dir frames \
        [--uv smplx_uv_2023.npz --texture skin.png] \
        [--res-x 720 --res-y 1080] [--rot-x 0]
"""
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")  # headless GPU; must precede GL import

import argparse
import numpy as np
import trimesh
import pyrender
import imageio.v2 as imageio
from PIL import Image


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--uv", default=None)
    ap.add_argument("--texture", default=None)
    ap.add_argument("--textures", default=None,
                    help="comma list of per-body textures (one per person); "
                         "overrides --texture for multi-character scenes")
    ap.add_argument("--res-x", type=int, default=720)
    ap.add_argument("--res-y", type=int, default=1080)
    ap.add_argument("--rot-x", type=float, default=0.0,
                    help="deg about X. SMPL-X is Y-up (=pyrender), so 0 by default.")
    ap.add_argument("--flip-v", action="store_true",
                    help="flip texture V if it renders upside-down")
    ap.add_argument("--bg", default="0.05,0.05,0.07")
    ap.add_argument("--no-ground", action="store_true", help="omit floor plane")
    ap.add_argument("--no-shadows", action="store_true", help="disable shadow maps")
    ap.add_argument("--framing", default="full", choices=["full", "medium", "face"],
                    help="shot size: full body / waist-up / head (for lip-sync)")
    return ap.parse_args()


def ground_plane(all_verts, half=4.0):
    """Large floor quad at foot height to ground the character + catch shadows."""
    y = float(all_verts.reshape(-1, 3)[:, 1].min())
    cx, cz = all_verts.reshape(-1, 3)[:, 0].mean(), all_verts.reshape(-1, 3)[:, 2].mean()
    v = np.array([[cx - half, y, cz - half], [cx + half, y, cz - half],
                  [cx + half, y, cz + half], [cx - half, y, cz + half]], np.float32)
    f = np.array([[0, 1, 2], [0, 2, 3]], np.int64)
    tm = trimesh.Trimesh(vertices=v, faces=f, process=False)
    mat = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.32, 0.33, 0.36, 1.0], metallicFactor=0.0, roughnessFactor=1.0)
    return pyrender.Mesh.from_trimesh(tm, material=mat, smooth=False)


def rot_x_mat(deg):
    r = np.radians(deg)
    c, s = np.cos(r), np.sin(r)
    m = np.eye(4)
    m[1, 1], m[1, 2], m[2, 1], m[2, 2] = c, -s, s, c
    return m


def build_mesh(verts, faces, uv=None, tex_img=None, flip_v=False):
    """Textured if uv+tex given (unwrapped per-corner for correct seams), else clay."""
    if uv is not None and tex_img is not None:
        loops = faces.shape[0] * 3
        v2 = verts[faces].reshape(-1, 3)
        f2 = np.arange(loops, dtype=np.int64).reshape(-1, 3)
        uv2 = np.asarray(uv[:loops], dtype=np.float32).copy()
        if flip_v:
            uv2[:, 1] = 1.0 - uv2[:, 1]
        vis = trimesh.visual.TextureVisuals(uv=uv2, image=tex_img)
        tm = trimesh.Trimesh(vertices=v2, faces=f2, visual=vis, process=False)
        return pyrender.Mesh.from_trimesh(tm, smooth=False)
    tm = trimesh.Trimesh(vertices=verts, faces=faces, process=False)
    mat = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.85, 0.66, 0.55, 1.0], metallicFactor=0.0,
        roughnessFactor=0.7)
    return pyrender.Mesh.from_trimesh(tm, material=mat, smooth=True)


def frame_camera(all_verts, framing="full", aspect=0.6667):
    """Front camera (Y-up, looking down -Z). Shot size set by `framing`. Fits BOTH
    height and width (so a row of characters isn't cropped at the sides)."""
    lo = all_verts.reshape(-1, 3).min(0)
    hi = all_verts.reshape(-1, 3).max(0)
    center = (lo + hi) / 2.0
    height = float(hi[1] - lo[1])
    width = float(hi[0] - lo[0])
    yfov = np.pi / 4.0
    xfov = 2.0 * np.arctan(np.tan(yfov / 2.0) * aspect)
    # target height fraction (0=feet,1=head-top) and fit factor (smaller=closer)
    tgt_frac, fit = {"full": (0.45, 0.62), "medium": (0.66, 0.40),
                     "face": (0.90, 0.16)}[framing]
    dist_h = (height * fit) / np.tan(yfov / 2.0)
    dist_w = (width * 0.62) / np.tan(xfov / 2.0)     # keep the whole row in frame
    dist = max(dist_h, dist_w) + 0.4
    target = np.array([center[0], lo[1] + tgt_frac * height, center[2]])
    eye = np.array([center[0], target[1] + 0.04 * height, center[2] + dist])
    return yfov, _aim(eye, target), center


def main():
    a = parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    for old in os.listdir(a.out_dir):       # clear stale frames so ffmpeg can't mux them
        if old.startswith("frame_") and old.endswith(".png"):
            os.remove(os.path.join(a.out_dir, old))
    d = np.load(a.clip, allow_pickle=True)
    verts = np.asarray(d["verts"], dtype=np.float32)
    faces = np.asarray(d["faces"], dtype=np.int64)
    if verts.ndim == 3:               # (F,V,3) -> (1,F,V,3) single person
        verts = verts[None]
    P, F = verts.shape[0], verts.shape[1]

    R = rot_x_mat(a.rot_x)
    if a.rot_x:
        for p in range(P):
            verts[p] = (R[:3, :3] @ verts[p].reshape(-1, 3).T).T.reshape(F, -1, 3)

    uv = None
    tex_imgs = []                     # one per person (cycled if fewer than P)
    if a.uv and os.path.exists(a.uv):
        paths = ([p for p in a.textures.split(",")] if a.textures
                 else ([a.texture] if a.texture else []))
        paths = [p for p in paths if p and os.path.exists(p)]
        if paths:
            with np.load(a.uv) as u:
                uv = u["uv_coordinates"]
            tex_imgs = [Image.open(p).convert("RGB") for p in paths]
            print(f"[render] textured: uv={uv.shape} {len(tex_imgs)} texture(s)")
    if not tex_imgs:
        print("[render] clay (no texture)")

    yfov, cam_pose, center = frame_camera(verts, a.framing, a.res_x / a.res_y)
    bg = [float(x) for x in a.bg.split(",")]

    cam = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=a.res_x / a.res_y)
    # 3-ish point key/fill/rim
    lights = [
        (pyrender.DirectionalLight(color=np.ones(3), intensity=3.5),
         _aim([2, 2, 3], center)),
        (pyrender.DirectionalLight(color=np.ones(3), intensity=1.5),
         _aim([-3, 1, 2], center)),
        (pyrender.DirectionalLight(color=np.ones(3), intensity=1.2),
         _aim([0, 2, -3], center)),
    ]
    r = pyrender.OffscreenRenderer(a.res_x, a.res_y)
    flags = pyrender.RenderFlags.NONE
    if not a.no_shadows:
        flags |= pyrender.RenderFlags.SHADOWS_DIRECTIONAL
    ground = None if a.no_ground else ground_plane(verts)

    for fi in range(F):
        scene = pyrender.Scene(bg_color=bg + [1.0],
                               ambient_light=[0.35, 0.35, 0.38])
        if ground is not None:
            scene.add(ground)
        for p in range(P):
            tex = tex_imgs[p % len(tex_imgs)] if tex_imgs else None
            scene.add(build_mesh(verts[p, fi], faces, uv, tex, a.flip_v))
        scene.add(cam, pose=cam_pose)
        for lt, pose in lights:
            scene.add(lt, pose=pose)
        color, _ = r.render(scene, flags=flags)
        imageio.imwrite(os.path.join(a.out_dir, f"frame_{fi:04d}.png"), color)
        if fi % 12 == 0:
            print(f"[render] frame {fi+1}/{F}")
    r.delete()
    print(f"[render] done -> {a.out_dir} ({F} frames)")


def _aim(eye, target):
    """4x4 pose for a light at `eye` looking at `target` (Y-up)."""
    eye = np.asarray(eye, float)
    target = np.asarray(target, float)
    fwd = target - eye
    fwd /= (np.linalg.norm(fwd) + 1e-9)
    right = np.cross(fwd, [0, 1, 0])
    right /= (np.linalg.norm(right) + 1e-9)
    up = np.cross(right, fwd)
    m = np.eye(4)
    m[:3, 0], m[:3, 1], m[:3, 2], m[:3, 3] = right, up, -fwd, eye
    return m


if __name__ == "__main__":
    main()
