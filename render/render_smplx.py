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
from mouth_parts import load_params      # mouth/teeth gate (tunable in mouth_params.json)


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
    ap.add_argument("--framing", default="full", choices=["full", "medium", "face", "head"],
                    help="shot size: full body / waist-up / face+shoulders / head-only close-up")
    ap.add_argument("--tex-size", type=int, default=1024,
                    help="downscale textures to NxN on load (0=full); 4K re-upload/frame is the bottleneck")
    ap.add_argument("--room", action="store_true",
                    help="warm interior set (floor+walls) instead of void/bare ground")
    ap.add_argument("--floor-tex", default="", help="floor texture image for --room")
    ap.add_argument("--wall-tex", default="", help="wall texture image for --room")
    ap.add_argument("--cut", action="store_true",
                    help="cut to a medium shot of the active speaker per beat (needs shot_speaker)")
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


def _quad(corners, color, rough=1.0, tex=None, rep=1.0):
    v = np.array(corners, np.float32)
    f = np.array([[0, 1, 2], [0, 2, 3]], np.int64)
    if tex is not None:                 # textured (tiled rep x rep)
        uv = np.array([[0, 0], [rep, 0], [rep, rep], [0, rep]], np.float32)
        tm = trimesh.Trimesh(v, f, visual=trimesh.visual.TextureVisuals(uv=uv, image=tex),
                             process=False)
        m = pyrender.Mesh.from_trimesh(tm, smooth=False)
        m.primitives[0].material.doubleSided = True
        return m
    tm = trimesh.Trimesh(v, f, process=False)
    mat = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=color + [1.0], metallicFactor=0.0, roughnessFactor=rough,
        doubleSided=True)               # double-sided so winding/normals don't matter
    return pyrender.Mesh.from_trimesh(tm, material=mat, smooth=False)


def room_set(all_verts, floor_tex=None, wall_tex=None):
    """A warm interior: floor + back wall + two side walls around the cast, so they stand
    in a room (not a black void). Textured if floor_tex/wall_tex given, else flat color."""
    p = all_verts.reshape(-1, 3)
    y0 = float(p[:, 1].min())
    cx, cz = float(p[:, 0].mean()), float(p[:, 2].mean())
    W = max(3.2, (float(p[:, 0].max()) - float(p[:, 0].min())) * 1.3)   # room half-width
    back = cz - 1.4                     # back wall (behind cast; camera looks -Z)
    front = cz + 3.2                    # floor extends toward camera
    h = 2.7                             # wall height
    floor = [0.40, 0.36, 0.32]; wall = [0.56, 0.50, 0.45]; side = [0.50, 0.45, 0.40]
    return [
        _quad([[cx - W, y0, back], [cx + W, y0, back],
               [cx + W, y0, front], [cx - W, y0, front]], floor, tex=floor_tex, rep=4.0),  # floor
        _quad([[cx - W, y0, back], [cx + W, y0, back],
               [cx + W, y0 + h, back], [cx - W, y0 + h, back]], wall, tex=wall_tex, rep=2.5),  # back
        _quad([[cx - W, y0, back], [cx - W, y0, front],
               [cx - W, y0 + h, front], [cx - W, y0 + h, back]], side, tex=wall_tex, rep=2.5),  # L
        _quad([[cx + W, y0, back], [cx + W, y0, front],
               [cx + W, y0 + h, front], [cx + W, y0 + h, back]], side, tex=wall_tex, rep=2.5),  # R
    ]


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
    if framing == "head":                       # tight head-only close-up (lip-sync)
        ty = float(hi[1]) - 0.12                # ~head centre
        dist = 0.14 / np.tan(yfov / 2.0) + 0.05  # fit ~0.28 m tall
        target = np.array([center[0], ty, center[2]])
        eye = np.array([center[0], ty + 0.01, center[2] + dist])
        return yfov, _aim(eye, target), center
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

    # optional teeth/tongue mouth geometry (SMPL-X has none); (P,F,M,3)+(P,F) gate
    mverts = mfaces = mcolors = mgate = None
    if "mouth_verts" in d.files:
        mverts = np.asarray(d["mouth_verts"], np.float32)
        if mverts.ndim == 3:
            mverts = mverts[None]
        mfaces = np.asarray(d["mouth_faces"], np.int64)
        mcolors = np.asarray(d["mouth_colors"], np.uint8)
        mgate = np.asarray(d["mouth_gate"], np.float32)
        if mgate.ndim == 1:
            mgate = mgate[None]
        print(f"[render] mouth geometry: {mverts.shape}")
    gate_thr = load_params().get("gate", 0.12)         # show teeth only when mouth this open
    shot_speaker = np.asarray(d["shot_speaker"]) if "shot_speaker" in d.files else None
    beat_frames = np.asarray(d["beat_frames"]) if "beat_frames" in d.files else None

    R = rot_x_mat(a.rot_x)
    if a.rot_x:
        for p in range(P):
            verts[p] = (R[:3, :3] @ verts[p].reshape(-1, 3).T).T.reshape(F, -1, 3)
            if mverts is not None:
                mverts[p] = (R[:3, :3] @ mverts[p].reshape(-1, 3).T).T.reshape(F, -1, 3)

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
            if a.tex_size and tex_imgs and tex_imgs[0].width > a.tex_size:
                # 4K textures are re-uploaded to the GPU every frame (the render bottleneck);
                # at full-body 720p the face is ~80px so 1024 is plenty. ~16x less upload.
                tex_imgs = [im.resize((a.tex_size, a.tex_size), Image.LANCZOS) for im in tex_imgs]
            print(f"[render] textured: uv={uv.shape} {len(tex_imgs)} texture(s) @ {tex_imgs[0].size}")
    if not tex_imgs:
        print("[render] clay (no texture)")

    yfov, cam_pose, center = frame_camera(verts, a.framing, a.res_x / a.res_y)
    bg = [float(x) for x in a.bg.split(",")]

    # per-frame camera: cut to a medium shot of the active speaker, per beat (static within
    # a beat, like a dialogue edit). Falls back to the single wide shot.
    cam_poses = None
    if a.cut and shot_speaker is not None and beat_frames is not None:
        aspect = a.res_x / a.res_y
        def single(p, fr):                                  # head+shoulders on character p
            return frame_camera(verts[p, min(fr, F - 1)][None], "face", aspect)[1]
        def wide(fr):                                       # the whole room/cast
            return frame_camera(verts[:, min(fr, F - 1)], "full", aspect)[1]
        cam_poses = np.repeat(cam_pose[None], F, axis=0)
        est = min(34, F // 6)                               # establishing wide (~1.4s)
        cam_poses[:est] = wide(est // 2)
        off = 0; ncuts = 1
        for bi, n in enumerate(beat_frames):
            n = int(n); s = int(shot_speaker[off]); b0, b1 = off, off + n
            seg0 = max(b0, est)
            if seg0 < b1:
                if n > 72:                                  # long beat: insert a reaction cut
                    r0 = b0 + int(n * 0.55); r1 = min(r0 + 22, b1)
                    lis = [k for k in range(P) if k != s]
                    lp = lis[bi % len(lis)]
                    cam_poses[seg0:r0] = single(s, (seg0 + r0) // 2)
                    cam_poses[r0:r1] = single(lp, (r0 + r1) // 2)   # listener reaction
                    cam_poses[r1:b1] = single(s, (r1 + b1) // 2)
                    ncuts += 3
                else:
                    cam_poses[seg0:b1] = single(s, (seg0 + b1) // 2)
                    ncuts += 1
            off = b1
        cf = max(F - 26, est)                               # close on a wide for resolution
        cam_poses[cf:] = wide((cf + F) // 2); ncuts += 1
        print(f"[render] camera cuts: {ncuts} shots over {len(beat_frames)} beats")

    cam = pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=a.res_x / a.res_y)
    # warm, soft, fairly even key/fill/rim — high key so its shadow falls down (not a big
    # diagonal across the back wall); strong fill to keep contrast low (soft look).
    warm = np.array([1.0, 0.95, 0.88])
    cool = np.array([0.9, 0.94, 1.0])
    lights = [
        (pyrender.DirectionalLight(color=warm, intensity=3.0), _aim([1.5, 3.2, 2.5], center)),
        (pyrender.DirectionalLight(color=cool, intensity=2.2), _aim([-2.5, 1.6, 2.2], center)),
        (pyrender.DirectionalLight(color=warm, intensity=1.0), _aim([0, 2.6, -2.5], center)),
    ]
    r = pyrender.OffscreenRenderer(a.res_x, a.res_y)
    flags = pyrender.RenderFlags.NONE
    if not a.no_shadows:
        flags |= pyrender.RenderFlags.SHADOWS_DIRECTIONAL
    if a.room:
        ftex = Image.open(a.floor_tex).convert("RGB") if a.floor_tex and os.path.exists(a.floor_tex) else None
        wtex = Image.open(a.wall_tex).convert("RGB") if a.wall_tex and os.path.exists(a.wall_tex) else None
        set_meshes = room_set(verts, ftex, wtex)
        bg = [0.10, 0.09, 0.08]                 # warm so wall edges blend
        ambient = [0.45, 0.43, 0.40]
    else:
        set_meshes = [] if a.no_ground else [ground_plane(verts)]
        ambient = [0.35, 0.35, 0.38]

    for fi in range(F):
        scene = pyrender.Scene(bg_color=bg + [1.0], ambient_light=ambient)
        for sm in set_meshes:
            scene.add(sm)
        for p in range(P):
            tex = tex_imgs[p % len(tex_imgs)] if tex_imgs else None
            scene.add(build_mesh(verts[p, fi], faces, uv, tex, a.flip_v))
            if mverts is not None and mgate[p, fi] > gate_thr:   # interior only when clearly open
                mt = trimesh.Trimesh(mverts[p, fi], mfaces, vertex_colors=mcolors, process=False)
                scene.add(pyrender.Mesh.from_trimesh(mt, smooth=False))
        scene.add(cam, pose=(cam_poses[fi] if cam_poses is not None else cam_pose))
        for lt, pose in lights:
            scene.add(lt, pose=pose)
        color, _ = r.render(scene, flags=flags)
        imageio.imwrite(os.path.join(a.out_dir, f"frame_{fi:04d}.png"), color)
        if fi % 24 == 0:
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
