#!/usr/bin/env python3
"""Teeth + tongue + dark interior for the SMPL-X mouth (which has none -> open mouth
renders as a black void). Builds a tiny vertex-colored mesh anchored to the lip region;
the lower teeth + tongue + cavity-bottom drop with the jaw so it tracks speech.

ALL geometry/colors are tunable in render/mouth_params.json (edit + run render/teeth.sh).
Returns (mverts (F,16,3), faces (8,3), colors (16,3) uint8).
"""
import json
import os
import numpy as np

DEFAULTS = {
    "width": 0.42, "y_drop": 0.009, "anchor_up": 0.004, "drop_scale": 0.06, "gate": 0.12,
    "upper":  {"y0": 0.0015, "y1": 0.0075, "z": -0.004, "w": 0.95, "color": [206, 201, 187]},
    "lower":  {"y0": -0.0075, "y1": -0.0025, "z": -0.013, "w": 0.85, "color": [206, 201, 187]},
    "tongue": {"y0": -0.007, "y1": -0.001, "z": -0.017, "w": 0.70, "color": [150, 80, 84]},
    "cavity": {"y0": -0.011, "y1": 0.010, "z": -0.026, "w": 1.05, "color": [24, 11, 13]},
}
_PARAMS_PATH = os.path.join(os.path.dirname(__file__), "mouth_params.json")


def load_params(path=None):
    p = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    path = path or _PARAMS_PATH
    if os.path.exists(path):
        for k, v in json.load(open(path)).items():
            if k.startswith("_"):
                continue
            if isinstance(v, dict) and isinstance(p.get(k), dict):
                p[k].update(v)
            else:
                p[k] = v
    return p


def build_mouth(verts, lip_idx, jaw_rad, yaw_rad=0.0, params=None):
    P = params or load_params()
    F = verts.shape[0]
    L = verts[:, lip_idx, :]                        # (F, n, 3)
    # Anchor to verts that MOVE with the jaw (the lower lip / mouth). The nose is the most-
    # forward part of the lip region but doesn't move, so a "front-most" anchor put the teeth
    # on the nose; jaw-motion variance excludes it automatically.
    var_y = L[:, :, 1].var(0)
    move = var_y >= np.percentile(var_y, 65)
    if move.sum() < 8:
        move = L[0, :, 2] >= np.percentile(L[0, :, 2], 80)
    sel = L[0, move]
    cx = float(sel[:, 0].mean())
    cy = float(sel[:, 1].mean()) + P["anchor_up"] - P["y_drop"]   # ~mouth line
    zlip = float(sel[:, 2].mean())
    hw = (float(sel[:, 0].max()) - float(sel[:, 0].min())) * 0.5 * P["width"]
    d = np.clip(np.asarray(jaw_rad), 0, None) * P["drop_scale"]

    def quad(y0, y1, z, w):
        xw = hw * w
        return np.array([[cx - xw, y0, zlip + z], [cx + xw, y0, zlip + z],
                         [cx + xw, y1, zlip + z], [cx - xw, y1, zlip + z]], np.float32)

    up, lo, to, ca = P["upper"], P["lower"], P["tongue"], P["cavity"]
    # Frame-0 rest geometry (4 quads, 16 verts), oriented to the frame-0 head yaw.
    mv0 = np.concatenate([
        quad(cy + up["y0"], cy + up["y1"], up["z"], up["w"]),       # upper teeth (skull)
        quad(cy + lo["y0"], cy + lo["y1"], lo["z"], lo["w"]),       # lower teeth
        quad(cy + to["y0"], cy + to["y1"], to["z"], to["w"]),       # tongue
        quad(cy + ca["y0"], cy + ca["y1"], ca["z"], ca["w"]),       # cavity
    ], 0).astype(np.float32)
    if yaw_rad:
        c, s = np.cos(yaw_rad), np.sin(yaw_rad)
        dx = mv0[:, 0] - cx; dz = mv0[:, 2] - zlip
        mv0[:, 0] = cx + c * dx + s * dz
        mv0[:, 2] = zlip - s * dx + c * dz

    # Rigidly attach the mouth parts to the HEAD per frame. The head moves a LOT with TalkSHOW
    # gestures (lean/turn ~15cm); anchoring at frame 0 left the teeth floating behind the head.
    # Kabsch-fit the skull cap (top-of-head verts: rigid, no jaw/expression) frame0 -> frame i,
    # apply that transform to mv0, then drop the lower teeth/tongue/cavity-bottom with the jaw.
    # rigid set = the head SHELL above the mouth (forehead, skull, sides) — 3D-spread so the
    # Kabsch rotation is well-conditioned (a flat skull cap alone is near-planar -> unstable).
    rig = np.where(verts[0, :, 1] >= cy + 0.02)[0]
    if rig.size < 30:
        rig = np.where(verts[0, :, 1] >= np.percentile(verts[0, :, 1], 88))[0]
    P0 = verts[0, rig]; P0c = P0.mean(0); P0d = P0 - P0c
    drop_idx = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13]                   # lower, tongue, cavity-bottom
    down = np.array([0.0, -1.0, 0.0], np.float32)
    MV = np.empty((F, 16, 3), np.float32)
    for i in range(F):
        Pi = verts[i, rig]; Pic = Pi.mean(0)
        H = P0d.T @ (Pi - Pic)
        U, _, Vt = np.linalg.svd(H)
        dsign = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1.0, 1.0, dsign]) @ U.T                 # frame0 -> frame i rotation
        mvi = mv0 @ R.T + (Pic - R @ P0c)                          # rigid follow head
        mvi[drop_idx] += float(d[i]) * (R @ down)                  # jaw-follow drop (head-local)
        MV[i] = mvi

    faces = []
    for g in range(4):                                             # 4 quads -> 8 tris
        b = g * 4
        faces += [[b, b + 1, b + 2], [b, b + 2, b + 3]]
    faces = np.array(faces, np.int64)

    col = np.zeros((16, 3), np.uint8)                              # 4 verts per quad
    for g, part in enumerate((up, lo, to, ca)):
        col[g * 4:g * 4 + 4] = part["color"]
    return MV, faces, col
