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
    upper = quad(cy + up["y0"], cy + up["y1"], up["z"], up["w"])    # static (skull)
    MV = []
    for i in range(F):
        di = float(d[i])                                            # jaw-follow drop
        lower = quad(cy + lo["y0"] - di, cy + lo["y1"] - di, lo["z"], lo["w"])
        tongue = quad(cy + to["y0"] - di, cy + to["y1"] - di, to["z"], to["w"])
        cavity = quad(cy + ca["y0"] - di, cy + ca["y1"], ca["z"], ca["w"])  # bottom drops
        MV.append(np.concatenate([upper, lower, tongue, cavity], 0))
    MV = np.stack(MV, 0).astype(np.float32)                         # (F, 16, 3)

    # rotate parts about Y around the mouth centre to track the head's yaw (else on an angled
    # head they face the camera and clip through the nose/cheek).
    if yaw_rad:
        c, s = np.cos(yaw_rad), np.sin(yaw_rad)
        dx = MV[..., 0] - cx; dz = MV[..., 2] - zlip
        MV[..., 0] = cx + c * dx + s * dz
        MV[..., 2] = zlip - s * dx + c * dz

    faces = []
    for g in range(4):                                             # 4 quads -> 8 tris
        b = g * 4
        faces += [[b, b + 1, b + 2], [b, b + 2, b + 3]]
    faces = np.array(faces, np.int64)

    col = np.zeros((16, 3), np.uint8)                              # 4 verts per quad
    for g, part in enumerate((up, lo, to, ca)):
        col[g * 4:g * 4 + 4] = part["color"]
    return MV, faces, col
