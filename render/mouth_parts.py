#!/usr/bin/env python3
"""Teeth + tongue + dark interior for the SMPL-X mouth (which has none -> open mouth renders
as a black void). AUTO-FITS the mouth opening every frame: each part is anchored between the
actual upper and lower lip (measured per frame), so it resizes/repositions with the mouth and
rides with the head — one set of params works on every frame, not just one.

Params (render/mouth_params.json), per part: v0,v1 = vertical band in lip-space where 0=lower
lip, 1=upper lip (so upper teeth ~0.7-1.0, lower ~0-0.3); z = depth behind the lip (m, more
negative=deeper); w = width vs the mouth width; color RGB. Globals: width (overall x scale),
anchor_up (shift the band), gate (jaw-open before teeth show, applied at render). Tune live:
render/teeth_viser.sh -> http://<host>:8772
"""
import json
import os
import numpy as np

DEFAULTS = {
    "width": 1.0, "anchor_up": 0.0, "gate": 0.08,
    "upper":  {"v0": 0.70, "v1": 1.00, "z": -0.004, "w": 0.95, "color": [236, 232, 222]},
    "lower":  {"v0": 0.00, "v1": 0.30, "z": -0.008, "w": 0.85, "color": [220, 216, 205]},
    "tongue": {"v0": 0.06, "v1": 0.55, "z": -0.013, "w": 0.72, "color": [178, 76, 80]},
    "cavity": {"v0": -0.05, "v1": 1.05, "z": -0.022, "w": 1.05, "color": [22, 10, 12]},
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
    L = verts[:, lip_idx, :]                         # (F, n, 3)
    var = L[:, :, 1].var(0)                          # per-lip-vert vertical motion
    upper_m = var <= np.percentile(var, 40)          # upper lip (rigid w/ head, low jaw motion)
    lower_m = var >= np.percentile(var, 65)          # lower lip (drops with the jaw)
    if upper_m.sum() < 3: upper_m = np.ones(len(var), bool)
    if lower_m.sum() < 3: lower_m = np.ones(len(var), bool)

    # head orientation per frame from the skull shell above the mouth (3D-spread -> stable Kabsch)
    rig = np.where(verts[0, :, 1] >= cy_thresh(verts))[0]
    P0 = verts[0, rig]; P0c = P0.mean(0); P0d = P0 - P0c
    cyaw, syaw = np.cos(yaw_rad), np.sin(yaw_rad)
    r0 = np.array([cyaw, 0.0, -syaw]); f0 = np.array([syaw, 0.0, cyaw])   # yaw-rotated X / Z

    up, lo, to, ca = P["upper"], P["lower"], P["tongue"], P["cavity"]
    parts = (up, lo, to, ca)
    aup = float(P.get("anchor_up", 0.0)); wmul = float(P.get("width", 1.0))
    MV = np.empty((F, 16, 3), np.float32)
    for i in range(F):
        Pi = verts[i, rig]; Pic = Pi.mean(0)
        H = P0d.T @ (Pi - Pic)
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
        ri = R @ r0; fi = R @ f0                      # mouth right / outward axes this frame
        Lp = L[i]
        uc = Lp[upper_m].mean(0); lc = Lp[lower_m].mean(0)   # upper/lower lip centres (auto)
        axis = uc - lc                                # lower->upper; length = current opening
        W = float((Lp @ ri).max() - (Lp @ ri).min())  # mouth width this frame
        out16 = []
        for part in parts:
            xw = part["w"] * wmul * W * 0.5
            base = ri * xw
            p0 = lc + (part["v0"] + aup) * axis + part["z"] * fi   # bottom edge (v0)
            p1 = lc + (part["v1"] + aup) * axis + part["z"] * fi   # top edge (v1)
            out16 += [p0 - base, p0 + base, p1 + base, p1 - base]
        MV[i] = np.array(out16, np.float32)

    faces = []
    for g in range(4):                               # 4 quads -> 8 tris
        b = g * 4
        faces += [[b, b + 1, b + 2], [b, b + 2, b + 3]]
    faces = np.array(faces, np.int64)
    col = np.zeros((16, 3), np.uint8)
    for g, part in enumerate(parts):
        col[g * 4:g * 4 + 4] = part["color"]
    return MV, faces, col


def cy_thresh(verts):
    y = verts[0, :, 1]
    return np.percentile(y, 90)                       # skull/forehead/eyes region
