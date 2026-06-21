#!/usr/bin/env python3
"""Teeth + tongue + dark interior for the SMPL-X mouth (which has none -> open mouth
renders as a black void). Builds a tiny vertex-colored mesh anchored to the lip
region; the lower teeth + tongue drop with the jaw so it tracks speech. Hidden
behind the lips when closed, visible when open.

Returns (mverts (F,M,3), faces (T,3), colors (M,3) uint8) to render alongside the body.
"""
import numpy as np


def build_mouth(verts, lip_idx, jaw_rad, drop_scale=0.06, width=0.42, y_drop=0.009, yaw_rad=0.0):
    """Mouth interior: a DARK recessed cavity (fills the void) + a subtle, dim, deeply
    recessed upper-teeth strip + tongue. Deliberately not a bright white bar. The lower
    teeth are folded into the cavity (a separate bright strip read as fake)."""
    F = verts.shape[0]
    L = verts[:, lip_idx, :]                       # (F, n, 3)
    # Anchor to verts that MOVE with the jaw (the lower lip / mouth) — the nose is the
    # most-forward part of the lip region but doesn't move, so a "front-most" anchor put
    # the teeth on the nose. Jaw-motion variance excludes the nose automatically.
    var_y = L[:, :, 1].var(0)
    move = var_y >= np.percentile(var_y, 65)
    if move.sum() < 8:                             # fallback if little jaw motion in clip
        move = L[0, :, 2] >= np.percentile(L[0, :, 2], 80)
    sel = L[0, move]                               # rest positions of the moving lip verts
    cx = float(sel[:, 0].mean())
    cy = float(sel[:, 1].mean()) + 0.004 - y_drop  # ~mouth line (lower lip + small up)
    zlip = float(sel[:, 2].mean())
    hw = (float(sel[:, 0].max()) - float(sel[:, 0].min())) * 0.5 * width
    d = np.clip(np.asarray(jaw_rad), 0, None) * drop_scale

    def quad(y0, y1, z, xw):
        return np.array([[cx - xw, y0, z], [cx + xw, y0, z],
                         [cx + xw, y1, z], [cx - xw, y1, z]], np.float32)

    upper = quad(cy + 0.002, cy + 0.0055, zlip - 0.013, hw * 0.92)   # dim upper teeth, recessed
    MV = []
    for i in range(F):
        di = float(d[i])
        tongue = quad(cy - 0.007 - di, cy - 0.001 - di, zlip - 0.018, hw * 0.7)
        cavity = quad(cy - 0.010 - di, cy + 0.009, zlip - 0.024, hw * 1.0)   # dark, deepest
        MV.append(np.concatenate([upper, tongue, cavity], 0))
    MV = np.stack(MV, 0).astype(np.float32)                 # (F, 12, 3)

    # rotate parts about Y around the mouth centre to track the head's yaw (else they
    # face the camera on an angled head and clip through the nose/cheek).
    if yaw_rad:
        c, s = np.cos(yaw_rad), np.sin(yaw_rad)
        dx = MV[..., 0] - cx; dz = MV[..., 2] - zlip
        MV[..., 0] = cx + c * dx + s * dz
        MV[..., 2] = zlip - s * dx + c * dz

    faces = []
    for g in range(3):
        b = g * 4
        faces += [[b, b + 1, b + 2], [b, b + 2, b + 3]]
    faces = np.array(faces, np.int64)

    col = np.zeros((12, 3), np.uint8)
    col[0:4] = [188, 182, 168]      # upper teeth (dim, not bright white)
    col[4:8] = [132, 70, 74]        # tongue
    col[8:12] = [26, 11, 13]        # dark interior cavity
    return MV, faces, col
