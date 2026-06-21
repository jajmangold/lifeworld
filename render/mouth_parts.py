#!/usr/bin/env python3
"""Teeth + tongue + dark interior for the SMPL-X mouth (which has none -> open mouth
renders as a black void). Builds a tiny vertex-colored mesh anchored to the lip
region; the lower teeth + tongue drop with the jaw so it tracks speech. Hidden
behind the lips when closed, visible when open.

Returns (mverts (F,M,3), faces (T,3), colors (M,3) uint8) to render alongside the body.
"""
import numpy as np


def build_mouth(verts, lip_idx, jaw_rad, drop_scale=0.06, width=0.5, y_drop=0.009):
    F = verts.shape[0]
    m0 = verts[0, lip_idx]
    # anchor to the FRONT-MOST lip verts (actual lip surface), not the broad region's
    # centroid (which spans chin->philtrum and sits too high/forward). y_drop lowers it
    # a touch (the front-lip mean still skews slightly toward the upper lip).
    front = m0[m0[:, 2] >= np.percentile(m0[:, 2], 75)]
    cx = float(front[:, 0].mean()); cy = float(front[:, 1].mean()) - y_drop
    hw = (float(front[:, 0].max()) - float(front[:, 0].min())) * 0.5 * width
    zc = float(front[:, 2].mean()) - 0.010   # recess just behind the lip surface
    d = np.clip(np.asarray(jaw_rad), 0, None) * drop_scale   # lower-group drop per frame

    def quad(y0, y1, z, xw):
        return np.array([[cx - xw, y0, z], [cx + xw, y0, z],
                         [cx + xw, y1, z], [cx - xw, y1, z]], np.float32)

    upper = quad(cy + 0.001, cy + 0.006, zc, hw)            # static upper teeth (small)
    MV = []
    for i in range(F):
        di = float(d[i])
        lower = quad(cy - 0.006 - di, cy - 0.001 - di, zc, hw)        # lower teeth (drops)
        tongue = quad(cy - 0.005 - di, cy - 0.001 - di, zc + 0.003, hw * 0.75)
        back = quad(cy - 0.008 - di, cy + 0.008, zc - 0.010, hw * 1.05)  # dark interior
        MV.append(np.concatenate([upper, lower, tongue, back], 0))
    MV = np.stack(MV, 0).astype(np.float32)                 # (F, 16, 3)

    faces = []
    for g in range(4):
        b = g * 4
        faces += [[b, b + 1, b + 2], [b, b + 2, b + 3]]
    faces = np.array(faces, np.int64)

    col = np.zeros((16, 3), np.uint8)
    col[0:4] = [232, 230, 218]      # upper teeth
    col[4:8] = [232, 230, 218]      # lower teeth
    col[8:12] = [168, 88, 92]       # tongue
    col[12:16] = [38, 16, 18]       # dark interior
    return MV, faces, col
