#!/usr/bin/env python3
"""Teeth + tongue + dark interior for the SMPL-X mouth, RIGGED TO THE BONES.

SMPL-X has no teeth; an open mouth renders as a black void. We add a tiny mesh and attach it
to the actual skeleton via the model's skinning weights:
  - upper teeth  -> the SKULL bone (head joint 15) — fixed to the cranium/maxilla
  - lower teeth + tongue -> the JAW bone (joint 22) — rotate with the mandible (jaw_pose)
  - cavity -> top edge on the skull, bottom edge on the jaw, so it STRETCHES open by itself
Per frame, each bone's rigid transform is recovered by Kabsch-fitting its skinned verts
(rest->frame). Placement (the gum lines / mouth width / face normal) is detected automatically
from where those bones meet the lip region. So it opens with the real jaw and never detaches —
no per-frame fitting, minimal params. Falls back to a lip-centroid estimate if no model given.

Params (render/mouth_params.json), per part: depth = recess behind the lip (m), h = strip
height (m), w = width vs mouth width, color RGB. Globals: width (x scale), gate (jaw-open
before teeth show, applied at render).
"""
import json
import os
import numpy as np

JAW_J, HEAD_J = 22, 15

DEFAULTS = {
    "width": 1.0, "gate": 0.08,
    "upper":  {"depth": 0.005, "h": 0.007, "w": 0.92, "color": [236, 232, 222]},
    "lower":  {"depth": 0.005, "h": 0.007, "w": 0.85, "color": [220, 216, 205]},
    "tongue": {"depth": 0.011, "h": 0.012, "w": 0.72, "color": [178, 76, 80]},
    "cavity": {"depth": 0.018, "h": 0.0, "w": 1.05, "color": [22, 10, 12]},
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


def _kabsch(A, B):
    """Rigid R,t with B ~= A @ R.T + t (maps rest set A to frame set B)."""
    ca = A.mean(0); cb = B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return R, cb - R @ ca


def _faces_cols(parts):
    faces = []
    for g in range(4):
        b = g * 4
        faces += [[b, b + 1, b + 2], [b, b + 2, b + 3]]
    faces += [[f[0], f[2], f[1]] for f in faces]      # back-faces too (double-sided)
    col = np.zeros((16, 3), np.uint8)
    for g, part in enumerate(parts):
        col[g * 4:g * 4 + 4] = part["color"]
    return np.array(faces, np.int64), col


def _atlas_uv(nparts, N):
    """UVs into a vertical 4-band atlas (band per part: upper/lower teeth, tongue, cavity).
    rowA = gum/lip edge -> top of band; rowB = into the mouth -> bottom of band."""
    uv = []
    for p in range(nparts):
        uv += [[c / N, (p + 0.10) / nparts] for c in range(N + 1)]   # rowA
        uv += [[c / N, (p + 0.90) / nparts] for c in range(N + 1)]   # rowB
    return np.array(uv, np.float32)


def build_mouth(verts, lip_idx, jaw_rad, yaw_rad=0.0, params=None, model=None, return_uv=False):
    P = params or load_params()
    F = verts.shape[0]
    lip = np.asarray(lip_idx)
    up_p, lo_p, to_p, ca_p = P["upper"], P["lower"], P["tongue"], P["cavity"]
    parts = (up_p, lo_p, to_p, ca_p)
    gw = float(P.get("width", 1.0))
    jr = np.clip(np.asarray(jaw_rad, np.float32), 0, None) if jaw_rad is not None else np.zeros(F)

    if model is None:
        return _fallback(verts, lip, jr, yaw_rad, parts, gw, return_uv)

    Wt = model.lbs_weights
    Wt = Wt.detach().cpu().numpy() if hasattr(Wt, "detach") else np.asarray(Wt)
    jaw_v = np.where(Wt[:, JAW_J] > 0.5)[0]
    skull_v = np.where(Wt[:, HEAD_J] > 0.5)[0]
    lip_jaw = lip[np.isin(lip, jaw_v)]            # lower lip (on the jaw bone)
    lip_skull = lip[np.isin(lip, skull_v)]        # upper lip (on the skull bone)
    if len(jaw_v) < 8 or len(skull_v) < 8 or len(lip_jaw) < 3 or len(lip_skull) < 3:
        return _fallback(verts, lip, jr, yaw_rad, parts, gw)

    rest_i = int(jr.argmin())                      # most-closed frame -> clean gum lines
    Rv = verts[rest_i]
    su = Rv[lip_skull]; sj = Rv[lip_jaw]
    # up DIRECTION from the lip-band centroids (~2cm apart -> stable; the inner-edge difference
    # is ~0 on a closed mouth and its direction would be pure noise -> tilted teeth).
    up = su.mean(0) - sj.mean(0)
    up = up / np.linalg.norm(up) if np.linalg.norm(up) > 1e-6 else np.array([0.0, 1.0, 0.0])
    # INNER lip edges (the mouth line) for placement: extremes along up, not world-y (robust to
    # head turn/roll). upper lip's lowest / lower lip's highest along up.
    uc = su[(su @ up) <= np.percentile(su @ up, 35)].mean(0)
    lc = sj[(sj @ up) >= np.percentile(sj @ up, 65)].mean(0)
    # right = widest lip spread in the plane perpendicular to up (tracks head roll); normal out.
    Lc = Rv[lip] - Rv[lip].mean(0)
    Lc = Lc - np.outer(Lc @ up, up)
    _, _, Vt = np.linalg.svd(Lc, full_matrices=False)
    right = Vt[0]; right = right - (right @ up) * up; right /= np.linalg.norm(right)
    normal = np.cross(right, up); normal /= np.linalg.norm(normal)
    if ((uc + lc) / 2 - Rv[skull_v].mean(0)) @ normal < 0:
        normal = -normal
    right = np.cross(up, normal)
    Wm = float(np.ptp(Rv[lip] @ right))            # mouth width

    # ---- curved gum lines (the dental arch): sample the inner lip edge across the mouth so the
    # teeth BEND with the gumline (front forward, sides receding) instead of a flat bar ----
    N = 7                                          # segments per arch

    def gum_line(pts, inner_sign, wfrac):
        proj = pts @ right
        c = 0.5 * (np.percentile(proj, 5) + np.percentile(proj, 95))
        half = 0.5 * (np.percentile(proj, 95) - np.percentile(proj, 5)) * wfrac
        out = []
        for x in np.linspace(c - half, c + half, N + 1):
            m = np.abs(proj - x) <= max(half / N, 1e-4) * 1.6
            sub = pts[m] if m.sum() >= 2 else pts[np.argsort(np.abs(proj - x))[:4]]
            spu = sub @ up
            sel = sub[spu <= np.percentile(spu, 45)] if inner_sign < 0 else sub[spu >= np.percentile(spu, 55)]
            out.append(sel.mean(0) if len(sel) else sub.mean(0))
        return np.array(out, np.float32)           # (N+1, 3), follows the arch in 3D

    ug = gum_line(su, -1, gw)                       # upper gum (skull), against the upper lip
    lg = gum_line(sj, +1, gw)                       # lower gum (jaw); shared span so cavity aligns

    # each part = two polylines (rowA against the lip, rowB extending into the mouth); the gum
    # points already carry the arch curve + the tiny recess keeps them just behind the lips.
    def strip(line, hsign, part):
        a = line - part["depth"] * normal
        b = a + hsign * part["h"] * up
        return a, b
    geo = [(strip(ug, -1, up_p), 0, 0, up_p["color"]),    # upper teeth: skull/skull
           (strip(lg, +1, lo_p), 1, 1, lo_p["color"]),    # lower teeth: jaw/jaw
           (strip(lg, +1, to_p), 1, 1, to_p["color"]),    # tongue: jaw/jaw
           ((ug - ca_p["depth"] * normal, lg - ca_p["depth"] * normal), 0, 1, ca_p["color"])]  # cavity: skull/jaw

    rest, bones, cols, faces = [], [], [], []
    for (rowA, rowB), boneA, boneB, color in geo:
        base = len(rest)
        rest += list(rowA) + list(rowB)
        bones += [boneA] * (N + 1) + [boneB] * (N + 1)
        cols += [color] * (2 * (N + 1))
        for k in range(N):
            a0, a1 = base + k, base + k + 1
            b0, b1 = base + N + 1 + k, base + N + 1 + k + 1
            faces += [[a0, a1, b1], [a0, b1, b0]]
    faces += [[f[0], f[2], f[1]] for f in faces]   # double-sided
    rest = np.array(rest, np.float32)
    bones = np.array(bones); faces = np.array(faces, np.int64)
    col = np.array(cols, np.uint8)

    def sub(a, n=400):
        return a if len(a) <= n else a[np.linspace(0, len(a) - 1, n).astype(int)]
    ss, js = sub(skull_v), sub(jaw_v)
    s0, j0 = verts[rest_i, ss], verts[rest_i, js]
    M = len(rest)
    MV = np.empty((F, M, 3), np.float32)
    for i in range(F):
        Rs, ts = _kabsch(s0, verts[i, ss])
        Rj, tj = _kabsch(j0, verts[i, js])
        MV[i, bones == 0] = rest[bones == 0] @ Rs.T + ts
        MV[i, bones == 1] = rest[bones == 1] @ Rj.T + tj
    if return_uv:
        return MV, faces, col, _atlas_uv(len(geo), N)
    return MV, faces, col


def _fallback(verts, lip, jr, yaw_rad, parts, gw, return_uv=False):
    """No model: estimate gum lines from jaw-variance lip split, no bone rig (best-effort)."""
    L = verts[:, lip, :]
    var = L[:, :, 1].var(0)
    upper_m = var <= np.percentile(var, 40); lower_m = var >= np.percentile(var, 65)
    cyaw, syaw = np.cos(yaw_rad), np.sin(yaw_rad)
    r0 = np.array([cyaw, 0.0, -syaw]); n0 = np.array([syaw, 0.0, cyaw]); up0 = np.array([0.0, 1.0, 0.0])
    up_p, lo_p, to_p, ca_p = parts
    MV = np.empty((len(verts), 16, 3), np.float32)
    for i in range(len(verts)):
        uc = L[i, upper_m].mean(0); lc = L[i, lower_m].mean(0)
        Wm = float(np.ptp(L[i] @ r0))

        def strip(anchor, hsign, part):
            xw = part["w"] * gw * Wm * 0.5; base = anchor - part["depth"] * n0
            edge = base + hsign * part["h"] * up0
            return [base - xw * r0, base + xw * r0, edge + xw * r0, edge - xw * r0]
        xwc = ca_p["w"] * gw * Wm * 0.5; dc = ca_p["depth"]
        MV[i] = np.array(strip(uc, -1, up_p) + strip(lc, 1, lo_p) + strip(lc, 1, to_p) +
                         [uc - dc * n0 - xwc * r0, uc - dc * n0 + xwc * r0,
                          lc - dc * n0 + xwc * r0, lc - dc * n0 - xwc * r0], np.float32)
    faces, col = _faces_cols(parts)
    if return_uv:
        uv = []
        for p in range(4):
            uv += [[0, (p + .1) / 4], [1, (p + .1) / 4], [1, (p + .9) / 4], [0, (p + .9) / 4]]
        return MV, faces, col, np.array(uv, np.float32)
    return MV, faces, col
