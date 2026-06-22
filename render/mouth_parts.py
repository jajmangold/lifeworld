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


def build_mouth(verts, lip_idx, jaw_rad, yaw_rad=0.0, params=None, model=None):
    P = params or load_params()
    F = verts.shape[0]
    lip = np.asarray(lip_idx)
    up_p, lo_p, to_p, ca_p = P["upper"], P["lower"], P["tongue"], P["cavity"]
    parts = (up_p, lo_p, to_p, ca_p)
    gw = float(P.get("width", 1.0))
    jr = np.clip(np.asarray(jaw_rad, np.float32), 0, None) if jaw_rad is not None else np.zeros(F)

    if model is None:
        return _fallback(verts, lip, jr, yaw_rad, parts, gw)

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
    # INNER lip edges (the mouth line), not the lip-band centroids (which sit ~2cm apart even
    # closed): upper lip's lowest verts / lower lip's highest verts.
    su = Rv[lip_skull]; sj = Rv[lip_jaw]
    uc = su[su[:, 1] <= np.percentile(su[:, 1], 35)].mean(0)   # inner upper lip
    lc = sj[sj[:, 1] >= np.percentile(sj[:, 1], 65)].mean(0)   # inner lower lip
    # Anatomical axes (PCA is unreliable on a near-closed mouth — the thin vertical slit makes
    # it swap the vertical and depth axes -> sideways teeth). up = lower->upper lip; right from
    # the head yaw, orthogonalized; normal = right x up, forced outward.
    up = uc - lc
    up = up / np.linalg.norm(up) if np.linalg.norm(up) > 1e-6 else np.array([0.0, 1.0, 0.0])
    # right = widest lip-spread direction IN THE PLANE perpendicular to up (so it tracks the
    # head's actual roll/turn at the rest frame; width >> depth there -> unambiguous).
    Lc = Rv[lip] - Rv[lip].mean(0)
    Lc = Lc - np.outer(Lc @ up, up)                # drop the up component
    _, _, Vt = np.linalg.svd(Lc, full_matrices=False)
    right = Vt[0]; right = right - (right @ up) * up; right /= np.linalg.norm(right)
    normal = np.cross(right, up); normal /= np.linalg.norm(normal)
    if ((uc + lc) / 2 - Rv[skull_v].mean(0)) @ normal < 0:
        normal = -normal                           # point OUTWARD (away from head centre)
    right = np.cross(up, normal)                    # re-orthonormalize
    Wm = float(np.ptp(Rv[lip] @ right))            # mouth width

    def strip(anchor, hsign, part):
        xw = part["w"] * gw * Wm * 0.5
        base = anchor - part["depth"] * normal
        edge = base + hsign * part["h"] * up
        return [base - xw * right, base + xw * right, edge + xw * right, edge - xw * right]

    upper = strip(uc, -1.0, up_p)                   # extends DOWN from the upper gum
    lower = strip(lc, +1.0, lo_p)                   # extends UP from the lower gum
    tongue = strip(lc, +1.0, to_p)
    xwc = ca_p["w"] * gw * Wm * 0.5; dc = ca_p["depth"]
    cavity = [uc - dc * normal - xwc * right, uc - dc * normal + xwc * right,   # top -> skull
              lc - dc * normal + xwc * right, lc - dc * normal - xwc * right]   # bottom -> jaw
    rest16 = np.array(upper + lower + tongue + cavity, np.float32)
    bone = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1])   # 0=skull, 1=jaw

    def sub(a, n=400):
        return a if len(a) <= n else a[np.linspace(0, len(a) - 1, n).astype(int)]
    ss, js = sub(skull_v), sub(jaw_v)
    s0, j0 = verts[rest_i, ss], verts[rest_i, js]
    MV = np.empty((F, 16, 3), np.float32)
    for i in range(F):
        Rs, ts = _kabsch(s0, verts[i, ss])
        Rj, tj = _kabsch(j0, verts[i, js])
        MV[i, bone == 0] = rest16[bone == 0] @ Rs.T + ts
        MV[i, bone == 1] = rest16[bone == 1] @ Rj.T + tj

    faces, col = _faces_cols(parts)
    return MV, faces, col


def _fallback(verts, lip, jr, yaw_rad, parts, gw):
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
    return MV, faces, col
