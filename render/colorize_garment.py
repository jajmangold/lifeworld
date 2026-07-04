"""Luminance-preserving recolor of a garment albedo map (the artifact-free way to change a suit's COLOR).
Projection-baking a Klein-edited render only works for spatial PATTERN changes on clean, contiguous UVs;
for a plain recolor it aliases badly on fragmented UV islands (jacket -> digital-camo). Instead: keep the
original albedo's luminance (all the woven detail, folds, AO baked in) and push only the chroma (a,b in LAB)
toward the target. sd.cpp/Klein still DRIVES it -- we sample the target color from the Klein-edited render's
changed region -- so natural-language control ("deep navy", "burgundy") is preserved, but the application is
a clean per-texel color transfer with no seams, no aliasing.

  python3 colorize_garment.py <orig_map.png> <out_map.png> <view_front.png> <edited_front.png> [Lscale]

Run in a cv2/numpy container (swap-server). If the view/edited pair is omitted, falls back to a fixed navy.
"""
import sys, cv2, numpy as np

orig_path, out_path = sys.argv[1], sys.argv[2]
view_p = sys.argv[3] if len(sys.argv) > 3 else None
edit_p = sys.argv[4] if len(sys.argv) > 4 else None
Lscale = float(sys.argv[5]) if len(sys.argv) > 5 else 0.55

orig = cv2.imread(orig_path)
lab = cv2.cvtColor(orig, cv2.COLOR_BGR2LAB).astype(np.float32)
L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]

# --- target chroma: sampled from Klein's edited render where it changed the garment, else fixed navy ---
if view_p and edit_p:
    v = cv2.imread(view_p).astype(np.float32); e = cv2.imread(edit_p).astype(np.float32)
    if v is not None and e is not None:
        if v.shape != e.shape: e = cv2.resize(e, (v.shape[1], v.shape[0]))
        diff = np.abs(e - v).sum(2)                          # where Klein repainted the garment
        m = diff > (diff.max() * 0.25)
        elab = cv2.cvtColor(e.astype(np.uint8), cv2.COLOR_BGR2LAB).astype(np.float32)
        ta = float(np.median(elab[..., 1][m])); tb = float(np.median(elab[..., 2][m]))
        tL = float(np.median(elab[..., 0][m]))
        print(f"[colorize] sampled target LAB a={ta:.0f} b={tb:.0f} L={tL:.0f} from {int(m.sum())} changed px")
    else:
        ta, tb, tL = 137.0, 99.0, 39.0
else:
    ta, tb, tL = 137.0, 99.0, 39.0                           # deep navy #12244f

# recolor only the neutral fabric (leave already-saturated bits like skin/logos alone)
chroma = np.sqrt((A - 128) ** 2 + (B - 128) ** 2)
mask = chroma < 25
A2 = np.where(mask, ta, A)
B2 = np.where(mask, tb, B)
L2 = np.where(mask, np.clip(L * Lscale + tL * 0.25, 0, 255), L)   # keep relative shading, seat into target's value range
res = cv2.cvtColor(np.dstack([L2, A2, B2]).astype(np.uint8), cv2.COLOR_LAB2BGR)
cv2.imwrite(out_path, res)
print(f"COLORIZE_OK -> {out_path} (recolored {mask.mean()*100:.0f}% neutral texels)")
