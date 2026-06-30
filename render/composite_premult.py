"""Premultiplied-over composite of the transparent character frames onto the static env bg.
EEVEE outputs premultiplied alpha, so the correct over is  out = rgb_premult + bg*(1-alpha)  (NO
unpremultiply divide, which clamps edge pixels to white -> halo; and NO straight overlay, which
darkens edges -> dark fringe). Writes c%04d.png next to the f%04d.png inputs.

Edge feather (kills the thin silhouette "matte line"): even a correct premult-over leaves a crisp
~2-3px alpha step, which a bright subject on a dark bg shows as a hair-thin edge line. We pull the
matte in by ERODE px (drops the outermost edge sliver) and soften it over FEATHER px so the whole
silhouette dissolves into the bg instead of ending on a hard line. To stay edge-accurate we
unpremultiply to true color first (guarded), then re-over with the feathered alpha.
  FEATHER=<sigma px, default 1.2>  ERODE=<px, default 1>  python3 composite_premult.py <dir>
  FEATHER=0 -> plain premult-over (no feather).
"""
import cv2, numpy as np, glob, os, sys
d = sys.argv[1]
FEATHER = float(os.environ.get("FEATHER", "1.5"))   # gaussian sigma in px for the alpha edge ramp
ERODE   = int(os.environ.get("ERODE", "1"))          # px to pull the matte in before feathering
bg = cv2.imread(os.path.join(d, "bg.png")).astype(np.float32)
frames = sorted(glob.glob(os.path.join(d, "f[0-9][0-9][0-9][0-9].png")))
ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ERODE + 1, 2 * ERODE + 1)) if ERODE > 0 else None
for f in frames:
    im = cv2.imread(f, cv2.IMREAD_UNCHANGED)
    a = im[:, :, 3].astype(np.float32) / 255.0
    rgb = im[:, :, :3].astype(np.float32)            # already premultiplied by EEVEE
    if FEATHER > 0:
        # true (straight) color, guarded so fully-transparent pixels don't blow up (they get a'~0)
        color = np.clip(rgb / np.maximum(a[..., None], 1e-3), 0, 255)
        af = cv2.erode(a, ker) if ker is not None else a
        af = cv2.GaussianBlur(af, (0, 0), FEATHER)
        af = af[..., None]
        out = (color * af + bg * (1.0 - af)).clip(0, 255).astype(np.uint8)
    else:
        out = (rgb + bg * (1.0 - a[..., None])).clip(0, 255).astype(np.uint8)
    cv2.imwrite(f.replace(os.sep + "f", os.sep + "c"), out)
print(f"COMPOSITE_OK {len(frames)} frames (premult-over, feather={FEATHER} erode={ERODE})")
