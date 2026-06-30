"""Premultiplied-over composite of the transparent character frames onto the static env bg.
EEVEE outputs premultiplied alpha, so the correct over is  out = rgb_premult + bg*(1-alpha)  (NO
unpremultiply divide, which clamps edge pixels to white -> halo; and NO straight overlay, which
darkens edges -> dark fringe). Writes c%04d.png next to the f%04d.png inputs.
  python3 composite_premult.py <anchor_anim_dir>
"""
import cv2, numpy as np, glob, os, sys
d = sys.argv[1]
bg = cv2.imread(os.path.join(d, "bg.png")).astype(np.float32)
frames = sorted(glob.glob(os.path.join(d, "f[0-9][0-9][0-9][0-9].png")))
for f in frames:
    im = cv2.imread(f, cv2.IMREAD_UNCHANGED)
    a = im[:, :, 3:4].astype(np.float32) / 255.0
    rgb = im[:, :, :3].astype(np.float32)            # already premultiplied by EEVEE
    out = (rgb + bg * (1.0 - a)).clip(0, 255).astype(np.uint8)
    cv2.imwrite(f.replace(os.sep + "f", os.sep + "c"), out)
print(f"COMPOSITE_OK {len(frames)} frames (premult-over)")
