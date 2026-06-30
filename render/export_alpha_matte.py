"""Export per-frame alpha mattes (m%04d.png) from the transparent character frames f%04d.png.
The premium FlashVSR composite uses these to confine diffusion sharpening INSIDE the head silhouette
(eroded matte), so FlashVSR's edge-ringing never haloes the silhouette boundary.
  python3 export_alpha_matte.py <anchor_anim_dir> <out_dir>
"""
import cv2, glob, os, sys
d, o = sys.argv[1], sys.argv[2]
os.makedirs(o, exist_ok=True)
fs = sorted(glob.glob(os.path.join(d, "f[0-9][0-9][0-9][0-9].png")))
for i, f in enumerate(fs):
    im = cv2.imread(f, cv2.IMREAD_UNCHANGED)
    cv2.imwrite(os.path.join(o, "m%04d.png" % (i + 1)), im[:, :, 3])
print("MATTE_OK", len(fs), "->", o)
