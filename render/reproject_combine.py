"""Combine per-view garment reprojections into one UV base-color map. Weighted-average the per-view color
bakes by their facing weight, then blend that OVER the original map so ONLY the visible (restyled) regions
change and the unseen back stays original. Run in a cv2/numpy container (swap-server).
  python3 reproject_combine.py <viewdir> <original_map.png> <out_map.png> <v1,v2,...>
"""
import sys, cv2, numpy as np

viewdir, orig_path, out_path, views = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(",")
orig = cv2.imread(orig_path).astype(np.float32)
H, W = orig.shape[:2]
acc = np.zeros((H, W, 3), np.float32); wsum = np.zeros((H, W, 1), np.float32)
for v in views:
    col = cv2.imread(f"{viewdir}/bake_col_{v}.png")
    wt = cv2.imread(f"{viewdir}/bake_w_{v}.png", cv2.IMREAD_GRAYSCALE)
    if col is None or wt is None:
        continue
    col = cv2.resize(col, (W, H)).astype(np.float32)
    wt = (cv2.resize(wt, (W, H)).astype(np.float32) / 255.0)[..., None]
    acc += col * wt; wsum += wt
reproj = acc / np.clip(wsum, 1e-3, None)
blend = np.clip(wsum * 1.4, 0, 1)                    # where the garment was seen -> use restyle; else original
final = orig * (1 - blend) + reproj * blend
cv2.imwrite(out_path, final.astype(np.uint8))
print(f"COMBINE_OK -> {out_path} (blended {sum(1 for v in views)} views over original {W}x{H})")
