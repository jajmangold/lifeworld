"""Host composite for the Klein head bake: blend the front-projected Klein head onto the original
head texture (Image_2) with a feathered facing mask, then CARVE the eyes back to original so blinks
read correctly (baking Klein's open-eye projection over the eye/eyelid/iris region broke full blinks).
Inputs (from hair_reproject.py frontbake): hb_orig_head.png, hb_baked_front.png, hb_facing.png.
  python3 hair_compose.py <orig> <baked_front> <facing> <out_tex.png>
Eye-carve coords are measured on a 1024 atlas and scaled to the actual texture size.
"""
import sys, cv2, numpy as np
orig  = cv2.imread(sys.argv[1], cv2.IMREAD_UNCHANGED)
baked = cv2.imread(sys.argv[2], cv2.IMREAD_UNCHANGED)
face  = cv2.imread(sys.argv[3], cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
out_path = sys.argv[4]
H, W = orig.shape[:2]
if baked.shape[:2] != (H, W): baked = cv2.resize(baked, (W, H))
if face.shape[:2]  != (H, W): face  = cv2.resize(face,  (W, H))
cov = (baked[:, :, 3].astype(np.float32) / 255.0) if baked.shape[2] == 4 else (baked.sum(2) > 8).astype(np.float32)
# facing-weighted, front-only blend (smoothstep)
t = np.clip((face - 0.18) / (0.55 - 0.18), 0, 1); m = t * t * (3 - 2 * t)
m = m * cov
# --- carve eyes (keep original eye/eyelid + iris island) so blinks stay full ---
sx, sy = W / 1024.0, H / 1024.0
carve = np.ones((H, W), np.float32)
cv2.ellipse(carve, (int(240*sx), int(168*sy)), (int(150*sx), int(48*sy)), 0, 0, 360, 0.0, -1)  # both eye apertures
carve[0:int(360*sy), int(535*sx):W] = 0.0                                                       # iris island (top-right)
carve = cv2.GaussianBlur(carve, (0, 0), 6.0)
m = cv2.GaussianBlur(m * carve, (0, 0), 3.0)[..., None]
out = (baked[:, :, :3].astype(np.float32) * m + orig[:, :, :3].astype(np.float32) * (1 - m)).clip(0, 255).astype(np.uint8)
res = orig.copy(); res[:, :, :3] = out
cv2.imwrite(out_path, res[:, :, :3])
print("HEADTEX_OK (eyes carved) meanM=%.3f -> %s" % (m.mean(), out_path))
