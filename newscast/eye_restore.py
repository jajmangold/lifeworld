"""Eye-preservation for the premium path. FlashVSR-face is a diffusion super-res with NO eye-carve (unlike
the GFPGAN keepeyes restore), so it re-opens/smooths the 3-frame blinks that the render produces. This
composites the REAL eyes (with the full blink + eye motion) from the pre-FlashVSR video back over FlashVSR's
sharper output, soft-masked around each eye. Eyes come out a touch softer than FlashVSR but with correct
blinks — the right trade (eyes are small; a wrong-open blink is far more noticeable than slight softness).

  python3 eye_restore.py <flashvsr.mp4> <pre_flashvsr.mp4> <faces.json> <out.mp4>

faces.json = the swap's saved per-frame [bbox, kps] (kps in pre-video pixel coords); kps[0],kps[1] = eyes.
Handles the resolution difference (FlashVSR is 2x the pre video) by scaling kps + upscaling the eye patch.
"""
import sys, json, cv2, numpy as np

fv_path, pre_path, faces_path, out_path = sys.argv[1:5]
faces = json.load(open(faces_path)) if faces_path and faces_path != "-" else None
fv = cv2.VideoCapture(fv_path); pre = cv2.VideoCapture(pre_path)
fps = fv.get(cv2.CAP_PROP_FPS) or 25
W = int(fv.get(cv2.CAP_PROP_FRAME_WIDTH)); H = int(fv.get(cv2.CAP_PROP_FRAME_HEIGHT))
pw = int(pre.get(cv2.CAP_PROP_FRAME_WIDTH)) or W
scale = W / float(pw)                       # FlashVSR is upscaled vs pre (typically 2x)
out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
n = done = 0
while True:
    okf, ff = fv.read(); okp, pf = pre.read()
    if not okf:
        break
    if okp and faces and n < len(faces) and faces[n] is not None:
        _, kps = faces[n]
        kps = np.asarray(kps, np.float32)
        eyes = kps[:2] * scale              # left/right eye centres in FlashVSR pixels
        pf_up = cv2.resize(pf, (W, H), interpolation=cv2.INTER_CUBIC) if pf.shape[1] != W else pf
        eye_sp = float(np.linalg.norm(eyes[1] - eyes[0]))
        # JUST the eye aperture (eyeball + lid + lashes) -- two tiny ellipses, NOT the whole socket. Keeps the
        # composited region minimal so there's no tone patch across the under-eye/brow; only the actual eye
        # (which FlashVSR wrongly re-opens) is restored.
        rx = max(8, int(eye_sp * 0.26)); ry = max(5, int(eye_sp * 0.15))
        mask = np.zeros((H, W), np.float32)
        for ex, ey in eyes:
            cv2.ellipse(mask, (int(ex), int(ey)), (rx, ry), 0, 0, 360, 1.0, -1)
        k = max(5, (int(eye_sp * 0.09) | 1))                 # tight feather (small patch, minimal bleed)
        mask = cv2.GaussianBlur(mask, (k, k), 0)[..., None]
        # TONE-MATCH the pre-FlashVSR eye patch to FlashVSR's local tone. FlashVSR shifts the face's overall
        # brightness/color; without this the pasted eyes keep the pre tone -> a visible tone patch around the
        # eyes. Match per-channel mean+std over the (skin-dominated) eye region so the paste is seamless.
        m = mask[..., 0] > 0.04
        if m.any():
            pf_up = pf_up.astype(np.float32)
            for c in range(3):
                sd, dd = pf_up[..., c][m], ff[..., c][m].astype(np.float32)
                ss = sd.std() or 1.0
                pf_up[..., c] = np.clip((pf_up[..., c] - sd.mean()) * (dd.std() / ss) + dd.mean(), 0, 255)
            # mild unsharp so the patch's crispness better matches FlashVSR's diffusion sharpening
            pf_up = np.clip(pf_up * 1.35 - cv2.GaussianBlur(pf_up, (0, 0), 1.1) * 0.35, 0, 255)
        ff = (pf_up * mask + ff.astype(np.float32) * (1 - mask)).astype(np.uint8)
        done += 1
    out.write(ff); n += 1
out.release()
print(f"EYE_RESTORE_OK {n} frames, {done} eye-composited -> {out_path}")
