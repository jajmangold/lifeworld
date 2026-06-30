import sys, os, json, subprocess
import cv2, numpy as np

stem  = sys.argv[1]
src   = sys.argv[2]                      # original full-res source
scale = 2

box = json.load(open(f"/workspace/myinput/headbox_{stem}.json"))
bx, by, bw, bh = box["x"], box["y"], box["w"], box["h"]
W2, H2 = box["W"]*scale, box["H"]*scale

head = f"/workspace/myinput/head_up_{stem}.mp4"
tmp  = f"/workspace/myinput/_composite_{stem}.mp4"
out  = f"/workspace/outputs/flashvsr/wan2gp_face_fast_{stem}_2x.mp4"
os.makedirs(os.path.dirname(out), exist_ok=True)

# aligned upscaled-head size (center-crop of bw*scale x bh*scale)
ch_probe = cv2.VideoCapture(head)
hw = int(ch_probe.get(cv2.CAP_PROP_FRAME_WIDTH))
hh = int(ch_probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
ch_probe.release()
# head_up may be rendered at a higher scale than the 2x canvas (ULTRA=4x). Downscale it onto the canvas
# (supersampling -> sharper face). HEAD_SCALE = head_up's native scale; the loop resizes hf to (hw,hh).
HEAD_SCALE = int(os.environ.get("HEAD_SCALE", "2"))
if HEAD_SCALE != scale:
    f_ds = scale / float(HEAD_SCALE)
    hw, hh = int(round(hw * f_ds)), int(round(hh * f_ds))
px = bx*scale + (bw*scale - hw)//2
py = by*scale + (bh*scale - hh)//2
print(f"[cmp] canvas {W2}x{H2}; head {hw}x{hh} at ({px},{py})", flush=True)

# box ramp mask (feathers the crop rectangle into the Lanczos bg)
r = 160
ramp = np.linspace(0, 1, r, dtype=np.float32)
boxmask = np.ones((hh, hw), np.float32)
boxmask[:r, :] *= ramp[:, None]; boxmask[-r:, :] *= ramp[::-1, None]
boxmask[:, :r] *= ramp[None, :]; boxmask[:, -r:] *= ramp[None, ::-1]

# HEAD MATTE: if a per-frame alpha matte is staged at myinput/matte_<stem>/m%04d.png, intersect it
# (eroded inward + feathered) with the box mask. This keeps the diffusion-sharpened pixels strictly
# INSIDE the head silhouette, so FlashVSR's edge-ringing halo never reaches the silhouette boundary —
# the boundary stays the clean feathered Lanczos edge. EROD/FEAT are in 1440p px.
matte_dir = f"/workspace/myinput/matte_{stem}"
use_matte = os.path.isdir(matte_dir)
EROD = int(os.environ.get("MATTE_EROD", "14"))
FEAT = float(os.environ.get("MATTE_FEAT", "12"))
ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*EROD+1, 2*EROD+1))
print(f"[cmp] head-matte={'ON' if use_matte else 'OFF'} erod={EROD} feat={FEAT}", flush=True)

cs = cv2.VideoCapture(src); chv = cv2.VideoCapture(head)
fps = cs.get(cv2.CAP_PROP_FPS)
vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W2, H2))
i = 0
while True:
    ok, fr = cs.read()
    if not ok: break
    okh, hf = chv.read()
    canvas = cv2.resize(fr, (W2, H2), interpolation=cv2.INTER_LANCZOS4)
    if okh:
        if hf.shape[1] != hw or hf.shape[0] != hh:
            hf = cv2.resize(hf, (hw, hh), interpolation=cv2.INTER_LANCZOS4)
        m = boxmask
        if use_matte:
            mp = f"{matte_dir}/m%04d.png" % (i+1)
            mt = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
            if mt is not None:
                if mt.shape[1] != W2 or mt.shape[0] != H2:
                    mt = cv2.resize(mt, (W2, H2), interpolation=cv2.INTER_LINEAR)
                mt_box = mt[py:py+hh, px:px+hw].astype(np.float32) / 255.0
                mt_box = cv2.erode(mt_box, ker)
                mt_box = cv2.GaussianBlur(mt_box, (0, 0), FEAT)
                m = boxmask * mt_box
        m3 = m[:, :, None]
        reg = canvas[py:py+hh, px:px+hw].astype(np.float32)
        canvas[py:py+hh, px:px+hw] = np.clip(reg*(1-m3) + hf.astype(np.float32)*m3, 0, 255).astype(np.uint8)
    vw.write(canvas); i += 1
cs.release(); chv.release(); vw.release()
print(f"[cmp] wrote {i} frames -> {tmp}", flush=True)

subprocess.run(["ffmpeg", "-y", "-i", tmp, "-i", src, "-map", "0:v:0", "-map", "1:a:0?",
                "-c:v", "libx264", "-crf", "17", "-preset", "medium", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-shortest", out],
               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
print(f"[cmp] DONE -> {out}", flush=True)
