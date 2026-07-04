"""RESIDENT VAE DECODE DAEMON. The LTX VAE (~2GB) sits resident on ONE card and decodes the raw LTX latents
(.pt) produced by the resident denoise daemons -> low-res frames (PNG) ready to ship to rtx0 FlashVSR. Because
the VAE is tiny and resident, decode is fast with no eviction; one decode card feeds several generation cards.
Watches LATDIR for *.pt (+ *.meta.json), writes <out>/<stem>/f*.png, moves the .pt to done/. Args: [LATDIR]"""
import os, sys, time, glob, json, gc, signal
import torch
from diffusers import AutoencoderKLLTX2Video
from PIL import Image
import numpy as np

DT = torch.bfloat16
DEV = "cuda:0"
LATDIR = sys.argv[1] if len(sys.argv) > 1 else "/root/ComfyUI/output/latents"
DONE = os.path.join(LATDIR, "done"); os.makedirs(DONE, exist_ok=True)
import glob as _g
MODEL = sorted(_g.glob("/root/.cache/huggingface/hub/models--OzzyGT--LTX-2.3-Distilled-1.1-sdnq-dynamic-int4/snapshots/*"))[0]
_run = {"go": True}
for s in (signal.SIGTERM, signal.SIGINT): signal.signal(s, lambda *a: _run.update(go=False))

t = time.time()
vae = AutoencoderKLLTX2Video.from_pretrained(MODEL, subfolder="vae", torch_dtype=DT)
try:
    vae.enable_tiling(); vae.use_framewise_decoding = True; vae.tile_sample_min_num_frames = 16
except Exception: pass
vae = vae.to(DEV).eval()
dev = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
print(f"[decoded dev={dev} t={time.time()-t:.1f}] VAE resident on {DEV} | VRAM {torch.cuda.memory_allocated()/1e9:.2f}GB | watching {LATDIR}", flush=True)

def decode(pt_path, meta):
    lat = torch.load(pt_path, map_location="cpu").to(DEV, DT)
    with torch.no_grad():
        vid = vae.decode(lat, None, return_dict=False)[0]
    # postprocess to uint8 PIL frames (mirror video_processor.postprocess for 'pil')
    v = vid.float()                       # [B,C,F,H,W] in ~[-1,1]
    v = ((v.clamp(-1, 1) + 1) / 2 * 255).round().to(torch.uint8)[0]  # [C,F,H,W]
    v = v.permute(1, 2, 3, 0).cpu().numpy()  # [F,H,W,C]
    return [Image.fromarray(fr) for fr in v]

_pid = os.getpid()
while _run["go"]:
    todo = sorted(glob.glob(os.path.join(LATDIR, "*.pt")))
    if not todo: time.sleep(0.5); continue
    pt0 = todo[0]; stem = os.path.basename(pt0)[:-3]
    pt = pt0 + f".busy{_pid}"
    try:
        os.rename(pt0, pt)          # atomic claim: only one decode daemon wins this latent
    except OSError:
        continue                     # another daemon grabbed it first
    mp = os.path.join(LATDIR, f"{stem}.meta.json")
    meta = json.load(open(mp)) if os.path.exists(mp) else {}
    outdir = os.path.join(LATDIR, "frames", stem); os.makedirs(outdir, exist_ok=True)
    ts = time.time()
    try:
        imgs = decode(pt, meta)
        for i, im in enumerate(imgs): im.save(os.path.join(outdir, f"f{i:03d}.png"))
        a = np.asarray(imgs[0].convert("L")).astype(float); rc = float(np.corrcoef(a[:-1].ravel(), a[1:].ravel())[0,1])
        print(f"[dec {stem}] {len(imgs)}f -> {time.time()-ts:.1f}s row-corr={rc:.3f} {'REAL' if rc>0.9 else 'SUSPECT'} -> {outdir}", flush=True)
    except Exception as e:
        print(f"[dec {stem}] FAILED {repr(e)[:160]}", flush=True)
    finally:
        gc.collect(); torch.cuda.empty_cache()
        try:
            os.rename(pt, os.path.join(DONE, os.path.basename(pt0)))  # store as <stem>.pt (dispatch checks this)
            if os.path.exists(mp): os.rename(mp, os.path.join(DONE, os.path.basename(mp)))
        except Exception: pass
print("[decoded] stopped", flush=True)
