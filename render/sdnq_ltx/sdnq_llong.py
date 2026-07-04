"""LEVER 3 — LATENT-SLIDING long LTX-2.3 clips on the proven offload path. Unlike sdnq_long.py (which DECODES
every window to pixels and RE-ENCODES the overlap to condition the next -> N VAE round-trips, N transformer
evictions), this keeps everything in LATENT space: each window runs output_type='latent' (no decode), and the
next window is conditioned on the PREVIOUS window's tail LATENTS by monkeypatching pipe.vae.encode to return
them (so the VAE is NEVER touched mid-stream). One vae.decode at the very end -> the ~44s transformer eviction
is paid ONCE, not per window. Continuity is latent-space (no lossy decode/encode round-trip).

Caches: embeds + connectors (fixed prompt). Args: W H TOTAL_FRAMES WIN OVL_LAT STEPS [SEED]"""
import os, sys, time, glob, inspect, hashlib, types, gc
import torch, sdnq  # noqa
from diffusers import DiffusionPipeline, LTX2ConditionPipeline
from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
from sdnq.loader import apply_sdnq_options_to_model
from PIL import Image
import numpy as np

DT = torch.bfloat16
W   = int(sys.argv[1]) if len(sys.argv) > 1 else 512
H   = int(sys.argv[2]) if len(sys.argv) > 2 else 768
TOT = int(sys.argv[3]) if len(sys.argv) > 3 else 89    # total pixel frames wanted
WIN = int(sys.argv[4]) if len(sys.argv) > 4 else 57    # per-window pixel frames (8n+1), fits one card
OVLL= int(sys.argv[5]) if len(sys.argv) > 5 else 2     # overlap in LATENT frames (~8 pixel frames each)
STEPS = int(sys.argv[6]) if len(sys.argv) > 6 else 8
# official Lightricks DISTILLED_SIGMAS (8 steps) — the distilled model's trained trajectory, NOT linspace.
# Using them makes 8 steps clean (kills the watermark-ghost) and stays fast. See sdnq_prod2.py.
DISTILLED_SIGMAS_8 = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875]
SEED  = int(sys.argv[7]) if len(sys.argv) > 7 else 42
PATH = sorted(glob.glob("/root/.cache/huggingface/hub/models--OzzyGT--LTX-2.3-Distilled-1.1-sdnq-dynamic-int4/snapshots/*"))[0]
# rich character-locked prompt (MUST match sdnq_gen._DEF_POS exactly to reuse its embed/conn caches)
POS = ("Medium close-up vertical portrait of a distinguished male television news anchor in his late fifties, "
    "with neatly styled silver-gray side-parted hair and a clean-shaven, lightly tanned face with friendly blue eyes, "
    "seated at a modern broadcast desk. He wears a tailored navy suit over a light blue dress shirt and a blue "
    "patterned silk tie. He speaks directly to the camera with calm, articulate, authoritative expressions, his lips "
    "moving naturally and his gaze steady. Soft key lighting from the front-left illuminates his face against a gently "
    "blurred dark blue studio background. The footage looks like real broadcast television: photorealistic, with "
    "natural skin texture, visible pores and fine detail. The camera holds a steady, locked-off shot.")
NEG = "cartoon, cgi, 3d render, plastic skin, waxy, blurry, distorted, extra people, on-screen text, watermark, logo"
EMB  = f"/root/ComfyUI/output/embeds_{hashlib.md5((POS+'||'+NEG).encode()).hexdigest()[:10]}.pt"
CONN = f"/root/ComfyUI/output/conn_{hashlib.md5((POS+'||'+NEG).encode()).hexdigest()[:10]}.pt"

t = time.time()
base = DiffusionPipeline.from_pretrained(PATH, torch_dtype=DT)
base.transformer = apply_sdnq_options_to_model(base.transformer, dtype=DT, use_quantized_matmul=True)
pipe = LTX2ConditionPipeline(**{k: v for k, v in base.components.items()
                                if k in set(inspect.signature(LTX2ConditionPipeline.__init__).parameters)})
try:
    pipe.vae.enable_tiling()
    # enable_tiling() only turns on SPATIAL tiling; the final decode is many frames at once -> temporal OOM.
    # Turn on FRAMEWISE (temporal) decoding so vae.decode tiles+blends along frames -> fits + no seams.
    pipe.vae.use_framewise_decoding = True
    pipe.vae.tile_sample_min_num_frames = 16
except Exception as e: print("tiling warn", e)
pipe.enable_model_cpu_offload()
# embeds cache (fixed prompt)
assert os.path.exists(EMB), "need cached embeds (run sdnq_gen once with this prompt)"
pe, pm, ne, nm = torch.load(EMB, map_location="cpu")
g = lambda x: (x.to("cuda", DT) if (torch.is_tensor(x) and x.is_floating_point()) else (x.to("cuda") if torch.is_tensor(x) else None))
EMB_KW = dict(prompt_embeds=g(pe), prompt_attention_mask=g(pm))
if ne is not None: EMB_KW.update(negative_prompt_embeds=g(ne), negative_prompt_attention_mask=g(nm))
# connector cache: compute ONCE if missing (else every one of the ~31 windows re-streams the 6.35GB module).
if os.path.exists(CONN):
    _ct = tuple(torch.load(CONN, map_location="cpu"))
    print("[connectors] loaded from cache", flush=True)
else:
    with torch.no_grad():                                  # gs=1.0 -> connectors see pos-only (matches window calls)
        _ct = tuple(x.detach().cpu() for x in pipe.connectors(g(pe), g(pm), padding_side="left"))
    torch.save(_ct, CONN); print("[connectors] computed + cached", flush=True)
pipe.connectors = lambda pe_, pm_, padding_side="left": tuple(x.to(pe_.device) for x in _ct)
print(f"[t={time.time()-t:.1f}] pipeline ready (offload + caches)", flush=True)

_real_encode = pipe.vae.encode
def denoise_window(num, cond_tail_raw, seed):
    """Run one window in latent space. cond_tail_raw: [B,C,OVLL,H,W] raw-vae latents to condition frame 0 on
    (or None for the first window). Returns raw-vae latents [B,C,f,H,W]."""
    kw = dict(width=W, height=H, num_frames=num, num_inference_steps=STEPS, guidance_scale=1.0, frame_rate=25,
              output_type="latent", generator=torch.Generator("cpu").manual_seed(seed), **EMB_KW)
    if STEPS == 8: kw["sigmas"] = DISTILLED_SIGMAS_8   # correct distilled schedule -> no watermark
    if cond_tail_raw is None:
        kw["conditions"] = []
        out = pipe(**kw)
    else:
        klat = cond_tail_raw.shape[2]
        kpix = (klat - 1) * pipe.vae_temporal_compression_ratio + 1     # dummy pixel frames (content ignored)
        dummy = torch.zeros(kpix, 3, H, W, dtype=pipe.vae.dtype)
        # inject: vae.encode returns the precomputed tail latents instead of encoding pixels
        pipe.vae.encode = lambda x, *a, **k: types.SimpleNamespace(latents=cond_tail_raw.to("cuda", pipe.vae.dtype))
        try:
            kw["conditions"] = [LTX2VideoCondition(frames=dummy, index=0, strength=1.0)]
            out = pipe(**kw)
        finally:
            pipe.vae.encode = _real_encode
    return out.frames    # [B,C,f,H,W] raw-vae latents

# ---- slide ----
all_lat = None
w = 0
while True:
    tail = None if all_lat is None else all_lat[:, :, -OVLL:].contiguous()
    tw = time.time()
    lat = denoise_window(WIN, tail, SEED + w)
    if all_lat is None:
        all_lat = lat
    else:
        all_lat = torch.cat([all_lat, lat[:, :, OVLL:]], dim=2)   # drop the OVLL conditioned frames (== prev tail)
    fnow = (all_lat.shape[2] - 1) * pipe.vae_temporal_compression_ratio + 1
    print(f"[window {w}] {WIN}f in {time.time()-tw:.1f}s -> latent frames {all_lat.shape[2]} (~{fnow} pixels)", flush=True)
    gc.collect(); torch.cuda.empty_cache()
    w += 1
    if fnow >= TOT or w > 60: break   # safety cap (1 min @512x768 ~= 31 windows)

# ---- ONE decode at the end (the only VAE-on-GPU / transformer-eviction event) ----
td = time.time()
with torch.no_grad():
    video = pipe.vae.decode(all_lat.to(pipe.vae.dtype).to("cuda"), None, return_dict=False)[0]
imgs = pipe.video_processor.postprocess_video(video, output_type="pil")[0]
imgs = imgs[:TOT]
print(f"[decode] {all_lat.shape[2]} latent frames -> {len(imgs)} pixel frames in {time.time()-td:.1f}s", flush=True)

os.makedirs("/root/ComfyUI/output/sdnq_llong", exist_ok=True)
for i, im in enumerate(imgs): im.save(f"/root/ComfyUI/output/sdnq_llong/f{i:03d}.png")
# verify: intra-frame structure + cross-seam continuity (frame at each window boundary vs its neighbor)
def rc(a, b):
    a = np.asarray(a.convert("L")).astype(float); b = np.asarray(b.convert("L")).astype(float)
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
intra = rc(imgs[0], imgs[0])
seam = min((rc(imgs[i-1], imgs[i]) for i in range(1, len(imgs))), default=1.0)
a0 = np.asarray(imgs[0].convert("L")).astype(float); struct = float(np.corrcoef(a0[:-1].ravel(), a0[1:].ravel())[0,1])
print(f"[LLONG] {W}x{H} {len(imgs)}f in {w} windows | total {time.time()-t:.1f}s | struct={struct:.3f} "
      f"{'REAL' if struct>0.9 else 'BAD'} | worst adjacent-frame corr={seam:.3f} (continuity)", flush=True)
