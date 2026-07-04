"""TWO-STAGE DISTILLED LTX-2.3 (the official production path) on the offload+caches base. Stage 1 denoises at
HALF resolution (cheap, 1/4 tokens) with DISTILLED_SIGMAS (8 steps); the LTX2 spatial x2 latent upsampler
(ltx-2.3-spatial-upscaler-x2-1.1) upscales the latents; Stage 2 refines at FULL res from the upscaled latents
with STAGE_2_DISTILLED_SIGMAS (3 steps, starts at sigma 0.909) -> sharper full-res output than single-stage.
Caches: embeds + connectors (fixed prompt). Args: W H FRAMES [SEED] [OUTDIR]"""
import os, sys, time, glob, inspect, hashlib
import torch, sdnq  # noqa
from diffusers import DiffusionPipeline, LTX2ConditionPipeline
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from sdnq.loader import apply_sdnq_options_to_model
from safetensors.torch import load_file
from PIL import Image
import numpy as np

DT = torch.bfloat16
W = int(sys.argv[1]) if len(sys.argv) > 1 else 512       # FINAL width
H = int(sys.argv[2]) if len(sys.argv) > 2 else 768       # FINAL height
F = int(sys.argv[3]) if len(sys.argv) > 3 else 57
SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 42
OUTDIR = sys.argv[5] if len(sys.argv) > 5 else "/root/ComfyUI/output/two_stage"
PATH = sorted(glob.glob("/root/.cache/huggingface/hub/models--OzzyGT--LTX-2.3-Distilled-1.1-sdnq-dynamic-int4/snapshots/*"))[0]
UPS  = glob.glob("/root/.cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/*/ltx-2.3-spatial-upscaler-x2-1.1.safetensors")[0]
POS = ("Vertical portrait video of a professional female news anchor speaking to camera, waist-up, "
       "natural realistic skin with visible pores, soft studio lighting, modern broadcast news set, documentary realism.")
NEG = "cartoon, cgi, 3d render, plastic skin, waxy, blurry, distorted, extra people"
EMB  = f"/root/ComfyUI/output/embeds_{hashlib.md5((POS+'||'+NEG).encode()).hexdigest()[:10]}.pt"
CONN = f"/root/ComfyUI/output/conn_{hashlib.md5((POS+'||'+NEG).encode()).hexdigest()[:10]}.pt"
DISTILLED_SIGMAS_8 = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875]  # stage 1 (8 steps)
STAGE_2_SIGMAS_3   = [0.909375, 0.725, 0.421875]                                          # stage 2 (3 steps, +terminal 0)

t = time.time()
base = DiffusionPipeline.from_pretrained(PATH, torch_dtype=DT)
base.transformer = apply_sdnq_options_to_model(base.transformer, dtype=DT, use_quantized_matmul=True)
pipe = LTX2ConditionPipeline(**{k: v for k, v in base.components.items()
                                if k in set(inspect.signature(LTX2ConditionPipeline.__init__).parameters)})
try:
    pipe.vae.enable_tiling()
    pipe.vae.use_framewise_decoding = True; pipe.vae.tile_sample_min_num_frames = 16
except Exception as e: print("tiling warn", e)
pipe.enable_model_cpu_offload()
# fixed-prompt caches
if os.path.exists(CONN):
    _ct = tuple(torch.load(CONN, map_location="cpu"))
    pipe.connectors = lambda pe_, pm_, padding_side="left": tuple(x.to(pe_.device) for x in _ct)
    print("[connectors] bypassed via cache", flush=True)
assert os.path.exists(EMB), "need cached embeds (run sdnq_prod2 once)"
pe, pm, ne, nm = torch.load(EMB, map_location="cpu")
g = lambda x: (x.to("cuda", DT) if (torch.is_tensor(x) and x.is_floating_point()) else (x.to("cuda") if torch.is_tensor(x) else None))
EMB_KW = dict(prompt_embeds=g(pe), prompt_attention_mask=g(pm))
if ne is not None: EMB_KW.update(negative_prompt_embeds=g(ne), negative_prompt_attention_mask=g(nm))
# upsampler (kept on CPU; moved to cuda only for the single upsample call)
ups = LTX2LatentUpsamplerModel(in_channels=128, mid_channels=1024, num_blocks_per_stage=4, dims=3,
                               spatial_upsample=True, temporal_upsample=False, use_rational_resampler=False)
ups.load_state_dict(load_file(UPS), strict=True); ups = ups.to(DT).eval()
print(f"[t={time.time()-t:.1f}] pipeline + x2 upsampler ready", flush=True)

def adain(latents, ref, factor=1.0):
    """Match per-(batch,channel) mean/std of `latents` to `ref` (official upscale post-filter)."""
    out = latents.clone()
    for i in range(latents.size(0)):
        for c in range(latents.size(1)):
            r_sd, r_mu = torch.std_mean(ref[i, c]); l_sd, l_mu = torch.std_mean(latents[i, c])
            out[i, c] = ((latents[i, c] - l_mu) / (l_sd + 1e-8)) * r_sd + r_mu
    return torch.lerp(latents, out, factor)

# ---- STAGE 1: half-res latents ----
t1 = time.time()
s1 = pipe(width=W // 2, height=H // 2, num_frames=F, num_inference_steps=8, sigmas=DISTILLED_SIGMAS_8,
          guidance_scale=1.0, frame_rate=25, conditions=[], output_type="latent",
          generator=torch.Generator("cpu").manual_seed(SEED), **EMB_KW).frames  # raw-vae latents [B,128,f,H/2/32,W/2/32]
print(f"[stage1] {W//2}x{H//2} -> latents {tuple(s1.shape)} in {time.time()-t1:.1f}s", flush=True)

# ---- UPSAMPLE x2 (on raw/unnormalized latents) ----
# NOTE: adain stat-matching (official optional post-filter) was tested and DRIFTS IDENTITY (masculinized the
# "female" anchor) — disabled. Set ADAIN=1 to re-enable if you see latent-stat artifacts on other prompts.
ADAIN = bool(int(os.environ.get("ADAIN", "0")))
tu = time.time()
ups.to("cuda")
with torch.no_grad():
    up = ups(s1.to("cuda", DT))
if ADAIN:
    up = adain(up, torch.nn.functional.interpolate(s1.to("cuda", DT), size=up.shape[-3:], mode="nearest"))
ups.to("cpu"); torch.cuda.empty_cache()
print(f"[upsample] -> {tuple(up.shape)} in {time.time()-tu:.1f}s", flush=True)

# ---- STAGE 2: full-res refine from the upscaled latents ----
t2 = time.time()
out = pipe(width=W, height=H, num_frames=F, num_inference_steps=3, sigmas=STAGE_2_SIGMAS_3,
           guidance_scale=1.0, frame_rate=25, conditions=[], latents=up.to(DT),
           generator=torch.Generator("cpu").manual_seed(SEED), **EMB_KW)
imgs = out.frames[0]
imgs = [f if isinstance(f, Image.Image) else Image.fromarray((np.asarray(f).clip(0,1)*255).astype("uint8")) for f in imgs]
print(f"[stage2] {W}x{H} refine+decode in {time.time()-t2:.1f}s", flush=True)

os.makedirs(OUTDIR, exist_ok=True)
for i, im in enumerate(imgs): im.save(f"{OUTDIR}/f{i:03d}.png")
a = np.asarray(imgs[0].convert("L")).astype(float); rc = float(np.corrcoef(a[:-1].ravel(), a[1:].ravel())[0,1])
print(f"[TWO-STAGE] {W}x{H} {len(imgs)}f | total {time.time()-t:.1f}s | row-corr={rc:.3f} {'REAL' if rc>0.9 else 'BAD'}", flush=True)
