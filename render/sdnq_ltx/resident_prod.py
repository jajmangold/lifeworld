"""TRANSFORMER-RESIDENT SDNQ int4 LTX-2.3 on ONE sm_70 card. No cpu_offload, no wrappers, single device ->
no corruption. Transformer(11.56GB)+connectors stay resident on cuda:0 FOREVER (never re-streamed over x1 =
the big win). The bf16 VAE (moves cleanly, unlike SDNQ) shuttles: on CPU during denoise (frees activation
headroom), moved to cuda:0 for the once-per-clip decode, moved back. gemma excluded (cached embeds).

Design lets a daemon serve clips at ~denoise+decode only (~80-100s) vs 296s offloaded. Args: W H F STEPS [SEED] [REPEAT]"""
import os, sys, time, hashlib, inspect, glob, gc
import torch, sdnq  # noqa
from diffusers import DiffusionPipeline, LTX2ConditionPipeline
from sdnq.loader import apply_sdnq_options_to_model
from PIL import Image
import numpy as np

DT = torch.bfloat16
W  = int(sys.argv[1]) if len(sys.argv) > 1 else 512
H  = int(sys.argv[2]) if len(sys.argv) > 2 else 768
F  = int(sys.argv[3]) if len(sys.argv) > 3 else 41
STEPS  = int(sys.argv[4]) if len(sys.argv) > 4 else 8
SEED   = int(sys.argv[5]) if len(sys.argv) > 5 else 42
REPEAT = int(sys.argv[6]) if len(sys.argv) > 6 else 2   # how many clips to time (amortization proof)
PATH = sorted(glob.glob("/root/.cache/huggingface/hub/models--OzzyGT--LTX-2.3-Distilled-1.1-sdnq-dynamic-int4/snapshots/*"))[0]
POS = ("Vertical portrait video of a professional female news anchor speaking to camera, waist-up, "
       "natural realistic skin with visible pores, soft studio lighting, modern broadcast news set, documentary realism.")
NEG = "cartoon, cgi, 3d render, plastic skin, waxy, blurry, distorted, extra people"
EMB = f"/root/ComfyUI/output/embeds_{hashlib.md5((POS+'||'+NEG).encode()).hexdigest()[:10]}.pt"
DEV = "cuda:0"

t = time.time()
base = DiffusionPipeline.from_pretrained(PATH, torch_dtype=DT)
base.transformer = apply_sdnq_options_to_model(base.transformer, dtype=DT, use_quantized_matmul=True)
comp = {k: v for k, v in base.components.items()
        if k in set(inspect.signature(LTX2ConditionPipeline.__init__).parameters)}
comp["text_encoder"] = None
pipe = LTX2ConditionPipeline(**comp)
try: pipe.vae.enable_tiling()
except Exception as e: print("tiling warn", e)
del base; gc.collect(); torch.cuda.empty_cache()
if not os.path.exists(EMB): print("NO CACHED EMBEDS"); sys.exit(1)
pe, pm, ne, nm = torch.load(EMB, map_location="cpu")
g = lambda x: (x.to(DEV, DT) if (torch.is_tensor(x) and x.is_floating_point()) else (x.to(DEV) if torch.is_tensor(x) else None))

# --- connector output is FIXED for a fixed prompt (runs once, line 1258). The connectors module is 6.35GB
#     and can't co-reside with the 11.56GB transformer. So precompute the connector triple ONCE (transformer
#     still on CPU), cache it, free the connectors, then load the transformer resident and BYPASS connectors. ---
CONN = f"/root/ComfyUI/output/conn_{hashlib.md5((POS+'||'+NEG).encode()).hexdigest()[:10]}.pt"
# audio_vae/vocoder are only used to DECODE audio (non-latent branch). With output_type='latent' they are
# never called -> keep them on CPU (config still readable) to free VRAM for denoise activations.
for n in ("audio_vae", "vocoder"):
    m = getattr(pipe, n, None)
    if m is not None: m.to("cpu")
pipe.vae.to("cpu")
if os.path.exists(CONN):
    cpe, cape, cam = (x.to(DEV) for x in torch.load(CONN, map_location="cpu"))
    print(f"[t={time.time()-t:.1f}] connector triple loaded from cache (connectors NEVER loaded on GPU)", flush=True)
else:
    pipe.connectors.to(DEV)
    with torch.no_grad():
        cpe, cape, cam = pipe.connectors(g(pe), g(pm), padding_side="left")
    cpe, cape, cam = cpe.detach(), cape.detach(), cam.detach()
    torch.save(tuple(x.cpu() for x in (cpe, cape, cam)), CONN)
    pipe.connectors.to("cpu"); gc.collect(); torch.cuda.empty_cache()
    print(f"[t={time.time()-t:.1f}] connector triple computed+cached; connectors freed from GPU", flush=True)

def _conn_stub(prompt_embeds, prompt_attention_mask, padding_side="left"):
    return cpe, cape, cam          # fixed prompt -> fixed output; ignore inputs
pipe.connectors = _conn_stub       # bypass the 6.35GB module entirely

# RESIDENT: transformer stays on cuda:0 forever (never re-streamed over x1).
pipe.transformer.to(DEV)
# _execution_device would otherwise pick CPU (vae/audio_vae parked there) and run denoise on CPU. The only
# ACTIVE module is the resident transformer on cuda:0, so pin exec there. Correct: single active device.
pipe.__class__._execution_device = property(lambda self: torch.device(DEV))
torch.cuda.synchronize()
print(f"[t={time.time()-t:.1f}] resident: transformer on {DEV}; connectors bypassed; vae parked CPU; "
      f"VRAM {torch.cuda.memory_allocated()/1e9:.2f}GB; exec={pipe._execution_device}", flush=True)

def run(seed):
    marks = {}
    def cb(p, s, ts, kw):
        now = time.time()
        if s == 0: marks["first"] = now
        marks["last"] = now
        return kw
    gc.collect(); torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.time()
    # 1) DENOISE only -> latents (VAE on CPU, not touched). output_type='latent' skips in-pipe decode.
    lat = pipe(prompt_embeds=g(pe), prompt_attention_mask=g(pm),
               negative_prompt_embeds=g(ne), negative_prompt_attention_mask=g(nm),
               width=W, height=H, num_frames=F, num_inference_steps=STEPS, guidance_scale=1.0,
               frame_rate=25, conditions=[], generator=torch.Generator("cpu").manual_seed(seed),
               output_type="latent", callback_on_step_end=cb).frames  # denormalized latents [B,C,F,H,W]
    torch.cuda.synchronize(); t_den = time.time()
    # 2) shuttle VAE on-card (activations now freed), decode resident, shuttle back
    gc.collect(); torch.cuda.empty_cache()
    pipe.vae.to(DEV); t_vae_on = time.time()
    with torch.no_grad():
        video = pipe.vae.decode(lat.to(pipe.vae.dtype).to(DEV), None, return_dict=False)[0]
    imgs = pipe.video_processor.postprocess_video(video, output_type="pil")[0]
    torch.cuda.synchronize(); t_dec = time.time()
    peak = torch.cuda.max_memory_allocated()/1e9
    pipe.vae.to("cpu"); gc.collect(); torch.cuda.empty_cache()
    setup = marks.get("first", t0) - t0; den = marks.get("last", t0) - marks.get("first", t0)
    a = np.asarray(imgs[0]).astype(np.float32)
    c = float(np.corrcoef(a[:-1].ravel(), a[1:].ravel())[0, 1])
    return imgs, dict(total=t_dec - t0, setup=setup, denoise=den, vae_on=t_vae_on - t_den,
                      decode=t_dec - t_vae_on, peak=peak, corr=c)

for r in range(REPEAT):
    imgs, m = run(SEED + r)
    tag = "cold" if r == 0 else f"warm{r}"
    print(f"[RESIDENT {tag}] {W}x{H} {F}f -> {m['total']:.1f}s (setup {m['setup']:.1f} + denoise {m['denoise']:.1f} "
          f"+ vae-on {m['vae_on']:.1f} + decode {m['decode']:.1f}) | peak {m['peak']:.2f}GB | "
          f"row-corr={m['corr']:.3f} {'REAL' if m['corr']>0.9 else 'GARBAGE!'}", flush=True)
    if r == REPEAT - 1:
        os.makedirs("/root/ComfyUI/output/resident_prod", exist_ok=True)
        for i, im in enumerate(imgs): im.save(f"/root/ComfyUI/output/resident_prod/f{i:03d}.png")
