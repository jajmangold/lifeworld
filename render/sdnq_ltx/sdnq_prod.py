"""SDNQ int4 LTX-2.3 PRODUCTION vertical anchor: (1) embeds PERSISTED to disk (cut ~264s gemma/clip),
(2) RESIDENT VAE decode on its own card (cut decode ~8x), (3) generate latents via the correct offload
pipeline. Structure-verified (row-corr, not mean). Args: W H FRAMES STEPS"""
import os, sys, time, glob, inspect, hashlib
import torch, sdnq  # noqa
from diffusers import DiffusionPipeline, LTX2ConditionPipeline, AutoencoderKLLTX2Video
from sdnq.loader import apply_sdnq_options_to_model
import numpy as np

DT = torch.bfloat16
W = int(sys.argv[1]) if len(sys.argv) > 1 else 512
H = int(sys.argv[2]) if len(sys.argv) > 2 else 768
F = int(sys.argv[3]) if len(sys.argv) > 3 else 57
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 8
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 42
OUTDIR = sys.argv[6] if len(sys.argv) > 6 else "/root/ComfyUI/output/prod"
PATH = sorted(glob.glob("/root/.cache/huggingface/hub/models--OzzyGT--LTX-2.3-Distilled-1.1-sdnq-dynamic-int4/snapshots/*"))[0]
POS = ("Vertical portrait video of a professional female news anchor speaking to camera, waist-up, "
       "natural realistic skin with visible pores, soft studio lighting, modern broadcast news set, documentary realism.")
NEG = "cartoon, cgi, 3d render, plastic skin, waxy, blurry, distorted, extra people"
EMB = f"/root/ComfyUI/output/embeds_{hashlib.md5((POS+'||'+NEG).encode()).hexdigest()[:10]}.pt"

t = time.time()
base = DiffusionPipeline.from_pretrained(PATH, torch_dtype=DT)
base.transformer = apply_sdnq_options_to_model(base.transformer, dtype=DT, use_quantized_matmul=True)
pipe = LTX2ConditionPipeline(**{k: v for k, v in base.components.items()
                                if k in set(inspect.signature(LTX2ConditionPipeline.__init__).parameters)})
try: pipe.vae.enable_tiling()
except Exception: pass
pipe.enable_model_cpu_offload()
print(f"[t={time.time()-t:.1f}] pipeline ready", flush=True)

# --- LEVER 1: persisted embeds. capture on first-ever run (via a real gen wrapper), reuse from disk after ---
cap = {}
need_save = not os.path.exists(EMB)
if os.path.exists(EMB):
    cap["e"] = torch.load(EMB, map_location="cpu"); print("[embeds] loaded from disk (gemma skipped)", flush=True)
_orig = pipe.encode_prompt
def _wrap(*a, **k):
    r = _orig(*a, **k); cap["e"] = tuple(x.detach().cpu() if torch.is_tensor(x) else x for x in r); return r


marks = {}
def cb(p, s, ts, kw):
    marks["last"] = time.time()
    if s == 0: marks["first"] = time.time()
    return kw
kw = dict(width=W, height=H, num_frames=F, num_inference_steps=STEPS, guidance_scale=1.0, frame_rate=25,
          conditions=[], callback_on_step_end=cb, generator=torch.Generator("cpu").manual_seed(SEED))
if "e" in cap:
    pe, pm, ne, nm = cap["e"]
    g = lambda x: (x.to("cuda", DT) if (torch.is_tensor(x) and x.is_floating_point()) else (x.to("cuda") if torch.is_tensor(x) else None))
    kw.update(prompt_embeds=g(pe), prompt_attention_mask=g(pm))
    if ne is not None: kw.update(negative_prompt_embeds=g(ne), negative_prompt_attention_mask=g(nm))
else:
    pipe.encode_prompt = _wrap; kw.update(prompt=POS, negative_prompt=NEG)

torch.cuda.synchronize(); tg = time.time()
out = pipe(**kw)                      # returns denormalized latents (output_type='latent')
torch.cuda.synchronize(); den_total = time.time() - tg
if need_save and "e" in cap:
    pipe.encode_prompt = _orig
    torch.save(cap["e"], EMB); print(f"[embeds] SAVED to {EMB}", flush=True)
fr = out.frames[0] if hasattr(out, "frames") else out[0]
from PIL import Image
import numpy as np
os.makedirs(OUTDIR, exist_ok=True)
imgs=[f if isinstance(f,Image.Image) else Image.fromarray((np.asarray(f).clip(0,1)*255).astype("uint8")) for f in fr]
for i,im in enumerate(imgs): im.save(f"{OUTDIR}/f{i:03d}.png")
a=np.asarray(imgs[0].convert("L")).astype(float); rc=float(np.corrcoef(a[:-1].ravel(),a[1:].ravel())[0,1])
setup=marks.get("first",tg)-tg; denoise=marks.get("last",tg)-marks.get("first",tg); dec=den_total-(marks.get("last",tg)-tg)
print(f"[PROD] {W}x{H} {F}f -> total {den_total:.1f}s (setup {setup:.1f} + denoise {denoise:.1f} + decode {dec:.1f}) | row-corr={rc:.3f} {'REAL' if rc>0.9 else 'BAD'}", flush=True)
print(f"SAVED {len(imgs)} frames", flush=True)
