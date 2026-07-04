"""SDNQ int4 LTX-2.3 PRODUCTION vertical anchor on the PROVEN offload path, with two fixed-prompt caches:
(1) LEVER 1 embeds PERSISTED to disk (skip gemma, ~-115s/clip), (2) LEVER 1b connector-output CACHED +
the 6.35GB connectors module BYPASSED (it runs once on the fixed prompt -> fixed output; under offload it
would otherwise be streamed over PCIe x1 every clip). Structure-verified (row-corr, not mean). Args: W H FRAMES STEPS [SEED] [OUTDIR]"""
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
# The distilled model was trained on a HAND-TUNED sigma trajectory, NOT linspace. Feeding linspace at low
# steps caused the watermark-ghost artifact. These are the official Lightricks DISTILLED_SIGMAS (8 steps);
# using them makes 8 steps clean (no watermark) — the correct fix, faster than bumping steps.
DISTILLED_SIGMAS_8 = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875]
SEED = int(sys.argv[5]) if len(sys.argv) > 5 else 42
OUTDIR = sys.argv[6] if len(sys.argv) > 6 else "/root/ComfyUI/output/prod"
PATH = sorted(glob.glob("/root/.cache/huggingface/hub/models--OzzyGT--LTX-2.3-Distilled-1.1-sdnq-dynamic-int4/snapshots/*"))[0]
# parametrized: prompt via env PROMPT/NEG; STG via env STG_SCALE (0=off). Rich character-locked default prompt
# following the LTX-2.3 guide (4-8 sentences, explicit character, NO text/signage since the model can't render it).
_DEF_POS = ("Medium close-up vertical portrait of a distinguished male television news anchor in his late fifties, "
    "with neatly styled silver-gray side-parted hair and a clean-shaven, lightly tanned face with friendly blue eyes, "
    "seated at a modern broadcast desk. He wears a tailored navy suit over a light blue dress shirt and a blue "
    "patterned silk tie. He speaks directly to the camera with calm, articulate, authoritative expressions, his lips "
    "moving naturally and his gaze steady. Soft key lighting from the front-left illuminates his face against a gently "
    "blurred dark blue studio background. The footage looks like real broadcast television: photorealistic, with "
    "natural skin texture, visible pores and fine detail. The camera holds a steady, locked-off shot.")
POS = os.environ.get("PROMPT", _DEF_POS)
NEG = os.environ.get("NEG", "cartoon, cgi, 3d render, plastic skin, waxy, blurry, distorted, extra people, on-screen text, watermark, logo")
STG_SCALE = float(os.environ.get("STG_SCALE", "0"))
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

# --- LEVER 1b: bypass the 6.35GB connectors with the cached fixed-prompt output (no per-clip x1 stream) ---
CONN = f"/root/ComfyUI/output/conn_{hashlib.md5((POS+'||'+NEG).encode()).hexdigest()[:10]}.pt"
if os.path.exists(CONN):
    _ctrip = tuple(x for x in torch.load(CONN, map_location="cpu"))
    def _conn_stub(prompt_embeds, prompt_attention_mask, padding_side="left"):
        d = prompt_embeds.device
        return tuple(x.to(d) for x in _ctrip)   # fixed prompt -> fixed output; ignore inputs
    pipe.connectors = _conn_stub
    print("[connectors] BYPASSED via cache (6.35GB module never streamed)", flush=True)
else:
    print("[connectors] no cache -> running real connectors (will cache-miss stream)", flush=True)

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
if STEPS == 8: kw["sigmas"] = DISTILLED_SIGMAS_8   # correct distilled schedule (else fall back to linspace)
if STG_SCALE > 0:                                  # spatio-temporal guidance: coherence+quality, +1 fwd/step
    kw.update(stg_scale=STG_SCALE, spatio_temporal_guidance_blocks=[28])  # [28] = LTX-2.3 recommended block
    print(f"[STG] enabled scale={STG_SCALE} blocks=[28]", flush=True)
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
# also cache the connectors output for the resident daemons (they require conn_<hash>.pt too)
if not os.path.exists(CONN) and "e" in cap:
    pe2, pm2 = cap["e"][0], cap["e"][1]
    gg = lambda x: (x.to("cuda", DT) if (torch.is_tensor(x) and x.is_floating_point()) else (x.to("cuda") if torch.is_tensor(x) else None))
    with torch.no_grad():
        _ct = tuple(x.detach().cpu() for x in pipe.connectors(gg(pe2), gg(pm2), padding_side="left"))
    torch.save(_ct, CONN); print(f"[connectors] SAVED to {CONN}", flush=True)
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
