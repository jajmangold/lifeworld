"""Temporal-windowed SDNQ int4 LTX-2.3 on ONE sm_70 card -> arbitrarily long FULL-RES clips.
Each window fits the card; every window after the first is conditioned on the previous window's tail
frames (LTX2VideoCondition) for continuity, so the seams stay coherent. Embeds cached once (task #13).
Args: WIDTH HEIGHT TOTAL_FRAMES WIN OVERLAP STEPS"""
import os, sys, time, glob
import torch, sdnq  # noqa
from diffusers import DiffusionPipeline, LTX2ConditionPipeline
from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
from sdnq.loader import apply_sdnq_options_to_model
from PIL import Image
import numpy as np

DT = torch.bfloat16
W  = int(sys.argv[1]) if len(sys.argv) > 1 else 768
H  = int(sys.argv[2]) if len(sys.argv) > 2 else 448
TOTAL = int(sys.argv[3]) if len(sys.argv) > 3 else 121
WIN   = int(sys.argv[4]) if len(sys.argv) > 4 else 57   # 8n+1, fits one card
OVL   = int(sys.argv[5]) if len(sys.argv) > 5 else 9    # frames of overlap/condition
STEPS = int(sys.argv[6]) if len(sys.argv) > 6 else 8
PATH = sorted(glob.glob("/root/.cache/huggingface/hub/models--OzzyGT--LTX-2.3-Distilled-1.1-sdnq-dynamic-int4/snapshots/*"))[0]
POS = ("Vertical portrait video of a professional female news anchor speaking to camera, waist-up, "
       "natural realistic skin with visible pores, soft studio lighting, modern broadcast news set, documentary realism.")
NEG = "cartoon, cgi, 3d render, plastic skin, waxy, blurry, distorted, extra people"

t = time.time()
base = DiffusionPipeline.from_pretrained(PATH, torch_dtype=DT)
base.transformer = apply_sdnq_options_to_model(base.transformer, dtype=DT, use_quantized_matmul=True)
import inspect
_initp = set(inspect.signature(LTX2ConditionPipeline.__init__).parameters)
pipe = LTX2ConditionPipeline(**{k: v for k, v in base.components.items() if k in _initp})  # reuse SDNQ components, no recast
try: pipe.vae.enable_tiling()
except Exception: pass
pipe.enable_model_cpu_offload()
print(f"[t={time.time()-t:.1f}] condition pipeline ready", flush=True)

# cache prompt embeds on the first window (wrap encode_prompt), reuse after -> no gemma re-encode per window
cap = {}
_orig = pipe.encode_prompt
def _wrap(*a, **k):
    r = _orig(*a, **k); cap["e"] = tuple(x.detach().cpu() if torch.is_tensor(x) else x for x in r); return r

def gen(num, conds, run):
    kw = dict(width=W, height=H, num_frames=num, num_inference_steps=STEPS, guidance_scale=1.0,
              frame_rate=25, conditions=conds, generator=torch.Generator("cpu").manual_seed(42 + run))
    if "e" in cap:
        pe, pm, ne, nm = cap["e"]
        g = lambda x: (x.to("cuda", DT) if (torch.is_tensor(x) and x.is_floating_point()) else (x.to("cuda") if torch.is_tensor(x) else None))
        kw.update(prompt_embeds=g(pe), prompt_attention_mask=g(pm))
        if ne is not None: kw.update(negative_prompt_embeds=g(ne), negative_prompt_attention_mask=g(nm))
    else:
        pipe.encode_prompt = _wrap
        kw.update(prompt=POS, negative_prompt=NEG)
    out = pipe(**kw)
    if "e" not in cap: pipe.encode_prompt = _orig
    fr = out.frames[0] if hasattr(out, "frames") else out[0]
    return [f if isinstance(f, Image.Image) else Image.fromarray((np.asarray(f).clip(0,1)*255).astype("uint8")) for f in fr]

all_frames, prev_tail, w = [], None, 0
while len(all_frames) < TOTAL:
    conds = [] if prev_tail is None else [LTX2VideoCondition(frames=prev_tail, index=0, strength=1.0)]
    t = time.time()
    fr = gen(WIN, conds, w)
    add = fr if prev_tail is None else fr[OVL:]        # drop the overlap we already have
    all_frames.extend(add)
    prev_tail = fr[-OVL:]
    print(f"[window {w}] {len(fr)}f in {time.time()-t:.1f}s -> total {len(all_frames)}", flush=True)
    import gc; gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    w += 1
    if w > 12: break

all_frames = all_frames[:TOTAL]
os.makedirs("/root/ComfyUI/output/sdnq_long", exist_ok=True)
for i, im in enumerate(all_frames): im.save(f"/root/ComfyUI/output/sdnq_long/f{i:03d}.png")
a = np.asarray(all_frames[0]); print(f"SAVED {len(all_frames)} frames | frame0 mean={a.mean():.1f}", flush=True)
