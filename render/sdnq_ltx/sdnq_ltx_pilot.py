"""SDNQ int4 LTX-2.3 pilot on sm_70 (CMP 100-210, 16GB). Loads the pre-quantized dynamic-int4
diffusers pipeline, enables Triton quantized matmul (no fp16 dequant balloon), CPU-offloads the
encoder/connectors, and times a small T2V gen. Args: WIDTH HEIGHT FRAMES STEPS."""
import os, sys, time, inspect
import torch
import sdnq  # noqa: F401  (registers the SDNQ quantization backend for diffusers)
from diffusers import DiffusionPipeline
from sdnq.loader import apply_sdnq_options_to_model

import os as _os
_d=_os.environ.get("PILOT_DTYPE","fp16"); DT={"fp16":torch.float16,"bf16":torch.bfloat16,"fp32":torch.float32}[_d]
W = int(sys.argv[1]) if len(sys.argv) > 1 else 768
H = int(sys.argv[2]) if len(sys.argv) > 2 else 448
F = int(sys.argv[3]) if len(sys.argv) > 3 else 25
STEPS = int(sys.argv[4]) if len(sys.argv) > 4 else 8
PATH = open("/tmp/sdnq_model_path.txt").read().strip()

def vram(tag):
    a = torch.cuda.memory_allocated()/1e9; p = torch.cuda.max_memory_allocated()/1e9
    print(f"  VRAM[{tag}] alloc={a:.2f} peak={p:.2f} GB", flush=True)

t = time.time()
pipe = DiffusionPipeline.from_pretrained(PATH, torch_dtype=DT, local_files_only=True)
print(f"[t={time.time()-t:.1f}] pipeline loaded: {type(pipe).__name__}", flush=True)

# Enable Triton quantized matmul on the int4 transformer -> compute stays quantized, no fp16 dequant.
if torch.cuda.is_available():
    pipe.transformer = apply_sdnq_options_to_model(pipe.transformer, dtype=DT, dequantize_fp32=True, use_quantized_matmul=True)
    print("[sdnq] quantized matmul applied to transformer", flush=True)

# transformer runs on GPU; encoder/connectors/vae offload to CPU RAM (130GB) between steps.
# bound VAE decode memory for long clips (121f) via spatial+temporal tiling
for _fn in ("enable_tiling", "enable_slicing"):
    try: getattr(pipe.vae, _fn)(); print(f"[vae] {_fn}", flush=True)
    except Exception as _e: print(f"[vae] no {_fn}", flush=True)
pipe.enable_model_cpu_offload()

# introspect the pipeline call signature and pass only accepted kwargs
sig = set(inspect.signature(pipe.__call__).parameters)
print("[call params]", sorted(p for p in sig if p not in ("self",))[:24], flush=True)
POS = ("Photorealistic broadcast video of a professional female news anchor speaking to camera, "
       "natural realistic skin with visible pores, soft studio lighting, documentary realism.")
NEG = "cartoon, cgi, 3d render, plastic skin, waxy, blurry, distorted, extra people"
want = dict(prompt=POS, negative_prompt=NEG, width=W, height=H, num_frames=F,
            num_inference_steps=STEPS, guidance_scale=1.0, frame_rate=25,
            generator=torch.Generator("cpu").manual_seed(42))
kw = {k: v for k, v in want.items() if k in sig}
print("[using kwargs]", {k: kw[k] for k in kw if k not in ("prompt","negative_prompt","generator")}, flush=True)

RUNS = int(_os.environ.get("PILOT_RUNS", "1"))
if "callback_on_step_end" in sig:
    marks = {}
    def cb(pipe_, step, ts, kwargs):
        m = time.time()
        if step == 0: marks["first"] = m
        marks["last"] = m
        return kwargs
    kw["callback_on_step_end"] = cb
for run in range(RUNS):
    marks = {}
    if "callback_on_step_end" in sig:
        kw["callback_on_step_end"] = lambda p, s, ts, kwargs: (marks.__setitem__("first" if s == 0 else "x", time.time()), marks.__setitem__("last", time.time()), kwargs)[-1]
    torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t = time.time()
    out = pipe(**kw)
    torch.cuda.synchronize(); dt = time.time() - t
    tag = "COLD" if run == 0 else f"WARM{run}"
    if marks.get("first"):
        enc = marks["first"] - t; den = marks["last"] - marks["first"]; dec = (t + dt) - marks["last"]
        print(f"[SAMPLE {tag}] {dt:.1f}s = encode/setup {enc:.1f}s + denoise {den:.1f}s ({den/STEPS:.1f}s/step) + decode/post {dec:.1f}s", flush=True)
    else:
        print(f"[SAMPLE {tag}] {W}x{H} {F}f {STEPS}steps -> {dt:.1f}s", flush=True)
    vram(f"post-{tag}")

# save first frame + all frames
frames = getattr(out, "frames", None) or getattr(out, "videos", None) or out[0]
seq = frames[0] if isinstance(frames, (list, tuple)) else frames
os.makedirs("/root/ComfyUI/output/sdnq", exist_ok=True)
from PIL import Image
import numpy as np
def to_img(x):
    if isinstance(x, Image.Image): return x
    a = np.asarray(x)
    if a.dtype != np.uint8: a = (a.clip(0,1)*255).astype("uint8")
    return Image.fromarray(a)
imgs = [to_img(f) for f in seq]
imgs[0].save("/root/ComfyUI/output/sdnq/frame0.png")
for i, im in enumerate(imgs): im.save(f"/root/ComfyUI/output/sdnq/f{i:03d}.png")
print(f"SAVED {len(imgs)} frames -> /root/ComfyUI/output/sdnq/", flush=True)
