"""RESIDENT + temporal-windowed SDNQ int4 LTX-2.3: each component pinned to its own card (no CPU-offload
shuffle, gemma encodes on-GPU in seconds not 230s on CPU). VAE co-located with the transformer so the
condition-latent path stays on one device; gemma + connectors on other cards (x1-cheap: only ~1MB text
conditioning crosses). Windowed for arbitrary length on the transformer card. Args: W H TOTAL WIN OVL STEPS"""
import os, sys, time, glob, inspect, gc
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
WIN   = int(sys.argv[4]) if len(sys.argv) > 4 else 41
OVL   = int(sys.argv[5]) if len(sys.argv) > 5 else 9
STEPS = int(sys.argv[6]) if len(sys.argv) > 6 else 8
PATH = sorted(glob.glob("/root/.cache/huggingface/hub/models--OzzyGT--LTX-2.3-Distilled-1.1-sdnq-dynamic-int4/snapshots/*"))[0]
POS = ("Photorealistic broadcast video of a professional female news anchor speaking to camera, "
       "natural realistic skin with visible pores, soft studio lighting, documentary realism.")
NEG = "cartoon, cgi, 3d render, plastic skin, waxy, blurry, distorted, extra people"
ng = torch.cuda.device_count()

t = time.time()
base = DiffusionPipeline.from_pretrained(PATH, torch_dtype=DT)
base.transformer = apply_sdnq_options_to_model(base.transformer, dtype=DT, use_quantized_matmul=True)
pipe = LTX2ConditionPipeline(**{k: v for k, v in base.components.items()
                                if k in set(inspect.signature(LTX2ConditionPipeline.__init__).parameters)})
try: pipe.vae.enable_tiling()
except Exception: pass

# manual placement: transformer ALONE on card0 (needs the full card for activations); gemma/connectors/vae
# each on their own card. VAE off card0 -> we move its encode outputs back to cuda:0 (below) so the
# condition-latent path lands on the exec device.
gemma_c = 1 if ng > 1 else 0
conn_c  = 2 if ng > 2 else gemma_c
# vae_c=0 co-locates the video VAE with the transformer (CORRECT output, forces small windows). Moving it to
# its own card (vae_c=3) gives big windows + ~4min/5s BUT the diffusers LTX VAE has device-coupled internals
# beyond latents_mean/std -> decode returns black. Left at 0 (correct) until the pipeline decode is patched.
vae_c   = 0
# only the 1.4GB VIDEO vae moves off card0 (frees DiT headroom); audio_vae+vocoder are tiny (~0.4GB) and
# stay co-located on card0 so the audio path never crosses devices.
place = {"transformer": 0, "text_encoder": gemma_c, "connectors": conn_c,
         "vae": vae_c, "audio_vae": 0, "vocoder": 0}
for nm, idx in place.items():
    c = getattr(pipe, nm, None)
    if c is not None and hasattr(c, "to"):
        c.to(f"cuda:{idx}")
# The pipeline denormalizes video latents with vae.latents_mean/std at PIPELINE level (on the exec device,
# cuda:0) — but those buffers moved to the VAE's card. Pin them back to cuda:0 so the denorm op matches.
import torch as _t
for b in ("latents_mean", "latents_std"):
    if hasattr(pipe.vae, b) and _t.is_tensor(getattr(pipe.vae, b)):
        setattr(pipe.vae, b, getattr(pipe.vae, b).to("cuda:0"))
# VAE lives on vae_c but the pipeline concatenates its encoded condition latents with noise on cuda:0.
# Wrap encode/decode so their tensor OUTPUTS land on the exec device (cuda:0).
def _out_to(mod, dev):
    _orig = mod.forward
    def fwd(*a, **k):
        r = _orig(*a, **k)
        def mv(x):
            if torch.is_tensor(x): return x.to(dev)
            if hasattr(x, "sample") and hasattr(x, "mean"):  # a Distribution: move its params
                for attr in ("mean", "std", "logvar", "var", "parameters"):
                    if hasattr(x, attr) and torch.is_tensor(getattr(x, attr)): setattr(x, attr, getattr(x, attr).to(dev))
                return x
            if hasattr(x, "latent_dist"): x.latent_dist = mv(x.latent_dist); return x
            if hasattr(x, "sample") and torch.is_tensor(x.sample): x.sample = x.sample.to(dev); return x
            return x
        return mv(r)
    mod.forward = fwd
EXEC = torch.device("cuda:0")
def align(module):
    dev = next(module.parameters()).device
    orig = module.forward
    def fwd(*a, **k):
        a = [x.to(dev) if torch.is_tensor(x) else x for x in a]
        k = {kk: (vv.to(dev) if torch.is_tensor(vv) else vv) for kk, vv in k.items()}
        return orig(*a, **k)
    module.forward = fwd
for nm in ("transformer", "text_encoder", "connectors"):   # these run via .forward()
    c = getattr(pipe, nm, None)
    if c is not None and hasattr(c, "parameters"):
        try: align(c)
        except StopIteration: pass
# VAE is called via .encode()/.decode() (not .forward()); wrap those: inputs -> VAE card, outputs -> cuda:0
def _mv(x, dev):
    if torch.is_tensor(x): return x.to(dev)
    if hasattr(x, "latent_dist"):
        d = x.latent_dist
        for attr in ("mean", "std", "logvar", "var"):
            if hasattr(d, attr) and torch.is_tensor(getattr(d, attr)): setattr(d, attr, getattr(d, attr).to(dev))
        if hasattr(d, "parameters") and torch.is_tensor(d.parameters): d.parameters = d.parameters.to(dev)
        return x
    if hasattr(x, "sample") and torch.is_tensor(x.sample): x.sample = x.sample.to(dev); return x
    return x
def wrap_method(mod, name, in_dev, out_dev):
    orig = getattr(mod, name)
    def m(*a, **k):
        a = [x.to(in_dev) if torch.is_tensor(x) else x for x in a]
        k = {kk: (vv.to(in_dev) if torch.is_tensor(vv) else vv) for kk, vv in k.items()}
        return _mv(orig(*a, **k), out_dev)
    setattr(mod, name, m)
_vdev = torch.device(f"cuda:{vae_c}")
for meth in ("encode", "decode"):
    if hasattr(pipe.vae, meth): wrap_method(pipe.vae, meth, _vdev, EXEC)
pipe.__class__._execution_device = property(lambda self: EXEC)
print(f"[t={time.time()-t:.1f}] resident pipeline placed across {ng} cards (transformer+vae=0, gemma={gemma_c}, connectors={conn_c})", flush=True)

def gen(num, conds, run):
    kw = dict(width=W, height=H, num_frames=num, num_inference_steps=STEPS, guidance_scale=1.0,
              frame_rate=25, prompt=POS, negative_prompt=NEG, conditions=conds,
              generator=torch.Generator("cpu").manual_seed(42 + run))
    out = pipe(**kw)
    fr = out.frames[0] if hasattr(out, "frames") else out[0]
    return [f if isinstance(f, Image.Image) else Image.fromarray((np.asarray(f).clip(0,1)*255).astype("uint8")) for f in fr]

all_frames, prev_tail, w = [], None, 0
while len(all_frames) < TOTAL:
    conds = [] if prev_tail is None else [LTX2VideoCondition(frames=prev_tail, index=0, strength=1.0)]
    t = time.time()
    fr = gen(WIN, conds, w)
    all_frames.extend(fr if prev_tail is None else fr[OVL:])
    prev_tail = fr[-OVL:]
    print(f"[window {w}] {len(fr)}f in {time.time()-t:.1f}s -> total {len(all_frames)}", flush=True)
    gc.collect(); torch.cuda.empty_cache()
    w += 1
    if w > 12: break

all_frames = all_frames[:TOTAL]
os.makedirs("/root/ComfyUI/output/sdnq_res", exist_ok=True)
for i, im in enumerate(all_frames): im.save(f"/root/ComfyUI/output/sdnq_res/f{i:03d}.png")
a = np.asarray(all_frames[0]); print(f"SAVED {len(all_frames)} frames | frame0 mean={a.mean():.1f}", flush=True)
