"""RESIDENT DENOISE DAEMON — the Volta farm engine. Transformer RESIDENT (no offload), gemma-less, connectors
cached+freed, LOW-RES denoise-only -> saves raw LTX latents (.pt) per job. ~23.6s/clip warm, VmRSS ~2GB so
all 16 cards run (no RAM cap). Decode + FlashVSR happen downstream on rtx0. One daemon per CUDA_VISIBLE_DEVICES.
Job = JSON in JOBS/: {w,h,frames,steps,seed,out}. Writes out/<job>.pt (latents) + out/<job>.meta.json. Args: [JOBS_DIR]"""
import os, sys, time, glob, inspect, hashlib, json, gc, signal
import torch, sdnq  # noqa
from diffusers import DiffusionPipeline, LTX2ConditionPipeline
from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
from PIL import Image
from sdnq.loader import apply_sdnq_options_to_model
import numpy as np

DT = torch.bfloat16
DEV = "cuda:0"
JOBS = sys.argv[1] if len(sys.argv) > 1 else "/root/ComfyUI/output/resdjobs"
DONE = os.path.join(JOBS, "done"); os.makedirs(DONE, exist_ok=True)
PATH = sorted(glob.glob("/root/.cache/huggingface/hub/models--OzzyGT--LTX-2.3-Distilled-1.1-sdnq-dynamic-int4/snapshots/*"))[0]
POS = ("Medium shot of a distinguished male television news anchor in his late fifties, with neatly styled "
    "silver-gray side-parted hair and a clean-shaven, lightly tanned face with friendly blue eyes, seated at a "
    "modern broadcast desk with his hands resting on the desk. He wears a tailored navy suit over a light blue "
    "dress shirt and a blue patterned silk tie. He speaks directly to the camera with calm, articulate, authoritative "
    "expressions, his lips moving naturally and his gaze steady. Even flat studio lighting against a solid uniform "
    "bright chroma-green screen background. The footage looks like real broadcast television: photorealistic, with "
    "natural skin texture, visible pores and fine detail. The camera holds a steady, locked-off shot.")
NEG = "cartoon, cgi, 3d render, plastic skin, waxy, blurry, distorted, extra people, on-screen text, watermark, logo"
_h = hashlib.md5((POS+'||'+NEG).encode()).hexdigest()[:10]
EMB = f"/root/ComfyUI/output/embeds_{_h}.pt"; CONN = f"/root/ComfyUI/output/conn_{_h}.pt"
DISTILLED_SIGMAS_8 = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875]
_run = {"go": True}
for s in (signal.SIGTERM, signal.SIGINT): signal.signal(s, lambda *a: _run.update(go=False))
def rss_gb():
    for ln in open("/proc/self/status"):
        if ln.startswith("VmRSS"): return int(ln.split()[1]) / 1e6
    return 0.0

t = time.time()
assert os.path.exists(EMB) and os.path.exists(CONN), "run sdnq_gen once to build caches"
base = DiffusionPipeline.from_pretrained(PATH, torch_dtype=DT, text_encoder=None)
base.transformer = apply_sdnq_options_to_model(base.transformer, dtype=DT, use_quantized_matmul=True)
comp = {k: v for k, v in base.components.items() if k in set(inspect.signature(LTX2ConditionPipeline.__init__).parameters)}
comp["text_encoder"] = None
pipe = LTX2ConditionPipeline(**comp)
_ct = tuple(torch.load(CONN, map_location="cpu"))
pipe.connectors = lambda pe_, pm_, padding_side="left": tuple(x.to(pe_.device) for x in _ct)
# RESIDENT transformer + VAE (VAE encodes frame-0 conditioning images); audio parked on CPU.
pipe.transformer.to(DEV)
pipe.vae.to(DEV)
try: pipe.vae.enable_tiling()
except Exception: pass
for n in ("audio_vae", "vocoder"):
    m = getattr(pipe, n, None)
    if m is not None: m.to("cpu")
pipe.__class__._execution_device = property(lambda self: torch.device(DEV))
del base; gc.collect(); torch.cuda.synchronize()
pe, pm, ne, nm = torch.load(EMB, map_location="cpu")
g = lambda x: (x.to(DEV, DT) if (torch.is_tensor(x) and x.is_floating_point()) else (x.to(DEV) if torch.is_tensor(x) else None))
EMB_KW = dict(prompt_embeds=g(pe), prompt_attention_mask=g(pm))
if ne is not None: EMB_KW.update(negative_prompt_embeds=g(ne), negative_prompt_attention_mask=g(nm))
dev = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
print(f"[resd dev={dev} t={time.time()-t:.1f}] transformer RESIDENT | VRAM {torch.cuda.memory_allocated()/1e9:.2f}GB | VmRSS {rss_gb():.1f}GB | watching {JOBS}", flush=True)

def denoise(job):
    w, h = int(job["w"]), int(job["h"]); nf = int(job["frames"]); steps = int(job.get("steps", 8)); seed = int(job.get("seed", 42))
    conds = []
    cp = job.get("cond")                       # optional frame-0 conditioning image (e.g. a green library seed)
    if cp and os.path.exists(cp):
        img = Image.open(cp).convert("RGB").resize((w, h))
        conds.append(LTX2VideoCondition(frames=img, index=0, strength=float(job.get("cond_strength", 0.6))))
        # multi-keyframe: pin the LAST frame too -> locks framing (kills zoom drift), motion stays in the middle
        if job.get("cond_last"):
            li = job.get("cond_last")
            lp = li if isinstance(li, str) and os.path.exists(li) else cp
            limg = img if lp == cp else Image.open(lp).convert("RGB").resize((w, h))
            conds.append(LTX2VideoCondition(frames=limg, index=nf - 1, strength=float(job.get("cond_strength_last", 0.8))))
        # optional mid keyframe (allows a natural arc instead of a rigid boomerang)
        mp = job.get("cond_mid")
        if mp and os.path.exists(mp):
            conds.append(LTX2VideoCondition(frames=Image.open(mp).convert("RGB").resize((w, h)),
                                            index=nf // 2, strength=float(job.get("cond_strength_mid", 0.5))))
    kw = dict(width=w, height=h, num_frames=nf, num_inference_steps=steps, guidance_scale=1.0, frame_rate=25,
              conditions=conds, output_type="latent", generator=torch.Generator("cpu").manual_seed(seed), **EMB_KW)
    if steps == 8: kw["sigmas"] = DISTILLED_SIGMAS_8
    return pipe(**kw).frames   # raw LTX latents [B,C,f,H,W]

while _run["go"]:
    todo = sorted(glob.glob(os.path.join(JOBS, "*.json")))
    if not todo: time.sleep(0.5); continue
    jf = todo[0]
    try: job = json.load(open(jf))
    except Exception as e: os.rename(jf, jf + ".bad"); print(f"[bad {jf}] {e}", flush=True); continue
    outdir = job.get("out", os.path.join(JOBS, "out")); os.makedirs(outdir, exist_ok=True)
    stem = os.path.basename(jf)[:-5]; ts = time.time()
    try:
        lat = denoise(job)
        torch.save(lat.cpu(), os.path.join(outdir, f"{stem}.pt"))
        json.dump({"w": job["w"], "h": job["h"], "frames": job["frames"], "seed": job.get("seed", 42),
                   "shape": list(lat.shape)}, open(os.path.join(outdir, f"{stem}.meta.json"), "w"))
        print(f"[job {stem}] {job['w']}x{job['h']} {job['frames']}f -> latents {tuple(lat.shape)} {time.time()-ts:.1f}s -> {outdir}/{stem}.pt", flush=True)
    except Exception as e:
        print(f"[job {stem}] FAILED {repr(e)[:160]}", flush=True)
    finally:
        gc.collect(); torch.cuda.empty_cache()
        try: os.rename(jf, os.path.join(DONE, os.path.basename(jf)))
        except Exception: pass
print("[resd] stopped", flush=True)
