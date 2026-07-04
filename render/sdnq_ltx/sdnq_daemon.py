"""Resident SDNQ int4 LTX-2.3 daemon on ONE sm_70 card. Loads the pipeline ONCE (pays the ~146s setup a
single time) then serves clip jobs at ~150s each (denoise + decode only). This is the correct "keep it
resident" lever: we keep the *process* alive, NOT hand-pin the VAE (that broke the offload chain). The
pipeline still uses enable_model_cpu_offload for every job -> pixels stay correct (row-corr ~0.99).

Job protocol (dirt simple, farm-friendly): drop a JSON file in JOBS/, we pick it up, render, write the
mp4/frames to its "out" dir, then move the job file to JOBS/done/. JSON keys: w,h,frames,steps,seed,out,
[pos],[neg]. Poll loop; SIGTERM-clean.

Env: JOBS dir via arg1 (default /root/ComfyUI/output/jobs). CUDA_VISIBLE_DEVICES pins the card."""
import os, sys, time, json, glob, hashlib, inspect, signal
import torch, sdnq  # noqa
from diffusers import DiffusionPipeline, LTX2ConditionPipeline
from sdnq.loader import apply_sdnq_options_to_model
from PIL import Image
import numpy as np

DT = torch.bfloat16
JOBS = sys.argv[1] if len(sys.argv) > 1 else "/root/ComfyUI/output/jobs"
DONE = os.path.join(JOBS, "done")
os.makedirs(DONE, exist_ok=True)
PATH = sorted(glob.glob("/root/.cache/huggingface/hub/models--OzzyGT--LTX-2.3-Distilled-1.1-sdnq-dynamic-int4/snapshots/*"))[0]
DEF_POS = ("Vertical portrait video of a professional female news anchor speaking to camera, waist-up, "
           "natural realistic skin with visible pores, soft studio lighting, modern broadcast news set, documentary realism.")
DEF_NEG = "cartoon, cgi, 3d render, plastic skin, waxy, blurry, distorted, extra people"
EMBDIR = "/root/ComfyUI/output"

_run = {"go": True}
signal.signal(signal.SIGTERM, lambda *a: _run.update(go=False))
signal.signal(signal.SIGINT,  lambda *a: _run.update(go=False))

# ---- load pipeline ONCE ----
t = time.time()
base = DiffusionPipeline.from_pretrained(PATH, torch_dtype=DT)
base.transformer = apply_sdnq_options_to_model(base.transformer, dtype=DT, use_quantized_matmul=True)
pipe = LTX2ConditionPipeline(**{k: v for k, v in base.components.items()
                                if k in set(inspect.signature(LTX2ConditionPipeline.__init__).parameters)})
try: pipe.vae.enable_tiling()
except Exception: pass
pipe.enable_model_cpu_offload()
print(f"[daemon t={time.time()-t:.1f}] pipeline resident on {os.environ.get('CUDA_VISIBLE_DEVICES','?')}; watching {JOBS}", flush=True)

# ---- embed cache (same scheme as prod: hash of pos||neg) ----
def embeds_for(pos, neg):
    emb = os.path.join(EMBDIR, f"embeds_{hashlib.md5((pos+'||'+neg).encode()).hexdigest()[:10]}.pt")
    if os.path.exists(emb):
        return torch.load(emb, map_location="cpu"), None            # cached tuple, no capture needed
    cap = {}
    _orig = pipe.encode_prompt
    def _wrap(*a, **k):
        r = _orig(*a, **k); cap["e"] = tuple(x.detach().cpu() if torch.is_tensor(x) else x for x in r); return r
    pipe.encode_prompt = _wrap
    return None, (emb, cap, _orig, _wrap)

def gen(job):
    w, h = int(job["w"]), int(job["h"]); nf = int(job["frames"]); steps = int(job.get("steps", 8))
    seed = int(job.get("seed", 42)); pos = job.get("pos", DEF_POS); neg = job.get("neg", DEF_NEG)
    cached, cap = embeds_for(pos, neg)
    kw = dict(width=w, height=h, num_frames=nf, num_inference_steps=steps, guidance_scale=1.0,
              frame_rate=25, conditions=[], generator=torch.Generator("cpu").manual_seed(seed))
    if cached is not None:
        pe, pm, ne, nm = cached
        g = lambda x: (x.to("cuda", DT) if (torch.is_tensor(x) and x.is_floating_point()) else (x.to("cuda") if torch.is_tensor(x) else None))
        kw.update(prompt_embeds=g(pe), prompt_attention_mask=g(pm))
        if ne is not None: kw.update(negative_prompt_embeds=g(ne), negative_prompt_attention_mask=g(nm))
    else:
        kw.update(prompt=pos, negative_prompt=neg)
    out = pipe(**kw)
    if cap is not None:                                             # first time for this prompt -> persist + restore
        emb, capd, _orig, _ = cap
        if "e" in capd: torch.save(capd["e"], emb)
        pipe.encode_prompt = _orig
    fr = out.frames[0]
    imgs = [f if isinstance(f, Image.Image) else Image.fromarray((np.asarray(f).clip(0,1)*255).astype("uint8")) for f in fr]
    return imgs

def verify(imgs):
    a = np.asarray(imgs[0]).astype(np.float32)
    c = float(np.corrcoef(a[:-1].ravel(), a[1:].ravel())[0, 1])
    return c

# ---- serve loop ----
while _run["go"]:
    todo = sorted(glob.glob(os.path.join(JOBS, "*.json")))
    if not todo:
        time.sleep(1.0); continue
    jf = todo[0]
    try:
        with open(jf) as f: job = json.load(f)
    except Exception as e:
        os.rename(jf, jf + ".bad"); print(f"[bad job {jf}] {e}", flush=True); continue
    outdir = job.get("out", os.path.join(EMBDIR, "daemon_out"))
    os.makedirs(outdir, exist_ok=True)
    ts = time.time()
    try:
        imgs = gen(job)
        c = verify(imgs)
        for i, im in enumerate(imgs): im.save(os.path.join(outdir, f"f{i:03d}.png"))
        print(f"[job {os.path.basename(jf)}] {job['w']}x{job['h']} {job['frames']}f -> {time.time()-ts:.1f}s | row-corr={c:.3f} {'REAL' if c>0.9 else 'SUSPECT'} -> {outdir}", flush=True)
    except Exception as e:
        print(f"[job {os.path.basename(jf)}] FAILED {repr(e)[:160]}", flush=True)
    finally:
        import gc; gc.collect(); torch.cuda.empty_cache()
        try: os.rename(jf, os.path.join(DONE, os.path.basename(jf)))
        except Exception: pass
print("[daemon] stopped", flush=True)
