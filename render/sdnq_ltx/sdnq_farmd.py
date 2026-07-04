"""VOLTA FARM DAEMON — the RAM-slim + amortized-setup generation engine. Both levers combined:
  LEVER A (RAM-slim): load WITHOUT gemma (text_encoder=None, ~-24GB) and free the connectors module
                      (cached, ~-6GB) -> ~15GB VmRSS/process (vs ~28GB) -> ~2x concurrent jobs in 130GB.
  LEVER B (daemon):   load the pipeline ONCE, serve many jobs from a watch dir -> the ~77s setup is paid
                      once, not per clip.
Designed for LOW-RES generation (cheap on Volta) that a 3060 FlashVSR pass upscales 2x. One card per daemon;
run N daemons (one per CUDA_VISIBLE_DEVICES) for the farm. Job = JSON in JOBS/: {w,h,frames,steps,seed,out}.
Result frames -> job['out']/f*.png, job moved to JOBS/done/. Args: [JOBS_DIR]"""
import os, sys, time, glob, inspect, hashlib, json, gc, signal
import torch, sdnq  # noqa
from diffusers import DiffusionPipeline, LTX2ConditionPipeline
from sdnq.loader import apply_sdnq_options_to_model
from PIL import Image
import numpy as np

DT = torch.bfloat16
JOBS = sys.argv[1] if len(sys.argv) > 1 else "/root/ComfyUI/output/farmjobs"
DONE = os.path.join(JOBS, "done"); os.makedirs(DONE, exist_ok=True)
PATH = sorted(glob.glob("/root/.cache/huggingface/hub/models--OzzyGT--LTX-2.3-Distilled-1.1-sdnq-dynamic-int4/snapshots/*"))[0]
POS = ("Medium close-up vertical portrait of a professional female television news anchor in her mid-thirties, "
    "with shoulder-length auburn hair and warm brown eyes, seated at a modern broadcast desk. She wears a tailored "
    "navy blazer over a cream blouse, with subtle natural makeup and small gold stud earrings. She speaks directly "
    "to the camera with calm, articulate expressions, her lips moving naturally and her gaze steady. Soft key "
    "lighting from the front-left illuminates her face against a gently blurred dark blue studio background. The "
    "footage looks like real broadcast television: photorealistic, with natural skin texture, visible pores and fine "
    "detail. The camera holds a steady, locked-off shot.")
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
assert os.path.exists(EMB) and os.path.exists(CONN), "run sdnq_gen once to build embed+connector caches"
base = DiffusionPipeline.from_pretrained(PATH, torch_dtype=DT, text_encoder=None)   # LEVER A: no gemma
base.transformer = apply_sdnq_options_to_model(base.transformer, dtype=DT, use_quantized_matmul=True)
comp = {k: v for k, v in base.components.items()
        if k in set(inspect.signature(LTX2ConditionPipeline.__init__).parameters)}
comp["text_encoder"] = None
pipe = LTX2ConditionPipeline(**comp)
try:
    pipe.vae.enable_tiling(); pipe.vae.use_framewise_decoding = True; pipe.vae.tile_sample_min_num_frames = 16
except Exception: pass
_ct = tuple(torch.load(CONN, map_location="cpu"))
pipe.connectors = lambda pe_, pm_, padding_side="left": tuple(x.to(pe_.device) for x in _ct)  # LEVER A: free 6.35GB
del base; gc.collect()
pipe.enable_model_cpu_offload()
pe, pm, ne, nm = torch.load(EMB, map_location="cpu")
g = lambda x: (x.to("cuda", DT) if (torch.is_tensor(x) and x.is_floating_point()) else (x.to("cuda") if torch.is_tensor(x) else None))
EMB_KW = dict(prompt_embeds=g(pe), prompt_attention_mask=g(pm))
if ne is not None: EMB_KW.update(negative_prompt_embeds=g(ne), negative_prompt_attention_mask=g(nm))
dev = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
print(f"[farmd dev={dev} t={time.time()-t:.1f}] ready | VmRSS={rss_gb():.1f}GB | watching {JOBS}", flush=True)

def gen(job):
    w, h = int(job["w"]), int(job["h"]); nf = int(job["frames"]); steps = int(job.get("steps", 8)); seed = int(job.get("seed", 42))
    kw = dict(width=w, height=h, num_frames=nf, num_inference_steps=steps, guidance_scale=1.0, frame_rate=25,
              conditions=[], generator=torch.Generator("cpu").manual_seed(seed), **EMB_KW)
    if steps == 8: kw["sigmas"] = DISTILLED_SIGMAS_8
    out = pipe(**kw)
    return [f if isinstance(f, Image.Image) else Image.fromarray((np.asarray(f).clip(0,1)*255).astype("uint8")) for f in out.frames[0]]

while _run["go"]:
    todo = sorted(glob.glob(os.path.join(JOBS, "*.json")))
    if not todo: time.sleep(1.0); continue
    jf = todo[0]
    try: job = json.load(open(jf))
    except Exception as e: os.rename(jf, jf + ".bad"); print(f"[bad {jf}] {e}", flush=True); continue
    outdir = job.get("out", os.path.join(JOBS, "out", os.path.basename(jf)[:-5])); os.makedirs(outdir, exist_ok=True)
    ts = time.time()
    try:
        imgs = gen(job)
        for i, im in enumerate(imgs): im.save(os.path.join(outdir, f"f{i:03d}.png"))
        a = np.asarray(imgs[0].convert("L")).astype(float); rc = float(np.corrcoef(a[:-1].ravel(), a[1:].ravel())[0,1])
        print(f"[job {os.path.basename(jf)}] {job['w']}x{job['h']} {job['frames']}f -> {time.time()-ts:.1f}s row-corr={rc:.3f} {'REAL' if rc>0.9 else 'SUSPECT'} -> {outdir}", flush=True)
    except Exception as e:
        print(f"[job {os.path.basename(jf)}] FAILED {repr(e)[:160]}", flush=True)
    finally:
        gc.collect(); torch.cuda.empty_cache()
        try: os.rename(jf, os.path.join(DONE, os.path.basename(jf)))
        except Exception: pass
print("[farmd] stopped", flush=True)
