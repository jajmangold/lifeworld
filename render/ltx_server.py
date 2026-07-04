"""Resident LTX-2.3 server for the realism pass — keeps the 13GB model LOADED across clips so we don't pay the
~6min reload per clip (wgp.py --process reloads every time). Uses WanGP's in-process API (shared.api.WanGPSession):
the session holds the model between submit_task() calls. Runs inside the wan2gp docker on rtx0.

Job protocol (a watched dir, mirrors the flashvsr server pattern):
  drop  /workspace/ltxjobs/<name>.job.json  = a flat settings-OVERRIDES dict (must include 'model_type' +
        'video_guide' as an absolute path; the server merges it over get_default_settings()).
  server writes /workspace/ltxjobs/<name>.out (first generated file path) or <name>.err on failure.

  start: docker exec -e CUDA_VISIBLE_DEVICES=<gpu> wan2gp bash -lc 'cd /workspace && python3 ltx_server.py'
  ready: prints LTX_SERVER_READY once the model is warmed.
"""
import os, sys, json, time, glob, shutil, traceback
sys.path.insert(0, "/workspace"); os.chdir("/workspace")
from shared.api import WanGPSession

JOBS = "/workspace/ltxjobs"
OUTDIR = os.path.join(JOBS, "out_resident")
os.makedirs(OUTDIR, exist_ok=True)
# LTX_PROFILE=3 (LowRAM-HighVRAM = model RESIDENT on GPU, no PCIe weight streaming) — only fits if the model is
# small enough (Q3 ~10GB + freed text encoder). Default: inherit wgp_config.json (profile 4 = stream).
_prof = os.environ.get("LTX_PROFILE", "")
_cli = ["--profile", _prof] if _prof else []
session = WanGPSession(output_dir=OUTDIR, console_output=True, cli_args=_cli).ensure_ready()
# EAGER self-warmup: hydrate the model to THIS card at startup, THEN report READY. This lets ltx_farm.sh
# stagger the farm — start one server, wait for its READY (= weights loaded), then the next — so only ONE card
# hydrates the ~10GB weights at a time. Critical on this 30GB-RAM box (4 concurrent loads swap-thrash it).
# Done directly via submit_task (not the shared queue) so no atomic-claim race with sibling servers.
_warm = os.environ.get("LTX_WARMUP", os.path.join(JOBS, "_warmup.mp4"))
_model = os.environ.get("LTX_MODEL", "ltx2_22B_distilled_gguf_q3_k_s")
if os.path.exists(_warm):
    try:
        s = session.get_default_settings(_model)
        s.update({"video_guide": _warm, "video_prompt_type": "VGU", "image_prompt_type": "",
                  "keep_frames_video_guide": "", "denoising_strength": 0.5, "resolution": "512x320",
                  "video_length": 25, "seed": 1, "num_inference_steps": 8, "guidance_scale": 1.0,
                  "prompt": "a professional news anchor", "negative_prompt": ""})
        session.submit_task(s).result()
        print("LTX_WARMED (model resident on card)", flush=True)
    except Exception as e:
        print("LTX_WARM_WARN", repr(e), flush=True)
print(f"LTX_SERVER_READY (profile={_prof or 'config-default'})", flush=True)

GPU = os.environ.get("CUDA_VISIBLE_DEVICES", "?")
while True:
    for jf in sorted(glob.glob(os.path.join(JOBS, "*.job.json"))):
        name = os.path.basename(jf)[:-9]
        # ATOMIC CLAIM so a FARM of servers (1 per GPU) can share one queue without racing: os.rename is atomic,
        # so exactly one server wins the job; the losers get FileNotFoundError and move on. This load-balances
        # across all 4 cards automatically (whichever server is free grabs the next job).
        claimed = os.path.join(JOBS, name + ".proc")
        try:
            os.rename(jf, claimed)
        except OSError:
            continue
        done = os.path.join(JOBS, name + ".out"); err = os.path.join(JOBS, name + ".err")
        try:
            ov = json.load(open(claimed)); os.remove(claimed)
            print(f"LTX_CLAIM {name} on gpu{GPU}", flush=True)
            mt = ov.pop("model_type", "ltx2_22B_distilled_gguf_q4_k_m")
            settings = session.get_default_settings(mt)     # full valid settings for the model
            settings.update(ov)                             # overlay our realism-pass overrides
            t0 = time.time()
            job = session.submit_task(settings)             # model already resident -> no reload
            res = job.result()
            # the API path can exceed the 255-char FS limit (long prompt in the name) and truncate on disk, so
            # resolve to the actual newest output file and copy it to a clean, space-free name for the caller.
            newest = max(glob.glob(os.path.join(OUTDIR, "*.mp4")), key=os.path.getmtime, default="")
            safe = os.path.join(JOBS, name + "_out.mp4")
            if newest:
                shutil.copy(newest, safe); os.chmod(safe, 0o644)
            open(done, "w").write(safe if newest else "")
            print(f"LTX_DONE {name} {round(time.time()-t0,1)}s -> {safe}", flush=True)
        except Exception as e:
            open(err, "w").write(f"{e}\n{traceback.format_exc()}")
            print(f"LTX_ERR {name}: {e}", flush=True)
    time.sleep(2)
