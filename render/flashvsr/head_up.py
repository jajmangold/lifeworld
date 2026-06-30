import sys, os, time
sys.path.insert(0, "/workspace/plugins")
os.chdir("/workspace")
import torch, imageio, numpy as np
import importlib

stem = sys.argv[1]
mod = importlib.import_module("wan2gp-flashvsr.plugin")
dm  = importlib.import_module("wan2gp-flashvsr.src.models.download_manager")

# ULTRA finish (steal from ComfyUI-FlashVSR_Ultra_Fast): 4x + full-block + unload_dit so the MCU-sized
# face gets supersampled detail; tiled DiT/VAE + unload_dit keep it inside 12GB (slower). Default = 2x tiny.
ULTRA = os.environ.get("ULTRA", "0") == "1"
device, dtype = "cuda", torch.bfloat16
scale   = 4 if ULTRA else 2
variant = "tiny-long"
full_block = ULTRA
unload  = ULTRA
sparse_r, kv, local_r, out_quality, c_fix = 2.0, 3, 11, 8, True

inp = f"/workspace/myinput/head_crop_{stem}.mp4"
out = f"/workspace/myinput/head_up_{stem}.mp4"

reader = imageio.get_reader(inp)
fps = int(round(reader.get_meta_data().get('fps', 25)))
frames = [torch.from_numpy(fr.astype(np.float32)/255.0).to(dtype) for fr in reader]
reader.close()
vt = torch.stack(frames, 0)
fc = vt.shape[0]
print(f"[hu] {fc} frames {tuple(vt.shape)} @ {fps} | ULTRA={ULTRA} scale={scale} full_block={full_block}", flush=True)
pad = mod.next_8n5(fc) - fc
if pad > 0:
    vt = torch.cat([vt, vt[-1:].repeat(pad,1,1,1)], 0)

t0 = time.time()
pipeline = dm.load_pipeline(variant=variant, device=device, torch_dtype=dtype, model_version="FlashVSR-v1.1")
print(f"[hu] pipeline {time.time()-t0:.1f}s", flush=True)

th, tw, F = mod.get_input_params(vt, scale=scale)
LQ = mod.input_tensor_generator(vt, device, scale=scale, dtype=dtype)
topk = sparse_r * 768 * 1280 / (th * tw)
print(f"[hu] one-pass {vt.shape[2]}x{vt.shape[1]} -> {tw}x{th}, F={F}", flush=True)
ts = time.time()
pipeline(LQ_video=LQ, num_frames=F, height=th, width=tw, topk_ratio=topk,
         output_path=out, quality=int(out_quality),
         prompt="", negative_prompt="", cfg_scale=1.0, num_inference_steps=1, seed=0,
         tiled=True, is_full_block=full_block, if_buffer=True, kv_ratio=int(kv),
         local_range=int(local_r), unload_dit=unload, fps=fps, color_fix=c_fix)
print(f"[hu] DONE in {time.time()-ts:.1f}s (total {time.time()-t0:.1f}s) -> {out}", flush=True)
