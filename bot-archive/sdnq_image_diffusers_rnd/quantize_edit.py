#!/usr/bin/env python3
"""Self-quantize the base Qwen-Image-Edit transformer to a QMM-compatible SDNQ variant.

The published Disty0 uint4-**svd-r32** model can't use SDNQ quantized-matmul (svd_up
shape bug) -> stuck on the slow bf16-dequant path (~98 s/step on Volta). We re-quantize
the base transformer as **uint4, no-svd, use_quantized_matmul=True** (int8 DP4A), which
Volta supports (same recipe as the farm's Wan/Z-Image int8). group_size=0 for fast MatMul.

Then assemble a pipeline dir: our quantized transformer/ + the (byte-identical) TE, VAE,
tokenizer, processor, scheduler from the already-downloaded svd model.

Run inside sdnq-mgpu:
  CUDA_VISIBLE_DEVICES=GPU-<uuid> python3 quantize_edit.py
"""
import os
import shutil
import torch
import diffusers
from sdnq import SDNQConfig

BASE = os.environ.get("BASE", "Qwen/Qwen-Image-Edit-2511")
SVD_SNAP = os.environ["SVD_SNAP"]                     # existing svd model snapshot (for TE/VAE/etc)
OUT = os.environ.get("OUT", "/root/sdnq_image/models/qwen-edit-2511-uint4-noSvd-qmm")
WTYPE = os.environ.get("WTYPE", "uint4")
QDEV = os.environ.get("QDEV", "cuda:0")               # quantize on GPU (streams per-layer); 'cpu' fallback

os.makedirs(OUT, exist_ok=True)

qcfg = SDNQConfig(
    weights_dtype=WTYPE,
    use_svd=False,                    # <-- the fix: no svd -> QMM works
    use_quantized_matmul=True,        # int8 DP4A on Volta
    quantized_matmul_dtype="int8",
    group_size=0,                     # disable grouping -> faster MatMul (like Z-Image int8)
    quantization_device=QDEV,
)
print(f"[quant] loading+quantizing {BASE} transformer -> {WTYPE} no-svd QMM (dev={QDEV})")
t = diffusers.QwenImageTransformer2DModel.from_pretrained(
    BASE, subfolder="transformer", torch_dtype=torch.bfloat16,
    quantization_config=qcfg, local_files_only=True,
)
print("[quant] saving transformer ->", OUT)
t.save_pretrained(os.path.join(OUT, "transformer"))
del t
torch.cuda.empty_cache()

# assemble the rest of the pipeline dir from the identical svd model components
print("[quant] assembling pipeline dir from", SVD_SNAP)
for item in os.listdir(SVD_SNAP):
    if item == "transformer":
        continue
    src = os.path.join(SVD_SNAP, item)
    dst = os.path.join(OUT, item)
    if os.path.exists(dst):
        continue
    if os.path.isdir(src):
        shutil.copytree(src, dst, symlinks=False)
    else:
        shutil.copy2(src, dst)
print("[quant] DONE ->", OUT)
