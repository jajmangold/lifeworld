# sdnq_image — models, cards, and LoRA workflows

SDNQ-quantized **image** models on the Volta farm (sm_70), separate from the LTX/Wan
**video** stack. Run inside the `sdnq-mgpu` container (torch 2.9 / diffusers 0.38 /
sdnq 0.2 / transformers 4.56). Scripts live in `/root/sdnq_image/` (host:
`/srv/nvme-data/containers/comfy/storage/sdnq_image/`); outputs land in
`/root/ComfyUI/output/`.

## Cards (pin by UUID — container idx ≠ host idx, golden rule #6)

| Model | Script | Host card | UUID | Notes |
|---|---|---|---|---|
| Z-Image-Turbo (SDNQ int8, ~6B) | `zimage_gen.py` | 9 (V100) | `GPU-7540b3cb-…` | txt2img, **fast/practical**, ≤768² |
| Qwen-Image-Edit-2511 (SDNQ uint4-svd, 20B) | `qwen_edit.py` | 8 (CMP) | `GPU-534081fb-…` | edit, **sequential offload** (slow) |
| Qwen-Image-Edit-2509 (SDNQ uint4-svd, 20B) | `qwen_edit.py` (`QWENEDIT_MODEL=…2509`) | 7 (CMP) | `GPU-c0e49412-…` | edit, same pipeline class |

`CUDA_VISIBLE_DEVICES=GPU-<uuid>` pins a process to one physical card unambiguously.

## sm_70 realities that shaped these scripts

- **No flash-attention** → SDPA needs a big contiguous buffer; attention memory is
  quadratic in latent tokens. Keep edit/gen resolution ≤768² (512² for the 20B edit).
- **No int8/fp8 tensor cores** (Turing+ only) → SDNQ **quantized-matmul is OFF**
  (`QMM=1` to try; expect fallback). Weights are uint4/int8 *storage*; compute dequants
  to bf16 (works, not tensor-core accelerated).
- **20B edit models don't fit resident on one 16 GB Volta.** Measured footprint:
  transformer 10.8 GB + Qwen2.5-VL TE 5.05 GB + VAE 0.24 GB = 16.1 GB (> 15.77 usable).
  Every single-card variant was tried and hits a wall:
  - `enable_model_cpu_offload` — the SDNQ TE does **not** evict (accelerate hook skips
    quantized modules): TE 5 GB + transformer 10.8 GB both stay resident → OOM even at 256².
  - resident transformer + **CPU** VAE — works, but the video VAE's fp32 conv3d on CPU
    is ~2.5 min, and SDNQ uint4→bf16 dequant workspace + attention still OOM at 512².
  - resident transformer + **GPU** VAE (even tiled) — the conv3d falls back to a CPU-only
    `slow_conv3d` kernel because cuDNN can't get a workspace in the ~5 GB left with the
    transformer resident → CUDA `NotImplementedError`. Works only when the transformer
    is absent (which defeats the point).
  - **PCIe-gen1 ×1** makes any host↔device transfer of the 10.8 GB transformer ~40 s,
    so streaming (`enable_sequential_cpu_offload`) is ~110 s/step.
  → `qwen_edit.py` defaults to `OFFLOAD=seq` (sequential offload): the only reliably
  correct single-card mode, **~7 min/512² image**. `OFFLOAD=manual` (+`VAE_CPU=1`) is
  kept for experimentation but still OOMs at usable resolutions.
- **Fast edit needs multi-card** (not yet built): since 2509 & 2511 share a
  byte-identical Qwen2.5-VL TE + VAE, one **encode card** (TE+VAE resident, free VRAM →
  GPU VAE works) can serve **both** transformer cards (2511, 2509) resident on their own
  cards. Only tiny latents cross the bus → seconds/image. This is the recommended build.
- Z-Image (6 B int8) runs with plain `enable_model_cpu_offload` and IS the practical/fast
  path today (~2 min/768²).

## LoRA registry

`loras/qwen-image-edit-2511/` and `loras/qwen-image-edit-2509/` (symlinks into the HF
cache; targets are container-internal `/root/.cache/...` paths). Stack via comma-lists:
`--lora a.safetensors,b.safetensors --lora-scale 1.0,1.0`.

## Recipes (from model cards + the BFS ComfyUI workflow JSONs)

### Z-Image-Turbo (txt2img)
`python3 zimage_gen.py --prompt "…" --width 768 --height 768 --steps 8 --cfg 1.0`
Turbo distill: 8 steps, CFG 1.0. Confirmed producing clean photoreal headshots.

### Qwen-Image-Edit-2511 + Lightning (fast edit)
`--lora loras/qwen-image-edit-2511/lightning-4step.safetensors --steps 4 --cfg 1.0`
4-step distill (≈10× fewer steps than the 40-step base). euler / simple scheduler.

### CustomLightning (cleaner distill, 2511)
`custom-lightning.safetensors` — 4-or-8-step distill that corrects the oversaturation /
plastic-skin of plain Lightning. Use 4 or 8 steps. Preferred in the BFS workflows.

### Multiple-Angles (2511) — camera control
Trigger: `<sks> [azimuth] [elevation] [distance]`, LoRA weight 0.8–1.0.
- azimuth (8): front / front-right/left quarter / side / back-right/left quarter / back
- elevation (4): low-angle (-30°) / eye-level (0°) / elevated (30°) / high-angle (60°)
- distance (3): close-up (×0.6) / medium (×1.0) / wide (×1.8)
e.g. `<sks> front view eye-level shot medium shot`

### BFS Best-Face-Swap (head/face swap) — from the author's workflow JSON
2 input images. V3+/2511: **Picture 1 = body/base, Picture 2 = face** (inverted from v1/v2).
Stack: `bfs-head-v5.safetensors` (2511) or `bfs-head-vN.safetensors` (2509) **+
custom-lightning**. euler / simple, ~8–20 steps, **CFG 2.5**, model shift 1, CFGNorm.
Prompt (verbatim, V5):
> head_swap: start with Picture 1 as the base image, keeping its lighting, environment,
> and background. remove the head from Picture 1 completely and replace it with the head
> from Picture 2, strictly preserving the hair, eye color, and nose structure of Picture
> 2. copy the eye direction, head rotation, and micro-expressions from Picture 1. high
> quality, sharp details, 4k

### InScene (2509) — scene-coherent edits
Descriptive prompts, no trigger word: "Make a shot in the same scene of …, camera angle
shifts slightly …". Base recipe steps 50 / CFG 4.0 (fewer steps if stacked with a distill).

### Next-Scene (2509) — next-shot continuation (civitai v2-3000)
`loras/qwen-image-edit-2509/next-scene_lora-v2-3000.safetensors`. Prompt prefix
"Next Scene:" then describe the following shot. Pair with a distill LoRA for speed.
