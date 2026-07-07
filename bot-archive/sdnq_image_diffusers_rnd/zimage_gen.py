#!/usr/bin/env python3
"""Z-Image-Turbo (SDNQ int8) txt2img on a single sm_70 card.

Runs inside the sdnq-mgpu container (torch 2.9 / diffusers 0.38 / sdnq 0.2).
Pin the card by UUID from the host, e.g.:
  CUDA_VISIBLE_DEVICES=GPU-7540b3cb-... python3 zimage_gen.py --prompt "..."

sm_70 notes: bf16 storage/compute works (no tensor cores, so it is emulated but
correct). Quantized-matmul (SDNQ QMM) is Turing+ only -> OFF by default here;
set QMM=1 to try it. torch.compile gated behind COMPILE=1.
"""
import os
import argparse
import torch
import diffusers
from sdnq import SDNQConfig  # noqa: F401  (import registers the SDNQ quant loader)
from sdnq.loader import apply_sdnq_options_to_model

MODEL = os.environ.get("ZIMAGE_MODEL", "Disty0/Z-Image-Turbo-SDNQ-int8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative", default="")
    ap.add_argument("--out", default="/root/ComfyUI/output/zimage_sdnq.png")
    ap.add_argument("--steps", type=int, default=int(os.environ.get("STEPS", 8)))
    ap.add_argument("--cfg", type=float, default=float(os.environ.get("CFG", 1.0)))
    ap.add_argument("--width", type=int, default=int(os.environ.get("W", 1024)))
    ap.add_argument("--height", type=int, default=int(os.environ.get("H", 1024)))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("SEED", 42)))
    args = ap.parse_args()

    print(f"[zimage] loading {MODEL} (bf16)")
    pipe = diffusers.ZImagePipeline.from_pretrained(MODEL, torch_dtype=torch.bfloat16)

    if os.environ.get("QMM", "0") == "1":
        print("[zimage] enabling SDNQ quantized-matmul (Turing+; may fall back on sm_70)")
        pipe.transformer = apply_sdnq_options_to_model(pipe.transformer, use_quantized_matmul=True)
        pipe.text_encoder = apply_sdnq_options_to_model(pipe.text_encoder, use_quantized_matmul=True)

    # sm_70 has no flash-attention: SDPA needs a large contiguous buffer, so keep
    # the (bf16) text encoder on CPU during the diffusion loop to leave VRAM for it.
    if os.environ.get("OFFLOAD", "1") == "1":
        print("[zimage] enable_model_cpu_offload (frees VRAM for sm_70 SDPA buffer)")
        pipe.enable_model_cpu_offload()
    else:
        pipe.to("cuda")

    if os.environ.get("COMPILE", "0") == "1":
        pipe.transformer = torch.compile(pipe.transformer)

    g = torch.Generator(device="cpu").manual_seed(args.seed)
    print(f"[zimage] gen {args.width}x{args.height} steps={args.steps} cfg={args.cfg} seed={args.seed}")
    out = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative or None,
        num_inference_steps=args.steps,
        guidance_scale=args.cfg,
        width=args.width,
        height=args.height,
        generator=g,
    )
    out.images[0].save(args.out)
    print("SAVED", args.out)


if __name__ == "__main__":
    main()
