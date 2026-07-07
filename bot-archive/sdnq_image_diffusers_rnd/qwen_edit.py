#!/usr/bin/env python3
"""Qwen-Image-Edit-2511 (SDNQ uint4-svd-r32) instruction image edit on one sm_70 card.

Runs inside the sdnq-mgpu container (torch 2.9 / diffusers 0.38 / sdnq 0.2).
Pin the card by UUID from the host, e.g.:
  CUDA_VISIBLE_DEVICES=GPU-534081fb-... python3 qwen_edit.py \
      --image in.png --prompt "make the tie red"

The 20B transformer in uint4 (~10 GB) + the Qwen2.5-VL text encoder do not both
fit resident on 16 GB, so we default to enable_model_cpu_offload() (OFFLOAD=1).
Slow over the PCIe-gen1 riser but correct. Set OFFLOAD=0 to force fully-resident.

Lightning few-step distill: pass --lora <path-to-4step.safetensors> (STEPS=4,
CFG=1.0). LoRA-on-quantized may not apply cleanly; base path is the fallback.

sm_70: QMM (SDNQ quantized-matmul) is Turing+ only -> OFF by default (QMM=1 to try).
"""
import os
import argparse
import torch
import diffusers
from PIL import Image
from sdnq import SDNQConfig  # noqa: F401  (import registers the SDNQ quant loader)
from sdnq.loader import apply_sdnq_options_to_model

MODEL = os.environ.get("QWENEDIT_MODEL", "Disty0/Qwen-Image-Edit-2511-SDNQ-uint4-svd-r32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", nargs="+", required=True, help="one or more input images")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--negative", default="")
    ap.add_argument("--out", default="/root/ComfyUI/output/qwen_edit_sdnq.png")
    ap.add_argument("--steps", type=int, default=int(os.environ.get("STEPS", 20)))
    ap.add_argument("--cfg", type=float, default=float(os.environ.get("CFG", 4.0)))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("SEED", 42)))
    ap.add_argument("--width", type=int, default=int(os.environ.get("W", 512)),
                    help="edit resolution; sm_70 has no flash-attn so keep <=768")
    ap.add_argument("--height", type=int, default=int(os.environ.get("H", 512)))
    ap.add_argument("--lora", default=os.environ.get("LORA", ""))
    ap.add_argument("--lora-scale", default=os.environ.get("LORA_SCALE", "1.0"),
                    help="single float, or comma-list matching --lora order")
    args = ap.parse_args()

    # DTYPE: bf16 is the farm default but has NO tensor cores on Volta (slow matmul) and
    # xformers' sm_70 kernel rejects it. **fp16 uses V100 fp16 tensor cores AND unlocks
    # xformers memory-efficient attention** -> the fast path on sm_70.
    DT = {"fp16": torch.float16, "bf16": torch.bfloat16}[os.environ.get("DTYPE", "bf16")]
    print(f"[qwenedit] loading {MODEL} dtype={DT}")
    _dm = os.environ.get("DEVICE_MAP", "")
    if _dm:
        print(f"[qwenedit] device_map={_dm}")
        pipe = diffusers.QwenImageEditPlusPipeline.from_pretrained(MODEL, torch_dtype=DT, device_map=_dm)
    else:
        pipe = diffusers.QwenImageEditPlusPipeline.from_pretrained(MODEL, torch_dtype=DT)

    # Compat shim: this model's Qwen2.5-VL config nests the vision token ids under
    # `text_config`, but transformers 4.56 get_rope_index reads them flat off the
    # top-level config. Promote them so we don't have to bump the shared farm's
    # transformers. (Wrap, don't fight — golden rule #3.)
    cfg = pipe.text_encoder.config
    sub = getattr(cfg, "text_config", None)
    for attr in ("vision_start_token_id", "vision_end_token_id", "vision_token_id",
                 "image_token_id", "video_token_id"):
        if getattr(cfg, attr, None) is None and sub is not None and getattr(sub, attr, None) is not None:
            setattr(cfg, attr, getattr(sub, attr))
            print(f"[qwenedit] shim: config.{attr} = {getattr(cfg, attr)}")

    # Volta has no PyTorch-SDPA flash/mem-efficient kernel, but xformers' own kernel
    # DOES run on sm_70 -> memory-efficient attention (no full N*N matrix). flash_attn
    # is Ampere+ only; sageattention int8 fails to compile on sm_70.
    _ab = os.environ.get("ATTN_BACKEND", "")
    if _ab and hasattr(pipe.transformer, "set_attention_backend"):
        try:
            pipe.transformer.set_attention_backend(_ab)
            print(f"[qwenedit] attention backend = {_ab}")
        except Exception as e:
            print(f"[qwenedit] set_attention_backend({_ab}) failed: {e}")

    # LoRAs can be stacked: repeat --lora / --lora-scale, or comma-separate.
    loras = [x for x in args.lora.split(",") if x]
    scales = [float(x) for x in str(args.lora_scale).split(",")]
    adapter_names = []
    for i, lp in enumerate(loras):
        name = f"lora{i}"
        print(f"[qwenedit] loading LoRA {lp} as {name}")
        # offline mode needs dir + weight_name split when given a file path
        if os.path.isfile(lp):
            pipe.load_lora_weights(os.path.dirname(lp), weight_name=os.path.basename(lp), adapter_name=name)
        else:
            pipe.load_lora_weights(lp, adapter_name=name)
        adapter_names.append(name)
    if adapter_names:
        sc = scales if len(scales) == len(adapter_names) else [scales[0]] * len(adapter_names)
        pipe.set_adapters(adapter_names, adapter_weights=sc)

    if os.environ.get("QMM", "0") == "1":
        print("[qwenedit] enabling SDNQ quantized-matmul (Turing+; may fall back on sm_70)")
        pipe.transformer = apply_sdnq_options_to_model(pipe.transformer, use_quantized_matmul=True)
        pipe.text_encoder = apply_sdnq_options_to_model(pipe.text_encoder, use_quantized_matmul=True)

    imgs = [Image.open(p).convert("RGB") for p in args.image]
    if args.width and args.height:
        imgs = [im.resize((args.width, args.height), Image.LANCZOS) for im in imgs]
    g = torch.Generator(device="cpu").manual_seed(args.seed)
    dev = torch.device("cuda")

    # Default = seq (sequential CPU offload): the only mode that reliably produces a
    # correct edit on ONE 16 GB Volta. ~7 min/512^2 over the PCIe-gen1 riser. The
    # 'manual' modes (resident transformer + CPU/GPU VAE) were explored but every
    # single-card variant OOMs: SDNQ uint4->bf16 dequant workspace + no-flash-attn
    # attention exceed 16 GB, and the video VAE's conv3d needs a cuDNN workspace that
    # only exists when the transformer is NOT resident. Fast edit => multi-card
    # (shared Qwen2.5-VL TE+VAE on an encode card, transformer resident on its own).
    mode = os.environ.get("OFFLOAD", "seq")
    call_kwargs = dict(image=imgs, num_inference_steps=args.steps, true_cfg_scale=args.cfg,
                       height=args.height or None, width=args.width or None, generator=g)

    # Volta has no flash/mem-efficient SDPA kernel (only math, which materializes the
    # full N*N attention matrix ~ the per-block memory spike). Slice attention so it is
    # computed in chunks -> big drop in per-block peak. Default on for the GPU-resident modes.
    if os.environ.get("ATTN_SLICE", "1") == "1" and mode in ("multicard", "manual", "cuda"):
        try:
            pipe.enable_attention_slicing(int(os.environ.get("ATTN_SLICE_SIZE", "1")))
            print("[qwenedit] attention slicing ON")
        except Exception as e:
            print("[qwenedit] attention slicing unavailable:", e)

    if mode == "manual":
        # SDNQ's quantized text encoder does NOT evict under accelerate's
        # enable_model_cpu_offload (measured: TE 5.05 GB + transformer 10.8 GB both
        # stay resident -> OOM even at 256^2). So we sequence it by hand:
        #   1) TE + VAE on GPU, encode prompt (+negative), 2) move TE to CPU & free,
        #   3) transformer resident, denoise. Peak ~13.5 GB @512^2 -> fits AND fast.
        import gc
        print("[qwenedit] manual: encode with TE on GPU")
        pipe.text_encoder.to(dev)
        with torch.no_grad():
            pe, pem = pipe.encode_prompt(prompt=args.prompt, image=imgs, device=dev)
            call_kwargs.update(prompt_embeds=pe, prompt_embeds_mask=pem)
            if args.cfg and float(args.cfg) != 1.0:
                npe, npem = pipe.encode_prompt(prompt=args.negative or " ", image=imgs, device=dev)
                call_kwargs.update(negative_prompt_embeds=npe, negative_prompt_embeds_mask=npem)
        # encode_prompt can hand back CPU tensors; the transformer runs on GPU.
        for k in ("prompt_embeds", "prompt_embeds_mask", "negative_prompt_embeds", "negative_prompt_embeds_mask"):
            if call_kwargs.get(k) is not None:
                call_kwargs[k] = call_kwargs[k].to(dev)
        print("[qwenedit] manual: evict TE -> CPU, load transformer")
        pipe.text_encoder.to("cpu"); gc.collect(); torch.cuda.empty_cache()
        pipe.transformer.to(dev)
        # With TE/VAE on CPU, the pipeline's _execution_device resolves to CPU and it
        # would put the noise latents on CPU (transformer is on GPU -> device mismatch).
        # Pin execution device to the transformer's card.
        type(pipe)._execution_device = property(lambda self: dev)
        # VAE placement. With the 10.8 GB transformer resident, a whole-image conv3d
        # can't get a cuDNN workspace in the ~5 GB left -> it falls to a CPU-only
        # slow_conv3d kernel and errors on CUDA. Two fixes:
        #   VAE_CPU=1 : bounce the VAE through CPU in fp32 (robust but ~2-3 min/img;
        #               bf16-on-CPU conv3d effectively hangs, so fp32).
        #   default   : keep the VAE on GPU but TILED -> small per-tile conv3d fits
        #               the free VRAM and runs fast.
        if os.environ.get("VAE_CPU", "0") == "1":
            pipe.vae.to("cpu", torch.float32)
            _eiv = pipe._encode_vae_image
            pipe._encode_vae_image = lambda image, generator: _eiv(image.to("cpu", torch.float32), generator).to(image.device, image.dtype)
            _dec = pipe.vae.decode
            pipe.vae.decode = lambda z, *a, **k: _dec(z.to("cpu", torch.float32), *a, **k)
        else:
            pipe.vae.to(dev)
            pipe.vae.enable_tiling()
            pipe.vae.enable_slicing()
    elif mode == "multicard":
        # FAST PATH. Two GPUs visible: cuda:0 = encode card (Qwen2.5-VL TE + VAE, both
        # resident with plenty of free VRAM so the video-VAE conv3d gets a cuDNN
        # workspace and runs on-GPU), cuda:1 = the 10.8 GB transformer alone (fits the
        # attention buffer at good res). Only embeds + tiny latents cross the bus.
        edev, tdev = torch.device("cuda:0"), torch.device("cuda:1")
        print(f"[qwenedit] multicard: encode={edev} denoise={tdev}")
        pipe.text_encoder.to(edev); pipe.vae.to(edev)
        pipe.transformer.to(tdev)
        mv = lambda t: t.to(tdev) if t is not None else None  # mask is None when all-ones
        with torch.no_grad():
            pe, pem = pipe.encode_prompt(prompt=args.prompt, image=imgs, device=edev)
            call_kwargs.update(prompt_embeds=mv(pe), prompt_embeds_mask=mv(pem))
            if args.cfg and float(args.cfg) != 1.0:
                npe, npem = pipe.encode_prompt(prompt=args.negative or " ", image=imgs, device=edev)
                call_kwargs.update(negative_prompt_embeds=mv(npe), negative_prompt_embeds_mask=mv(npem))
        # denoise happens on tdev; VAE stays on edev and we bridge tensors across cards.
        type(pipe)._execution_device = property(lambda self: tdev)
        _eiv = pipe._encode_vae_image
        pipe._encode_vae_image = lambda image, generator: _eiv(image.to(edev), generator).to(tdev)
        _dec = pipe.vae.decode
        pipe.vae.decode = lambda z, *a, **k: _dec(z.to(edev), *a, **k)
    elif mode == "balanced":
        # pipeline already sharded across GPUs via DEVICE_MAP=balanced at load; just run.
        print("[qwenedit] balanced device_map: pipeline pre-placed")
        call_kwargs.update(prompt=args.prompt, negative_prompt=args.negative or None)
    elif mode == "seq":
        print("[qwenedit] enable_sequential_cpu_offload (slow fallback)")
        pipe.enable_sequential_cpu_offload()
        call_kwargs.update(prompt=args.prompt, negative_prompt=args.negative or None)
    else:
        pipe.to("cuda")
        call_kwargs.update(prompt=args.prompt, negative_prompt=args.negative or None)

    if os.environ.get("PROFILE", "0") == "1" and hasattr(pipe.transformer, "transformer_blocks"):
        tdv = pipe.transformer.device
        def _mk(i):
            def _h(m, inp, out):
                torch.cuda.synchronize(tdv)
                print(f"[prof] block{i}: alloc={torch.cuda.memory_allocated(tdv)/2**30:.2f} peak={torch.cuda.max_memory_allocated(tdv)/2**30:.2f}", flush=True)
            return _h
        def _pre(name):
            def _h(m, inp):
                torch.cuda.synchronize(tdv)
                print(f"[prof] ->{name}: alloc={torch.cuda.memory_allocated(tdv)/2**30:.2f}", flush=True)
            return _h
        for name, child in pipe.transformer.named_children():
            if name != "transformer_blocks":
                child.register_forward_pre_hook(_pre(name))
                child.register_forward_hook(lambda m, i, o, n=name: print(f"[prof] <-{n}: alloc={torch.cuda.memory_allocated(tdv)/2**30:.2f}", flush=True))
        for i, b in enumerate(pipe.transformer.transformer_blocks):
            if i < 3:
                b.register_forward_pre_hook(_pre(f"block{i}"))
        torch.cuda.reset_peak_memory_stats(tdv)

    if os.environ.get("NAN_CHECK", "0") == "1" and hasattr(pipe.transformer, "transformer_blocks"):
        state = {"first": None}
        def _nanhook(i):
            def _h(m, inp, out):
                o = out[0] if isinstance(out, tuple) else out
                if isinstance(o, torch.Tensor) and state["first"] is None and not torch.isfinite(o).all():
                    mx = o.abs()[torch.isfinite(o)].max().item() if torch.isfinite(o).any() else float("inf")
                    state["first"] = i
                    print(f"[nan] first non-finite at block{i}; max finite abs so far={mx:.1f}", flush=True)
            return _h
        for i, b in enumerate(pipe.transformer.transformer_blocks):
            b.register_forward_hook(_nanhook(i))

    print(f"[qwenedit] edit {args.width}x{args.height} steps={args.steps} cfg={args.cfg} seed={args.seed} imgs={len(imgs)} mode={mode}")
    out = pipe(**call_kwargs)
    out.images[0].save(args.out)
    for i in range(torch.cuda.device_count()):
        print(f"[mem] cuda:{i} peak_alloc={torch.cuda.max_memory_allocated(i)/2**30:.2f}GB")
    print("SAVED", args.out)


if __name__ == "__main__":
    main()
