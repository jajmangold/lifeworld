# ARCHIVED — Qwen-Image-Edit diffusers/SDNQ R&D (superseded)

This is the diffusers/SDNQ exploration for Qwen-Image-Edit on sm_70. It is **superseded by the
production sd.cpp servers** in `lm-stack/qwen-edit-sdcpp/` (see that README + memory
`sdnq-image-models-volta.md`).

Why it's dead: on Volta, diffusers can't be both fast and correct — fp16 uses tensor cores but the
model overflows to NaN; bf16 is correct but has no tensor cores; SDNQ int8 QMM needs a triton bf16
compile Volta lacks. ggml/sd.cpp (native mixed precision) won. Kept only as the record of what was tried:

- `qwen_edit.py` — every diffusers offload/attention mode explored (seq/manual/multicard/balanced/fp16/xformers). All dead ends.
- `quantize_edit.py` — self-quantize to uint4-no-svd + QMM (proved QMM gives no speedup on sm_70).
- `zimage_gen.py` — diffusers Z-Image (superseded by the sd.cpp z-image server on card 14).
- `WORKFLOWS.md` — R&D-era notes + LoRA recipe reference (multi-angle / bfs / inscene / next-scene triggers).
