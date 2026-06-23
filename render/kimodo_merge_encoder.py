#!/usr/bin/env python3
"""One-time: merge LLM2Vec's mntp+supervised LoRA into the base in fp16 (CPU) and save, so it can
then be loaded in 8-bit on the GPU (bitsandbytes can't merge LoRA into a quantized base). Run in
kimodo:volta with TEXT_ENCODERS_DIR + HF_HOME set."""
import os
os.environ.pop("KIMODO_QUANT", None)        # force the normal fp16 merge path
import torch
from transformers import AutoTokenizer
from kimodo.model.llm2vec.llm2vec import LLM2Vec

TED = os.environ["TEXT_ENCODERS_DIR"]
mntp = os.path.join(TED, "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp")
sup = os.path.join(TED, "McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised")
out = os.path.join(TED, "merged-llm2vec-fp16")

m = LLM2Vec.from_pretrained(base_model_name_or_path=mntp, peft_model_name_or_path=sup,
                            torch_dtype=torch.float16, merge_peft=True)   # merges both (fp16, OK)
mdl = m.model                                          # fully-merged base; drop lingering peft tags
for attr in ("peft_config", "_hf_peft_config_loaded"):
    if hasattr(mdl, attr):
        try: delattr(mdl, attr)
        except Exception: setattr(mdl, attr, False)
mdl.save_pretrained(out, safe_serialization=True)
AutoTokenizer.from_pretrained(mntp).save_pretrained(out)
print("MERGED_OK", out)
