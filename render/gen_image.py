#!/usr/bin/env python3
"""Quick SD1.5 txt2img (uses Champ's local SD1.5 weights). python gen_image.py <out.png> "<prompt>" [W H]"""
import sys, torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
out = sys.argv[1]
prompt = sys.argv[2]
W = int(sys.argv[3]) if len(sys.argv) > 3 else 512
H = int(sys.argv[4]) if len(sys.argv) > 4 else 768
pipe = StableDiffusionPipeline.from_pretrained("/champ/pretrained_models/stable-diffusion-v1-5",
                                               torch_dtype=torch.float16, safety_checker=None)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
pipe = pipe.to("cuda")
g = torch.Generator("cuda").manual_seed(7)
img = pipe(prompt, negative_prompt="cropped head, out of frame, blurry, deformed hands, extra limbs, lowres, watermark, text",
           num_inference_steps=30, guidance_scale=7.5, height=H, width=W, generator=g).images[0]
img.save(out)
print("IMG_OK", out, img.size)
