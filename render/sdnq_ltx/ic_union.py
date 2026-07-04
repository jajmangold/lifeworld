import glob, inspect, os, time, types, torch, sdnq
from diffusers import DiffusionPipeline, LTX2ConditionPipeline
from diffusers.pipelines.ltx2.pipeline_ltx2_condition import retrieve_latents
from diffusers.utils.torch_utils import randn_tensor
from sdnq.loader import apply_sdnq_options_to_model
from PIL import Image
import numpy as np
DT=torch.bfloat16; DEV="cuda:0"
W,H,STEPS=256,384,8
NPX=int(os.environ.get("NPX","25"))       # pixel frames for output == guide (must be 1 mod 8)
OUT_PX=NPX; GUIDE_PX=NPX
DS=int(os.environ.get("DS","2")); STR=float(os.environ.get("STR","1.0"))
SEED=int(os.environ.get("SEED","5")); OUTDIR=os.environ.get("OUTDIR","icu_out")
TOTAL=OUT_PX  # guide OVERLAPS output frames (frame_idx=0), so num_frames == output frames
CTRL=os.environ.get("CTRL","ctrl_armsup")
PATH=sorted(glob.glob("/root/.cache/huggingface/hub/models--OzzyGT--LTX-2.3-Distilled-1.1-sdnq-dynamic-int4/snapshots/*"))[0]
LORA="/root/ComfyUI/models/loras/ltxv/ltx2/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
EMB="/root/ComfyUI/output/embeds_fcb0316dc2.pt"; CONN="/root/ComfyUI/output/conn_fcb0316dc2.pt"
SIG=[1.0,0.99375,0.9875,0.98125,0.975,0.909375,0.725,0.421875]
base=DiffusionPipeline.from_pretrained(PATH,torch_dtype=DT,text_encoder=None)
base.transformer=apply_sdnq_options_to_model(base.transformer,dtype=DT,use_quantized_matmul=False)
comp={k:v for k,v in base.components.items() if k in set(inspect.signature(LTX2ConditionPipeline.__init__).parameters)};comp["text_encoder"]=None
pipe=LTX2ConditionPipeline(**comp)
_ct=tuple(torch.load(CONN,map_location="cpu"));pipe.connectors=lambda pe_,pm_,padding_side="left":tuple(x.to(pe_.device) for x in _ct)
pipe.transformer.to(DEV); pipe.vae.to(DEV)
try: pipe.vae.enable_tiling()
except: pass
for n in ("audio_vae","vocoder"):
    m=getattr(pipe,n,None); m.to("cpu") if m is not None else None
pipe.__class__._execution_device=property(lambda self: torch.device(DEV))
pe,pm,ne,nm=torch.load(EMB,map_location="cpu")
g=lambda x:(x.to(DEV,DT) if (torch.is_tensor(x) and x.is_floating_point()) else (x.to(DEV) if torch.is_tensor(x) else None))
EMB_KW=dict(prompt_embeds=g(pe),prompt_attention_mask=g(pm))
if ne is not None: EMB_KW.update(negative_prompt_embeds=g(ne),negative_prompt_attention_mask=g(nm))
pipe.load_lora_weights(LORA)

# --- grid geometry ---
Hl=H//32; Wl=W//32                      # 12, 8
outFl=(OUT_PX-1)//8+1                    # 4
gFl=(GUIDE_PX-1)//8+1                    # 4
GF=outFl                                 # grid = output frames; guide OVERLAPS same temporal coords (frame_idx=0)
Hg=Hl//DS; Wg=Wl//DS                     # 6, 4  compact guide spatial
N_O=outFl*Hl*Wl                          # 384
N_G=gFl*Hg*Wg                            # 96
# --- control frames -> half-res video tensor [-1,1] ---
ctrl_pils=[Image.open(f).convert("RGB") for f in sorted(glob.glob(f"/root/ComfyUI/output/{CTRL}/c_*.png"))][:GUIDE_PX]
ctrl_t=pipe.video_processor.preprocess_video(ctrl_pils, height=H//DS, width=W//DS).to(DEV, pipe.vae.dtype)

# SUBJECT-only masking: keep guide tokens only where the depth is FIGURE (near/bright), drop background
# guide tokens so the unconstrained bg renders GREEN from the prompt -> clean chroma key.
SUBJECT=int(os.environ.get("SUBJECT","0")); FIGTHR=float(os.environ.get("FIGTHR","-0.7"))
# figure mask at compact guide LATENT res (gFl,Hg,Wg), in (f,h',w') pack order (temporal VAE /8 -> gFl frames)
fig=torch.nn.functional.interpolate(ctrl_t.float(), size=(gFl,Hg,Wg), mode="trilinear", align_corners=False)  # [1,3,gFl,Hg,Wg]
figm=(fig[0].mean(0) > FIGTHR)                                                                        # [gFl,Hg,Wg] True=figure
figflat=figm.reshape(-1).to(DEV) if SUBJECT else torch.ones(gFl*Hg*Wg,dtype=torch.bool,device=DEV)   # [gFl*Hg*Wg]

# token indices into the (GF,Hl,Wl) grid, in (f,h,w) pack order.
# ComfyUI append_keyframe: guide keeps temporal coords [0,gFl) (frame_idx=0), OVERLAPPING output frames.
sf,sh=Hl*Wl,Wl
out_idx=[f*sf+h*sh+w for f in range(0,outFl) for h in range(Hl) for w in range(Wl)]         # output frames [0,outFl)
gui_idx_all=[f*sf+h*sh+w for f in range(0,gFl) for h in range(0,Hl,DS) for w in range(0,Wl,DS)]  # ALL ::DS positions
gui_idx=[gi for gi,keep in zip(gui_idx_all, figflat.tolist()) if keep]                        # figure-only guide positions
N_O=outFl*Hl*Wl; N_G=len(gui_idx)
IDX=torch.tensor(out_idx+gui_idx,dtype=torch.long,device=DEV)
print(f"[icu] control {ctrl_t.shape} DS={DS} STR={STR} CTRL={CTRL} SUBJECT={SUBJECT} LoRA={pipe.get_active_adapters()} N_O={N_O} N_G={N_G}/{gFl*Hg*Wg}",flush=True)

# --- monkeypatch prepare_latents: compact [output | guide] sequence ---
def compact_prepare(self, conditions=None, batch_size=1, num_channels_latents=128, height=512, width=768,
                    num_frames=121, noise_scale=1.0, dtype=None, device=None, generator=None, latents=None):
    gl=retrieve_latents(self.vae.encode(ctrl_t), generator=generator, sample_mode="argmax")
    gl=self._normalize_latents(gl, self.vae.latents_mean, self.vae.latents_std).to(device=device,dtype=dtype)  # [B,C,gFl,Hg,Wg]
    guide=self._pack_latents(gl,self.transformer_spatial_patch_size,self.transformer_temporal_patch_size)       # [B,gFl*Hg*Wg,C]
    guide=guide[:, figflat, :]                                                                                  # [B,N_G,C] figure-only
    zero_out=torch.zeros((batch_size,num_channels_latents,outFl,Hl,Wl),device=device,dtype=dtype)
    out=self._pack_latents(zero_out,self.transformer_spatial_patch_size,self.transformer_temporal_patch_size)   # [B,N_O,C]
    clean=torch.cat([out,guide],dim=1)                                                                          # [B,480,C]
    msk=torch.cat([out.new_zeros(batch_size,N_O,1), out.new_full((batch_size,N_G,1),STR)],dim=1)                # [B,480,1]
    noise=randn_tensor(clean.shape,generator=generator,device=clean.device,dtype=clean.dtype)
    sm=(1.0-msk)*noise_scale; lat=noise*sm+clean*(1-sm)
    print(f"[icu] prepared packed={tuple(lat.shape)} cond_tokens={int((msk>0).sum())}",flush=True)
    return lat,msk,clean
pipe.prepare_latents=types.MethodType(compact_prepare,pipe)

# --- monkeypatch rope.prepare_video_coords: gather output+guide coords from full grid ---
_orig_pvc=pipe.transformer.rope.prepare_video_coords
def custom_pvc(batch_size, num_frames, height, width, device, fps=24.0):
    full=_orig_pvc(batch_size, GF, Hl, Wl, device, fps=fps)   # [B,3,GF*Hl*Wl,2]
    return full[:, :, IDX, :]                                  # [B,3,480,2]
pipe.transformer.rope.prepare_video_coords=custom_pvc

# --- monkeypatch _unpack_latents: crop output tokens, unpack as (outFl,Hl,Wl) ---
_orig_unpack=pipe._unpack_latents
def custom_unpack(latents, num_frames, height, width, patch_size=1, patch_size_t=1):
    return _orig_unpack(latents[:, :N_O, :], outFl, Hl, Wl, patch_size, patch_size_t)
pipe._unpack_latents=custom_unpack

def decode(lat):
    with torch.no_grad(): v=pipe.vae.decode(lat.to(DEV,DT),None,return_dict=False)[0]
    v=((v.float().clamp(-1,1)+1)/2*255).round().to(torch.uint8)[0].permute(1,2,3,0).cpu().numpy(); return [Image.fromarray(f) for f in v]
try:
    t=time.time(); lat=pipe(width=W,height=H,num_frames=TOTAL,num_inference_steps=STEPS,guidance_scale=1.0,frame_rate=25,
        conditions=[],output_type="latent",generator=torch.Generator("cpu").manual_seed(SEED),sigmas=SIG,**EMB_KW).frames
    print(f"[icu] gen OK latents={tuple(lat.shape)} {time.time()-t:.1f}s",flush=True)
    pipe.transformer.to("cpu"); torch.cuda.empty_cache()
    imgs=decode(lat); od="/root/ComfyUI/output/"+OUTDIR; os.makedirs(od,exist_ok=True)
    for i,im in enumerate(imgs): im.save(f"{od}/f{i:03d}.png")
    print(f"[icu] decoded {len(imgs)} output frames -> {od}",flush=True)
except Exception as e:
    import traceback; print("[icu] FAILED",repr(e)[:300]); traceback.print_exc()
print("ICU_DONE",flush=True)
