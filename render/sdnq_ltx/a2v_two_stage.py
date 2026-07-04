"""T0.1b spike — a2v (audio-driven talking) + two-stage SPATIAL UPSCALER for CRISP 512x768 lips on one card.
Stage 1: a2v denoise at 256x384 (audio-frozen) -> latents. Spatial x2 latent upsampler -> 512x768 latents.
Stage 2: 3-step refine at 512x768 (audio STILL frozen) -> decode. Goal: keep the lip motion, gain sharpness.
Uses offload (fits 512x768 like sdnq_two_stage). Env: QMM(1) NPX(57) MODSCALE(3) SEED WAV OUTDIR."""
import glob, inspect, os, time, types, wave, numpy as np, torch, torchaudio, sdnq
from diffusers import DiffusionPipeline, LTX2ConditionPipeline
from diffusers.pipelines.ltx2.pipeline_ltx2_condition import retrieve_latents
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from sdnq.loader import apply_sdnq_options_to_model
from safetensors.torch import load_file
from PIL import Image
DT=torch.bfloat16; DEV="cuda:0"
W,H = int(os.environ.get("W","512")), int(os.environ.get("H","768"))   # FINAL res (stage1 = half)
NPX=int(os.environ.get("NPX","57")); FR=25; MODSCALE=float(os.environ.get("MODSCALE","3.0"))
SEED=int(os.environ.get("SEED","5")); OUTDIR=os.environ.get("OUTDIR","a2v_2stage")
WAV=os.environ.get("WAV","/root/ComfyUI/output/anchor_line.wav")
QMM=bool(int(os.environ.get("QMM","1")))
PATH=sorted(glob.glob("/root/.cache/huggingface/hub/models--OzzyGT--LTX-2.3-Distilled-1.1-sdnq-dynamic-int4/snapshots/*"))[0]
UPS=glob.glob("/root/.cache/huggingface/hub/models--Lightricks--LTX-2.3/snapshots/*/ltx-2.3-spatial-upscaler-x2-1.1.safetensors")[0]
EMB="/root/ComfyUI/output/embeds_fcb0316dc2.pt"; CONN="/root/ComfyUI/output/conn_fcb0316dc2.pt"
DISTILLED_8=[1.0,0.99375,0.9875,0.98125,0.975,0.909375,0.725,0.421875]
STAGE2_3=[0.909375,0.725,0.421875]
SR=16000; NFFT=1024; HOP=160; NMELS=64

def load_wav(p):
    w=wave.open(p,'rb'); sr=w.getframerate(); ch=w.getnchannels(); n=w.getnframes()
    a=np.frombuffer(w.readframes(n),dtype=np.int16).astype(np.float32)/32768.0
    return torch.from_numpy(a.reshape(-1,ch).T.copy()), sr

base=DiffusionPipeline.from_pretrained(PATH,torch_dtype=DT,text_encoder=None)
base.transformer=apply_sdnq_options_to_model(base.transformer,dtype=DT,use_quantized_matmul=QMM)
comp={k:v for k,v in base.components.items() if k in set(inspect.signature(LTX2ConditionPipeline.__init__).parameters)};comp["text_encoder"]=None
pipe=LTX2ConditionPipeline(**comp)
_ct=tuple(torch.load(CONN,map_location="cpu"));pipe.connectors=lambda pe_,pm_,padding_side="left":tuple(x.to(pe_.device) for x in _ct)
pe,pm,ne,nm=torch.load(EMB,map_location="cpu")
g=lambda x:(x.to(DEV,DT) if (torch.is_tensor(x) and x.is_floating_point()) else (x.to(DEV) if torch.is_tensor(x) else None))
EMB_KW=dict(prompt_embeds=g(pe),prompt_attention_mask=g(pm))
if ne is not None: EMB_KW.update(negative_prompt_embeds=g(ne),negative_prompt_attention_mask=g(nm))

# ---- encode driving audio (audio_vae briefly on GPU, before offload) ----
pipe.audio_vae.to(DEV)
wav,sr=load_wav(WAV)
if sr!=SR: wav=torchaudio.functional.resample(wav,sr,SR)
if wav.shape[0]==1: wav=wav.repeat(2,1)
wav=wav[:2].unsqueeze(0).to(DEV,torch.float32)
mel_t=torchaudio.transforms.MelSpectrogram(sample_rate=SR,n_fft=NFFT,win_length=NFFT,hop_length=HOP,f_min=0.0,f_max=SR/2.0,
    n_mels=NMELS,window_fn=torch.hann_window,center=True,pad_mode="reflect",power=1.0,mel_scale="slaney",norm="slaney").to(DEV)
mel=torch.log(torch.clamp(mel_t(wav),min=1e-5)).permute(0,1,3,2).contiguous()
with torch.no_grad():
    raw_audio=retrieve_latents(pipe.audio_vae.encode(mel.to(pipe.audio_vae.dtype)), sample_mode="argmax").float()
CLEAN=pipe._normalize_audio_latents(pipe._pack_audio_latents(raw_audio), pipe.audio_vae.latents_mean, pipe.audio_vae.latents_std).to(DT)
pipe.audio_vae.to("cpu")
print(f"[a2v2] audio raw={tuple(raw_audio.shape)} clean={tuple(CLEAN.shape)} QMM={QMM} final={W}x{H} NPX={NPX}",flush=True)

# ---- spatial x2 upsampler ----
ups=LTX2LatentUpsamplerModel(in_channels=128,mid_channels=1024,num_blocks_per_stage=4,dims=3,
                             spatial_upsample=True,temporal_upsample=False,use_rational_resampler=False)
ups.load_state_dict(load_file(UPS),strict=True); ups=ups.to(DT).eval()

# ---- FREEZE audio (device-aware for offload) ----
_orig_fwd=pipe.transformer.forward
def fwd(self,*a,audio_hidden_states=None,audio_timestep=None,**kw):
    if audio_hidden_states is not None:
        ah=CLEAN
        if ah.shape[0]!=audio_hidden_states.shape[0]: ah=ah.repeat(audio_hidden_states.shape[0]//ah.shape[0],1,1)
        audio_hidden_states=ah.to(audio_hidden_states.device,audio_hidden_states.dtype)
    if audio_timestep is not None: audio_timestep=torch.zeros_like(audio_timestep)
    return _orig_fwd(*a,audio_hidden_states=audio_hidden_states,audio_timestep=audio_timestep,**kw)
pipe.transformer.forward=types.MethodType(fwd,pipe.transformer)
try: pipe.vae.enable_tiling()
except: pass
pipe.enable_model_cpu_offload()

AKW=dict(modality_scale=MODSCALE, audio_latents=raw_audio.to(torch.float32), conditions=[], **EMB_KW)
try:
    T=time.time()
    # STAGE 1: talking motion at half res (256x384)
    t1=time.time()
    s1=pipe(width=W//2,height=H//2,num_frames=NPX,num_inference_steps=8,sigmas=DISTILLED_8,guidance_scale=1.0,
            frame_rate=FR,output_type="latent",generator=torch.Generator("cpu").manual_seed(SEED),**AKW).frames
    print(f"[a2v2] stage1 {W//2}x{H//2} -> {tuple(s1.shape)} {time.time()-t1:.1f}s",flush=True)
    # UPSAMPLE x2 (raw latents)
    tu=time.time(); ups.to("cuda")
    with torch.no_grad(): up=ups(s1.to("cuda",DT))
    ups.to("cpu"); torch.cuda.empty_cache()
    print(f"[a2v2] upsample -> {tuple(up.shape)} {time.time()-tu:.1f}s",flush=True)
    # STAGE 2: refine at 512x768 (audio still frozen), 3 steps
    t2=time.time()
    out=pipe(width=W,height=H,num_frames=NPX,num_inference_steps=3,sigmas=STAGE2_3,guidance_scale=1.0,
             frame_rate=FR,latents=up.to(DT),output_type="pil",generator=torch.Generator("cpu").manual_seed(SEED),**AKW)
    imgs=out.frames[0]
    imgs=[f if isinstance(f,Image.Image) else Image.fromarray((np.asarray(f).clip(0,1)*255).astype("uint8")) for f in imgs]
    od="/root/ComfyUI/output/"+OUTDIR; os.makedirs(od,exist_ok=True)
    for i,im in enumerate(imgs): im.save(f"{od}/f{i:03d}.png")
    print(f"[a2v2] stage2 {W}x{H} refine+decode {time.time()-t2:.1f}s | {len(imgs)}f total {time.time()-T:.1f}s -> {od}",flush=True)
except Exception as e:
    import traceback; print("[a2v2] FAILED",repr(e)[:300]); traceback.print_exc()
print("A2V2_DONE",flush=True)
