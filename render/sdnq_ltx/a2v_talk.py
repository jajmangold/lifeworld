import glob, inspect, os, time, types, wave, numpy as np, torch, torchaudio, sdnq
from diffusers import DiffusionPipeline, LTX2ConditionPipeline, AutoencoderKLLTX2Audio
from diffusers.pipelines.ltx2.pipeline_ltx2_condition import retrieve_latents
from sdnq.loader import apply_sdnq_options_to_model
from PIL import Image
DT=torch.bfloat16; DEV="cuda:0"
W,H,STEPS=int(os.environ.get("W","256")),int(os.environ.get("H","384")),8
NPX=int(os.environ.get("NPX","73"))          # pixel frames == 8k+1 ; 73 -> 2.92s @25fps
FR=25; MODSCALE=float(os.environ.get("MODSCALE","3.0"))
SEED=int(os.environ.get("SEED","5")); OUTDIR=os.environ.get("OUTDIR","a2v_out")
WAV=os.environ.get("WAV","/root/ComfyUI/output/anchor_line.wav")
PATH=sorted(glob.glob("/root/.cache/huggingface/hub/models--OzzyGT--LTX-2.3-Distilled-1.1-sdnq-dynamic-int4/snapshots/*"))[0]
EMB="/root/ComfyUI/output/embeds_fcb0316dc2.pt"; CONN="/root/ComfyUI/output/conn_fcb0316dc2.pt"
SIG=[1.0,0.99375,0.9875,0.98125,0.975,0.909375,0.725,0.421875]
SR=16000; NFFT=1024; HOP=160; NMELS=64

def load_wav(p):
    w=wave.open(p,'rb'); sr=w.getframerate(); ch=w.getnchannels(); n=w.getnframes()
    a=np.frombuffer(w.readframes(n),dtype=np.int16).astype(np.float32)/32768.0
    return torch.from_numpy(a.reshape(-1,ch).T.copy()), sr

base=DiffusionPipeline.from_pretrained(PATH,torch_dtype=DT,text_encoder=None)
base.transformer=apply_sdnq_options_to_model(base.transformer,dtype=DT,use_quantized_matmul=bool(int(os.environ.get("QMM","0"))))
comp={k:v for k,v in base.components.items() if k in set(inspect.signature(LTX2ConditionPipeline.__init__).parameters)};comp["text_encoder"]=None
pipe=LTX2ConditionPipeline(**comp)
_ct=tuple(torch.load(CONN,map_location="cpu"));pipe.connectors=lambda pe_,pm_,padding_side="left":tuple(x.to(pe_.device) for x in _ct)
pipe.transformer.to(DEV); pipe.vae.to(DEV); pipe.audio_vae.to(DEV)
try: pipe.vae.enable_tiling()
except: pass
pipe.__class__._execution_device=property(lambda self: torch.device(DEV))
pe,pm,ne,nm=torch.load(EMB,map_location="cpu")
g=lambda x:(x.to(DEV,DT) if (torch.is_tensor(x) and x.is_floating_point()) else (x.to(DEV) if torch.is_tensor(x) else None))
EMB_KW=dict(prompt_embeds=g(pe),prompt_attention_mask=g(pm))
if ne is not None: EMB_KW.update(negative_prompt_embeds=g(ne),negative_prompt_attention_mask=g(nm))

# ---- encode driving audio (validated official mel path) ----
wav,sr=load_wav(WAV)
if sr!=SR: wav=torchaudio.functional.resample(wav,sr,SR)
if wav.shape[0]==1: wav=wav.repeat(2,1)
wav=wav[:2].unsqueeze(0).to(DEV,torch.float32)
mel_t=torchaudio.transforms.MelSpectrogram(sample_rate=SR,n_fft=NFFT,win_length=NFFT,hop_length=HOP,f_min=0.0,f_max=SR/2.0,
    n_mels=NMELS,window_fn=torch.hann_window,center=True,pad_mode="reflect",power=1.0,mel_scale="slaney",norm="slaney").to(DEV)
mel=torch.log(torch.clamp(mel_t(wav),min=1e-5)).permute(0,1,3,2).contiguous()
with torch.no_grad():
    raw_audio=retrieve_latents(pipe.audio_vae.encode(mel.to(pipe.audio_vae.dtype)), sample_mode="argmax")  # [1,8,L,16]
raw_audio=raw_audio.float()
Laud=raw_audio.shape[2]
# clean packed+normalized audio the transformer should see EVERY step (frozen conditioning)
CLEAN=pipe._normalize_audio_latents(pipe._pack_audio_latents(raw_audio), pipe.audio_vae.latents_mean, pipe.audio_vae.latents_std).to(DEV,DT)
print(f"[a2v] audio raw={tuple(raw_audio.shape)} Laud={Laud} clean_packed={tuple(CLEAN.shape)} MODSCALE={MODSCALE}",flush=True)

# ---- FREEZE audio: wrap transformer.forward to always feed clean audio at timestep 0 ----
_orig_fwd=pipe.transformer.forward
def fwd(self, *a, audio_hidden_states=None, audio_timestep=None, **kw):
    if audio_hidden_states is not None:
        ah=CLEAN
        if ah.shape[0]!=audio_hidden_states.shape[0]:
            ah=ah.repeat(audio_hidden_states.shape[0]//ah.shape[0],1,1)
        audio_hidden_states=ah.to(audio_hidden_states.dtype)
    if audio_timestep is not None:
        audio_timestep=torch.zeros_like(audio_timestep)
    return _orig_fwd(*a, audio_hidden_states=audio_hidden_states, audio_timestep=audio_timestep, **kw)
pipe.transformer.forward=types.MethodType(fwd,pipe.transformer)

def decode(lat):
    with torch.no_grad(): v=pipe.vae.decode(lat.to(DEV,DT),None,return_dict=False)[0]
    v=((v.float().clamp(-1,1)+1)/2*255).round().to(torch.uint8)[0].permute(1,2,3,0).cpu().numpy(); return [Image.fromarray(f) for f in v]
try:
    t=time.time()
    out=pipe(width=W,height=H,num_frames=NPX,num_inference_steps=STEPS,guidance_scale=1.0,frame_rate=FR,
        modality_scale=MODSCALE, audio_latents=raw_audio.to(torch.float32),
        conditions=[],output_type="latent",generator=torch.Generator("cpu").manual_seed(SEED),sigmas=SIG,**EMB_KW)
    lat=out.frames
    print(f"[a2v] gen OK video_latents={tuple(lat.shape)} {time.time()-t:.1f}s",flush=True)
    pipe.transformer.to("cpu"); torch.cuda.empty_cache()
    imgs=decode(lat); od="/root/ComfyUI/output/"+OUTDIR; os.makedirs(od,exist_ok=True)
    for i,im in enumerate(imgs): im.save(f"{od}/f{i:03d}.png")
    print(f"[a2v] decoded {len(imgs)} frames -> {od}",flush=True)
except Exception as e:
    import traceback; print("[a2v] FAILED",repr(e)[:300]); traceback.print_exc()
print("A2V_DONE",flush=True)
