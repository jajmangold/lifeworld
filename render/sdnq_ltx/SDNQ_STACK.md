# SDNQ LTX-2.3 Volta Farm — Stack Reference

**Audience:** future LLM agents (and humans) working on the synthetic-news-anchor pipeline.
**Scope:** the resident LTX-2.3 generation farm on the Volta mining rig, its control/audio extensions,
downstream finishing, hard-won rules, measured performance, and open gaps. Companion to the visual
`system_map.html` in this dir. Keep this file updated when you change the stack.

---

## 0. TL;DR

- A **22B audio-video diffusion transformer (LTX-2.3-Distilled-1.1)** runs **broadcast-photoreal video on
  Volta mining GPUs** (CMP 100-210, sm_70, 16 GB, **no bf16 tensor cores, no flash-attn, PCIe Gen1 ×1**).
- The unlock: **SDNQ int4 quantization** (`OzzyGT/LTX-2.3-Distilled-1.1-sdnq-dynamic-int4`) + **one heavy
  component per card** (no CPU offload → transformer never re-streams over the ×1 riser).
- **Production engine = resident daemons**: `sdnq_resd.py` (denoise, one per gen card) + `sdnq_decoded.py`
  (VAE decode) launched by `resd_farm.sh`; `dispatch.sh` ships latents → decode → rtx0 FlashVSR → finals.
  ~**23.6 s/clip** warm at low-res, 8 distilled steps, `use_quantized_matmul=True`.
- **Extensions built this session** (both on the resident diffusers `LTX2ConditionPipeline`):
  - `ic_union.py` — **IC-LoRA depth/pose control** + **green subject-mask keying** (grounded anchor).
  - `a2v_talk.py` — **native audio-driven talking** (frozen audio latent + cross-modal attn) → real
    lip-sync + head motion + identity in ONE pass. Replaces the swap+muse bolt-on.
- **Long clips** = `sdnq_llong.py`: latent-sliding windows (no per-window decode) + ONE framewise decode.

Container: `sdnq-mgpu` (has the model cache, diffusers, sdnq, torchaudio). All scripts live in
`/root/ComfyUI/` inside the container; source of truth is `render/sdnq_ltx/` in the repo.

---

## 1. The rig & the hard constraints

- **16 GPUs**: 11× CMP 100-210 (Volta sm_70, 16 GB, mining card), 4× Tesla V100 (16 GB), 1× K620. Each on a
  **PCIe Gen1 ×1 riser (~0.2 GB/s)** — moving a 6–12 GB module across it costs tens of seconds.
- **No bf16 tensor cores** → bf16 matmul is emulated/slow; **no flash-attention on sm_70** → attention is
  O(N²) in memory (this sets the resolution ceiling, see §7).
- **No NVENC** → all video encode is CPU ffmpeg.
- Implication: **keep every heavy module resident on its own card; never offload across the riser mid-stream.**

---

## 2. The core unlock

1. **SDNQ int4** (`apply_sdnq_options_to_model(transformer, dtype=bf16, use_quantized_matmul=True)`): the 22B
   DiT fits in ~11.5 GB, leaving ~4.5 GB for activations on a 16 GB card.
   - `use_quantized_matmul=True` → Triton int4 matmul (FAST, production). Used by all `sdnq_*` production
     scripts.
   - `use_quantized_matmul=False` → eager dequant path (SLOWER, ~2×). `ic_union.py`/`a2v_talk.py` currently
     use `False` — **TODO: test `True` on those for a ~2× speedup** (see §10).
2. **Component placement** (resident daemons): DiT on its own card (needs all 16 GB for activations);
   Gemma-3 text encoder, connectors, and VAE each on other cards OR skipped/cached. Only ~1 MB of text
   conditioning crosses the ×1 riser per clip.
3. **Distilled 8-step schedule** — MUST use the official `DISTILLED_SIGMAS_8 =
   [1.0,0.99375,0.9875,0.98125,0.975,0.909375,0.725,0.421875]`, NOT `linspace`. Wrong schedule →
   watermark-ghost artifacts. Stage-2 refine uses `STAGE_2_DISTILLED_SIGMAS` (3 steps from sigma 0.909).

---

## 3. Production farm (resident daemons)

### Engines
- **`sdnq_resd.py`** — RESIDENT DENOISE DAEMON. Transformer + VAE resident on ONE card (gemma dropped,
  connectors cached+bypassed, audio_vae/vocoder parked on CPU). Watches a jobs dir, denoise-only, writes
  **raw LTX latents** `out/<job>.pt` + `.meta.json`. ~23.6 s/clip warm, VmRSS ~2 GB (so all cards run).
  One daemon per `CUDA_VISIBLE_DEVICES`.
  - **Job = JSON** in `JOBS/`: `{w,h,frames,steps,seed,out}` + optional frame conditioning:
    `cond` (frame-0 image path), `cond_strength` (def 0.6), `cond_last`+`cond_strength_last` (def 0.8),
    `cond_mid`+`cond_strength_mid` (def 0.5). Frames must be **8k+1** (e.g. 25, 57, 73, 89).
- **`sdnq_decoded.py`** — RESIDENT VAE DECODE DAEMON. LTX VAE (~2 GB) resident on ONE card, atomically
  claims `.pt` latents from a shared inbox, writes low-res PNG frames. One decode card feeds several gen
  cards (decode ~12 s vs gen ~24 s → run ~2 gen : 1 decode).

### Launch / operate
- **`resd_farm.sh [N_GEN=6] [N_DEC=3]`** — kills old daemons, starts N_GEN denoise (cards 0..N_GEN-1) +
  N_DEC decode (cards N_GEN..). `CUDA_VISIBLE_DEVICES` within the container are 0..(N-1), mapped in the
  order the container was created. Wait ~85 s for `RESIDENT` in `/tmp/resd*.log`.
- **`dispatch.sh [once|watch]`** (host-side, env `NGEN=6 NFVSR=4 FVSR_PORT0=8811`) — routes gen latents →
  decode inbox → decode daemons → **rtx0 FlashVSR** (NFVSR workers round-robin, 2× upscale) → `finals/`.
  Needs docker (`sdnq-mgpu`) + ssh/scp to `rtx0`.

### CRITICAL: `CUDA_VISIBLE_DEVICES` ≠ `nvidia-smi -i` index
CUDA enumerates by PCI/fastest-first; nvidia-smi by its own order. e.g. daemon launched with
`CUDA_VISIBLE_DEVICES=5` showed up as **nvidia-smi index 4**. When freeing a card for a test, kill the
daemon then poll the physical card that actually dropped to ~7 MiB.

---

## 4. Generation modes (choose per need)

| Script | Mode | When |
|---|---|---|
| `sdnq_resd.py` + farm | resident denoise-only, low-res, farm-parallel | **PRODUCTION** — many short clips |
| `sdnq_gen.py` / `sdnq_prod2.py` | one-shot offload path, embed+connector caches | single clip, offload base |
| `sdnq_long.py` | temporal windows, **decode+re-encode per window** | long clip, simpler (N VAE round-trips) |
| `sdnq_llong.py` | **latent-sliding windows, ONE framewise decode** | long clip, EFFICIENT (see §5) |
| `sdnq_two_stage.py` | half-res denoise → latent ×2 upscaler → full-res refine | sharper full-res |
| `sdnq_ltx_resident.py` | resident + windowed, component-per-card | resident long |
| `resident_prod.py`, `sdnq_daemon.py`, `sdnq_farmd.py` | earlier resident/daemon strategies | reference |
| `ic_union.py` | IC-LoRA depth/pose control + green subject-mask | grounded/posed anchor (§6) |
| `a2v_talk.py` | native audio-driven talking | lip-synced anchor (§6) |

---

## 5. Long clips: latent-sliding windows + single decode (`sdnq_llong.py`)

**This is the "latent window overlapping thing + only sample the VAE at the end".** It already exists.

- Each window runs `output_type='latent'` (NO decode). The next window is conditioned on the PREVIOUS
  window's **tail latents** by monkeypatching `pipe.vae.encode` to return them (VAE never touched
  mid-stream). `all_lat = torch.cat([all_lat, lat[:, :, OVLL:]], dim=2)` drops the overlap.
- **ONE `vae.decode` at the very end.** The transformer-eviction/VAE-on-GPU cost is paid ONCE.
- **Decode-OOM fix (important):** a single 32-latent decode OOMs with only spatial tiling. `sdnq_llong.py`
  sets `pipe.vae.use_framewise_decoding = True` + `pipe.vae.tile_sample_min_num_frames = 16` → temporal
  tiling → fits + no seams. (My standalone decode probe OOM'd precisely because it lacked this.)
- Args: `W H TOTAL_FRAMES WIN OVL_LAT STEPS [SEED]`. Uses `use_quantized_matmul=True` + offload base.
- Verifies continuity via adjacent-frame row-correlation at window seams.

**Gap:** `sdnq_llong.py` windows are TEXT-driven (fixed prompt). For **audio-driven long talking**, combine
its windowing with `a2v_talk.py`'s audio freeze — each window needs its audio slice + the tail-latent
continuation. Not yet integrated (see §10).

---

## 6. Control & audio extensions (this session)

### 6a. IC-LoRA depth/pose control — `ic_union.py`  (memory: `ic-lora-resident-control`)
- Union-control IC-LoRA (`ltx-2.3-22b-ic-lora-union-control-ref0.5`) engages on the SDNQ farm.
- **The fix (after 4 failed attempts):** feed the control as COMPACT guide tokens at **OVERLAPPING rope
  coords** — guide frames `[0,gFl)` at the `::2` even grid positions, SAME temporal coords as the output
  (frame_idx=0). Monkeypatch `prepare_latents` / `rope.prepare_video_coords` / `_unpack_latents`.
- **Green subject-mask keying (`SUBJECT=1`):** drop the BACKGROUND guide tokens (keep guides only where the
  depth is figure) → the unconstrained bg renders GREEN from the prompt → clean chroma key, no dark halo.
- Env: `NPX`(8k+1) `CTRL`(control-frames dir) `SUBJECT` `SEED` `OUTDIR` `STR` `DS` `FIGTHR`.

### 6b. Native audio-driven talking — `a2v_talk.py`  (memory: `a2v-audio-driven-talking-sdnq`)
- **Replaces swap+muse.** LTX-2.3 is natively audio-video; a driving audio → lips+head+expression in one
  pass via bidirectional cross-modal attention. Official ref: `~/archives/ltx2-trainer` (Lightricks/LTX-2)
  `a2vid_two_stage.py` — "video-only denoising, audio FROZEN".
- **Audio encode (validated, mel round-trip corr=0.9996):** wav → resample 16 kHz → stereo → torchaudio
  `MelSpectrogram(n_fft=1024, win=1024, hop=160, n_mels=64, f_min=0, f_max=8000, hann, center, reflect,
  power=1.0, slaney, slaney)` → `log(clamp(mel,1e-5))` → permute `[B,C,time,64]` → `audio_vae.encode`
  (argmax) → `[1,8,L,16]` (L≈25 latent/sec, aligns 1:1 with 25 fps). **Params are from
  `ltx_core/model/audio_vae/ops.py` — do NOT guess them.** torchaudio.load needs torchcodec (absent) → use
  stdlib `wave`. Vocoder is `LTX2VocoderWithBWE` (48 kHz); not needed — mux the original TTS.
- **Freeze:** wrap `transformer.forward` to override `audio_hidden_states` = clean packed+normalized audio
  (`_normalize_audio_latents(_pack_audio_latents(raw))`) and `audio_timestep = zeros` EVERY step. Pass
  `audio_latents=raw` to `__call__` (sets audio_num_frames/coords). `modality_scale=3.0` enables native
  modality-isolation guidance (steers toward AV sync; +1 fwd pass/step, VRAM-neutral).
- **Recommended params** (official): modality_scale 3.0, audio_cfg 7.0, video_cfg 3.0, STG block [29].
- **Status:** works — identity consistent (no swap), head motion strong, lips MOVE but subtly (mouth is
  res-starved, §7). Env: `W H NPX MODSCALE SEED OUTDIR WAV`.

---

## 7. Performance (measured on the SDNQ farm, one 16 GB card)

**a2v-driven path (eager `use_quantized_matmul=False` + audio + modality) — resolution sweep, 73 frames:**

| Res | tokens | full 8-step (mod=3 / mod=1) | peak VRAM | mouth px (latent) | fits? |
|---|---|---|---|---|---|
| 256×384 | 960 | 108 s / 66 s | 14.7 GB | 35 (1.1) | ✓ |
| 320×448 | 1400 | 122 s / 65 s | 15.0 GB | 43 (1.35) | ✓ |
| 384×512 | 1920 | 163 s / 86 s | **15.6 GB** | 52 (1.6) | ✓ **(ceiling)** |
| 448×640+ | ≥2800 | — | **OOM** | — | ✗ |

- **Hard ceiling ≈ 384×512 on 16 GB** (single fwd-pass O(N²) attention; no flash-attn on sm_70). Diffusers
  attention-slicing DOESN'T help (LTX2 uses a custom `LTX2PerturbedAttnProcessor` that ignores it).
- `modality_scale` doubles TIME, same VRAM (the two passes run sequentially).
- **Production base** (`use_quantized_matmul=True`, no audio, no modality) is much faster: **~23.6 s/clip**.
  The sweep above is the slow eager path — divide by ~2–4 for the quantized-matmul base.
- **VAE decode** (framewise): per 10-latent chunk 3.8 s (256×384) / 5.3 s (320×448) / 7.1 s (384×512).
  A single 32-latent decode OOMs → framewise/temporal tiling required (§5).
- **10 s clip (≈32 latent frames, ~4 latent-windows), resident, one card, eager+mod=3 extrapolated:**
  ~7.5 min (256×384) / ~8.5 min (320×448) / ~11.3 min (384×512); ~half at mod=1; decode adds only ~15–25 s.
  **Across 6 gen cards in parallel → ~1.3–1.9 min amortized per 10 s clip.** On the quantized-matmul base
  path these drop substantially — re-measure once a2v runs with `use_quantized_matmul=True`.

**Crisp lip-sync (mouth ≥ ~2.5 latent px, i.e. 512×768) does NOT fit one 16 GB card.** Real fix =
multi-GPU tensor/sequence parallelism across the farm (the rig is built for fan-out). FlashVSR upscaling
sharpens but does NOT add lip MOTION (motion is generated at the base res).

---

## 8. Downstream finishing (host `studio/`, mostly job-queue over docker)

- **swap-server** (`swap/run_swap.sh`, `:8088`): face-swap. Job JSON → `output/swap_jobs/<name>.json`
  `{src,video,out,keepeyes,enhance,feather,passes,save_faces}`; `/o/` = `output/`. `passes:2` = double-swap.
- **muse-server** (MuseTalk, `:9090`): lip-sync. Job → `output/muse_jobs/` `{video,audio,out}`; `/io/`=output.
  (Superseded by `a2v_talk.py` for anchor lip-sync — muse needs a big face + degrades small/low-res mouths.)
- **qwen3-tts** (voxserver on **amd1**, `http://amd1:8064` — `/mono` POST; override `TTS_URL`):
  `newscast/synth_voice.py --text --ref /work/refX.wav`.
- **FlashVSR** (rtx0, workers `:8811-881N`): `flashvsr_face.sh <stem>` (face-crop, needs a detectable face)
  OR full-frame `curl :8811/upscale?mode=video&scale=2`. 2× upscale.
- **Graphics/composite**: `newscast/broadcast_finish.sh` (ticker/lower-third/bug, 1440p landscape),
  `newscast/broadcast_gfx.py`. For portrait clips, hand-composite with ffmpeg chroma-key over
  `newscast/assets/studio_bg.png` (clean set) — see `output/grounded_anchor_full.mp4` recipe.
- **make_anchor.sh** (host): the OLD Blender-render → swap → muse → FlashVSR anchor pipeline (CG-avatar
  path). The SDNQ farm + a2v is the newer, photoreal-direct path.
- **a2f_lipsync.py** (ARCHIVED → `bot-archive/render-experiments/`): Audio2Face-3D ONNX → ARKit JSON — was
  the SMPL-X/Blender avatar lip path, superseded by `a2v_talk.py`. Old note: for the SMPL-X path, not
  the LTX path).

---

## 9. Hard-won rules (violating these has cost reboots / hours)

1. **NEVER `kill -9` a live CUDA process** — corrupts the driver (GPUs go "Unknown Error", container wedges,
   needs a host reboot). Always **SIGTERM + poll VRAM until <2 GB** before reusing a card.
2. **Free-a-card pattern for tests:** `pkill -TERM -f resdjobs5` → poll the physical card to <2 GB → run
   test with `CUDA_VISIBLE_DEVICES=5` → restart the daemon after. Keep the farm at 6 gen + 3 decode.
3. **Prompt must be BYTE-IDENTICAL** to reuse the `embeds_<md5>.pt` / `conn_<md5>.pt` caches. The connectors
   module is 6.35 GB over ×1 — caching its output is a huge win; a changed prompt silently re-streams it.
4. **8-step distilled sigmas, not linspace** (watermark-ghost otherwise). Frames must be **8k+1**.
5. **Long decode needs framewise/temporal tiling** (`use_framewise_decoding=True`), else temporal OOM.
6. **container GPU index ≠ CUDA_VISIBLE_DEVICES** (§3).
7. Recreate `sdnq-mgpu` with `--gpus '"device=1,2,3,..."'` (index form in a script file; UUID form trips the
   CDI modifier).

---

## 10. Known gaps & next steps

- **a2v + `use_quantized_matmul=True`**: `a2v_talk.py`/`ic_union.py` use the slow eager path. Test the
  Triton path for ~2× — verify the audio cross-attn + LoRA still work quantized.
- **a2v long clips**: integrate `a2v_talk.py` audio-freeze into `sdnq_llong.py` windowing (per-window audio
  slice + tail-latent continuation) → arbitrary-length talking. Not built.
- **Crisp lip-sync**: needs 512×768 = multi-GPU parallelism (single card caps at 384×512). Scope a
  tensor/sequence-parallel split across 2–3 farm cards.
- **Verify a2v is truly audio-DRIVEN** (not idle motion): silent/different-audio differential.
- **a2v identity control**: currently identity comes from the baked prompt; add a reference-image condition
  (head-shot) for locked identity per talent.

---

## 11. File index (`render/sdnq_ltx/`)

```
sdnq_resd.py        resident denoise daemon (PROD engine)      resd_farm.sh   farm launcher
sdnq_decoded.py     resident VAE decode daemon                 dispatch.sh    latents->decode->rtx0 FlashVSR
sdnq_gen.py         one-shot offload base + caches             sdnq_farm.sh   (older farm launcher)
sdnq_prod2.py       prod offload variant                       system_map.html visual rig/placement map
sdnq_long.py        windowed (decode-per-window)               sdnq_two_stage.py half-res->upscaler->refine
sdnq_llong.py       windowed LATENT-sliding + 1 framewise decode  <-- long-clip engine
sdnq_ltx_resident.py resident + windowed, component-per-card   sdnq_daemon.py  resident daemon (early)
sdnq_farmd.py       RAM-slim + daemon                          resident_prod.py transformer-resident, VAE shuttle
sdnq_ltx_pilot.py   first sm_70 pilot                          ic_union.py    IC-LoRA control + green mask
a2v_talk.py         native audio-driven talking                bake_caches.py (in-container) embed/conn baker
```
Caches live at `/root/ComfyUI/output/embeds_<md5>.pt`, `conn_<md5>.pt` (keyed on prompt md5).
Model: `OzzyGT/LTX-2.3-Distilled-1.1-sdnq-dynamic-int4` (HF cache in `sdnq-mgpu`). Official trainer/pipelines
checked out at `~/archives/ltx2-trainer` (authoritative for mel params, A2Vid, IC-LoRA internals).
