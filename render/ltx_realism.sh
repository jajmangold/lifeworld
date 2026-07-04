#!/bin/bash
# LTX-2.3 v2v REALISM PASS over a talking-head video. Photorealizes a 3D-render anchor (skin/hair/lighting)
# while PRESERVING identity/pose/wardrobe/background via a low-denoise video-to-video pass. This is the route
# to full photorealism the render->swap->muse stack can't reach (our base is a raster render). Runs headless in
# the wan2gp docker on rtx0 (LTX-2.3 distilled Q4 gguf, fits one 3060). See memory [[ltx-realism-pass]].
#
#   ltx_realism.sh <in.mp4> <out.mp4> [denoise=0.5] [gpu=0]
#
# The LTX distilled pipeline changes the frame count/timing (a temporal upsampler is baked in), so we RETIME
# the output back to the input's exact duration + fps and re-mux the input's audio -> guaranteed A/V sync.
set -e
IN=$1; OUT=$2; DEN=${3:-0.65}; GPU=${4:-0}   # 0.65 = realism sweet spot (control mode holds identity; >0.65 plateaus)
SPATIAL=${NEWS_LTX_SPATIAL-}       # VAE spatial upscale ('vae2') is NOT available for video in this LTX build;
GRAIN=${NEWS_LTX_GRAIN:-0.03}      # instead get real detail by generating at higher base res (NEWS_LTX_RES).
MODEL=${NEWS_LTX_MODEL:-ltx2_22B_distilled_gguf_q3_k_s}   # DEFAULT Q3_K_S: quality==Q4, fits RESIDENT on-card
                                                          # (profile 3, ~10.5GB) so a FARM of 4 servers (1/GPU) runs.
TGT=${NEWS_LTX_RES:-1024}          # gen long-edge px = the speed<->realism dial. 1024=~4.2min (fast, photoreal),
                                   # 1152=~6min (hero shots), 768=too soft, 1536=slow over-crank. Realism comes
                                   # from denoise/reference, not brute res. LTX-2.3 has NO teacache; sage attn is on.
[ -z "$IN" ] || [ -z "$OUT" ] && { echo "usage: ltx_realism.sh <in.mp4> <out.mp4> [denoise] [gpu]"; exit 1; }
BOT=/srv/nvme-data/containers/live/studio; cd "$BOT"
RTX="ssh -o BatchMode=yes josh@rtx0"; W=/mnt/datadisk/containers/wan2gp
NAME=$(basename "$OUT" .mp4)
log(){ echo "[ltx $(date +%H:%M:%S)] $*"; }

# source timing (to conform the output back to)
FPS=$(ffprobe -v error -select_streams v -show_entries stream=r_frame_rate -of csv=p=0 "$IN" | head -1 | awk -F/ '{printf "%.4f", ($2?$1/$2:$1)}')
DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$IN")
W0=$(ffprobe -v error -select_streams v -show_entries stream=width -of csv=p=0 "$IN")
H0=$(ffprobe -v error -select_streams v -show_entries stream=height -of csv=p=0 "$IN")
# LTX wants dims divisible by 32 and length = 8n+1. Downscale long edge to <=1280 keeping AR, round to /32.
LW=$(python3 -c "w=$W0;s=$TGT/w; print(int(round(w*s/32))*32)")
LH=$(python3 -c "w=$W0;h=$H0;s=$TGT/w; print(int(round(h*s/32))*32)")
NF0=$(python3 -c "print(int(round($DUR*$FPS)))")
NF=$(python3 -c "n=$NF0; print(((n-1)//8)*8+1)")   # nearest 8n+1 <= n
log "in ${W0}x${H0} ${NF0}f @${FPS}fps ${DUR}s -> LTX ${LW}x${LH} ${NF}f denoise=$DEN gpu=$GPU"

# 1) prep LTX source (scaled, exact 8n+1 frames), stage to rtx0
SRC=output/${NAME}_ltxsrc.mp4
ffmpeg -y -i "$IN" -vf "scale=${LW}:${LH},fps=${FPS}" -frames:v "$NF" -an "$SRC" 2>/dev/null
$RTX "mkdir -p $W/ltxjobs" ; scp -q "$SRC" josh@rtx0:$W/ltxjobs/${NAME}_src.mp4

# 2) build the task queue (copy an existing task's full param dict, override for a realism v2v pass)
PROMPT="Photorealistic broadcast video of a professional news anchor speaking to camera, natural realistic skin with visible pores and subtle texture, real individual hair strands, soft natural lighting, shot on a professional broadcast camera, documentary realism, sharp detailed eyes. Keep the same person, expression, pose, framing and background."
NEG="cartoon, cgi, 3d render, plastic skin, waxy, blurry, distorted, extra people"
# 2) RESIDENT SERVER: keep the 13GB LTX model loaded across clips (~7min/clip saved: cold job ~574s, warm
#    ~155s). Start ltx_server.py if it isn't up, then submit this clip as a settings-overrides job. VGU control
#    mode = per-frame conditioning (no drift). The server merges these over the model's default settings.
# any ready server (single or a farm from ltx_farm.sh) will grab the job from the shared queue (atomic claim).
if ! $RTX "grep -laq LTX_SERVER_READY $W/ltxjobs/server*.log 2>/dev/null"; then
  log "no LTX server up — starting one on gpu$GPU (warm-up ~model load)..."
  scp -q "$BOT/render/ltx_server.py" josh@rtx0:$W/ltx_server.py
  $RTX "docker exec -d -e CUDA_VISIBLE_DEVICES=$GPU -e LTX_PROFILE=3 wan2gp bash -lc 'cd /workspace && python3 ltx_server.py > /workspace/ltxjobs/server.log 2>&1'"
  for i in $(seq 1 90); do $RTX "grep -laq LTX_SERVER_READY $W/ltxjobs/server*.log 2>/dev/null" && break; sleep 6; done
fi
$RTX "rm -f $W/ltxjobs/${NAME}.out $W/ltxjobs/${NAME}.err"
$RTX "cd $W && python3 - <<PY
import json
# CONTROL-VIDEO v2v: video_guide + video_prompt_type 'VGU' (V=control video, G=guide/denoise, U=raw frames)
# conditions EVERY frame on the corresponding source frame -> no drift over long clips.
job={'model_type':'${MODEL}','prompt':'''$PROMPT''','negative_prompt':'''$NEG''',
 'video_guide':'/workspace/ltxjobs/${NAME}_src.mp4','video_prompt_type':'VGU','image_prompt_type':'',
 'keep_frames_video_guide':'','denoising_strength':$DEN,'resolution':'${LW}x${LH}','video_length':$NF,
 'seed':42,'num_inference_steps':8,'guidance_scale':1.0,'temporal_upsampling':'',
 'spatial_upsampling':'${SPATIAL}','film_grain_intensity':${GRAIN},'film_grain_saturation':0.5}
json.dump(job, open('ltxjobs/${NAME}.job.json','w')); print('JOB_SUBMITTED')
PY"

# 3) wait for the resident server to finish this clip
log "LTX v2v via resident server..."
for i in $(seq 1 240); do
  $RTX "test -f $W/ltxjobs/${NAME}.out -o -f $W/ltxjobs/${NAME}.err" && break; sleep 10
done
$RTX "test -f $W/ltxjobs/${NAME}.out" || { log "LTX FAILED:"; $RTX "head -20 $W/ltxjobs/${NAME}.err 2>/dev/null"; exit 1; }

# 4) pull raw output, retime to source duration + fps, re-mux the input's audio. The server copies its result to
#    a deterministic clean name ($W/ltxjobs/<name>_out.mp4 on the host = /workspace/ltxjobs/... in the container).
$RTX "test -s $W/ltxjobs/${NAME}_out.mp4 && cp $W/ltxjobs/${NAME}_out.mp4 $W/ltxjobs/${NAME}_raw.mp4 && chmod 644 $W/ltxjobs/${NAME}_raw.mp4" \
  || { log "no LTX output mp4"; exit 1; }
scp -q josh@rtx0:$W/ltxjobs/${NAME}_raw.mp4 output/${NAME}_ltxraw.mp4
RDUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 output/${NAME}_ltxraw.mp4)
RATIO=$(python3 -c "print($DUR/$RDUR)")
log "retime LTX ${RDUR}s -> ${DUR}s (x$RATIO) @${FPS}fps + mux audio"
ffmpeg -y -i output/${NAME}_ltxraw.mp4 -i "$IN" -filter_complex "[0:v]setpts=PTS*${RATIO},fps=${FPS}[v]" \
  -map "[v]" -map 1:a? -c:v libx264 -pix_fmt yuv420p -crf 17 -c:a aac -shortest "$OUT" 2>/dev/null
log "DONE -> $OUT ($(ffprobe -v error -show_entries stream=width,height -of csv=p=0:s=x "$OUT" | head -1))"
