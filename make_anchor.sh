#!/bin/bash
# One-shot photoreal talking-anchor builder.
#   make_anchor.sh --audio out/x.wav --out out/final.mp4 [opts]
# Stages: proc_perf -> render(rtx0) -> [fast: mux] OR [full: swap -> MuseTalk] -> [OTS] -> final.mp4
#
# Options:
#   --mood neutral|warm|serious|alert   (default serious)
#   --brow 1.4   --nod 1.0   --browbase 0.05   --seed 7
#   --face anchorM.png        face-swap source (full mode; default anchorM.png)
#   --fast                    viseme mouth, render-only (skip swap+MuseTalk) ~4min vs ~full
#   --premium                 FlashVSR-face 2x diffusion upscale -> 1440p (rtx0, +~4min; replaces GFPGAN)
#   --baked [--bakedtex T]    use the pre-baked anchorM head texture (identity in the render) -> SKIP swap
#                             stage entirely (~7min/40s saved). Default tex viverse_avatar/anchorM_head_baked.png
#   --ots "img|LABEL|end; img2|LABEL2|end2"   over-the-shoulder panels (seconds = segment end times)
set -e
BOT=/srv/nvme-data/containers/projects/bot
SAMPL=/mnt/datadisk/containers/sampl
RTX="ssh -o BatchMode=yes josh@rtx0"
cd "$BOT"

MOOD=serious; BROW=1.0; NOD=1.0; BROWBASE=""; SEED=7; FACE=anchorM.png; FAST=0; PREMIUM=0; BAKED=0; BAKEDTEX=anchorM_head_baked.png; OTS=""; AUDIO=""; OUT=""
while [ $# -gt 0 ]; do case "$1" in
  --audio) AUDIO=$2; shift 2;; --out) OUT=$2; shift 2;; --mood) MOOD=$2; shift 2;;
  --brow) BROW=$2; shift 2;; --nod) NOD=$2; shift 2;; --browbase) BROWBASE=$2; shift 2;;
  --seed) SEED=$2; shift 2;; --face) FACE=$2; shift 2;; --fast) FAST=1; shift;;
  --premium) PREMIUM=1; shift;;
  --baked) BAKED=1; shift;;
  --bakedtex) BAKEDTEX=$2; shift 2;;
  --ots) OTS=$2; shift 2;; *) echo "unknown arg: $1"; exit 1;; esac; done
[ -z "$AUDIO" ] && { echo "need --audio"; exit 1; }
[ -z "$OUT" ] && { echo "need --out"; exit 1; }
NAME=$(basename "$OUT" .mp4)
AB=$(basename "$AUDIO")
mkdir -p output
log(){ echo "[make_anchor $(date +%H:%M:%S)] $*"; }

# 1) PERFORMANCE ------------------------------------------------------------------
BB=""; [ -n "$BROWBASE" ] && BB="--browbase $BROWBASE"
VIS=""; [ "$FAST" = 1 ] && VIS="--visemes"
log "proc_perf mood=$MOOD brow=$BROW nod=$NOD fast=$FAST"
docker run --rm -v "$BOT":/io -w /io mp-extract:1.0 python3 proc_perf.py \
  --audio /io/"$AUDIO" --mood "$MOOD" --brow "$BROW" --nod "$NOD" $BB $VIS --seed "$SEED" \
  --out /io/output/${NAME}.perf.json | grep -aE 'PERF_OK|Error'

# 2) RENDER on rtx0 ---------------------------------------------------------------
log "render on rtx0..."
# avatar.glb + newsroom_pano.png persist on rtx0; (re)stage them if missing
$RTX "test -f $SAMPL/avatar.glb" || scp -q viverse_avatar/avatar.glb josh@rtx0:$SAMPL/
$RTX "test -f $SAMPL/newsroom_pano.png" || scp -q output/newsroom_pano.png josh@rtx0:$SAMPL/
scp -q output/${NAME}.perf.json render_anchor_anim.py josh@rtx0:$SAMPL/
BENV=""
if [ "$BAKED" = 1 ]; then
  $RTX "test -f $SAMPL/$BAKEDTEX" || scp -q viverse_avatar/$BAKEDTEX josh@rtx0:$SAMPL/
  BENV="BAKED_HEAD_TEX=/work/$BAKEDTEX "
  log "baked-texture mode: $BAKEDTEX (identity baked into render; swap will be skipped)"
fi
$RTX "docker exec sampl bash -lc 'cd /work && rm -f output/anchor_anim/f*.png && ${BENV}CUDA_VISIBLE_DEVICES=0 /opt/blender/blender --background --python render_anchor_anim.py -- --arkit /work/${NAME}.perf.json > /work/output/${NAME}_render.log 2>&1; echo DONE_RC=\$? >> /work/output/${NAME}_render.log'"
$RTX "docker exec sampl bash -lc 'cd /work/output/anchor_anim && ffmpeg -y -framerate 25 -i f%04d.png -loop 1 -i bg.png -filter_complex \"[0:v]unpremultiply=inplace=1[c];[1:v][c]overlay=shortest=1,format=yuv420p\" -c:v libx264 -crf 18 /work/output/${NAME}_silent.mp4 2>&1 | tail -1'" >/dev/null
scp -q josh@rtx0:$SAMPL/output/${NAME}_silent.mp4 output/${NAME}_silent.mp4
log "render done -> output/${NAME}_silent.mp4"

# 3) MOUTH/IDENTITY ---------------------------------------------------------------
if [ "$FAST" = 1 ]; then
  log "fast mode: mux audio (viseme mouth, no swap/muse)"
  ffmpeg -y -i output/${NAME}_silent.mp4 -i "$AUDIO" -c:v libx264 -pix_fmt yuv420p -crf 18 -c:a aac -b:a 192k -shortest output/${NAME}_talk.mp4 2>/dev/null
else
  if [ "$BAKED" = 1 ]; then
    # identity is baked into the render -> no per-frame swap; muse runs on the render directly.
    log "baked mode: skipping swap; MuseTalk on the baked render"
    MUSE_IN=${NAME}_silent
  else
    log "swap (keepeyes, resident server) -> $FACE"
    docker ps --format '{{.Names}}' | grep -q '^swap-server$' || { log "starting swap-server..."; bash "$BOT/swap/run_swap.sh"; \
      for i in $(seq 1 20); do docker logs --tail 5 swap-server 2>&1 | tr '\r' '\n' | grep -q SWAP_SERVER_READY && break; sleep 3; done; }
    docker exec swap-server chmod 777 /o/swap_jobs 2>/dev/null || true
    rm -f output/swap_jobs/${NAME}.done output/swap_jobs/${NAME}.err
    # swap WITHOUT GFPGAN (soft, fast) + save detected faces — GFPGAN is moved to a final restore pass
    # after MuseTalk so it also sharpens the soft 256px muse mouth (same total compute, better quality).
    printf '{"src":"/o/%s","video":"/o/%s_silent.mp4","out":"/o/%s_swap.mp4","keepeyes":true,"enhance":false,"save_faces":"/o/%s.faces.json"}\n' "$FACE" "$NAME" "$NAME" "$NAME" > output/swap_jobs/${NAME}.json
    while [ ! -f output/swap_jobs/${NAME}.done ] && [ ! -f output/swap_jobs/${NAME}.err ]; do sleep 3; done
    [ -f output/swap_jobs/${NAME}.err ] && { echo "SWAP ERR: $(cat output/swap_jobs/${NAME}.err)"; exit 1; }
    log "swap done: $(cat output/swap_jobs/${NAME}.done)"
    MUSE_IN=${NAME}_swap
  fi
  # 16k audio for muse
  ffmpeg -y -i "$AUDIO" -ar 16000 -ac 1 output/${NAME}_16k.wav 2>/dev/null
  log "MuseTalk (last)..."
  rm -f output/muse_jobs/${NAME}.done output/muse_jobs/${NAME}.err
  printf '{"video":"/io/%s.mp4","audio":"/io/%s_16k.wav","out":"/io/%s_talk.mp4"}\n' "$MUSE_IN" "$NAME" "$NAME" > output/muse_jobs/${NAME}.json
  while [ ! -f output/muse_jobs/${NAME}.done ] && [ ! -f output/muse_jobs/${NAME}.err ]; do sleep 5; done
  [ -f output/muse_jobs/${NAME}.err ] && { echo "MUSE ERR: $(cat output/muse_jobs/${NAME}.err)"; exit 1; }
  log "muse done: $(cat output/muse_jobs/${NAME}.done)"
  # FINAL FACE FINISH: --premium => FlashVSR-face 2x diffusion upscale on rtx0 (1440p, replaces GFPGAN).
  # else => GFPGAN-keepeyes restore reusing the swap's saved faces (720p, fast).
  if [ "$PREMIUM" = 1 ]; then
    log "FlashVSR-face 2x premium upscale on rtx0 (~4min)..."
    WAN=/mnt/datadisk/containers/wan2gp
    scp -q output/${NAME}_talk.mp4 josh@rtx0:$WAN/myinput/${NAME}.mp4
    $RTX "bash $WAN/flashvsr_face.sh ${NAME}" 2>&1 | grep -aE 'FLASHVSR_OK|Error|Traceback' | tail -3
    rm -f output/${NAME}_talk.mp4   # muse wrote it as root; scp can't overwrite, only the josh-owned dir lets us unlink
    scp -q josh@rtx0:$WAN/outputs/flashvsr/wan2gp_face_fast_${NAME}_2x.mp4 output/${NAME}_talk.mp4
    OTSW=860   # OTS panels at 2x for the 1440p canvas
    log "flashvsr done -> $(ffprobe -v error -show_entries stream=width,height -of csv=p=0:s=x output/${NAME}_talk.mp4 2>/dev/null | head -1)"
  else
    # GFPGAN restore: reuse the swap's saved faces if present, else re-detect (baked mode has no swap)
    log "final GFPGAN restore (sharpen mouth)..."
    docker ps --format '{{.Names}}' | grep -q '^swap-server$' || { bash "$BOT/swap/run_swap.sh"; \
      for i in $(seq 1 20); do docker logs --tail 5 swap-server 2>&1 | tr '\r' '\n' | grep -q SWAP_SERVER_READY && break; sleep 3; done; \
      docker exec swap-server chmod 777 /o/swap_jobs 2>/dev/null || true; }
    FACESARG=""; [ -f output/${NAME}.faces.json ] && FACESARG=",\"faces\":\"/o/${NAME}.faces.json\""
    rm -f output/swap_jobs/${NAME}r.done output/swap_jobs/${NAME}r.err
    printf '{"mode":"restore","video":"/o/%s_talk.mp4","out":"/o/%s_sharp.mp4","keepeyes":true%s}\n' "$NAME" "$NAME" "$FACESARG" > output/swap_jobs/${NAME}r.json
    while [ ! -f output/swap_jobs/${NAME}r.done ] && [ ! -f output/swap_jobs/${NAME}r.err ]; do sleep 3; done
    [ -f output/swap_jobs/${NAME}r.err ] && { echo "RESTORE ERR: $(cat output/swap_jobs/${NAME}r.err)"; exit 1; }
    # restore output is video-only mp4v; re-mux the audio from the muse output, back to ${NAME}_talk.mp4
    ffmpeg -y -i output/${NAME}_sharp.mp4 -i output/${NAME}_talk.mp4 -map 0:v -map 1:a \
      -c:v libx264 -pix_fmt yuv420p -crf 18 -c:a copy -shortest output/${NAME}_sharpav.mp4 2>/dev/null
    mv output/${NAME}_sharpav.mp4 output/${NAME}_talk.mp4
    rm -f output/${NAME}_sharp.mp4
    log "restore done: $(cat output/swap_jobs/${NAME}r.done)"
  fi
fi

# 4) OTS GRAPHICS (optional) ------------------------------------------------------
if [ -n "$OTS" ]; then
  log "building OTS panels..."
  PANELARGS=""; i=0
  IFS=';' read -ra SEGS <<< "$OTS"
  for seg in "${SEGS[@]}"; do
    seg=$(echo "$seg" | sed 's/^ *//;s/ *$//'); [ -z "$seg" ] && continue
    IMG=$(echo "$seg" | cut -d'|' -f1); LAB=$(echo "$seg" | cut -d'|' -f2); END=$(echo "$seg" | cut -d'|' -f3)
    P=output/${NAME}_ots_${i}.png
    docker run --rm -v "$BOT":/b -w /b mp-extract:1.0 python3 newscast/ots_panel.py "/b/$IMG" "$LAB" "/b/$P" --w "${OTSW:-430}" | grep -aE 'OTS_OK|Error'
    PANELARGS="$PANELARGS $P $END"; i=$((i+1))
  done
  log "compositing OTS..."
  bash newscast/ots_compose.sh output/${NAME}_talk.mp4 "$OUT" $PANELARGS | grep -aE 'OTS_COMPOSE_OK|Error'
else
  cp output/${NAME}_talk.mp4 "$OUT"
fi
log "DONE -> $OUT"
