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
#   --ots "img|LABEL|end; img2|LABEL2|end2"   over-the-shoulder panels (seconds = segment end times)
set -e
BOT=/srv/nvme-data/containers/projects/bot
SAMPL=/mnt/datadisk/containers/sampl
RTX="ssh -o BatchMode=yes josh@rtx0"
cd "$BOT"

MOOD=serious; BROW=1.0; NOD=1.0; BROWBASE=""; SEED=7; FACE=anchorM.png; FAST=0; OTS=""; AUDIO=""; OUT=""
while [ $# -gt 0 ]; do case "$1" in
  --audio) AUDIO=$2; shift 2;; --out) OUT=$2; shift 2;; --mood) MOOD=$2; shift 2;;
  --brow) BROW=$2; shift 2;; --nod) NOD=$2; shift 2;; --browbase) BROWBASE=$2; shift 2;;
  --seed) SEED=$2; shift 2;; --face) FACE=$2; shift 2;; --fast) FAST=1; shift;;
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
$RTX "docker exec sampl bash -lc 'cd /work && rm -f output/anchor_anim/f*.png && CUDA_VISIBLE_DEVICES=0 /opt/blender/blender --background --python render_anchor_anim.py -- --arkit /work/${NAME}.perf.json > /work/output/${NAME}_render.log 2>&1; echo DONE_RC=\$? >> /work/output/${NAME}_render.log'"
$RTX "docker exec sampl bash -lc 'cd /work/output/anchor_anim && ffmpeg -y -framerate 25 -i f%04d.png -c:v libx264 -pix_fmt yuv420p -crf 18 /work/output/${NAME}_silent.mp4 2>&1 | tail -1'" >/dev/null
scp -q josh@rtx0:$SAMPL/output/${NAME}_silent.mp4 output/${NAME}_silent.mp4
log "render done -> output/${NAME}_silent.mp4"

# 3) MOUTH/IDENTITY ---------------------------------------------------------------
if [ "$FAST" = 1 ]; then
  log "fast mode: mux audio (viseme mouth, no swap/muse)"
  ffmpeg -y -i output/${NAME}_silent.mp4 -i "$AUDIO" -c:v libx264 -pix_fmt yuv420p -crf 18 -c:a aac -b:a 192k -shortest output/${NAME}_talk.mp4 2>/dev/null
else
  log "swap (keepeyes) -> $FACE"
  docker run --rm --gpus '"device=0"' -v "$BOT/swap":/s -v "$BOT/output":/o -w /s inswap:local \
    python local_swap_keepeyes.py /o/"$FACE" /o/${NAME}_silent.mp4 /o/${NAME}_swap.mp4 2>&1 | grep -aE 'SWAP_OK|Error'
  # 16k audio for muse
  ffmpeg -y -i "$AUDIO" -ar 16000 -ac 1 output/${NAME}_16k.wav 2>/dev/null
  log "MuseTalk (last)..."
  rm -f output/muse_jobs/${NAME}.done output/muse_jobs/${NAME}.err
  printf '{"video":"/io/%s_swap.mp4","audio":"/io/%s_16k.wav","out":"/io/%s_talk.mp4"}\n' "$NAME" "$NAME" "$NAME" > output/muse_jobs/${NAME}.json
  while [ ! -f output/muse_jobs/${NAME}.done ] && [ ! -f output/muse_jobs/${NAME}.err ]; do sleep 5; done
  [ -f output/muse_jobs/${NAME}.err ] && { echo "MUSE ERR: $(cat output/muse_jobs/${NAME}.err)"; exit 1; }
  log "muse done: $(cat output/muse_jobs/${NAME}.done)"
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
    docker run --rm -v "$BOT":/b -w /b mp-extract:1.0 python3 newscast/ots_panel.py "/b/$IMG" "$LAB" "/b/$P" | grep -aE 'OTS_OK|Error'
    PANELARGS="$PANELARGS $P $END"; i=$((i+1))
  done
  log "compositing OTS..."
  bash newscast/ots_compose.sh output/${NAME}_talk.mp4 "$OUT" $PANELARGS | grep -aE 'OTS_COMPOSE_OK|Error'
else
  cp output/${NAME}_talk.mp4 "$OUT"
fi
log "DONE -> $OUT"
