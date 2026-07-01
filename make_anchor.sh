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
BOT=/srv/nvme-data/containers/live/studio
SAMPL=/mnt/datadisk/containers/sampl
RTX="ssh -o BatchMode=yes josh@rtx0"
cd "$BOT"

MOOD=serious; BROW=1.0; NOD=1.0; BROWBASE=""; SEED=7; FACE=anchorM.png; FAST=0; PREMIUM=0; BAKED=0; BAKEDTEX=anchorM_head_baked.png; HEADTEX=anchorM_klein_head.png; SCREEN=""; OTS=""; AUDIO=""; OUT=""; RESTORE=0.6; FORMAT=anchor_wall; BG=""; PANO_ENV=""; GLBFILE=""; KEEPEYES=true; CHARFILE=""; ULTRA=0
while [ $# -gt 0 ]; do case "$1" in
  --audio) AUDIO=$2; shift 2;; --out) OUT=$2; shift 2;; --mood) MOOD=$2; shift 2;;
  --brow) BROW=$2; shift 2;; --nod) NOD=$2; shift 2;; --browbase) BROWBASE=$2; shift 2;;
  --seed) SEED=$2; shift 2;; --face) FACE=$2; shift 2;; --fast) FAST=1; shift;;
  --premium) PREMIUM=1; shift;;
  --ultra) PREMIUM=1; ULTRA=1; shift;;   # FlashVSR-face 4x ULTRA (vs premium 2x)
  --baked) BAKED=1; shift;;
  --bakedtex) BAKEDTEX=$2; shift 2;;
  --headtex) HEADTEX=$2; shift 2;;   # improved skin/hair base texture (swap still runs). DEFAULT anchorM_klein_head.png; pass --headtex "" to disable
  --screen) SCREEN=$2; shift 2;;     # broadcast mode: 3D video-wall image (reframe MCU + anchor left + screen right)
  --restore) RESTORE=$2; shift 2;;   # GFPGAN restore strength 0..1 (default 0.6; 1.0=waxy, 0=muse-soft)
  --format) FORMAT=$2; shift 2;;     # anchor_wall (default, needs --screen) | fullscreen_anchor | ots
  --bg) BG=$2; shift 2;;             # composite background plate (flat photo) — replaces the studio bg.png (fallback; --pano is better)
  --pano) PANO_ENV=$2; shift 2;;     # field/outdoor equirectangular env pano: LIGHTS the character AND renders the bg (proper on-location look)
  --glb) GLBFILE=$2; shift 2;;       # swap the character mesh (e.g. a viverse reporter VRM/GLB) instead of the anchor
  --character) CHARFILE=$2; shift 2;; # BlenderKit Rigify .blend -> render via render_character.py (viverse-free); swap still runs
  --keepeyes) KEEPEYES=$2; shift 2;; # keep the RENDER's eyes during face-swap (true, default). false => swap brings the source's eyes (fixes viverse dark-eye meshes)
  --ots) OTS=$2; shift 2;; *) echo "unknown arg: $1"; exit 1;; esac; done
[ -z "$AUDIO" ] && { echo "need --audio"; exit 1; }
[ -z "$OUT" ] && { echo "need --out"; exit 1; }
NAME=$(basename "$OUT" .mp4)
AB=$(basename "$AUDIO")
mkdir -p output
# CLIP FACTORY hooks: per-segment render frame dir (so concurrent renders don't clobber) + per-resource
# flocks. With FLOCK_DIR set (by newscast/factory.py) the kernel serializes each scarce resource
# (render=rtx0 GPU0, premium=rtx0 GPU1, swap+muse=V100 servers) while different segments overlap across
# resources. Unset FLOCK_DIR -> LK/UNLK are no-ops -> single-clip behaviour unchanged.
ADIR="anchor_anim_${NAME}"                         # per-segment frame dir on rtx0 (/work/output/$ADIR)
LK(){ [ -n "$FLOCK_DIR" ] && { eval "exec $2>$FLOCK_DIR/$1.lock"; flock "$2"; }; return 0; }   # LK <res> <fd>
UNLK(){ [ -n "$FLOCK_DIR" ] && flock -u "$2"; return 0; }
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
CB=""
if [ -n "$CHARFILE" ]; then          # BlenderKit character: stage the .blend + the generic renderer
  CB="char_${NAME}.blend"; scp -q "$CHARFILE" josh@rtx0:$SAMPL/"$CB"; scp -q render/render_character.py josh@rtx0:$SAMPL/
  HEADTEX=""                         # not applicable to a BlenderKit character
  log "character mode: $(basename "$CHARFILE") -> render_character.py"
fi
BENV=""
if [ "$BAKED" = 1 ]; then
  $RTX "test -f $SAMPL/$BAKEDTEX" || scp -q viverse_avatar/$BAKEDTEX josh@rtx0:$SAMPL/
  BENV="BAKED_HEAD_TEX=/work/$BAKEDTEX "
  log "baked-texture mode: $BAKEDTEX (identity baked into render; swap will be skipped)"
elif [ -n "$HEADTEX" ]; then
  # improved skin/hair base texture (Klein-edited, front-projected); swap STILL runs on top.
  scp -q viverse_avatar/$HEADTEX josh@rtx0:$SAMPL/
  BENV="BAKED_HEAD_TEX=/work/$HEADTEX "
  log "headtex mode: $HEADTEX (improved skin/hair base; per-frame swap still runs)"
fi
if [ -n "$SCREEN" ]; then
  SB=$(basename "$SCREEN"); scp -q "$SCREEN" josh@rtx0:$SAMPL/"$SB"
  BENV="${BENV}NEWS_SCREEN=/work/$SB "
  log "broadcast mode: 3D video wall $SB (reframe MCU + anchor left)"
fi
if [ -n "$GLBFILE" ]; then
  # custom character mesh (viverse VRM/GLB). Blender's glTF importer needs a .glb/.gltf extension, so a
  # .vrm (which IS glb) is staged as .glb. Same viverse rig as the anchor (Avatar_* bones), so idle
  # motion + blink aliases just work; disable the anchor head-tex (--headtex "") so her own skin shows.
  GB=$(basename "$GLBFILE"); GB="${GB%.vrm}.glb"; scp -q "$GLBFILE" josh@rtx0:$SAMPL/"$GB"
  BENV="${BENV}NEWS_GLB=/work/$GB "
  log "custom mesh: $GB (viverse character)"
fi
if [ -n "$PANO_ENV" ]; then
  # field/outdoor env: the pano LIGHTS the character and is rendered as bg.png through the camera, so
  # lighting + background are one coherent world (correct on-location look). Supersedes --bg photo hack.
  PB=$(basename "$PANO_ENV"); scp -q "$PANO_ENV" josh@rtx0:$SAMPL/"$PB"
  BENV="${BENV}NEWS_PANO=/work/$PB "
  BG=""   # pano renders the background itself; don't also overwrite bg.png with a flat photo
  log "field env: $PB (equirectangular pano lights + backs the standup)"
fi
# segment FORMAT (anchor variants): reframe without a wall. fullscreen_anchor=centered, ots=offset (box in post).
case "$FORMAT" in
  fullscreen_anchor) BENV="${BENV}NEWS_MCU=1 NEWS_SHIFTX=0 "; log "format: fullscreen_anchor (centered MCU)";;
  ots)               BENV="${BENV}NEWS_MCU=1 NEWS_SHIFTX=0.33 "; log "format: ots (anchor left, OTS box in post)";;
esac
# forward framing/env overrides set in the caller's environment (per-character tuning, e.g. a viverse
# avatar with different proportions needs its own headroom/lens): NEWS_AIM/DIST/LENS/FSTOP/WALLROT/ENVSTR.
for _v in NEWS_AIM NEWS_DIST NEWS_LENS NEWS_FSTOP NEWS_WALLROT NEWS_ENVSTR NEWS_SHIFTX; do
  eval "_val=\${$_v}"; [ -n "$_val" ] && BENV="${BENV}$_v=$_val "
done
LK render 201   # serialize rtx0 GPU0 across segments (released right after Blender exits)
if [ -n "$CHARFILE" ]; then          # BlenderKit Rigify character (render_character.py)
  $RTX "docker exec sampl bash -lc 'cd /work && mkdir -p output/$ADIR && rm -f output/$ADIR/f*.png && ${BENV}NEWS_CAMSIDE=${NEWS_CAMSIDE:--1} ANCHOR_OUT=/work/output/$ADIR/ CUDA_VISIBLE_DEVICES=0 /opt/blender/blender -b --python render_character.py -- --blend /work/$CB --arkit /work/${NAME}.perf.json > /work/output/${NAME}_render.log 2>&1; echo DONE_RC=\$? >> /work/output/${NAME}_render.log'"
else                                 # viverse avatar (render_anchor_anim.py)
  $RTX "docker exec sampl bash -lc 'cd /work && mkdir -p output/$ADIR && rm -f output/$ADIR/f*.png && ${BENV}ANCHOR_OUT=/work/output/$ADIR/ CUDA_VISIBLE_DEVICES=0 /opt/blender/blender --background --python render_anchor_anim.py -- --arkit /work/${NAME}.perf.json > /work/output/${NAME}_render.log 2>&1; echo DONE_RC=\$? >> /work/output/${NAME}_render.log'"
fi
UNLK render 201
# optional background plate (e.g. field backdrop for a reporter standup): overwrite the studio bg.png
# with the given image, scaled to COVER the frame. Character stays lit by the studio HDRI (fine for a
# brief standup); only the composited backdrop changes.
if [ -n "$BG" ]; then
  BGB=$(basename "$BG"); scp -q "$BG" josh@rtx0:$SAMPL/"$BGB"
  $RTX "docker exec sampl python3 -c \"import cv2,numpy as np,sys; d='/work/output/$ADIR'; b=cv2.imread(d+'/bg.png'); H,W=b.shape[:2]; s=cv2.imread('/work/$BGB'); sh,sw=s.shape[:2]; f=max(W/sw,H/sh); r=cv2.resize(s,(int(sw*f+0.5),int(sh*f+0.5))); y=(r.shape[0]-H)//2; x=(r.shape[1]-W)//2; cv2.imwrite(d+'/bg.png', r[y:y+H,x:x+W]); print('BG_SET',W,H)\"" 2>&1 | grep -aE 'BG_SET|Error' | tail -1
  log "bg plate: $BGB (field backdrop)"
fi
# premultiplied-over composite (clean silhouette edges) then encode (CPU — off the GPU lock)
$RTX "docker exec sampl bash -lc 'cd /work/output/$ADIR && rm -f c[0-9]*.png && python3 /work/composite_premult.py /work/output/$ADIR'" 2>&1 | grep -aE 'COMPOSITE_OK|Error' | tail -1
$RTX "docker exec sampl bash -lc 'cd /work/output/$ADIR && ffmpeg -y -framerate 25 -i c%04d.png -c:v libx264 -pix_fmt yuv420p -crf 18 /work/output/${NAME}_silent.mp4 2>&1 | tail -1'" >/dev/null
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
    LK swap 202   # serialize the V100 swap-server across segments
    # swap WITHOUT GFPGAN (soft, fast) + save detected faces — GFPGAN is moved to a final restore pass
    # after MuseTalk so it also sharpens the soft 256px muse mouth (same total compute, better quality).
    printf '{"src":"/o/%s","video":"/o/%s_silent.mp4","out":"/o/%s_swap.mp4","keepeyes":%s,"enhance":false,"save_faces":"/o/%s.faces.json"}\n' "$FACE" "$NAME" "$NAME" "$KEEPEYES" "$NAME" > output/swap_jobs/${NAME}.json
    while [ ! -f output/swap_jobs/${NAME}.done ] && [ ! -f output/swap_jobs/${NAME}.err ]; do sleep 3; done
    UNLK swap 202
    [ -f output/swap_jobs/${NAME}.err ] && { echo "SWAP ERR: $(cat output/swap_jobs/${NAME}.err)"; exit 1; }
    log "swap done: $(cat output/swap_jobs/${NAME}.done)"
    MUSE_IN=${NAME}_swap
  fi
  # 16k audio for muse
  # noise-gate the MuseTalk audio: hard-silence the pauses so the mouth doesn't jitter on
  # low-level breath/TTS-floor (loudnorm raises that floor). Gated pauses -> neutral closed mouth.
  ffmpeg -y -i "$AUDIO" -af "highpass=f=70,agate=threshold=0.045:ratio=12:attack=6:release=180:range=0.0,agate=threshold=0.02:ratio=6:attack=10:release=250:range=0.0" -ar 16000 -ac 1 output/${NAME}_16k.wav 2>/dev/null
  log "MuseTalk (last)..."
  rm -f output/muse_jobs/${NAME}.done output/muse_jobs/${NAME}.err
  LK muse 203   # serialize the V100 muse-server across segments
  printf '{"video":"/io/%s.mp4","audio":"/io/%s_16k.wav","out":"/io/%s_talk.mp4"}\n' "$MUSE_IN" "$NAME" "$NAME" > output/muse_jobs/${NAME}.json
  while [ ! -f output/muse_jobs/${NAME}.done ] && [ ! -f output/muse_jobs/${NAME}.err ]; do sleep 5; done
  UNLK muse 203
  [ -f output/muse_jobs/${NAME}.err ] && { echo "MUSE ERR: $(cat output/muse_jobs/${NAME}.err)"; exit 1; }
  log "muse done: $(cat output/muse_jobs/${NAME}.done)"
  # PAUSE-LIP SETTLE: MuseTalk animates the lips even on true silence -> jitter in quiet moments. Read
  # per-frame DENSE lip landmarks (insightface 2d106 in swap-server), smooth the lip trajectory through
  # each silent run, and warp the mouth into the settled shape (kept a touch open). Natural settle driven
  # by the real shapes, no teeth smear. Works in swap AND baked mode (self-detects lips).
  log "pause-lip settle (dense-lip warp)..."
  cp newscast/lip_landmarks.py output/lip_landmarks.py
  docker ps --format '{{.Names}}' | grep -q '^swap-server$' || { bash "$BOT/swap/run_swap.sh"; \
    for i in $(seq 1 20); do docker logs --tail 5 swap-server 2>&1 | tr '\r' '\n' | grep -q SWAP_SERVER_READY && break; sleep 3; done; }
  LK swap 202
  docker exec swap-server python3 /o/lip_landmarks.py /o/${NAME}_talk.mp4 /o/${NAME}_lips.npy 2>&1 | grep -aE 'LIP_OK|Error' | tail -1
  UNLK swap 202
  if [ -f output/${NAME}_lips.npy ]; then
    docker run --rm -v "$BOT":/io -w /io mp-extract:1.0 python3 newscast/mouth_settle.py \
      /io/output/${NAME}_talk.mp4 /io/output/${NAME}_lips.npy /io/output/${NAME}_16k.wav /io/output/${NAME}_settle.mp4 2>&1 | grep -aE 'MOUTH_SETTLE_OK|Error'
    if [ -f output/${NAME}_settle.mp4 ]; then
      ffmpeg -y -i output/${NAME}_settle.mp4 -i output/${NAME}_talk.mp4 -map 0:v -map 1:a? \
        -c:v libx264 -pix_fmt yuv420p -crf 17 -c:a aac -shortest output/${NAME}_setav.mp4 2>/dev/null
      rm -f output/${NAME}_talk.mp4 output/${NAME}_settle.mp4; mv output/${NAME}_setav.mp4 output/${NAME}_talk.mp4
    fi
  fi
  # FINAL FACE FINISH: --premium => FlashVSR-face 2x diffusion upscale on rtx0 (1440p, replaces GFPGAN).
  # else => GFPGAN-keepeyes restore reusing the swap's saved faces (720p, fast).
  if [ "$PREMIUM" = 1 ]; then
    log "FlashVSR-face 2x premium upscale on rtx0 (~4min)..."
    WAN=/mnt/datadisk/containers/wan2gp
    scp -q output/${NAME}_talk.mp4 josh@rtx0:$WAN/myinput/${NAME}.mp4
    # head matte from the render alpha frames -> confine FlashVSR sharpening inside the silhouette
    # (no diffusion edge-ringing halo). composite.py auto-uses myinput/matte_<name>/ if present.
    scp -q render/export_alpha_matte.py josh@rtx0:$SAMPL/
    scp -q render/flashvsr/composite.py render/flashvsr/detect_crop.py render/flashvsr/flashvsr_face.sh josh@rtx0:$WAN/
    # matte from THIS segment's render frame dir (clip factory: $ADIR is per-segment)
    $RTX "docker exec sampl bash -lc 'cd /work && python3 export_alpha_matte.py output/$ADIR output/matte_${NAME}'" 2>&1 | grep -aE 'MATTE_OK|Error'
    $RTX "rm -rf $WAN/myinput/matte_${NAME}; cp -r $SAMPL/output/matte_${NAME} $WAN/myinput/"
    # wait for the resident FlashVSR queue coordinator (:8800) to have >=1 ready worker.
    $RTX "for i in \$(seq 1 40); do curl -sf -m3 http://localhost:8800/health 2>/dev/null | grep -q true && break; sleep 3; done"
    # NO premium flock: the queue coordinator load-balances concurrent submits across its workers, so
    # multiple factory segments upscale on DIFFERENT GPUs in parallel (premium is no longer the 1-GPU bottleneck).
    $RTX "ULTRA=${ULTRA:-0} bash $WAN/flashvsr_face.sh ${NAME}" 2>&1 | grep -aE 'FLASHVSR_OK|Error|Traceback' | tail -3
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
    printf '{"mode":"restore","video":"/o/%s_talk.mp4","out":"/o/%s_sharp.mp4","keepeyes":true,"strength":%s%s}\n' "$NAME" "$NAME" "$RESTORE" "$FACESARG" > output/swap_jobs/${NAME}r.json
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
