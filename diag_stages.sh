#!/bin/bash
# Per-stage diagnostic: render -> swap -> muse(raw, kept) -> GFPGAN restore.
# Preserves every intermediate so we can attribute eyes/skin/mouth defects to a stage.
set -e
BOT=/srv/nvme-data/containers/live/studio
SAMPL=/mnt/datadisk/containers/sampl
RTX="ssh -o BatchMode=yes josh@rtx0"
cd "$BOT"
NAME=diag
AUDIO=output/diag.wav
log(){ echo "[diag $(date +%H:%M:%S)] $*"; }

# 1) performance
log "proc_perf"
docker run --rm -v "$BOT":/io -w /io mp-extract:1.0 python3 proc_perf.py \
  --audio /io/$AUDIO --mood serious --brow 1.0 --nod 1.0 --seed 7 \
  --out /io/output/${NAME}.perf.json 2>&1 | grep -aE 'PERF_OK|Error' || true

# 2) render on rtx0 (NO baked tex -> original avatar face; swap adds identity)
log "render on rtx0"
$RTX "test -f $SAMPL/avatar.glb" || scp -q viverse_avatar/avatar.glb josh@rtx0:$SAMPL/
$RTX "test -f $SAMPL/newsroom_pano.png" || scp -q output/newsroom_pano.png josh@rtx0:$SAMPL/
scp -q output/${NAME}.perf.json render_anchor_anim.py josh@rtx0:$SAMPL/
$RTX "docker exec sampl bash -lc 'cd /work && rm -f output/anchor_anim/f*.png && CUDA_VISIBLE_DEVICES=0 /opt/blender/blender --background --python render_anchor_anim.py -- --arkit /work/${NAME}.perf.json > /work/output/${NAME}_render.log 2>&1; echo RC=\$?'"
$RTX "docker exec sampl bash -lc 'cd /work/output/anchor_anim && rm -f c[0-9]*.png && python3 /work/composite_premult.py /work/output/anchor_anim'" 2>&1 | grep -aE 'COMPOSITE_OK|Error' | tail -1
$RTX "docker exec sampl bash -lc 'cd /work/output/anchor_anim && ffmpeg -y -framerate 25 -i c%04d.png -c:v libx264 -pix_fmt yuv420p -crf 18 /work/output/${NAME}_silent.mp4 2>&1 | tail -1'" >/dev/null
scp -q josh@rtx0:$SAMPL/output/${NAME}_silent.mp4 output/${NAME}_silent.mp4
log "render done -> output/${NAME}_silent.mp4"

# 3) swap (keepeyes, no GFPGAN) -> _swap, save faces
log "swap"
docker exec swap-server chmod 777 /o/swap_jobs 2>/dev/null || true
rm -f output/swap_jobs/${NAME}.done output/swap_jobs/${NAME}.err
printf '{"src":"/o/anchorM.png","video":"/o/%s_silent.mp4","out":"/o/%s_swap.mp4","keepeyes":true,"enhance":false,"save_faces":"/o/%s.faces.json"}\n' "$NAME" "$NAME" "$NAME" > output/swap_jobs/${NAME}.json
while [ ! -f output/swap_jobs/${NAME}.done ] && [ ! -f output/swap_jobs/${NAME}.err ]; do sleep 2; done
[ -f output/swap_jobs/${NAME}.err ] && { echo "SWAP ERR: $(cat output/swap_jobs/${NAME}.err)"; exit 1; }
log "swap done"

# 4) MuseTalk -> _muse_raw (KEEP, no restore)
ffmpeg -y -i $AUDIO -ar 16000 -ac 1 output/${NAME}_16k.wav 2>/dev/null
log "muse"
rm -f output/muse_jobs/${NAME}.done output/muse_jobs/${NAME}.err
printf '{"video":"/io/%s_swap.mp4","audio":"/io/%s_16k.wav","out":"/io/%s_muse_raw.mp4"}\n' "$NAME" "$NAME" "$NAME" > output/muse_jobs/${NAME}.json
while [ ! -f output/muse_jobs/${NAME}.done ] && [ ! -f output/muse_jobs/${NAME}.err ]; do sleep 3; done
[ -f output/muse_jobs/${NAME}.err ] && { echo "MUSE ERR: $(cat output/muse_jobs/${NAME}.err)"; exit 1; }
log "muse done -> output/${NAME}_muse_raw.mp4"

# 5) GFPGAN restore (keepeyes, reuse faces) -> _restored
log "restore"
FACESARG=""; [ -f output/${NAME}.faces.json ] && FACESARG=",\"faces\":\"/o/${NAME}.faces.json\""
rm -f output/swap_jobs/${NAME}r.done output/swap_jobs/${NAME}r.err
printf '{"mode":"restore","video":"/o/%s_muse_raw.mp4","out":"/o/%s_restored.mp4","keepeyes":true%s}\n' "$NAME" "$NAME" "$FACESARG" > output/swap_jobs/${NAME}r.json
while [ ! -f output/swap_jobs/${NAME}r.done ] && [ ! -f output/swap_jobs/${NAME}r.err ]; do sleep 2; done
[ -f output/swap_jobs/${NAME}r.err ] && { echo "RESTORE ERR: $(cat output/swap_jobs/${NAME}r.err)"; exit 1; }
log "restore done -> output/${NAME}_restored.mp4"
log "ALL STAGES DONE"
