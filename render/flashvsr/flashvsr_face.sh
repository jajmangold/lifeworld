#!/bin/bash
# flashvsr_face.sh <stem>  — face-only FlashVSR upscale. Input: myinput/<stem>.mp4
# Output: outputs/flashvsr/wan2gp_face_fast_<stem>_2x.mp4 . See bot/FLASHVSR_FACE_UPSCALE.md.
# Upscales the head crop via the resident FlashVSR async QUEUE coordinator (:8800), which load-balances
# across N resident workers (one model per GPU) — no per-call reload, no 2nd model copy (that was the
# OOM), and concurrent factory segments run on DIFFERENT workers in parallel. /upscale is the sync
# back-compat path (submit+wait). ULTRA=1 -> 4x + full-block.
set -e
STEM=$1
IMG=deepbeepmeep/wan2gp:3060
cd /mnt/datadisk/containers/wan2gp
SRC=/workspace/myinput/${STEM}.mp4
ULTRA=${ULTRA:-0}
SCALE=2; [ "$ULTRA" = 1 ] && SCALE=4
FVSR=${FVSR_URL:-http://localhost:8800}
DC="docker run --rm -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --entrypoint bash -v $PWD:/workspace $IMG -c"
$DC "cd /workspace && MAX_BOX=${MAX_BOX:-464} python3 detect_crop.py $SRC $STEM"
# upscale the head crop via the resident server (no head_up.py docker-run, no model reload)
curl -sf -X POST --data-binary @myinput/head_crop_${STEM}.mp4 \
     "$FVSR/upscale?mode=video&scale=${SCALE}&ext=mp4&ultra=${ULTRA}" \
     -o myinput/head_up_${STEM}.mp4
$DC "cd /workspace && HEAD_SCALE=$SCALE python3 composite.py $STEM $SRC"
echo "FLASHVSR_OK -> outputs/flashvsr/wan2gp_face_fast_${STEM}_2x.mp4 (ultra=$ULTRA, resident server)"
