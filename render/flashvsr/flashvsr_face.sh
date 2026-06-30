#!/bin/bash
# flashvsr_face.sh <stem>  — face-only FlashVSR upscale. Input: myinput/<stem>.mp4
# Output: outputs/flashvsr/wan2gp_face_fast_<stem>_2x.mp4 . See bot/FLASHVSR_FACE_UPSCALE.md.
# ULTRA=1 -> head_up runs 4x + full-block + unload_dit (tiled) for a sharper MCU-sized face; the
# composite supersamples the 4x head back onto the 2x canvas. Slower, fits 12GB via tiling/unload.
set -e
STEM=$1
IMG=deepbeepmeep/wan2gp:3060
cd /mnt/datadisk/containers/wan2gp
SRC=/workspace/myinput/${STEM}.mp4
ULTRA=${ULTRA:-0}
HEAD_SCALE=2; [ "$ULTRA" = 1 ] && HEAD_SCALE=4
DC="docker run --rm -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --entrypoint bash -v $PWD:/workspace $IMG -c"
$DC "cd /workspace && MAX_BOX=${MAX_BOX:-464} python3 detect_crop.py $SRC $STEM"
docker run --rm --gpus '"device=1"' -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e ULTRA=$ULTRA \
  --entrypoint bash -v "$PWD":/workspace $IMG -c "cd /workspace && ULTRA=$ULTRA python3 -u head_up.py $STEM"
$DC "cd /workspace && HEAD_SCALE=$HEAD_SCALE python3 composite.py $STEM $SRC"
echo "FLASHVSR_OK -> outputs/flashvsr/wan2gp_face_fast_${STEM}_2x.mp4 (ultra=$ULTRA)"
