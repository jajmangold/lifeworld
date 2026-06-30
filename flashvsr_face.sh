#!/bin/bash
# flashvsr_face.sh <stem>  — 2x face-only FlashVSR upscale. Input: myinput/<stem>.mp4
# Output: outputs/flashvsr/wan2gp_face_fast_<stem>_2x.mp4 . See bot/FLASHVSR_FACE_UPSCALE.md.
set -e
STEM=$1
IMG=deepbeepmeep/wan2gp:3060
cd /mnt/datadisk/containers/wan2gp
SRC=/workspace/myinput/${STEM}.mp4
DC="docker run --rm -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --entrypoint bash -v $PWD:/workspace $IMG -c"
$DC "cd /workspace && python3 detect_crop.py $SRC $STEM"
docker run --rm --gpus '"device=1"' -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --entrypoint bash -v "$PWD":/workspace $IMG -c "cd /workspace && python3 -u head_up.py $STEM"
$DC "cd /workspace && python3 composite.py $STEM $SRC"
echo "FLASHVSR_OK -> outputs/flashvsr/wan2gp_face_fast_${STEM}_2x.mp4"
