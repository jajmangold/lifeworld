#!/usr/bin/env bash
# Fast lip-sync iteration: cached ARKit json -> head-only contact sheet in ~13s
# (one container: bake + render, no A2F re-run). Edit face_drive.py / jaw, re-run this.
#   render/preview.sh [arkit_json] [gender] [betas_csv]
set -euo pipefail
ARKIT="${1:-/a2f/test.arkit.json}"; GENDER="${2:-female}"; BETAS="${3:-}"
docker run --rm --gpus device=7 -e NVIDIA_DRIVER_CAPABILITIES=all -e PYTHONPATH=/work \
  -v /srv/nvme-data/containers/projects/sampl:/work \
  -v /srv/nvme-data/containers/live/studio:/lw \
  -v /mnt/24tb/a2f:/a2f \
  lifeworld-preview python3 /lw/render/preview.py --arkit "$ARKIT" --gender "$GENDER" \
    ${BETAS:+--betas "$BETAS"} --out /a2f/preview.png
cp /mnt/24tb/a2f/preview.png /srv/nvme-data/containers/live/studio/output/preview.png
echo "preview -> output/preview.png"
