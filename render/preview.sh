#!/usr/bin/env bash
# Fast lip-sync iteration: cached ARKit json -> head-only contact sheet in ~13s
# (one container: bake + render, no A2F re-run). Edit face_drive.py / jaw, re-run this.
#   render/preview.sh [arkit_json] [gender] [betas_csv]
set -euo pipefail
ARKIT="${1:-/a2f/test.arkit.json}"; GENDER="${2:-female}"; BETAS="${3:-}"
docker run --rm --gpus device=7 -e NVIDIA_DRIVER_CAPABILITIES=all -e PYTHONPATH=/work \
  -v "${LIFEWORLD_SAMPLES_DIR:-./samples}":/work \
  -v "${LIFEWORLD_BOT_DIR:-./bot}":/lw \
  -v "${A2F_DIR:-./a2f}":/a2f \
  lifeworld-preview python3 /lw/render/preview.py --arkit "$ARKIT" --gender "$GENDER" \
    ${BETAS:+--betas "$BETAS"} --out /a2f/preview.png
cp "${A2F_DIR:-./a2f}/preview.png" "${LIFEWORLD_BOT_DIR:-./bot}/output/preview.png"
echo "preview -> output/preview.png"
