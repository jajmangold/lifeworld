#!/usr/bin/env bash
# Manually tune the teeth/mouth: edit render/mouth_params.json, then run this (~15s).
# Shows the open mouth (front + 3/4 view) cropped to the bite -> output/teeth.png.
# The same mouth_params.json drives the real cinematic (render/scene_cine.py), so once it
# looks right here, re-render with: python3 render/scene_cine.py --reuse
set -euo pipefail
docker run --rm --gpus device=7 -e NVIDIA_DRIVER_CAPABILITIES=all -e PYTHONPATH=/work \
  -v "${LIFEWORLD_SAMPLES_DIR:-./samples}":/work \
  -v "${LIFEWORLD_BOT_DIR:-./bot}":/lw \
  -v "${A2F_DIR:-./a2f}":/a2f \
  lifeworld-preview python3 /lw/render/teeth_preview.py --out /a2f/teeth.png "$@"
cp "${A2F_DIR:-./a2f}/teeth.png" "${LIFEWORLD_BOT_DIR:-./bot}/output/teeth.png"
echo "-> output/teeth.png  (edit render/mouth_params.json and re-run to adjust)"
