#!/usr/bin/env bash
# Real-time teeth tuner web UI. Runs inside the GPU container (body kept warm), exposes :8771.
#   render/teeth_ui.sh            # start (detached), then open http://<host>:8771
#   render/teeth_ui.sh stop       # stop it
set -euo pipefail
NAME=teeth-ui
if [ "${1:-}" = "stop" ]; then docker rm -f $NAME 2>/dev/null && echo "stopped"; exit 0; fi
docker rm -f $NAME 2>/dev/null || true
docker run -d --name $NAME --gpus device=7 -e NVIDIA_DRIVER_CAPABILITIES=all -e PYTHONPATH=/work \
  -p 8771:8771 \
  -v "${LIFEWORLD_SAMPLES_DIR:-./samples}":/work \
  -v "${LIFEWORLD_BOT_DIR:-./bot}":/lw \
  -v "${A2F_DIR:-./a2f}":/a2f \
  lifeworld-preview python3 /lw/render/teeth_ui_live.py >/dev/null
echo "starting teeth-ui (warming the body ~10s)…"
for i in $(seq 1 30); do
  if docker logs $NAME 2>&1 | grep -q "LIVE ->"; then echo "READY -> http://$(hostname -I | awk '{print $1}'):8771"; exit 0; fi
  sleep 1
done
echo "(still warming; check: docker logs $NAME)"
