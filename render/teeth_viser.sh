#!/usr/bin/env bash
# Live 3D teeth tuner (viser): Theo's head in the browser, teeth update as you drag.
#   render/teeth_viser.sh         # start (detached) -> http://<host>:8772
#   render/teeth_viser.sh stop
set -euo pipefail
NAME=teeth-viser
if [ "${1:-}" = "stop" ]; then docker rm -f $NAME 2>/dev/null && echo stopped; exit 0; fi
docker rm -f $NAME 2>/dev/null || true
docker run -d --name $NAME -e PYTHONPATH=/work -p 8772:8772 \
  -v /srv/nvme-data/containers/projects/sampl:/work \
  -v /srv/nvme-data/containers/projects/bot:/lw \
  lifeworld-viser python3 /lw/render/teeth_viser.py >/dev/null
echo "starting (baking Theo ~15s)…"
for i in $(seq 1 40); do
  docker logs $NAME 2>&1 | grep -q "viser LIVE" && { echo "READY -> http://$(hostname -I | awk '{print $1}'):8772"; exit 0; }
  docker logs $NAME 2>&1 | grep -qiE "Traceback|Error" && { docker logs $NAME 2>&1 | tail -20; exit 1; }
  sleep 1
done
echo "(still starting; docker logs $NAME)"
