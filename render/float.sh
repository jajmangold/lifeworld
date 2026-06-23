#!/bin/bash
# Resident FLOAT talking-head server. Loads the model once; warm gens skip the cold start.
#   render/float.sh server                       # start (GPU $FLOAT_GPU, default 9), port 8210
#   render/float.sh gen <ref.png> <audio.wav> <out.mp4> [nfe]   # paths under bot/output (host)
#   render/float.sh health | stop | logs
set -e
NAME=float-server; PORT=8217; GPU=${FLOAT_GPU:-9}
FLOATDIR=/srv/nvme-data/containers/projects/sampl/tools/float
IO=/srv/nvme-data/containers/projects/bot/output          # shared io: host <-> /io in container

case "$1" in
server)
  docker rm -f "$NAME" 2>/dev/null || true
  docker run -d --name "$NAME" --gpus "\"device=$GPU\"" -e NVIDIA_DRIVER_CAPABILITIES=all \
    -e TORCH_HOME=/float/.torch -e HF_HOME=/float/.hf -e FLOAT_PORT=$PORT \
    -p $PORT:$PORT \
    -v "$FLOATDIR":/float -v "$IO":/io -w /float float:volta \
    python float_server.py
  echo "starting $NAME on :$PORT (GPU $GPU); 'float.sh health' to poll readiness" ;;
gen)
  # $2 ref, $3 audio, $4 out  — names are under bot/output (mapped to /io)
  REF=/io/$(basename "$2"); AUD=/io/$(basename "$3"); OUT=/io/$(basename "$4"); NFE=${5:-10}
  curl -s -X POST "http://127.0.0.1:$PORT/gen" -H 'Content-Type: application/json' \
    -d "{\"ref_path\":\"$REF\",\"audio_path\":\"$AUD\",\"out_path\":\"$OUT\",\"nfe\":$NFE}"
  echo ;;
health) curl -s "http://127.0.0.1:$PORT/health"; echo ;;
logs)   docker logs --tail 30 "$NAME" 2>&1 | tr '\r' '\n' | tail -20 ;;
stop)   docker rm -f "$NAME" ;;
*) echo "usage: float.sh {server|gen <ref> <aud> <out> [nfe]|health|logs|stop}" ;;
esac
