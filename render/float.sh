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
    -e FLOAT_FACE_THR=${FLOAT_FACE_THR:-0.2} \
    -e PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128 \
    -p $PORT:$PORT \
    -v "$FLOATDIR":/float -v "$IO":/io -w /float float:volta \
    python float_server.py
  echo "starting $NAME on :$PORT (GPU $GPU); 'float.sh health' to poll readiness" ;;
gen)
  # $2 ref, $3 audio, $4 out (names under bot/output = /io). Optional env:
  #   NFE GAIN BIAS EMO  e.g.  GAIN=1.7 BIAS="15:6,6:4" EMO=happy render/float.sh gen a.png b.wav c.mp4
  REF=/io/$(basename "$2"); AUD=/io/$(basename "$3"); OUT=/io/$(basename "$4")
  curl -s -X POST "http://127.0.0.1:$PORT/gen" -H 'Content-Type: application/json' \
    -d "{\"ref_path\":\"$REF\",\"audio_path\":\"$AUD\",\"out_path\":\"$OUT\",\"nfe\":${NFE:-10},\"gain\":${GAIN:-1.0},\"bias\":\"${BIAS:-}\"$([ -n "$EMO" ] && echo ",\"emo\":\"$EMO\"")}"
  echo ;;
health) curl -s "http://127.0.0.1:$PORT/health"; echo ;;
logs)   docker logs --tail 30 "$NAME" 2>&1 | tr '\r' '\n' | tail -20 ;;
stop)   docker rm -f "$NAME" ;;
*) echo "usage: float.sh {server|gen <ref> <aud> <out> [nfe]|health|logs|stop}" ;;
esac
