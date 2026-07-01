#!/bin/bash
# Launch the resident face-swap server (loads insightface+inswapper+gfpgan ONCE, then polls a job queue).
# Eliminates the ~25s model-load paid per old `docker run inswap:local`.
#   run_swap.sh         # (re)create + start
#   run_swap.sh stop
# Submit: write bot/output/swap_jobs/<n>.json {"src","video","out"[,"keepeyes"]} -> <n>.done/.err
set -e
NAME=swap-server
GPU=${SWAP_GPU:-0}
BOT=/srv/nvme-data/containers/live/studio
[ "$1" = "stop" ] && { docker rm -f "$NAME" 2>/dev/null && echo "stopped $NAME"; exit 0; }
docker rm -f "$NAME" 2>/dev/null || true
docker run -d --name "$NAME" --restart unless-stopped --gpus "\"device=$GPU\"" \
  -e SWAP_JOBS=/o/swap_jobs -w /s \
  -v "$BOT/swap":/s -v "$BOT/output":/o \
  inswap:local python swap_server.py
echo "started $NAME (inswap:local, GPU $GPU). poll: docker logs --tail 5 $NAME | grep SWAP_SERVER_READY"
