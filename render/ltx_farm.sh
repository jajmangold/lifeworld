#!/bin/bash
# FARM of resident LTX-2.3 servers — one per RTX 3060, each UUID-PINNED to a fixed card — so up to N clips
# photorealize in PARALLEL. Each holds a Q3_K_S DiT RESIDENT (profile 3, ~10.5GB) and all share ONE atomic job
# queue (ltx_server.py claims via atomic os.rename -> no races). ltx_realism.sh just drops jobs; the free card
# grabs the next -> ~Nx throughput.
#
# STAGGERED startup: this box has only 30GB RAM, so 4 concurrent 10GB weight-hydrations swap-thrash it to death.
# We start ONE server, wait for its LTX_SERVER_READY (the server self-warms => weights already resident), THEN
# start the next. So only ONE card ever hydrates at a time — safe on reboots and cold starts.
#
#   ltx_farm.sh [ngpus=4]     |     ltx_farm.sh stop
set -e
BOT=/srv/nvme-data/containers/live/studio
RTX="ssh -o BatchMode=yes josh@rtx0"; W=/mnt/datadisk/containers/wan2gp
N=${1:-4}

if [ "$1" = "stop" ]; then
  $RTX "docker exec wan2gp bash -lc 'for p in \$(pgrep -f ltx_server.py); do kill -9 \$p 2>/dev/null; done'" || true
  echo "farm stopped"; exit 0
fi

# free the cards + clean slate
$RTX "docker update --restart=no flashvsr-w1 flashvsr-w2 flashvsr-queue >/dev/null 2>&1; docker stop flashvsr-w1 flashvsr-w2 flashvsr-queue" >/dev/null 2>&1 || true
$RTX "docker exec wan2gp bash -lc 'for p in \$(pgrep -f ltx_server.py); do kill -9 \$p 2>/dev/null; done; sleep 3; rm -f /workspace/ltxjobs/*.proc /workspace/ltxjobs/server_gpu*.log /workspace/ltxjobs/warm_*'" || true
scp -q "$BOT/render/ltx_server.py" josh@rtx0:$W/ltx_server.py
[ -f /tmp/_warmup.mp4 ] || ffmpeg -y -i "$BOT/output/jessica_v2.mp4" -vf "scale=512:320,fps=25" -frames:v 25 -an /tmp/_warmup.mp4 2>/dev/null
$RTX "test -f $W/ltxjobs/_warmup.mp4" || scp -q /tmp/_warmup.mp4 josh@rtx0:$W/ltxjobs/_warmup.mp4

# stable per-index GPU UUIDs (index can shuffle across reboots; UUID pins each server to its physical card)
mapfile -t UUIDS < <($RTX "nvidia-smi --query-gpu=uuid --format=csv,noheader")
echo "GPUs: ${#UUIDS[@]} | launching $N servers STAGGERED (one hydrates at a time)"

# RAM guard: each server pins its ~10GB weights in RAM; on this 30GB box that caps the farm well below 4. After
# each server settles, if free RAM is too low to safely load ANOTHER, stop adding servers (auto-fit to RAM).
RAM_FLOOR=${LTX_RAM_FLOOR:-11}   # GB of free RAM required to start the next server's ~10GB hydration
UP=0
for g in $(seq 0 $((N-1))); do
  FREE=$($RTX "free -g | awk 'NR==2{print \$7}'")   # 'available' GB
  if [ "$g" -gt 0 ] && [ "${FREE:-0}" -lt "$RAM_FLOOR" ]; then
    echo "  STOP: only ${FREE}GB free (< ${RAM_FLOOR}GB) — 30GB RAM fits $UP servers. (encoder-precompute or more RAM -> 4)"; break
  fi
  U=${UUIDS[$g]}
  $RTX "docker exec -d -e CUDA_VISIBLE_DEVICES=$U -e LTX_PROFILE=3 wan2gp bash -lc 'cd /workspace && python3 ltx_server.py > /workspace/ltxjobs/server_gpu$g.log 2>&1'"
  echo -n "  gpu$g ($U) hydrating"
  ok=0
  for i in $(seq 1 120); do
    $RTX "grep -aq LTX_SERVER_READY $W/ltxjobs/server_gpu$g.log 2>/dev/null" && { echo " -> READY"; ok=1; break; }
    $RTX "grep -aq 'Traceback\|Error' $W/ltxjobs/server_gpu$g.log 2>/dev/null" && { echo " -> FAILED"; $RTX "tail -8 $W/ltxjobs/server_gpu$g.log"; break; }
    echo -n "."; sleep 6
  done
  [ "$ok" = 1 ] && UP=$((UP+1))
  sleep 4   # let RAM settle post-hydration before the next check
done
echo "farm up: $UP server(s), Q3_K_S resident, UUID-pinned, staggered."
$RTX "free -g | awk 'NR==2{print \"RAM used/total: \"\$3\"/\"\$2\" GB, available: \"\$7\" GB\"}'"
