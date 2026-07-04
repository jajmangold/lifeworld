#!/bin/bash
# dispatch.sh — host-side glue for the resident farm. Routes:
#   gen daemons' latents (.pt) -> shared decode inbox -> decode daemons -> low-res frames
#   -> rtx0 FlashVSR (NFVSR workers, round-robin) 2x -> finals/
# Runs a single sweep (call from cron/loop, or pass `watch` to loop). Needs: docker (sdnq-mgpu), ssh/scp rtx0.
#
#   ./dispatch.sh [once|watch]        env: NGEN=6 NFVSR=4 FVSR_PORT0=8811
# rtx0 must run NFVSR FlashVSR workers on consecutive ports FVSR_PORT0..FVSR_PORT0+NFVSR-1 (one per RTX 3060):
#   for w in 0 1 2 3; do ssh rtx0 "cd /mnt/datadisk/containers/wan2gp && docker run -d --name fvsr-w$w \
#     --gpus \"\\\"device=$w\\\"\" -e FLASHVSR_PORT=$((8811+w)) -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
#     -v \$PWD:/workspace -p $((8811+w)):$((8811+w)) --entrypoint python3 deepbeepmeep/wan2gp:3060 /workspace/flashvsr_server.py"; done
set -e
C=sdnq-mgpu
NGEN=${NGEN:-6}
NFVSR=${NFVSR:-4}
FVSR_PORT0=${FVSR_PORT0:-8811}
LAT=/root/ComfyUI/output/latents
FIN=/tmp/claude-1000/finals; mkdir -p "$FIN"
FVSR_IN=/mnt/datadisk/fvsr/FlashVSR-Pro/inputs
FVSR_OUT=/mnt/datadisk/fvsr/FlashVSR-Pro/results
STATE="$FIN/.seen"; touch "$STATE"

ship() {   # $1=stem  $2=worker index (0..NFVSR-1)
  local stem="$1" w="$2" port=$((FVSR_PORT0 + $2))
  exec 9>"$FIN/.fvsr_w$w.lock"; flock 9         # one clip per worker at a time
  local tmp; tmp=$(mktemp -d)
  docker cp $C:$LAT/frames/$stem/. "$tmp"/ 2>/dev/null
  ffmpeg -y -framerate 25 -i "$tmp/f%03d.png" -c:v libx264 -pix_fmt yuv420p -crf 14 "$tmp/lr.mp4" 2>/dev/null
  scp -q "$tmp/lr.mp4" rtx0:$FVSR_IN/${stem}_lr.mp4
  ssh rtx0 "curl -sf -X POST --data-binary @$FVSR_IN/${stem}_lr.mp4 'http://localhost:$port/upscale?mode=video&scale=2&ext=mp4' -o $FVSR_OUT/${stem}_2x.mp4"
  scp -q rtx0:$FVSR_OUT/${stem}_2x.mp4 "$FIN/${stem}_512x768.mp4"
  echo "$stem" >> "$STATE"; rm -rf "$tmp"; flock -u 9
  echo "[dispatch w$w] $stem -> $FIN/${stem}_512x768.mp4"
}
export -f ship; export C LAT FIN FVSR_IN FVSR_OUT STATE FVSR_PORT0

sweep() {
  # 1) funnel new latents from every gen daemon into the shared decode inbox (prefix by gen card)
  for c in $(seq 0 $((NGEN-1))); do
    docker exec $C bash -lc "cd /root/ComfyUI/output/resdjobs$c/out 2>/dev/null && for f in *.pt; do [ -e \"\$f\" ] || continue; mv \"\$f\" $LAT/${c}_\$f; mv \"\${f%.pt}.meta.json\" $LAT/${c}_\${f%.pt}.meta.json 2>/dev/null; done" 2>/dev/null || true
  done
  # 2) collect finished-and-unshipped stems (decode wrote frames AND moved the .pt to done/)
  local ready=() i=0
  for stem in $(docker exec $C bash -lc "ls $LAT/frames 2>/dev/null" 2>/dev/null); do
    grep -qx "$stem" "$STATE" && continue
    docker exec $C bash -lc "ls $LAT/done/${stem}.pt >/dev/null 2>&1" || continue
    ready+=("$stem $((i % NFVSR))"); i=$((i+1))
  done
  # 3) ship concurrently across the NFVSR FlashVSR workers (per-worker flock serializes each)
  [ ${#ready[@]} -eq 0 ] && return
  printf '%s\n' "${ready[@]}" | xargs -P "$NFVSR" -n2 bash -c 'ship "$0" "$1"'
}

if [ "${1:-once}" = watch ]; then while true; do sweep; sleep 5; done; else sweep; fi
