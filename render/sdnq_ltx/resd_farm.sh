#!/bin/bash
# resd_farm.sh — launch the RESIDENT low-res anchor farm inside the sdnq-mgpu container.
# Layout: N_GEN resident denoise daemons (one/card) + N_DEC resident VAE decode daemons.
#   - denoise daemon (sdnq_resd.py): transformer RESIDENT ~12GB, saves raw LTX latents (.pt) per job.
#   - decode daemon (sdnq_decoded.py): VAE RESIDENT ~1.5GB, atomic-claims latents from the SHARED inbox
#     ($LAT), writes low-res frames. Multiple decode daemons share one inbox race-free (rename claim).
# Gen throughput ~23s/clip, decode ~12s/clip -> run ~2 gen per 1 decode. dispatch.sh ships frames -> rtx0 FlashVSR.
#
#   ./resd_farm.sh [N_GEN=6] [N_DEC=3]
# Container sdnq-mgpu must expose >= N_GEN+N_DEC GPUs (recreate with more --gpus device=... if needed).
# CUDA_VISIBLE_DEVICES within the container are 0..(N_GEN+N_DEC-1), mapped in the order the container was
# created. Keep any shared/occupied host card LAST so it lands a light decode daemon, not a 12GB gen daemon.
set -e
C=sdnq-mgpu
N_GEN=${1:-6}
N_DEC=${2:-3}
LAT=/root/ComfyUI/output/latents          # shared latent inbox for all decode daemons
A="PYTORCH_ALLOC_CONF=expandable_segments:True"

NEED=$((N_GEN+N_DEC))
HAVE=$(docker exec $C bash -lc 'python3 -c "import torch;print(torch.cuda.device_count())" 2>/dev/null' | tail -1)
[ "${HAVE:-0}" -ge "$NEED" ] || { echo "ERROR: container sees $HAVE GPUs, need $NEED (N_GEN=$N_GEN N_DEC=$N_DEC)"; exit 1; }

docker exec $C bash -lc "pkill -TERM -f 'sdnq_resd.py|sdnq_decoded.py' 2>/dev/null; sleep 3; mkdir -p $LAT/done" || true
# gen daemons -> cards 0..N_GEN-1
for c in $(seq 0 $((N_GEN-1))); do
  docker exec $C bash -lc "mkdir -p /root/ComfyUI/output/resdjobs$c/done"
  docker exec -d $C bash -lc "cd /root/ComfyUI && CUDA_VISIBLE_DEVICES=$c $A python3 sdnq_resd.py /root/ComfyUI/output/resdjobs$c > /tmp/resd$c.log 2>&1"
  echo "gen daemon    -> card $c   (jobs: output/resdjobs$c/*.json)"
done
# decode daemons -> cards N_GEN..N_GEN+N_DEC-1, all sharing the $LAT inbox (atomic claim)
for d in $(seq 0 $((N_DEC-1))); do
  card=$((N_GEN+d))
  docker exec -d $C bash -lc "cd /root/ComfyUI && CUDA_VISIBLE_DEVICES=$card $A python3 sdnq_decoded.py $LAT > /tmp/decoded$d.log 2>&1"
  echo "decode daemon -> card $card   (shared inbox: $LAT)"
done
echo "wait ~85s for 'RESIDENT' in /tmp/resd*.log and 'VAE resident' in /tmp/decoded*.log"
