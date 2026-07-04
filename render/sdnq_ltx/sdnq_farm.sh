#!/bin/bash
# Parallel SDNQ LTX-2.3 vertical-anchor farm on the sm_70 box. Each job = one clip on one card (offload),
# embeds cached to disk (gemma skipped). Runs inside the comfyui/sibling container. RAM caps concurrency
# (~28GB/job in 130GB -> ~4-6 jobs). Usage: sdnq_farm.sh <N_jobs> [frames=57] [container=sdnq-mgpu]
set -e
N=${1:-4}; F=${2:-57}; C=${3:-sdnq-mgpu}
echo "launching $N parallel ${F}f vertical-anchor jobs in $C (cards 0..$((N-1)))"
for i in $(seq 0 $((N-1))); do
  docker exec -d "$C" bash -lc "cd /root/ComfyUI && CUDA_VISIBLE_DEVICES=$i PYTORCH_ALLOC_CONF=expandable_segments:True \
    python3 sdnq_prod.py 512 768 $F 8 $((1000+i)) /root/ComfyUI/output/farm_$i > /tmp/farm_$i.log 2>&1"
done
echo "jobs launched. tail: docker exec $C bash -lc 'grep -a PROD /tmp/farm_*.log'"
