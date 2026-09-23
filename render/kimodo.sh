#!/usr/bin/env bash
# Kimodo text->SMPL-X motion with a PERSISTENT 8-bit GPU text-encoder server (loads the 14GB
# LLM2Vec once -> ~25s/motion instead of ~250s/run).
#   render/kimodo.sh server                       # start/ensure the encoder server
#   render/kimodo.sh gen "a person waves" wave    # generate + render -> output/wave.mp4
#   render/kimodo.sh stop                         # stop the server (frees the GPU)
# Env: ENC_GPU (encoder GPU, default 5), GEN_GPU (diffusion+render GPU, default 6).
set -euo pipefail
SAMPL="${LIFEWORLD_SAMPLES_DIR:-./samples}"
BOT="${LIFEWORLD_BOT_DIR:-./bot}"
MERGED=/work/tools/kimodo/text_encoders/merged-llm2vec-fp16
ENC_GPU="${ENC_GPU:-5}"; GEN_GPU="${GEN_GPU:-6}"; PORT=9550

case "${1:-gen}" in
server)
  docker rm -f kimodo-encoder 2>/dev/null || true
  docker run -d --name kimodo-encoder --gpus "\"device=$ENC_GPU\"" --network host \
    -e NVIDIA_DRIVER_CAPABILITIES=compute,utility -e KIMODO_QUANT=8bit -e KIMODO_MERGED="$MERGED" \
    -e HF_HOME=/work/tools/hf-cache -e HF_HUB_DISABLE_XET=1 \
    -e TEXT_ENCODERS_DIR=/work/tools/kimodo/text_encoders -e PYTHONPATH=/work/tools/kimodo_code \
    -e GRADIO_SERVER_PORT=$PORT -e GRADIO_SERVER_NAME=0.0.0.0 \
    -v "$SAMPL":/work -w /work/tools/kimodo_code kimodo:volta \
    python -m kimodo.scripts.run_text_encoder_server >/dev/null
  echo "encoder server starting on :$PORT (GPU $ENC_GPU). Ready when: curl -s -o/dev/null -w'%{http_code}' 127.0.0.1:$PORT  == 200 (~1-2 min)" ;;
stop)
  docker rm -f kimodo-encoder 2>/dev/null && echo "stopped" || echo "not running" ;;
gen)
  prompt="${2:?usage: kimodo.sh gen \"prompt\" [name]}"; name="${3:-kimodo}"
  curl -sf -o /dev/null --max-time 5 "http://127.0.0.1:$PORT/" || { echo "server not up -> run: render/kimodo.sh server"; exit 1; }
  docker run --rm --gpus "\"device=$GEN_GPU\"" --network host -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    -e TEXT_ENCODER_MODE=api -e TEXT_ENCODER_URL="http://127.0.0.1:$PORT" \
    -e HF_HOME=/work/tools/hf-cache -e HF_TOKEN -e HF_HUB_DISABLE_XET=1 -e PYTHONPATH=/work/tools/kimodo_code \
    -v "$SAMPL":/work -w /work/tools/kimodo_code kimodo:volta \
    python -m kimodo.scripts.generate "$prompt" --model Kimodo-SMPLX-RP-v1 --duration 4 \
    --diffusion_steps 100 --no-postprocess --output "/work/output/$name"
  docker run --rm --gpus "\"device=$GEN_GPU\"" -e NVIDIA_DRIVER_CAPABILITIES=all -e PYTHONPATH=/work \
    -v "$SAMPL":/work -v "$BOT":/lw lifeworld-preview \
    python3 /lw/render/kimodo_render.py "/work/output/${name}_amass.npz" "/work/output/$name.mp4" | grep KIMODO_RENDER_OK
  cp "$SAMPL/output/$name.mp4" "$BOT/output/$name.mp4"; echo "-> output/$name.mp4" ;;
*) echo "usage: kimodo.sh {server|gen \"prompt\" [name]|stop}"; exit 1 ;;
esac
