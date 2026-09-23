#!/usr/bin/env bash
# Viseme / lip-sync probe for one dialogue line: runs a phoneme recognizer on the audio and
# plots the expected visemes (open 'ah' / round 'oo' / closed 'm-b-p' ...) against what A2F is
# actually doing (jawOpen / round / mouthClose), with a vowel-based lag number. Spot sync issues.
#   render/probe.sh [line]      # line defaults to theo_0; e.g. render/probe.sh priya_0
set -euo pipefail
LINE="${1:-theo_0}"
SAMPL="${LIFEWORLD_SAMPLES_DIR:-./samples}"
docker run --rm -v "$SAMPL":/work -v "$SAMPL/tools/TalkSHOW":/ts -v "${LIFEWORLD_BOT_DIR:-./bot}":/lw \
  -e PYTHONPATH=/ts/pydeps -e HF_HOME=/ts/hf-cache sampl:dev bash -lc \
  "\$SAMPL_VENV/bin/python /lw/render/viseme_probe.py --wav /work/output/scene/$LINE.wav \
   --arkit /work/output/scene/$LINE.arkit.json --out /work/output/scene/probe_$LINE.png" | grep -E "VOWEL_LAG|PROBE_OK"
cp "$SAMPL/output/scene/probe_$LINE.png" "${LIFEWORLD_BOT_DIR:-./bot}/output/probe_$LINE.png"
echo "-> output/probe_$LINE.png"
