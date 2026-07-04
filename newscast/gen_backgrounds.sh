#!/usr/bin/env bash
set -uo pipefail
Z="python3 /tmp/claude-1000/-srv-nvme-data-containers-projects-bot/78ba17f7-8831-40be-97c7-8116a9c0bd11/scratchpad/zgen.py"
NEG="people, person, anchor, crowd, text, words, letters, numbers, watermark, logo, caption, ticker text, blurry, low quality"
BASE="cinematic modern television broadcast studio set, large curved glowing video wall, dramatic studio lighting, shallow depth of field, premium, empty, no people, "
g(){ $Z "$BASE$2" "$NEG" 1344 768 newscast/cast/bg_$1.png $((RANDOM)) && echo "  bg_$1"; }
g 00_intro    "deep blue and crimson signature brand lighting, sleek futuristic news set, bold and premium"
g 01_venezuela "somber muted grey-blue tones, abstract seismic ripple map on the video wall, serious gravitas"
g 02_iran     "global world map and ocean on the video wall, cool steel blue geopolitical tone"
g 03_starmer  "refined British themed set, subtle union jack blue and red on the video wall, parliamentary"
g 04_bolton   "washington dc capitol dome and justice motif on the video wall, navy and grey, official"
g 05_fed      "financial markets set, abstract glowing green and gold data streams on the video wall, wall street"
g 06_openai   "futuristic technology set, neon cyber circuit board patterns on the video wall, electric blue and purple"
g 07_heat     "climate set, intense warm orange and deep red heat glow with a blazing sun motif on the video wall"
g 08_weather  "television weather center, large radar and satellite map on the video wall, blue meteorology graphics glow"
g 09_sports   "energetic sports desk studio, stadium floodlights, dynamic red and electric blue, big arena video wall"
g 10_signoff  "deep blue and crimson signature brand lighting, sleek futuristic news set, warm closing tone"
echo BG_DONE
