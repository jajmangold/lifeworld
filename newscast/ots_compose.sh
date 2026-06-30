#!/bin/bash
# Composite over-the-shoulder panels onto the finished talking-anchor video, switching at segment
# boundaries. Usage: ots_compose.sh <in.mp4> <out.mp4> <panel1.png> <end1> [<panel2.png> <end2> ...]
# Panels appear upper-right; each panelN shows from the previous end to <endN> (seconds).
set -e
IN="$1"; OUT="$2"; shift 2
POS="x=W-w-28:y=36"
inputs=(-i "$IN"); fc=""; cur="[0:v]"; prev=0; idx=1
while [ $# -ge 2 ]; do
  png="$1"; end="$2"; shift 2
  inputs+=(-i "$png")
  fc+="${cur}[${idx}:v]overlay=${POS}:enable='between(t,${prev},${end})'[v${idx}];"
  cur="[v${idx}]"; prev="$end"; idx=$((idx+1))
done
fc="${fc%;}"
last="${cur}"
ffmpeg -y "${inputs[@]}" -filter_complex "$fc" -map "$last" -map 0:a \
  -c:v libx264 -pix_fmt yuv420p -crf 18 -c:a copy "$OUT"
echo "OTS_COMPOSE_OK -> $OUT"
