#!/bin/bash
# BlenderKit character talking-head (viverse-free). Thin wrapper over make_anchor.sh --character with
# character-appropriate defaults: photoreal face-swap (keepeyes=false so the swap's real eyes come in),
# fullscreen MCU framing, field/studio HDRI, no viverse head-tex. Full pipeline:
#   render_character -> swap(photoreal face) -> MuseTalk -> pause-lip settle -> premium/ultra -> final.
#   make_character.sh --char <asset.blend> --audio x.wav --out out.mp4
#        [--face reporter_face.png] [--pano output/field_pano_up.png] [--premium | --ultra] [--seed N]
#   --premium = FlashVSR-face 2x (1440p) ; --ultra = FlashVSR-face 4x
set -e
BOT=/srv/nvme-data/containers/live/studio; cd "$BOT"
CHAR=""; AUDIO=""; OUT=""; FACE=reporter_face.png; PANO=output/field_pano_up.png; EXTRA=""
while [ $# -gt 0 ]; do case "$1" in
  --char) CHAR=$2; shift 2;; --audio) AUDIO=$2; shift 2;; --out) OUT=$2; shift 2;;
  --face) FACE=$2; shift 2;; --pano) PANO=$2; shift 2;;
  --premium) EXTRA="$EXTRA --premium"; shift;;
  --ultra) EXTRA="$EXTRA --ultra"; shift;;
  --seed) EXTRA="$EXTRA --seed $2"; shift 2;;
  *) echo "unknown arg: $1"; exit 1;; esac; done
[ -z "$CHAR" ]  && { echo "need --char <asset.blend>"; exit 1; }
[ -z "$AUDIO" ] && { echo "need --audio"; exit 1; }
[ -z "$OUT" ]   && { echo "need --out"; exit 1; }
exec bash make_anchor.sh --character "$CHAR" --audio "$AUDIO" --out "$OUT" \
  --face "$FACE" --keepeyes false --headtex "" --format fullscreen_anchor --pano "$PANO" $EXTRA
