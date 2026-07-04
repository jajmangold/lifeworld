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
CHAR=""; AUDIO=""; OUT=""; FACE=""; PANO=output/field_pano_up.png; EXTRA=""; PASSES=1
while [ $# -gt 0 ]; do case "$1" in
  --char) CHAR=$2; shift 2;; --audio) AUDIO=$2; shift 2;; --out) OUT=$2; shift 2;;
  --face) FACE=$2; shift 2;; --pano) PANO=$2; shift 2;;
  --premium) EXTRA="$EXTRA --premium"; shift;;
  --ultra) EXTRA="$EXTRA --ultra"; shift;;
  --ltx) EXTRA="$EXTRA --ltx"; shift;;   # LTX-2.3 realism pass (photorealize; the route past the rendered look)
  --swappasses) PASSES=$2; shift 2;;   # 1 (default, enough on the photoreal HumGen base) | 2 (double-swap, for tough off-angle shots)
  --seed) EXTRA="$EXTRA --seed $2"; shift 2;;
  *) echo "unknown arg: $1"; exit 1;; esac; done
[ -z "$CHAR" ]  && { echo "need --char <asset.blend>"; exit 1; }
[ -z "$AUDIO" ] && { echo "need --audio"; exit 1; }
[ -z "$OUT" ]   && { echo "need --out"; exit 1; }
# Swap a REAL face onto the 3D character (keepeyes=false so the real source's eyes come in). On the photoreal
# HumGen render, a SINGLE pass is already photoreal (verified single==double); --swappasses 2 (double-swap) is
# available for weaker/off-angle shots where one pass still reads CG. --face overrides the default identity.
FACE=${FACE:-reporter_face.png}
exec bash make_anchor.sh --character "$CHAR" --audio "$AUDIO" --out "$OUT" \
  --face "$FACE" --keepeyes false --swappasses "$PASSES" --headtex "" --format fullscreen_anchor --pano "$PANO" $EXTRA
