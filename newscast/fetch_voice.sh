#!/usr/bin/env bash
set -uo pipefail
BOT=/srv/nvme-data/containers/live/studio; CR=/srv/nvme-data/containers/projects/CrispASR
Y=$BOT/newscast/bin/yt-dlp; OUT=$1; G=$2; Q=$3
tmp=$(mktemp -d)
$Y --no-warnings -x --audio-format wav --postprocessor-args "ExtractAudio:-ar 24000 -ac 1" \
   --match-filter "duration > 50 & duration < 1500" --download-sections "*00:40-00:58" \
   -o "$tmp/c%(autonumber)s.%(ext)s" "ytsearch6:$Q" >/dev/null 2>&1 || true
echo "downloaded $(ls $tmp/*.wav 2>/dev/null|wc -l) candidates for $OUT"
chosen=$(docker run --rm -v "$tmp":/t -v $BOT/newscast:/n lifeworld-pyrender python3 /n/f0.py --pick /t $G 2>/dev/null | tail -1 | tr -d '[:space:]')
if [ -n "$chosen" ] && [ -s "$tmp/$chosen" ]; then cp "$tmp/$chosen" "$CR/voices/$OUT.wav"; echo "PICKED $OUT <- $chosen"; else echo "FETCH_FAIL $OUT"; fi
rm -rf "$tmp"
