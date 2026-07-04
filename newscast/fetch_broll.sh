#!/usr/bin/env bash
# Fetch short b-roll sections for motion-heavy stories (demo use; copyright per source).
# Prefer <=720p, normal-length videos; grab a 12s section.
set -uo pipefail
BOT=/srv/nvme-data/containers/live/studio
Y="$BOT/newscast/bin/yt-dlp"
declare -A Q=(
  [01_venezuela]="earthquake collapsed building rescue aftermath"
  [02_iran]="navy destroyer warship at sea"
  [05_fed]="wall street stock exchange trading floor"
  [07_heat]="extreme heat wave sun city summer haze"
  [08_worldcup]="soccer stadium crowd cheering"
)
for id in "${!Q[@]}"; do
  out="$BOT/newscast/clips/${id}.mp4"
  [ -s "$out" ] && { echo "skip $id"; continue; }
  echo "fetch $id : ${Q[$id]}"
  timeout 180 "$Y" --no-warnings --quiet \
    --match-filter "duration > 20 & duration < 900" \
    -f "best[height<=720][ext=mp4]/best[height<=720]/best[ext=mp4]" \
    --download-sections "*00:15-00:27" --force-keyframes-at-cuts \
    -o "$BOT/newscast/clips/${id}.%(ext)s" "ytsearch3:${Q[$id]}" 2>&1 | grep -iE "Destination|Download|ERROR|Merging" | tail -2
  [ -s "$out" ] && echo "  OK $id $(du -h "$out" 2>/dev/null|cut -f1)" || echo "  FAIL $id"
done
echo "BROLL_DONE"; ls -la "$BOT/newscast/clips/" 2>/dev/null | awk '{print $9,$5}'
