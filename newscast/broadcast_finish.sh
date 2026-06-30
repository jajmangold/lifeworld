#!/bin/bash
# Composite the FFNN graphics package over a finished 1440p anchor clip, with animation:
#  - bug (top-right, static)
#  - lower-third (slides in from left ~1s, slides out near end)
#  - ticker: static bar + scrolling headlines (behind the LIVE tab) + time box + clock
#  - subtle music bed under the VO
#   broadcast_finish.sh <in.mp4> <out.mp4> [music.mp3]
set -e
IN=$1; OUT=$2; MUSIC=$3
BOT=/srv/nvme-data/containers/projects/bot; cd "$BOT"
G=output
DUR=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$IN")
TW=$(ffprobe -v error -select_streams v -show_entries stream=width -of csv=p=0 "$G/gfx_ticker_text.png")
LTW=$(ffprobe -v error -select_streams v -show_entries stream=width -of csv=p=0 "$G/gfx_lower_third.png")
CLOCK=$(date +"%-I\\:%M %p")     # escape the colon for ffmpeg drawtext
SPEED=190                       # ticker px/sec
OUTTRO=$(awk "BEGIN{print $DUR-1.6}")    # lower-third slides out

# overlay graph:
#  [bg over in] -> ticker bar; [+text] scroll; [+fg] tab/time cover; [+bug]; [+lower third slide]; drawtext clock
FF="[0:v][1:v]overlay=0:H-92[a];\
[a][2:v]overlay=x='250 - mod(t*$SPEED\,$TW)':y=H-92[b];\
[b][3:v]overlay=0:H-92[c];\
[c][4:v]overlay=x=W-overlay_w-34:y=34[d];\
[d][5:v]overlay=x='if(lt(t,1),-$LTW, if(lt(t,1.6), -$LTW+($LTW+70)*(t-1)/0.6, if(gt(t,$OUTTRO), 70-($LTW+70)*(t-$OUTTRO)/1.6, 70)))':y=H-92-overlay_h-26[v]"

if [ -n "$MUSIC" ] && [ -f "$MUSIC" ]; then
  ffmpeg -y -i "$IN" -i "$G/gfx_ticker_bg.png" -i "$G/gfx_ticker_text.png" -i "$G/gfx_ticker_fg.png" \
    -i "$G/gfx_bug.png" -i "$G/gfx_lower_third.png" -stream_loop -1 -i "$MUSIC" \
    -filter_complex "$FF;[0:a]aformat=fltp:44100:stereo,volume=1.0[vo];[6:a]aformat=fltp:44100:stereo,volume=0.10[mu];[vo][mu]amix=inputs=2:duration=first:dropout_transition=0[aout]" \
    -map "[v]" -map "[aout]" -c:v libx264 -pix_fmt yuv420p -crf 17 -c:a aac -b:a 192k -shortest "$OUT"
else
  ffmpeg -y -i "$IN" -i "$G/gfx_ticker_bg.png" -i "$G/gfx_ticker_text.png" -i "$G/gfx_ticker_fg.png" \
    -i "$G/gfx_bug.png" -i "$G/gfx_lower_third.png" \
    -filter_complex "$FF" -map "[v]" -map 0:a -c:v libx264 -pix_fmt yuv420p -crf 17 -c:a aac -b:a 192k "$OUT"
fi
echo "BROADCAST_FINISH_OK -> $OUT"
