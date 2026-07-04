#!/bin/bash
# ltx_anchor_segment.sh — LTX-clip anchor segment (order: muse -> swap[last] -> chunked FlashVSR -> matte -> composite -> graphics)
#   ./ltx_anchor_segment.sh NAME GREEN_CLIP AUDIO_WAV [FACE=anchorM.png] [SET=newsroom.png] [MUSIC]
set -e
BOT=/srv/nvme-data/containers/live/studio
STU=/srv/nvme-data/containers/live/studio
NAME=$1; GREEN=$2; AUDIO=$3; FACE=${4:-anchorM.png}; SET=${5:-$STU/newscast/assets/newsroom.png}; MUSIC=${6:-$STU/newscast/music/news_bed.mp3}
O=$BOT/output; FIN=/mnt/datadisk/fvsr/FlashVSR-Pro
log(){ echo "[ltxseg $NAME] $*"; }
cp "$GREEN" $O/${NAME}_src.mp4
# 1) muse (motion on the green LTX face)
ffmpeg -y -i "$AUDIO" -af "highpass=f=70,agate=threshold=0.045:ratio=12:attack=6:release=180:range=0.0" -ar 16000 -ac 1 $O/${NAME}_16k.wav 2>/dev/null
rm -f $O/muse_jobs/${NAME}.done $O/muse_jobs/${NAME}.err
printf '{"video":"/io/%s_src.mp4","audio":"/io/%s_16k.wav","out":"/io/%s_muse.mp4"}\n' "$NAME" "$NAME" "$NAME" > $O/muse_jobs/${NAME}.json
until [ -f $O/muse_jobs/${NAME}.done ] || [ -f $O/muse_jobs/${NAME}.err ]; do sleep 4; done
[ -f $O/muse_jobs/${NAME}.err ] && { echo "MUSE ERR"; exit 1; }; log "muse done"
# 2) swap LAST (identity + re-sharpen the muse mouth)
rm -f $O/swap_jobs/${NAME}.done $O/swap_jobs/${NAME}.err
printf '{"src":"/o/%s","video":"/o/%s_muse.mp4","out":"/o/%s_swap.mp4","keepeyes":true,"enhance":true,"feather":true,"passes":2}\n' "$FACE" "$NAME" "$NAME" > $O/swap_jobs/${NAME}.json
until [ -f $O/swap_jobs/${NAME}.done ] || [ -f $O/swap_jobs/${NAME}.err ]; do sleep 3; done
[ -f $O/swap_jobs/${NAME}.err ] && { echo "SWAP ERR: $(cat $O/swap_jobs/${NAME}.err)"; exit 1; }; log "swap done"
# 3) chunked FlashVSR (avoid the 3060 OOM on long clips): 48-frame chunks, rotate workers, concat
CH=$O/${NAME}_fvsr; rm -rf $CH; mkdir -p $CH
ffmpeg -y -i $O/${NAME}_swap.mp4 -c:v libx264 -pix_fmt yuv420p -g 48 -force_key_frames "expr:gte(t,n_forced*1.92)" -f segment -segment_time 1.92 -reset_timestamps 1 $CH/seg%03d.mp4 2>/dev/null
i=0; : > $CH/concat.txt
for seg in $CH/seg*.mp4; do
  port=$((8811 + i % 4)); b=$(basename $seg)
  scp -q $seg rtx0:$FIN/inputs/${NAME}_$b
  ssh rtx0 "curl -sf -m 240 -X POST --data-binary @$FIN/inputs/${NAME}_$b 'http://localhost:$port/upscale?mode=video&scale=2&ext=mp4' -o $FIN/results/${NAME}_${b%.mp4}_2x.mp4"
  scp -q rtx0:$FIN/results/${NAME}_${b%.mp4}_2x.mp4 $CH/${b%.mp4}_2x.mp4
  echo "file '$CH/${b%.mp4}_2x.mp4'" >> $CH/concat.txt; i=$((i+1))
done
ffmpeg -y -f concat -safe 0 -i $CH/concat.txt -c:v libx264 -pix_fmt yuv420p -crf 15 $O/${NAME}_anchor.mp4 2>/dev/null
log "flashvsr(chunked x$i) -> $(ffprobe -v error -select_streams v -show_entries stream=width,height -of csv=p=0 $O/${NAME}_anchor.mp4 2>/dev/null)"
# 4) matte + composite behind the newsroom desk @1440p
python3 $STU/render/ltx_matte_composite.py $O/${NAME}_anchor.mp4 "$SET" $O/${NAME}_composited.mp4 "$AUDIO" 2560 1440
log "composited @1440p"
# 5) graphics package (bug + lower-third + ticker + music) via the existing finisher
bash $STU/newscast/broadcast_finish.sh $O/${NAME}_composited.mp4 $O/${NAME}_final.mp4 "$MUSIC" 2>&1 | grep -aE "BROADCAST_FINISH_OK|Error" | tail -1
log "FINAL -> $O/${NAME}_final.mp4"
