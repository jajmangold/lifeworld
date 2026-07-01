#!/usr/bin/env bash
# Qwen-in-the-loop hair scale-fit: adjust scale/dy, render, let Qwen judge size+position, converge.
set -uo pipefail
cd /srv/nvme-data/containers/projects/bot/viverse
BASE="data/hair_woman01.vrm"          # composed (brown-tinted) cartoonish hair on the realistic head
S=${S:-0.5}; DY=${DY:-0.0}
ask() {  # $1 = png -> prints "SIZE: .. POS: .." lines
  python3 - "$1" <<'PY'
import sys,base64,json,urllib.request
b64=base64.b64encode(open(sys.argv[1],"rb").read()).decode()
ask=("3D avatar of a woman with hair. Judge ONLY the hair FIT. Is the hair the right SIZE for her head, "
 "or too big / too small? Is it POSITIONED right, or sitting too high / too low (covering the face)? "
 "Answer EXACTLY two lines:\nSIZE: <big|small|good>\nPOS: <high|low|good>")
body=json.dumps({"model":"qwen","temperature":0.7,"top_p":0.8,"top_k":20,"presence_penalty":1.5,
 "repeat_penalty":1.0,"max_tokens":120,"messages":[{"role":"user","content":[
 {"type":"text","text":ask},{"type":"image_url","image_url":{"url":"data:image/png;base64,"+b64}}]}]}).encode()
req=urllib.request.Request("http://localhost:8021/v1/chat/completions",data=body,headers={"Content-Type":"application/json"})
print(json.load(urllib.request.urlopen(req,timeout=120))["choices"][0]["message"]["content"])
PY
}
for i in 1 2 3 4 5 6; do
  python3 tools/hairfit.py --in "$BASE" --out data/hair_fit.vrm --scale "$S" --dy "$DY" >/dev/null
  docker compose exec -T browser python /app/tools/vrm_render.py --vrm /data/hair_fit.vrm --view face --env --out /data/hair_fit_$i.png >/dev/null 2>&1
  V=$(ask "data/hair_fit_$i.png")
  SZ=$(echo "$V" | sed -n 's/.*SIZE:[[:space:]]*\([a-z]*\).*/\1/p' | head -1)
  PO=$(echo "$V" | sed -n 's/.*POS:[[:space:]]*\([a-z]*\).*/\1/p' | head -1)
  echo "[iter $i] scale=$S dy=$DY -> SIZE=$SZ POS=$PO"
  DONE=1
  case "$SZ" in big) S=$(python3 -c "print(round($S*0.72,3))"); DONE=0;; small) S=$(python3 -c "print(round($S*1.25,3))"); DONE=0;; esac
  case "$PO" in low) DY=$(python3 -c "print(round($DY+0.03,3))"); DONE=0;; high) DY=$(python3 -c "print(round($DY-0.03,3))"); DONE=0;; esac
  [ "$DONE" = 1 ] && { echo "CONVERGED scale=$S dy=$DY (iter $i) -> data/hair_fit_$i.png"; cp data/hair_fit_$i.png data/hair_fit_best.png; break; }
  cp data/hair_fit_$i.png data/hair_fit_best.png
done
echo "HAIRLOOP_DONE final scale=$S dy=$DY"
