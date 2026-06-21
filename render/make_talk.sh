#!/usr/bin/env bash
# Talking SMPL-X clip, end to end:
#   Higgs-TTS (text->wav)  ->  LAM_A2E (wav->ARKit, HOST-side)  ->
#   bake_talk (SMPL-X face)  ->  render_smplx (GPU, Blender-free)  ->  mp4
#
# LAM and Higgs are host services (127.0.0.1), so the TTS + LAM steps run on the
# host; bake (CPU) and render (GPU) run in containers. Usage:
#   render/make_talk.sh "Some line of dialogue." female 7
set -euo pipefail
TEXT="${1:-Hi. I am finally awake.}"
GENDER="${2:-female}"
GPU="${3:-7}"
FPS=24

SAMPL=/srv/nvme-data/containers/projects/sampl
BOT=/srv/nvme-data/containers/projects/bot
OUT=$SAMPL/output
TEX=$SAMPL/assets/smplx_texture_f_alb_eyefix.png   # clothed + patched eye (see fix_eye_texture.py)

# 1. TTS (Higgs :8055) + 2. LAM ARKit (:8202) — host python, stdlib only
python3 - "$TEXT" "$OUT" <<'PY'
import json, sys, urllib.request, wave
text, out = sys.argv[1], sys.argv[2]
spe = f"{out}/speech.wav"
body = json.dumps({"input": text, "response_format": "wav"}).encode()
req = urllib.request.Request("http://127.0.0.1:8055/v1/audio/speech", data=body,
                             headers={"Content-Type": "application/json"})
open(spe, "wb").write(urllib.request.urlopen(req, timeout=180).read())
dur = (lambda w: w.getnframes()/w.getframerate())(wave.open(spe, "rb"))
print(f"TTS {dur:.2f}s -> {spe}")
# LAM a2e (multipart)
b = "----lamform"; wav = open(spe, "rb").read()
mp = (f"--{b}\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"a.wav\"\r\n"
      f"Content-Type: audio/wav\r\n\r\n").encode() + wav + \
     (f"\r\n--{b}\r\nContent-Disposition: form-data; name=\"id_idx\"\r\n\r\n0\r\n--{b}--\r\n").encode()
r = urllib.request.Request("http://127.0.0.1:8202/a2e", data=mp,
                           headers={"Content-Type": f"multipart/form-data; boundary={b}"})
try:
    jid = json.load(urllib.request.urlopen(r, timeout=180))["job_id"]
    ak = json.load(urllib.request.urlopen(f"http://127.0.0.1:8202/jobs/{jid}/anim.arkit.json", timeout=60))
    json.dump(ak, open(f"{out}/speech.arkit.json", "w"))
    print(f"LAM ok: {len(ak['weights'])} frames")
    open(f"{out}/.frames", "w").write(str(round(dur*24)))
except Exception as e:
    print("LAM unavailable, will use amplitude fallback:", e)
    open(f"{out}/.frames", "w").write(str(round(dur*24)))
PY
FRAMES=$(cat "$OUT/.frames")
ARKIT=""; [ -f "$OUT/speech.arkit.json" ] && ARKIT="--arkit /work/output/speech.arkit.json"

# 3. bake face (CPU, sampl on PYTHONPATH for face.talk)
docker run --rm -v "$SAMPL":/work -v "$BOT":/lw -w /work -e PYTHONPATH=/work sampl:dev \
  bash -lc "\$SAMPL_VENV/bin/python /lw/render/bake_talk.py --audio /work/output/speech.wav \
    $ARKIT --frames $FRAMES --gender $GENDER --model-dir /work/models --out /work/output/clip_talk.npz"

# 4. render (GPU, Blender-free)
docker run --rm --gpus "device=$GPU" -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics \
  -v "$SAMPL":/work -v "$BOT":/lw -w /work lifeworld-pyrender bash -c "
set -e
python3 /lw/render/render_smplx.py --clip /work/output/clip_talk.npz --out-dir /work/output/frames_talk \
  --uv /work/assets/smplx_uv_2023.npz --texture $TEX --framing medium --rot-x 0 --res-x 720 --res-y 900
ffmpeg -y -loglevel error -framerate $FPS -i /work/output/frames_talk/frame_%04d.png \
  -c:v libx264 -pix_fmt yuv420p -crf 18 /work/output/talk_lam.mp4"
echo "DONE -> $OUT/talk_lam.mp4"
