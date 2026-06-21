#!/usr/bin/env python3
"""Cinematic multi-character scene: generate a short dialogue, give each character a
voice + lip-sync, and render them as distinct textured SMPL-X people in one shot
with audio. Orchestrates host services (DeepSeek, Higgs TTS :8055, LAM :8202) and
two containers (sampl:dev bake, lifeworld-pyrender render+mux).

Run on the HOST:  python3 render/scene_cine.py
"""
import base64
import json
import os
import subprocess
import sys
import urllib.request
import wave

SAMPL = "/srv/nvme-data/containers/projects/sampl"
BOT = "/srv/nvme-data/containers/projects/bot"
SCENE = f"{SAMPL}/output/scene"          # shared via /work mount
HIGGS, LAM = "http://127.0.0.1:8055", "http://127.0.0.1:8202"
FPS = 24

# cast: row facing camera, angled slightly inward; distinct gender/shape/voice/texture
CAST = [
    {"name": "Mara",  "gender": "female", "betas": [1.5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
     "pos": [-0.85, 0.0], "yaw_deg": 22, "tex": "f", "voice": None, "a2f_id": "Claire"},
    {"name": "Theo",  "gender": "male",   "betas": [0.5, 1.0, 0, 0, 0, 0, 0, 0, 0, 0],
     "pos": [0.0, 0.25], "yaw_deg": 0,  "tex": "m", "voice": "male", "a2f_id": "Mark"},
    {"name": "Priya", "gender": "female", "betas": [-1.2, -0.5, 0, 0, 0, 0, 0, 0, 0, 0],
     "pos": [0.9, 0.0],  "yaw_deg": -22, "tex": "f", "voice": None, "a2f_id": "Claire"},
]
A2F_DIR = "/mnt/24tb/a2f"
PERSONA = {"Mara": "tidy, anxious, protective of her food",
           "Theo": "easygoing, forgetful musician, sheepish",
           "Priya": "blunt, funny peacemaker"}
SEED = ("Saturday morning; Mara's labeled leftovers vanished from the fridge and the "
        "three roommates hash it out.")


def deepseek(system, user, temp=0.9):
    body = {"model": os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
            "temperature": temp, "max_tokens": 200,
            "response_format": {"type": "json_object"}}
    last = None
    for _ in range(4):                       # DeepSeek occasionally returns an empty body
        try:
            req = urllib.request.Request(
                f"{os.environ.get('DEEPSEEK_BASE_URL','https://api.deepseek.com')}/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"})
            txt = json.load(urllib.request.urlopen(req, timeout=60))["choices"][0]["message"]["content"]
            try:
                return json.loads(txt)
            except Exception:
                s, e = txt.find("{"), txt.rfind("}")
                return json.loads(txt[s:e + 1])
        except Exception as ex:
            last = ex
    raise last


def tts(text, out_wav, voice=None):
    # NOTE: this Higgs build does NOT interpret <|emotion:..|> tokens — it speaks them
    # aloud — so we never prepend them. Voice variety comes from reference_audio only.
    body = {"input": text, "response_format": "wav"}
    if voice == "male":
        ref = f"{SAMPL}/assets/voices/male_ref.wav"
        body["reference_audio"] = base64.b64encode(open(ref, "rb").read()).decode()
        body["reference_text"] = "Promised you would finish the nursery this weekend."
    req = urllib.request.Request(f"{HIGGS}/v1/audio/speech", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    open(out_wav, "wb").write(urllib.request.urlopen(req, timeout=180).read())


def lam(wav, out_json):
    b = "----lamform"; data = open(wav, "rb").read()
    mp = (f"--{b}\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"a.wav\"\r\n"
          f"Content-Type: audio/wav\r\n\r\n").encode() + data + \
         (f"\r\n--{b}\r\nContent-Disposition: form-data; name=\"id_idx\"\r\n\r\n0\r\n--{b}--\r\n").encode()
    r = urllib.request.Request(f"{LAM}/a2e", data=mp,
                               headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    jid = json.load(urllib.request.urlopen(r, timeout=180))["job_id"]
    ak = json.load(urllib.request.urlopen(f"{LAM}/jobs/{jid}/anim.arkit.json", timeout=60))
    json.dump(ak, open(out_json, "w"))


def main():
    os.makedirs(SCENE, exist_ok=True)
    names = [c["name"] for c in CAST]

    scene_json = f"{SCENE}/scene.json"
    if "--reuse" in sys.argv and os.path.exists(scene_json):
        # FAST re-render: reuse cached dialogue/TTS/A2F, only re-bake + re-render.
        beats = json.load(open(scene_json))["beats"]
        print(f"[scene_cine] --reuse: {len(beats)} cached beats (skipping dialogue/TTS/A2F)")
    else:
        # 1. generate a short ordered dialogue (2 rounds)
        transcript, beats = [], []
        for rnd in range(2):
            for i, c in enumerate(CAST):
                convo = "\n".join(f"{t['who']}: {t['say']}" for t in transcript[-6:])
                out = deepseek(
                    f"You are {c['name']}: {PERSONA[c['name']]}. In a room with "
                    f"{', '.join(n for n in names if n != c['name'])}. ONE short spoken line. Reply ONLY JSON.",
                    f"SCENE: {SEED}\nSo far:\n{convo or '(start)'}\nYour line. "
                    '{"say":"<one short sentence>"}')
                say = out.get("say", "").strip()
                transcript.append({"who": c["name"], "say": say})
                wav = f"{SCENE}/{c['name'].lower()}_{rnd}.wav"
                ak = f"{SCENE}/{c['name'].lower()}_{rnd}.arkit.json"
                tts(say, wav, c["voice"])       # lip-sync (A2F) batched below
                dur = wave.open(wav, "rb").getnframes() / wave.open(wav, "rb").getframerate()
                beats.append({"speaker": i, "audio": wav.replace(SAMPL, "/work"),
                              "arkit": ak.replace(SAMPL, "/work"), "wav_host": wav,
                              "a2f_id": c["a2f_id"], "dur": dur})
                print(f"  {c['name']}: {say}  ({dur:.1f}s)")

        # 1b. Audio2Face-3D lip-sync for every line in ONE container session (CMP/V100, ORT)
        manifest = f"{SCENE}/a2f_manifest.json"
        json.dump([{"wav": b["audio"].replace("/work", "/asset"),
                    "out": b["arkit"].replace("/work", "/asset"),
                    "identity": b["a2f_id"]} for b in beats], open(manifest, "w"))
        run(["docker", "run", "--rm", "--gpus", "device=1",
             "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
             "-v", f"{SAMPL}:/asset", "-v", f"{BOT}:/lw", "-v", f"{A2F_DIR}:/a2f", "-w", "/lw",
             "lifeworld-a2f", "python3", "render/a2f_lipsync.py", "--a2f", "/a2f",
             "--manifest", "/asset/output/scene/a2f_manifest.json"])

        # 2. scene config for the baker
        chars = [{"name": c["name"], "gender": c["gender"], "betas": c["betas"],
                  "pos": c["pos"], "yaw_deg": c["yaw_deg"]} for c in CAST]
        cfg = {"fps": FPS, "characters": chars,
               "beats": [{"speaker": b["speaker"], "audio": b["audio"], "arkit": b["arkit"]} for b in beats]}
        json.dump(cfg, open(scene_json, "w"))

    # 3. bake the multi-person clip (CPU, sampl:dev)
    run(["docker", "run", "--rm", "-v", f"{SAMPL}:/work", "-v", f"{BOT}:/lw",
         "-w", "/work", "-e", "PYTHONPATH=/work", "sampl:dev", "bash", "-lc",
         "$SAMPL_VENV/bin/python /lw/render/bake_scene.py --config /work/output/scene/scene.json "
         "--out /work/output/scene/clip.npz"])

    # 4. concat audio (in beat order) on the host-visible files
    listf = f"{SCENE}/audio.txt"   # paths must be CONTAINER paths (ffmpeg runs in-container)
    open(listf, "w").write("".join(f"file '{b['audio']}'\n" for b in beats))

    # 5. ensure male eye-fixed texture exists, then render + mux (lifeworld-pyrender)
    texf = f"{SAMPL}/assets/smplx_texture_f_alb_eyefix.png"
    texm = f"{SAMPL}/assets/smplx_texture_m_alb_eyefix.png"
    if not os.path.exists(texm):
        import shutil; shutil.copy(f"{SAMPL}/assets/smplx_texture_m_alb.png", texm)
        run(["docker", "run", "--rm", "-v", f"{SAMPL}:/work", "-v", f"{BOT}:/lw",
             "-w", "/work", "-e", "PYTHONPATH=/work", "lifeworld-pyrender", "python3", "-c",
             "from tools_prep.fix_eye_texture import patch; patch('/work/assets/smplx_texture_m_alb_eyefix.png')"])
    tex_arg = ",".join("/work/assets/smplx_texture_{}_alb_eyefix.png".format(c["tex"]) for c in CAST)
    run(["docker", "run", "--rm", "--gpus", "device=7", "-e", "NVIDIA_DRIVER_CAPABILITIES=all",
         "-v", f"{SAMPL}:/work", "-v", f"{BOT}:/lw", "-w", "/work", "lifeworld-pyrender", "bash", "-lc",
         "python3 /lw/render/render_smplx.py --clip /work/output/scene/clip.npz "
         "--out-dir /work/output/scene/frames --uv /work/assets/smplx_uv_2023.npz "
         f"--textures {tex_arg} --framing full --rot-x 0 --res-x 1280 --res-y 720 && "
         "ffmpeg -y -loglevel error -f concat -safe 0 -i /work/output/scene/audio.txt "
         "-c:a aac /work/output/scene/audio.aac && "
         f"ffmpeg -y -loglevel error -framerate {FPS} -i /work/output/scene/frames/frame_%04d.png "
         "-i /work/output/scene/audio.aac -c:v libx264 -pix_fmt yuv420p -crf 18 -shortest "
         "/work/output/scene/scene_cine.mp4"])

    out = f"{SAMPL}/output/scene/scene_cine.mp4"
    print(f"\nSCENE_CINE_OK -> {out}")
    subprocess.run(["cp", out, f"{BOT}/output/scene_cine.mp4"])


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print("CMD FAIL:", " ".join(cmd[:6]), "...\n", p.stdout[-1500:], p.stderr[-1500:])
        sys.exit(1)


if __name__ == "__main__":
    main()
