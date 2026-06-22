#!/usr/bin/env python3
"""Data-driven lip-sync calibration. Generate many varied TTS lines -> A2F -> phoneme-align,
then POOL all frames to find the single time-offset that best aligns expected visemes (from
the audio's phonemes) to A2F's jawOpen. Per-line lag is noisy; pooled over many lines a
systematic A2F latency (if any) emerges robustly.

  python3 render/sync_calibrate.py            # full run (~5 min)
  python3 render/sync_calibrate.py --reuse    # skip TTS/A2F, just re-analyze
"""
import base64
import json
import os
import subprocess
import sys
import urllib.request

SAMPL = "/srv/nvme-data/containers/projects/sampl"
BOT = "/srv/nvme-data/containers/projects/bot"
A2F = "/mnt/24tb/a2f"
HIGGS = "http://127.0.0.1:8055"
CALIB = f"{A2F}/calib"
REUSE = "--reuse" in sys.argv

# phonetically varied lines (open AH, round OO, wide EE, bilabial M/B/P, fricatives) x 2 voices
SENT = [
    "Open the door and look outside.", "We bought two blue balloons.",
    "She sees the green trees easily.", "My mama makes maple pie.",
    "How now brown cow.", "The quick fox jumps over.",
    "Please pour more warm water.", "I really need a meeting today.",
    "Five funny vampires flyffff.", "Who knew you used your new shoe.",
    "Bob probably brought the bread.", "Eat each sweet treat please.",
    "All aboard the autumn train.", "Wow, what a wonderful world.",
    "Peter piped a pretty pepper.", "Sea shells shimmer softly.",
    "Go slow over the cold road.", "Mama mia, here we go again.",
    "Try the dry rye bread.", "Loud crowds shout aloud.",
]


def tts(text, out_wav, voice):
    body = {"input": text, "response_format": "wav"}
    if voice == "male":
        ref = f"{SAMPL}/assets/voices/male_ref.wav"
        body["reference_audio"] = base64.b64encode(open(ref, "rb").read()).decode()
        body["reference_text"] = "Promised you would finish the nursery this weekend."
    req = urllib.request.Request(f"{HIGGS}/v1/audio/speech", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    open(out_wav, "wb").write(urllib.request.urlopen(req, timeout=180).read())


def main():
    os.makedirs(CALIB, exist_ok=True)
    samples = [(t, "male" if i % 2 else "female") for i, t in enumerate(SENT)]
    manifest = []
    for i, (text, voice) in enumerate(samples):
        wav = f"{CALIB}/{i:03d}.wav"
        if not REUSE or not os.path.exists(wav):
            print(f"[tts {i+1}/{len(samples)}] {voice}: {text}")
            tts(text, wav, voice)
        manifest.append({"wav": f"/a2f/calib/{i:03d}.wav", "out": f"/a2f/calib/{i:03d}.arkit.json",
                         "identity": "James" if voice == "male" else "Claire"})
    json.dump(manifest, open(f"{CALIB}/manifest.json", "w"))
    json.dump({f"{i:03d}": t for i, (t, _) in enumerate(samples)}, open(f"{CALIB}/transcripts.json", "w"))

    if not REUSE:
        print("[A2F] batch lip-sync...")
        subprocess.run(["docker", "run", "--rm", "--gpus", "device=1",
                        "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
                        "-v", f"{A2F}:/a2f", "-v", f"{BOT}:/lw", "-w", "/lw", "lifeworld-a2f",
                        "python3", "render/a2f_lipsync.py", "--a2f", "/a2f",
                        "--manifest", "/a2f/calib/manifest.json"], check=True)

    print("[analyze] phoneme-align + pool...")
    subprocess.run(["docker", "run", "--rm", "-v", f"{SAMPL}:/work",
                    "-v", f"{SAMPL}/tools/TalkSHOW:/ts", "-v", f"{BOT}:/lw", "-v", f"{A2F}:/a2f",
                    "-e", "PYTHONPATH=/ts/pydeps", "-e", "HF_HOME=/ts/hf-cache",
                    "-e", "TORCH_HOME=/a2f/torch", "sampl:dev",
                    "bash", "-lc", "$SAMPL_VENV/bin/python /lw/render/calib_analyze.py --dir /a2f/calib"],
                   check=True)


if __name__ == "__main__":
    main()
