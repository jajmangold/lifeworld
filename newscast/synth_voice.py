"""Synthesize an arbitrary line/script with the resident Qwen3-TTS (/mono) using ANY reference voice
(--ref = a wav path inside the qwen3dia container's /work). Reuses the deterministic TTS normalizer and
the <=28-word turn split (avoids runaway). Used for the anchor and for distinct reporter/correspondent
voices.  python3 newscast/synth_voice.py --text "..." [--ref /work/reporter_ref.wav] [--seed 11] --out out.wav
"""
import sys, os, re, io, wave, json, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize_tts import normalize

def arg(f, d=None): return sys.argv[sys.argv.index(f)+1] if f in sys.argv else d
raw = arg("--text") or (open(arg("--textfile")).read() if arg("--textfile") else None)
if raw is None: sys.exit("need --text or --textfile")
REF = arg("--ref", "/work/ref1.wav"); SEED = int(arg("--seed", "42")); OUT = arg("--out", "output/voice.wav")

text = normalize(raw)
sents = re.split(r'(?<=[.!?])\s+', text.strip())
turns, cur, n = [], [], 0
for s in sents:
    cur.append(s); n += len(s.split())
    if n >= 28: turns.append(" ".join(cur)); cur = []; n = 0
if cur: turns.append(" ".join(cur))

allpcm, sr = [], None
for i, t in enumerate(turns):
    body = json.dumps({"text": t, "ref1": REF, "seed": SEED}).encode()
    req = urllib.request.Request("http://localhost:8064/mono", data=body, headers={"Content-Type": "application/json"})
    wav = urllib.request.urlopen(req, timeout=600).read()
    w = wave.open(io.BytesIO(wav)); sr = w.getframerate(); allpcm.append(w.readframes(w.getnframes()))
    allpcm.append(b'\x00\x00' * int(sr*0.35))
    print(f"  turn {i+1}/{len(turns)} {w.getnframes()/sr:.1f}s")
out = wave.open(OUT, "wb"); out.setnchannels(1); out.setsampwidth(2); out.setframerate(sr)
out.writeframes(b''.join(allpcm)); out.close()
print("WROTE", OUT, "sr", sr)
