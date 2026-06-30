"""Synthesize the anchor read from output/news_package.json with the resident Qwen3-TTS (/mono) and
master it. Picks the TTS-ready 'spoken_script' (DeepSeek-normalized) if present, else the plain
'script', and ALWAYS runs the deterministic normalize_tts backstop so digits/acronyms/units are never
spoken wrong by the no-frontend base clone. Splits into <=~28-word turns to avoid runaway.
  python3 newscast/synth_anchor.py [--seed N]   -> output/anchor_vo_raw.wav (+ master step in caller)
"""
import json, re, io, wave, urllib.request, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from normalize_tts import normalize

pkg = json.load(open("output/news_package.json"))
raw = pkg.get("spoken_script") or pkg["script"]
text = normalize(raw)                      # backstop: catches anything DeepSeek left as digits/acronyms
print("SPOKEN:", text[:200])

sents = re.split(r'(?<=[.!?])\s+', text.strip())
turns, cur, n = [], [], 0
for s in sents:
    cur.append(s); n += len(s.split())
    if n >= 28:
        turns.append(" ".join(cur)); cur = []; n = 0
if cur: turns.append(" ".join(cur))

seed = int(sys.argv[sys.argv.index("--seed")+1]) if "--seed" in sys.argv else 42
allpcm, sr = [], None
for i, t in enumerate(turns):
    body = json.dumps({"text": t, "seed": seed}).encode()
    req = urllib.request.Request("http://localhost:8064/mono", data=body, headers={"Content-Type": "application/json"})
    wav = urllib.request.urlopen(req, timeout=600).read()
    w = wave.open(io.BytesIO(wav)); sr = w.getframerate(); allpcm.append(w.readframes(w.getnframes()))
    allpcm.append(b'\x00\x00' * int(sr*0.35))
    print(f"  turn {i+1}/{len(turns)} {w.getnframes()/sr:.1f}s")
out = wave.open("output/anchor_vo_raw.wav", "wb")
out.setnchannels(1); out.setsampwidth(2); out.setframerate(sr); out.writeframes(b''.join(allpcm)); out.close()
print("WROTE output/anchor_vo_raw.wav sr", sr)
