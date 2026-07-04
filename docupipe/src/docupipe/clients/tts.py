"""Narration via the qwen3-tts server (amd1 voxserver, config.TTS_URL). /mono = single narrator (deep male anchor voice)."""
import json
import urllib.request
import wave
from .. import config, cache


def narrate(text: str, out_wav: str, *, ref=None, temperature=0.85, seed=42) -> float:
    """Synthesize narration to out_wav (24k mono). Returns duration in seconds. Cached."""
    ref = ref or config.NARRATOR_REF
    if cache.have(out_wav):
        return _dur(out_wav)
    payload = {"text": text, "ref1": ref, "temperature": temperature, "seed": seed}
    req = urllib.request.Request(
        config.TTS_URL.rstrip("/") + "/mono", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        cache.atomic_write_bytes(out_wav, r.read())
    return _dur(out_wav)


def _dur(p: str) -> float:
    with wave.open(p) as w:
        return w.getnframes() / float(w.getframerate())
