"""Vision judge — Qwen3.5-9B + mmproj (TurboQuant llama.cpp, OpenAI /v1). Scores how well a
candidate image fits a beat AND how period-appropriate it is. Resilient: returns None if the
endpoint is down, so the heuristic curator stands alone."""
import base64
import json
import mimetypes
import os
import urllib.request
from .. import config, cache


def available() -> bool:
    try:
        with urllib.request.urlopen(config.VISION_URL.rstrip("/") + "/models", timeout=4) as r:
            return b'"id"' in r.read(2000)
    except Exception:
        return False


def _data_url(path, max_side=512):
    """Downscale to a small JPEG data URL (fast image prefill; scoring needs no detail)."""
    small = cache.path("vthumb", cache.key("vt", path, os.path.getsize(path)), "jpg")
    if not cache.have(small):
        import subprocess
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path, "-frames:v", "1",
                        "-update", "1", "-vf", f"scale='min({max_side},iw)':-2", small],
                       timeout=40)
    b = open(small, "rb").read()
    mime = mimetypes.guess_type(small)[0] or "image/jpeg"
    return f"data:{mime};base64," + base64.b64encode(b).decode()


SYS = ("You are a documentary photo editor. Judge a candidate archival image for a film beat. "
       "Reply ONLY compact JSON: {\"relevance\":0-1, \"period\":0-1, \"is_photo_of_subject\":bool, "
       "\"caption\":str}. relevance=how well it depicts/evokes the beat; period=how much it looks "
       "like a genuine pre-1930 archival photo/document (1.0 = clearly old B&W/sepia; 0.0 = modern).")


def score(image_path, beat_text, topic, *, use_cache=True):
    """Return {relevance, period, is_photo_of_subject, caption} or None if unavailable."""
    ck = cache.path("vscore", cache.key("vs", image_path, beat_text, topic), "json")
    if use_cache and cache.have(ck):
        return cache.load_json(ck)
    try:
        body = {
            "model": config.VISION_MODEL, "temperature": 0.1, "max_tokens": 200,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": [
                {"type": "text", "text": f"TOPIC: {topic}\nBEAT: {beat_text}\nJudge this image:"},
                {"type": "image_url", "image_url": {"url": _data_url(image_path)}}]}],
        }
        req = urllib.request.Request(config.VISION_URL.rstrip("/") + "/chat/completions",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            txt = json.load(r)["choices"][0]["message"]["content"]
        s, e = txt.find("{"), txt.rfind("}")
        out = json.loads(txt[s:e + 1])
        if use_cache:
            cache.save_json(ck, out)
        return out
    except Exception:
        return None
