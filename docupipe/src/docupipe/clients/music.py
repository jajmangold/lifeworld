"""Original score via the resident ACE-Step 1.5 server (--enable-api on :7861). Generates a short
INSTRUMENTAL bed matched to the episode's mood; media.add_music_and_master loops + ducks it under the
narration. Copyright-clean (fully synthesized). Cached by caption. No vocals — a documentary score."""
import json
import time
import urllib.request
from .. import config, cache

BASE = config.ACESTEP_URL.rstrip("/")


def available() -> bool:
    try:
        urllib.request.urlopen(BASE + "/", timeout=4)
        return True
    except Exception:
        return False


def _post(path, body, timeout=120):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def score(tags: str, out_wav: str, *, duration=60.0, bpm=72, key="", seed=7) -> str:
    """Generate an instrumental cue for the tag-set `tags` and write it to out_wav. "" on failure.
    ACE-Step craft (per the docs): the caption drives ~70% of quality — pass a SHORT, SPECIFIC
    comma-tag list (5-12 concrete tags: genre + concrete instruments + 1-2 emotion words); do NOT
    put tempo/bpm/key/duration in the caption — those go in the metadata params (bpm/keyscale).
    Instrumental = empty lyrics + a 'instrumental' tag. Cached by (tags,duration,bpm,key,seed)."""
    duration = max(10.0, min(float(duration), 90.0))     # reliable range on this Volta; loop to fill
    ck = cache.path("score", cache.key("ace", tags, duration, bpm, key, seed), "mp3")
    if cache.have(ck):
        cache.atomic_write_bytes(out_wav, open(ck, "rb").read())
        return out_wav
    payload = {
        "lyrics": "[instrumental]",
        "prompt": tags.strip().rstrip(",") + ", instrumental",
        "thinking": True, "bpm": int(bpm), "vocal_language": "en",
        "inference_steps": 8, "guidance_scale": 7.0, "audio_duration": float(duration),
        "use_random_seed": False, "seed": seed, "audio_format": "mp3",
        "model": "acestep-v15-turbo",
    }
    if key:
        payload["keyscale"] = key
    try:
        resp = _post("/release_task", payload)
        task_id = resp["data"]["task_id"]
        deadline = time.time() + 1800
        while time.time() < deadline:
            q = _post("/query_result", {"task_id_list": [task_id]})
            item = (q.get("data") or [{}])[0]
            # status is an int: 1 = done. result is a JSON-encoded string [{file,url,...}].
            if item.get("status") == 1 and item.get("result"):
                res = json.loads(item["result"])
                r0 = res[0] if isinstance(res, list) else res
                url = r0.get("url", "")               # "/v1/audio?path=..." served by the app
                data = urllib.request.urlopen(BASE + url, timeout=180).read()
                cache.atomic_write_bytes(ck, data)
                cache.atomic_write_bytes(out_wav, data)
                return out_wav
            if item.get("status") in (2, -1, "failed", "error"):
                print(f"[music] ACE-Step task failed: {str(item)[:200]}", flush=True)
                return ""
            time.sleep(5)
    except Exception as e:  # noqa
        print(f"[music] ACE-Step error: {e}", flush=True)
        return ""
    return ""
