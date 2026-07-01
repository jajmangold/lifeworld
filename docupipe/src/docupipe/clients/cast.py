"""Cast of fabricated 'researchers' for talking-head interview cutaways. Each has a name, title,
a Z-Image photoreal headshot, and a distinct TTS voice ref. Portraits generated once, cached."""
import base64
import json
import os
import urllib.request
from .. import config, cache

BOT_OUT = "/srv/nvme-data/containers/live/studio/output"

# voice refs are paths inside the qwen3dia container (/work/*). ref1=deep male narrator (reserved).
# A small ENSEMBLE of experts with DISTINCT lenses, so interview cutaways add perspective/friction
# rather than filler. The writers' room picks the speaker whose expertise fits each analysis beat.
RESEARCHERS = [
    {"id": "hale", "name": "Dr. Eleanor Hale", "title": "Historian",
     "voice_ref": "/work/ref2.wav", "lens": "social/period history — the world and its people",
     "look": "a woman historian in her 50s, warm intelligent expression, shoulder-length auburn hair"},
    {"id": "vance", "name": "Prof. Marcus Vance", "title": "Criminologist",
     "voice_ref": "/work/ref_res2.wav", "lens": "criminal behavior, method, motive and pattern",
     "look": "a man criminologist in his 60s, grey beard, glasses, tweed jacket, thoughtful"},
    {"id": "reyes", "name": "Detective (Ret.) Sofia Reyes", "title": "Cold-Case Investigator",
     "voice_ref": "/work/ref_res3.wav", "lens": "investigation, evidence, what police got right/wrong",
     "look": "a Latina woman in her 50s, short dark hair, composed, dark blazer, direct gaze"},
    {"id": "okafor", "name": "James Okafor", "title": "Investigative Journalist",
     "voice_ref": "/work/ref_res4.wav", "lens": "the human cost, the overlooked, the injustice",
     "look": "a Black man in his 40s, close-cropped hair, open collar, earnest expression"},
]
BY_ID = {r["id"]: r for r in RESEARCHERS}


def roster_brief() -> str:
    """One-line-per-expert menu the writers' room uses to cast interview beats by lens."""
    return "\n".join(f"  - id='{r['id']}': {r['name']}, {r['title']} — {r['lens']}" for r in RESEARCHERS)

PORTRAIT_STYLE = ("professional documentary interview headshot photograph, soft studio key light, "
                  "shallow depth of field, blurred bookshelf background, seated, looking slightly "
                  "off-camera, photorealistic, 50mm, natural skin, no text, no watermark")
PORTRAIT_NEG = "cartoon, illustration, painting, cgi, deformed, extra fingers, text, watermark"


def portrait(researcher, seed=42) -> str:
    """Generate/caches a photoreal headshot; returns a path staged in bot/output (for --face)."""
    r = researcher
    ck = cache.path("portrait", cache.key("por", r["id"], r["look"], seed), "png")
    if not cache.have(ck):
        body = {"prompt": f"{r['look']}, {PORTRAIT_STYLE}", "negative_prompt": PORTRAIT_NEG,
                "steps": 8, "cfg_scale": 1.0, "width": 1216, "height": 832,
                "sampler_name": "Euler", "seed": seed}
        req = urllib.request.Request(config.ZIMAGE_URL.rstrip("/") + "/sdapi/v1/txt2img",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=240) as resp:
            cache.atomic_write_bytes(ck, base64.b64decode(json.load(resp)["images"][0]))
    # stage a copy into bot/output where make_anchor's swap-server can read it (/o/<name>)
    staged = os.path.join(BOT_OUT, f"researcher_{r['id']}.png")
    if not cache.have(staged) or os.path.getsize(staged) != os.path.getsize(ck):
        cache.atomic_write_bytes(staged, open(ck, "rb").read())
    return staged
