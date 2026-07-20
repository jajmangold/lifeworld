"""Generative gap-fill — Z-Image Turbo (sdapi) makes period illustrations for beats that have
no archival image. Styled as vintage charcoal/etching so they read as 'artist's depiction' and
blend with the archival grade (NOT photoreal — avoids passing synthetic images as real footage).
Resilient + cached."""
import base64
import json
import urllib.request
from .. import config, cache

STYLE = ("vintage 1910s archival illustration, charcoal and ink etching, black and white "
         "engraving, period-accurate, grainy halftone, dramatic chiaroscuro, somber, "
         "no text, no watermark, no signature")
NEG = "color, modern, contemporary, photograph, photo, text, caption, signature, watermark, blurry"


def available() -> bool:
    try:
        urllib.request.urlopen(config.ZIMAGE_URL.rstrip("/") + "/sdapi/v1/options", timeout=4)
        return True
    except Exception:
        return False


def illustration(scene: str, out_png: str, *, seed=7, steps=8, w=1216, h=832) -> str:
    """Generate a period illustration of `scene`. Returns out_png, or "" on failure."""
    ck = cache.path("illus", cache.key("ill", scene, seed, steps), "png")
    if cache.have(ck):
        return ck
    body = {"prompt": f"{scene}. {STYLE}", "negative_prompt": NEG,
            "steps": steps, "cfg_scale": 1.0, "width": w, "height": h,
            "sampler_name": "Euler", "seed": seed}
    try:
        req = urllib.request.Request(config.ZIMAGE_URL.rstrip("/") + "/sdapi/v1/txt2img",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=240) as r:
            img = base64.b64decode(json.load(r)["images"][0])
        cache.atomic_write_bytes(ck, img)
        return ck
    except Exception:
        return ""
