"""Premium superscale via the resident FlashVSR server on rtx0 (warm pipeline, HTTP).
Diffusion VSR recovers detail on degraded 'hero' stills; ESRGAN stays the fast default for the
rest (it's better for clean stills). Resilient: returns the original on any failure."""
import os
import urllib.request
from .. import config, cache


def available() -> bool:
    """True if the FlashVSR endpoint is up — works for a single worker ('ok') or the
    queue coordinator (a readiness map with at least one true)."""
    try:
        with urllib.request.urlopen(config.FLASHVSR_URL.rstrip("/") + "/health", timeout=4) as r:
            body = r.read(2000)
        return b'"ok"' in body or b'true' in body
    except Exception:
        return False


def flashvsr(image: str, scale=2, ultra=False) -> str:
    """4x/2x a still through the resident FlashVSR server. Returns a cached PNG, or the input on failure."""
    out = cache.path("fvsr", cache.key("fv", image, os.path.getsize(image), scale, ultra), "png")
    if cache.have(out):
        return out
    ext = (os.path.splitext(image)[1].lstrip(".") or "jpg").lower()
    url = (config.FLASHVSR_URL.rstrip("/")
           + f"/upscale?mode=image&scale={scale}&ultra={'1' if ultra else '0'}&ext={ext}")
    try:
        req = urllib.request.Request(url, data=open(image, "rb").read(),
                                     headers={"Content-Type": "application/octet-stream"})
        with urllib.request.urlopen(req, timeout=600) as r:
            data = r.read()
        if not data.startswith(b"\x89PNG"):
            return image
        cache.atomic_write_bytes(out, data)
        return out
    except Exception:
        return image
