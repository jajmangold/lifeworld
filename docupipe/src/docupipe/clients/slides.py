"""Presenton slide decks for explainer beats (timelines, suspect line-ups, data). Generates a
PDF deck via the local Presenton service, renders each page to a PNG. Resilient + cached."""
import base64
import json
import os
import subprocess
import urllib.request
from .. import config, cache

PRESENTON_URL = os.environ.get("DOCUPIPE_PRESENTON_URL", "http://localhost:5001")
PRESENTON_AUTH = os.environ.get("DOCUPIPE_PRESENTON_AUTH", "admin:docupipe1")
# Presenton writes to /app_data (container) == presenton_data (host)
DATA_HOST = os.path.join(config.ROOT, "presenton_data")


def available() -> bool:
    try:
        req = urllib.request.Request(PRESENTON_URL.rstrip("/") + "/api/v1/auth/status")
        req.add_header("Authorization", "Basic " + base64.b64encode(PRESENTON_AUTH.encode()).decode())
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def deck_pages(content: str, n_slides=4, tone="educational") -> list:
    """Generate a deck and return a list of PNG page paths (host), or [] on failure."""
    ck = cache.path("deck", cache.key("deck", content, n_slides), "json")
    if cache.have(ck):
        pages = cache.load_json(ck)
        if all(cache.have(p) for p in pages):
            return pages
    try:
        body = json.dumps({"content": content, "n_slides": n_slides, "language": "English",
                           "export_as": "pdf", "tone": tone, "verbosity": "concise",
                           "include_title_slide": True}).encode()
        req = urllib.request.Request(PRESENTON_URL.rstrip("/") + "/api/v1/ppt/presentation/generate",
                                     data=body, headers={"Content-Type": "application/json"})
        req.add_header("Authorization", "Basic " + base64.b64encode(PRESENTON_AUTH.encode()).decode())
        with urllib.request.urlopen(req, timeout=600) as r:
            out = json.load(r)
        pdf_host = out["path"].replace("/app_data", DATA_HOST)
        if not cache.have(pdf_host):
            return []
        # render each PDF page to a PNG
        stem = cache.path("deck", cache.key("pg", pdf_host), "")
        os.makedirs(os.path.dirname(stem), exist_ok=True)
        subprocess.run(["pdftoppm", "-png", "-r", "150", pdf_host, stem], timeout=120)
        pages = sorted(p for p in
                       (os.path.join(os.path.dirname(stem), f) for f in os.listdir(os.path.dirname(stem)))
                       if p.startswith(stem) and p.endswith(".png"))
        cache.save_json(ck, pages)
        return pages
    except Exception:
        return []
