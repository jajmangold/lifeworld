"""Rights-gated public-domain image fetch. Phase 1: Wikimedia Commons + Library of Congress.
Every returned asset carries license + attribution; nothing without a resolved rights flag."""
import json
import os
import re
import urllib.parse
import urllib.request
from .. import config, cache

UA = {"User-Agent": config.HTTP_UA}  # Commons 403s without a descriptive UA


def _get_json(url):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def _download(url, dest):
    if cache.have(dest):
        return dest
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=60) as r:
        cache.atomic_write_bytes(dest, r.read())
    return dest


# ── Wikimedia Commons ────────────────────────────────────────────────
_OK_LIC = re.compile(r"(public domain|pd-|cc0|cc-by|cc by|attribution)", re.I)
_BAD_LIC = re.compile(r"(non.?commercial|cc-by-nc|no.?deriv|fair use|copyright)", re.I)


def commons_search(query, n=8):
    base = "https://commons.wikimedia.org/w/api.php?"
    q = base + urllib.parse.urlencode({
        "action": "query", "format": "json", "generator": "search",
        "gsrsearch": f"filetype:bitmap {query}", "gsrnamespace": 6, "gsrlimit": n,
        "prop": "imageinfo", "iiprop": "url|extmetadata|mime|size",
        "iiurlwidth": 1600})
    data = _get_json(q)
    out = []
    for pg in (data.get("query", {}).get("pages", {}) or {}).values():
        ii = (pg.get("imageinfo") or [{}])[0]
        if not ii or "image" not in ii.get("mime", ""):
            continue
        em = ii.get("extmetadata", {})
        lic = (em.get("LicenseShortName", {}) or {}).get("value", "") or \
              (em.get("License", {}) or {}).get("value", "")
        artist = re.sub(r"<[^>]+>", "", (em.get("Artist", {}) or {}).get("value", "")).strip()
        rights_ok = bool(_OK_LIC.search(lic)) and not _BAD_LIC.search(lic)
        out.append({
            "source": "wikimedia_commons", "title": pg.get("title", ""),
            "page_url": ii.get("descriptionurl", ""),
            "image_url": ii.get("thumburl") or ii.get("url"),
            "full_url": ii.get("url"),
            "license": lic or "unknown", "attribution": artist or "Wikimedia Commons",
            "rights_ok": rights_ok, "w": ii.get("width"), "h": ii.get("height")})
    return out


# ── Library of Congress (free-to-use sets; gate on rights_advisory) ──
def loc_search(query, n=8):
    url = "https://www.loc.gov/photos/?" + urllib.parse.urlencode(
        {"q": query, "fo": "json", "c": n, "fa": "access-restricted:false"})
    try:
        data = _get_json(url)
    except Exception:
        return []
    out = []
    for it in (data.get("results", []) or [])[:n]:
        img = it.get("image_url") or []
        full = ("https:" + img[-1]) if img and img[-1].startswith("//") else (img[-1] if img else "")
        if not full:
            continue
        out.append({
            "source": "library_of_congress", "title": it.get("title", ""),
            "page_url": it.get("id", ""), "image_url": full, "full_url": full,
            "license": "LoC — no known restrictions", "attribution": "Library of Congress",
            "rights_ok": True})
    return out


_STOP = {"the", "a", "an", "of", "in", "on", "at", "and", "scene", "photo", "image",
         "1910s", "1900s", "night", "street", "front", "page", "cover"}


def _keywords(q):
    return {w for w in re.findall(r"[a-z]{4,}", q.lower()) if w not in _STOP}


def find_images(query, n=6, broaden=None):
    """Rights-OK candidates across archives. Tries the specific query, then broader
    fallbacks; LoC results are kept only if their title shares a keyword (relevance guard)."""
    cands, words = [], query.split()
    attempts = [query]
    if len(words) > 4:
        attempts.append(" ".join(words[:4]))     # broaden: first 4 words
    if broaden:
        attempts.append(broaden)                  # final fallback (e.g. the topic)
    kw = _keywords(query) | (_keywords(broaden) if broaden else set())
    for q in attempts:
        try:
            cands += [c for c in commons_search(q, n) if c["rights_ok"]]
        except Exception:
            pass
        if cands:
            break
    if not cands:                                 # LoC fallback, relevance-guarded
        try:
            for c in loc_search(query, n):
                if not kw or _keywords(c["title"]) & kw:
                    cands.append(c)
        except Exception:
            pass
    # dedup by image url, keep order
    seen, out = set(), []
    for c in cands:
        u = c.get("full_url") or c.get("image_url")
        if u and u not in seen:
            seen.add(u)
            out.append(c)
    return out[:n]


def wikipedia_images(topic, n=15):
    """The topic's Wikipedia article's OWN images — hand-curated, captioned, usually
    period-correct. High-precision source. Rights-gated via Commons imageinfo."""
    base = "https://en.wikipedia.org/w/api.php?"
    try:
        pg = _get_json(base + urllib.parse.urlencode({
            "action": "query", "format": "json", "titles": topic,
            "generator": "images", "gimlimit": n,
            "prop": "imageinfo", "iiprop": "url|extmetadata|mime|size", "iiurlwidth": 1600}))
    except Exception:
        return []
    out = []
    for p in (pg.get("query", {}).get("pages", {}) or {}).values():
        ii = (p.get("imageinfo") or [{}])[0]
        mime = ii.get("mime", "")
        if not ii or "image" not in mime or "svg" in mime:
            continue
        em = ii.get("extmetadata", {})
        lic = (em.get("LicenseShortName", {}) or {}).get("value", "")
        artist = re.sub(r"<[^>]+>", "", (em.get("Artist", {}) or {}).get("value", "")).strip()
        if not (_OK_LIC.search(lic) and not _BAD_LIC.search(lic)):
            continue
        out.append({"source": "wikipedia", "title": p.get("title", ""),
                    "page_url": ii.get("descriptionurl", ""),
                    "image_url": ii.get("thumburl") or ii.get("url"), "full_url": ii.get("url"),
                    "license": lic or "PD", "attribution": artist or "Wikimedia Commons",
                    "rights_ok": True, "w": ii.get("width"), "h": ii.get("height")})
    return out


def saturation(path) -> float:
    """Mean color saturation 0..1 via a tiny ffmpeg RGB dump. Period B&W/sepia -> low;
    modern color -> high. A free era/archival-ness signal."""
    import subprocess
    import numpy as np
    try:
        r = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", "scale=24:24",
                            "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
                           capture_output=True, timeout=30)
        a = np.frombuffer(r.stdout[:24 * 24 * 3], dtype=np.uint8).astype(np.float32)
        if a.size < 24 * 24 * 3:
            return 0.5
        a = a.reshape(-1, 3)
        mx, mn = a.max(1), a.min(1)
        return float(np.mean(np.where(mx > 0, (mx - mn) / mx, 0.0)))
    except Exception:
        return 0.5


_YEAR = re.compile(r"\b(18\d\d|19[0-2]\d)\b")          # <=1929 -> period
_NEWS = re.compile(r"(newspaper|times-picayune|item|headline|front page|letter|clipping)", re.I)
_MAP = re.compile(r"\bmap\b", re.I)


def shot_type_of(cand):
    t = (cand.get("title", "") + " " + cand.get("image_query", "")).lower()
    if _MAP.search(t):
        return "map"
    if _NEWS.search(t):
        return "newspaper"
    return "photo"


def archival_score(cand, sat):
    """Higher = more archival/period. Used to rank the pool (lower saturation wins)."""
    s = (1.0 - sat) * 1.0                                   # B&W/sepia preferred
    if _YEAR.search(cand.get("title", "")):
        s += 0.6                                            # year <=1929 in title
    w, h = cand.get("w") or 0, cand.get("h") or 0
    if min(w, h) >= 600:
        s += 0.2
    if cand.get("source") in ("wikipedia", "library_of_congress"):
        s += 0.25                                           # curated / archival sources
    return s


def download(asset, slug):
    """Download an asset's image into assets/ ; return local path or None."""
    url = asset.get("full_url") or asset.get("image_url")
    if not url:
        return None
    ext = os.path.splitext(urllib.parse.urlparse(url).path)[1].lower() or ".jpg"
    if ext not in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".gif"):
        ext = ".jpg"
    dest = os.path.join(config.ASSETS_DIR, cache.key("img", url) + ext)
    try:
        return _download(url, dest)
    except Exception:
        return None
