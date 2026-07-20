"""Grounded research: SearXNG search -> fetch top pages -> strip text. Returns sources[]."""
import json
import re
import urllib.parse
import urllib.request
from .. import config, cache

_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


def search(query: str, n=6):
    ck = cache.key("searx", query, n)
    cp = cache.path("searx", ck, "json")
    if cache.have(cp):
        return cache.load_json(cp)
    url = config.SEARXNG_URL.rstrip("/") + "/search?" + urllib.parse.urlencode(
        {"q": query, "format": "json"})
    req = urllib.request.Request(url, headers={"User-Agent": config.HTTP_UA})
    with urllib.request.urlopen(req, timeout=30) as r:
        data = json.load(r)
    out = [{"title": x.get("title", ""), "url": x.get("url", ""),
            "snippet": x.get("content", "")} for x in data.get("results", [])[:n]]
    cache.save_json(cp, out)
    return out


def fetch_text(url: str, max_chars=8000):
    ck = cache.key("page", url)
    cp = cache.path("page", ck, "txt")
    if cache.have(cp):
        return open(cp, encoding="utf-8").read()
    try:
        req = urllib.request.Request(url, headers={"User-Agent": config.HTTP_UA})
        with urllib.request.urlopen(req, timeout=30) as r:
            html = r.read(1_500_000).decode("utf-8", "ignore")
        html = re.sub(r"<(script|style|nav|footer|header)[^>]*>.*?</\1>", " ", html,
                      flags=re.S | re.I)
        text = _WS.sub(" ", _TAG.sub(" ", html)).strip()[:max_chars]
    except Exception as e:  # noqa
        text = f"[fetch failed: {e}]"
    cache.atomic_write_bytes(cp, text.encode())
    return text


def gather(queries, per_query=5, fetch_top=2):
    """Run several queries, fetch the top pages of each, return deduped sources with text."""
    seen, sources = set(), []
    for q in queries:
        for hit in search(q, per_query):
            u = hit["url"]
            norm = re.sub(r"#.*$", "", u)
            if not u or norm in seen:
                continue
            seen.add(norm)
            sources.append(hit)
    # fetch text for the first `fetch_top` of each query's results (cap total)
    for s in sources[: max(fetch_top * len(queries), 8)]:
        s["text"] = fetch_text(s["url"])
    return sources
