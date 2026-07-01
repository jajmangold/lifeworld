"""News research from Wikipedia's Portal:Current events — the structured, daily-curated, sourced feed the
NNS writers' room reads. Parses a day's subpage into stories {category, headline, summary, links, sources}
and pulls Commons lead images (with license + attribution) for OTS/b-roll. Attribution flows into the
provenance/cue_sheet like docupipe. Pure stdlib (urllib) — runs anywhere.
  python3 wiki_news.py [YYYY-MM-DD] [--images]           -> prints JSON of the day's stories
  from wiki_news import fetch_day, lead_image
"""
import sys, re, json, urllib.request, urllib.parse, datetime

UA = {"User-Agent": "NNS-newsbot/1.0 (newscast research; contact ops)"}
API = "https://en.wikipedia.org/w/api.php"
MONTHS = ["January","February","March","April","May","June","July","August","September","October","November","December"]

def _get(params):
    params = {**params, "format": "json", "formatversion": "2"}
    req = urllib.request.Request(API + "?" + urllib.parse.urlencode(params), headers=UA)
    return json.load(urllib.request.urlopen(req, timeout=25))

def _clean(t):
    t = re.sub(r"\[https?://[^\s\]]+\s*\([^)]*\)\]", "", t)      # drop [url (Source)] citations
    t = re.sub(r"\[\[[^\]|]*\|([^\]]+)\]\]", r"\1", t)           # [[a|b]] -> b
    t = re.sub(r"\[\[([^\]]+)\]\]", r"\1", t)                    # [[a]]   -> a
    t = t.replace("'''", "").replace("''", "")
    return re.sub(r"\s+", " ", t).strip()

def _links(t):   return re.findall(r"\[\[([^\]|#]+)", t)                    # wikilinked article titles
def _sources(t): return [{"name": n.strip().replace("''", ""), "url": u}
                         for u, n in re.findall(r"\[(https?://[^\s\]]+)\s*\(([^)]*)\)\]", t)]

def fetch_day(date=None):
    """date: 'YYYY-MM-DD' or None=today. -> list of story dicts."""
    d = datetime.date.fromisoformat(date) if date else datetime.date.today()
    page = f"Portal:Current_events/{d.year}_{MONTHS[d.month-1]}_{d.day}"
    wt = _get({"action": "parse", "page": page, "prop": "wikitext"}).get("parse", {}).get("wikitext", "")
    m = re.search(r"content=(.*)", wt, re.S)
    body = m.group(1) if m else wt
    stories, category, ancestors = [], None, {}       # ancestors: depth -> topic text
    for line in body.splitlines():
        line = line.strip()
        cat = re.match(r"^'''(.+?)'''$", line)
        if cat:
            category = cat.group(1); ancestors = {}; continue
        li = re.match(r"^(\*+)\s*(.*)$", line)
        if not li: continue
        depth, text = len(li.group(1)), li.group(2)
        ancestors[depth] = _clean(text) or ancestors.get(depth, "")
        srcs = _sources(text)
        if srcs or (depth >= 3 and len(_clean(text)) > 40):    # a leaf news item
            headline = next((ancestors[k] for k in sorted(ancestors) if k < depth and ancestors[k]), "")
            stories.append({"category": category, "headline": headline or _clean(text)[:80],
                            "summary": _clean(text), "links": _links(text), "sources": srcs,
                            "date": d.isoformat()})
    return [s for s in stories if s["summary"]]

def lead_image(title):
    """Lead image for a wikilinked article, with Commons license + attribution (or None)."""
    q = _get({"action": "query", "titles": title, "prop": "pageimages",
              "piprop": "original|thumbnail", "pithumbsize": "1280", "redirects": "1"})
    pg = (q.get("query", {}).get("pages") or [{}])[0]
    img = pg.get("original") or pg.get("thumbnail")
    if not img: return None
    # filename: prefer pageimage, else derive from the upload URL (…/commons/x/xx/<File>.jpg)
    fname = pg.get("pageimage") or urllib.parse.unquote(img["source"].split("/")[-1])
    fname = re.sub(r"^\d+px-", "", fname)                       # strip thumb prefix if present
    if not fname: return None
    meta = _get({"action": "query", "titles": "File:" + fname, "prop": "imageinfo",
                 "iiprop": "extmetadata|url"})
    ii = ((meta.get("query", {}).get("pages") or [{}])[0].get("imageinfo") or [{}])[0]
    ex = ii.get("extmetadata", {}) or {}
    def g(k): return re.sub("<[^>]+>", "", (ex.get(k, {}) or {}).get("value", "") or "").strip()
    return {"url": img.get("source"), "width": img.get("width"), "file": fname,
            "license": g("LicenseShortName"), "artist": g("Artist"),
            "credit": g("Credit"), "descriptionurl": ii.get("descriptionurl"),
            "rights_ok": "fair use" not in g("UsageTerms").lower()}

if __name__ == "__main__":
    date = next((a for a in sys.argv[1:] if re.match(r"\d{4}-\d\d-\d\d", a)), None)
    stories = fetch_day(date)
    if "--images" in sys.argv:
        WEAK = re.compile(r"^(Flag of|Coat of arms|President of|Prime Minister of|Ministry|Ambassador)", re.I)
        for s in stories:
            # prefer the specific story topic (headline) + non-generic links; flags/offices are weak b-roll
            cands = [s["headline"]] + [l for l in s["links"] if not WEAK.match(l)]
            for t in list(dict.fromkeys(cands))[:4]:
                try:
                    im = lead_image(t)
                except Exception:
                    im = None
                if im and im.get("url") and im.get("rights_ok"): s["image"] = im; break
    print(json.dumps(stories, indent=2, ensure_ascii=False))
    print(f"\n{len(stories)} stories", file=sys.stderr)
