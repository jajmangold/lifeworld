"""Geographic helpers for docgfx maps. Fetches REAL street geometry from OpenStreetMap (Overpass)
for a map's bounding box so the base reads as an actual city plan, not a stylized grid. Cached."""
import json
import urllib.parse
import urllib.request
from .. import config, cache

OVERPASS = "https://overpass-api.de/api/interpreter"
# highway classes -> weight tier (major roads thicker/darker, residential thin)
_MAJOR = {"motorway", "trunk", "primary", "secondary"}
_MID = {"tertiary", "unclassified", "residential"}


def bbox_of(points, pad=0.45):
    lats = [p["lat"] for p in points]
    lngs = [p["lng"] for p in points]
    dlat = max(max(lats) - min(lats), 0.01) * pad
    dlng = max(max(lngs) - min(lngs), 0.01) * pad
    return (min(lats) - dlat, min(lngs) - dlng, max(lats) + dlat, max(lngs) + dlng)  # s,w,n,e


def streets(points, max_ways=2600):
    """Return [{coords:[[lng,lat],...], tier:0|1|2, water:bool}, ...] for the points' bbox."""
    s, w, n, e = bbox_of(points)
    ck = cache.path("osm", cache.key("osm", round(s, 4), round(w, 4), round(n, 4), round(e, 4)), "json")
    if cache.have(ck):
        return cache.load_json(ck)
    q = (f"[out:json][timeout:40];(way['highway']({s},{w},{n},{e});"
         f"way['waterway'='river']({s},{w},{n},{e});way['natural'='water']({s},{w},{n},{e}););out geom;")
    try:
        req = urllib.request.Request(OVERPASS, data=urllib.parse.urlencode({"data": q}).encode(),
                                     headers={"User-Agent": config.HTTP_UA})
        with urllib.request.urlopen(req, timeout=90) as r:
            data = json.load(r)
    except Exception:
        return []
    out = []
    for el in data.get("elements", []):
        geom = el.get("geometry")
        if not geom or len(geom) < 2:
            continue
        coords = [[g["lon"], g["lat"]] for g in geom]   # Overpass uses lon
        tags = el.get("tags", {})
        if tags.get("waterway") == "river" or tags.get("natural") == "water":
            out.append({"coords": coords, "tier": 0, "water": True})
            continue
        hw = tags.get("highway", "")
        tier = 0 if hw in _MAJOR else (1 if hw in _MID else 2)
        out.append({"coords": coords, "tier": tier, "water": False})
        if len(out) >= 2600:
            break
    # major/water first (drawn under), so they read as the spine
    out.sort(key=lambda x: (not x["water"], x["tier"]))
    cache.save_json(ck, out)
    return out
