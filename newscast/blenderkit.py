"""BlenderKit (blendkit.com) headless client — search + download rigged royalty-free character/hair/
wardrobe assets to replace the dumped viverse avatar source. Search is public; DOWNLOAD needs the
account API key (blenderkit.com -> profile -> API keys, or the addon token). Set BLENDERKIT_API_KEY or
pass --key.  Assets land in assets/blenderkit/<slug>/.
  python3 blenderkit.py search "business woman rigged" [--type model]
  python3 blenderkit.py get <assetBaseId> [--key KEY] [--gltf]     -> downloads the asset file
"""
import os, sys, json, time, uuid, urllib.request, urllib.parse, urllib.error

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))        # live/studio
DEST = os.path.join(ROOT, "assets", "blenderkit")
API = "https://www.blendkit.com/api/v1"
COOKIE_FILE = os.path.expanduser("~/.config/studio/blendkit.cookies")     # session (never committed)
UA = {"User-Agent": "BlenderKit-Client/1.0", "Accept": "application/json"}   # WAF blocks bot UAs

TOKEN_FILE = os.path.expanduser("~/.config/studio/blenderkit.token")

def _key(explicit=None):
    k = (explicit or os.environ.get("BLENDERKIT_API_KEY", "")).strip()
    if k: return k
    if os.path.isfile(TOKEN_FILE):                       # OAuth access_token from blenderkit_login.py
        try: return (json.load(open(TOKEN_FILE)).get("access_token") or "").strip()
        except Exception: return ""
    return ""

def _auth(explicit_key=None):
    """Prefer a Bearer API key; else fall back to the stored browser session cookie."""
    k = _key(explicit_key)
    if k: return {"Authorization": "Bearer " + k}
    if os.path.isfile(COOKIE_FILE):
        return {"Cookie": open(COOKIE_FILE).read().strip()}
    return {}

def search(query, asset_type="model", n=12, only_rigged=False):
    url = f"{API}/search/?" + urllib.parse.urlencode(
        {"query": query, "asset_type": asset_type, "order": "-score", "page_size": str(n)})
    r = json.load(urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30))
    out = []
    for a in r.get("results", []):
        p = a.get("dictParameters", {}) or {}
        if only_rigged and not p.get("rig"):
            continue
        fmts = sorted({f.get("fileType") for f in (a.get("files") or []) if f.get("downloadUrl")})
        out.append({"id": a.get("assetBaseId"), "name": a.get("name"), "author": (a.get("author") or {}).get("firstName", ""),
                    "rig": bool(p.get("rig")), "animated": bool(p.get("animated")),
                    "license": a.get("license"), "formats": fmts})
    return out

def _asset(asset_base_id, key=None):
    url = f"{API}/search/?" + urllib.parse.urlencode({"query": f"asset_base_id:{asset_base_id}"})
    r = json.load(urllib.request.urlopen(urllib.request.Request(url, headers={**UA, **_auth(key)}), timeout=30))
    res = r.get("results", [])
    return res[0] if res else None

def download(asset_base_id, key=None, prefer_gltf=False):
    auth = _auth(key)
    if not auth:
        raise SystemExit("no auth — set BLENDERKIT_API_KEY or store the session cookie at " + COOKIE_FILE)
    a = _asset(asset_base_id, key)
    if not a: raise SystemExit(f"asset {asset_base_id} not found")
    files = [f for f in (a.get("files") or []) if f.get("downloadUrl")]
    order = (lambda f: (f.get("fileType") != "gltf",)) if prefer_gltf else (lambda f: (f.get("fileType") != "blend",))
    files.sort(key=order)
    if not files: raise SystemExit("no downloadable files")
    f = files[0]
    hdr = {**UA, **auth}
    # step 1: the download endpoint returns JSON with the real (signed S3) file path. The endpoint is
    # RATE-LIMITED (anti-bulk) -> fresh scene_uuid per attempt + backoff on 401/403.
    base = f["downloadUrl"]
    meta = None
    for attempt in range(6):
        dl = base + ("&" if "?" in base else "?") + urllib.parse.urlencode({"scene_uuid": str(uuid.uuid4())})
        try:
            meta = json.load(urllib.request.urlopen(urllib.request.Request(dl, headers=hdr), timeout=60))
            break
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 429) and attempt < 5:
                wait = 6 * (attempt + 1)
                print(f"  rate-limited ({e.code}), retry in {wait}s...", flush=True); time.sleep(wait); continue
            raise
    file_url = (meta or {}).get("filePath") or (meta or {}).get("file_path") or (meta or {}).get("url")
    if not file_url: raise SystemExit(f"no file path in download response: {list(meta or {})}")
    slug = "".join(c if c.isalnum() else "_" for c in (a.get("name") or asset_base_id))[:40]
    d = os.path.join(DEST, slug); os.makedirs(d, exist_ok=True)
    ext = f.get("fileType", "blend")
    out = os.path.join(d, f"asset.{ext}")
    # fetch the signed CDN file with a real UA (the asset CDN WAF 403s the default urllib UA)
    with urllib.request.urlopen(urllib.request.Request(file_url, headers={"User-Agent": UA["User-Agent"]}), timeout=600) as r, open(out, "wb") as w:
        while chunk := r.read(1 << 20):
            w.write(chunk)
    json.dump(a, open(os.path.join(d, "meta.json"), "w"))
    print("BLENDERKIT_OK ->", out, f"({os.path.getsize(out)//1024}KB, {ext}, rig={a.get('dictParameters',{}).get('rig')})")
    return out

if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "search":
        for r in search(sys.argv[2], only_rigged="--rigged" in sys.argv):
            print(f"  {'RIG' if r['rig'] else '   '} {'ANI' if r['animated'] else '   '} | {r['name'][:42]:42s} "
                  f"| {','.join(r['formats']) or '-':14s} | {r['license'][:12]} | {r['id']}")
    elif len(sys.argv) >= 3 and sys.argv[1] == "get":
        k = sys.argv[sys.argv.index("--key")+1] if "--key" in sys.argv else None
        download(sys.argv[2], key=k, prefer_gltf="--gltf" in sys.argv)
    else:
        print(__doc__)
