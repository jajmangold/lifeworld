"""Fetch real, freely-licensed images from Wikimedia for the newscast over-shoulder panels.
For each story, try candidate Wikipedia page titles -> grab the lead pageimage -> save.
Output: newscast/assets/wiki_<id>.jpg (only where a good image was found).
"""
import json, urllib.request, urllib.parse, os
BOT = "/srv/nvme-data/containers/live/studio"
UA = {"User-Agent": "newscast-demo/1.0 (research)"}
# candidate Wikipedia titles per story (first that yields an image wins)
CAND = {
    "01_venezuela": ["2026 Venezuela earthquakes", "Caracas", "La Guaira"],
    "02_iran": ["Strait of Hormuz", "Bandar Abbas"],
    "03_starmer": ["Keir Starmer"],
    "04_bolton": ["John Bolton"],
    "05_fed": ["Marriner S. Eccles Federal Reserve Board Building", "Federal Reserve"],
    "06_openai": ["OpenAI", "Integrated circuit"],
    "07_heat": ["2026 European heat waves", "Heat wave"],
    "08_worldcup": ["2026 FIFA World Cup", "MetLife Stadium"],
}
def thumb(title):
    q = urllib.parse.urlencode({"action":"query","titles":title,"prop":"pageimages",
                                "pithumbsize":"900","format":"json"})
    r = urllib.request.urlopen(urllib.request.Request(f"https://en.wikipedia.org/w/api.php?{q}", headers=UA), timeout=15)
    d = json.load(r); pages = d["query"]["pages"]
    for _, p in pages.items():
        if "thumbnail" in p: return p["thumbnail"]["source"]
    return None
for sid, titles in CAND.items():
    out = f"{BOT}/newscast/assets/wiki_{sid}.jpg"
    got = None
    for t in titles:
        try:
            u = thumb(t)
            if u:
                urllib.request.urlretrieve(*( [urllib.request.Request(u, headers=UA)] )) if False else None
                req = urllib.request.Request(u, headers=UA)
                data = urllib.request.urlopen(req, timeout=20).read()
                open(out, "wb").write(data); got = (t, u); break
        except Exception as e:
            print("  miss", sid, t, str(e)[:60])
    print(("OK  " if got else "NONE") + f" {sid}" + (f" <- {got[0]}" if got else ""))
