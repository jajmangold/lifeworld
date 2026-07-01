"""News writers' room — the `news` show profile's writers'-room stage. Turns a DAY of Wikipedia
Portal:Current events into an ordered NNS rundown of renderable Segments, reusing docupipe's LLM client.
  research (wiki_news) -> select/order rundown (EP) -> write on-air copy per story (writer) -> Segments.
Produces a segment plan (list of Segment dicts) ready for the render dispatcher. No GPU here — CPU/LLM only.
  python3 -m studio.news_room [YYYY-MM-DD] [--n 6]      -> prints the rundown JSON
"""
import os, sys, json, argparse

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # live/studio
sys.path.insert(0, os.path.join(_ROOT, "docupipe", "src"))     # reuse docupipe's llm client + cache
sys.path.insert(0, os.path.join(_ROOT, "newscast"))            # wiki_news
from docupipe.clients import llm                                # noqa: E402
import wiki_news                                                # noqa: E402

CATALOG = os.path.join(_ROOT, "catalog")
def _cat(name): return json.load(open(os.path.join(CATALOG, name)))

# which segment kinds the news profile allows, and the talent/set each implies
NEWS_KINDS = {
    "anchor_wall":       {"talent": "graham_pierce", "set": "newsroom"},   # anchor + video wall (default)
    "vo_broll":          {"talent": "graham_pierce", "set": "broll_plate"},# anchor VO over full-screen b-roll
    "ots":               {"talent": "graham_pierce", "set": "newsroom"},   # over-the-shoulder box
    "reporter_pkg":      {"talent": "elise_warren",  "set": "flood_field"},# field correspondent standup + VO
    "title":             {"talent": None,            "set": None},         # bumper card
}

EP_SYS = ("You are the executive producer of NNS, a national TV newscast anchored by Graham Pierce. "
          "Build TODAY'S RUNDOWN from the wire stories. Lead with the highest-impact story; group by "
          "news value, not category. Assign each selected story a FORMAT: 'anchor_wall' (default, anchor "
          "at desk with a video-wall graphic), 'vo_broll' (anchor voiceover over full-screen footage, for "
          "visually-driven stories), 'ots' (over-the-shoulder box), or 'reporter_pkg' (send a field "
          "correspondent — use for 1-2 on-the-ground disaster/conflict stories max). Keep it tight.")

WRITER_SYS = ("You are a network news writer for NNS. Write BROADCAST copy for anchor Graham Pierce: "
              "active voice, present tense, plain spoken sentences a TTS voice will read cleanly (spell "
              "out nothing weird, no headlines-ese). 2-4 sentences for the anchor read.")


def research(date=None):
    return wiki_news.fetch_day(date)


def select_rundown(stories, n=6):
    """EP picks + orders n stories into a rundown with formats."""
    wire = [{"i": i, "category": s["category"], "headline": s["headline"], "summary": s["summary"][:280]}
            for i, s in enumerate(stories)]
    out = llm.chat_json("bulk", EP_SYS,
        f"Wire stories (JSON):\n{json.dumps(wire, ensure_ascii=False)}\n\n"
        f"Select the top {n} for tonight and order them. Return JSON: "
        '{"cold_open": "<one-sentence teaser of the top story>", '
        '"rundown": [{"i": <story index>, "format": "anchor_wall|vo_broll|ots|reporter_pkg", '
        '"block": "<2-4 word slug>", "why": "<why it made air>"}]}', temperature=0.3)
    return out


def write_segment(story, fmt):
    """Writer produces on-air copy + graphics + b-roll queries for one story/format."""
    reporter = fmt == "reporter_pkg"
    out = llm.chat_json("writer", WRITER_SYS,
        f"Story: {json.dumps({'headline': story['headline'], 'summary': story['summary'], 'sources':[s['name'] for s in story['sources']]}, ensure_ascii=False)}\n"
        f"Format: {fmt}. Return JSON: "
        '{"anchor_read": "<2-4 spoken sentences>", "headline": "<UPPERCASE lower-third, <=48 chars>", '
        '"kicker": "<1-2 word tag e.g. BREAKING>", "broll_queries": ["<visual search terms>", ...]'
        + (', "reporter_read": "<correspondent standup + VO, 3-5 sentences>", "location": "<CITY, ST>"' if reporter else '')
        + '}', temperature=0.5)
    return out


def build_plan(date=None, n=6):
    stories = research(date)
    if not stories:
        return {"date": date, "segments": [], "note": "no wire stories for this date"}
    rd = select_rundown(stories, n)
    segments = [{"id": 0, "order": 0, "kind": "title", "spec": {"title": "NNS", "subtitle": rd.get("cold_open", "")},
                 "talent_ref": None, "set_ref": None, "narration": ""}]
    for k, item in enumerate(rd.get("rundown", []), 1):
        st = stories[item["i"]]; fmt = item["format"] if item["format"] in NEWS_KINDS else "anchor_wall"
        copy = write_segment(st, fmt)
        bind = NEWS_KINDS[fmt]
        seg = {"id": k, "order": k, "kind": fmt, "block": item.get("block"),
               "talent_ref": bind["talent"], "set_ref": bind["set"],
               "narration": copy.get("anchor_read", ""),
               "spec": {"headline": copy.get("headline", st["headline"][:48]).upper(),
                        "kicker": copy.get("kicker", ""), "broll_queries": copy.get("broll_queries", [])},
               "sources": st["sources"], "image_query": st["headline"]}
        if st.get("image"): seg["asset"] = st["image"]
        if fmt == "reporter_pkg":
            seg["spec"]["reporter_read"] = copy.get("reporter_read", "")
            seg["spec"]["location"] = copy.get("location", "")
        segments.append(seg)
    return {"date": date or "today", "show_type": "news", "profile": "news", "brand_ref": "nns",
            "title": rd.get("cold_open", "NNS Newscast"), "segments": segments}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("date", nargs="?", default=None)
    ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    print(json.dumps(build_plan(a.date, a.n), indent=2, ensure_ascii=False))
