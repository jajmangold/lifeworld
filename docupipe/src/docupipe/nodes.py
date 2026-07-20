"""LangGraph nodes for the docuseries pipeline (Phase 1)."""
import json
import os
from langgraph.config import get_stream_writer
from langgraph.types import interrupt, Send
from . import config, provenance, cache
from .clients import (research, archives, tts, media, llm, restore, vision, generate,
                      superscale, cast, anchor, slides, docgfx, music)


def _progress(**kw):
    print(f"   · {kw.get('stage','')}: {kw.get('msg','')}"
          + (f" [seg {kw['seg']}]" if 'seg' in kw else ""), flush=True)
    try:
        get_stream_writer()(kw)
    except Exception:
        pass


# ═════════════════════════════════════════════════════════════════════
#  THE WRITERS' ROOM  —  research → deep-research (fan-out) → story bible
#  → write acts (fan-out) → assemble → showrunner edit.  A single LLM call
#  can't turn a mystery into a network-grade episode; this decomposes the
#  job the way a real documentary is actually built.  Topic-agnostic.
# ═════════════════════════════════════════════════════════════════════

# ── 1. RESEARCH (broad) — facts AND the cast + threads a story is built from
RESEARCH_SYS = (
    "You are the lead researcher for a documentary. Using ONLY the provided sources, build the "
    "raw material a writers' room needs. Prefer primary/reputable sources; flag disputes.\n"
    "Return JSON: {\n"
    " \"title\": str,\n"
    " \"overview\": str (400-600 words — the whole story, chronological, with the human stakes),\n"
    " \"key_facts\": [{\"fact\": str, \"source_url\": str}] (15-30, specific and concrete),\n"
    " \"entities\": [{\"name\": str, \"type\": \"victim\"|\"suspect\"|\"investigator\"|\"witness\""
    "|\"place\"|\"organization\"|\"other\", \"role\": str (why they matter, one line)}] "
    "(the PEOPLE and places a viewer would follow — 6-12; real named individuals first),\n"
    " \"threads\": [{\"name\": str, \"question\": str}] (the distinct narrative/investigative "
    "threads a series would weave — e.g. a theory, an injustice, a mystery within the mystery — 4-8),\n"
    " \"open_questions\": [str]\n}\n"
    "Do not invent anything absent from the sources.")


def research_node(state):
    topic = state["topic"]
    cp = cache.path("node_research", cache.key(topic, "v2"), "json")
    if cache.have(cp):
        out = cache.load_json(cp)
        _progress(stage="research", msg=f"cached: {len(out.get('entities', []))} entities, "
                  f"{len(out.get('threads', []))} threads")
        return {"research": out, "title": out.get("title", topic)}
    _progress(stage="research", msg=f"searching: {topic}")
    queries = [topic, f"{topic} what happened timeline", f"{topic} people victims suspects",
               f"{topic} investigation theories", f"{topic} history background"]
    srcs = research.gather(queries, per_query=6, fetch_top=3)
    corpus = "\n\n".join(
        f"SOURCE [{s['url']}] {s['title']}\n{s.get('text', s.get('snippet',''))[:5000]}"
        for s in srcs[:14])
    out = llm.chat_json("bulk", RESEARCH_SYS,
                        f"TOPIC: {topic}\n\nSOURCES:\n{corpus}", max_tokens=6000)
    out["sources"] = [{"title": s["title"], "url": s["url"]} for s in srcs]
    cache.save_json(cp, out)
    _progress(stage="research", msg=f"{len(out.get('key_facts', []))} facts, "
              f"{len(out.get('entities', []))} entities, {len(out.get('threads', []))} threads")
    return {"research": out, "title": out.get("title", topic)}


# ── 2. DEEP RESEARCH (fan-out) — one rich dossier per key person / thread
DOSSIER_SYS = (
    "You are a documentary researcher writing a DOSSIER on one subject for the writers' room, "
    "using ONLY the provided sources plus the story context. The goal is DEPTH and HUMANITY — the "
    "detail that makes a viewer care.\n"
    "Return JSON: {\n"
    " \"subject\": str, \"kind\": str,\n"
    " \"humanizing\": str (2-4 sentences — who this person really was: age, work, family, a telling "
    "detail; for a place/thread, its texture and significance),\n"
    " \"role_in_story\": str (how they drive the narrative),\n"
    " \"key_moments\": [{\"when\": str, \"what\": str}] (specific scene-able moments, 3-6),\n"
    " \"quotes\": [{\"text\": str, \"speaker\": str, \"source_url\": str}] (verbatim primary quotes "
    "if the sources contain them — headlines, letters, testimony; [] if none),\n"
    " \"disputes\": [str] (contested claims / competing interpretations),\n"
    " \"sources\": [str]\n}\n"
    "Never invent facts or quotes. If the sources are thin, say so honestly in the fields.")


def deep_research_fanout(state):
    r = state.get("research", {})
    ents = r.get("entities", []) or []
    threads = r.get("threads", []) or []
    # prioritise people (victims/suspects/investigators) then threads; cap the fan-out
    order = {"victim": 0, "suspect": 1, "investigator": 2, "witness": 3}
    ents = sorted(ents, key=lambda e: order.get(e.get("type"), 5))
    items = [{"subject": e["name"], "kind": e.get("type", "other"), "role": e.get("role", "")}
             for e in ents if e.get("name")]
    items += [{"subject": t["name"], "kind": "thread", "role": t.get("question", "")}
              for t in threads if t.get("name")]
    items = items[:config.MAX_ENTITIES]
    _progress(stage="deep_research", msg=f"{len(items)} dossiers")
    return [Send("deep_research", {"item": it, "topic": state["topic"],
                                   "overview": r.get("overview", "")}) for it in items]


def deep_research_node(state):
    it, topic = state["item"], state["topic"]
    cp = cache.path("node_dossier", cache.key(topic, it["subject"], it["kind"]), "json")
    if cache.have(cp):
        return {"dossiers": [cache.load_json(cp)]}
    q = f"{it['subject']} {topic}"
    srcs = research.gather([q, f"{it['subject']} biography life", f"{q} what happened"],
                           per_query=5, fetch_top=2)
    corpus = "\n\n".join(f"SOURCE [{s['url']}] {s['title']}\n{s.get('text', s.get('snippet',''))[:4500]}"
                         for s in srcs[:8])
    user = (f"STORY: {topic}\nCONTEXT: {state.get('overview','')[:1500]}\n\n"
            f"SUBJECT: {it['subject']} ({it['kind']}) — {it.get('role','')}\n\nSOURCES:\n{corpus}")
    try:
        d = llm.chat_json("bulk", DOSSIER_SYS, user, max_tokens=3000)
    except Exception:
        d = {"subject": it["subject"], "kind": it["kind"], "humanizing": "", "role_in_story": it.get("role", ""),
             "key_moments": [], "quotes": [], "disputes": [], "sources": []}
    d.setdefault("subject", it["subject"]); d.setdefault("kind", it["kind"])
    cache.save_json(cp, d)
    _progress(stage="deep_research", msg=f"dossier: {it['subject']}")
    return {"dossiers": [d]}


# ── 3. STORY BIBLE (showrunner) — cast, threads, multi-act beat sheet
BIBLE_SYS = (
    "You are the SHOWRUNNER of a prestige documentary (think Ken Burns, ESPN 30 for 30, Netflix "
    "true-crime). From the research + dossiers, architect ONE episode that would air on a real "
    "network. Ground everything in the material; dramatizing real events is allowed, inventing "
    "facts is not.\n"
    "PRINCIPLES: open on a world/place 'before it all went wrong'; make the audience CARE about "
    "specific named people (victims and the accused, not just the villain); weave MULTIPLE threads "
    "and pay each one off; use red herrings and competing theories honestly; escalate every act; "
    "give present-tense 'aliveness' via expert interviews used as an active investigation; end on a "
    "resonant reflection and the unresolved question.\n"
    "STRUCTURE: a TEASER (cold open) + THREE acts, roughly 25% / 50% / 25% of runtime. Act II is "
    "the long middle: the shady characters, the false leads, the turn.\n"
    "VISUAL GRAMMAR: the episode is carried by a NARRATOR speaking over archival images. Interview "
    "cutaways to an expert are OCCASIONAL — about ONE per act (3-5 total across the episode), placed "
    "where analysis or perspective genuinely helps. MOST beats are narrator VO with shot_hint = "
    "photo/newspaper/map/letter/timeline/evidence/stat. Set \"speaker\" ONLY on the few real interview "
    "beats (shot_hint='interview'); leave it empty everywhere else.\n"
    f"Expert roster for those interview beats (use their id):\n{{ROSTER}}\n"
    "Return JSON: {\n"
    " \"logline\": str, \"target_minutes\": num,\n"
    " \"music_brief\": {\"palette\": str (shared instrumentation — 3-5 CONCRETE instruments that make "
    "every cue sound like one score, rooted in the story's era/place/genre), \"key\": str (e.g. 'D minor'), "
    "\"bpm\": num (slow underscore, ~60-85), \"cues\": {\"tension\": str, \"dread\": str, \"elegy\": str}} "
    "— each cue is a SHORT tag-set (3-6 words: emotion + a texture) for that phase of the arc (teaser/"
    "rising = tension, dark middle = dread, close = elegy). The score MUST be ON-THEME (e.g. 1918 New "
    "Orleans -> funeral jazz: muted trumpet, clarinet, upright bass; a 1970s case -> uneasy analog synth; "
    "a seafaring tragedy -> low strings + accordion), NOT a generic sad-piano bed. Do NOT put bpm/key in "
    "the tag strings. Instrumental only.\n"
    " \"cast\": [{\"name\": str, \"role\": str, \"human_hook\": str (why we care), \"arc\": str}] (3-6 "
    "people we FOLLOW),\n"
    " \"threads\": [{\"name\": str, \"payoff\": str}],\n"
    " \"acts\": [{\"act\": num (0=teaser,1,2,3), \"title\": str, \"goal\": str (what this act does to "
    "the viewer), \"beats\": [{\"purpose\": str (e.g. 'introduce Rosie; make her human'), \"thread\": "
    "str, \"who\": str (person(s) featured), \"shot_hint\": str (photo|newspaper|map|letter|interview|"
    "timeline|evidence|stat), \"speaker\": str (expert id, only if interview)}] }]\n}\n"
    "Plan enough beats across all acts to fill the target runtime at ~150 words/min (a beat ≈ 20-45s). "
    "Do NOT write the narration yet — this is the blueprint.")


def story_bible_node(state):
    topic = state["topic"]
    # dedupe dossiers by subject (operator.add reducer can accumulate on re-run of a thread)
    seen, dossiers = set(), []
    for d in state.get("dossiers", []):
        k = (d.get("subject", ""), d.get("kind", ""))
        if k not in seen:
            seen.add(k); dossiers.append(d)
    r = state.get("research", {})
    cp = cache.path("node_bible", cache.key(topic, config.TARGET_MINUTES,
                                            sorted(d.get("subject", "") for d in dossiers)), "json")
    if cache.have(cp):
        b = cache.load_json(cp)
        _progress(stage="bible", msg=f"cached: {len(b.get('acts', []))} acts")
        return {"story_bible": b}
    _progress(stage="bible", msg="architecting episode")
    dtxt = "\n\n".join(
        f"### {d.get('subject')} ({d.get('kind')})\n{d.get('humanizing','')}\nROLE: {d.get('role_in_story','')}\n"
        f"MOMENTS: " + "; ".join(f"{m.get('when','')}: {m.get('what','')}" for m in d.get('key_moments', [])) +
        ("\nQUOTES: " + " | ".join(f'"{q.get("text","")}" —{q.get("speaker","")}' for q in d.get('quotes', []))
         if d.get('quotes') else "") +
        ("\nDISPUTED: " + "; ".join(d.get('disputes', [])) if d.get('disputes') else "")
        for d in dossiers)
    user = (f"TOPIC: {topic}\nTARGET RUNTIME: {config.TARGET_MINUTES} minutes\n\n"
            f"OVERVIEW:\n{r.get('overview','')}\n\n"
            f"THREADS:\n" + "\n".join(f"- {t.get('name')}: {t.get('question')}" for t in r.get('threads', [])) +
            f"\n\nDOSSIERS:\n{dtxt}")
    b = llm.chat_json("writer", BIBLE_SYS.replace("{ROSTER}", cast.roster_brief()),
                      user, temperature=0.6, max_tokens=8000)
    b.setdefault("target_minutes", config.TARGET_MINUTES)
    cache.save_json(cp, b)
    _progress(stage="bible", msg=f"{len(b.get('acts', []))} acts, {len(b.get('cast', []))} leads")
    return {"story_bible": b}


# ── 4. WRITE ACTS (fan-out) — full narration, per act, with room to breathe
BEAT_FORMAT = (
    "Each beat is an object with: \"kind\" (\"cold_open\" for the very first beat of the teaser, else "
    "\"narration\"), \"narration\" (the VO), \"image_query\" (a concrete, literal archive-search phrase "
    "for a period public-domain visual), and \"shot_type\":\n"
    " - 'photo' (default) | 'newspaper' (a clipping/headline)\n"
    " - 'letter' — a verbatim quoted letter/message/testimony: put ONLY the quoted words in \"quote\" "
    "and use narration as the lead-in\n"
    " - 'interview' — an expert speaks: set \"speaker\" to an expert id and put THEIR first-person "
    "words (2-4 sentences, a real insight or argument) in \"narration\"\n"
    " - 'map' — set \"map_title\", \"map_subtitle\", \"map_points\":[{name,lat,lng}] (approx real "
    "coords); \"map_route\":true ONLY for a real path someone travelled\n"
    " - 'timeline' — \"gfx_title\" + \"events\":[{date,label}] (4-7)\n"
    " - 'evidence' — \"gfx_title\" + \"nodes\":[{label}] (3-6) + \"links\":[[i,j]]\n"
    " - 'stat' — \"stat_value\" (a WORD or number; a lone '0' reads as a ring — prefer 'NONE') + "
    "\"stat_label\"\n"
    "Most beats are 'photo'/'newspaper'; use graphics/interviews where they genuinely serve the story.")

WRITE_ACT_SYS = (
    "You are writing the narration for ONE act of a documentary episode, from the showrunner's beat "
    "sheet. Write like the best in the business: scene reconstruction ('It is just past midnight...'), "
    "concrete sensory detail, present-tense immediacy, subtext, and real stakes. Make named people "
    "HUMAN before the plot turns on them. Ground every claim in the research/dossiers; dramatize real "
    "events but invent nothing. No cliché, no restating what the image shows, no 'AI' meta-notes.\n"
    "Pacing ~150 words/min; honor the beat sheet's intent but expand each beat into real, vivid VO "
    "(a beat is usually 40-110 words).\n"
    "CRITICAL: the DEFAULT voice is the NARRATOR speaking over the archival image named by "
    "image_query — shot_type is 'photo'/'newspaper'/'map'/'letter'/'timeline'/'evidence'/'stat'. "
    "Use shot_type='interview' ONLY for a beat the sheet explicitly marks interview (with a speaker "
    "id) — and then the narration must be that expert's OWN first-person words, not narrator VO. Do "
    "NOT turn ordinary narration beats into interviews. At most one interview in this act.\n"
    + BEAT_FORMAT + "\n"
    "Return JSON: {\"segments\": [ <beat>, ... ]} for THIS act only, in order.")


def write_act_fanout(state):
    b = state.get("story_bible", {})
    acts = sorted(b.get("acts", []), key=lambda a: a.get("act", 0))
    _progress(stage="write", msg=f"{len(acts)} acts")
    return [Send("write_act", {"act": a, "bible": b, "topic": state["topic"],
                               "dossiers": state.get("dossiers", []),
                               "research": state.get("research", {})}) for a in acts]


def _dossier_digest(dossiers, limit=6000):
    txt = "\n\n".join(
        f"### {d.get('subject')} ({d.get('kind')})\n{d.get('humanizing','')}\n"
        f"MOMENTS: " + "; ".join(f"{m.get('when','')}: {m.get('what','')}" for m in d.get('key_moments', [])) +
        ("\nQUOTES: " + " | ".join(f'"{q.get("text","")}"' for q in d.get('quotes', [])) if d.get('quotes') else "")
        for d in dossiers)
    return txt[:limit]


def write_act_node(state):
    a, b, topic = state["act"], state["bible"], state["topic"]
    an = a.get("act", 0)
    cp = cache.path("node_act", cache.key(topic, an, a.get("title", ""),
                                          config.TARGET_MINUTES, len(state.get("dossiers", []))), "json")
    if cache.have(cp):
        out = cache.load_json(cp)
    else:
        beats = "\n".join(
            f"  {i+1}. purpose={bt.get('purpose','')} | thread={bt.get('thread','')} | "
            f"who={bt.get('who','')} | shot={bt.get('shot_hint','photo')}"
            + (f" | speaker={bt.get('speaker')}" if bt.get('speaker') else "")
            for i, bt in enumerate(a.get("beats", [])))
        others = "  " + "\n  ".join(f"Act {o.get('act')}: {o.get('title')} — {o.get('goal')}"
                                    for o in b.get("acts", []) if o.get("act") != an)
        user = (f"EPISODE: {b.get('logline','')}\nTOPIC: {topic}\n"
                f"CAST WE FOLLOW:\n" + "\n".join(f"  - {c.get('name')}: {c.get('human_hook')}" for c in b.get('cast', [])) +
                f"\n\nEXPERTS (interview ids):\n{cast.roster_brief()}\n\n"
                f"THIS ACT — Act {an}: {a.get('title','')}\nGOAL: {a.get('goal','')}\nBEAT SHEET:\n{beats}\n\n"
                f"OTHER ACTS (for continuity, do not write them):\n{others}\n\n"
                f"RESEARCH DIGEST:\n{_dossier_digest(state.get('dossiers', []))}")
        out = llm.chat_json("writer", WRITE_ACT_SYS, user, temperature=0.75, max_tokens=8000)
        cache.save_json(cp, out)
    _progress(stage="write", msg=f"act {an}: {len(out.get('segments', []))} beats")
    return {"act_drafts": [{"act": an, "segments": out.get("segments", [])}]}


# ── 5. ASSEMBLE SCRIPT — order acts, flatten to the segment list downstream wants
def _cap_interviews(segs, max_n=5):
    """Hard guardrail: the LLM sometimes tags most/all beats as 'interview' (a wall of talking
    heads — monotonous and hugely expensive to render). Keep interviews occasional and SPREAD OUT;
    demote the surplus to a narrator VO still whose shot type is inferred from its image_query."""
    iv = [s for s in segs if s.get("shot_type") == "interview"]
    if len(iv) <= max_n:
        return segs
    # keep interviews evenly spaced across the episode; demote the rest
    keep = set(id(iv[round(k * (len(iv) - 1) / (max_n - 1))]) for k in range(max_n))
    for s in segs:
        if s.get("shot_type") == "interview" and id(s) not in keep:
            s["shot_type"] = archives.shot_type_of({"title": "", "image_query": s.get("image_query", "")})
            s["speaker"] = ""
    return segs


def _norm_seg(s, i, topic):
    return {"id": i, "kind": s.get("kind", "narration"),
            "narration": (s.get("narration", "") or "").strip(),
            "image_query": s.get("image_query", topic),
            "shot_type": s.get("shot_type", "photo"),
            "quote": (s.get("quote", "") or "").strip(),
            "speaker": (s.get("speaker", "") or "").strip(),
            "map_title": s.get("map_title", ""), "map_subtitle": s.get("map_subtitle", ""),
            "map_points": s.get("map_points", []), "map_route": bool(s.get("map_route", False)),
            "gfx_title": s.get("gfx_title", ""), "events": s.get("events", []),
            "nodes": s.get("nodes", []), "links": s.get("links", []),
            "stat_value": s.get("stat_value", ""), "stat_label": s.get("stat_label", ""),
            "enhanced": False}


def assemble_script_node(state):
    # dedupe act drafts by act number (reducer accumulation guard), order, flatten
    by_act = {}
    for d in state.get("act_drafts", []):
        by_act[d["act"]] = d          # keep last
    topic = state["topic"]
    raw = []
    for an in sorted(by_act):
        for s in by_act[an].get("segments", []):
            raw.append((an, s))
    # the very first beat of the whole episode is the cold open
    segs = []
    for i, (an, s) in enumerate(raw):
        seg = _norm_seg(s, i, topic)
        seg["act"] = an
        segs.append(seg)
    if segs:
        segs[0]["kind"] = "cold_open"
    segs = _cap_interviews(segs)
    words = sum(len(s["narration"].split()) for s in segs)
    _progress(stage="assemble_script", msg=f"{len(segs)} beats, ~{words} words "
              f"(~{words // config.WORDS_PER_MIN} min)")
    return {"segments": segs, "title": state.get("story_bible", {}).get("logline", state.get("title"))}


# ── 6. SHOWRUNNER EDIT — critique the whole draft against a rubric, revise weak beats
EDITOR_SYS = (
    "You are the SHOWRUNNER giving final notes on a documentary script, then delivering the revised "
    "cut. Judge it as a broadcast professional would and FIX what falls short.\n"
    "RUBRIC: (1) Do we CARE about specific named people — are victims/accused human, not props? "
    "(2) Are the threads set up and PAID OFF, or dropped? (3) Is it CLEAR — can a first-time viewer "
    "follow who/when/where? (4) Does each act ESCALATE (no sag, no rush)? (5) Present-tense aliveness, "
    "scene reconstruction, subtext — not a dry recap? (6) Everything grounded (no invented facts/quotes)? "
    "(7) A strong cold open and a resonant close?\n"
    "Rewrite weak beats IN PLACE (sharpen VO, add a human detail, fix a confusing jump, strengthen a "
    "thread hand-off). Keep the same beat objects/shape and shot_types; you may split or merge a beat "
    "or adjust ordering for clarity, but keep it grounded and keep the runtime near target. No 'AI' notes.\n"
    "PRESERVE THE VISUAL GRAMMAR: most beats are narrator VO over archival images "
    "(photo/newspaper/map/letter/timeline/evidence/stat); interviews stay OCCASIONAL (~one per act). "
    "Do NOT convert narration beats into interviews, and do not collapse the shot-type variety.\n"
    "Return JSON: {\"notes\": {\"scores\": {\"care\":1-10,\"threads\":1-10,\"clarity\":1-10,"
    "\"escalation\":1-10,\"craft\":1-10}, \"summary\": str, \"fixed\": [str]}, "
    "\"segments\": [ <full revised beat list, same format> ]}.")


def script_editor_node(state):
    segs = state.get("segments", [])
    topic = state["topic"]
    cp = cache.path("node_editor", cache.key(topic, config.EDITOR_PASSES,
                                             [s["narration"] for s in segs]), "json")
    if cache.have(cp):
        out = cache.load_json(cp)
        _progress(stage="editor", msg=f"cached: scores {out.get('notes', {}).get('scores', {})}")
        return {"segments": out["segments"], "editor_notes": out.get("notes", {})}
    b = state.get("story_bible", {})
    cur = json.dumps([{k: s[k] for k in ("kind", "narration", "shot_type", "speaker", "quote",
                                         "image_query") if s.get(k)} for s in segs])[:24000]
    notes, out_segs = {}, segs
    for p in range(max(1, config.EDITOR_PASSES)):
        _progress(stage="editor", msg=f"showrunner pass {p+1}")
        user = (f"LOGLINE: {b.get('logline','')}\nTARGET: {config.TARGET_MINUTES} min\n"
                f"CAST WE SHOULD CARE ABOUT:\n" + "\n".join(f"  - {c.get('name')}: {c.get('human_hook')}"
                                                            for c in b.get('cast', [])) +
                f"\nTHREADS TO PAY OFF:\n" + "\n".join(f"  - {t.get('name')}: {t.get('payoff')}"
                                                       for t in b.get('threads', [])) +
                f"\n\nCURRENT SCRIPT (beats):\n{cur}")
        try:
            res = llm.chat_json("writer", EDITOR_SYS, user, temperature=0.5, max_tokens=9000)
        except Exception:
            break
        new = res.get("segments")
        if new:
            out_segs = [_norm_seg(s, i, topic) for i, s in enumerate(new)]
            if out_segs:
                out_segs[0]["kind"] = "cold_open"
            out_segs = _cap_interviews(out_segs)
            cur = json.dumps([{k: s[k] for k in ("kind", "narration", "shot_type", "speaker",
                              "quote", "image_query") if s.get(k)} for s in out_segs])[:24000]
        notes = res.get("notes", {})
    cache.save_json(cp, {"segments": out_segs, "notes": notes})
    _progress(stage="editor", msg=f"scores {notes.get('scores', {})}")
    return {"segments": out_segs, "editor_notes": notes}



# ── 2b. CURATE (pool -> assign): global, reviewable, deterministic ───
def curate_node(state):
    """Build a topic-wide pool of rights-clean images, rank by archival-ness (low color
    saturation + period year + curated source), then assign the best-fitting unused image
    per beat. Beats marked 'letter' stay quote cards. Decided before the expensive render."""
    segs = [dict(s) for s in state["segments"]]
    topic = state["topic"]
    cp = cache.path("node_curate", cache.key(topic, [s["image_query"] for s in segs]), "json")
    if cache.have(cp):
        _progress(stage="curate", msg="cached")
        return {"segments": cache.load_json(cp)}

    # 1. gather pool: Wikipedia-article images (curated/period) + topic + per-beat queries + LoC
    pool, seen = [], set()
    pool += archives.wikipedia_images(topic, 20)
    queries = [topic, f"{topic} New Orleans", f"{topic} 1919"] + \
              [s["image_query"] for s in segs if s.get("shot_type") != "letter"]
    for q in queries:
        try:
            pool += [c for c in archives.commons_search(q, 6) if c.get("rights_ok")]
        except Exception:
            pass
    try:
        pool += archives.loc_search(f"New Orleans {topic}", 8)
    except Exception:
        pass
    uniq = []
    for c in pool:
        u = c.get("full_url") or c.get("image_url")
        if u and u not in seen:
            seen.add(u)
            uniq.append(c)
    pool = uniq[:40]
    _progress(stage="curate", msg=f"pool={len(pool)} candidates")

    # 2. download + score each candidate. If the vision judge is up, enrich ONCE per
    #    candidate (caption + period) — caption beats the filename for relevance matching.
    vis = vision.available()
    _progress(stage="curate", msg=f"vision judge: {'on' if vis else 'off (heuristic)'}")
    for c in pool:
        p = archives.download(c, topic)
        c["_path"] = p
        if not p:
            c["_score"] = -9
            continue
        score = archives.archival_score(c, archives.saturation(p))
        if vis:
            vr = vision.score(p, f"a documentary about {topic}", topic)
            if vr:
                c["_caption"] = vr.get("caption", "")
                per = vr.get("period")
                if per is not None:
                    score += 1.5 * float(per)        # vision-confirmed period photo
        c["_score"] = score

    # 3. assign best-fitting image per beat: relevance + archival score, with a strong
    #    reuse penalty (use-count) and an extra penalty for repeating on adjacent beats.
    # Reserve real archival MAP scans for the map beat: a period city plan is a spatial
    # asset (and usually the strongest-scoring image in the pool), so it must not land on an
    # atmospheric photo beat. If one exists it renders on the 'map' beat as a graded still —
    # a real 1919 map beats the procedural docgfx map. Excluded from the general still pool.
    def _ismap(c):
        return archives.shot_type_of({"title": c.get("title", "")}) == "map"
    map_cands = [c for c in pool if c.get("_path") and _ismap(c)]
    best_map = max(map_cands, key=lambda c: c["_score"]) if map_cands else None
    map_ids = {id(c) for c in map_cands}
    usecount = {}
    prev_id = None
    for s in segs:
        st0 = s.get("shot_type")
        if st0 == "map" and best_map:                   # real period map -> graded still
            s["asset"] = {k: best_map[k] for k in ("source", "title", "page_url", "license",
                                                    "attribution", "rights_ok") if k in best_map}
            s["image_path"] = best_map["_path"]
            prev_id = None
            continue
        if st0 in ("letter", "interview", "slide", "map", "timeline", "evidence", "stat"):
            prev_id = None
            continue                                    # quote card / talking head / graphic, no image
        if s.get("kind") == "cold_open":
            # the hook is an evocative scene-setter: an archival still rarely matches it and a
            # dense document kills the mood. Reserve it for a generated period illustration of
            # the described scene (styled etching -> reads as 'artist's depiction', not footage).
            prev_id = None
            continue
        kw = archives._keywords(s["image_query"]) | archives._keywords(topic)
        best, bestv = None, -1e9
        for c in pool:
            if not c.get("_path") or id(c) in map_ids:  # map scans reserved for the map beat
                continue
            desc = c.get("title", "") + " " + c.get("_caption", "")   # caption enriches match
            rel = len(archives._keywords(desc) & kw)
            v = rel * 2.0 + c["_score"] - 5.0 * usecount.get(id(c), 0)
            if id(c) == prev_id:
                v -= 8.0                                 # never repeat back-to-back if avoidable
            if v > bestv:
                best, bestv = c, v
        if best:
            usecount[id(best)] = usecount.get(id(best), 0) + 1
            prev_id = id(best)
            s["asset"] = {k: best[k] for k in ("source", "title", "page_url", "license",
                                               "attribution", "rights_ok") if k in best}
            s["image_path"] = best["_path"]
            # a beat that carries an archival raster image is only ever 'photo' or
            # 'newspaper' — never a docgfx graphic type ('map'/'timeline'/... need
            # structured data, not a still). A matched *map image* renders as a photo still.
            st = s.get("shot_type", "photo")
            if st != "newspaper":
                st = "newspaper" if archives.shot_type_of(
                    {"title": best.get("title", ""),
                     "image_query": s["image_query"]}) == "newspaper" else "photo"
            s["shot_type"] = st
    # beats with no archival match (and not a letter) -> generative period illustration
    for s in segs:
        if not s.get("image_path") and s.get("shot_type") not in ("letter", "interview", "slide", "map", "timeline", "evidence", "stat"):
            s["shot_type"] = "illustration"
    # evidence beats: pin a REAL archival photo into each card (matched from the pool by label,
    # spread so cards differ) so the board reads as a true investigation wall, not empty frames.
    for s in segs:
        if s.get("shot_type") != "evidence":
            continue
        used = {}
        for nd in s.get("nodes", []):
            if not isinstance(nd, dict) or nd.get("img") or not nd.get("label"):
                continue
            kw = archives._keywords(nd["label"]) | archives._keywords(topic)
            best, bv = None, -1e9
            for c in pool:
                if not c.get("_path"):
                    continue
                rel = len(archives._keywords(c.get("title", "") + " " + c.get("_caption", "")) & kw)
                v = rel * 2.0 + c.get("_score", 0) - 3.0 * used.get(id(c), 0)
                if v > bv:
                    best, bv = c, v
            if best:
                nd["img"] = os.path.abspath(best["_path"])
                used[id(best)] = used.get(id(best), 0) + 1
    cache.save_json(cp, segs)
    placed = sum(1 for s in segs if s.get("image_path"))
    illus = sum(1 for s in segs if s.get("shot_type") == "illustration")
    _progress(stage="curate", msg=f"assigned {placed}/{len(segs)} archival, {illus} illustration")
    return {"segments": segs}


# ── 3. REVIEW (human-in-the-loop) ────────────────────────────────────
def review_script(state):
    decision = interrupt({"action": "approve_script", "title": state.get("title"),
                          "segments": state["segments"]})
    if isinstance(decision, dict) and decision.get("segments"):
        return {"segments": decision["segments"], "script_approved": True}
    return {"script_approved": True}


# ── 4. FAN-OUT + RENDER ──────────────────────────────────────────────
def fan_out(state):
    return [Send("render_segment", {"seg": s, "job_id": state["job_id"]})
            for s in state["segments"]]


def render_segment(state):
    """Execute the treatment decided by curate: upscale -> archival grade + shot-grammar
    Ken-Burns for assigned stills; quote card for letters; wrapped text card as last resort."""
    seg = dict(state["seg"])
    job = state["job_id"]
    rdir = os.path.join(config.RENDERS_DIR, job)
    os.makedirs(rdir, exist_ok=True)
    sid = seg["id"]
    shot = seg.get("shot_type", "photo")
    _progress(stage="render", seg=sid, msg=f"{shot}: {seg.get('image_query','')[:40]}")
    cue = [f"Segment {sid}: AI-synthesized narration (Qwen3-TTS, designed voice)"]

    # narration — interview beats speak in the RESEARCHER's voice, everything else the narrator
    wav = os.path.join(rdir, f"seg{sid:02d}.wav")
    vref = None
    if shot == "interview":
        vref = cast.BY_ID.get(seg.get("speaker") or "hale", cast.RESEARCHERS[0])["voice_ref"]
    dur = tts.narrate(seg["narration"], wav, ref=vref) if seg.get("narration") else 0.0
    dur = max(dur, config.STILL_MIN_SEC)

    clip = os.path.join(rdir, f"seg{sid:02d}.mp4")
    img_path = seg.get("image_path")
    rendered = False
    if shot == "letter" and seg.get("quote"):
        media.quote_card(seg["quote"], state.get("title", ""), dur, clip, narration_wav=wav)
        cue.append(f"Segment {sid}: quote card (verbatim historical text)")
        rendered = True
    elif shot == "interview":                       # talking-head researcher (news pipeline)
        r = cast.BY_ID.get(seg.get("speaker") or "hale", cast.RESEARCHERS[0])
        try:
            th = ""
            if anchor.available():
                port = cast.portrait(r)              # Z-Image headshot (cached)
                bg = config.INTERVIEW_BG if os.path.isfile(config.INTERVIEW_BG) else None
                th = anchor.talking_head(port, wav, clip, bg=bg)
            if th:                                   # make_anchor: swap face + MuseTalk + restore
                media.interview_segment(th, r["name"], r["title"], clip)
                cue.append(f"Segment {sid}: researcher interview ({r['name']}, {r['title']})")
                rendered = True
        except Exception as e:  # noqa
            cue.append(f"Segment {sid}: interview render failed ({str(e)[:50]})")
    elif shot == "map" and img_path and os.path.exists(img_path):
        # a real archival map scan beats the procedural one — render it as a graded still
        try:
            up = restore.upscale(img_path)
            media.still_segment(up, wav, dur, clip, shot_type="map", zoom_in=(sid % 2 == 0))
            cue.append(f"Segment {sid}: real period map (archival scan), archival grade")
            rendered = True
        except Exception as e:  # noqa
            cue.append(f"Segment {sid}: map still failed ({str(e)[:40]})")
    elif shot == "map":                             # on-brand vintage animated map (docgfx)
        pts = seg.get("map_points") or []
        if pts and docgfx.available():
            spec = docgfx.map_spec(seg.get("map_title") or state.get("topic", "").upper(),
                                   seg.get("map_subtitle", ""), pts, route=bool(seg.get("map_route")))
            if docgfx.clip(spec, wav, dur, clip):
                cue.append(f"Segment {sid}: animated map ({len(pts)} locations)")
                rendered = True
    elif shot == "timeline" and seg.get("events") and docgfx.available():
        if docgfx.clip(docgfx.timeline_spec(seg.get("gfx_title") or "TIMELINE", seg["events"]), wav, dur, clip):
            cue.append(f"Segment {sid}: animated timeline ({len(seg['events'])} events)")
            rendered = True
    elif shot == "evidence" and seg.get("nodes") and docgfx.available():
        if docgfx.clip(docgfx.evidence_spec(seg.get("gfx_title") or "THE CONNECTIONS",
                                            seg["nodes"], seg.get("links", [])), wav, dur, clip):
            cue.append(f"Segment {sid}: evidence board ({len(seg['nodes'])} cards)")
            rendered = True
    elif shot == "stat" and seg.get("stat_value") and docgfx.available():
        if docgfx.clip(docgfx.stat_spec(seg["stat_value"], seg.get("stat_label", "")), wav, dur, clip):
            cue.append(f"Segment {sid}: stat card")
            rendered = True
    elif shot == "slide":                           # (legacy) Presenton explainer slide
        pages = slides.deck_pages(seg.get("narration", ""), n_slides=1) if slides.available() else []
        if pages:
            media.slide_segment(pages[0], wav, dur, clip, zoom_in=(sid % 2 == 0))
            cue.append(f"Segment {sid}: explainer slide (Presenton)")
            rendered = True
    elif shot == "illustration":                    # no archival image -> period illustration
        scene = seg.get("image_query") or seg.get("narration", "")[:120]
        gen = generate.illustration(scene, "") if generate.available() else ""
        if gen:
            media.still_segment(gen, wav, dur, clip, shot_type="photo", zoom_in=(sid % 2 == 0))
            cue.append(f"Segment {sid}: generated period illustration")
            rendered = True
    elif img_path and os.path.exists(img_path):
        try:
            if config.PREMIUM_UPSCALE and superscale.available():
                up = superscale.flashvsr(img_path, scale=2)   # resident FlashVSR (rtx0)
                if up != img_path:
                    cue.append(f"Segment {sid}: FlashVSR superscale of public-domain still")
            else:
                up = restore.upscale(img_path)              # ESRGAN if small; else original
                if up != img_path:
                    cue.append(f"Segment {sid}: ESRGAN upscale of public-domain still")
            media.still_segment(up, wav, dur, clip, shot_type=shot, zoom_in=(sid % 2 == 0))
            cue.append(f"Segment {sid}: archival grade (sepia/grain/vignette) applied")
            rendered = True
        except Exception as e:  # noqa
            cue.append(f"Segment {sid}: image render failed ({str(e)[:50]}); text card")
    if not rendered:
        media.title_card(seg.get("narration", "")[:140], dur, clip, narration_wav=wav)
        if seg.get("kind") != "cold_open":
            cue.append(f"Segment {sid}: no public-domain image found; text card used")
    seg.update(wav_path=wav, dur=dur, clip_path=clip)
    return {"rendered": [seg], "cue_sheet": cue}


# ── 5. ASSEMBLE ──────────────────────────────────────────────────────
def assemble_node(state):
    # dedupe by id (the `rendered` reducer is operator.add; re-running a job on the
    # same thread_id would otherwise accumulate — and concat each beat more than once).
    by_id = {}
    for s in state["rendered"]:
        by_id[s["id"]] = s                      # keep the last render of each beat
    segs = sorted(by_id.values(), key=lambda s: s["id"])
    title = state.get("title", state["topic"])
    job = state["job_id"]
    edir = os.path.join(config.EPISODES_DIR, job)
    os.makedirs(edir, exist_ok=True)

    problems = provenance.gate(segs)
    if problems:
        return {"errors": problems}

    intro = os.path.join(edir, "intro.mp4")
    media.title_card(title, 4.0, intro)
    clips = [intro] + [s["clip_path"] for s in segs if s.get("clip_path")]

    body = os.path.join(edir, "_body.mp4")
    media.concat(clips, body)
    episode = os.path.join(edir, "episode.mp4")

    # original score — ACE-Step. An EVOLVING bed matched to the arc: distinct cues (tension ->
    # dread -> elegy), one shared instrumentation palette, placed at the act turns and ducked under VO.
    score_wav = None
    if config.MUSIC_ENABLE and music.available():
        mb = state.get("story_bible", {}).get("music_brief", {}) or {}
        palette = mb.get("palette", "") or mb.get("instrumentation", "")
        key = mb.get("key", "")
        bpm = int(mb.get("bpm", 72) or 72)
        cue_tags = mb.get("cues") or {}
        # default arc if the bible didn't specify per-cue tags
        if not cue_tags:
            cue_tags = {"tension": "sparse, uneasy, suspenseful", "dread": "ominous, heavy, dark",
                        "elegy": "mournful, reflective, grief"}
        try:
            import subprocess as _sp
            total = float(_sp.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                                   "-of", "default=nk=1:nw=1", body], capture_output=True,
                                  text=True).stdout.strip() or 0)
            plan = [("tension", 0.0, 0.30), ("dread", 0.30, 0.75), ("elegy", 0.75, 1.0)]
            cues = []
            for i, (mood, a, b) in enumerate(plan):
                tags = ", ".join(x for x in [palette, cue_tags.get(mood, mood)] if x)
                cp = music.score(tags, os.path.join(edir, f"cue_{mood}.mp3"),
                                 duration=min(config.MUSIC_SECONDS, 75), bpm=bpm, key=key, seed=7 + i)
                if cp:
                    cues.append((cp, a, b))
            if cues and total > 0:
                score_wav = media.score_bed(cues, total, os.path.join(edir, "score.wav")) or None
                if score_wav:
                    _progress(stage="assemble", msg=f"score: {len(cues)}-cue evolving bed (ACE-Step)")
        except Exception as e:  # noqa
            _progress(stage="assemble", msg=f"score skipped ({str(e)[:40]})")
            score_wav = None
    media.add_music_and_master(body, score_wav, episode)

    credits = provenance.write_credits(edir, segs, state.get("cue_sheet", []))
    _progress(stage="assemble", msg=f"episode -> {episode}")
    return {"episode_path": episode, "credits_path": credits}
