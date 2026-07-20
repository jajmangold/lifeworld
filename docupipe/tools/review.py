#!/usr/bin/env python3
"""Art-direct a docgfx frame with the local qwen9b-vis model.
  usage: review.py <image> <type> [context]     type: map|timeline|evidence|stat|still|illustration|interview

Uses Qwen's OFFICIAL non-thinking VL sampling (temp 0.7 / top_p 0.8 / top_k 20 / presence_penalty 1.5 /
rep 1.0) — near-greedy (low temp) makes Qwen loop/degrade. Sends 1100px (grain must be visible or the
judge falsely reports 'no texture'). Asks a SPECIFIC per-type checklist, not open-ended 'critique'."""
import base64
import json
import subprocess
import sys
import urllib.request

import os
VISION_URL = os.environ.get("QWEN_URL", "https://amd0.python-bull.ts.net/v1/chat/completions")  # qwen9b on amd0
VISION_MODEL = os.environ.get("QWEN_MODEL", "Qwen3.5-9B-UD-Q4_K_XL.gguf")

# per-type checklists — concrete, answerable PASS/FAIL items beat vague "critique this"
CHECK = {
    "map": [
        "BASE: does a real period city plan read here (streets/blocks/river visible), or is it a near-empty field with floating pins?",
        "PAPER: is aged-paper texture + uneven toning visible (not a flat digital fill)?",
        "TYPE: are labels a period letterpress serif (IM Fell-like), not a clean modern/UI sans?",
        "TITLE: is the title a period cartouche/engraved plate, not a flat modern panel/UI card?",
        "MARKERS: do the location markers look drawn/engraved & era-appropriate, not modern app map-pins?",
        "GRADE: sepia + film grain present and even across the whole frame?",
    ],
    "timeline": [
        "SPINE: does the timeline read as an etched/inked line on a document, not a thin clean vector rule?",
        "PAPER: aged paper texture + grain visible (not flat beige)?",
        "TYPE: dates & labels in a period serif with character; is contrast readable?",
        "LAYOUT: does it feel like an archival exhibit (a little organic), or a symmetric corporate slide?",
        "COLOR: are date & label colors cohesive/period, not a 'modern UI' accent-color scheme?",
    ],
    "evidence": [
        "BOARD: does it read as a physical cork/pin board with depth, not flat graphics?",
        "STRING: do connectors look like real red string/yarn (sagging, physical)?",
        "CARDS: aged photo/card frames, pins, slight rotation — tactile not vector?",
        "PAPER/GRADE: texture + grain present?",
        "TYPE: handwritten/typewriter labels vs clean sans?",
    ],
    "stat": [
        "FOCUS: one dominant number, held, with clear hierarchy?",
        "SUBSTRATE: does the number sit on a period card/ledger/paper, not a floating HUD?",
        "TYPE: period slab/serif with weight; not modern UI?",
        "GRADE: sepia + grain even across frame?",
    ],
    "interview": [
        "SUBJECT: does this read as a documentary EXPERT interview (historian in a study/archive/neutral set), NOT a live TV news anchor at a news desk?",
        "BACKDROP: is the background period/neutral/subject-appropriate (books, archive, dark set), NOT a modern glass-tower newsroom with city bokeh?",
        "LOWER-THIRD: is the name/title chyron tasteful and documentary-style, not a breaking-news bug?",
        "GRADE: does the shot sit tonally with an archival film (some warmth/filmic treatment), or does it clash as bright modern broadcast?",
        "FRAMING: is the subject framed like an interview (slightly off-center, eyeline), not centered news-presenter?",
    ],
    "_default": [
        "Does it read as genuine archival documentary material vs a corporate slide?",
        "Aged paper/texture + even film grain present?",
        "Period-appropriate typography (not modern/UI)?",
        "Composition & focal hierarchy?",
    ],
}


def review(image, gtype="_default", context=""):
    small = image.rsplit(".", 1)[0] + "_rvw.jpg"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", image, "-vf", "scale=1100:-2", "-q:v", "3", small],
                   timeout=30)
    b = base64.b64encode(open(small, "rb").read()).decode()
    items = CHECK.get(gtype, CHECK["_default"])
    checklist = "\n".join(f"{i+1}. {c}" for i, c in enumerate(items))
    sysmsg = (
        "You are a senior broadcast motion-graphics art director for archival/period documentaries "
        "(think Ken Burns, ESPN 30 for 30, Netflix true-crime). You judge whether a graphic reads as "
        "genuine archival material vs a corporate slide. Be specific, concrete, and blunt — cite exactly "
        "what you see in the frame. Do not invent problems that aren't visible; if something passes, say so.")
    user = (f"This is a {gtype.upper()} graphic for a period true-crime documentary. {context}\n\n"
            f"Go through this checklist. For EACH item answer PASS or FAIL and give ONE specific, "
            f"actionable fix (name the exact element and change):\n{checklist}\n\n"
            f"Then give an OVERALL score /10 and the single highest-impact next change.")
    body = {"model": VISION_MODEL, "chat_template_kwargs": {"enable_thinking": False},
            "temperature": 0.7, "top_p": 0.8, "top_k": 20, "presence_penalty": 1.5,
            "repetition_penalty": 1.0, "max_tokens": 700,
            "messages": [{"role": "system", "content": sysmsg},
                         {"role": "user", "content": [
                             {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b}},
                             {"type": "text", "text": user}]}]}
    req = urllib.request.Request(VISION_URL, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=150))["choices"][0]["message"]["content"]


if __name__ == "__main__":
    print(review(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "_default",
                 sys.argv[3] if len(sys.argv) > 3 else ""))
