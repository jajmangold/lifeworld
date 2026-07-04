"""Rights gate + attribution credits + GenAI cue sheet (APA disclosure)."""
import os
from . import config


def gate(segments) -> list:
    """Return list of rights problems; empty = OK to assemble."""
    problems = []
    for s in segments:
        a = s.get("asset")
        if s.get("image_path") and (not a or not a.get("rights_ok")):
            problems.append(f"segment {s.get('id')}: image has unresolved rights")
    return problems


def credits_text(segments, cue_sheet) -> str:
    lines = ["SOURCES & ATTRIBUTIONS", ""]
    seen = set()
    for s in segments:
        a = s.get("asset") or {}
        key = (a.get("attribution"), a.get("title"))
        if a and key not in seen:
            seen.add(key)
            lines.append(f"• {a.get('title','(image)')} — {a.get('attribution','')} "
                         f"[{a.get('license','')}] {a.get('page_url','')}")
    return "\n".join(lines)


def write_credits(job_dir, segments, cue_sheet) -> str:
    p = os.path.join(job_dir, "credits.txt")
    open(p, "w", encoding="utf-8").write(credits_text(segments, cue_sheet))
    return p
