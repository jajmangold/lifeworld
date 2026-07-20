"""Rights gate + attribution credits + GenAI cue sheet (APA disclosure)."""
import os
from . import config


def _visual_assets(segment):
    if segment.get("image_path"):
        yield f"segment {segment.get('id')}", segment.get("asset") or {}
    for index, node in enumerate(segment.get("nodes") or []):
        if isinstance(node, dict) and node.get("img"):
            yield f"segment {segment.get('id')} evidence node {index}", node.get("asset") or {}


def gate(segments) -> list:
    """Return list of rights problems; empty = OK to assemble."""
    problems = []
    for s in segments:
        for location, asset in _visual_assets(s):
            if not asset.get("rights_ok"):
                problems.append(f"{location}: image has unresolved rights")
    return problems


def credits_text(segments, cue_sheet) -> str:
    lines = ["SOURCES & ATTRIBUTIONS", ""]
    seen = set()
    for s in segments:
        for _, asset in _visual_assets(s):
            key = (asset.get("attribution"), asset.get("title"))
            if asset and key not in seen:
                seen.add(key)
                lines.append(f"• {asset.get('title','(image)')} — {asset.get('attribution','')} "
                             f"[{asset.get('license','')}] {asset.get('page_url','')}")
    return "\n".join(lines)


def write_credits(job_dir, segments, cue_sheet) -> str:
    p = os.path.join(job_dir, "credits.txt")
    open(p, "w", encoding="utf-8").write(credits_text(segments, cue_sheet))
    return p
