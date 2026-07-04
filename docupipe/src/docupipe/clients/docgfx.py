"""Documentary motion-graphics client. Drives the Node renderer (docgfx/render.js: d3+jsdom->resvg
-> vintage frames), then media.graphic_segment applies the shared archival grain/vignette + VO.
Deterministic + cached by spec. Replaces Presenton for on-brand explainer beats (maps, timelines...)."""
import json
import os
import shutil
import subprocess
from .. import config, cache
from . import media, geo

DOCGFX_DIR = os.path.join(config.ROOT, "docgfx")


def available() -> bool:
    return os.path.isfile(os.path.join(DOCGFX_DIR, "render.js")) and \
        os.path.isdir(os.path.join(DOCGFX_DIR, "node_modules"))


def render_frames(spec: dict) -> str:
    """Run the Node renderer for a spec; returns the frames dir (cached by spec)."""
    key = cache.key("docgfx", spec)
    frames = os.path.join(config.CACHE_DIR, "docgfx", key)
    done = frames + ".done"
    if cache.have(done):
        return frames
    if os.path.isdir(frames):
        shutil.rmtree(frames)
    os.makedirs(frames, exist_ok=True)
    spec = dict(spec, out_dir=frames)
    sp = os.path.join(frames, "spec.json")
    with open(sp, "w") as f:
        json.dump(spec, f)
    r = subprocess.run(["node", "render.js", sp], cwd=DOCGFX_DIR,
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0 or "DOCGFX_OK" not in r.stdout:
        raise RuntimeError("docgfx render failed: " + (r.stderr or r.stdout)[-400:])
    open(done, "w").write("ok")
    return frames


def clip(spec: dict, narration_wav: str, dur: float, out: str) -> str:
    """Full graphic beat: render vintage frames -> archival grain/vignette finish + VO. "" on failure."""
    try:
        spec = dict(spec, duration=round(dur, 3), fps=config.FPS, W=config.W, H=config.H)
        frames = render_frames(spec)
        media.graphic_segment(frames, narration_wav, dur, out)
        return out
    except Exception:
        return ""


def map_spec(title, subtitle, points, route=True) -> dict:
    """points: [{name, lat, lng, labelBelow?}]. Fetches REAL OSM streets for the bbox as the base."""
    pts = [p for p in points if p.get("lat") and p.get("lng")]
    try:
        st = geo.streets(pts)
    except Exception:
        st = []
    return {"type": "map", "title": title, "subtitle": subtitle, "route": route,
            "points": pts, "streets": st}


def timeline_spec(title, events) -> dict:
    """events: [{date, label}]."""
    return {"type": "timeline", "title": title, "events": events}


def evidence_spec(title, nodes, links) -> dict:
    """nodes: [{label}]; links: [[i,j], ...] (indices into nodes)."""
    return {"type": "evidence", "title": title, "nodes": nodes, "links": links}


def stat_spec(value, label) -> dict:
    return {"type": "stat", "value": str(value), "label": label}
