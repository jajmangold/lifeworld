"""Non-talking segment renderers — VO-over-broll, title cards, plain narration-over-image. These WRAP the
existing ffmpeg/Ken-Burns builders in newscast/ (build_vo_broll.py, build_title.py) and the docupipe
Ken-Burns node. No talking head, so no swap/muse/flashvsr — just image(s) + narration + motion + graphics.
Kept thin so the dispatcher can compose a full episode from mixed segment kinds.
"""
import os, subprocess
from .registry import register

BOT = "/srv/nvme-data/containers/live/studio"

@register("vo_broll")
def render_vo_broll(seg, ctx):
    out = os.path.join(BOT, "output", f"seg_{seg['id']}.mp4")
    spec = seg.get("spec", {})
    cmd = ["python3", os.path.join(BOT, "newscast", "build_vo_broll.py"),
           "--audio", seg["wav_path"], "--out", out]
    if spec.get("broll"):
        cmd += ["--broll", *spec["broll"]]
    if seg.get("image_path"):
        cmd += ["--image", seg["image_path"]]
    subprocess.run(cmd, check=True)
    seg["clip_path"] = out
    return seg

@register("title", "cold_open")
def render_title(seg, ctx):
    out = os.path.join(BOT, "output", f"seg_{seg['id']}.mp4")
    spec = seg.get("spec", {})
    cmd = ["python3", os.path.join(BOT, "newscast", "build_title.py"), "--out", out]
    for k in ("headline", "kicker", "bg"):
        if spec.get(k):
            cmd += [f"--{k}", spec[k]]
    if seg.get("wav_path"):
        cmd += ["--audio", seg["wav_path"]]
    subprocess.run(cmd, check=True)
    seg["clip_path"] = out
    return seg

@register("narration")
def render_narration(seg, ctx):
    # narration-over-still (Ken Burns) — the docuseries default; reuse docupipe's renderer if present.
    out = os.path.join(BOT, "output", f"seg_{seg['id']}.mp4")
    kb = os.path.join(BOT, "docupipe", "src", "docupipe", "render", "kenburns.py")
    script = kb if os.path.exists(kb) else os.path.join(BOT, "newscast", "build_vo_broll.py")
    subprocess.run(["python3", script, "--audio", seg["wav_path"],
                    "--image", seg.get("image_path", ""), "--out", out], check=True)
    seg["clip_path"] = out
    return seg
