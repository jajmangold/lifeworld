"""Anchor/reporter renderer — wraps make_character.sh (the locked tail: render_character -> graft HAAR hair
-> swap -> MuseTalk -> settle -> FlashVSR -> broadcast_finish) with catalog resolution. Resolves the talent's
BlenderKit character (via the wardrobe), face-swap source, voice-driven wav, HAAR hair groom, and the set's
HDRI pano, then shells the working script. Handles kinds: fullscreen_anchor, anchor_wall, ots, reporter_pkg.
"""
import os, json, subprocess
from .registry import register
from ..services import hair as hair_svc

BOT = "/srv/nvme-data/containers/live/studio"

def _wardrobe():
    return json.load(open(os.path.join(BOT, "catalog", "wardrobe.json"))).get("characters", {})

def _char_blend(avatar_ref):
    # avatar_ref like "wardrobe:business_female" -> the .blend path
    if avatar_ref and avatar_ref.startswith("wardrobe:"):
        w = _wardrobe().get(avatar_ref.split(":", 1)[1])
        if w:
            return os.path.join(BOT, w["blend"])
    return None

@register("fullscreen_anchor", "anchor_wall", "ots", "reporter_pkg")
def render_anchor(seg, ctx):
    talent = ctx["talent"][seg["talent_ref"]]
    setspec = ctx.get("sets", {}).get(seg.get("set_ref"), {})
    wav = seg["wav_path"]                                  # produced upstream by the tts service
    out = os.path.join(BOT, "output", f"seg_{seg['id']}.mp4")
    fmt = {"anchor_wall": "anchor_wall"}.get(seg["kind"], "fullscreen_anchor" if seg["kind"] != "ots" else "ots")

    env = os.environ.copy()
    env["NEWS_CAMSIDE"] = "-1"
    h = hair_svc.resolve(talent.get("hair"))
    if h:
        env["NEWS_HAIR_GROOM"] = h["groom_remote"]; env["NEWS_HAIR_MEL"] = str(h["melanin"])
        if h.get("face_slim"):
            env["NEWS_FACE_SLIM"] = str(h["face_slim"])

    char = _char_blend(talent.get("avatar"))
    face = os.path.basename(talent["face"])               # make_anchor: --face is relative to output/ (=/o)
    if char:                                              # BlenderKit character path (make_character)
        cmd = ["bash", f"{BOT}/make_character.sh", "--char", char, "--audio", wav, "--out", out,
               "--face", face, "--premium"]
        pano = setspec.get("env")
        if pano:
            cmd += ["--pano", pano]
    else:                                                 # legacy viverse avatar (make_anchor --glb/default)
        cmd = ["bash", f"{BOT}/make_anchor.sh", "--audio", wav, "--out", out, "--face", face,
               "--format", fmt, "--premium"]
        if talent.get("avatar"):
            cmd += ["--glb", talent["avatar"]]
        if setspec.get("env"):
            cmd += ["--pano", setspec["env"]]

    print(f"[render_anchor] seg {seg['id']} talent={seg['talent_ref']} kind={seg['kind']} hair={bool(h)}", flush=True)
    subprocess.run(cmd, env=env, check=True)
    seg["clip_path"] = out
    return seg
