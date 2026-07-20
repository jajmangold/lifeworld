"""Researcher talking-head via the NEWS pipeline (make_anchor.sh): render avatar -> face-swap the
researcher portrait -> MuseTalk lip-sync -> restore/premium. Same process the news segments use.
Heavy (~8-15 min/clip on rtx0); cached by (face,audio)."""
import os
import subprocess
from .. import config, cache

BOT = config.LEGACY_STUDIO_ROOT
BOT_OUT = os.path.join(BOT, "output")


def available() -> bool:
    if not BOT or not os.path.isfile(os.path.join(BOT, "make_anchor.sh")):
        return False
    try:
        out = subprocess.run(["docker", "ps", "--format", "{{.Names}}"],
                             capture_output=True, text=True, timeout=10).stdout
        return "swap-server" in out and "muse-server" in out
    except Exception:
        return False


def talking_head(portrait_path: str, wav_path: str, out_mp4: str, *, premium=False,
                 mood="serious", nod=1.1, bg=None) -> str:
    """Run make_anchor.sh with the researcher's face+voice. `bg` = a background plate image
    (composited behind the studio-lit subject via --bg) to replace the newsroom set.
    Returns out_mp4 or "" on failure."""
    ck = cache.path("anchor", cache.key("anc", portrait_path, wav_path, premium, bg or ""), "mp4")
    if cache.have(ck):
        return ck
    # stage face + audio into bot/output (make_anchor reads /o/<name> + output/<name>)
    face = os.path.join(BOT_OUT, "res_face_" + cache.key("f", portrait_path) + ".png")
    aud = os.path.join(BOT_OUT, "res_aud_" + cache.key("a", wav_path) + ".wav")
    cache.atomic_write_bytes(face, open(portrait_path, "rb").read())
    cache.atomic_write_bytes(aud, open(wav_path, "rb").read())
    dst = os.path.join(BOT_OUT, "res_out_" + cache.key("o", portrait_path, wav_path) + ".mp4")
    cmd = ["bash", "make_anchor.sh", "--audio", os.path.relpath(aud, BOT),
           "--face", os.path.basename(face), "--mood", mood, "--nod", str(nod),
           "--out", os.path.relpath(dst, BOT)]
    if bg and os.path.isfile(bg):
        cmd += ["--bg", os.path.relpath(bg, BOT)]
    if premium:
        cmd.append("--premium")
    try:
        r = subprocess.run(cmd, cwd=BOT, capture_output=True, text=True, timeout=2400)
        if r.returncode != 0 or not os.path.exists(dst):
            return ""
        cache.atomic_write_bytes(ck, open(dst, "rb").read())
        return ck
    except Exception:
        return ""
