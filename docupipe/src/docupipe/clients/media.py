"""ffmpeg documentary assembler: Ken-Burns stills timed to narration, title cards,
concat, ducked music bed, loudness master. All craft defaults from config."""
import math
import os
import subprocess
from .. import config, cache

W, H, FPS = config.W, config.H, config.FPS


def _run(args):
    r = subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"] + args,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("ffmpeg failed:\n" + r.stderr[-1500:])


def _esc(t: str) -> str:
    return t.replace("\\", "\\\\").replace(":", "\\:").replace("'", "’").replace("%", "\\%")


def _wrap_lines(text: str, width=40):
    """Word-wrap to a list of lines of ~width chars."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return lines or [""]


def _stack(lines, size, color, font, line_h, ycenter="(h)/2", extra=""):
    """Render wrapped lines as SEPARATE centered drawtext filters (no embedded newline
    chars, which some ffmpeg/font combos render as tofu boxes). Vertically centered block."""
    n = len(lines)
    parts = []
    for i, ln in enumerate(lines):
        # y = block_top + i*line_h ; block_top = ycenter - n*line_h/2
        y = f"({ycenter})-{n}*{line_h}/2+{i}*{line_h}"
        parts.append(f"drawtext=text='{_esc(ln)}':fontcolor={color}:fontsize={size}:"
                     f"x=(w-text_w)/2:y={y}:fontfile={font}{(':' + extra) if extra else ''}")
    return ",".join(parts)


_SERIF_B = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf"
_SERIF_I = "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf"
_SERIF = "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"
_SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
# Distribution-provided fonts and generated backgrounds keep publication assets provenance-safe.
_PERIOD_SERIF = _SERIF
_PERIOD_SERIF_I = _SERIF_I


def normalize_image(image: str) -> str:
    """Transcode any input (gif/tiff/cmyk/png/jpg) to a clean single-frame RGB PNG.
    Avoids demuxer quirks (e.g. animated-GIF + -loop) and color-space issues."""
    png = os.path.splitext(image)[0] + ".norm.png"
    if cache.have(png):
        return png
    _run(["-i", image, "-frames:v", "1", "-vf", "format=rgb24", png])
    return png


# ── Archival visual language ─────────────────────────────────────────
# Unifying grade applied to EVERY still so disparate sources (B&W newspaper, sepia
# photo, even a stray modern color shot) read as one intentional vintage film.
GRADE = ("colorchannelmixer=.393:.769:.189:0:.349:.686:.168:0:.272:.534:.131,"  # sepia
         "curves=preset=darker,eq=contrast=1.08:brightness=-0.02:saturation=0.85,"
         "noise=alls=7:allf=t+u,vignette=PI/5")


def _zoompan(shot_type, frames, zoom_in):
    """Shot-grammar motion. z/x/y expressions for zoompan (operates on the 2x canvas)."""
    inc, zmax = config.KENBURNS_ZOOM_PER_FRAME, config.KENBURNS_MAX_ZOOM
    if shot_type == "map":                       # slow lateral pan, minimal zoom
        return ("1.15", f"'(iw-iw/zoom)*on/{frames}'", "'ih/2-(ih/zoom/2)'")
    if shot_type in ("newspaper", "document"):   # push IN on the headline (top)
        return (f"'min(zoom+{inc},{zmax})'", "'iw/2-(iw/zoom/2)'", "'0'")
    # photo/portrait: gentle zoom, focal bias slightly above center (faces/subjects)
    z = f"'min(zoom+{inc},{zmax})'" if zoom_in else f"'if(eq(on,0),{zmax},max(zoom-{inc},1.0))'"
    return (z, "'iw/2-(iw/zoom/2)'", "'ih/2.35-(ih/zoom/2)'")


def still_segment(image: str, narration_wav: str, dur: float, out: str,
                  shot_type="photo", zoom_in=True):
    """Treated archival still -> motion clip with VO. Pipeline in one ffmpeg pass:
    grade(sepia+grain+vignette) -> blurred-cover 16:9 compose (no stretch/bars) ->
    shot-grammar Ken-Burns. Anti-jitter: compose at 2x then zoompan down to target."""
    if cache.have(out):
        return out
    image = normalize_image(image)
    frames = max(1, int(math.ceil(dur * FPS)))
    z, x, y = _zoompan(shot_type, frames, zoom_in)
    BW, BH = W * 2, H * 2
    fc = (
        f"[0:v]{GRADE}[g];[g]split[bg][fg];"
        f"[bg]scale={BW}:{BH}:force_original_aspect_ratio=increase,crop={BW}:{BH},"
        f"boxblur=28:2,eq=brightness=-0.32[bgb];"
        f"[fg]scale={BW}:{BH}:force_original_aspect_ratio=decrease[fgf];"
        f"[bgb][fgf]overlay=(W-w)/2:(H-h)/2[c];"
        f"[c]zoompan=z={z}:d={frames}:x={x}:y={y}:s={W}x{H}:fps={FPS},format=yuv420p[v]")
    _run(["-loop", "1", "-i", image, "-i", narration_wav, "-filter_complex", fc,
          "-map", "[v]", "-map", "1:a", "-t", f"{dur:.3f}",
          "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
          "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-shortest", out])
    return out


# backwards-compat alias
def ken_burns_segment(image, narration_wav, dur, out, zoom_in=True):
    return still_segment(image, narration_wav, dur, out, "photo", zoom_in)


def title_card(text: str, dur: float, out: str, subtitle: str = "", narration_wav=None):
    """Period masthead title: dark serif on generated parchment, framed by an
    engraved double rule, finished with the archival grain + vignette (not a black slide)."""
    if cache.have(out):
        return out
    tlines = [ln.upper() for ln in _wrap_lines(text, 24)]
    fs = 100 if len(tlines) <= 1 else (78 if len(tlines) == 2 else 58)
    lh = int(fs * 1.28)
    ink = "0x241a10"
    yc = "(h)/2-10"
    draw = _stack(tlines, fs, ink, _PERIOD_SERIF, lh, ycenter=yc)
    # engraved double rule under the title (masthead), spanning the centre third
    block_h = lh * len(tlines)
    ry = f"(ih)/2-10+{block_h // 2 + 46}"   # drawbox: iw/ih are the FRAME dims (w/h = box dims)
    draw += (f",drawbox=x=(iw-560)/2:y={ry}:w=560:h=3:color={ink}@0.85:t=fill"
             f",drawbox=x=(iw-560)/2:y={ry}+9:w=560:h=1:color={ink}@0.6:t=fill")
    if subtitle:
        draw += "," + _stack(_wrap_lines(subtitle, 46), 30, ink + "@0.8", _PERIOD_SERIF_I, 44,
                             ycenter=f"(h)/2-10+{block_h // 2 + 90}")
    # parchment bg scaled to cover, slightly dimmed for contrast, then the archival finish
    fc = (f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
          f"eq=brightness=-0.04:contrast=1.02[bg];"
          f"[bg]{draw},noise=alls=7:allf=t+u,vignette=PI/5,format=yuv420p[v]")
    args = ["-f", "lavfi", "-i", f"color=c=0xefe4cb:s={W}x{H}:r={FPS}"]
    if narration_wav:                       # real VO
        args += ["-i", narration_wav]
    else:                                   # silent track so concat keeps a uniform audio stream
        args += ["-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=48000"]
    args += ["-filter_complex", fc, "-map", "[v]", "-map", "1:a",
             "-t", f"{dur:.3f}", "-r", str(FPS), "-c:v", "libx264", "-crf", "18",
             "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
             "-shortest", out]
    _run(args)
    return out


def quote_card(text, attribution, dur, out, narration_wav=None):
    """A signature typographic card for a famous quote/letter — graded dark parchment,
    large serif italic, slow drift. Used e.g. for the Axeman's jazz letter."""
    if cache.have(out):
        return out
    qlines = _wrap_lines(text, 38)
    qlines[0] = "“" + qlines[0]
    qlines[-1] = qlines[-1] + "”"
    body = _stack(qlines, 52, "0xEDE3C8", _SERIF_I, 74, ycenter="(h)/2-20")
    if attribution:
        body += (f",drawtext=text='— {_esc(attribution)}':fontcolor=0xB59A6A:fontsize=30:"
                 f"x=(w-text_w)/2:y=h-h/4:fontfile={_SERIF}")
    bg = (f"color=c=0x140f0a:s={W}x{H}:r={FPS}:d={dur:.3f},"
          f"vignette=PI/4,noise=alls=6:allf=t")
    args = ["-f", "lavfi", "-i", bg]
    if narration_wav:
        args += ["-i", narration_wav]
    else:
        args += ["-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=48000"]
    args += ["-vf", body, "-t", f"{dur:.3f}", "-c:v", "libx264", "-crf", "18",
             "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
             "-shortest", out]
    _run(args)
    return out


def graphic_segment(frames_dir: str, narration_wav: str, dur: float, out: str):
    """Assemble docgfx PNG frames (frames/f%05d.png, already vintage-toned) into a clip, apply the
    SHARED archival grain + vignette (the unifier — same grain as the stills), and mux narration.
    No sepia re-tone (frames are already toned) — just grain/vignette/contrast so it matches."""
    if cache.have(out):
        return out
    vf = ("noise=alls=17:allf=t+u,eq=contrast=1.06:saturation=0.96,"
          "vignette=PI/6,format=yuv420p")
    _run(["-framerate", str(FPS), "-i", os.path.join(frames_dir, "f%05d.png"),
          "-i", narration_wav, "-vf", vf, "-t", f"{dur:.3f}",
          "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
          "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-shortest", out])
    return out


def slide_segment(image: str, narration_wav: str, dur: float, out: str, zoom_in=True):
    """Explainer slide (Presenton page) full-frame with narrator VO + gentle push-in. NO sepia
    grade — slides are modern graphics; just fit to frame on a dark bg and a slow zoom."""
    if cache.have(out):
        return out
    image = normalize_image(image)
    frames = max(1, int(math.ceil(dur * FPS)))
    z = f"min(zoom+{config.KENBURNS_ZOOM_PER_FRAME},1.12)" if zoom_in else "1.0"
    fc = (f"color=c=0x111318:s={W}x{H}:r={FPS}[bg];"
          f"[0:v]scale={W}:{H}:force_original_aspect_ratio=decrease[fg];"
          f"[bg][fg]overlay=(W-w)/2:(H-h)/2,scale={W*2}:{H*2},"
          f"zoompan=z='{z}':d={frames}:x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':"
          f"s={W}x{H}:fps={FPS},format=yuv420p[v]")
    _run(["-loop", "1", "-i", image, "-i", narration_wav, "-filter_complex", fc,
          "-map", "[v]", "-map", "1:a", "-t", f"{dur:.3f}", "-c:v", "libx264", "-preset", "medium",
          "-crf", "18", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
          "-shortest", out])
    return out


def interview_segment(talking_mp4: str, name: str, title: str, out: str, dur=None):
    """Take a talking-head clip, fit to 1080p, add a lower-third chyron (name + title), light
    contemporary grade (NOT sepia — interviews read as present-day). Keeps the clip's audio."""
    if cache.have(out):
        return out
    y0 = H - 200
    lt = (f"drawbox=x=0:y={y0}:w=iw:h=150:color=black@0.55:t=fill,"
          f"drawbox=x=90:y={y0 + 24}:w=6:h=100:color=0x7a3323:t=fill,"   # oxblood, not bright red
          f"drawtext=text='{_esc(name)}':fontcolor=0xF2E9D8:fontsize=46:x=130:y={y0 + 34}:"
          f"fontfile={_PERIOD_SERIF},"
          f"drawtext=text='{_esc(title)}':fontcolor=0xC9B79A:fontsize=28:x=132:y={y0 + 92}:"
          f"fontfile={_PERIOD_SERIF_I}")
    # documentary interviews read present-day, but warm/filmic — not sterile broadcast: warm the
    # whites, lift shadows slightly, desaturate the harsh blues, add a touch of film grain.
    vf = (f"scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
          f"eq=saturation=0.82:contrast=1.05:gamma_b=0.94:gamma_r=1.05,"
          f"noise=alls=5:allf=t,{lt},format=yuv420p")
    args = ["-i", talking_mp4, "-vf", vf]
    if dur:
        args += ["-t", f"{dur:.3f}"]
    args += ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
             "-c:a", "aac", "-b:a", "192k", "-ar", "48000", out]
    _run(args)
    return out


def concat(clips, out):
    """Concatenate same-spec clips via the concat demuxer (re-encode for safety)."""
    if not clips:
        raise ValueError("no clips")
    lst = out + ".txt"
    with open(lst, "w") as f:
        for c in clips:
            f.write(f"file '{os.path.abspath(c)}'\n")
    # force constant 30fps + CFR: clips can differ in framerate (e.g. the 25fps talking-head
    # or an image-loop card), and the concat demuxer + re-encode otherwise drifts timestamps.
    _run(["-f", "concat", "-safe", "0", "-i", lst, "-r", str(FPS), "-fps_mode", "cfr",
          "-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
          "-c:a", "aac", "-b:a", "192k", "-ar", "48000", out])
    os.remove(lst)
    return out


def score_bed(cues, total_dur: float, out: str) -> str:
    """Build an EVOLVING music bed that changes with the episode's arc. `cues` = ordered list of
    (cue_path, start_frac, end_frac); each cue is looped to fill its time-span and the spans are
    concatenated with short fades so the score shifts (tension -> dread -> elegy) at the act turns.
    Returns out (a wav), or "" if no usable cue. add_music_and_master then ducks it under the VO."""
    cues = [(p, a, b) for (p, a, b) in cues if p and os.path.exists(p) and b > a]
    if not cues:
        return ""
    tmpdir = os.path.dirname(out)
    spans = []
    for i, (p, a, b) in enumerate(cues):
        L = max(1.0, (b - a) * total_dur)
        sp = os.path.join(tmpdir, f"_span{i}.wav")
        # loop the cue to fill the span, fade in/out so hard joins between cues are smooth
        _run(["-stream_loop", "-1", "-i", p, "-t", f"{L:.3f}",
              "-af", f"afade=t=in:st=0:d=1.2,afade=t=out:st={max(0.0, L-1.5):.3f}:d=1.5",
              "-ar", "48000", "-ac", "2", sp])
        spans.append(sp)
    lst = out + ".txt"
    with open(lst, "w") as f:
        for sp in spans:
            f.write(f"file '{os.path.abspath(sp)}'\n")
    _run(["-f", "concat", "-safe", "0", "-i", lst, "-ar", "48000", "-ac", "2", out])
    os.remove(lst)
    for sp in spans:
        try:
            os.remove(sp)
        except OSError:
            pass
    return out


def add_music_and_master(video_in: str, music_wav: str, out: str):
    """Mix the video's narration with a music bed ducked under it, then loudnorm to target."""
    duck = config.MUSIC_DUCK_DB
    lufs = config.LOUDNESS_LUFS
    if music_wav and cache.have(music_wav):
        # [1] music, keyed (ducked) by [0] narration via sidechaincompress; then amix; loudnorm
        fc = (f"[0:a]asplit=2[nar][key];"
              f"[1:a]aloop=loop=-1:size=2e9,volume={duck}dB[mus];"
              f"[mus][key]sidechaincompress=threshold=0.03:ratio=8:attack=20:release=400[ducked];"
              f"[nar][ducked]amix=inputs=2:duration=first:dropout_transition=0[mix];"
              f"[mix]loudnorm=I={lufs}:TP=-1.5:LRA=11[a]")
        _run(["-i", video_in, "-i", music_wav, "-filter_complex", fc,
              "-map", "0:v", "-map", "[a]", "-c:v", "copy",
              "-c:a", "aac", "-b:a", "192k", "-ar", "48000", out])
    else:
        _run(["-i", video_in, "-af", f"loudnorm=I={lufs}:TP=-1.5:LRA=11",
              "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-ar", "48000", out])
    return out
