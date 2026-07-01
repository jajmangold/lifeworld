"""Multi-aspect delivery: derive a 9:16 vertical SHORT (YouTube Shorts / TikTok) from a 16:9 clean master
— cheap ffmpeg reframe, no GPU. The master already paid the render/swap/muse/flashvsr cost; this just
recomposes: blurred cover background + the master in an upper video window + NNS bug + a bold headline
caption panel (verticals need big on-screen text, no ticker). Part of S5 multi-aspect finish.
  python3 reframe_vertical.py --master clip.mp4 --headline "..." [--kicker BREAKING] [--music bed.mp3]
                             [--bug output/gfx_bug.png] --out short.mp4
"""
import sys, os, subprocess, textwrap

def arg(f, d=None): return sys.argv[sys.argv.index(f)+1] if f in sys.argv else d
M = arg("--master"); OUT = arg("--out", "output/short.mp4")
HEAD = (arg("--headline", "") or "").upper(); KICK = (arg("--kicker", "") or "").upper()
MUSIC = arg("--music", ""); BUG = arg("--bug", "output/gfx_bug.png")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
W, H = 1080, 1920
VW, VY = 1080, 420                       # video window: full width, upper-middle
NAVY = "0x0c1630"; RED = "0xc8142e"

def esc(t): return t.replace("\\", "").replace(":", "\\:").replace("'", "’")

# pre-wrap headline to <=3 lines (~22 chars) — drawtext has no auto-wrap
lines = textwrap.wrap(HEAD, width=22)[:3]
CY = 1074                                # caption panel top (below the ~608px-tall video window)
PANEL_H = 130 + len(lines) * 76          # grow the panel to fit the wrapped headline

fc = (f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},"
      f"boxblur=26:3,eq=brightness=-0.30:saturation=0.75[bg];"
      f"[0:v]scale={VW}:-2[fg];"
      f"[bg][fg]overlay=(W-w)/2:{VY}[v0];")
i = 1
inputs = ["-i", M]
if os.path.isfile(BUG):
    inputs += ["-i", BUG]; fc += f"[{i}:v]scale=180:-1[bug];[v0][bug]overlay=W-w-46:52[v1];"; prev = "v1"; i += 1
else:
    prev = "v0"
# caption panel + kicker tab + headline lines (drawn on the composited stream)
draw = f"[{prev}]drawbox=x=0:y={CY}:w={W}:h={PANEL_H}:color={NAVY}@0.92:t=fill"
draw += f",drawbox=x=0:y={CY}:w={W}:h=8:color={RED}@1:t=fill"
if KICK:
    kw = 34 + len(KICK) * 26
    draw += (f",drawbox=x=54:y={CY+34}:w={kw}:h=64:color={RED}@1:t=fill"
             f",drawtext=fontfile={FONT}:text='{esc(KICK)}':fontcolor=white:fontsize=40:x=72:y={CY+46}")
for n, ln in enumerate(lines):
    draw += (f",drawtext=fontfile={FONT}:text='{esc(ln)}':fontcolor=white:fontsize=62:"
             f"x=56:y={CY+118+n*76}")
fc += draw + "[v]"

amap, aopt = "", []
cmd = ["ffmpeg", "-y"] + inputs
if MUSIC and os.path.isfile(MUSIC):
    cmd += ["-stream_loop", "-1", "-i", MUSIC]
    fc += (f";[0:a]aformat=fltp:44100:stereo,volume=1.0[vv];[{i}:a]aformat=fltp:44100:stereo,volume=0.07[mm];"
           f"[vv][mm]amix=inputs=2:duration=first[a]")
    aopt = ["-map", "[a]"]
else:
    aopt = ["-map", "0:a?"]
cmd += ["-filter_complex", fc, "-map", "[v]"] + aopt + [
    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "19", "-r", "30",
    "-c:a", "aac", "-b:a", "192k", "-shortest", OUT]
subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
print("VERTICAL_OK ->", OUT, f"({W}x{H})")
