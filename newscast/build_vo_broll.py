"""VO + full-screen b-roll segment (no anchor). Full-frame story footage (image -> slow Ken-Burns, or a
video) under the anchor VO, with the NNS graphics package (bug + headline bar + ticker). Pure ffmpeg/PIL,
no GPU. Duration = VO length.
  python3 build_vo_broll.py --broll img_or_video --audio vo.wav --headline "..." [--kicker BREAKING]
                            [--music news_bed.mp3] --out out.mp4
Requires the gfx elements (newscast/broadcast_gfx.py already generated output/gfx_*). Falls back to the
branded gradient if no --broll.
"""
import sys, os, subprocess, json
from PIL import Image, ImageDraw, ImageFont
import numpy as np

def arg(f, d=None): return sys.argv[sys.argv.index(f)+1] if f in sys.argv else d
BROLL = arg("--broll", "")
AUDIO = arg("--audio")
OUT = arg("--out", "output/vo.mp4")
MUSIC = arg("--music", "")
pkg = json.load(open("output/news_package.json")) if os.path.exists("output/news_package.json") else {}
HEAD = arg("--headline", pkg.get("headline", "BREAKING NEWS"))
KICK = arg("--kicker", pkg.get("kicker", "BREAKING"))
W, H = 2560, 1440
RED = (200, 20, 46, 255)
def probe(p, k): return subprocess.run(["ffprobe","-v","error","-show_entries",k,"-of","csv=p=0",p],
                                       capture_output=True, text=True).stdout.strip()
DUR = float(probe(AUDIO, "format=duration") or "10")
fps = 25; nf = int(DUR*fps)

# headline bar (bottom, above ticker): kicker tab + headline
def Fnt(sz, bold=True): return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else ""), sz)
bar = Image.new("RGBA", (W, 300), (0,0,0,0)); d = ImageDraw.Draw(bar)
d.rectangle((0, 120, W, 300), fill=(10,16,32,225))          # translucent band
d.rectangle((0, 120, W, 128), fill=RED)
kf = Fnt(46); kw = d.textlength(KICK, font=kf)
d.rectangle((90, 150, 90+kw+56, 214), fill=RED); d.text((118, 160), KICK, font=kf, fill=(255,255,255,255))
hf = Fnt(84); d.text((90, 224), HEAD.upper()[:46], font=hf, fill=(255,255,255,255))
os.makedirs("output", exist_ok=True); bar.save("output/_vo_headbar.png")

# base b-roll layer -> full-frame video
if BROLL and os.path.exists(BROLL) and BROLL.lower().endswith((".mp4",".mov",".mkv",".webm")):
    base = ["-stream_loop","-1","-i",BROLL]
    vbase = f"[0:v]scale={W}:{H}:force_original_aspect_ratio=increase,crop={W}:{H},trim=0:{DUR},setpts=PTS-STARTPTS"
elif BROLL and os.path.exists(BROLL):
    base = ["-loop","1","-t",str(DUR),"-i",BROLL]
    vbase = (f"[0:v]scale=2816:1584:force_original_aspect_ratio=increase,crop=2816:1584,"
             f"zoompan=z='min(zoom+0.0005,1.10)':d={nf}:s={W}x{H}:fps={fps}")
else:                                                         # branded gradient fallback
    g = np.zeros((H,W,3),np.uint8); topc,botc=np.array([10,16,32]),np.array([22,32,60])
    for y in range(H): g[y]=(topc+(botc-topc)*(y/H)).astype(np.uint8)
    Image.fromarray(g).save("output/_vo_bg.png")
    base = ["-loop","1","-t",str(DUR),"-i","output/_vo_bg.png"]
    vbase = f"[0:v]scale={W}:{H}"

# darken slightly for text legibility
vbase += ",eq=brightness=-0.06,format=yuva420p[bg];"
G = "output"
# inputs: 0=broll, 1=audio, 2=bug, 3=ticker_bg, 4=ticker_text, 5=ticker_fg, 6=headbar, [7=music]
TW = probe(G+"/gfx_ticker_text.png", "stream=width") or "4000"
fc = (vbase +
      f"[bg][2:v]overlay=W-overlay_w-34:34[a];" +
      f"[a][3:v]overlay=0:H-92[b];" +
      f"[b][4:v]overlay=x='250-mod(t*190\\,{TW})':y=H-92[c];" +
      f"[c][5:v]overlay=0:H-92[d];" +
      f"[d][6:v]overlay=0:H-92-overlay_h:enable='gte(t,0.4)'[v]")   # headline band sits just above the ticker
cmd = ["ffmpeg","-y"] + base + ["-i",AUDIO,
       "-i",G+"/gfx_bug.png","-i",G+"/gfx_ticker_bg.png","-i",G+"/gfx_ticker_text.png",
       "-i",G+"/gfx_ticker_fg.png","-i","output/_vo_headbar.png"]
amap = "[1:a]aformat=fltp:44100:stereo,volume=1.0[vo]"
if MUSIC and os.path.exists(MUSIC):
    cmd += ["-stream_loop","-1","-i",MUSIC]
    amap = "[1:a]aformat=fltp:44100:stereo,volume=1.0[vv];[7:a]aformat=fltp:44100:stereo,volume=0.08[mm];[vv][mm]amix=inputs=2:duration=first[vo]"
cmd += ["-filter_complex", fc+";"+amap, "-map","[v]","-map","[vo]",
        "-c:v","libx264","-pix_fmt","yuv420p","-crf","18","-r",str(fps),"-c:a","aac","-b:a","192k","-shortest",OUT]
subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for f in ("output/_vo_headbar.png","output/_vo_bg.png"):
    try: os.remove(f)
    except Exception: pass
print("VO_BROLL_OK ->", OUT)
