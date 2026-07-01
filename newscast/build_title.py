"""Title / bumper card segment (no anchor) — animated NNS branding card. Composes a 1440p card
(NNS logo + program title + subtitle, navy/red) then animates it (fade in, slow push, fade out) over a
music sting. Pure PIL+ffmpeg, no GPU.
  python3 build_title.py --title "EVENING REPORT" [--subtitle "..."] [--dur 5] [--music news_bed.mp3] --out out.mp4
"""
import sys, os, subprocess
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np

def arg(f, d=None):
    return sys.argv[sys.argv.index(f)+1] if f in sys.argv else d
BOT = "."   # run with cwd = repo root (or /io inside mp-extract)
TITLE = arg("--title", "EVENING REPORT")
SUB = arg("--subtitle", "")
DUR = float(arg("--dur", "5"))
MUSIC = arg("--music", "")
OUT = arg("--out", "output/title.mp4")
W, H = 2560, 1440
RED = (200, 20, 46, 255)
def F(sz, bold=True): return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else ""), sz)

# navy gradient + node motif (matches the on-set graphics)
top, bot = np.array([10, 16, 32]), np.array([22, 32, 60])
g = np.zeros((H, W, 3), np.uint8)
for y in range(H): g[y] = (top + (bot-top)*(y/H)).astype(np.uint8)
im = Image.fromarray(g); d = ImageDraw.Draw(im, "RGBA")
rng = np.random.default_rng(7)
pts = [(int(rng.uniform(0.03,0.97)*W), int(rng.uniform(0.03,0.97)*H)) for _ in range(30)]
for i,(x,y) in enumerate(pts):
    for x2,y2 in pts[i+1:]:
        if (x-x2)**2+(y-y2)**2 < (W*0.15)**2: d.line((x,y,x2,y2), fill=(200,70,90,32), width=2)
for x,y in pts: d.ellipse((x-5,y-5,x+5,y+5), fill=(210,90,110,120))
im = im.filter(ImageFilter.GaussianBlur(0.6)); d = ImageDraw.Draw(im, "RGBA")

# NNS logo (clean the baked checkerboard), centered upper
logo = Image.open(BOT+"/newscast/assets/nns_logo.png").convert("RGB"); arr = np.array(logo).astype(np.int16)
bg = arr.min(2) >= 190; arr[bg] = 255; arr = arr.astype(np.uint8)
c = np.where(~bg); y0,y1,x0,x1 = c[0].min(),c[0].max(),c[1].min(),c[1].max()
logo = Image.fromarray(arr).crop((x0-8,y0-8,x1+8,y1+8))
lw = 900; logo = logo.resize((lw, int(lw*logo.height/logo.width)))
plate_w, plate_h = lw+80, logo.height+70
plate = Image.new("RGBA", (plate_w, plate_h), (0,0,0,0)); pd = ImageDraw.Draw(plate)
pd.rounded_rectangle((0,0,plate_w-1,plate_h-1), radius=28, fill=(255,255,255,240), outline=RED, width=7)
plate.paste(logo, (40, 35))
im.paste(plate, ((W-plate_w)//2, 300), plate)
# title + accent + subtitle
tf = F(120); tw = d.textlength(TITLE, font=tf)
ty = 300 + plate_h + 90
d.text(((W-tw)//2, ty), TITLE, font=tf, fill=(255,255,255,255))
d.rectangle(((W-460)//2, ty+150, (W+460)//2, ty+158), fill=RED)
if SUB:
    sf = F(52, False); sw = d.textlength(SUB, font=sf)
    d.text(((W-sw)//2, ty+190), SUB, font=sf, fill=(235,180,190,255))
os.makedirs("output", exist_ok=True)
im.convert("RGB").save("output/_titlecard.png")

# animate: slow push-in (zoompan) + fade in/out, + music sting
fps = 25; nf = int(DUR*fps)
vf = (f"scale=2688:1512,zoompan=z='min(zoom+0.0006,1.06)':d={nf}:s={W}x{H}:fps={fps},"
      f"fade=t=in:st=0:d=0.5,fade=t=out:st={DUR-0.6}:d=0.6,format=yuv420p")
cmd = ["ffmpeg","-y","-loop","1","-t",str(DUR),"-i","output/_titlecard.png"]
if MUSIC and os.path.exists(MUSIC):
    cmd += ["-i",MUSIC,"-filter_complex",
            f"[0:v]{vf}[v];[1:a]atrim=0:{DUR},afade=t=in:st=0:d=0.3,afade=t=out:st={DUR-0.8}:d=0.8,volume=0.6[a]",
            "-map","[v]","-map","[a]","-c:a","aac","-b:a","192k","-shortest"]
else:
    cmd += ["-vf",vf,"-an"]
cmd += ["-c:v","libx264","-pix_fmt","yuv420p","-crf","18","-r",str(fps),OUT]
subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
os.remove("output/_titlecard.png")
print("TITLE_OK ->", OUT)
