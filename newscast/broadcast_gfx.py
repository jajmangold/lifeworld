"""Generate the FFNN broadcast graphics-package elements (1440p) as RGBA PNGs:
  bug.png         corner network logo
  lower_third.png anchor name + title bar (slides in)
  ticker_bg.png   bottom ticker bar (FFNN tab + accent + time box bg)
  ticker_text.png long transparent strip of the scrolling headlines
Reads output/news_package.json. Run in mp-extract (PIL+numpy). Colors = FFNN blue.
  python3 broadcast_gfx.py
"""
import json, os
from PIL import Image, ImageDraw, ImageFont
B = "/io"
pkg = json.load(open(B + "/output/news_package.json"))
W, H = 2560, 1440
BLUE = (200, 20, 46, 255)      # NNS accent (red) — name kept for minimal churn
NAVY = (12, 20, 38, 255)
LT_NAVY = (14, 24, 46, 235)
WHITE = (255, 255, 255, 255)
LOGO = B + "/newscast/assets/nns_logo.png"
def F(sz, bold=True):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else ""), sz)
def newimg(w, h): return Image.new("RGBA", (w, h), (0, 0, 0, 0))

# ---- bug: NNS logo on a white rounded plate (logo PNG has a baked checkerboard bg + dark marks) ----
import numpy as np
logo = Image.open(LOGO).convert("RGB")
arr = np.array(logo).astype(np.int16)
# the file has a BAKED checkerboard (light/lt-gray pixels) instead of real alpha — erase it: any pixel
# whose darkest channel is light (>=190) is background/checkerboard -> set to pure white.
bgmask = arr.min(2) >= 190
arr[bgmask] = 255
arr = arr.astype(np.uint8)
content = np.where(~bgmask)                                   # autocrop to the real logo marks
y0, y1, x0, x1 = content[0].min(), content[0].max(), content[1].min(), content[1].max()
logo = Image.fromarray(arr).crop((x0 - 8, y0 - 8, x1 + 8, y1 + 8))
bw = 260; logo = logo.resize((bw, int(bw * logo.height / logo.width)))
pad = 22; plate_w, plate_h = bw + 2 * pad, logo.height + 2 * pad
bug = newimg(plate_w, plate_h)
d = ImageDraw.Draw(bug)
d.rounded_rectangle((0, 0, plate_w - 1, plate_h - 1), radius=18, fill=(255, 255, 255, 235), outline=BLUE, width=5)
bug.paste(logo, (pad, pad))
bug.save(B + "/output/gfx_bug.png")

# ---- lower third: kicker tab + anchor name + title ----
lt = newimg(1200, 230)
d = ImageDraw.Draw(lt)
# main bar
d.rectangle((0, 70, 980, 150), fill=LT_NAVY)
d.rectangle((0, 150, 980, 196), fill=(8, 18, 38, 230))
# blue accent edge
d.rectangle((0, 70, 12, 196), fill=BLUE)
# kicker tab
kf = F(34); kick = "NNS NEWSDESK"
kw = d.textlength(kick, font=kf)
d.rectangle((30, 24, 30 + kw + 44, 24 + 52), fill=BLUE)
d.text((52, 34), kick, font=kf, fill=WHITE)
# name + title
d.text((34, 80), pkg["anchor_name"].upper(), font=F(60), fill=WHITE)
d.text((36, 156), "NNS ANCHOR", font=F(30, False), fill=(235, 180, 190, 255))
lt.save(B + "/output/gfx_lower_third.png")

# ---- ticker bar: BG (navy bar) + FG (LIVE tab left, time box right) so text scrolls BEHIND the tabs ----
th = 92
bg = newimg(W, th); d = ImageDraw.Draw(bg)
d.rectangle((0, 0, W, th), fill=NAVY)
d.rectangle((0, 0, W, 5), fill=BLUE)                      # top accent line
bg.save(B + "/output/gfx_ticker_bg.png")
fg = newimg(W, th); d = ImageDraw.Draw(fg)
d.rectangle((0, 5, 250, th), fill=BLUE)                   # brand tab (not "LIVE" — these go to YouTube)
d.text((40, 22), "NNS", font=F(46), fill=WHITE)
fg.save(B + "/output/gfx_ticker_fg.png")

# ---- scrolling ticker text strip (transparent, long) ----
sep = "      ♦      "
line = sep.join(pkg["ticker"]) + sep
tf = F(40)
tmp = ImageDraw.Draw(newimg(10, 10))
lw = int(tmp.textlength(line, font=tf))
strip = newimg(lw + 80, th)
d = ImageDraw.Draw(strip)
d.text((0, 24), line, font=tf, fill=(225, 235, 255, 255))
strip.save(B + "/output/gfx_ticker_text.png")
print("GFX_OK bug+lower_third+ticker (ticker_text w=%d)" % (lw + 80))
