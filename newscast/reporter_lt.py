"""Reporter lower-third PNG (name + location + 'NNS CORRESPONDENT'), NNS navy/red style, sized for a
2560x1440 frame. No 'LIVE' (these go to YouTube). Runs in mp-extract (PIL available).
  python3 reporter_lt.py --name "Elise Warren" --location "KANSAS CITY, KS" --out output/reporter_lt.png
"""
import sys
from PIL import Image, ImageDraw, ImageFont

def arg(f, d=""): return sys.argv[sys.argv.index(f)+1] if f in sys.argv else d
NAME = arg("--name", "NNS Correspondent").upper()
LOC = arg("--location", "").upper()
OUT = arg("--out", "output/reporter_lt.png")

WHITE = (255, 255, 255, 255)
RED = (200, 20, 46, 255)
NAVY = (12, 22, 48, 235)
SUB = (235, 180, 190, 255)
def F(sz, bold=True):
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else ""), sz)

lt = Image.new("RGBA", (1400, 260), (0, 0, 0, 0))
d = ImageDraw.Draw(lt)
d.rectangle((0, 78, 1120, 168), fill=NAVY)              # name bar
d.rectangle((0, 168, 1120, 220), fill=(8, 18, 38, 230)) # sub bar
d.rectangle((0, 78, 14, 220), fill=RED)                 # accent edge
# kicker tab
kf = F(38); kick = "ON THE SCENE"; kw = d.textlength(kick, font=kf)
d.rectangle((30, 22, 30 + kw + 48, 22 + 58), fill=RED)
d.text((54, 33), kick, font=kf, fill=WHITE)
# name + correspondent/location line
d.text((36, 90), NAME[:26], font=F(66), fill=WHITE)
sub = "NNS CORRESPONDENT" + (u"  ·  " + LOC if LOC else "")
d.text((38, 176), sub[:52], font=F(32, False), fill=SUB)
lt.save(OUT)
print("REPORTER_LT_OK ->", OUT)
