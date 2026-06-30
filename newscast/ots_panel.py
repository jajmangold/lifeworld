#!/usr/bin/env python3
"""Build an over-the-shoulder (OTS) 'TV screen bubble' panel PNG (RGBA): rounded-rect screen with the
story image, a thin accent frame, a glossy highlight, a bottom label bar, and an outer drop shadow.
  python ots_panel.py <story_image> <label_text> <out.png> [--w 430] [--accent 40,120,230]
"""
import sys
from PIL import Image, ImageDraw, ImageFont, ImageFilter

def arg(flag, d):
    return sys.argv[sys.argv.index(flag)+1] if flag in sys.argv else d

IMG, LABEL, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
W = int(arg("--w", "430"))
accent = tuple(int(x) for x in arg("--accent", "40,120,230").split(","))

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
pad = 26            # space for shadow
rad = 22           # corner radius
frame = 4          # accent frame thickness
barh = 46          # label bar height
imw = W - 2*pad
imh = int(imw * 0.62)                 # 16:10-ish screen
screen_h = imh + barh
fullw, fullh = W, screen_h + 2*pad

canvas = Image.new("RGBA", (fullw, fullh), (0, 0, 0, 0))
draw = ImageDraw.Draw(canvas)

# ---- drop shadow ----
sh = Image.new("RGBA", (fullw, fullh), (0, 0, 0, 0))
sd = ImageDraw.Draw(sh)
sd.rounded_rectangle([pad+6, pad+8, pad+imw+6, pad+screen_h+8], rad, fill=(0, 0, 0, 150))
sh = sh.filter(ImageFilter.GaussianBlur(10))
canvas.alpha_composite(sh)

# ---- panel body (dark) ----
x0, y0, x1, y1 = pad, pad, pad+imw, pad+screen_h
draw.rounded_rectangle([x0, y0, x1, y1], rad, fill=(14, 18, 26, 255))

# ---- story image, cropped to the screen area, rounded ----
story = Image.open(IMG).convert("RGB")
sw, sh2 = story.size
scale = max(imw/sw, imh/sh2)
story = story.resize((int(sw*scale), int(sh2*scale)))
left = (story.width - imw)//2; top = (story.height - imh)//2
story = story.crop((left, top, left+imw, top+imh))
mask = Image.new("L", (imw, imh), 0)
md = ImageDraw.Draw(mask)
md.rounded_rectangle([0, 0, imw-1, imh-1], rad-4, fill=255)
md.rectangle([0, imh-rad, imw-1, imh-1], fill=255)   # square bottom (meets label bar)
canvas.paste(story, (x0, y0), mask)

# ---- label bar ----
draw = ImageDraw.Draw(canvas)
by0 = y0 + imh
draw.rectangle([x0, by0, x1, y1-1], fill=accent + (255,))
draw.rounded_rectangle([x0, by0-rad, x1, y1], rad, fill=accent + (255,))
draw.rectangle([x0, by0-rad, x1, by0+4], fill=accent + (255,))
# label text (auto-fit)
txt = LABEL.upper()
fs = 24
while fs > 12:
    f = ImageFont.truetype(FONT, fs)
    if draw.textlength(txt, font=f) <= imw-28: break
    fs -= 1
f = ImageFont.truetype(FONT, fs)
tw = draw.textlength(txt, font=f)
draw.text((x0 + (imw-tw)/2, by0 + (barh-fs)/2 - 2), txt, font=f, fill=(255, 255, 255, 255))

# ---- accent frame around the screen ----
draw.rounded_rectangle([x0, y0, x1, y1], rad, outline=accent + (255,), width=frame)
# inner hairline
draw.rounded_rectangle([x0+frame, y0+frame, x1-frame, y1-frame], rad-frame, outline=(255, 255, 255, 60), width=1)

# ---- subtle glossy highlight on the screen (top third) ----
gloss = Image.new("RGBA", (fullw, fullh), (0, 0, 0, 0))
gd = ImageDraw.Draw(gloss)
gd.rounded_rectangle([x0+frame, y0+frame, x1-frame, y0+int(imh*0.38)], rad-frame, fill=(255, 255, 255, 26))
gloss = gloss.filter(ImageFilter.GaussianBlur(6))
canvas.alpha_composite(gloss)

canvas.save(OUT)
print(f"OTS_OK {OUT} size={fullw}x{fullh} label='{txt}' fontsize={fs}")
