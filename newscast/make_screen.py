"""Generate the on-set video-wall story graphic (1920x1080) from output/news_package.json.
NNS palette (navy + red). NO logo here — the network logo appears ONLY in the corner bug.
The renderer flips this horizontally (screen plane faces away from camera) -> writes screen_graphic_f.png.
  python3 make_screen.py            # -> output/screen_graphic.png (+ _f flipped)
Run in mp-extract (PIL+numpy+cv2).
"""
import json
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np, cv2
B = "/io"
pkg = json.load(open(B + "/output/news_package.json"))
W, Hh = 1920, 1080
RED = (200, 20, 46)
top = np.array([10, 16, 32]); bot = np.array([24, 34, 64])     # navy gradient
g = np.zeros((Hh, W, 3), np.uint8)
for y in range(Hh): g[y] = (top + (bot - top) * (y / Hh)).astype(np.uint8)
im = Image.fromarray(g); d = ImageDraw.Draw(im, "RGBA")
# node motif
rng = np.random.default_rng(7)
pts = [(int(rng.uniform(0.05, 0.95) * W), int(rng.uniform(0.12, 0.95) * Hh)) for _ in range(26)]
for i, (x, y) in enumerate(pts):
    for x2, y2 in pts[i + 1:]:
        if (x - x2) ** 2 + (y - y2) ** 2 < (W * 0.16) ** 2:
            d.line((x, y, x2, y2), fill=(200, 70, 90, 40), width=2)
for x, y in pts: d.ellipse((x - 6, y - 6, x + 6, y + 6), fill=(210, 90, 110, 150))
im = im.filter(ImageFilter.GaussianBlur(0.6)); d = ImageDraw.Draw(im, "RGBA")
def F(sz, bold=True): return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf" % ("-Bold" if bold else ""), sz)
kick = pkg["kicker"]; kf = F(46); kw = d.textlength(kick, font=kf)
d.rectangle((90, 150, 90 + kw + 60, 222), fill=RED + (255,)); d.text((120, 166), kick, font=kf, fill=(255, 255, 255, 255))
hl = pkg["headline"].upper(); hf = F(110); words = hl.split(); lines = []; cur = ""
for w in words:
    if d.textlength((cur + " " + w).strip(), font=hf) < W - 180: cur = (cur + " " + w).strip()
    else: lines.append(cur); cur = w
lines.append(cur); y = 290
for ln in lines: d.text((90, y), ln, font=hf, fill=(255, 255, 255, 255)); y += 128
sub = pkg.get("subhead") or pkg.get("screen_label", "")
d.text((92, y + 12), sub, font=F(46, False), fill=(235, 180, 190, 255))
d.rectangle((92, y + 92, 652, y + 100), fill=RED + (255,))
im.convert("RGB").save(B + "/output/screen_graphic.png")
cv2.imwrite(B + "/output/screen_graphic_f.png", cv2.flip(cv2.imread(B + "/output/screen_graphic.png"), 1))
print("SCREEN_OK (NNS, no logo) -> screen_graphic.png + _f")
