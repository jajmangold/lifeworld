#!/usr/bin/env python3
"""Generate a 4-band inner-mouth texture atlas for mouth_parts UVs (upper teeth, lower teeth,
tongue, palate/throat). Procedural but reads as real with the glossy material. Top of each band =
gum/lip edge, bottom = into the mouth. python gen_mouth_atlas.py <out.png>"""
import sys
import numpy as np
from PIL import Image

W, BH = 256, 128                     # band width, band height (4 bands -> 512 tall)
rng = np.random.default_rng(7)
bands = []


def vgrad(top, bot, h=BH):
    t = np.linspace(0, 1, h)[:, None, None]
    g = np.array(top, float)[None, None] * (1 - t) + np.array(bot, float)[None, None] * t  # (h,1,3)
    return np.repeat(g, W, axis=1)                       # (h,W,3)


def teeth(top_enamel, bot_enamel, n_teeth=9):
    img = vgrad(top_enamel, bot_enamel)
    x = np.arange(W)
    # per-tooth rounded shading: brighter center, darker toward each gap
    phase = (x % (W / n_teeth)) / (W / n_teeth)          # 0..1 across a tooth
    shade = 1.0 - 0.28 * (np.abs(phase - 0.5) * 2) ** 1.6   # bright mid, dark edges
    img *= shade[None, :, None]
    # dark separation lines between teeth
    for k in range(n_teeth + 1):
        xc = int(k * W / n_teeth)
        for dx, f in [(-1, .55), (0, .4), (1, .55)]:
            xi = (xc + dx) % W; img[:, xi] *= f
    # no pink gum line — pure enamel; the real gum is hidden behind the lip anyway (user wants
    # less gum). Keep just a faint warm tint at the very top edge.
    g = (np.linspace(1, 0, BH)[:, None, None] ** 16) * 0.3
    img = img * (1 - g) + np.array([210, 180, 165])[None, None] * g
    img += rng.normal(0, 4, img.shape)                    # subtle grain
    return img


def tongue():
    img = vgrad([78, 36, 40], [120, 60, 64])              # dim wet interior (not bright pink)
    x = np.arange(W)
    sulcus = np.exp(-((x - W / 2) / 6.0) ** 2)            # median groove down the middle
    img *= (1 - 0.35 * sulcus)[None, :, None]
    img += rng.normal(0, 5, img.shape)                    # papillae stipple
    return img


def palate():
    img = vgrad([34, 14, 16], [9, 4, 5])                  # dark throat, slight red
    img += rng.normal(0, 2, img.shape)
    return img


bands = [teeth([214, 198, 168], [248, 246, 240]),         # upper teeth
         teeth([206, 190, 162], [242, 240, 232], 8),       # lower teeth
         tongue(), palate()]
atlas = np.clip(np.concatenate(bands, 0), 0, 255).astype(np.uint8)
Image.fromarray(atlas).save(sys.argv[1] if len(sys.argv) > 1 else "mouth_atlas.png")
print("ATLAS_OK", atlas.shape)
