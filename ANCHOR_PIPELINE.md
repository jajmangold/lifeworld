# Anchor Pipeline — code → watchable mp4

How `make_anchor.sh` turns a WAV of narration into a photoreal talking news‑anchor clip.
Every model, every mask, every compositing step, in order.

```
make_anchor.sh --audio out/x.wav --out output/final.mp4 [--mood --brow --nod --browbase
                --seed --face anchorM.png --fast --premium --baked [--bakedtex T] --ots "...">]
```

Stage flow:
```
proc_perf ─▶ render(rtx0) ─▶ premult composite ─▶ encode silent.mp4
   │
   ├─ FAST ───────────────▶ mux audio ──────────────────────────────────┐
   └─ FULL ─▶ [swap | baked-skip] ─▶ MuseTalk ─▶ [GFPGAN restore | FlashVSR premium] ─┤
                                                                          ▼
                                                              [OTS panels] ─▶ final.mp4
```

---

## Machines & containers

| Host | Hardware | Role | Containers |
|---|---|---|---|
| **V100 box** (local) | "V100" GPUs are hacked‑BIOS CMP mining cards (crippled PCIe; per‑frame time swings 2–5× under contention) | orchestration, swap, lip‑sync, graphics | `muse-server` (GPU6), `swap-server`/`inswap` (GPU0), `mp-extract:1.0` (CPU performance + OTS) |
| **rtx0** | 4× RTX 3060 12 GB, sm86 | Blender render + FlashVSR | `sampl` (Blender `/opt/blender/blender`, `/mnt/datadisk/containers/sampl`→`/work`), `wan2gp` (`deepbeepmeep/wan2gp:3060`) |

Work is shuttled host↔rtx0 over `ssh/scp` (`josh@rtx0`). `avatar.glb` and `newsroom_pano.png` persist on rtx0 and are re‑staged only if missing.

---

## Models

| Model | Where | Job | Detail |
|---|---|---|---|
| **insightface buffalo_l** | swap-server | face detect + 512‑d embed + 5‑pt kps | drives swap target alignment and the keepeyes mask |
| **inswapper_128.onnx** | swap-server | identity face swap | operates at 128 px, blends back via warp |
| **gfpgan-1024.onnx** | swap-server | face restore/sharpen | fixed‑shape 512×512 fp32; final pass sharpens the soft MuseTalk mouth |
| **MuseTalk** (+ TAESD VAE) | muse-server | audio→lip‑sync | resident, optimized; inpaints mouth region only, TAESD‑encoded latents |
| **FlashVSR‑v1.1** | wan2gp (rtx0) | diffusion super‑res | head‑only 2× upscale to 1440p (`--premium`); Sparse SageAttention |
| **TAESD** | muse-server | fast VAE | tiny autoencoder for MuseTalk latents |

(Parked: **FastAvatar** — single‑image 3D‑Gaussian face, FLAME/DINO/gsplat. Good at angles but soft/static; not in the pipeline.)

---

## Stage 1 — Performance (`proc_perf.py`, mp-extract:1.0, CPU)

Audio → `output/${NAME}.perf.json`: per‑frame ARKit blendshape + head‑pose track driven by the speech envelope and `--mood/--brow/--nod/--browbase/--seed`. `--fast` adds `--visemes` (renders a mouth directly, so swap+MuseTalk can be skipped).

---

## Stage 2 — Render on rtx0 (`render_anchor_anim.py`, Blender EEVEE_NEXT)

The VIVERSE **avatar.glb** is rendered against the newsroom HDRI, animated by the perf JSON.

**Avatar head geometry** (the parts that matter):
- `head_Opaque_Material_Meshes_Mesh` (2615 verts), material `…_Material`, base‑color **`Image_2`** 1024², UV layer **`UVMap`** — the face skin.
- `head_Transparent` (130 verts, `Image_0` 512²) — the "fuzz" hairline shell (optionally hidden via `HIDE_HEAD_FUZZ`).
- others: `body_Opaque` (Image_5 256²), `clothing_Opaque` (Image_8 1024²), `cornea_Transparent`, `Icosphere`.

**Baked mode** (`BAKED_HEAD_TEX` env): the head_Opaque base‑color image is swapped for `anchorM_head_baked.png` (loaded sRGB) so the render already carries anchorM's identity — no per‑frame face swap downstream. See *Face‑bake* below.

**Lighting:** `KEY=130, FILL=45, RIM=90`; the rim area light sits behind+above (`cx‑0.3, cy‑1.5, topz+0.7`) to separate the silhouette from the bokeh background.

**Camera:** `lens=LENS`, located `(cx, mx.y+H*0.62, aimz)`, rot `(90°,0,180°)`; **1280×720**; view transform **AgX**, exposure `EXPO`, env strength `ENVSTR=0.75`.

**Render = background plate + transparent character (post‑composite)**, the key to clean edges:
```python
sc.render.film_transparent=False           # 1) hide all meshes, render the bg plate once
... hide all meshes ...; render -> bg.png
... restore visibility ...
sc.render.film_transparent=True            # 2) render character RGBA over transparency
sc.render.image_settings.color_mode="RGBA"
render animation -> f0001.png … (premultiplied alpha)
```

---

## Stage 2b — Premultiplied‑over composite (`composite_premult.py`, rtx0)

EEVEE writes **premultiplied** alpha: edge pixels have RGB already multiplied by their (fractional) alpha (edge RGB ~151 vs opaque hair ~200). The only correct composite is **premult‑over, no divide**:

```python
a   = im[:,:,3:4]/255.0
rgb = im[:,:,:3]            # already premultiplied by EEVEE
out = (rgb + bg*(1.0 - a)).clip(0,255)     # f####.png -> c####.png
```

> Why this and nothing else:
> - **Straight overlay** (`rgb*a + bg*(1‑a)`) double‑darkens the edge → **dark fringe**.
> - **Unpremultiply → overlay** (e.g. ffmpeg `unpremultiply,overlay`) divides edge RGB by a tiny alpha and clamps to white → **bright halo**.
> - `rgb_pm + bg*(1‑α)` is exactly the over operator for premultiplied source → invisible seam.

Then ffmpeg encodes `c%04d.png` → `${NAME}_silent.mp4` (25 fps, crf 18), pulled back to the host.

---

## Stage 3 — Mouth / identity

### FAST (`--fast`)
Viseme mouth is already in the render; just mux audio → `${NAME}_talk.mp4`. Done (~4 min).

### FULL — identity first
**(a) Swap** (skipped in `--baked`): swap-server runs insightface+inswapper on `${NAME}_silent.mp4` → `${NAME}_swap.mp4`.
- `keepeyes:true` — insightface 5‑pt kps carve **eye holes** out of the swap mask so the avatar's real blinks/eye motion survive the swap.
- `enhance:false` — **no GFPGAN here**; it's deferred to the final pass so the same compute also sharpens the soft MuseTalk mouth.
- `save_faces:"…faces.json"` — caches detected face boxes/kps so the restore pass needn't re‑detect.

**(baked)** identity is already in the texture → swap stage skipped entirely (~7 min/40 s saved); MuseTalk runs on the render directly. Restore re‑detects faces (0 misses observed).

**(b) MuseTalk** (muse-server): `${MUSE_IN}.mp4` + 16 kHz mono wav → `${NAME}_talk.mp4`. Inpaints only the mouth region (head stays locked), TAESD‑encoded.

### Final face finish
**Default — GFPGAN restore** (swap-server, `mode:restore`): 512×512 fp32 restore sharpens the ~256 px MuseTalk mouth. `keepeyes:true` again; reuses `faces.json` if present (else re‑detect). Restore is video‑only mp4v → audio is re‑muxed from the muse output. Mask is an **elliptical feathered face mask** so the sharpened face blends into the unchanged surroundings. 720p.

**`--premium` — FlashVSR‑face 2× upscale** (rtx0 wan2gp): diffusion super‑res on the **head only**, Lanczos on everything else → **2560×1440**. Replaces GFPGAN. (`rm -f` the root‑owned `_talk.mp4` before scp, then pull the result back.) Sets `OTSW=860` so OTS panels match the 2× canvas. See *FlashVSR face‑only* below.

---

## Stage 4 — OTS graphics (optional, `--ots`)

`--ots "img|LABEL|end; img2|LABEL2|end2"` (seconds = segment **end** times). For each segment, `ots_panel.py` builds an over‑the‑shoulder panel (`--w ${OTSW:-430}`, =860 at 1440p), then `ots_compose.sh` overlays them onto `${NAME}_talk.mp4` → `$OUT`. No `--ots` → straight copy to `$OUT`.

---

## Face‑bake (replaces per‑frame swap) — `bake/`

One‑time: bake anchorM's identity into the avatar's head UV texture so every render is already anchorM. Full `--baked` run ≈ 8 min vs ≈ 21 min.

1. **`bake_source.py`** — render a clean frontal head: even world strength **0.85** (true‑albedo; 1.6 baked in lighting → pale skin), lens 55, cam at `mx.y+H*1.45`, RGBA transparent, **Standard** view transform (not AgX — we want raw albedo) → `frontal.png` + `cam.json`.
2. **swap** anchorM onto `frontal.png` → `_bake_swapped.png` (insightface+inswapper, normal swap).
3. **`bake_face.py`** (Cycles EMIT bake):
   - Recreate the exact front camera from `cam.json`.
   - Add a **`Proj`** UV layer via a **UV_PROJECT** modifier from that camera (`aspect 1:1`), then **apply** it so camera‑projected UVs are baked into mesh data.
   - **Bake the projected swapped face** into native `UVMap` space: temp Emission material samples `_bake_swapped.png` through the `Proj` UV; Cycles `bake(type='EMIT')`, margin 8, target 1024² → `baked_face.png`.
   - **Bake a facing mask** into `UVMap`: `dot(world_normal, -cam_fwd)` (note the **negation** — front faces give a positive value; without it front faces bake black). → `baked_facing.png`.
   - Composite face×smoothstep(mask) over the original `Image_2` → **`anchorM_head_baked.png`** — front‑facing coverage only, feathered where the head turns away from camera.

The baked texture is gitignored but reproducible from `bake/`.

---

## FlashVSR face‑only (`--premium`) — rtx0 wan2gp

Sharpen only the head with diffusion VSR, Lanczos the rest. ~11× faster than full‑frame on 12 GB (one region, no tiling). `flashvsr_face.sh <stem>` chains:

1. **`detect_crop.py`** — Haar `frontalface_default` every 10th frame; union all boxes, expand ×2.2 to a square, clamp, round to mult‑of‑16 → `head_crop.mp4` + `headbox.json`. Box is **static** (fine for fixed camera).
2. **`head_up.py`** — FlashVSR‑v1.1 `tiny-long` pipeline, whole crop as one region, `color_fix=True`, scale 2, bf16, GPU device 1 → `head_up.mp4`. Cost scales with frame count, not region size (~2 s/frame‑iter).
3. **`composite.py`** — Lanczos 2× the full frame to 2560×1440, then **feather‑blend** the sharp upscaled head crop back in with an **`r=160` px ramp** on all four box edges (`reg*(1‑mask)+head*mask`), mux original audio (`libx264 crf 17`, AAC).

**Geometry gotcha:** `get_input_params` center‑crops the 2× region to a multiple of 128 (e.g. 464×432 box → 896×768), so paste offsets must add back the trimmed margin:
```
px = box_x*2 + (box_w*2 - aligned_w)//2
py = box_y*2 + (box_h*2 - aligned_h)//2
```

---

## The edge saga (why composite math matters)

Three stacked artifacts, each fixed by getting one mask/blend right:

1. **Dark fringe** — straight‑alpha overlay of premultiplied EEVEE output. Fix: premult‑over.
2. **Bright halo** — `unpremultiply→overlay` divides edge RGB by ~0 alpha, ffmpeg clamps to white. Fix: numpy `rgb_pm + bg*(1‑α)` (ffmpeg can't express premult‑over), in `composite_premult.py`.
3. **Box seam (neck/ear)** — FlashVSR head‑crop boundary; 56 px feather too narrow at 1440p. Fix: widen `composite.py` feather ramp `r` 56→**160** (CPU re‑composite only, no re‑render).

---

## Masking — quick reference

| Mask | Stage | Purpose |
|---|---|---|
| **keepeyes** (insightface 5‑pt kps → eye holes) | swap + restore | preserve real avatar blinks through swap/restore |
| **GFPGAN elliptical feather** | final restore | blend sharpened face into unchanged surroundings |
| **facing mask** `dot(normal, -cam_fwd)` + smoothstep | face‑bake | limit baked identity to front‑facing head, feather the turn |
| **premultiplied‑over** `rgb_pm + bg*(1‑α)` | render composite | correct silhouette edge (no fringe/halo) |
| **FlashVSR feather** `r=160` head crop | premium upscale | invisible head/background seam at 1440p |

---

## Output sizes & timings (≈)

| Mode | Resolution | Wall time |
|---|---|---|
| `--fast` | 1280×720 | ~4 min |
| full (swap + GFPGAN) | 1280×720 | ~21 min |
| `--baked` (no swap) | 1280×720 | ~8 min |
| `--premium` (FlashVSR) | 2560×1440 | +~4 min (replaces GFPGAN) |
