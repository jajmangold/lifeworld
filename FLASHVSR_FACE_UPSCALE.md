# FlashVSR face-only fast upscale (talking heads)

Fast, clean 2×/4× upscale of a fixed-camera talking-head video by running diffusion
super-resolution **only on the head** and Lanczos-upscaling everything else. ~11× faster
than full-frame FlashVSR on a 12 GB GPU, with no quality loss where it matters.

- **Host:** `rtx0` (4× RTX 3060, 12 GB each, sm86)
- **App:** Wan2GP at `/mnt/datadisk/containers/wan2gp` (image `deepbeepmeep/wan2gp:3060`)
- **Plugin:** `Hakaze/wan2gp-flashvsr` in `plugins/wan2gp-flashvsr` (official FlashVSR-v1.1
  weights + Sparse SageAttention)
- **Scripts:** `detect_crop.py`, `head_up.py`, `composite.py` (in the wan2gp dir on rtx0)

---

## Why face-only

Full-frame FlashVSR on a 720p clip needs spatial tiling on a 12 GB card (the model is
~10.5 GB resident). That means 6 tiles × 500 frames ≈ **25 min**, and the tiles can leave
faint seams on flat/soft backgrounds — the blurry studio bokeh is exactly the content
diffusion VSR hallucinates worst.

A talking head only needs the **face/hair/skin** sharpened. So:

1. Crop the head region (one stable box for a fixed camera).
2. Run FlashVSR on just that crop — small enough to be **one pass, no tiling**.
3. Lanczos-upscale the full frame (the soft background loses nothing to Lanczos).
4. Feather-composite the sharp head back in; mux the original audio.

Result: same diffusion detail on the face, invisible seam, and the slow part now scales
with frame count only (one region instead of six).

### Performance (1280×720 → 2560×1440)

| Clip | Frames | Full-frame (6 tiles) | **Face crop, 1 pass** |
|---|---|---|---|
| wrapper_full_test_talk | 500 | ~25 min | **131 s** |
| iran_full | 1022 | (~50 min est.) | **259 s** |

(+ ~7–52 s one-time model load.)

---

## Pipeline

### 1. `detect_crop.py` — find the head box + write the crop
- OpenCV Haar `haarcascade_frontalface_default` on every 10th frame.
- Unions all detections, expands ×2.2 to a square (covers hair/forehead/chin/neck +
  motion margin), clamps to frame, rounds to mult-of-16.
- Writes `myinput/head_crop.mp4` and `myinput/headbox.json` (`x,y,w,h,W,H,fps`).

### 2. `head_up.py` — FlashVSR the crop (single region, no tiling)
- Loads the `tiny-long` FlashVSR-v1.1 pipeline via the plugin's bundled
  `src.models.download_manager.load_pipeline` (models auto-download to `ckpts/flashvsr`).
- Feeds the whole crop as one region (`get_input_params` → one `pipeline(...)` call),
  `color_fix=True`, scale 2, bf16. Writes `myinput/head_up.mp4`.
- ~2 s/frame-iter; cost is frame count, not region size.

### 3. `composite.py` — Lanczos bg + feathered head + audio
- Lanczos 2× each source frame to 2560×1440.
- Feather-blends the upscaled head (56 px ramp) onto the canvas.
- Muxes original audio (`-c:v libx264 -crf 17`, AAC). Writes
  `outputs/flashvsr/wan2gp_face_fast_2x.mp4`.

#### Geometry gotcha
`get_input_params` center-crops the 2× crop to a multiple of 128, so a 464×432 box →
**896×768** output (not 928×864). Paste offset must account for the trimmed margin:

```
px = box_x*2 + (box_w*2 - aligned_w)//2     # e.g. 400*2 + (928-896)//2 = 816
py = box_y*2 + (box_h*2 - aligned_h)//2     # e.g.   0*2 + (864-768)//2 =  48
```

---

## Run it

The scripts are parameterized: `detect_crop.py <src> <stem>`, `head_up.py <stem>`,
`composite.py <stem> <src>`. `<stem>` names the intermediate/output files
(`head_crop_<stem>.mp4`, `head_up_<stem>.mp4`, `wan2gp_face_fast_<stem>_2x.mp4`).

```bash
# on rtx0, in /mnt/datadisk/containers/wan2gp
IMG=deepbeepmeep/wan2gp:3060
STEM=iran
SRC=/workspace/myinput/iran_full.mp4      # put your clip in myinput/ first
DC="docker run --rm -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --entrypoint bash -v $PWD:/workspace $IMG -c"

# 1. detect + crop (CPU)
$DC "cd /workspace && python3 detect_crop.py $SRC $STEM"

# 2. upscale the head (GPU — pin one card)
docker run --rm --gpus '"device=1"' -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    --entrypoint bash -v $PWD:/workspace $IMG -c "cd /workspace && python3 -u head_up.py $STEM"

# 3. composite + audio (CPU)
$DC "cd /workspace && python3 composite.py $STEM $SRC"
# -> outputs/flashvsr/wan2gp_face_fast_${STEM}_2x.mp4
```

Composite auto-derives geometry: it reads the head box from `headbox_<stem>.json` and the
aligned upscaled size from `head_up_<stem>.mp4`, so any crop size works without editing.

---

## Tuning

- **Faster:** use the `tiny` variant (vs `tiny-long`) for ≤120-frame clips; tighter crop;
  scale 2 instead of 4.
- **Bigger/sharper face:** raise the crop expansion factor (×2.2) or scale to 4.
- **Visible seam:** increase the feather ramp `r` in `composite.py`, or grow the box so the
  boundary lands on plain shirt/background, never across facial features.

## Caveats

- The head box is **static** — ideal for a fixed-camera anchor. A moving/walking subject
  needs per-frame face tracking (replace the union box with per-frame boxes + a stabilized
  crop window).
- Haar can miss extreme head turns; for robustness swap in `insightface`/`facexlib`
  (both present in the image, but they download detector weights).

## Related

- Full-frame path + the background on this whole setup (FlashVSR-Pro vs wan2gp-flashvsr,
  the 12 GB memory wall, the color-fix `.view` bug) is in the agent memory note
  `flashvsr-rtx0`.
