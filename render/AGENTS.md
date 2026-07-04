# render/ — AGENTS.md
> The renderers: Blender talking-anchor/character frames + the FlashVSR / LTX photoreal finishing passes.

## What it does
Produces the raster talking-head frames that the anchor pipeline swaps + lip-syncs downstream, plus the
photoreal upscale/realism finishers. **ACTIVE path:** the Blender EEVEE character renderer
(`render_character.py`) driven by `make_anchor.sh`/`make_character.sh`, the wardrobe/hair/texture ops it
depends on (`wardrobe.py`, `gen_hair.py`+`graft_hair.py`, `fix_hair.py`, `slim_face.py`, ESRGAN texture
upscale), the FlashVSR face upscaler (`flashvsr/`), and the LTX-2.3 v2v realism pass (`ltx_realism.sh`).
`humgen_gen.py` (HumGen3D own-base human) is the current direction for base meshes.
**LEGACY (ARCHIVED, do not build on):** the whole SMPL-X / Kimodo / FLOAT / MDM / teeth / calibration /
Audio2Face avatar cluster has been **moved to `/mnt/24tb/nvme-offload/bot-archive/render-experiments/`**
(2026-07-04 — superseded by the LTX-2.3 SDNQ farm + `render/sdnq_ltx/a2v_talk.py`). Includes
`combine_motion_face.py`, `scene_cine.py`, `mdm_*`, `kimodo_*`, `float_*`, `teeth_*`, `bake_smplx.py`,
`mp_extract/mp_pose.py`, `a2f_lipsync.py`, `face_drive.py`, plus earlier `render_smplx.py`/`calib_*`.
Recover from git history or the archive. Active hair ops (`fix_hair.py`, `graft_hair.py`) were kept.

## Run
Everything runs from `studio/`. The renderers are invoked by the top-level orchestrators, not directly:
```
bash make_anchor.sh    --audio out/x.wav --out out/final.mp4 [--premium|--ultra] [--pano env.exr] [--character asset.blend]
bash make_character.sh --char asset.blend --audio out/x.wav --out out/final.mp4      # thin wrapper over make_anchor.sh --character
```
Wardrobe / texture / hair ops (called by the above, also standalone):
```
python3 render/wardrobe.py {list|show|textures|retexture} <char> [--method recolor|klein|esrgan] ...
python3 render/gen_hair.py "<prompt>"          # HAAR strand groom via resident hairgen service
```
FlashVSR face upscale (resident async queue :8800; `ULTRA=1`→4x): `bash render/flashvsr/flashvsr_face.sh <stem>`
LTX realism pass (wan2gp docker on rtx0): `bash render/ltx_realism.sh <in.mp4> <out.mp4> [denoise=0.65] [gpu]`

## Key files
- `render_character.py` — source of truth for the viverse-free Rigify/BlenderKit render (framing, HDRI light, ARKit-driven blinks/brows/idle, premult frames). MuseTalk owns the mouth; face-swap runs downstream.
- `wardrobe.py` — CLI manager over `catalog/wardrobe.json` + character/garment textures (BlenderKit dl, ESRGAN upscale, Klein edit, Blender repack). Ties the loose texture ops together.
- `gen_hair.py` + `graft_hair.py` + `fix_hair.py` — HAAR groom generation, grafting onto the head bone, and the EEVEE hair/cloth de-plasticiser (fixes "wet rubber" female-character hair).
- `flashvsr/flashvsr_face.sh` (+ `detect_crop.py`, `composite.py`) — head-crop diffusion upscale; the `--premium/--ultra` finisher.
- `ltx_realism.sh` / `ltx_server.py` — LTX-2.3 low-denoise v2v photoreal pass (retimes + re-muxes to preserve A/V sync).
- (Legacy SMPL-X/motion cluster is archived — see the LEGACY note above.)

## Gotchas & rules
- Blender renders run remotely on rtx0 (`ssh josh@rtx0`) via the orchestrators, not on the local host.
- LTX distilled changes frame count/timing (baked temporal upsampler) — `ltx_realism.sh` MUST retime/re-mux back to the input's duration/fps; don't strip that.
- Renderer emits **premultiplied** frames for the premult-over composite — keep alpha premult through the chain.
- FlashVSR uses the resident queue (:8800) to avoid the second-model-copy OOM; don't add a per-call model reload.
- This dir is a grab-bag of experiments — confirm a file is on the `make_anchor.sh` call graph before assuming it's live.

## Related
- `render/sdnq_ltx/SDNQ_STACK.md` — the LTX-2.3 SDNQ resident farm (documented separately; do not duplicate here).
- Up: `make_anchor.sh` / `make_character.sh` / `proc_perf.py` (perf → render). Down: swap + MuseTalk stages, then `output/`.
