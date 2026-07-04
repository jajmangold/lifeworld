# viverse_avatar/ — AGENTS.md
> VIVERSE avatar meshes/rigs (VRM-as-GLB) + baked head textures + hair/skin retexture tools that feed the Blender anchor render path.

## What it does
Holds the canonical talking-head anchor: `avatar.glb`, a VIVERSE character exported from
`.vrm` (which is already glTF) with an `Avatar_*` bone rig and ARKit-name blendshapes
(eyeBlinkLeft, jawOpen, mouthSmile…). `render_anchor_anim.py` (studio root) imports this
mesh, drives idle + ARKit face motion, and renders the newsroom anchor.
Also here: **retexture tools** — reproject/bake head & clothing base-color, ESRGAN-upscale
maps, and Klein-4B instruction edits — that produce `avatar_hires.glb` and the **baked head
textures** used for identity. Mostly a staging/asset dir + one-off Blender helper scripts;
the many UUID `.glb` files are alternate avatar candidates, and `reproj/`, `tex_dump/`,
`inspect*.py`, `bones.py`, `bs_test.py` are experimental/scratch.

## Run
Blender helpers run headless with the avatar staged at `/work/avatar.glb`:
```
blender --background --python reproject.py       -- render|bake   # clothing/body reproject bake
blender --background --python hair_reproject.py  -- render|bake   # HEAD hair reproject bake
python3 hair_compose.py <orig> <baked_front> <facing> <out_tex>   # host composite (carves eyes)
blender --background --python inspect_avatar.py                   # list meshes/blendshapes/bones
```
`klein_edit.py` posts an image+instruction to the resident `klein-proxy` (FLUX.2 Klein-4B).
No entrypoints — these are invoked ad hoc while producing the baked textures below.

## Key files / assets
- `avatar.glb` — canonical anchor mesh/rig (VIVERSE VRM→GLB, ~13 MB, `Avatar_*` bones, ARKit blendshapes).
- `anchorM_head_baked.png`, `anchorM_klein_head.png` — baked head base-color textures; injected via the renderer's `BAKED_HEAD_TEX` hook (keeps the per-frame face swap, just upgrades hair/skin).
- `avatar_hires.glb` — ESRGAN base-color-upscaled variant (`apply_hires.py`); `avatar_restyled/skin*` — retexture intermediates.
- `reproject.py` / `hair_reproject.py` / `hair_compose.py` — view-render → edit → project-bake texture pipeline.
- `Catwalk_Idle_04.bvh` — idle motion clip; `*-*-*.glb` (UUIDs) — alternate avatar candidates.

## Gotchas & rules
- The render path depends on `Avatar_*` bone naming (`Avatar_Head/Neck/Spine`) — don't rename; `render_character.py` has a fallback for Rigify `head/neck/spine`.
- `.vrm` is already glTF: stage/copy it as `.glb` (see `MESH_model_original_*.vrm~hmac=*.glb`).
- Baking over the eye region breaks blinks — `hair_compose.py` carves eyes/iris back to original; keep that carve when regenerating head textures.
- Blender scripts hardcode `/work/avatar.glb` and dirs like `/h` (hires maps); stage inputs there. Global rules inherited from `studio/AGENTS.md`.

## Related
- `../render_anchor_anim.py` + `make_anchor.sh` (consume `avatar.glb` + baked head tex), `render/render_character.py` (Rigify path).
- `../splat/` & `../humgen/` — other avatar approaches.
