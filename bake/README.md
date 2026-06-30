# Face-bake: put a target identity on the VIVERSE avatar mesh (replaces per-frame swap)

One-time bake so the render produces the target face directly (no per-frame face-swap).
Front-projection coverage — ideal for a head-on anchor.

1. `bake_source.py` (rtx0/sampl) — render the avatar head front-on, evenly lit -> bake/frontal.png + cam.json
2. swap the target face onto frontal.png (inswap + GFPGAN, full identity) -> bake/_bake_swapped.png
3. `bake_face.py` (rtx0/sampl, Cycles) — UVProject the swapped image from the front cam onto head_Opaque,
   bake into UVMap space -> baked_face.png + a facing-mask bake baked_facing.png
4. numpy composite: new = baked_face*mask + orig_head_tex*(1-mask), mask = smoothstep(facing)&projected
   -> viverse_avatar/anchorM_head_baked.png
5. render_anchor_anim.py picks it up via env BAKED_HEAD_TEX (or --bakedtex); make_anchor.sh --baked
   sets it and skips the swap stage.

Target texture lives on head_Opaque material (Image_2, 1024x1024, UVMap).
