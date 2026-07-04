"""Generate an OWN base human with Human Generator (HumGen3D) — headless, on rtx0's Blender.

Why own the base: harvesting whole BlenderKit characters means inheriting each artist's topology, fragmented
UVs (broke the projection-retexture), glossy/in-face hair, fixed outfit, and random rig naming. A HumGen base
gives us ONE canonical, photoreal foundation: clean single-material SSS skin body, FACS facial shapekeys (the
talking rig), a 100+ bone body rig (idle motion), consistent naming. New character = preset + body/face params
+ skin tone + HAAR groom + wardrobe outfit. All DATA, matching the Studio philosophy.

Headless requirements (the addon + content live on rtx0's sampl container):
  addon:   /work/hg_scripts/addons/HumGen3D      (env BLENDER_USER_SCRIPTS=/work/hg_scripts)
  content: /work/hg_content/                     (get_prefs().filepath; extracted .hgpack packs)
Base packs ship humans/hair/poses but NO outfit garments (outfits are a separate paid HumGen pack) -> we
clothe the base from the wardrobe (BlenderKit garment + reproject/recolor). LICENSE: HumGen addon=GPL (free to
run/modify); ASSETS are royalty-free/commercial-capable BUT must never be distributed in an extractable form
(rule #4) -- we only ever publish RENDERED video (the .blend/textures stay on-box), so we're compliant.

  blender -b --python humgen_gen.py -- <gender> <preset_substr> <out.blend> [facial_rig=1]
"""
import bpy, sys, os

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else ["female", "", "/work/hg_out/base.blend", "1"]
GENDER = argv[0] if len(argv) > 0 else "female"
PRESET_SUBSTR = argv[1] if len(argv) > 1 else ""
OUT = argv[2] if len(argv) > 2 else "/work/hg_out/base.blend"
FACIAL_RIG = (argv[3] if len(argv) > 3 else "1") == "1"

bpy.ops.preferences.addon_enable(module="HumGen3D")
from HumGen3D.backend.preferences.preference_func import get_prefs
get_prefs().filepath = os.environ.get("HG_CONTENT", "/work/hg_content/")
from HumGen3D import Human

opts = Human.get_preset_options(GENDER)
pick = next((o for o in opts if PRESET_SUBSTR.lower() in o.lower()), opts[0]) if PRESET_SUBSTR else opts[0]
print("PRESET", pick, flush=True)
human = Human.from_preset(pick)
print("GENERATED name=", human.name, flush=True)

# FUTURE-PROOF lower-face slim: some presets ship a very wide chin/cheek (e.g. Jessica chin_width=1.78,
# cheek_fullness=1.03). A lower face wider than the swapped-in real face makes the render's jaw/cheek stick
# out past the swap. Clamp the widest lower-face keys so no generated base bulges beyond a normal face. Tune
# the caps via HG_CHIN_W / HG_CHEEK caps.
_chin_cap = float(os.environ.get("HG_CHIN_W", "0.3")); _cheek_cap = float(os.environ.get("HG_CHEEK", "0.35"))
try:
    for k in human.face.keys:
        if k.name == "chin_width" and k.value > _chin_cap: print("SLIM chin_width", round(k.value,2), "->", _chin_cap); k.value = _chin_cap
        elif k.name == "cheek_fullness" and k.value > _cheek_cap: print("SLIM cheek_fullness", round(k.value,2), "->", _cheek_cap); k.value = _cheek_cap
except Exception as e:
    print("FACE_SLIM_WARN", repr(e), flush=True)

if FACIAL_RIG:
    try:
        human.expression.load_facial_rig()
        print("FACIAL_RIG has_rig=", human.expression.has_facial_rig, flush=True)
    except Exception as e:
        print("FACIAL_RIG_WARN", repr(e), flush=True)

rig = human.objects.rig
print("RIG", rig.name, "bones=", len(rig.data.bones), flush=True)
for o in human.objects:
    if getattr(o, "type", None) == "MESH":
        mats = [s.material.name if s.material else None for s in o.material_slots]
        sks = len(o.data.shape_keys.key_blocks) if o.data.shape_keys else 0
        print(f"MESH {o.name} mats={mats} shapekeys={sks}", flush=True)

os.makedirs(os.path.dirname(OUT), exist_ok=True)
# HumGen saves texture paths RELATIVE to the blend (//../hg_content/...). The render pipeline stages the blend
# to a different dir depth on rtx0, which breaks those relatives (-> magenta "texture not found"). Fix: convert
# to ABSOLUTE (/work/hg_content/..., resolves anywhere on rtx0's sampl container). ORDER MATTERS: make_paths_
# absolute needs a saved .blend to resolve // against, so save FIRST, then make absolute, then save again.
# (For full off-rtx0 portability use human.process.baking.bake_all(folder_path=...) instead -- bakes the
# procedural skin to standalone maps; heavier, not needed while we always render on rtx0.)
bpy.ops.wm.save_as_mainfile(filepath=OUT)
bpy.ops.file.make_paths_absolute()
bpy.ops.wm.save_as_mainfile(filepath=OUT)
print(f"HUMGEN_SAVED {OUT}", flush=True)
