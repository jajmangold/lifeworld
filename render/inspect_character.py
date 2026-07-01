"""Inspect a BlenderKit character .blend to scope the generic loader: objects, armature bones (head/
neck/spine for idle motion), face shape keys, materials, hair, bbox.
  blender -b --python inspect_character.py -- <asset.blend>
"""
import bpy, sys
f = sys.argv[sys.argv.index("--") + 1]
bpy.ops.wm.open_mainfile(filepath=f)
sc = bpy.context.scene

print("\n=== OBJECTS ===")
for o in bpy.data.objects:
    extra = f"{len(o.data.vertices)}v" if o.type == "MESH" else ""
    print(f"  {o.type:9s} {o.name[:40]:40s} {extra}")

print("\n=== ARMATURE bones ===")
for o in bpy.data.objects:
    if o.type == "ARMATURE":
        bones = [b.name for b in o.data.bones]
        print(f"  armature '{o.name}' — {len(bones)} bones")
        for k in ("head", "neck", "spine", "chest"):
            hits = [b for b in bones if k in b.lower()]
            if hits: print(f"    {k}: {hits[:6]}")
        print(f"    sample: {bones[:12]}")

print("\n=== FACE shape keys (for expressions/blinks) ===")
found = False
for o in bpy.data.objects:
    if o.type == "MESH" and o.data.shape_keys:
        found = True
        ks = [k.name for k in o.data.shape_keys.key_blocks]
        print(f"  {o.name}: {len(ks)} keys -> {ks[:20]}")
if not found: print("  (no shape keys — MuseTalk drives the mouth, blinks would need bones/none)")

print("\n=== MATERIALS / hair hint ===")
mats = [m.name for m in bpy.data.materials]
print("  materials:", mats[:20])
print("  hair-ish:", [m for m in mats if any(h in m.lower() for h in ["hair", "eyebrow", "lash", "beard"])])

# overall bbox (world) of mesh geometry
import mathutils
lo = mathutils.Vector((1e9,)*3); hi = mathutils.Vector((-1e9,)*3)
for o in bpy.data.objects:
    if o.type == "MESH":
        for v in o.bound_box:
            w = o.matrix_world @ mathutils.Vector(v)
            lo = mathutils.Vector((min(lo[i], w[i]) for i in range(3)))
            hi = mathutils.Vector((max(hi[i], w[i]) for i in range(3)))
print(f"\n=== BBOX world: {[round(x,2) for x in lo]} .. {[round(x,2) for x in hi]}  (up axis guess: Z height {round(hi[2]-lo[2],2)})")
print("INSPECT_DONE")
