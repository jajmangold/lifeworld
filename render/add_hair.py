"""Bake a HAIR groom onto a bald viverse realistic head, in Blender, exporting a new GLB (hair baked in,
reused every render). Instead of fitting a mismatched hair shell, we DERIVE the hair from the head mesh
itself: duplicate the head, keep only the scalp/back/side region (above the hairline, not the face),
push it out along normals for volume + solidify, and keep the head's armature weights so the hair
deforms with the head. Perfect conform, animates for free.
  blender -b --python add_hair.py -- --in head.glb --out haired.glb [--color 4a3226]
          [--front 0.62] [--side 0.42] [--vol 0.012] [--thick 0.02] [--len 0.0]
Coords after glTF import: Z up, +Y = face front, -Y = back of head.
"""
import bpy, sys, math

argv = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
def a(f, d): return argv[argv.index(f)+1] if f in argv else d
INP = a("--in", "/work/reporter_avatar.glb"); OUT = a("--out", "/work/reporter_haired.glb")
COLOR = a("--color", "4a3226")
FRONT = float(a("--front", "0.62"))   # front hairline height (frac of head height); higher = more forehead
SIDE = float(a("--side", "0.42"))     # back/side coverage floor (frac); lower = longer down sides
VOL = float(a("--vol", "0.012"))      # push scalp out along normals (m) before solidify -> lift off skull
THICK = float(a("--thick", "0.02"))   # solidify thickness (m) -> hair volume
LEN = float(a("--len", "0.0"))        # extrude the bottom boundary down this many m -> length

def hex_lin(h):
    c = [int(h[i:i+2], 16)/255 for i in (0, 2, 4)]
    return [(v/12.92 if v <= 0.04045 else ((v+0.055)/1.055)**2.4) for v in c] + [1.0]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=INP)

# find the head skin mesh (material starts 'AvatarHead', pick the highest-poly one)
head = None
for o in bpy.data.objects:
    if o.type != "MESH": continue
    if any((m.name or "").startswith("AvatarHead") for m in o.data.materials if m):
        if head is None or len(o.data.vertices) > len(head.data.vertices): head = o
if head is None: raise SystemExit("no AvatarHead mesh")
print("head mesh:", head.name, len(head.data.vertices), "verts")

# bbox (world) for thresholds
import mathutils
cos = [head.matrix_world @ v.co for v in head.data.vertices]
zmin = min(c.z for c in cos); zmax = max(c.z for c in cos); zH = zmax - zmin
ymin = min(c.y for c in cos); ymax = max(c.y for c in cos); yD = ymax - ymin
z_front = zmin + FRONT*zH; z_side = zmin + SIDE*zH; y_mid = ymin + 0.52*yD
print(f"head z[{zmin:.3f},{zmax:.3f}] y[{ymin:.3f},{ymax:.3f}] z_front={z_front:.3f} z_side={z_side:.3f}")

# duplicate the head -> hair
hair = head.copy(); hair.data = head.data.copy(); hair.name = "Hair_Groom"
bpy.context.collection.objects.link(hair)
# keep only scalp/back/side verts: top band (above front hairline) OR back/side above ear level
import bmesh
bm = bmesh.new(); bm.from_mesh(hair.data)
bm.verts.ensure_lookup_table()
mw = hair.matrix_world
kill = []
for v in bm.verts:
    w = mw @ v.co
    keep = (w.z > z_front) or (w.y < y_mid and w.z > z_side)
    if not keep: kill.append(v)
bmesh.ops.delete(bm, geom=kill, context="VERTS")
# push remaining scalp OUT along normals so hair sits above the skull, not on it
bm.normal_update()
for v in bm.verts:
    v.co += v.normal * VOL
bm.to_mesh(hair.data); bm.free()
hair.data.update()

# clean loose bits, smooth, solidify for volume, optional length at the bottom edge
bpy.context.view_layer.objects.active = hair
for o in bpy.data.objects: o.select_set(False)
hair.select_set(True)
bpy.ops.object.mode_set(mode="EDIT")
bpy.ops.mesh.select_all(action="SELECT")
bpy.ops.mesh.delete_loose()
if LEN > 0:
    bpy.ops.mesh.select_all(action="DESELECT")
    bpy.ops.mesh.select_non_manifold()          # boundary loop (bottom edge)
    bpy.ops.mesh.extrude_region_move(TRANSFORM_OT_translate={"value": (0, 0, -LEN)})
bpy.ops.object.mode_set(mode="OBJECT")
bpy.ops.object.shade_smooth()
sol = hair.modifiers.new("sol", "SOLIDIFY"); sol.thickness = THICK; sol.offset = 1.0

# hair material
for s in list(hair.data.materials):
    pass
hair.data.materials.clear()
mat = bpy.data.materials.new("HairMat"); mat.use_nodes = True
bsdf = mat.node_tree.nodes.get("Principled BSDF")
bsdf.inputs["Base Color"].default_value = hex_lin(COLOR)
bsdf.inputs["Roughness"].default_value = 0.42
if "Specular IOR Level" in bsdf.inputs: bsdf.inputs["Specular IOR Level"].default_value = 0.5
hair.data.materials.append(mat)
print("hair verts kept:", len(hair.data.vertices))

# export whole scene (original body/head + hair) as glb
bpy.ops.export_scene.gltf(filepath=OUT, export_format="GLB", use_selection=False,
                          export_yup=True, export_skins=True, export_morph=True)
print("HAIR_BAKED ->", OUT)
