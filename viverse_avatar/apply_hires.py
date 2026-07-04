import bpy, sys
from mathutils import Vector
H = "/h"   # hires maps dir
SRC = "/work/avatar.glb"; OUT = "/work/avatar_hires.glb"; RENDER = "/h/hires_headshot.png"

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=SRC)

def tex_node(bsdf, slot):
    inp = bsdf.inputs.get(slot)
    if not inp or not inp.is_linked: return None
    n = inp.links[0].from_node
    if n.type == "TEX_IMAGE": return n
    for i in n.inputs:
        if i.is_linked and i.links[0].from_node.type == "TEX_IMAGE":
            return i.links[0].from_node
    return None

# mesh_key -> {slot: filename}.  Only BASE COLOR + hair are upscaled; normal/metallic/
# roughness stay native (Lanczos-upscaling them bleeds normals into the black UV
# background at island edges -> invalid normals -> black shading on seams).
PLAN = {
  "head_opaque":   {"Base Color":"head_color.png"},
  "body_opaque":   {"Base Color":"body_color.png"},
  "clothing":      {"Base Color":"clothing_color.png"},
  "head_transparent": {"Base Color":"hair_color.png","Alpha":"hair_color.png"},
}
cache = {}
def load(fn):
    if fn not in cache: cache[fn] = bpy.data.images.load(f"{H}/{fn}")
    return cache[fn]

for o in bpy.data.objects:
    if o.type != "MESH": continue
    key = next((k for k in PLAN if k in o.name.lower()), None)
    if not key: continue
    mat = o.material_slots[0].material
    bsdf = next(n for n in mat.node_tree.nodes if n.type=="BSDF_PRINCIPLED")
    for slot, fn in PLAN[key].items():
        node = tex_node(bsdf, slot)
        if node:
            old = node.image.name if node.image else None
            cs = node.image.colorspace_settings.name if node.image else "sRGB"
            node.image = load(fn)
            node.image.colorspace_settings.name = cs   # preserve Non-Color for normal/MR maps
            print(f"[hires] {key} {slot}: {old}({cs}) -> {fn}", flush=True)

bpy.ops.export_scene.gltf(filepath=OUT, export_format="GLB")
print(f"[hires] exported {OUT}", flush=True)

# head closeup render
mn=Vector((1e9,)*3); mx=Vector((-1e9,)*3)
for o in bpy.data.objects:
    if o.type=="MESH":
        for c in o.bound_box:
            w=o.matrix_world@Vector(c)
            for k in range(3): mn[k]=min(mn[k],w[k]); mx[k]=max(mx[k],w[k])
Ht=mx.z-mn.z; cx=(mn.x+mx.x)/2; cy=(mn.y+mx.y)/2
T=Vector((cx,cy,mn.z+0.90*Ht)); loc=T+Vector((0,1,0))*0.40*Ht
cd=bpy.data.cameras.new("c"); cam=bpy.data.objects.new("c",cd); bpy.context.scene.collection.objects.link(cam)
cam.location=loc; cd.lens=85; cam.rotation_euler=(T-loc).normalized().to_track_quat('-Z','Y').to_euler()
bpy.context.scene.camera=cam
w=bpy.data.worlds.new("W"); w.use_nodes=True; w.node_tree.nodes["Background"].inputs[1].default_value=1.4
bpy.context.scene.world=w
sc=bpy.context.scene; sc.render.engine="CYCLES"
try: sc.cycles.device="GPU"
except: pass
sc.cycles.samples=64; sc.render.resolution_x=1000; sc.render.resolution_y=1000
sc.render.filepath=RENDER; bpy.ops.render.render(write_still=True)
print("HIRES_DONE", flush=True)
