import bpy, sys, math
from mathutils import Vector
argv = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
NEWTEX = argv[0]                      # path to new_Image_8.png
OUTGLB = "/work/avatar_restyled.glb"
RENDER = "/work/reproj/verify_restyled.png"

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath="/work/avatar.glb")

# replace clothing base-color image
newimg = bpy.data.images.load(NEWTEX)
clothing = next(o for o in bpy.data.objects if o.type=="MESH" and "clothing" in o.name.lower())
mat = clothing.material_slots[0].material
nt = mat.node_tree
bsdf = next(n for n in nt.nodes if n.type=="BSDF_PRINCIPLED")
tex = bsdf.inputs["Base Color"].links[0].from_node
print(f"[apply] replacing {tex.image.name} -> {NEWTEX}", flush=True)
tex.image = newimg

# export new glb
bpy.ops.export_scene.gltf(filepath=OUTGLB, export_format="GLB")
print(f"[apply] exported {OUTGLB}", flush=True)

# verify render (front, lit)
mn=Vector((1e9,)*3); mx=Vector((-1e9,)*3)
for o in bpy.data.objects:
    if o.type=="MESH":
        for c in o.bound_box:
            w=o.matrix_world@Vector(c)
            for k in range(3): mn[k]=min(mn[k],w[k]); mx[k]=max(mx[k],w[k])
H=mx.z-mn.z; cx=(mn.x+mx.x)/2; cy=(mn.y+mx.y)/2
T=Vector((cx,cy,mn.z+0.58*H)); loc=T+Vector((0,1,0))*1.30*H
cd=bpy.data.cameras.new("c"); cam=bpy.data.objects.new("c",cd); bpy.context.scene.collection.objects.link(cam)
cam.location=loc; cd.lens=60; cam.rotation_euler=(T-loc).normalized().to_track_quat('-Z','Y').to_euler()
bpy.context.scene.camera=cam
w=bpy.data.worlds.new("W"); w.use_nodes=True; w.node_tree.nodes["Background"].inputs[1].default_value=1.4
bpy.context.scene.world=w
sc=bpy.context.scene; sc.render.engine="CYCLES"
try: sc.cycles.device="GPU"
except: pass
sc.cycles.samples=24; sc.render.resolution_x=768; sc.render.resolution_y=1024
sc.render.filepath=RENDER; bpy.ops.render.render(write_still=True)
print("APPLY_DONE", flush=True)
