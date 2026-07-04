import bpy, sys
from mathutils import Vector
argv = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
HEAD, BODY, SRCGLB, OUTGLB, RENDER = argv[0], argv[1], argv[2], argv[3], argv[4]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=SRCGLB)

def replace_basecolor(mesh_key, newpath):
    obj = next(o for o in bpy.data.objects if o.type=="MESH" and mesh_key in o.name.lower())
    mat = obj.material_slots[0].material; nt = mat.node_tree
    bsdf = next(n for n in nt.nodes if n.type=="BSDF_PRINCIPLED")
    tex = bsdf.inputs["Base Color"].links[0].from_node
    print(f"[skin] {mesh_key}: {tex.image.name}{tuple(tex.image.size)} -> {newpath}", flush=True)
    tex.image = bpy.data.images.load(newpath)

replace_basecolor("head", HEAD)   # head_Opaque (face/eyes/mouth)
replace_basecolor("body", BODY)   # body_Opaque (arms/hands/neck)

bpy.ops.export_scene.gltf(filepath=OUTGLB, export_format="GLB")
print(f"[skin] exported {OUTGLB}", flush=True)

# verify render (front, lit) framed on head/torso
mn=Vector((1e9,)*3); mx=Vector((-1e9,)*3)
for o in bpy.data.objects:
    if o.type=="MESH":
        for c in o.bound_box:
            w=o.matrix_world@Vector(c)
            for k in range(3): mn[k]=min(mn[k],w[k]); mx[k]=max(mx[k],w[k])
H=mx.z-mn.z; cx=(mn.x+mx.x)/2; cy=(mn.y+mx.y)/2
T=Vector((cx,cy,mn.z+0.86*H)); loc=T+Vector((0,1,0))*0.55*H   # head closeup
cd=bpy.data.cameras.new("c"); cam=bpy.data.objects.new("c",cd); bpy.context.scene.collection.objects.link(cam)
cam.location=loc; cd.lens=80; cam.rotation_euler=(T-loc).normalized().to_track_quat('-Z','Y').to_euler()
bpy.context.scene.camera=cam
w=bpy.data.worlds.new("W"); w.use_nodes=True; w.node_tree.nodes["Background"].inputs[1].default_value=1.4
bpy.context.scene.world=w
sc=bpy.context.scene; sc.render.engine="CYCLES"
try: sc.cycles.device="GPU"
except: pass
sc.cycles.samples=48; sc.render.resolution_x=900; sc.render.resolution_y=900
sc.render.filepath=RENDER; bpy.ops.render.render(write_still=True)
print("SKIN_DONE", flush=True)
