import bpy, sys
from mathutils import Vector
GLB=sys.argv[sys.argv.index("--")+1] if "--" in sys.argv else "/work/avatar_hires.glb"
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=GLB)
mn=Vector((1e9,)*3); mx=Vector((-1e9,)*3)
for o in bpy.data.objects:
    if o.type=="MESH":
        for c in o.bound_box:
            w=o.matrix_world@Vector(c)
            for k in range(3): mn[k]=min(mn[k],w[k]); mx[k]=max(mx[k],w[k])
H=mx.z-mn.z; cx=(mn.x+mx.x)/2; cy=(mn.y+mx.y)/2
T=Vector((cx,cy,mn.z+0.968*H)); loc=T+Vector((0,1,0))*0.20*H
cd=bpy.data.cameras.new("c"); cam=bpy.data.objects.new("c",cd); bpy.context.scene.collection.objects.link(cam)
cam.location=loc; cd.lens=95; cam.rotation_euler=(T-loc).normalized().to_track_quat('-Z','Y').to_euler()
bpy.context.scene.camera=cam
w=bpy.data.worlds.new("W"); w.use_nodes=True; w.node_tree.nodes["Background"].inputs[1].default_value=1.5
bpy.context.scene.world=w
sc=bpy.context.scene; sc.render.engine="CYCLES"
try: sc.cycles.device="GPU"
except: pass
sc.cycles.samples=64; sc.render.resolution_x=1100; sc.render.resolution_y=500
sc.render.filepath="/h/eye_closeup.png"; bpy.ops.render.render(write_still=True)
print("EYE_DONE")
