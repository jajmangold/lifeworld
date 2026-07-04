import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath="/work/avatar.glb")
# find all shape keys across meshes
drives={"eyeBlinkLeft":1.0,"eyeBlinkRight":1.0,"browInnerUp":0.8,
        "browOuterUpLeft":0.6,"browOuterUpRight":0.6,"mouthSmileLeft":0.5,
        "mouthSmileRight":0.5,"jawOpen":0.35}
hit={}
for o in bpy.data.objects:
    if o.type!="MESH" or not o.data.shape_keys: continue
    for kb in o.data.shape_keys.key_blocks:
        if kb.name in drives:
            kb.value=drives[kb.name]; hit[kb.name]=hit.get(kb.name,0)+1
print("DROVE:", hit)
# list all shape key names once (for reference)
names=set()
for o in bpy.data.objects:
    if o.type=="MESH" and o.data.shape_keys:
        for kb in o.data.shape_keys.key_blocks: names.add(kb.name)
print("ALLKEYS:", sorted(names))
import math
from mathutils import Vector
bpy.context.view_layer.update()
deps=bpy.context.evaluated_depsgraph_get()
mn=Vector((1e9,1e9,1e9));mx=Vector((-1e9,-1e9,-1e9))
for o in bpy.data.objects:
    if o.type!="MESH":continue
    ev=o.evaluated_get(deps)
    for c in ev.bound_box:
        w=ev.matrix_world@Vector(c)
        for i in range(3): mn[i]=min(mn[i],w[i]);mx[i]=max(mx[i],w[i])
cx=(mn.x+mx.x)/2;H=mx.z-mn.z;topz=mx.z
d=bpy.data.lights.new("k","AREA");d.energy=200;d.size=1.2
lo=bpy.data.objects.new("k",d);lo.location=(cx,mx.y+1.5,topz-H*0.08)
bpy.context.collection.objects.link(lo)
lo.rotation_euler=(Vector((cx,0,topz-H*0.10))-lo.location).normalized().to_track_quat('-Z','Y').to_euler()
cd=bpy.data.cameras.new("c");cd.lens=95
cam=bpy.data.objects.new("c",cd);bpy.context.collection.objects.link(cam)
cam.location=(cx,mx.y+H*0.34,topz-H*0.085)
cam.rotation_euler=(math.radians(90),0,math.radians(180))
bpy.context.scene.camera=cam
sc=bpy.context.scene
sc.render.engine="BLENDER_EEVEE_NEXT";sc.eevee.taa_render_samples=24
sc.render.resolution_x=512;sc.render.resolution_y=640
sc.view_settings.view_transform="Filmic"
sc.render.filepath="/work/output/anchor_env/bs_test.png"
bpy.ops.render.render(write_still=True)
print("WROTE bs_test")
