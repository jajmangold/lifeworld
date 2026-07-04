import bpy, math, mathutils, json
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath="/work/avatar.glb")
sk=set(); meshes=[]
for o in bpy.data.objects:
    if o.type=="MESH":
        meshes.append(o.name)
        if o.data.shape_keys:
            for k in o.data.shape_keys.key_blocks: sk.add(k.name)
print("MESHES:", meshes)
print("BLENDSHAPES(%d):"%len(sk), json.dumps(sorted(sk)))
arms=[o for o in bpy.data.objects if o.type=="ARMATURE"]
if arms:
    eb=[b.name for b in arms[0].pose.bones if "eye" in b.name.lower()]
    print("BONES:", len(arms[0].pose.bones), "EYE_BONES:", eb)
# render head-shoulders front
mn=mathutils.Vector((1e9,)*3); mx=mathutils.Vector((-1e9,)*3)
for o in bpy.data.objects:
    if o.type=="MESH":
        for c in o.bound_box:
            w=o.matrix_world@mathutils.Vector(c)
            for k in range(3): mn[k]=min(mn[k],w[k]); mx[k]=max(mx[k],w[k])
H=mx.z-mn.z; cx=(mn.x+mx.x)/2; topz=mx.z
cd=bpy.data.cameras.new("c"); cam=bpy.data.objects.new("c",cd); bpy.context.scene.collection.objects.link(cam)
# VRM faces +Y or -Y; try front along -Y
cam.location=(cx, mx.y+H*0.45, topz-H*0.12); cam.rotation_euler=(math.radians(90),0,math.radians(180)); cd.lens=75; bpy.context.scene.camera=cam
l=bpy.data.lights.new("L","AREA"); l.energy=400; l.size=2; lo=bpy.data.objects.new("L",l); bpy.context.scene.collection.objects.link(lo); lo.location=(cx-H*0.3,mx.y+H*0.6,topz); lo.rotation_euler=(math.radians(115),0,0)
w=bpy.data.worlds.new("W"); w.use_nodes=True; w.node_tree.nodes["Background"].inputs[1].default_value=1.2; bpy.context.scene.world=w
sc=bpy.context.scene; sc.render.engine="CYCLES"; sc.cycles.samples=16; sc.render.resolution_x=512; sc.render.resolution_y=640
sc.render.filepath="/work/avatar_front.png"; bpy.ops.render.render(write_still=True)
print("AVATAR_RENDER_DONE")
