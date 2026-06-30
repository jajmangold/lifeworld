import bpy, math, os
from mathutils import Vector
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath="/work/avatar.glb")
# replace head base-color texture with the baked anchorM texture
mat = bpy.data.materials["head_Opaque_Material_Meshes_Material"]
bsdf = next(n for n in mat.node_tree.nodes if n.type=="BSDF_PRINCIPLED")
texnode = bsdf.inputs["Base Color"].links[0].from_node
newimg = bpy.data.images.load("/work/bake/_new_head.png")
newimg.colorspace_settings.name = "sRGB"
texnode.image = newimg
print("replaced head texture")
# even lighting
world=bpy.data.worlds.new("W"); bpy.context.scene.world=world; world.use_nodes=True
bg=world.node_tree.nodes.get("Background"); bg.inputs["Color"].default_value=(1,1,1,1); bg.inputs["Strength"].default_value=1.5
# head bounds
head=bpy.data.objects["head_Opaque_Material_Meshes_Mesh"]; deps=bpy.context.evaluated_depsgraph_get()
mn=Vector((1e9,1e9,1e9));mx=Vector((-1e9,-1e9,-1e9)); ev=head.evaluated_get(deps)
for c in ev.bound_box:
    w=ev.matrix_world@Vector(c)
    for i in range(3): mn[i]=min(mn[i],w[i]);mx[i]=max(mx[i],w[i])
cx=(mn.x+mx.x)/2; cz=(mn.z+mx.z)/2; H=mx.z-mn.z; cy=(mn.y+mx.y)/2
sc=bpy.context.scene; sc.render.engine="BLENDER_EEVEE_NEXT"; sc.eevee.taa_render_samples=32
sc.render.film_transparent=True; sc.render.resolution_x=512; sc.render.resolution_y=512
sc.view_settings.view_transform="Standard"; sc.render.image_settings.file_format="PNG"
cam_d=bpy.data.cameras.new("c"); cam_d.lens=60; cam=bpy.data.objects.new("c",cam_d); bpy.context.collection.objects.link(cam); sc.camera=cam
ctr=Vector((cx,cy,cz+H*0.10)); dist=H*1.6
for deg in [0,25,45]:
    a=math.radians(deg)
    cam.location=ctr+Vector((math.sin(a)*dist, math.cos(a)*dist, 0))
    d=(ctr-cam.location).normalized(); cam.rotation_euler=d.to_track_quat('-Z','Y').to_euler()
    sc.render.filepath=f"/work/bake/test_{deg}.png"; bpy.ops.render.render(write_still=True)
    print("WROTE test_"+str(deg))
