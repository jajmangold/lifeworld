import bpy, math, os
from mathutils import Vector
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath="/work/avatar.glb")
os.makedirs("/work/bake", exist_ok=True)

# bounds of the HEAD mesh (face) for framing
head = bpy.data.objects["head_Opaque_Material_Meshes_Mesh"]
deps = bpy.context.evaluated_depsgraph_get()
mn = Vector((1e9,1e9,1e9)); mx = Vector((-1e9,-1e9,-1e9))
ev = head.evaluated_get(deps)
for c in ev.bound_box:
    w = ev.matrix_world @ Vector(c)
    for i in range(3): mn[i]=min(mn[i],w[i]); mx[i]=max(mx[i],w[i])
cx=(mn.x+mx.x)/2; cz=(mn.z+mx.z)/2; H=mx.z-mn.z
print("HEAD bounds z", round(mn.z,3), round(mx.z,3), "H", round(H,3), "y", round(mn.y,3), round(mx.y,3))

# EVEN lighting: bright uniform world (albedo-ish), no directional shadows
world = bpy.data.worlds.new("W"); bpy.context.scene.world = world
world.use_nodes = True; bg = world.node_tree.nodes.get("Background")
bg.inputs["Color"].default_value = (1,1,1,1); bg.inputs["Strength"].default_value = 1.6

# Camera: front (+Y), tight on the face
cam_d = bpy.data.cameras.new("c"); cam_d.lens = 55
cam = bpy.data.objects.new("c", cam_d); bpy.context.collection.objects.link(cam)
aimz = cz + H*0.10
cam.location = (cx, mx.y + H*1.45, aimz)
cam.rotation_euler = (math.radians(90), 0, math.radians(180))
bpy.context.scene.camera = cam

sc = bpy.context.scene
sc.render.engine = "BLENDER_EEVEE_NEXT"; sc.eevee.taa_render_samples = 32
sc.render.film_transparent = True   # transparent bg so the face/silhouette is clean
sc.render.resolution_x = 1024; sc.render.resolution_y = 1024
sc.view_settings.view_transform = "Standard"   # no tonemap -> closer to albedo
sc.render.image_settings.file_format = "PNG"; sc.render.image_settings.color_mode = "RGBA"
sc.render.filepath = "/work/bake/frontal.png"
bpy.ops.render.render(write_still=True)
print("WROTE /work/bake/frontal.png")
# also save the camera pose for the projection-bake step
import json
json.dump({"loc": list(cam.location), "rot": list(cam.rotation_euler), "lens": cam_d.lens,
           "res": 1024, "aimz": aimz}, open("/work/bake/cam.json","w"))
print("WROTE cam.json")
