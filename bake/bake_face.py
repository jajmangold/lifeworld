import bpy, json, math, os
from mathutils import Vector
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath="/work/avatar.glb")
os.makedirs("/work/bake", exist_ok=True)
cam_info = json.load(open("/work/bake/cam.json"))

head = bpy.data.objects["head_Opaque_Material_Meshes_Mesh"]

# --- recreate the EXACT front camera used for the swap source ---
cam_d = bpy.data.cameras.new("c"); cam_d.lens = cam_info["lens"]
cam = bpy.data.objects.new("c", cam_d); bpy.context.collection.objects.link(cam)
cam.location = Vector(cam_info["loc"]); cam.rotation_euler = cam_info["rot"]
bpy.context.scene.camera = cam
bpy.context.view_layer.update()
cam_fwd = (cam.matrix_world.to_3x3() @ Vector((0,0,-1))).normalized()

# --- projected UV layer from the camera ---
head.data.uv_layers.new(name="Proj")
mod = head.modifiers.new("uvproj", 'UV_PROJECT')
mod.uv_layer = "Proj"; mod.projector_count = 1
mod.projectors[0].object = cam
mod.aspect_x = 1.0; mod.aspect_y = 1.0
# apply the modifier so Proj UV is baked into the mesh data
bpy.context.view_layer.objects.active = head
bpy.ops.object.modifier_apply(modifier="uvproj")

swap_img = bpy.data.images.load("/work/bake/_bake_swapped.png")

def bake_emission(node_color_setup, out_path, samples=4):
    # build a temp material: Emission(color) -> output, with an active target image node
    m = bpy.data.materials.new("bk"); m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    emit = nt.nodes.new("ShaderNodeEmission")
    nt.links.new(emit.outputs[0], out.inputs["Surface"])
    node_color_setup(nt, emit)
    target = bpy.data.images.new("tgt", 1024, 1024, alpha=True)
    timg = nt.nodes.new("ShaderNodeTexImage"); timg.image = target
    nt.nodes.active = timg; timg.select = True
    head.data.materials.clear(); head.data.materials.append(m)
    # render UV = UVMap (the head's native layout) is the active render layer
    head.data.uv_layers["UVMap"].active_render = True
    head.data.uv_layers["UVMap"].active = True
    sc = bpy.context.scene
    sc.render.engine = "CYCLES"; sc.cycles.samples = samples
    try: sc.cycles.device = "GPU"
    except: pass
    sc.render.bake.margin = 8
    bpy.ops.object.select_all(action='DESELECT'); head.select_set(True)
    bpy.context.view_layer.objects.active = head
    bpy.ops.object.bake(type='EMIT')
    target.filepath_raw = out_path; target.file_format = 'PNG'; target.save()
    print("BAKED", out_path)

# 1) bake the projected swapped FACE into UVMap space
def face_setup(nt, emit):
    tex = nt.nodes.new("ShaderNodeTexImage"); tex.image = swap_img; tex.extension = 'EXTEND'
    uv = nt.nodes.new("ShaderNodeUVMap"); uv.uv_map = "Proj"
    nt.links.new(uv.outputs["UV"], tex.inputs["Vector"])
    nt.links.new(tex.outputs["Color"], emit.inputs["Color"])
bake_emission(face_setup, "/work/bake/baked_face.png")

# 2) bake a FACING mask = max(0, dot(world_normal, cam_fwd)) into UVMap space
def mask_setup(nt, emit):
    geo = nt.nodes.new("ShaderNodeNewGeometry")
    dot = nt.nodes.new("ShaderNodeVectorMath"); dot.operation = 'DOT_PRODUCT'
    dot.inputs[1].default_value = (-cam_fwd.x, -cam_fwd.y, -cam_fwd.z)
    nt.links.new(geo.outputs["Normal"], dot.inputs[0])
    nt.links.new(dot.outputs["Value"], emit.inputs["Color"])
bake_emission(mask_setup, "/work/bake/baked_facing.png")
print("BAKE_DONE")
