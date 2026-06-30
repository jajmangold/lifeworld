"""Hair reproject bake (mirrors reproject.py, retargeted to the HEAD).
Render flat-lit head views -> Klein-edit the hair on each -> bake the edited views back onto the
head_Opaque base-color (Image_2) via Window-coord projection + facing weight. A host composite then
applies the change ONLY where Klein actually edited (diff mask) so the face/skin stay original and
only the hair region updates. Result is a new head texture used by the renderer's BAKED_HEAD_TEX hook
(keeps the per-frame face swap; just improves the hair).
  blender --background --python hair_reproject.py -- render|bake
"""
import bpy, sys, math, json, os
from mathutils import Vector

argv = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
MODE = argv[0] if argv else "render"
WORK = "/work"
GLB  = f"{WORK}/avatar.glb"
VIEWDIR = f"{WORK}/hairproj"
os.makedirs(VIEWDIR, exist_ok=True)
RES = 1024
VIEWS = [("front", 0.0), ("left", 35.0), ("right", -35.0)]   # fixed-camera anchor: front + small turns

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=GLB)

# head_Opaque mesh + its base-color image
head = None
for o in bpy.data.objects:
    if o.type == "MESH" and "head" in o.name.lower() and "opaque" in o.name.lower():
        head = o
if head is None:
    for o in bpy.data.objects:
        if o.type == "MESH" and "head_Opaque" in o.name:
            head = o
print(f"[hair] head mesh = {head.name if head else None}", flush=True)

# bbox of the head mesh for framing
mn = Vector((1e9,)*3); mx = Vector((-1e9,)*3)
for c in head.bound_box:
    w = head.matrix_world @ Vector(c)
    for k in range(3):
        mn[k] = min(mn[k], w[k]); mx[k] = max(mx[k], w[k])
H = mx.z - mn.z
cx = (mn.x+mx.x)/2; cy = (mn.y+mx.y)/2
TARGET = Vector((cx, cy, mn.z + 0.62*H))     # head/hair center
R = 0.95*(mx.z-mn.z) + 0.35
print(f"[hair] target={tuple(round(v,3) for v in TARGET)} R={R:.3f}", flush=True)

def make_cam(name, angle_deg):
    a = math.radians(angle_deg)
    loc = TARGET + Vector((math.sin(a), math.cos(a), 0.0)) * R
    cd = bpy.data.cameras.new(name); cam = bpy.data.objects.new(name, cd)
    bpy.context.scene.collection.objects.link(cam)
    cam.location = loc; cd.lens = 70
    cam.rotation_euler = (TARGET - loc).normalized().to_track_quat('-Z', 'Y').to_euler()
    return cam

cams = {name: make_cam(name, ang) for name, ang in VIEWS}
sc = bpy.context.scene
sc.render.resolution_x = RES; sc.render.resolution_y = RES
sc.render.film_transparent = False
sc.render.engine = "CYCLES"
try: sc.cycles.device = "GPU"
except Exception: pass

if MODE == "render":
    w = bpy.data.worlds.new("W"); w.use_nodes = True
    bg = w.node_tree.nodes["Background"]
    bg.inputs["Color"].default_value = (1, 1, 1, 1)                 # NEW world defaults to dark grey 0.05!
    bg.inputs["Strength"].default_value = 1.0                      # even ~albedo
    sc.world = w
    sc.view_settings.view_transform = "Standard"                    # raw albedo, no Filmic darkening
    sc.cycles.samples = 12
    for name, _ in VIEWS:
        sc.camera = cams[name]
        sc.render.filepath = f"{VIEWDIR}/view_{name}.png"
        bpy.ops.render.render(write_still=True)
        print(f"[hair] rendered {name}", flush=True)
    json.dump({"views":[v[0] for v in VIEWS]}, open(f"{VIEWDIR}/meta.json","w"))
    print("RENDER_DONE", flush=True)

elif MODE == "bake":
    sc.cycles.samples = 1
    sc.render.bake.use_pass_direct = False
    sc.render.bake.use_pass_indirect = False
    sc.render.bake.margin = 8
    obj = head
    mat = obj.material_slots[0].material; mat.use_nodes = True; nt = mat.node_tree

    def clear(nt):
        for n in list(nt.nodes): nt.nodes.remove(n)
    def add_target(nt, img):
        tex = nt.nodes.new("ShaderNodeTexImage"); tex.image = img; nt.nodes.active = tex; return tex
    def setup_color(nt, edited_img):
        clear(nt)
        out = nt.nodes.new("ShaderNodeOutputMaterial"); emit = nt.nodes.new("ShaderNodeEmission")
        texc = nt.nodes.new("ShaderNodeTexCoord")
        img = nt.nodes.new("ShaderNodeTexImage"); img.image = edited_img; img.extension = "EXTEND"
        nt.links.new(texc.outputs["Window"], img.inputs["Vector"])
        nt.links.new(img.outputs["Color"], emit.inputs["Color"])
        nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])
    def setup_weight(nt, cam_loc):
        clear(nt)
        out = nt.nodes.new("ShaderNodeOutputMaterial"); emit = nt.nodes.new("ShaderNodeEmission")
        geo = nt.nodes.new("ShaderNodeNewGeometry")
        camv = nt.nodes.new("ShaderNodeCombineXYZ")
        camv.inputs[0].default_value, camv.inputs[1].default_value, camv.inputs[2].default_value = cam_loc
        sub = nt.nodes.new("ShaderNodeVectorMath"); sub.operation = "SUBTRACT"
        nt.links.new(camv.outputs[0], sub.inputs[0]); nt.links.new(geo.outputs["Position"], sub.inputs[1])
        norm = nt.nodes.new("ShaderNodeVectorMath"); norm.operation = "NORMALIZE"
        nt.links.new(sub.outputs[0], norm.inputs[0])
        dot = nt.nodes.new("ShaderNodeVectorMath"); dot.operation = "DOT_PRODUCT"
        nt.links.new(norm.outputs[0], dot.inputs[0]); nt.links.new(geo.outputs["Normal"], dot.inputs[1])
        clamp = nt.nodes.new("ShaderNodeClamp"); nt.links.new(dot.outputs["Value"], clamp.inputs["Value"])
        pw = nt.nodes.new("ShaderNodeMath"); pw.operation = "POWER"; pw.inputs[1].default_value = 3.0
        nt.links.new(clamp.outputs[0], pw.inputs[0]); nt.links.new(pw.outputs[0], emit.inputs["Color"])
        nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])

    for name, _ in VIEWS:
        sc.camera = cams[name]
        edited = bpy.data.images.load(f"{VIEWDIR}/edited_{name}.png")
        cam_loc = list(cams[name].location)
        col_img = bpy.data.images.new(f"col_{name}", RES, RES, alpha=False)
        setup_color(nt, edited); add_target(nt, col_img)
        bpy.ops.object.select_all(action="DESELECT"); obj.select_set(True); bpy.context.view_layer.objects.active = obj
        bpy.ops.object.bake(type="EMIT")
        col_img.filepath_raw = f"{VIEWDIR}/bake_col_{name}.png"; col_img.file_format = "PNG"; col_img.save()
        w_img = bpy.data.images.new(f"w_{name}", RES, RES, alpha=False)
        setup_weight(nt, cam_loc); add_target(nt, w_img)
        bpy.ops.object.bake(type="EMIT")
        w_img.filepath_raw = f"{VIEWDIR}/bake_w_{name}.png"; w_img.file_format = "PNG"; w_img.save()
        print(f"[hair] baked {name}", flush=True)
    print("BAKE_DONE", flush=True)

elif MODE == "frontbake":
    # Single front-camera projection bake (preserves Klein detail; right for a fixed frontal anchor).
    # Projects /work/hairproj/edited_front.png onto the head UVMap, + a facing mask, like bake_face.py.
    cam = cams["front"]; sc.camera = cam
    bpy.context.view_layer.update()
    cam_fwd = (cam.matrix_world.to_3x3() @ Vector((0, 0, -1))).normalized()
    # dump ORIGINAL head base-color (Image_2) for the host composite
    mat0 = head.material_slots[0].material
    orig = None
    for n in mat0.node_tree.nodes:
        if n.type == "TEX_IMAGE" and n.image is not None:
            orig = n.image; break
    if orig is not None:
        orig.filepath_raw = f"{VIEWDIR}/orig_head.png"; orig.file_format = "PNG"; orig.save()
        print(f"[hair] orig head tex {orig.size[0]}x{orig.size[1]}", flush=True)
    TEX = orig.size[0] if orig is not None else 1024
    # projected UV from the front camera
    head.data.uv_layers.new(name="Proj")
    mod = head.modifiers.new("uvproj", 'UV_PROJECT'); mod.uv_layer = "Proj"
    mod.projector_count = 1; mod.projectors[0].object = cam; mod.aspect_x = 1.0; mod.aspect_y = 1.0
    bpy.context.view_layer.objects.active = head
    bpy.ops.object.modifier_apply(modifier="uvproj")
    edited = bpy.data.images.load(f"{VIEWDIR}/edited_front.png")

    def bake_emit(setup, out_path):
        m = bpy.data.materials.new("bk"); m.use_nodes = True; nt = m.node_tree
        for n in list(nt.nodes): nt.nodes.remove(n)
        out = nt.nodes.new("ShaderNodeOutputMaterial"); emit = nt.nodes.new("ShaderNodeEmission")
        nt.links.new(emit.outputs[0], out.inputs["Surface"]); setup(nt, emit)
        target = bpy.data.images.new("tgt", TEX, TEX, alpha=True)
        timg = nt.nodes.new("ShaderNodeTexImage"); timg.image = target
        nt.nodes.active = timg; timg.select = True
        head.data.materials.clear(); head.data.materials.append(m)
        head.data.uv_layers["UVMap"].active_render = True; head.data.uv_layers["UVMap"].active = True
        sc.cycles.samples = 4; sc.render.bake.margin = 8
        bpy.ops.object.select_all(action='DESELECT'); head.select_set(True)
        bpy.context.view_layer.objects.active = head
        bpy.ops.object.bake(type='EMIT')
        target.filepath_raw = out_path; target.file_format = 'PNG'; target.save()

    def col_setup(nt, emit):
        tex = nt.nodes.new("ShaderNodeTexImage"); tex.image = edited; tex.extension = 'EXTEND'
        uv = nt.nodes.new("ShaderNodeUVMap"); uv.uv_map = "Proj"
        nt.links.new(uv.outputs["UV"], tex.inputs["Vector"]); nt.links.new(tex.outputs["Color"], emit.inputs["Color"])
    def mask_setup(nt, emit):
        geo = nt.nodes.new("ShaderNodeNewGeometry")
        dot = nt.nodes.new("ShaderNodeVectorMath"); dot.operation = 'DOT_PRODUCT'
        dot.inputs[1].default_value = (-cam_fwd.x, -cam_fwd.y, -cam_fwd.z)
        nt.links.new(geo.outputs["Normal"], dot.inputs[0]); nt.links.new(dot.outputs["Value"], emit.inputs["Color"])
    bake_emit(col_setup, f"{VIEWDIR}/baked_front.png")
    bake_emit(mask_setup, f"{VIEWDIR}/baked_facing.png")
    print("FRONTBAKE_DONE", flush=True)
