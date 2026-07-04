import bpy, sys, math, json, os
from mathutils import Vector

argv = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
MODE = argv[0] if argv else "render"          # "render" or "bake"
WORK = "/work"
GLB  = f"{WORK}/avatar.glb"
VIEWDIR = f"{WORK}/reproj"
os.makedirs(VIEWDIR, exist_ok=True)
RES = 1024
# orbit angles (deg) around vertical: 0=front, +/-40 sides, 180 back
VIEWS = [("front", 0.0), ("left", 40.0), ("right", -40.0), ("back", 180.0)]

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=GLB)

# ---- bounding box over mesh objects ----
mn = Vector((1e9,)*3); mx = Vector((-1e9,)*3)
clothing = None
for o in bpy.data.objects:
    if o.type == "MESH":
        if "clothing" in o.name.lower():
            clothing = o
        for c in o.bound_box:
            w = o.matrix_world @ Vector(c)
            for k in range(3):
                mn[k] = min(mn[k], w[k]); mx[k] = max(mx[k], w[k])
H = mx.z - mn.z
cx = (mn.x+mx.x)/2; cy = (mn.y+mx.y)/2
TARGET = Vector((cx, cy, mn.z + 0.58*H))      # upper torso
R = 1.30*H
print(f"[rp] H={H:.3f} target={tuple(round(v,3) for v in TARGET)} clothing={clothing.name if clothing else None}", flush=True)

def make_cam(name, angle_deg):
    a = math.radians(angle_deg)
    # front (a=0) on +Y side looking toward -Y
    loc = TARGET + Vector((math.sin(a), math.cos(a), 0.0)) * R
    cd = bpy.data.cameras.new(name); cam = bpy.data.objects.new(name, cd)
    bpy.context.scene.collection.objects.link(cam)
    cam.location = loc; cd.lens = 60
    # aim at TARGET
    direction = (TARGET - loc).normalized()
    cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()
    return cam

cams = {name: make_cam(name, ang) for name, ang in VIEWS}
sc = bpy.context.scene
sc.render.resolution_x = RES; sc.render.resolution_y = RES
sc.render.film_transparent = False

if MODE == "render":
    # flat even lighting ~ albedo
    w = bpy.data.worlds.new("W"); w.use_nodes = True
    w.node_tree.nodes["Background"].inputs[1].default_value = 1.6
    sc.world = w
    sc.render.engine = "CYCLES"
    try: sc.cycles.device = "GPU"
    except Exception: pass
    sc.cycles.samples = 8
    print(f"[rp] engine={sc.render.engine}", flush=True)
    for name, _ in VIEWS:
        sc.camera = cams[name]
        sc.render.filepath = f"{VIEWDIR}/view_{name}.png"
        bpy.ops.render.render(write_still=True)
        print(f"[rp] rendered {name}", flush=True)
    json.dump({"views":[v[0] for v in VIEWS]}, open(f"{VIEWDIR}/meta.json","w"))
    print("RENDER_DONE", flush=True)

elif MODE == "bake":
    import bpy
    sc.render.engine = "CYCLES"
    try: sc.cycles.device = "GPU"
    except: pass
    sc.cycles.samples = 1
    sc.render.bake.use_pass_direct = False
    sc.render.bake.use_pass_indirect = False
    sc.render.bake.margin = 8

    obj = clothing
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    # ensure single material with nodes; we rewrite its node tree per pass
    mat = obj.material_slots[0].material
    mat.use_nodes = True
    nt = mat.node_tree

    def clear(nt):
        for n in list(nt.nodes): nt.nodes.remove(n)

    def add_bake_target(nt, img):
        tex = nt.nodes.new("ShaderNodeTexImage"); tex.image = img
        nt.nodes.active = tex
        return tex

    def setup_color(nt, edited_img):
        clear(nt)
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        emit = nt.nodes.new("ShaderNodeEmission")
        texc = nt.nodes.new("ShaderNodeTexCoord")
        img = nt.nodes.new("ShaderNodeTexImage"); img.image = edited_img; img.extension = "EXTEND"
        nt.links.new(texc.outputs["Window"], img.inputs["Vector"])
        nt.links.new(img.outputs["Color"], emit.inputs["Color"])
        nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])

    def setup_weight(nt, cam_loc):
        clear(nt)
        out = nt.nodes.new("ShaderNodeOutputMaterial")
        emit = nt.nodes.new("ShaderNodeEmission")
        geo = nt.nodes.new("ShaderNodeNewGeometry")
        camv = nt.nodes.new("ShaderNodeCombineXYZ")
        camv.inputs[0].default_value = cam_loc[0]; camv.inputs[1].default_value = cam_loc[1]; camv.inputs[2].default_value = cam_loc[2]
        sub = nt.nodes.new("ShaderNodeVectorMath"); sub.operation = "SUBTRACT"
        nt.links.new(camv.outputs[0], sub.inputs[0]); nt.links.new(geo.outputs["Position"], sub.inputs[1])
        norm = nt.nodes.new("ShaderNodeVectorMath"); norm.operation = "NORMALIZE"
        nt.links.new(sub.outputs[0], norm.inputs[0])
        dot = nt.nodes.new("ShaderNodeVectorMath"); dot.operation = "DOT_PRODUCT"
        nt.links.new(norm.outputs[0], dot.inputs[0]); nt.links.new(geo.outputs["Normal"], dot.inputs[1])
        clamp = nt.nodes.new("ShaderNodeClamp")
        nt.links.new(dot.outputs["Value"], clamp.inputs["Value"])
        pw = nt.nodes.new("ShaderNodeMath"); pw.operation = "POWER"; pw.inputs[1].default_value = 3.0
        nt.links.new(clamp.outputs[0], pw.inputs[0])
        nt.links.new(pw.outputs[0], emit.inputs["Color"])
        nt.links.new(emit.outputs["Emission"], out.inputs["Surface"])

    for name, _ in VIEWS:
        sc.camera = cams[name]                      # CRITICAL: Window coords project from this camera
        edited = bpy.data.images.load(f"{VIEWDIR}/edited_{name}.png")
        cam_loc = list(cams[name].location)
        # color bake
        col_img = bpy.data.images.new(f"col_{name}", RES, RES, alpha=False)
        setup_color(nt, edited); add_bake_target(nt, col_img)
        bpy.context.view_layer.objects.active = obj; obj.select_set(True)
        bpy.ops.object.bake(type="EMIT")
        col_img.filepath_raw = f"{VIEWDIR}/bake_col_{name}.png"; col_img.file_format = "PNG"; col_img.save()
        # weight bake
        w_img = bpy.data.images.new(f"w_{name}", RES, RES, alpha=False)
        setup_weight(nt, cam_loc); add_bake_target(nt, w_img)
        bpy.ops.object.bake(type="EMIT")
        w_img.filepath_raw = f"{VIEWDIR}/bake_w_{name}.png"; w_img.file_format = "PNG"; w_img.save()
        print(f"[rp] baked {name}", flush=True)
    print("BAKE_DONE", flush=True)
