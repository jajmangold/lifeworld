"""Render -> (Klein-edit externally) -> reproject a character's GARMENT/skin texture. The right way to
retexture with Klein: Klein edits a RENDER (where it can see the suit), not the abstract UV atlas. This
renders flat-lit ~albedo views, and (after Klein edits them) bakes the edited views back onto the target
material's UV via camera Window-projection + a per-view facing weight, so wardrobe.py can blend them over
the original map (restyle only what's visible; keep the back/unseen regions original).

Generalized from viverse_avatar/reproject.py for any BlenderKit .blend + a named material.
  blender -b --python reproject_garment.py -- render <char.blend> <Material> <viewdir>
  blender -b --python reproject_garment.py -- bake   <char.blend> <Material> <viewdir>
render -> view_<v>.png (edit these -> edited_<v>.png). bake -> bake_col_<v>.png + bake_w_<v>.png.
"""
import bpy, sys, math, json, os
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
MODE, BLEND, MATERIAL, VIEWDIR = argv[0], argv[1], argv[2], argv[3]
os.makedirs(VIEWDIR, exist_ok=True)
RES = 1024
VIEWS = [("front", 0.0), ("left", 38.0), ("right", -38.0)]     # front-facing garment: 3 views cover the seen area

bpy.ops.wm.open_mainfile(filepath=BLEND)
for o in list(bpy.data.objects):
    if o.type in ("CAMERA", "LIGHT"): bpy.data.objects.remove(o, do_unlink=True)

# ALL meshes using MATERIAL (a garment material is often split across several separate mesh objects sharing one
# atlas -- e.g. jacket/pants/buttons/shoes -- baking only the first/last match silently missed the rest); bbox
# over all meshes for framing.
garments = []; mn = Vector((1e9,)*3); mx = Vector((-1e9,)*3)
for o in bpy.data.objects:
    if o.type != "MESH": continue
    if any(s.material and MATERIAL.lower() in s.material.name.lower() for s in o.material_slots):
        garments.append(o)
    for c in o.bound_box:
        w = o.matrix_world @ Vector(c)
        for k in range(3): mn[k] = min(mn[k], w[k]); mx[k] = max(mx[k], w[k])
if not garments: sys.exit(f"no mesh with material '{MATERIAL}'")
target = garments[0]
H = mx.z - mn.z; cx = (mn.x + mx.x) / 2; cy = (mn.y + mx.y) / 2
TARGET = Vector((cx, cy, mn.z + 0.60 * H)); R = 1.15 * H
side = float(os.environ.get("NEWS_CAMSIDE", "-1"))             # BlenderKit/Rigify faces -Y
print(f"[rg] H={H:.3f} target={tuple(round(v,3) for v in TARGET)} meshes={[o.name for o in garments]} side={side}", flush=True)

def make_cam(name, angle_deg):
    a = math.radians(angle_deg)
    face = Vector((math.sin(a), side * math.cos(a), 0.0))      # camera on the character's FRONT side (-Y for side<0)
    loc = TARGET + face * R
    cd = bpy.data.cameras.new(name); cam = bpy.data.objects.new(name, cd)
    bpy.context.scene.collection.objects.link(cam)
    cam.location = loc; cd.lens = 62
    cam.rotation_euler = (TARGET - loc).normalized().to_track_quat('-Z', 'Y').to_euler()
    return cam

cams = {n: make_cam(n, a) for n, a in VIEWS}
sc = bpy.context.scene
sc.render.resolution_x = RES; sc.render.resolution_y = RES; sc.render.film_transparent = False
sc.render.engine = "CYCLES"
try:
    sc.cycles.device = "GPU"
    pr = bpy.context.preferences.addons["cycles"].preferences; pr.compute_device_type = "CUDA"; pr.get_devices()
    for d in pr.devices: d.use = (d.type == "CUDA")
except Exception: pass

if MODE == "render":
    w = bpy.data.worlds.new("W"); w.use_nodes = True                  # flat even light ~ albedo
    w.node_tree.nodes["Background"].inputs[1].default_value = 1.5; sc.world = w
    sc.cycles.samples = 12
    for name, _ in VIEWS:
        sc.camera = cams[name]; sc.render.filepath = f"{VIEWDIR}/view_{name}.png"
        bpy.ops.render.render(write_still=True); print(f"[rg] rendered {name}", flush=True)
    json.dump({"views": [v[0] for v in VIEWS]}, open(f"{VIEWDIR}/meta.json", "w"))
    print("RENDER_DONE", flush=True)

elif MODE == "bake":
    # NOTE: ShaderNodeTexCoord "Window" output is NOT usable during a Cycles bake (baking rasterizes UV-space,
    # there is no per-bake camera/screen projection) -- it silently produced near-empty bakes. The robust fix is
    # a UV Project modifier per view: it projects the camera onto a REAL secondary UV layer (computed once, up
    # front, from the camera's actual perspective) which an ShaderNodeUVMap node can then sample during the bake.
    sc.cycles.samples = 32                  # AA the bake -- fine pinstripe patterns alias badly at 1spp through
                                             # the curved/steep-angle UV-project mapping (jacket lapels/sleeves)
    sc.render.bake.use_pass_direct = sc.render.bake.use_pass_indirect = False
    sc.render.bake.margin = 8
    obj = target
    mat = obj.material_slots[[i for i, s in enumerate(obj.material_slots)
                              if s.material and MATERIAL.lower() in s.material.name.lower()][0]].material
    mat.use_nodes = True; nt = mat.node_tree
    base_uv = obj.data.uv_layers.active.name if obj.data.uv_layers.active else obj.data.uv_layers[0].name

    bpy.ops.object.select_all(action="DESELECT")
    for o in garments: o.select_set(True)
    bpy.context.view_layer.objects.active = obj

    def make_proj_uv(name):
        # every garment mesh gets its OWN projected UV layer (same name), so the one shared shader node tree
        # resolves "proj_<name>" per-object when all garment meshes are baked together in a single call.
        uv_name = f"proj_{name}"
        for o in garments:
            if uv_name in o.data.uv_layers: o.data.uv_layers.remove(o.data.uv_layers[uv_name])
            o.data.uv_layers.new(name=uv_name)
            bpy.ops.object.select_all(action="DESELECT"); o.select_set(True); bpy.context.view_layer.objects.active = o
            mod = o.modifiers.new(f"UVProj_{name}", "UV_PROJECT")
            mod.uv_layer = uv_name; mod.projector_count = 1
            mod.projectors[0].object = cams[name]
            mod.aspect_x = mod.aspect_y = 1.0; mod.scale_x = mod.scale_y = 1.0
            bpy.ops.object.modifier_apply(modifier=mod.name)          # bakes the camera projection into real UV coords
            ob = o.data.uv_layers.get(base_uv) or o.data.uv_layers[0]
            o.data.uv_layers.active = ob                              # bake output must target the ORIGINAL atlas UV
        bpy.ops.object.select_all(action="DESELECT")
        for o in garments: o.select_set(True)
        bpy.context.view_layer.objects.active = obj
        return uv_name

    def clear(): [nt.nodes.remove(n) for n in list(nt.nodes)]
    def tgt(img): t = nt.nodes.new("ShaderNodeTexImage"); t.image = img; nt.nodes.active = t
    def setup_color(edited, uv_name):
        clear(); out = nt.nodes.new("ShaderNodeOutputMaterial"); emit = nt.nodes.new("ShaderNodeEmission")
        uvm = nt.nodes.new("ShaderNodeUVMap"); uvm.uv_map = uv_name
        im = nt.nodes.new("ShaderNodeTexImage"); im.image = edited; im.extension = "EXTEND"
        nt.links.new(uvm.outputs["UV"], im.inputs["Vector"])
        nt.links.new(im.outputs["Color"], emit.inputs["Color"]); nt.links.new(emit.outputs[0], out.inputs["Surface"])
    def setup_weight(cam_loc):
        clear(); out = nt.nodes.new("ShaderNodeOutputMaterial"); emit = nt.nodes.new("ShaderNodeEmission")
        geo = nt.nodes.new("ShaderNodeNewGeometry"); camv = nt.nodes.new("ShaderNodeCombineXYZ")
        for i in range(3): camv.inputs[i].default_value = cam_loc[i]
        sub = nt.nodes.new("ShaderNodeVectorMath"); sub.operation = "SUBTRACT"
        nt.links.new(camv.outputs[0], sub.inputs[0]); nt.links.new(geo.outputs["Position"], sub.inputs[1])
        nrm = nt.nodes.new("ShaderNodeVectorMath"); nrm.operation = "NORMALIZE"; nt.links.new(sub.outputs[0], nrm.inputs[0])
        dot = nt.nodes.new("ShaderNodeVectorMath"); dot.operation = "DOT_PRODUCT"
        nt.links.new(nrm.outputs[0], dot.inputs[0]); nt.links.new(geo.outputs["Normal"], dot.inputs[1])
        cl = nt.nodes.new("ShaderNodeClamp"); nt.links.new(dot.outputs["Value"], cl.inputs["Value"])
        pw = nt.nodes.new("ShaderNodeMath"); pw.operation = "POWER"; pw.inputs[1].default_value = 3.0
        nt.links.new(cl.outputs[0], pw.inputs[0]); nt.links.new(pw.outputs[0], emit.inputs["Color"])
        nt.links.new(emit.outputs[0], out.inputs["Surface"])
    for name, _ in VIEWS:
        uv_name = make_proj_uv(name)
        edited = bpy.data.images.load(f"{VIEWDIR}/edited_{name}.png")
        col = bpy.data.images.new(f"col_{name}", RES, RES); setup_color(edited, uv_name); tgt(col)
        bpy.ops.object.bake(type="EMIT")
        col.filepath_raw = f"{VIEWDIR}/bake_col_{name}.png"; col.file_format = "PNG"; col.save()
        wi = bpy.data.images.new(f"w_{name}", RES, RES); setup_weight(list(cams[name].location)); tgt(wi)
        bpy.ops.object.bake(type="EMIT")
        wi.filepath_raw = f"{VIEWDIR}/bake_w_{name}.png"; wi.file_format = "PNG"; wi.save()
        print(f"[rg] baked {name}", flush=True)
    print("BAKE_DONE", flush=True)
