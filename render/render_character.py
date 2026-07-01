"""Generic character renderer for a BlenderKit (Rigify) rigged .blend — the viverse-free path.
Loads the character, frames a medium-close-up, lights it with an HDRI (studio or field pano), drives
blinks/brows from the ARKit perf + subtle idle head motion, renders premultiplied frames for the
premult-over composite. MuseTalk owns the mouth downstream. A face-swap (photoreal identity) IS applied like the anchor path:
the BlenderKit face is semi-stylized, so swap brings a photoreal identity + blends the muse seam.
  blender -b --python render_character.py -- --blend char.blend --arkit perf.json
Env: NEWS_PANO (HDRI), ANCHOR_OUT (frame dir), NEWS_LENS/DIST/AIM/FSTOP/ENVSTR, NEWS_MCU.
"""
import bpy, sys, os, math, json
from mathutils import Vector, Euler

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
def a(f, d): return argv[argv.index(f) + 1] if f in argv else d
def ef(k, d): return float(os.environ.get(k, d))
BLEND = a("--blend", "/work/character.blend")
ARKIT = a("--arkit", "/work/perf.json")
PANO = os.environ.get("NEWS_PANO", "/work/newsroom_pano.png")
OUT = os.environ.get("ANCHOR_OUT", "/work/output/char_anim/")
if not OUT.endswith("/"): OUT += "/"
os.makedirs(OUT, exist_ok=True)

# ARKit -> Rigify shape-key aliases (this rig has eye_close/brow_* not ARKit names)
ALIAS = {"eyeBlinkLeft": ["eye_close.L"], "eyeBlinkRight": ["eye_close.R"],
         "browInnerUp": ["brow_up.L", "brow_up.R"], "browOuterUpLeft": ["brow_up.L"],
         "browOuterUpRight": ["brow_up.R"], "browDownLeft": ["brow_down.L"], "browDownRight": ["brow_down.R"]}
# Rigify control bones for idle motion (vs viverse Avatar_Head/Neck/Spine)
HEAD_B, NECK_B = "head", "neck"
SPINE_B = ["spine", "spine.001", "spine.002", "spine.003"]

A = json.load(open(ARKIT)); NAMES = A["arkit_names"]; W = A["weights"]; FPS = A.get("fps", 25.0)
NF = min(len(W), int(os.environ.get("MAXF", len(W))))    # MAXF: cap frames for quick tests
idx = {n: i for i, n in enumerate(NAMES)}
BLINK = {"eyeBlinkLeft", "eyeBlinkRight"}
def pctl(c, p): s = sorted(c); return s[max(0, min(len(s) - 1, int(p * len(s))))]
CURVE = {}
for nm in ["eyeBlinkLeft", "eyeBlinkRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight"]:
    if nm not in idx: continue
    col = [float(W[fi][idx[nm]]) for fi in range(NF)]
    base = pctl(col, 0.25 if nm in BLINK else 0.10)
    g = 2.7 if nm in BLINK else 0.5
    CURVE[nm] = [max(0.0, min(1.0, (v - base) * g)) for v in col]

bpy.ops.wm.open_mainfile(filepath=BLEND)
sc = bpy.context.scene
for o in list(bpy.data.objects):           # drop the asset's own cameras/lights; we set our own
    if o.type in ("CAMERA", "LIGHT"): bpy.data.objects.remove(o, do_unlink=True)
arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")

# character extent + head world position (for framing)
mn = Vector((1e9,)*3); mx = Vector((-1e9,)*3)
for o in bpy.data.objects:
    if o.type == "MESH":
        for c in o.bound_box:
            w = o.matrix_world @ Vector(c)
            for i in range(3): mn[i] = min(mn[i], w[i]); mx[i] = max(mx[i], w[i])
cx, cy = (mn.x + mx.x) / 2, (mn.y + mx.y) / 2
hb = arm.pose.bones.get(HEAD_B)
headw = (arm.matrix_world @ hb.head) if hb else Vector((cx, cy, mx.z - 0.15))
topz = mx.z

# ---- world HDRI ----
world = bpy.data.worlds.new("W"); sc.world = world; world.use_nodes = True
nt = world.node_tree; nt.nodes.clear()
tc = nt.nodes.new("ShaderNodeTexCoord"); mp = nt.nodes.new("ShaderNodeMapping")
mp.inputs["Rotation"].default_value[2] = math.radians(ef("NEWS_WALLROT", "180"))
et = nt.nodes.new("ShaderNodeTexEnvironment"); et.image = bpy.data.images.load(PANO)
bg = nt.nodes.new("ShaderNodeBackground"); bg.inputs["Strength"].default_value = ef("NEWS_ENVSTR", "1.1")
op = nt.nodes.new("ShaderNodeOutputWorld")
nt.links.new(tc.outputs["Generated"], mp.inputs["Vector"]); nt.links.new(mp.outputs["Vector"], et.inputs["Vector"])
nt.links.new(et.outputs["Color"], bg.inputs["Color"]); nt.links.new(bg.outputs["Background"], op.inputs["Surface"])

# ---- soft key/fill (env does most of the work) ----
def area(name, loc, energy, size):
    d = bpy.data.lights.new(name, "AREA"); d.energy = energy; d.size = size
    o = bpy.data.objects.new(name, d); o.location = loc; sc.collection.objects.link(o)
    o.rotation_euler = (Vector((cx, cy, headw.z)) - Vector(loc)).normalized().to_track_quat('-Z', 'Y').to_euler()
area("key", (cx - 0.6, cy + 1.6, headw.z + 0.2), 60, 1.4)
area("fill", (cx + 0.7, cy + 1.4, headw.z), 25, 1.8)

# ---- camera: MCU on the head, facing -Y (character faces +Y) ----
cd = bpy.data.cameras.new("cam"); cd.lens = ef("NEWS_LENS", "85")
cam = bpy.data.objects.new("cam", cd); sc.collection.objects.link(cam)
H = topz - mn.z
# character facing: BlenderKit/Rigify faces -Y (camera on -Y looking +Y); viverse faces +Y. NEWS_CAMSIDE=-1|+1
side = ef("NEWS_CAMSIDE", "-1")
camy = (mx.y + H * ef("NEWS_DIST", "0.42")) if side > 0 else (mn.y - H * ef("NEWS_DIST", "0.42"))
cam.location = (cx, camy, headw.z - H * ef("NEWS_AIM", "0.02"))
cam.rotation_euler = (math.radians(90), 0, math.radians(180 if side > 0 else 0)); cd.shift_x = ef("NEWS_SHIFTX", "0.0")
cd.dof.use_dof = True; cd.dof.focus_distance = (Vector(cam.location) - headw).length
cd.dof.aperture_fstop = ef("NEWS_FSTOP", "2.4")
sc.camera = cam

# ---- collect shape-key blocks, drive per frame + idle head motion ----
kb_by = {}
for o in bpy.data.objects:
    if o.type == "MESH" and o.data.shape_keys:
        for kb in o.data.shape_keys.key_blocks: kb_by.setdefault(kb.name, []).append(kb)
def targets(nm):
    return [nm] if nm in kb_by else [t for t in ALIAS.get(nm, []) if t in kb_by]

sc.render.fps = int(round(FPS)); sc.frame_start = 1; sc.frame_end = NF
sc.render.film_transparent = True
sc.render.image_settings.file_format = "PNG"; sc.render.image_settings.color_mode = "RGBA"
sc.render.resolution_x = 1280; sc.render.resolution_y = 720
try: sc.render.engine = "BLENDER_EEVEE_NEXT"
except Exception: sc.render.engine = "BLENDER_EEVEE"
sc.eevee.taa_render_samples = 64

for fi in range(NF):
    f = fi + 1; t = fi / FPS
    for nm, col in CURVE.items():
        for tn in targets(nm):
            for kb in kb_by[tn]: kb.value = col[fi]; kb.keyframe_insert("value", frame=f)
    hd = arm.pose.bones.get(HEAD_B); nk = arm.pose.bones.get(NECK_B)
    breath = math.sin(2 * math.pi * 0.25 * t) * 0.01
    for b in SPINE_B:
        pb = arm.pose.bones.get(b)
        if pb: pb.rotation_mode = "XYZ"; pb.rotation_euler = Euler((breath, 0, math.sin(2*math.pi*0.16*t)*0.005), "XYZ"); pb.keyframe_insert("rotation_euler", frame=f)
    if hd: hd.rotation_mode = "XYZ"; hd.rotation_euler = Euler((math.sin(2*math.pi*0.17*t)*0.02, 0, math.sin(2*math.pi*0.21*t+0.5)*0.02), "XYZ"); hd.keyframe_insert("rotation_euler", frame=f)
    if nk: nk.rotation_mode = "XYZ"; nk.rotation_euler = Euler((math.sin(2*math.pi*0.21*t+0.5)*0.012, 0, 0), "XYZ"); nk.keyframe_insert("rotation_euler", frame=f)
print("KEYED", NF, "frames")

# render env background once (character hidden) -> bg.png, then the character (transparent)
sc.render.film_transparent = False
vis = {o: o.hide_render for o in bpy.data.objects}
for o in bpy.data.objects:
    if o.type == "MESH": o.hide_render = True
sc.render.filepath = OUT + "bg.png"; bpy.ops.render.render(write_still=True)
for o, h in vis.items(): o.hide_render = h
sc.render.film_transparent = True
sc.render.filepath = OUT + "f"
bpy.ops.render.render(animation=True)
print("CHAR_RENDER_DONE ->", OUT)
