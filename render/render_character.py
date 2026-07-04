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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # so sibling modules (graft_hair, fix_hair) import
sys.path.insert(0, os.getcwd())

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
def a(f, d): return argv[argv.index(f) + 1] if f in argv else d
def ef(k, d): return float(os.environ.get(k, d))
BLEND = a("--blend", "/work/character.blend")
ARKIT = a("--arkit", "/work/perf.json")
PANO = os.environ.get("NEWS_PANO", "/work/newsroom_pano.png")
OUT = os.environ.get("ANCHOR_OUT", "/work/output/char_anim/")
if not OUT.endswith("/"): OUT += "/"
os.makedirs(OUT, exist_ok=True)

# ARKit -> shape-key aliases. Covers Rigify/MakeHuman (eye_close/brow_*) AND HumGen FACS (eyeBlink_L,
# browOuterUp_L, ...). targets() keeps only the names actually present, so one map serves every rig.
ALIAS = {"eyeBlinkLeft": ["eye_close.L", "eyeBlink_L"], "eyeBlinkRight": ["eye_close.R", "eyeBlink_R"],
         "browInnerUp": ["brow_up.L", "brow_up.R", "browInnerUp"],
         "browOuterUpLeft": ["brow_up.L", "browOuterUp_L"], "browOuterUpRight": ["brow_up.R", "browOuterUp_R"],
         "browDownLeft": ["brow_down.L", "browDown_L"], "browDownRight": ["brow_down.R", "browDown_R"],
         "jawOpen": ["jawOpen", "jaw_open", "mouthOpen"]}
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
# gentle jaw under MuseTalk: proc_perf emits a low jawOpen envelope; drive it lightly so the chin/jaw moves
# with speech (the mouth+jaw read as one) without fighting the 2D muse mouth. Absolute value, extra-tamed.
if "jawOpen" in idx:
    CURVE["jawOpen"] = [max(0.0, min(0.5, float(W[fi][idx["jawOpen"]]) * float(os.environ.get("NEWS_JAW", "0.7")))) for fi in range(NF)]

# per-frame HEAD motion from the perf: proc_perf authors a natural, speech-aware, safety-clipped idle
# (coherent drift + breathing + micro-saccades + emphasis nods on stressed peaks). Use it directly instead of
# a hand-rolled sin wobble (which read as uncanny). cols = [pitch, yaw, roll] radians.
HEAD = A.get("head"); HEAD_AMP = float(A.get("head_amp", 1.0)) * float(os.environ.get("NEWS_HEADAMP", "1.0"))

bpy.ops.wm.open_mainfile(filepath=BLEND)
if os.environ.get("NEWS_HAIRFIX"):               # de-plasticise hair/cloth materials (see fix_hair.py)
    import fix_hair
    _m = os.environ.get("NEWS_HAIRFIX_MATCH", "")
    fix_hair.tune_hair_cloth(_m.split(",") if _m and _m != "1" else None,
                             rough=ef("NEWS_HAIR_ROUGH", 0.7))
sc = bpy.context.scene
for o in list(bpy.data.objects):           # drop the asset's own cameras/lights; we set our own
    if o.type in ("CAMERA", "LIGHT"): bpy.data.objects.remove(o, do_unlink=True)
arm = next(o for o in bpy.data.objects if o.type == "ARMATURE")

# --- robust bone resolution: rigs name head/neck/spine differently (head vs Head vs Avatar_Head);
#     hardcoded lowercase names silently no-op'd idle motion on other rigs. Match flexibly. ---
_bn = {b.name.lower(): b.name for b in arm.pose.bones}
def _find_bone(cands):
    for c in cands:
        if c.lower() in _bn: return _bn[c.lower()]
    for c in cands:
        for ln, real in _bn.items():
            if c.lower() in ln: return real
    return None
HEAD_B = _find_bone(["head", "avatar_head", "def-head", "spine.006"]) or HEAD_B
NECK_B = _find_bone(["neck", "avatar_neck", "spine.004"]) or NECK_B
_sp = [_find_bone([s]) for s in ["spine", "spine.001", "spine.002", "spine.003", "chest", "avatar_spine"]]
SPINE_B = [b for b in dict.fromkeys(_sp) if b] or SPINE_B
print("BONES head=%s neck=%s spine=%s" % (HEAD_B, NECK_B, SPINE_B))

# --- optional HAAR hair graft (the flexible female-hair path); follows the head bone ---
_groom = os.environ.get("NEWS_HAIR_GROOM", "")
if _groom:
    import graft_hair
    graft_hair.graft(bpy, arm, _groom, HEAD_B,
                     melanin=ef("NEWS_HAIR_MEL", "0.5"), yaw=ef("NEWS_HAIR_YAW", "0"))

# optional lower-face slim (MakeHuman base meshes render 'chipmunk cheeks'; no cheek shape key exists)
if ef("NEWS_FACE_SLIM", "0") > 0:
    import slim_face
    slim_face.slim(bpy, arm, amount=ef("NEWS_FACE_SLIM", "0"))

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

# ---- broadcast 3-point lighting on the FRONT (camera) side ----
# BUG FIX: the old key/fill were placed at +Y (behind a character that faces -Y) so the face got no key light,
# only ambient env -> flat. Place a proper warm key + cool fill on the camera side (front = side*Y) + a back
# rim for separation. `side` (which way the character faces) must be known here, so compute it first.
side = ef("NEWS_CAMSIDE", "-1")
fy = 1.0 if side > 0 else -1.0                     # +Y is "front" for side>0, -Y for side<0
def area(name, loc, energy, size, color=None):
    d = bpy.data.lights.new(name, "AREA"); d.energy = energy; d.size = size
    if color: d.color = color
    o = bpy.data.objects.new(name, d); o.location = loc; sc.collection.objects.link(o)
    o.rotation_euler = (Vector((cx, cy, headw.z)) - Vector(loc)).normalized().to_track_quat('-Z', 'Y').to_euler()
area("key",  (cx - 0.75, cy + fy * 1.5, headw.z + 0.35), ef("NEWS_KEY", "150"), 1.2, (1.0, 0.95, 0.88))  # warm key, front-high-left
area("fill", (cx + 0.85, cy + fy * 1.4, headw.z + 0.0), ef("NEWS_FILL", "55"), 2.0, (0.90, 0.94, 1.0))   # cool soft fill, front-right
area("rim",  (cx + 0.5,  cy - fy * 1.3, headw.z + 0.55), ef("NEWS_RIM", "90"), 0.7)                       # back rim for separation

# ---- camera: MCU on the head, facing -Y (character faces +Y) ----
cd = bpy.data.cameras.new("cam"); cd.lens = ef("NEWS_LENS", "85")
cam = bpy.data.objects.new("cam", cd); sc.collection.objects.link(cam)
H = topz - mn.z
# character facing: BlenderKit/Rigify faces -Y (camera on -Y looking +Y); viverse faces +Y. NEWS_CAMSIDE=-1|+1
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

# CRITICAL: MakeHuman/Rigify rigs DRIVE these shape keys from eyelid/brow bones (SK-eyelid, etc.). A driver
# overrides our keyframed kb.value EVERY frame, so blinks AND brows silently do nothing (the shape has real
# deformation — verified — but the driver pins it to rest). Remove drivers on the keys we drive so keyframes apply.
_drive = set(t for nm in CURVE for t in targets(nm))
for o in bpy.data.objects:
    if o.type == "MESH" and o.data.shape_keys:
        for tn in _drive:
            try: o.data.shape_keys.driver_remove('key_blocks["%s"].value' % tn)
            except Exception: pass
print("BLINK/BROW drivers removed for:", sorted(_drive))

sc.render.fps = int(round(FPS)); sc.frame_start = 1; sc.frame_end = NF
sc.render.film_transparent = True
sc.render.image_settings.file_format = "PNG"; sc.render.image_settings.color_mode = "RGBA"
sc.render.resolution_x = 1280; sc.render.resolution_y = 720
try: sc.render.engine = "BLENDER_EEVEE_NEXT"
except Exception: sc.render.engine = "BLENDER_EEVEE"
sc.eevee.taa_render_samples = 64

# ---- SELF-FACE swap source: a frontal head close-up rendered with THIS shot's EXACT world + lights (same
#      renderer, same HDRI/tone), so when make_anchor face-swaps the character with its OWN face the source
#      tone matches the scene -> no identity mismatch, no swap seam. Rendered here at REST (before the
#      expression keyframing below) so it's neutral + eyes-open. Toggle with NEWS_SELF_FACE=0. ----
if os.environ.get("NEWS_SELF_FACE", "1") == "1":
    sfd = bpy.data.cameras.new("sfcam"); sfd.lens = 95
    sfc = bpy.data.objects.new("sfcam", sfd); sc.collection.objects.link(sfc)
    sft = headw + Vector((0, 0, 0.11))
    sfc.location = sft + (Vector((0, 0.85, 0)) if side > 0 else Vector((0, -0.85, 0)))
    sfc.rotation_euler = (sft - sfc.location).to_track_quat('-Z', 'Y').to_euler()
    _fp, _rx, _ry, _ft, _c0 = sc.render.filepath, sc.render.resolution_x, sc.render.resolution_y, sc.render.film_transparent, sc.camera
    sc.camera = sfc; sc.render.resolution_x = sc.render.resolution_y = 1024; sc.render.film_transparent = False
    sc.render.filepath = OUT + "self_face.png"; bpy.ops.render.render(write_still=True)
    sc.camera, sc.render.resolution_x, sc.render.resolution_y, sc.render.film_transparent, sc.render.filepath = _c0, _rx, _ry, _ft, _fp
    bpy.data.objects.remove(sfc, do_unlink=True)
    print("SELF_FACE_RENDERED ->", OUT + "self_face.png")

for fi in range(NF):
    f = fi + 1; t = fi / FPS
    for nm, col in CURVE.items():
        for tn in targets(nm):
            for kb in kb_by[tn]: kb.value = col[fi]; kb.keyframe_insert("value", frame=f)
    hd = arm.pose.bones.get(HEAD_B); nk = arm.pose.bones.get(NECK_B)
    # HEAD motion from the perf's authored idle (natural, speech-aware, clipped) — NOT a raw sin wobble.
    ph, yw, rl = (HEAD[fi] if (HEAD and fi < len(HEAD)) else (0.0, 0.0, 0.0))
    ph *= HEAD_AMP; yw *= HEAD_AMP; rl *= HEAD_AMP
    breath = math.sin(2 * math.pi * 0.25 * t) * 0.008        # just breathing on the spine (subtle)
    for b in SPINE_B:
        pb = arm.pose.bones.get(b)
        if pb: pb.rotation_mode = "XYZ"; pb.rotation_euler = Euler((breath, 0, 0), "XYZ"); pb.keyframe_insert("rotation_euler", frame=f)
    if hd: hd.rotation_mode = "XYZ"; hd.rotation_euler = Euler((ph, yw, rl), "XYZ"); hd.keyframe_insert("rotation_euler", frame=f)
    if nk: nk.rotation_mode = "XYZ"; nk.rotation_euler = Euler((ph * 0.4, yw * 0.4, 0), "XYZ"); nk.keyframe_insert("rotation_euler", frame=f)
print("KEYED", NF, "frames")

# render env background once (character hidden) -> bg.png, then the character (transparent)
sc.render.film_transparent = False
vis = {o: o.hide_render for o in bpy.data.objects}
for o in bpy.data.objects:
    if o.type in ("MESH", "CURVE", "CURVES"): o.hide_render = True   # incl. grafted hair curve
sc.render.filepath = OUT + "bg.png"; bpy.ops.render.render(write_still=True)
for o, h in vis.items(): o.hide_render = h
sc.render.film_transparent = True
sc.render.filepath = OUT + "f"
bpy.ops.render.render(animation=True)
print("CHAR_RENDER_DONE ->", OUT)
