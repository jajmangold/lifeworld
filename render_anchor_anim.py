import bpy, sys, math, os, json
from mathutils import Vector, Euler

argv = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
def argf(flag, d): return float(argv[argv.index(flag)+1]) if flag in argv else d
def args(flag, d): return argv[argv.index(flag)+1] if flag in argv else d
WALLROT = argf("--wallrot", 180.0)
ARM     = argf("--arm", 1.22)
SHO     = argf("--sho", 0.18)
LENS    = argf("--lens", 80.0)
ARKIT   = args("--arkit", "/work/anchor.arkit.json")
ENVSTR  = argf("--envstr", 0.75)   # newsroom env strength (was 1.0 = too bright)
KEY     = argf("--key", 130.0)
FILL    = argf("--fill", 45.0)
RIM     = argf("--rim", 0.0)        # back/hair rim light. 0 by default: the premult-over composite
                                    # separates the silhouette cleanly, so any rim>0 reads as a fake
                                    # bright "silver lining" halo around hair/shoulders. Dial up only
                                    # if a specific dark bg needs hair separation.
EXPO    = argf("--expo", -0.9)     # exposure stops (was -0.5)
GLB     = os.environ.get("NEWS_GLB", "/work/avatar.glb")   # swap the character mesh (e.g. a viverse reporter VRM)
# world environment (equirectangular): lights the character AND is rendered as the bg.png behind it, so
# lighting + background come from ONE coherent world. Override with NEWS_PANO for a field/outdoor env
# (e.g. a generated flood pano) instead of the newsroom studio — this is the RIGHT way to place a
# reporter on location (lit by the scene), not compositing over a flat photo.
PANO    = os.environ.get("NEWS_PANO", "/work/newsroom_pano.png")
_FIELD  = bool(os.environ.get("NEWS_PANO"))   # field mode: let the env do the lighting (flat overcast)
if _FIELD:                       # outdoor overcast standup: env light dominates; drop the warm studio
    ENVSTR = float(os.environ.get("NEWS_ENVSTR", "1.25"))   # brighter flat ambient from the sky
    KEY *= 0.35; FILL *= 0.55                               # cut the hot broadcast key so she matches the scene
    WALLROT = float(os.environ.get("NEWS_WALLROT", str(WALLROT)))  # aim the pano's good side at the camera
OUT     = os.environ.get("ANCHOR_OUT", "/work/output/anchor_anim/")  # per-segment frame dir (clip factory)
if not OUT.endswith("/"): OUT += "/"
os.makedirs(OUT, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

# ---- load ARKit driving coeffs (blinks/brows/micro from FLOAT->MediaPipe) ----
A = json.load(open(ARKIT))
NAMES = A["arkit_names"]; W = A["weights"]; FPS = A.get("fps", 25.0)
NF = min(len(W), int(argf("--maxf", 1e9)))
PROC = bool(A.get("proc", False))         # procedural performance -> apply weights directly
HEAD = A.get("head")                       # optional per-frame emphasis head offsets (rad)
HEAD_AMP = float(A.get("head_amp", 1.0))   # idle amplitude multiplier (mood)
# upper-face micro-expressions only (MuseTalk owns the mouth; we lock gaze => no eyeLook/mouth/jaw)
BLINK = {"eyeBlinkLeft","eyeBlinkRight"}                       # absolute (0 open .. 1 closed)
# NOTE: browDownLeft/Right dropped -> they cause the furrowed/stern frown. Keep only gentle lifts.
DELTA = {"browInnerUp","browOuterUpLeft","browOuterUpRight",
         "cheekSquintLeft","cheekSquintRight","eyeSquintLeft","eyeSquintRight",
         "eyeWideLeft","eyeWideRight","noseSneerLeft","noseSneerRight"}  # baseline-subtracted deltas
DRIVE = BLINK | DELTA
idx = {n:i for i,n in enumerate(NAMES)}
GAIN = 0.5        # tame the micro-expression deltas (softer brows)
BLINKGAIN = 2.7   # amplify blinks so partial closures read as full blinks
def pctl(col, p):
    s=sorted(col); return s[max(0,min(len(s)-1,int(p*len(s))))]
# precompute per-channel processed curves
CURVE = {}
DRIVE_NAMES = NAMES if PROC else DRIVE     # proc json only contains channels we want
for nm in DRIVE_NAMES:
    if nm not in idx: continue
    col=[float(W[fi][idx[nm]]) for fi in range(NF)]
    if PROC:
        pass                               # already final values, apply directly
    elif nm in BLINK:
        base=pctl(col,0.25)  # resting (eyes-open) level -> 0
        col=[max(0.0,min(1.0,(v-base)*BLINKGAIN)) for v in col]
    elif nm in DELTA:
        base=pctl(col,0.10)
        col=[max(0.0,min(1.0,(v-base)*GAIN)) for v in col]
    else:
        continue
    CURVE[nm]=col
# report blink peaks after boost
_b=CURVE.get("eyeBlinkLeft",[])
print("BLINK peaks(after boost):", sorted([round(max(_b[max(0,i-1):i+2]),2) for i in range(len(_b)) if _b[i]>0.5 and _b[i]==max(_b[max(0,i-2):i+3])])[-8:] if _b else [])
print("ARKIT", NF, "frames @", FPS, "fps | driving", len(CURVE), "shapes (blink+micro)")

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=GLB)
arm = [o for o in bpy.data.objects if o.type=="ARMATURE"][0]

# Optional: swap the head's base-color texture for a pre-baked face (anchorM baked once onto the UV);
# this makes the render produce the target identity directly, so the per-frame face-swap can be skipped.
_baked = os.environ.get("BAKED_HEAD_TEX") or args("--bakedtex", "")
if _baked:
    _hm = bpy.data.materials.get("head_Opaque_Material_Meshes_Material")
    if _hm and _hm.use_nodes:
        _b = next((n for n in _hm.node_tree.nodes if n.type=="BSDF_PRINCIPLED"), None)
        if _b and _b.inputs["Base Color"].is_linked:
            _tn = _b.inputs["Base Color"].links[0].from_node
            if _tn.type == "TEX_IMAGE":
                _img = bpy.data.images.load(_baked); _img.colorspace_settings.name = "sRGB"
                _tn.image = _img
                print("BAKED head texture applied:", _baked)
# hide the head_Transparent "fuzz" shell (thin transparent hairline layer) that reads as a dark rim
if os.environ.get("HIDE_HEAD_FUZZ"):
    _ht = bpy.data.objects.get("head_Transparent_Material_Meshes_Mesh")
    if _ht:
        _ht.hide_render = True
        print("hid head_Transparent fuzz shell")

# ---- arms down ----
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="POSE")
def baserot(name, axis, ang):
    pb = arm.pose.bones.get(name)
    if not pb: return
    pb.rotation_mode="XYZ"; e=list(pb.rotation_euler); e["XYZ".index(axis)]+=ang
    pb.rotation_euler=Euler(e,"XYZ")
baserot("Avatar_LeftArm","Z",ARM);  baserot("Avatar_RightArm","Z",-ARM)
baserot("Avatar_LeftShoulder","Z",SHO); baserot("Avatar_RightShoulder","Z",-SHO)
bpy.ops.object.mode_set(mode="OBJECT")
bpy.context.view_layer.update()

# ---- collect shape-key blocks per ARKit name across all meshes; lock gaze (eyeLook=0) ----
kb_by_name = {}
for o in bpy.data.objects:
    if o.type!="MESH" or not o.data.shape_keys: continue
    for kb in o.data.shape_keys.key_blocks:
        kb_by_name.setdefault(kb.name, []).append((o, kb))
# zero all eyeLook (gaze locked forward) once
for nm in [k for k in kb_by_name if k.startswith("eyeLook")]:
    for o,kb in kb_by_name[nm]: kb.value=0.0

sc = bpy.context.scene
sc.render.fps = int(round(FPS))
sc.frame_start = 1; sc.frame_end = NF

# ARKit -> native viverse shape-key aliases. The anchor glb was augmented with a full ARKit-52 set, but a
# RAW viverse avatar (e.g. a reporter VRM) only has native names. Alias the few micro-expressions we drive
# so blinks/brows fire on raw avatars too. Used ONLY when the ARKit name is absent (anchor unaffected).
ALIAS = {"eyeBlinkLeft":["Eye_Blink_L"], "eyeBlinkRight":["Eye_Blink_R"],
         "browInnerUp":["EyeB_Up_L","EyeB_Up_R"], "browOuterUpLeft":["EyeB_Up_L"],
         "browOuterUpRight":["EyeB_Up_R"], "eyeSquintLeft":["Eye_Squint_L"], "eyeSquintRight":["Eye_Squint_R"]}
def _targets(nm):
    if nm in kb_by_name: return [nm]
    return [a for a in ALIAS.get(nm, []) if a in kb_by_name]

# ---- keyframe driven blendshapes per frame + subtle body idle + micro head motion ----
spine = ["Avatar_Spine","Avatar_Spine1","Avatar_Spine2"]
def key_pose(name, f):
    pb = arm.pose.bones.get(name)
    if pb: pb.keyframe_insert("rotation_euler", frame=f)
for fi in range(NF):
    f = fi+1
    t = fi/FPS
    # facial: drive processed micro-expression curves, keyframe
    for nm, col in CURVE.items():
        v = col[fi]
        for tnm in _targets(nm):
            for o,kb in kb_by_name[tnm]:
                kb.value = v
                kb.keyframe_insert("value", frame=f)
    # body idle: breathing on spine, gentle head sway/nod (* mood head_amp) + emphasis nod (HEAD)
    bpy.ops.object.mode_set(mode="POSE")
    HA = HEAD_AMP
    ex, ey, ez = (HEAD[fi] if HEAD and fi < len(HEAD) else (0.0, 0.0, 0.0))
    breath = math.sin(2*math.pi*0.25*t)*0.012
    for b in spine:
        pb=arm.pose.bones.get(b)
        if pb: pb.rotation_mode="XYZ"; pb.rotation_euler=Euler((breath,0,math.sin(2*math.pi*0.16*t)*0.006*HA),"XYZ"); key_pose(b,f)
    hd=arm.pose.bones.get("Avatar_Head"); nk=arm.pose.bones.get("Avatar_Neck")
    if hd:
        hd.rotation_mode="XYZ"
        hd.rotation_euler=Euler((math.sin(2*math.pi*0.21*t+0.5)*0.02*HA + ex,
                                 math.sin(2*math.pi*0.13*t)*0.03*HA + ey,
                                 math.sin(2*math.pi*0.17*t)*0.015*HA + ez),"XYZ"); key_pose("Avatar_Head",f)
    if nk:
        nk.rotation_mode="XYZ"
        nk.rotation_euler=Euler((math.sin(2*math.pi*0.21*t+0.5)*0.012*HA + ex*0.4,0,0),"XYZ"); key_pose("Avatar_Neck",f)
    bpy.ops.object.mode_set(mode="OBJECT")
print("KEYED", NF, "frames")

# ---- bounds (frame 1) ----
sc.frame_set(1)
deps=bpy.context.evaluated_depsgraph_get()
mn=Vector((1e9,1e9,1e9));mx=Vector((-1e9,-1e9,-1e9))
for o in bpy.data.objects:
    if o.type!="MESH":continue
    ev=o.evaluated_get(deps)
    for c in ev.bound_box:
        w=ev.matrix_world@Vector(c)
        for i in range(3): mn[i]=min(mn[i],w[i]);mx[i]=max(mx[i],w[i])
cx=(mn.x+mx.x)/2;cy=(mn.y+mx.y)/2;H=mx.z-mn.z;topz=mx.z
aimz=topz-H*0.11

# ---- newsroom world ----
world=bpy.data.worlds.new("W");sc.world=world;world.use_nodes=True
nt=world.node_tree;nt.nodes.clear()
tc=nt.nodes.new("ShaderNodeTexCoord");mp=nt.nodes.new("ShaderNodeMapping")
mp.inputs["Rotation"].default_value[2]=math.radians(WALLROT)
et=nt.nodes.new("ShaderNodeTexEnvironment");et.image=bpy.data.images.load(PANO)
bg=nt.nodes.new("ShaderNodeBackground");bg.inputs["Strength"].default_value=ENVSTR
op=nt.nodes.new("ShaderNodeOutputWorld")
nt.links.new(tc.outputs["Generated"],mp.inputs["Vector"]);nt.links.new(mp.outputs["Vector"],et.inputs["Vector"])
nt.links.new(et.outputs["Color"],bg.inputs["Color"]);nt.links.new(bg.outputs["Background"],op.inputs["Surface"])

# ---- lights ----
def area(name,loc,energy,size):
    d=bpy.data.lights.new(name,"AREA");d.energy=energy;d.size=size
    o=bpy.data.objects.new(name,d);o.location=loc;bpy.context.collection.objects.link(o)
    o.rotation_euler=(Vector((cx,cy,topz-H*0.10))-Vector(loc)).normalized().to_track_quat('-Z','Y').to_euler()
area("key",(cx-0.6,cy+1.6,topz+0.2),KEY,1.4)
area("fill",(cx+0.7,cy+1.4,topz-0.1),FILL,1.8)
# rim/hair light BEHIND + above the head -> lifts the silhouette so the hair edge separates from the
# dark background (kills the dark contour rim) instead of falling off to a dark outline.
area("rim",(cx-0.3,cy-1.5,topz+0.7),RIM,1.0)

# ---- camera ----
cd=bpy.data.cameras.new("cam");cd.lens=LENS
cam=bpy.data.objects.new("cam",cd);bpy.context.collection.objects.link(cam)
cam.location=(cx,mx.y+H*0.62,aimz);cam.rotation_euler=(math.radians(90),0,math.radians(180))
sc.camera=cam

# ---- BROADCAST MODE: MCU reframe (any anchor format) + optional 3D video wall (anchor_wall only) ----
# NEWS_MCU=1 (or NEWS_SCREEN set) -> reframe to medium-close-up. NEWS_SCREEN=<image> -> add the 3D wall
# and default the anchor to the left third; without a screen (fullscreen_anchor/ots) default centered.
import os as _os
_SCREEN_IMG=_os.environ.get("NEWS_SCREEN")
_MCU=bool(_SCREEN_IMG) or _os.environ.get("NEWS_MCU")
_char_meshes=[o for o in bpy.data.objects if o.type=="MESH"]   # avatar meshes (before the screen is added)
_screen=None
def _ef(k,d): return float(_os.environ.get(k,d))
if _MCU:
    # reframe: medium close-up + horizontal lens shift (default: pushed left when a wall is present, else centered)
    cd.lens=_ef("NEWS_LENS","92")
    cd.shift_x=_ef("NEWS_SHIFTX", "0.33" if _SCREEN_IMG else "0.0")
    cam.location=(cx, mx.y+H*_ef("NEWS_DIST","0.42"), topz-H*_ef("NEWS_AIM","0.09"))
    if _FIELD:
        # long-lens standup: focus on her face, throw the environment out of focus (real DoF, so the
        # world bg gets a natural bokeh fall-off instead of the flat all-sharp env render).
        cd.dof.use_dof=True
        cd.dof.focus_distance=(Vector(cam.location)-Vector((cx,cy,topz-H*0.09))).length
        cd.dof.aperture_fstop=_ef("NEWS_FSTOP","2.2")
if _SCREEN_IMG:
    # 3D video wall: emissive plane standing in the set, angled toward camera, right of + behind the anchor.
    # NDC mapping (this rig): screen NDC x ~= 0.17 - 1.30*SX, NDC y centers ~0.5 at SZ~0.14. SX<0 -> frame right.
    bpy.ops.mesh.primitive_plane_add(size=1.0)
    _screen=bpy.context.active_object; _screen.name="news_screen"
    _screen.rotation_euler=(math.radians(90),0,math.radians(_ef("NEWS_YAW","14")))
    _screen.scale=(_ef("NEWS_W","0.50"), _ef("NEWS_H","0.28"), 1.0)
    _screen.location=(cx+_ef("NEWS_SX","-0.45"), cy+_ef("NEWS_SY","0.05"), topz-H*_ef("NEWS_SZ","0.08"))
    _sm=bpy.data.materials.new("screenmat");_sm.use_nodes=True;_smt=_sm.node_tree;_smt.nodes.clear()
    _so=_smt.nodes.new("ShaderNodeOutputMaterial");_se=_smt.nodes.new("ShaderNodeEmission")
    _st=_smt.nodes.new("ShaderNodeTexImage");_st.image=bpy.data.images.load(_SCREEN_IMG)
    _se.inputs["Strength"].default_value=_ef("NEWS_SCREEN_STR","2.4")
    _smt.links.new(_st.outputs["Color"],_se.inputs["Color"]);_smt.links.new(_se.outputs[0],_so.inputs["Surface"])
    _screen.data.materials.append(_sm)
    from bpy_extras.object_utils import world_to_camera_view as _w2c
    bpy.context.view_layer.update()
    _c=_w2c(sc,cam,_screen.matrix_world.translation)
    print("BROADCAST screen at",tuple(round(v,2) for v in _screen.location),"shift_x",cd.shift_x,
          "| NDC x=%.3f y=%.3f depth=%.2f"%(_c.x,_c.y,_c.z),flush=True)

# ---- render settings ----
sc.render.engine="BLENDER_EEVEE_NEXT"
# broadcast MCU magnifies the HASHED-material TAA dither (checker on the white shirt) -> more samples.
sc.eevee.taa_render_samples=int(os.environ.get("NEWS_SAMPLES", "64" if _MCU else "32"))
# opaque face/body should cast OPAQUE (not alpha-tested/dithered) shadows — the HASHED materials
# otherwise throw a stochastic checker into the neck/collar shadow on the white shirt. (Camera-side
# alpha cutout is untouched, so no black shoulder patches.)
for _o in bpy.data.objects:
    if _o.type=="MESH" and ("head_Opaque" in _o.name or "body_Opaque" in _o.name):
        for _sl in _o.material_slots:
            if _sl.material:
                try: _sl.material.use_transparent_shadow=False
                except Exception: pass
sc.render.resolution_x=1280;sc.render.resolution_y=720
sc.render.image_settings.file_format="PNG"
sc.view_settings.view_transform="AgX";sc.view_settings.exposure=EXPO

# Render the env BACKGROUND once (character hidden) -> bg.png. The character is then rendered on
# transparent and composited over bg in post (clean straight-alpha edges) instead of rendered directly
# over the env, which left a dark AA fringe at the head silhouette. The world still lights/reflects on
# the character either way.
sc.render.image_settings.color_mode="RGB"
sc.render.film_transparent=False
# In broadcast mode hide ONLY the avatar (the news_screen stays so it's baked into the static bg plate
# AND keeps lighting the anchor in the transparent pass). Otherwise hide all meshes (original behaviour).
_meshes=_char_meshes if _SCREEN_IMG else [o for o in bpy.data.objects if o.type=="MESH"]
_vis={o:o.hide_render for o in _meshes}
for o in _meshes: o.hide_render=True
sc.frame_set(1); sc.render.filepath=OUT+"bg"; bpy.ops.render.render(write_still=True)
for o in _meshes: o.hide_render=_vis[o]
print("rendered env background -> bg.png")

sc.render.film_transparent=True
sc.render.image_settings.color_mode="RGBA"
sc.render.filepath=OUT+"f"
bpy.ops.render.render(animation=True)
print("WROTE frames to",OUT)
