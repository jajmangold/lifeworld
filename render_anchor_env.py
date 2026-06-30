import bpy, sys, math, os
from mathutils import Vector, Euler, Quaternion

argv = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else []
def argf(flag, d):
    return float(argv[argv.index(flag)+1]) if flag in argv else d
TEST    = "--test" in argv
WALLROT = argf("--wallrot", 180.0)
ARM     = argf("--arm", 1.22)     # arm-down swing (rad) ~70deg
SHO     = argf("--sho", 0.18)     # shoulder drop (rad)
LENS    = argf("--lens", 80.0)
GLB     = "/work/avatar.glb"
PANO    = "/work/newsroom_pano.png"
OUT     = "/work/output/anchor_env/"
os.makedirs(OUT, exist_ok=True)

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=GLB)
arm = [o for o in bpy.data.objects if o.type=="ARMATURE"][0]

# ---- pose arms down (clean Avatar_* humanoid, T-pose) ----
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode="POSE")
def setrot(name, axis, ang):
    pb = arm.pose.bones.get(name)
    if not pb: return
    pb.rotation_mode = "XYZ"
    e = list(pb.rotation_euler)
    e["XYZ".index(axis)] += ang
    pb.rotation_euler = Euler(e, "XYZ")
# Left arm swings down via -Z, right via +Z (mirror). Tweak via --arm.
setrot("Avatar_LeftArm",  "Z",  ARM)
setrot("Avatar_RightArm", "Z", -ARM)
setrot("Avatar_LeftShoulder",  "Z",  SHO)
setrot("Avatar_RightShoulder", "Z", -SHO)
# gaze straight ahead (locked) — eyes neutral for now
bpy.ops.object.mode_set(mode="OBJECT")
bpy.context.view_layer.update()

# ---- bounds (from all meshes, after pose) ----
deps = bpy.context.evaluated_depsgraph_get()
mn = Vector(( 1e9, 1e9, 1e9)); mx = Vector((-1e9,-1e9,-1e9))
for o in bpy.data.objects:
    if o.type!="MESH": continue
    ev = o.evaluated_get(deps)
    for c in ev.bound_box:
        w = ev.matrix_world @ Vector(c)
        for i in range(3):
            mn[i]=min(mn[i],w[i]); mx[i]=max(mx[i],w[i])
cx=(mn.x+mx.x)/2; cy=(mn.y+mx.y)/2; H=mx.z-mn.z; topz=mx.z
print("BOUNDS x",round(mn.x,3),round(mx.x,3),"y",round(mn.y,3),round(mx.y,3),"z",round(mn.z,3),round(mx.z,3))

# ---- world: newsroom equirect env ----
world = bpy.data.worlds.new("W"); bpy.context.scene.world = world
world.use_nodes = True
nt = world.node_tree; nt.nodes.clear()
tc = nt.nodes.new("ShaderNodeTexCoord")
mp = nt.nodes.new("ShaderNodeMapping"); mp.inputs["Rotation"].default_value[2]=math.radians(WALLROT)
et = nt.nodes.new("ShaderNodeTexEnvironment"); et.image = bpy.data.images.load(PANO)
bg = nt.nodes.new("ShaderNodeBackground"); bg.inputs["Strength"].default_value=1.0
op = nt.nodes.new("ShaderNodeOutputWorld")
nt.links.new(tc.outputs["Generated"], mp.inputs["Vector"])
nt.links.new(mp.outputs["Vector"], et.inputs["Vector"])
nt.links.new(et.outputs["Color"], bg.inputs["Color"])
nt.links.new(bg.outputs["Background"], op.inputs["Surface"])

# ---- soft key + fill for face clarity ----
def area(name, loc, energy, size):
    d = bpy.data.lights.new(name, "AREA"); d.energy=energy; d.size=size
    o = bpy.data.objects.new(name, d); o.location=loc
    bpy.context.collection.objects.link(o)
    dirv = (Vector((cx,cy,topz-H*0.10))-Vector(loc)).normalized()
    o.rotation_euler = dirv.to_track_quat('-Z','Y').to_euler()
    return o
aimz = topz - H*0.11
area("key",  (cx-0.6, cy+1.6, topz+0.2), 200, 1.4)
area("fill", (cx+0.7, cy+1.4, topz-0.1), 70, 1.8)

# ---- camera: head + shoulders, in front (+Y) ----
cam_d = bpy.data.cameras.new("cam"); cam_d.lens=LENS
cam = bpy.data.objects.new("cam", cam_d); bpy.context.collection.objects.link(cam)
cam.location = (cx, mx.y + H*0.62, aimz)
cam.rotation_euler = (math.radians(90), 0, math.radians(180))  # look toward -Y (avatar front faces +Y)
bpy.context.scene.camera = cam

# ---- render ----
sc = bpy.context.scene
sc.render.engine = "BLENDER_EEVEE_NEXT"
sc.eevee.taa_render_samples = 32
sc.render.film_transparent = False
sc.render.resolution_x = 1280; sc.render.resolution_y = 720
sc.render.image_settings.file_format = "PNG"
sc.view_settings.view_transform = "Filmic"
sc.view_settings.exposure = -0.5

if TEST:
    sc.render.filepath = OUT + "test.png"
    bpy.ops.render.render(write_still=True)
    print("WROTE", sc.render.filepath)
else:
    sc.render.filepath = OUT + "f"
    sc.frame_start=1; sc.frame_end=int(argf("--frames",1))
    bpy.ops.render.render(write_still=True)
    print("WROTE frames to", OUT)
