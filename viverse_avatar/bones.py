import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath="/work/avatar.glb")
arm=[o for o in bpy.data.objects if o.type=="ARMATURE"][0]
print("ARMATURE:", arm.name)
for b in arm.pose.bones:
    print("BONE:", b.name)
