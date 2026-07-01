import bpy, sys
g=sys.argv[sys.argv.index("--")+1]; out=sys.argv[sys.argv.index("--")+2]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=g)
arm=[o for o in bpy.data.objects if o.type=="ARMATURE"][0]
hb=arm.pose.bones.get("Avatar_Head"); hw=arm.matrix_world @ hb.head
sc=bpy.context.scene
sc.world=bpy.data.worlds.new("W"); sc.world.use_nodes=True
sc.world.node_tree.nodes["Background"].inputs[1].default_value=3.5
cd=bpy.data.cameras.new("c"); cam=bpy.data.objects.new("c",cd); sc.collection.objects.link(cam)
cam.location=(hw.x, hw.y+0.6, hw.z+0.03); cam.rotation_euler=(1.5708,0,3.14159); cd.lens=68
sc.camera=cam
try: sc.render.engine="BLENDER_EEVEE_NEXT"
except Exception: sc.render.engine="BLENDER_EEVEE"
sc.render.resolution_x=512; sc.render.resolution_y=512; sc.render.filepath=out
bpy.ops.render.render(write_still=True); print("QUICK_OK")
