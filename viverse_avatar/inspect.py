import bpy, glob, math, mathutils, os, json
bpy.ops.wm.read_factory_settings(use_empty=True)
glbs=sorted(glob.glob("/work/*.glb"))
allshapes=set(); arms=[]; meshcount=0
for g in glbs:
    before=set(bpy.data.objects)
    try: bpy.ops.import_scene.gltf(filepath=g)
    except Exception as e: print("IMPORT_FAIL",os.path.basename(g),e); continue
    new=set(bpy.data.objects)-before
    sks=set()
    for o in new:
        if o.type=="MESH":
            meshcount+=1
            if o.data.shape_keys:
                for k in o.data.shape_keys.key_blocks: sks.add(k.name)
        if o.type=="ARMATURE": arms.append((os.path.basename(g),len(o.pose.bones)))
    allshapes|=sks
    print("GLB",os.path.basename(g),"meshes",[o.name for o in new if o.type=="MESH"],"shapes",len(sks))
print("TOTAL_MESHES",meshcount,"ARMATURES",arms)
print("ALLSHAPES",json.dumps(sorted(allshapes)))
