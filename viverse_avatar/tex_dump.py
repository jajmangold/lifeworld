import bpy, os
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath="/work/avatar.glb")
os.makedirs("/work/tex_dump", exist_ok=True)

print("=== MESH -> MATERIALS ===")
for o in bpy.data.objects:
    if o.type == "MESH":
        mats = [s.material.name if s.material else None for s in o.material_slots]
        print(f"MESH {o.name}: {mats}")

print("=== MATERIAL -> BASE COLOR IMAGE ===")
for m in bpy.data.materials:
    if not m.use_nodes: continue
    bsdf = next((n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    img = None
    if bsdf:
        inp = bsdf.inputs.get("Base Color")
        if inp and inp.is_linked:
            src = inp.links[0].from_node
            if src.type == "TEX_IMAGE" and src.image:
                img = src.image
    print(f"MAT {m.name}: base_color_img={img.name if img else None} size={tuple(img.size) if img else None}")

print("=== SAVE ALL IMAGES ===")
seen = set()
for img in bpy.data.images:
    if img.name in seen or img.size[0] == 0: continue
    seen.add(img.name)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in img.name)
    p = f"/work/tex_dump/{safe}.png"
    try:
        img.filepath_raw = p; img.file_format = "PNG"; img.save()
        print(f"SAVED {p} {tuple(img.size)}")
    except Exception as e:
        print(f"FAIL {img.name}: {e}")
print("DONE")
