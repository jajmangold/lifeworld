import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath="/work/avatar.glb")

def feeding_image(sock):
    if not sock.is_linked: return None
    n = sock.links[0].from_node
    # walk through Normal Map / other passthrough nodes
    if n.type == "TEX_IMAGE": return n.image
    for inp in n.inputs:
        if inp.is_linked:
            src = inp.links[0].from_node
            if src.type == "TEX_IMAGE": return src.image
    return None

print("=== MATERIAL TEXTURE SLOTS ===")
for o in bpy.data.objects:
    if o.type != "MESH": continue
    for s in o.material_slots:
        m = s.material
        if not m or not m.use_nodes: continue
        bsdf = next((n for n in m.node_tree.nodes if n.type=="BSDF_PRINCIPLED"), None)
        if not bsdf: continue
        print(f"\nMESH {o.name}  MAT {m.name}  blend={m.blend_method}")
        for slot in ("Base Color","Normal","Metallic","Roughness","Emission","Alpha","Specular IOR Level"):
            inp = bsdf.inputs.get(slot)
            if inp is None: continue
            img = feeding_image(inp)
            if img:
                print(f"   {slot:20s} -> {img.name:10s} {tuple(img.size)}")
print("DONE")
