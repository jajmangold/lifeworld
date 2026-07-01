"""List a character .blend's images + which material uses each as Base Color (albedo) — to decide what
to upscale. Only ALBEDO/base-color maps should be upscaled (never normal/roughness/metalness).
  blender -b --python dump_textures.py -- <asset.blend>
"""
import bpy, sys
bpy.ops.wm.open_mainfile(filepath=sys.argv[sys.argv.index("--") + 1])
print("\n=== IMAGES ===")
for im in bpy.data.images:
    if im.size[0]:
        print(f"  {im.name[:44]:44s} {im.size[0]}x{im.size[1]} packed={im.packed_file is not None} "
              f"cs={im.colorspace_settings.name}")
print("\n=== material -> Base Color / Normal / Roughness image ===")
for m in bpy.data.materials:
    if not m.use_nodes: continue
    bsdf = next((n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if not bsdf: continue
    for slot in ("Base Color", "Normal", "Roughness"):
        inp = bsdf.inputs.get(slot)
        if inp and inp.is_linked:
            n = inp.links[0].from_node
            # walk one hop (e.g. through Normal Map / sRGB nodes)
            img = getattr(n, "image", None)
            if not img:
                for i in n.inputs:
                    if i.is_linked and getattr(i.links[0].from_node, "image", None):
                        img = i.links[0].from_node.image; break
            if img:
                print(f"  {m.name:20s} {slot:11s} <- {img.name} ({img.size[0]}x{img.size[1]})")
print("DUMP_DONE")
