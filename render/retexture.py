"""Upscale/enhance a character's BASE-COLOR (albedo) texture maps and repack into a new .blend. Only
albedo (never normal/roughness — upscaling those breaks lighting). Two modes:
  blender -b --python retexture.py -- export <in.blend> <texdir>          # dump base-color PNGs
  blender -b --python retexture.py -- import <in.blend> <texdir> <out.blend>  # load up_<name>.png, repack
"""
import bpy, sys, os
argv = sys.argv[sys.argv.index("--") + 1:]
mode, blend, texdir = argv[0], argv[1], argv[2]
bpy.ops.wm.open_mainfile(filepath=blend)

nodes = {}                                   # image_name -> the TEX_IMAGE node feeding a Base Color
for m in bpy.data.materials:
    if not m.use_nodes: continue
    bsdf = next((n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
    if not bsdf: continue
    bc = bsdf.inputs.get("Base Color")
    if bc and bc.is_linked:
        n = bc.links[0].from_node
        if n.type == "TEX_IMAGE" and n.image and n.image.size[0]:
            nodes[n.image.name] = n

os.makedirs(texdir, exist_ok=True)
if mode == "export":
    for name, node in nodes.items():
        im = node.image
        p = os.path.join(texdir, name if name.lower().endswith(".png") else name + ".png")
        im.filepath_raw = p; im.file_format = "PNG"; im.save()
        print(f"EXPORT {name} {im.size[0]}x{im.size[1]} -> {p}")
    print("RETEX_EXPORT_DONE")
elif mode == "import":
    out = argv[3]
    for name, node in nodes.items():
        base = name if name.lower().endswith(".png") else name + ".png"
        up = os.path.join(texdir, "up_" + base)
        if os.path.exists(up):
            ni = bpy.data.images.load(up); ni.colorspace_settings.name = node.image.colorspace_settings.name
            ni.pack(); node.image = ni
            print(f"IMPORT {name} -> {ni.size[0]}x{ni.size[1]}")
    bpy.ops.wm.save_as_mainfile(filepath=out)
    print("RETEX_IMPORT_DONE ->", out)
