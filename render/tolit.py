"""Convert an UNLIT cartoonish viverse avatar (KHR_materials_unlit -> flat under HDRI) into LIT PBR so it
renders in our EEVEE pipeline like the anchor. For each material, rebuild a Principled BSDF driven by the
material's existing base-color image (or color). Exports a GLB. Lets us use an offline-composed cartoonish
avatar (chosen hairstyle that FITS the cartoonish head) on the anchor path.
  blender -b --python tolit.py -- --in x.glb --out x_lit.glb [--rough 0.55]
"""
import bpy, sys
def a(f, d):
    argv = sys.argv[sys.argv.index("--")+1:]; return argv[argv.index(f)+1] if f in argv else d
INP = a("--in", "/work/cartoon_reporter.glb"); OUT = a("--out", "/work/cartoon_lit.glb")
ROUGH = float(a("--rough", "0.55"))

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=INP)
for mat in bpy.data.materials:
    if not mat.use_nodes: continue
    nt = mat.node_tree
    imgs = [n for n in nt.nodes if n.type == "TEX_IMAGE"]
    # find current base color (from a texture, or an emission/bg color)
    base_img = imgs[0] if imgs else None
    col = (0.8, 0.8, 0.8, 1.0)
    for n in nt.nodes:
        for inp in n.inputs:
            if inp.name in ("Color", "Base Color") and not inp.is_linked and hasattr(inp, "default_value"):
                try: col = tuple(inp.default_value)
                except Exception: pass
    nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
    bsdf.inputs["Roughness"].default_value = ROUGH
    bsdf.inputs["Base Color"].default_value = col
    nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
    if base_img is not None and base_img.image is not None:
        ti = nt.nodes.new("ShaderNodeTexImage"); ti.image = base_img.image
        nt.links.new(ti.outputs["Color"], bsdf.inputs["Base Color"])
    mat.use_backface_culling = False
print("converted", len(bpy.data.materials), "materials -> PBR")
bpy.ops.export_scene.gltf(filepath=OUT, export_format="GLB", use_selection=False,
                          export_yup=True, export_skins=True, export_morph=True)
print("TOLIT_OK ->", OUT)
