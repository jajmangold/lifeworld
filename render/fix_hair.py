"""EEVEE hair/cloth material de-plasticiser. The recurring female-character failure (Business Female,
Formal woman) is HAIR/CLOTH rendering as wet rubber: a solid mesh with a default Principled BSDF that
has high specular + low roughness, so EEVEE gives it a glossy plastic sheen. This retunes those
materials toward how EEVEE actually reads as hair/fabric (per the FLUX/EEVEE hair-card guidance):

  * Roughness UP (~0.7)         -> kills the wet-plastic highlight; hair/cloth is a broad soft sheen.
  * Specular IOR Level DOWN     -> weakens the mirror hotspot that screams "rubber".
  * Anisotropic = 1             -> EEVEE has no hair BSDF, so this fakes the streaky lengthwise highlight.
  * Sheen small, Coat = 0       -> soft fabric/hair sheen, no clearcoat gloss.
  * Alpha -> DITHERED + double-sided -> real hair CARDS (alpha strand textures) show through, not as a shell.

Usage (standalone):  blender -b char.blend --python fix_hair.py -- --save out.blend [--match hair,wavy,bob,hijab,scarf]
Or import + call tune_hair_cloth(substrs, rough=0.7) after opening a .blend (render_character.py does this
when NEWS_HAIRFIX is set). Only touches materials whose name contains one of the match substrings — it never
touches skin/eyes. Solid rubber-TUBE hair (MakeHuman elvs_) still won't become photoreal (no strand geometry),
but this removes the plastic sheen; it shines on real hair cards and on satiny cloth (e.g. the hijab)."""
import bpy

DEFAULT_MATCH = ["hair", "wavy", "bob", "curl", "braid", "pony", "lash", "brow",
                 "hijab", "scarf", "veil", "cloth", "fabric", "dress", "shawl", "silk"]

def _set(bsdf, names, value):
    """Set the first Principled input whose name matches (handles Blender-version renames)."""
    for n in names:
        inp = bsdf.inputs.get(n)
        if inp is not None:
            try: inp.default_value = value; return True
            except Exception: pass
    return False

def _has_alpha(bsdf):
    a = bsdf.inputs.get("Alpha")
    return bool(a and (a.is_linked or a.default_value < 0.999))

def tune_hair_cloth(substrs=None, rough=0.7, spec=0.2, aniso=1.0, sheen=0.12):
    subs = [s.lower() for s in (substrs or DEFAULT_MATCH)]
    hit = []
    for m in bpy.data.materials:
        if not m.use_nodes or not any(s in m.name.lower() for s in subs):
            continue
        bsdf = next((n for n in m.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
        if not bsdf:
            continue
        _set(bsdf, ["Roughness"], rough)
        _set(bsdf, ["Specular IOR Level", "Specular"], spec)     # 4.x renamed "Specular"
        _set(bsdf, ["Anisotropic"], aniso)                       # fake streaky hair highlight
        _set(bsdf, ["Sheen Weight", "Sheen"], sheen)
        _set(bsdf, ["Coat Weight", "Clearcoat"], 0.0)            # no clearcoat gloss
        # double-sided so hair cards / thin cloth don't read as a hard shell
        if hasattr(m, "use_backface_culling"): m.use_backface_culling = False
        if hasattr(m, "show_transparent_back"): m.show_transparent_back = True
        if _has_alpha(bsdf):                                     # real hair-card alpha strands
            if hasattr(m, "surface_render_method"): m.surface_render_method = "DITHERED"  # EEVEE Next
            elif hasattr(m, "blend_method"): m.blend_method = "HASHED"                    # EEVEE legacy
        hit.append(m.name)
    print("FIX_HAIR tuned:", hit if hit else "(no matching materials)")
    return hit

if __name__ == "__main__":
    import sys
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    def a(f, d): return argv[argv.index(f) + 1] if f in argv else d
    match = a("--match", "")
    tune_hair_cloth(match.split(",") if match else None,
                    rough=float(a("--rough", "0.7")), spec=float(a("--spec", "0.2")))
    save = a("--save", "")
    if save:
        bpy.ops.wm.save_as_mainfile(filepath=save); print("FIX_HAIR saved ->", save)
