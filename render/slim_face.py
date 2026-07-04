"""Slim a character's lower face ('chipmunk cheeks' on MakeHuman base meshes — full cheeks + wide jaw, and
no cheek/jaw shape key to fix it). Narrows the cheek/jaw vertices toward the face midline with a smooth
falloff (no effect above the cheekbone or at the centerline, so the skull/forehead/nose are untouched).

CRITICAL: the head mesh has SHAPE KEYS, so its rest geometry is the 'Basis' shape key, NOT mesh.vertices[].co.
Moving only mesh.vertices does nothing. We compute an inward x-shift per vertex from the Basis, then apply the
SAME shift to every shape-key block (Basis + eye_close/brow/...) so the slim is consistent across all
expressions (shape keys store absolute positions, so a uniform delta preserves their relative deformation).

  from slim_face import slim ; slim(bpy, arm, amount=0.3)   # render_character calls this when NEWS_FACE_SLIM set
"""
from mathutils import Vector

FACEKEY = ("head", "face")

def slim(bpy, arm, amount=0.3):
    sc = bpy.context.scene
    head = next((o for o in sc.objects if o.type == "MESH" and any(k in o.name.lower() for k in FACEKEY)), None)
    if not head:
        print("SLIM_FACE: no head mesh"); return
    mw = head.matrix_world; mwi = mw.inverted()
    sk = head.data.shape_keys
    basis = sk.key_blocks["Basis"].data if sk else None
    def restco(i):                                   # rest position = Basis shape key (or mesh vert)
        return basis[i].co if basis else head.data.vertices[i].co
    hb = arm.pose.bones.get("head") or arm.pose.bones.get("Head")
    base_z = (arm.matrix_world @ hb.head).z              # ~ neck top / jaw base
    top_z = (arm.matrix_world @ hb.tail).z               # ~ crown
    H = top_z - base_z
    # smooth vertical band so the slim FADES OUT above the neck (no jaw->neck step, which clips the collar):
    cheekbone_z = base_z + 0.52 * H                      # fade-in top (0 above the cheekbone)
    full_top    = base_z + 0.34 * H
    full_bot    = base_z + 0.08 * H                      # full slim across the cheek/upper-jaw only
    chin_z      = base_z - 0.04 * H                      # fade-out bottom -> 0 (leaves the neck untouched)
    def _ss(t): t = max(0.0, min(1.0, t)); return t * t * (3 - 2 * t)
    def band_w(z):
        if z >= cheekbone_z or z <= chin_z: return 0.0
        if z >= full_top: return _ss((cheekbone_z - z) / (cheekbone_z - full_top))
        if z >= full_bot: return 1.0
        return _ss((z - chin_z) / (full_bot - chin_z))
    xs = [(mw @ restco(i)).x for i in range(len(head.data.vertices))]
    midx = (min(xs) + max(xs)) / 2
    halfw = 0.5 * (max(xs) - min(xs)) + 1e-6
    n = 0
    for i in range(len(head.data.vertices)):
        w = mw @ restco(i)
        wz = band_w(w.z)
        if wz <= 1e-4:
            continue
        latw = min(1.0, abs(w.x - midx) / halfw)                                            # only the lateral mass
        f = amount * wz * latw
        if f <= 1e-5:
            continue
        neww = Vector((midx + (w.x - midx) * (1.0 - f), w.y, w.z))
        delta = (mwi @ neww) - (mwi @ w)             # local-space inward shift for this vertex
        if sk:
            for kb in sk.key_blocks:                 # apply to EVERY shape key (keeps expressions consistent)
                kb.data[i].co += delta
        head.data.vertices[i].co += delta
        n += 1
    print(f"SLIM_FACE ok: {head.name}, {n} verts shifted (shape-key-aware={bool(sk)}), amount {amount}")
