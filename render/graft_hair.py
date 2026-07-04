"""Graft a HAAR strand groom onto a character in an open Blender scene, parented to the head bone so it
animates with the idle head motion. Called by render_character.py when NEWS_HAIR_GROOM is set. The real fix
for photoreal female hair (vs. the hijab stopgap): style/length/color come from the HAAR text prompt.

Key gotchas baked in (learned the hard way):
  * HAAR/NeuralHaircut output is Y-UP -> remap (x,y,z)->(x,-z,y) to Blender Z-up, else the hair points backward.
  * Hide native hair by OBJECT NAME only — the face/Head mesh often SHARES the hair material, so material-based
    hiding deletes the face.
  * Anchor to the SKULL (Head/face mesh) bbox, NOT the poofy native-hair bbox (which is oversized).
  * Per-strand melanin variance (Hair Info > Random) so it isn't a flat synthetic block.
  * CHILD_OF constraint to the head bone (bound at rest pose) so the hair follows head nods/turns.
"""
import numpy as np
from mathutils import Vector

HAIRKEY = ("hair", "wavy_bob", "ponytail", "braid", "updo")
FACEKEY = ("head", "face", "body", "skin", "eye", "brow", "lash", "teeth", "tongue")

def _has(n, keys): return any(k in n.lower() for k in keys)

def graft(bpy, arm, groom_ply, head_bone, melanin=0.5, yaw=0.0, dup=6, ppstrand=100):
    sc = bpy.context.scene
    # --- hide native scalp hair by object name; protect the face/head ---
    hair_objs = []
    for ob in sc.objects:
        if ob.type != "MESH" or _has(ob.name, FACEKEY):
            continue
        mats = [sl.material.name for sl in ob.material_slots if sl.material]
        if _has(ob.name, HAIRKEY) or (mats and all(_has(m, HAIRKEY) for m in mats)):
            hair_objs.append(ob)
    for ob in hair_objs:
        ob.hide_render = True; ob.hide_viewport = True
    # --- skull anchor ---
    def bbox(ob):
        P = np.array([[ (ob.matrix_world @ Vector(c))[i] for i in range(3)] for c in ob.bound_box])
        return P.mean(0), P.min(0), P.max(0)
    skull = next((ob for ob in sc.objects if ob.type == "MESH" and ob not in hair_objs
                  and _has(ob.name, ("head", "face"))), None)
    if skull:
        sctr, smin, smax = bbox(skull); ssize = smax - smin
    else:
        pb = arm.pose.bones.get(head_bone); ht = arm.matrix_world @ pb.tail
        sctr = np.array([ht.x, ht.y, ht.z]); ssize = np.array([0.19, 0.22, 0.27])
    HEAD_W, HEAD_TOP, HEAD_Y = float(ssize[0]), float(sctr[2] + ssize[2]*0.5), float(sctr[1])

    # --- load groom, Y-up -> Z-up, densify+fit ---
    bpy.ops.wm.ply_import(filepath=groom_ply)
    pc = bpy.context.selected_objects[0]
    V = np.array([v.co[:] for v in pc.data.vertices]); bpy.data.objects.remove(pc, do_unlink=True)
    n = len(V)//ppstrand; S = V[:n*ppstrand].reshape(n, ppstrand, 3)
    S = np.stack([S[..., 0], -S[..., 2], S[..., 1]], axis=-1)          # Y-up -> Z-up
    allp = S.reshape(-1, 3); gc = allp.mean(0); gsize = allp.max(0) - allp.min(0)
    scale = float(HEAD_W * 1.4 / max(gsize[0], 1e-6))
    th = np.radians(yaw); Rz = np.array([[np.cos(th), -np.sin(th), 0], [np.sin(th), np.cos(th), 0], [0, 0, 1]])
    def rs(p): return Rz @ ((p - gc) * scale)
    tp = np.array([rs(p) for p in allp]); tmax = tp.max(0); tcen = tp.mean(0)
    shift = np.array([-tcen[0], HEAD_Y + 0.015 - tcen[1], (HEAD_TOP + 0.01) - tmax[2]])
    rng = np.random.default_rng(1); trel = np.linspace(0, 1, ppstrand)[:, None]

    cu = bpy.data.curves.new("haar_hair", "CURVE"); cu.dimensions = "3D"
    cu.bevel_depth = 0.00016 * scale; cu.bevel_resolution = 0
    for strand in S:
        for d in range(dup):
            jit = (rng.normal(0, 0.0038, (1, 3)) * trel) if d else 0.0
            pts = strand + jit
            sp = cu.splines.new("POLY"); sp.points.add(ppstrand - 1)
            for k in range(ppstrand):
                w = rs(pts[k]) + shift; sp.points[k].co = (float(w[0]), float(w[1]), float(w[2]), 1.0)
    hair = bpy.data.objects.new("haar_hair", cu); sc.collection.objects.link(hair)

    # --- material: Principled Hair BSDF w/ per-strand melanin variance ---
    hm = bpy.data.materials.new("haar_hairmat"); hm.use_nodes = True
    nt = hm.node_tree; nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial"); hb = nt.nodes.new("ShaderNodeBsdfHairPrincipled")
    try: hb.parametrization = "MELANIN"
    except Exception: pass
    for nm, val in [("Melanin Redness", 0.4), ("Roughness", 0.3), ("Radial Roughness", 0.42)]:
        if nm in hb.inputs:
            try: hb.inputs[nm].default_value = val
            except Exception: pass
    if "Melanin" in hb.inputs:
        info = nt.nodes.new("ShaderNodeHairInfo"); mr = nt.nodes.new("ShaderNodeMapRange")
        mr.inputs["To Min"].default_value = max(0.0, melanin - 0.14); mr.inputs["To Max"].default_value = min(1.0, melanin + 0.14)
        nt.links.new(info.outputs["Random"], mr.inputs["Value"]); nt.links.new(mr.outputs["Result"], hb.inputs["Melanin"])
    nt.links.new(hb.outputs[0], out.inputs["Surface"]); hair.data.materials.append(hm)

    # --- follow the head bone (bind at current/rest pose) ---
    pb = arm.pose.bones.get(head_bone)
    if pb:
        con = hair.constraints.new("CHILD_OF"); con.target = arm; con.subtarget = head_bone
        con.inverse_matrix = (arm.matrix_world @ pb.matrix).inverted()
    print(f"GRAFT_HAIR ok: {len(S)} strands x{dup}, scale {scale:.3f}, hid {[o.name for o in hair_objs]}, head_bone {head_bone}")
    return hair
