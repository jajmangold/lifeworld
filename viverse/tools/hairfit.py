"""Scale/reseat a composed VRM's HAIR mesh so a cartoonish hairstyle fits a realistic (smaller) head.
The composer parents hair to Avatar_Head with its authored (cartoonish-size) transform -> oversized. This
post-processes the .vrm: find the hair node(s) (mesh with a material named 'Hair*'), multiply their scale
by --scale and shift by --dy (about the head). No binary edits (only node transforms in the JSON chunk).
Driven by Qwen-in-the-loop: render -> ask if too big/small/high/low -> adjust --scale/--dy -> repeat.
  python3 hairfit.py --in x.vrm --out y.vrm --scale 0.5 --dy 0.0
"""
import sys, json, struct

def arg(f, d=None): return sys.argv[sys.argv.index(f)+1] if f in sys.argv else d
INP = arg("--in"); OUT = arg("--out"); S = float(arg("--scale", "1.0")); DY = float(arg("--dy", "0.0"))

d = open(INP, "rb").read(); assert d[:4] == b"glTF"
off = 12; chunks = []
while off < len(d):
    clen = struct.unpack("<I", d[off:off+4])[0]; ctype = d[off+4:off+8]; cdata = d[off+8:off+8+clen]
    chunks.append([ctype, bytearray(cdata)]); off += 8 + clen
js = json.loads(bytes(chunks[0][1]))

mats = js.get("materials", [])
def is_hair_mesh(mi):
    for pr in js["meshes"][mi].get("primitives", []):
        mt = pr.get("material")
        if mt is not None and mats[mt].get("name", "").startswith("Hair"):
            return True
    return False

def mesh_center(mi):
    # bbox center of the mesh in its local space (from accessor min/max — no binary decode)
    lo = [1e9, 1e9, 1e9]; hi = [-1e9, -1e9, -1e9]
    for pr in js["meshes"][mi].get("primitives", []):
        acc = js["accessors"][pr["attributes"]["POSITION"]]
        mn = acc.get("min"); mx = acc.get("max")
        if not mn or not mx: continue
        for k in range(3): lo[k] = min(lo[k], mn[k]); hi[k] = max(hi[k], mx[k])
    return [(lo[k]+hi[k])/2 for k in range(3)]

hair_nodes = [i for i, n in enumerate(js.get("nodes", [])) if "mesh" in n and is_hair_mesh(n["mesh"])]
print("hair nodes:", [(i, js["nodes"][i].get("name")) for i in hair_nodes])
for i in hair_nodes:
    n = js["nodes"][i]
    sc = n.get("scale", [1.0, 1.0, 1.0])
    n["scale"] = [sc[0]*S, sc[1]*S, sc[2]*S]
    # scale ABOUT the hair centroid (not the node origin at the neck): translate by (1-S)*C so the hair
    # shrinks in place on the head instead of collapsing toward the root. --dy is an extra manual nudge.
    C = mesh_center(n["mesh"])
    tr = n.get("translation", [0.0, 0.0, 0.0])
    n["translation"] = [tr[0] + (1-S)*C[0], tr[1] + (1-S)*C[1] + DY, tr[2] + (1-S)*C[2]]
    print(f"  node {i}: scale->{n['scale']}  center={ [round(c,3) for c in C] }  translation->{n['translation']}")

nb = json.dumps(js).encode()
pad = (4 - (len(nb) % 4)) % 4
chunks[0][1] = bytearray(nb + b" " * pad)
body = b""
for ctype, cdata in chunks:
    p = (4 - (len(cdata) % 4)) % 4
    fill = b" " if ctype == b"JSON" else b"\x00"
    cd = bytes(cdata) + fill * p
    body += struct.pack("<I", len(cd)) + ctype + cd
open(OUT, "wb").write(b"glTF" + struct.pack("<II", 2, 12 + len(body)) + body)
print(f"wrote {OUT} (scale={S} dy={DY}, {len(hair_nodes)} hair node(s))")
