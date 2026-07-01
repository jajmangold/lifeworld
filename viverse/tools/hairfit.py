"""Fit a composed VRM's HAIR/HAT mesh to a realistic (smaller) head.
  --seat  : DETERMINISTIC. Measure the head mesh's size/crown in the hat's coordinate frame (Avatar_Head
            local) and scale+place the hat there. Knobs: --fit (width ratio, 1.0=head width) --lift
            (fraction of head height to raise the hat toward the crown).
  (legacy): --scale/--dy manual, scaling the hair node about its centroid.
Hair/hat node = mesh whose material starts with 'Hair'. Head = meshes whose material starts with 'AvatarHead'.
  python3 hairfit.py --in x.vrm --out y.vrm --seat --fit 1.1 --lift 0.15
  python3 hairfit.py --in x.vrm --out y.vrm --scale 0.6 --dy 0.1
"""
import sys, json, struct
import numpy as np

def arg(f, d=None): return sys.argv[sys.argv.index(f)+1] if f in sys.argv else d
INP = arg("--in"); OUT = arg("--out")
SEAT = "--seat" in sys.argv
FIT = float(arg("--fit", "1.1")); LIFT = float(arg("--lift", "0.15"))
S = float(arg("--scale", "1.0")); DY = float(arg("--dy", "0.0"))

d = open(INP, "rb").read(); assert d[:4] == b"glTF"
off = 12; chunks = []
while off < len(d):
    clen = struct.unpack("<I", d[off:off+4])[0]; ctype = d[off+4:off+8]; cdata = d[off+8:off+8+clen]
    chunks.append([ctype, bytearray(cdata)]); off += 8 + clen
js = json.loads(bytes(chunks[0][1]))
nodes = js.get("nodes", []); mats = js.get("materials", [])

def mat_startswith(mi, pre):
    for pr in js["meshes"][mi].get("primitives", []):
        mt = pr.get("material")
        if mt is not None and mats[mt].get("name", "").startswith(pre): return True
    return False

def mesh_bbox(mi):  # union of primitive POSITION accessor min/max (local mesh space)
    lo = np.array([1e9]*3); hi = np.array([-1e9]*3)
    for pr in js["meshes"][mi].get("primitives", []):
        acc = js["accessors"][pr["attributes"]["POSITION"]]
        if acc.get("min") and acc.get("max"):
            lo = np.minimum(lo, acc["min"]); hi = np.maximum(hi, acc["max"])
    return lo, hi

def quat_mat(q):
    x, y, z, w = q; M = np.eye(4)
    M[:3, :3] = np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])
    return M

def local_mat(n):
    if "matrix" in n: return np.array(n["matrix"]).reshape(4, 4).T
    T = np.eye(4); R = np.eye(4); Sc = np.eye(4)
    if "translation" in n: T[:3, 3] = n["translation"]
    if "rotation" in n: R = quat_mat(n["rotation"])
    if "scale" in n: Sc[:3, :3] = np.diag(n["scale"])
    return T @ R @ Sc

# world matrices via DFS from scene roots
world = {}
def dfs(i, parent):
    world[i] = parent @ local_mat(nodes[i])
    for c in nodes[i].get("children", []): dfs(c, world[i])
for r in js["scenes"][js.get("scene", 0)]["nodes"]: dfs(r, np.eye(4))

names = [n.get("name", "") for n in nodes]
head_bone = names.index("Avatar_Head")
Mh = world[head_bone]; Mh_inv = np.linalg.inv(Mh)

def corners(lo, hi):
    return np.array([[lo[0] if i&1 else hi[0], lo[1] if i&2 else hi[1], lo[2] if i&4 else hi[2], 1.0]
                     for i in range(8)]).T  # 4x8

hair_nodes = [i for i, n in enumerate(nodes) if "mesh" in n and mat_startswith(n["mesh"], "Hair")]
print("hair/hat nodes:", [(i, names[i]) for i in hair_nodes])

if SEAT:
    # head extent in Avatar_Head-local frame (skinned head accessors are model-space in rest pose)
    hlo = np.array([1e9]*3); hhi = np.array([-1e9]*3)
    for i, n in enumerate(nodes):
        if "mesh" in n and mat_startswith(n["mesh"], "AvatarHead"):
            lo, hi = mesh_bbox(n["mesh"]); C = Mh_inv @ corners(lo, hi)
            hlo = np.minimum(hlo, C[:3].min(1)); hhi = np.maximum(hhi, C[:3].max(1))
    Hc = (hlo + hhi) / 2; Hw = hhi[0] - hlo[0]; Hh = hhi[1] - hlo[1]
    print(f"head(AH-local) center={Hc.round(3)} width={Hw:.3f} height={Hh:.3f} crownY={hhi[1]:.3f}")
    target = np.array([Hc[0], Hc[1] + LIFT*Hh, Hc[2]])   # a bit above head center -> caps the crown
    for i in hair_nodes:
        H = local_mat(nodes[i]); lo, hi = mesh_bbox(nodes[i]["mesh"])
        Cc = H @ corners(lo, hi); clo = Cc[:3].min(1); chi = Cc[:3].max(1)
        Chat = (clo + chi) / 2; What = chi[0] - clo[0]
        s = (Hw / What) * FIT
        Tt = np.eye(4); Tt[:3, 3] = target
        Sc = np.eye(4); Sc[:3, :3] = np.diag([s, s, s])
        Tc = np.eye(4); Tc[:3, 3] = -Chat
        Hp = Tt @ Sc @ Tc @ H
        nodes[i].pop("translation", None); nodes[i].pop("rotation", None); nodes[i].pop("scale", None)
        nodes[i]["matrix"] = list(Hp.T.flatten())
        print(f"  seat node {i}: scale={s:.3f} hatW={What:.3f} -> target={target.round(3)}")
else:
    def mesh_center(mi):
        lo, hi = mesh_bbox(mi); return (lo + hi) / 2
    for i in hair_nodes:
        n = nodes[i]; sc = n.get("scale", [1.0, 1.0, 1.0]); n["scale"] = [sc[0]*S, sc[1]*S, sc[2]*S]
        C = mesh_center(n["mesh"]); tr = n.get("translation", [0.0, 0.0, 0.0])
        n["translation"] = [tr[0]+(1-S)*C[0], tr[1]+(1-S)*C[1]+DY, tr[2]+(1-S)*C[2]]
        print(f"  node {i}: scale->{n['scale']} translation->{[round(x,3) for x in n['translation']]}")

nb = json.dumps(js).encode(); pad = (4 - (len(nb) % 4)) % 4
chunks[0][1] = bytearray(nb + b" " * pad)
body = b""
for ctype, cdata in chunks:
    p = (4 - (len(cdata) % 4)) % 4; fill = b" " if ctype == b"JSON" else b"\x00"
    cd = bytes(cdata) + fill * p; body += struct.pack("<I", len(cd)) + ctype + cd
open(OUT, "wb").write(b"glTF" + struct.pack("<II", 2, 12 + len(body)) + body)
print(f"wrote {OUT}")
