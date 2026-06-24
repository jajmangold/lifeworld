#!/usr/bin/env python3
"""Stage 0: SMPL-X mesh as RIGGED Gaussian splats (one splat per vertex, textured), driven by our
FLAME params, rendered with gsplat. Splats ride the mesh by construction (positions recomputed from
the deformed mesh each frame) -> the 3D analog of FLOAT's warp, temporally exact.
  python splat_rig.py --out o.mp4 [--frames 16 --static --multiview --scale 0.008]"""
import os, sys, argparse, subprocess, json, numpy as np, torch
import smplx
from PIL import Image
from gsplat import rasterization
sys.path.insert(0,"/lw/splat"); from splat_build import build_face_splats
sys.path.insert(0, "/lw/render"); from face_flame import FlameDriver
dev = "cuda"
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--frames", type=int, default=16)
ap.add_argument("--static", action="store_true"); ap.add_argument("--multiview", action="store_true")
ap.add_argument("--arkit", default="/lw/output/greenman_mp6.json")
ap.add_argument("--scale", type=float, default=0.008); ap.add_argument("--res", type=int, default=320); ap.add_argument("--yaw", type=float, default=0.0); ap.add_argument("--mode", default="face"); ap.add_argument("--facecolor", default=None)
a = ap.parse_args()
F = 1 if a.static else a.frames
BAKED = np.load(a.facecolor) if a.facecolor else None

# ---- per-vertex texture color (assign each vertex the UV of a corner that uses it) ----
uv = np.load("/work/assets/smplx_uv_2023.npz")["uv_coordinates"].astype(np.float32)  # (loops,2)
tex = np.asarray(Image.open("/work/assets/smplx_texture_m_alb.png").convert("RGB"), np.float32) / 255.0
model = smplx.create("/work/models", model_type="smplx", gender="male", num_betas=10,
                     use_pca=False, flat_hand_mean=True, num_expression_coeffs=100, batch_size=F)
faces = model.faces.astype(np.int64)                         # (Ftri,3)
NV = 10475
vcol = np.zeros((NV, 3), np.float32); seen = np.zeros(NV, bool)
loops = faces.reshape(-1)                                    # corner->vertex
Ht, Wt = tex.shape[:2]
for li, vi in enumerate(loops):
    if not seen[vi]:
        u, vv = uv[li]; px = min(Wt-1, int(u*Wt)); py = min(Ht-1, int((1-vv)*Ht))
        vcol[vi] = tex[py, px]; seen[vi] = True
vcol_t = torch.tensor(vcol, device=dev)

# ---- drive: neutral (static) or FLAME expr/jaw/eye + head pose from greenman ----
expr = np.zeros((F, 100), np.float32); jaw = np.zeros((F, 3), np.float32)
leye = np.zeros((F, 3), np.float32); reye = np.zeros((F, 3), np.float32)
go = np.zeros((F, 3), np.float32); go[:, 1] = np.radians(a.yaw)
if not a.static:
    e, j, l, r = FlameDriver("/work/tools/mp2flame/mappings").drive(json.load(open(a.arkit)), F, 25, gain=0.6)
    expr, jaw, leye, reye = e, j*2.4, l, r
with torch.no_grad():
    V = model(global_orient=torch.from_numpy(go), expression=torch.from_numpy(expr),
              jaw_pose=torch.from_numpy(jaw), leye_pose=torch.from_numpy(leye),
              reye_pose=torch.from_numpy(reye), betas=torch.zeros(F,10)).vertices.numpy().astype(np.float32)
# frame the head
ys = V[0,:,1]; top=float(ys.max()); headc=top-0.13

def look_at(eye, tgt, up=(0,1,0)):
    eye=np.array(eye,np.float32); tgt=np.array(tgt,np.float32); up=np.array(up,np.float32)
    z=tgt-eye; z/=np.linalg.norm(z)                          # forward (OpenCV +Z)
    x=np.cross(z,up); x/=np.linalg.norm(x)
    y=np.cross(z,x)
    R=np.stack([x,y,z],0)                                    # world->cam rotation (rows)
    vm=np.eye(4,dtype=np.float32); vm[:3,:3]=R; vm[:3,3]=-R@eye
    return vm

dist=0.42; res=a.res; K=torch.tensor([[[res*1.2,0,res/2],[0,res*1.2,res/2],[0,0,1]]],dtype=torch.float32,device=dev)
quats=torch.tensor([[1,0,0,0]]*NV,dtype=torch.float32,device=dev)
scales=torch.full((NV,3),a.scale,device=dev); opac=torch.ones(NV,device=dev)
FRAMEDIR=os.path.join(os.path.dirname(a.out),"_splatframes"); os.makedirs(FRAMEDIR,exist_ok=True)
yaws = np.linspace(-40,40,F) if a.multiview else np.zeros(F)
for fi in range(F):
    th=np.radians(yaws[fi]); eye=[dist*np.sin(th), headc, dist*np.cos(th)]
    vm=torch.tensor(look_at(eye,[0,headc,0]),device=dev)[None]
    vf=V[fi if not a.static else 0]
    if a.mode=="face":
        mn,qz,sc,cl=build_face_splats(vf,faces,uv,tex,cols=BAKED)
        means=torch.tensor(mn,device=dev); q=torch.tensor(qz,device=dev); s=torch.tensor(sc,device=dev)
        op=torch.ones(len(mn),device=dev); col=torch.tensor(cl,device=dev)
        out,_,_=rasterization(means,q,s,op,col,vm,K,res,res)
    else:
        means=torch.tensor(vf,device=dev)
        out,_,_=rasterization(means,quats,scales,opac,vcol_t,vm,K,res,res)
    Image.fromarray((out[0].clamp(0,1)*255).byte().cpu().numpy()).save(f"{FRAMEDIR}/f_{fi:04d}.png")
if a.static:
    Image.open(f"{FRAMEDIR}/f_0000.png").save(a.out.replace(".mp4",".png")); print("SPLAT_STATIC_OK", a.out.replace(".mp4",".png"))
else:
    print("SPLAT_FRAMES_OK", FRAMEDIR, "F", F)
