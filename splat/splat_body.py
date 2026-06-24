#!/usr/bin/env python3
"""Stage 1: full SMPL-X BODY as rigged Gaussian splats, driven by Kimodo (body) + FLAME (face),
rendered with gsplat. Splats = per-vertex, ride the deforming mesh.
  python splat_body.py --kimodo amass.npz --arkit face.json --out o.mp4 [--frames 30 --scale 0.01 --yaw 0 --multiview --one]"""
import os, sys, argparse, json, numpy as np, torch, smplx
from PIL import Image
from gsplat import rasterization
sys.path.insert(0, "/lw/render"); from face_flame import FlameDriver
dev = "cuda"
ap = argparse.ArgumentParser()
ap.add_argument("--kimodo", required=True); ap.add_argument("--arkit", default="/lw/output/greenman_mp6.json")
ap.add_argument("--out", required=True); ap.add_argument("--frames", type=int, default=30)
ap.add_argument("--scale", type=float, default=0.012); ap.add_argument("--res", type=int, default=384)
ap.add_argument("--yaw", type=float, default=0.0); ap.add_argument("--multiview", action="store_true")
ap.add_argument("--one", action="store_true")
a = ap.parse_args()

def aa2mat(v):
    t=np.linalg.norm(v,axis=1,keepdims=True); t=np.clip(t,1e-8,None); k=v/t
    K=np.zeros((len(v),3,3),np.float32); K[:,0,1]=-k[:,2];K[:,0,2]=k[:,1];K[:,1,0]=k[:,2];K[:,1,2]=-k[:,0];K[:,2,0]=-k[:,1];K[:,2,1]=k[:,0]
    return (np.eye(3)[None]+np.sin(t)[:,:,None]*K+(1-np.cos(t))[:,:,None]*(K@K)).astype(np.float32)
def mat2aa(R):
    tr=np.clip((R[:,0,0]+R[:,1,1]+R[:,2,2]-1)/2,-1,1); ang=np.arccos(tr)
    v=np.stack([R[:,2,1]-R[:,1,2],R[:,0,2]-R[:,2,0],R[:,1,0]-R[:,0,1]],1); n=np.linalg.norm(v,axis=1,keepdims=True); n[n<1e-8]=1
    return ((v/n)*ang[:,None]).astype(np.float32)
def look_at(eye,tgt,up=(0,1,0)):
    eye=np.array(eye,np.float32);tgt=np.array(tgt,np.float32);up=np.array(up,np.float32)
    z=tgt-eye;z/=np.linalg.norm(z); x=np.cross(z,up);x/=np.linalg.norm(x); y=np.cross(z,x)
    vm=np.eye(4,dtype=np.float32);vm[:3,:3]=np.stack([x,y,z],0);vm[:3,3]=-np.stack([x,y,z],0)@eye; return vm

# Kimodo body
d=np.load(a.kimodo,allow_pickle=True); FPS=int(d["mocap_frame_rate"]) if "mocap_frame_rate" in d else 30
body=d["pose_body"].astype(np.float32); go0=d["root_orient"].astype(np.float32); tr0=d["trans"].astype(np.float32)
Ftot=len(body); F=1 if a.one else min(a.frames,Ftot)
body,go0,tr0=body[:F],go0[:F],tr0[:F]
th=np.radians(a.yaw); Rfix=np.array([[1,0,0],[0,0,1],[0,-1,0]],np.float32)
Ry=np.array([[np.cos(th),0,np.sin(th)],[0,1,0],[-np.sin(th),0,np.cos(th)]],np.float32); M=(Ry@Rfix).astype(np.float32)
go=mat2aa(M[None]@aa2mat(go0)); transl=(tr0@M.T).astype(np.float32)
# FLAME face (mouth) resampled to F
e,j,l,r=FlameDriver("/work/tools/mp2flame/mappings").drive(json.load(open(a.arkit)),F,FPS,gain=0.6); j=j*2.4
# texture vertex color
uv=np.load("/work/assets/smplx_uv_2023.npz")["uv_coordinates"].astype(np.float32)
tex=np.asarray(Image.open("/work/assets/smplx_texture_m_alb.png").convert("RGB"),np.float32)/255.0
model=smplx.create("/work/models",model_type="smplx",gender="male",num_betas=10,use_pca=False,flat_hand_mean=True,num_expression_coeffs=100,batch_size=F)
faces=model.faces.astype(np.int64); NV=10475; vcol=np.zeros((NV,3),np.float32); seen=np.zeros(NV,bool)
Ht,Wt=tex.shape[:2]
for li,vi in enumerate(faces.reshape(-1)):
    if not seen[vi]:
        u,vv=uv[li]; vcol[vi]=tex[min(Ht-1,int((1-vv)*Ht)),min(Wt-1,int(u*Wt))]; seen[vi]=True
vcol_t=torch.tensor(vcol,device=dev)
with torch.no_grad():
    V=model(betas=torch.zeros(F,10),global_orient=torch.from_numpy(go),body_pose=torch.from_numpy(body),
            transl=torch.from_numpy(transl),expression=torch.from_numpy(e),jaw_pose=torch.from_numpy(j),
            leye_pose=torch.from_numpy(l),reye_pose=torch.from_numpy(r)).vertices.numpy().astype(np.float32)
# full-body framing (fixed, fit whole body height)
allv=V.reshape(-1,3); mn=allv.min(0); mx=allv.max(0); ctr=(mn+mx)/2; bodyH=mx[1]-mn[1]
YFOV=0.85; dist=bodyH/(2*np.tan(0.5*YFOV))*1.15; res=a.res
K=torch.tensor([[[res/(2*np.tan(YFOV/2)),0,res/2],[0,res/(2*np.tan(YFOV/2)),res/2],[0,0,1]]],dtype=torch.float32,device=dev)
quats=torch.tensor([[1,0,0,0]]*NV,dtype=torch.float32,device=dev); scales=torch.full((NV,3),a.scale,device=dev); opac=torch.ones(NV,device=dev)
FRAMEDIR=os.path.join(os.path.dirname(a.out),"_splatframes"); os.makedirs(FRAMEDIR,exist_ok=True)
yaws=np.linspace(-45,45,F) if a.multiview else np.zeros(F)
for fi in range(F):
    ang=np.radians(yaws[fi]); eye=[ctr[0]+dist*np.sin(ang),ctr[1],ctr[2]+dist*np.cos(ang)]
    vm=torch.tensor(look_at(eye,[ctr[0],ctr[1],ctr[2]]),device=dev)[None]
    out,_,_=rasterization(torch.tensor(V[fi],device=dev),quats,scales,opac,vcol_t,vm,K,res,res)
    Image.fromarray((out[0].clamp(0,1)*255).byte().cpu().numpy()).save(f"{FRAMEDIR}/f_{fi:04d}.png")
print("SPLAT_BODY_OK", FRAMEDIR, "F", F)
