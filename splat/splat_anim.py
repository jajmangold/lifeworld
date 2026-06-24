#!/usr/bin/env python3
"""CAPSTONE: animated photoreal splat character. SHARP head gaussians are BOUND to the SMPL-X head
triangles (Gaussians-on-Mesh) so they ride head motion (turn) + FLAME jaw/expr (talk); body = SMPL-X
surface splats driven by Kimodo (gestures). All in 3D, one splat scene per frame.
  python splat_anim.py --ply head.ply --kimodo m.npz --arkit face.json --audio a.wav --out o.mp4
     [--frames 40 --res 420 --betas --hscale .9 --hcrop .66 --hy .05 --hz .04]"""
import os,sys,argparse,json,numpy as np,torch,smplx,subprocess
from plyfile import PlyData
from PIL import Image
from gsplat import rasterization
from scipy.spatial import cKDTree
sys.path.insert(0,"/lw/splat"); from splat_build import build_face_splats
sys.path.insert(0,"/lw/render"); from face_flame import FlameDriver
dev="cuda"
ap=argparse.ArgumentParser()
ap.add_argument("--ply",required=True); ap.add_argument("--kimodo",required=True)
ap.add_argument("--arkit",default="/o/greenman_mp6.json"); ap.add_argument("--audio",default="/o/sermon_6s.wav")
ap.add_argument("--out",required=True); ap.add_argument("--frames",type=int,default=40); ap.add_argument("--res",type=int,default=420)
ap.add_argument("--sam3d",default="/o/sam3d/greenman_skel.json"); ap.add_argument("--betas",action="store_true")
ap.add_argument("--hcrop",type=float,default=0.66); ap.add_argument("--hscale",type=float,default=0.9)
ap.add_argument("--hy",type=float,default=0.05); ap.add_argument("--hz",type=float,default=0.04); ap.add_argument("--closeup",action="store_true"); ap.add_argument("--jawgain",type=float,default=2.4); ap.add_argument("--upper",action="store_true")
a=ap.parse_args()
def look_at(eye,tgt,up=(0,1,0)):
    eye=np.array(eye,np.float32);tgt=np.array(tgt,np.float32);up=np.array(up,np.float32)
    z=tgt-eye;z/=np.linalg.norm(z);x=np.cross(z,up);x/=np.linalg.norm(x);y=np.cross(z,x)
    vm=np.eye(4,dtype=np.float32);vm[:3,:3]=np.stack([x,y,z],0);vm[:3,3]=-np.stack([x,y,z],0)@eye;return vm
def tri_frames(V,tri):                                   # tri (M,3) idx -> centroid (M,3), basis (M,3,3) cols=[t1,t2,n]
    P=V[tri]; c=P.mean(1); e1=P[:,1]-P[:,0]; e2=P[:,2]-P[:,0]
    n=np.cross(e1,e2); n/=np.clip(np.linalg.norm(n,axis=1,keepdims=True),1e-9,None)
    t1=e1/np.clip(np.linalg.norm(e1,axis=1,keepdims=True),1e-9,None); t2=np.cross(n,t1)
    return c.astype(np.float32), np.stack([t1,t2,n],2).astype(np.float32)

# --- SHARP head (load + crop) ---
p=PlyData.read(a.ply)["vertex"]; g=lambda k:np.asarray(p[k],np.float32)
hm=np.stack([g("x"),g("y"),g("z")],1); hs=np.exp(np.stack([g("scale_0"),g("scale_1"),g("scale_2")],1))
hq=np.stack([g("rot_0"),g("rot_1"),g("rot_2"),g("rot_3")],1); hq/=np.linalg.norm(hq,axis=1,keepdims=True)
ho=1/(1+np.exp(-g("opacity"))); hc=np.clip(0.2820948*np.stack([g("f_dc_0"),g("f_dc_1"),g("f_dc_2")],1)+0.5,0,1)
green=hc[:,1]-np.maximum(hc[:,0],hc[:,2]); ymin,ymax=hm[:,1].min(),hm[:,1].max()
keep=(green<0.02)&(ho>0.3)&(hm[:,1]<ymin+a.hcrop*(ymax-ymin))&(hm[:,2]<np.percentile(hm[:,2],85))
hm,hs,hq,ho,hc=hm[keep],hs[keep],hq[keep],ho[keep],hc[keep]
hcen=hm.mean(0); hheight=hm[:,1].max()-hm[:,1].min()

# --- Kimodo body ---
d=np.load(a.kimodo,allow_pickle=True); FPS=int(d["mocap_frame_rate"]) if "mocap_frame_rate" in d else 30
kb=d["pose_body"].astype(np.float32); F=min(a.frames,len(kb)); kb=kb[:F]
betas=np.zeros((1,10),np.float32)
if a.betas: betas[0]=np.array(json.load(open(a.sam3d))["shape_params"][:10],np.float32)
# FLAME talk (resampled to F)
e,j,l,r=FlameDriver("/work/tools/mp2flame/mappings").drive(json.load(open(a.arkit)),F,FPS,gain=0.6); j=j*a.jawgain

model=smplx.create("/work/models",model_type="smplx",gender="male",num_betas=10,use_pca=False,flat_hand_mean=True,num_expression_coeffs=100,batch_size=1)
faces=model.faces.astype(np.int64)
uv=np.load("/work/assets/smplx_uv_2023.npz")["uv_coordinates"].astype(np.float32)
tex=np.asarray(Image.open("/work/assets/smplx_texture_m_alb.png").convert("RGB"),np.float32)/255.0

# --- NEUTRAL (registration base): go=0, betas, A-pose ---
ap_pose=np.zeros((1,21,3),np.float32); ap_pose[0,15]=[0,0,-1.15]; ap_pose[0,16]=[0,0,1.15]
with torch.no_grad():
    o0=model(global_orient=torch.zeros(1,3),betas=torch.from_numpy(betas),body_pose=torch.from_numpy(ap_pose.reshape(1,-1)))
V0=o0.vertices.numpy()[0].astype(np.float32); J0=o0.joints.numpy()[0].astype(np.float32)
neck_y=float(J0[12,1]); chest_y=float(J0[9,1]); headj=J0[15]; crown=V0[:,1].max(); smplx_head_h=(crown-headj[1])*2.2
# register SHARP -> SMPL-X: scale by vertical SPAN (upper: crown->chest so shoulders land on body), 180-about-X
s=((crown-chest_y) if a.upper else smplx_head_h)/max(hheight,1e-6)*a.hscale
Hm0=hm-hcen; Hm0[:,1]*=-1; Hm0[:,2]*=-1; Hm0*=s
if a.upper:                                               # align bust top to crown so it fills crown->chest, no gap
    Hm0+=np.array([headj[0], crown-Hm0[:,1].max()+a.hy, headj[2]+a.hz],np.float32)
else:
    Hm0+=np.array([headj[0],headj[1]+0.04+a.hy,headj[2]+a.hz],np.float32)
hq=np.stack([-hq[:,1],hq[:,0],-hq[:,3],hq[:,2]],1); Hs=hs*s
# head triangles + bind each SHARP gaussian to nearest (local coords in triangle frame)
tcen=V0[faces].mean(1); _bc=(chest_y if a.upper else neck_y-0.02); head_tri=np.where(tcen[:,1]>_bc)[0]
c0,B0=tri_frames(V0,faces[head_tri])
nn=cKDTree(c0).query(Hm0)[1]                              # nearest head triangle per gaussian
local=np.einsum('nij,nj->ni', np.transpose(B0[nn],(0,2,1)), Hm0-c0[nn]).astype(np.float32)
print("SHARP head",len(Hm0),"bound to",len(head_tri),"head triangles")

def quat_mul(a,b):
    aw,ax,ay,az=a[:,0],a[:,1],a[:,2],a[:,3]; bw,bx,by,bz=b[:,0],b[:,1],b[:,2],b[:,3]
    return np.stack([aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw],1)
def mat2quat(R):                                          # (M,3,3)->(M,4) wxyz
    w=np.sqrt(np.clip(1+R[:,0,0]+R[:,1,1]+R[:,2,2],0,None))/2; w=np.clip(w,1e-8,None)
    return np.stack([w,(R[:,2,1]-R[:,1,2])/(4*w),(R[:,0,2]-R[:,2,0])/(4*w),(R[:,1,0]-R[:,0,1])/(4*w)],1)

# --- per-frame animate ---
FRAMEDIR=os.path.join(os.path.dirname(a.out),"_animframes"); os.makedirs(FRAMEDIR,exist_ok=True)
res=a.res; YFOV=0.85; fx=res/(2*np.tan(YFOV/2))
K=torch.tensor([[[fx,0,res/2],[0,fx,res/2],[0,0,1]]],dtype=torch.float32,device=dev)
bp=np.tile(ap_pose,(F,1,1)); bp[:,:14]=kb.reshape(F,-1,3)[:,:14]   # Kimodo torso/arms; keep A-pose? no-> use kimodo arms
bp=kb.reshape(F,-1,3).copy()                               # full Kimodo body (gestures + head/neck)
with torch.no_grad():
    modelF=smplx.create("/work/models",model_type="smplx",gender="male",num_betas=10,use_pca=False,flat_hand_mean=True,num_expression_coeffs=100,batch_size=F)
    O=modelF(global_orient=torch.zeros(F,3),betas=torch.from_numpy(np.tile(betas,(F,1))),
            body_pose=torch.from_numpy(bp.reshape(F,-1)),expression=torch.from_numpy(e),
            jaw_pose=torch.from_numpy(j),leye_pose=torch.from_numpy(l),reye_pose=torch.from_numpy(r))
VV=O.vertices.numpy().astype(np.float32)
for fi in range(F):
    V=VV[fi]
    c1,B1=tri_frames(V,faces[head_tri])
    Hm=c1[nn]+np.einsum('nij,nj->ni',B1[nn],local)        # SHARP head deformed (talk+turn)
    R=np.einsum('nij,nkj->nik',B1[nn],B0[nn])             # per-gaussian rotation B1 B0^T
    Hq=quat_mul(mat2quat(R),hq); Hq/=np.linalg.norm(Hq,axis=1,keepdims=True)
    bcen,bq,bs,bcol=build_face_splats(V,faces,uv,tex)
    keepb=bcen[:,1]<(chest_y if a.upper else headj[1]+0.04); bcen,bq,bs,bcol=bcen[keepb],bq[keepb],bs[keepb],bcol[keepb]
    means=np.concatenate([Hm,bcen]);quats=np.concatenate([Hq,bq]);scales=np.concatenate([Hs,bs])
    opac=np.concatenate([ho,np.ones(len(bcen))]);cols=np.concatenate([hc,bcol])
    T=[torch.tensor(x.astype(np.float32),device=dev) for x in (means,quats,scales,opac,cols)]
    if fi==0:
        mn,mx=means.min(0),means.max(0); ctr=((mn+mx)/2).copy(); bh=mx[1]-mn[1]; dist=bh/(2*np.tan(0.5*YFOV))*1.15
        if a.closeup:
            cy=(chest_y+0.12) if a.upper else (headj[1]-0.14); sh=0.78 if a.upper else 0.62
            ctr=np.array([0.0, cy, mx[2]],np.float32); dist=sh/(2*np.tan(0.5*YFOV))*1.1
    vm=torch.tensor(look_at([ctr[0],ctr[1],ctr[2]+dist],[ctr[0],ctr[1],ctr[2]]),device=dev)[None]
    out,_,_=rasterization(T[0],T[1],T[2],T[3],T[4],vm,K,res,res)
    Image.fromarray((out[0].clamp(0,1)*255).byte().cpu().numpy()).save(f"{FRAMEDIR}/f_{fi:04d}.png")
print("ANIM_FRAMES_OK",FRAMEDIR,"F",F)
