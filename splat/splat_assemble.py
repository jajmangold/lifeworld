#!/usr/bin/env python3
"""First assembly: photoreal SHARP head splats placed on the SMPL-X body surface splats (rough
registration by head size/position). Both face +z. Render front + a turn.
  python splat_assemble.py --ply head.ply --out o_ [--res 512]"""
import os,sys,argparse,numpy as np,torch,smplx
from plyfile import PlyData
from PIL import Image
from gsplat import rasterization
sys.path.insert(0,"/lw/splat"); from splat_build import build_face_splats
dev="cuda"
ap=argparse.ArgumentParser(); ap.add_argument("--ply",required=True); ap.add_argument("--out",required=True)
ap.add_argument("--res",type=int,default=512); ap.add_argument("--orbit",action="store_true")
a=ap.parse_args()
def look_at(eye,tgt,up=(0,1,0)):
    eye=np.array(eye,np.float32);tgt=np.array(tgt,np.float32);up=np.array(up,np.float32)
    z=tgt-eye;z/=np.linalg.norm(z);x=np.cross(z,up);x/=np.linalg.norm(x);y=np.cross(z,x)
    vm=np.eye(4,dtype=np.float32);vm[:3,:3]=np.stack([x,y,z],0);vm[:3,3]=-np.stack([x,y,z],0)@eye;return vm

# --- SHARP head splats ---
p=PlyData.read(a.ply)["vertex"]; g=lambda k:np.asarray(p[k],np.float32)
hm=np.stack([g("x"),g("y"),g("z")],1); hs=np.exp(np.stack([g("scale_0"),g("scale_1"),g("scale_2")],1))
hq=np.stack([g("rot_0"),g("rot_1"),g("rot_2"),g("rot_3")],1); hq/=np.linalg.norm(hq,axis=1,keepdims=True)
ho=1/(1+np.exp(-g("opacity"))); hc=np.clip(0.2820948*np.stack([g("f_dc_0"),g("f_dc_1"),g("f_dc_2")],1)+0.5,0,1)
# crop: drop green bg + keep head cluster (front + upper)
green=hc[:,1]-np.maximum(hc[:,0],hc[:,2]); ymin,ymax=hm[:,1].min(),hm[:,1].max()
keep=(green<0.02)&(ho>0.3)&(hm[:,1]<ymin+0.55*(ymax-ymin))&(hm[:,2]<np.percentile(hm[:,2],70))
hm,hs,hq,ho,hc=hm[keep],hs[keep],hq[keep],ho[keep],hc[keep]
print("SHARP head splats after crop:",len(hm))
# head metric box (SHARP frame, OpenCV y-down)
hcen=hm.mean(0); hheight=hm[:,1].max()-hm[:,1].min()

# --- SMPL-X body surface splats (neutral, faces +z via yaw 180) ---
model=smplx.create("/work/models",model_type="smplx",gender="male",num_betas=10,use_pca=False,flat_hand_mean=True,num_expression_coeffs=100,batch_size=1)
go=np.zeros((1,3),np.float32); go[:,1]=np.radians(180.0)
with torch.no_grad():
    o=model(global_orient=torch.from_numpy(go),betas=torch.zeros(1,10))
V=o.vertices.numpy()[0].astype(np.float32); J=o.joints.numpy()[0].astype(np.float32)
uv=np.load("/work/assets/smplx_uv_2023.npz")["uv_coordinates"].astype(np.float32)
tex=np.asarray(Image.open("/work/assets/smplx_texture_m_alb.png").convert("RGB"),np.float32)/255.0
bcen,bq,bs,bcol=build_face_splats(V,model.faces.astype(np.int64),uv,tex)
# SMPL-X head extent (head joint 15 to crown)
headj=J[15]; crown=V[:,1].max(); smplx_head_h=(crown-headj[1])*2.2

# --- register SHARP head -> SMPL-X head: scale + translate (both face +z) ---
s=smplx_head_h/max(hheight,1e-6)
# SHARP y is image-down; SMPL-X y is up -> flip Y when placing
Hm=hm.copy(); Hm-=hcen; Hm[:,1]*=-1; Hm[:,2]*=-1; Hm*=s
target=np.array([headj[0], headj[1]+0.04, headj[2]],np.float32)   # nudge up toward crown
Hm+=target
Hs=hs*s
# combine
hq=np.stack([-hq[:,1],hq[:,0],-hq[:,3],hq[:,2]],1)  # 180-about-X on quats
means=np.concatenate([Hm,bcen]); scales=np.concatenate([Hs,bs])
# SHARP quats: flipping Y negates the rotation's y-handedness; approximate by keeping quats (rough)
quats=np.concatenate([hq,bq]); opac=np.concatenate([ho,np.ones(len(bcen))]); cols=np.concatenate([hc,bcol])
T=[torch.tensor(x.astype(np.float32),device=dev) for x in (means,quats,scales,opac,cols)]
# camera: full body
allv=means; mn,mx=allv.min(0),allv.max(0); ctr=(mn+mx)/2; bh=mx[1]-mn[1]
YFOV=0.85; dist=bh/(2*np.tan(0.5*YFOV))*1.15; res=a.res; fx=res/(2*np.tan(YFOV/2))
K=torch.tensor([[[fx,0,res/2],[0,fx,res/2],[0,0,1]]],dtype=torch.float32,device=dev)
os.makedirs(os.path.dirname(a.out),exist_ok=True)
angs=np.radians(np.linspace(-30,30,12)) if a.orbit else [0.0]
for i,ang in enumerate(angs):
    eye=[ctr[0]+dist*np.sin(ang),ctr[1],ctr[2]+dist*np.cos(ang)]
    vm=torch.tensor(look_at(eye,[ctr[0],ctr[1],ctr[2]]),device=dev)[None]
    out,_,_=rasterization(T[0],T[1],T[2],T[3],T[4],vm,K,res,res)
    Image.fromarray((out[0].clamp(0,1)*255).byte().cpu().numpy()).save(f"{a.out}{i:03d}.png")
print("ASSEMBLE_OK head",len(hm),"body",len(bcen),"scale %.2f"%s)
