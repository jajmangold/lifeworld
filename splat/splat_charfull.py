#!/usr/bin/env python3
"""ONE consistent photoreal character: bind the FULL SHARP body (head+torso+clothes) to the SMPL-X
mesh (Gaussians-on-Mesh over the WHOLE body) -> no head/body seam, no material mismatch (it's all
one SHARP reconstruction). SMPL-X is the invisible rig only. Drive with Kimodo + FLAME.
  python splat_charfull.py --ply full.ply --kimodo m.npz --arkit f.json --audio a.wav --out o.mp4
     [--frames 40 --res 420 --betas --jawgain 0.5 --orbit --closeup]"""
import os,sys,argparse,json,numpy as np,torch,smplx
from plyfile import PlyData
from PIL import Image
from gsplat import rasterization
from scipy.spatial import cKDTree
sys.path.insert(0,"/lw/render"); from face_flame import FlameDriver
dev="cuda"
ap=argparse.ArgumentParser()
ap.add_argument("--ply",required=True); ap.add_argument("--kimodo",required=True)
ap.add_argument("--arkit",default="/o/greenman_mp6.json"); ap.add_argument("--audio",default="/o/sermon_6s.wav")
ap.add_argument("--out",required=True); ap.add_argument("--frames",type=int,default=40); ap.add_argument("--res",type=int,default=420)
ap.add_argument("--sam3d",default="/o/sam3d/greenman_skel.json"); ap.add_argument("--betas",action="store_true")
ap.add_argument("--jawgain",type=float,default=0.85); ap.add_argument("--orbit",action="store_true"); ap.add_argument("--armdown",type=float,default=1.25); ap.add_argument("--binddist",type=float,default=0.045); ap.add_argument("--closeup",action="store_true"); ap.add_argument("--cavity",action="store_true"); ap.add_argument("--armdamp",type=float,default=0.25); ap.add_argument("--k",type=int,default=4); ap.add_argument("--meshmode",action="store_true"); ap.add_argument("--lbs",action="store_true")
a=ap.parse_args()
def look_at(eye,tgt,up=(0,1,0)):
    eye=np.array(eye,np.float32);tgt=np.array(tgt,np.float32);up=np.array(up,np.float32)
    z=tgt-eye;z/=np.linalg.norm(z);x=np.cross(z,up);x/=np.linalg.norm(x);y=np.cross(z,x)
    vm=np.eye(4,dtype=np.float32);vm[:3,:3]=np.stack([x,y,z],0);vm[:3,3]=-np.stack([x,y,z],0)@eye;return vm
def tri_frames(V,tri):
    P=V[tri]; c=P.mean(1); e1=P[:,1]-P[:,0]; e2=P[:,2]-P[:,0]
    n=np.cross(e1,e2); n/=np.clip(np.linalg.norm(n,axis=1,keepdims=True),1e-9,None)
    t1=e1/np.clip(np.linalg.norm(e1,axis=1,keepdims=True),1e-9,None); t2=np.cross(n,t1)
    return c.astype(np.float32), np.stack([t1,t2,n],2).astype(np.float32)
def quat_mul(A,B):
    aw,ax,ay,az=A[:,0],A[:,1],A[:,2],A[:,3]; bw,bx,by,bz=B[:,0],B[:,1],B[:,2],B[:,3]
    return np.stack([aw*bw-ax*bx-ay*by-az*bz,aw*bx+ax*bw+ay*bz-az*by,aw*by-ax*bz+ay*bw+az*bx,aw*bz+ax*by-ay*bx+az*bw],1)
def mat2quat(R):
    w=np.sqrt(np.clip(1+R[:,0,0]+R[:,1,1]+R[:,2,2],0,None))/2; w=np.clip(w,1e-8,None)
    return np.stack([w,(R[:,2,1]-R[:,1,2])/(4*w),(R[:,0,2]-R[:,2,0])/(4*w),(R[:,1,0]-R[:,0,1])/(4*w)],1)

# --- SHARP full body: keep the person (drop green bg + far splats) ---
p=PlyData.read(a.ply)["vertex"]; g=lambda k:np.asarray(p[k],np.float32)
hm=np.stack([g("x"),g("y"),g("z")],1); hs=np.exp(np.stack([g("scale_0"),g("scale_1"),g("scale_2")],1))
hq=np.stack([g("rot_0"),g("rot_1"),g("rot_2"),g("rot_3")],1); hq/=np.linalg.norm(hq,axis=1,keepdims=True)
ho=1/(1+np.exp(-g("opacity"))); hc=np.clip(0.2820948*np.stack([g("f_dc_0"),g("f_dc_1"),g("f_dc_2")],1)+0.5,0,1)
green=hc[:,1]-np.maximum(hc[:,0],hc[:,2])
keep=np.ones(len(hm),bool) if a.meshmode else ((green<0.02)&(ho>0.3)&(hm[:,2]<np.percentile(hm[:,2],55)))
hm,hs,hq,ho,hc=hm[keep],hs[keep],hq[keep],ho[keep],hc[keep]
print("SHARP person splats:",len(hm))

# --- Kimodo + FLAME + betas ---
d=np.load(a.kimodo,allow_pickle=True); FPS=int(d["mocap_frame_rate"]) if "mocap_frame_rate" in d else 30
kb=d["pose_body"].astype(np.float32); F=min(a.frames,len(kb)); kb=kb[:F]
betas=np.zeros((1,10),np.float32)
if a.betas: betas[0]=np.array(json.load(open(a.sam3d))["shape_params"][:10],np.float32)
e,j,l,r=FlameDriver("/work/tools/mp2flame/mappings").drive(json.load(open(a.arkit)),F,FPS,gain=0.6); j=j*a.jawgain
model=smplx.create("/work/models",model_type="smplx",gender="male",num_betas=10,use_pca=False,flat_hand_mean=True,num_expression_coeffs=100,batch_size=1)
faces=model.faces.astype(np.int64)

# --- NEUTRAL bind pose: arms down to match greenman's standing photo ---
ap_pose=np.zeros((1,21,3),np.float32); ap_pose[0,15]=[0,0,-a.armdown]; ap_pose[0,16]=[0,0,a.armdown]
with torch.no_grad():
    o0=model(global_orient=torch.zeros(1,3),betas=torch.from_numpy(betas),body_pose=torch.from_numpy(ap_pose.reshape(1,-1)))
V0=o0.vertices.numpy()[0].astype(np.float32)
# register SHARP body -> SMPL-X: OpenCV->y-up (180-X), scale by HEIGHT, align feet + center
Hm0=hm.copy()
if not a.meshmode: Hm0[:,1]*=-1; Hm0[:,2]*=-1
sH=(V0[:,1].max()-V0[:,1].min())/max(Hm0[:,1].max()-Hm0[:,1].min(),1e-6)
Hm0*=sH; hq=hq if a.meshmode else np.stack([-hq[:,1],hq[:,0],-hq[:,3],hq[:,2]],1); Hs=hs*sH
Hm0[:,0]+=V0[:,0].mean()-Hm0[:,0].mean(); Hm0[:,2]+=V0[:,2].mean()-Hm0[:,2].mean()
Hm0[:,1]+=V0[:,1].min()-Hm0[:,1].min()                           # feet to feet
# bind via K-NEAREST-triangle SKINNING (blend K frames -> smooth across joints, no single-tri smear)
c0,B0=tri_frames(V0,faces); _tree=cKDTree(c0)
d1,_=_tree.query(Hm0); kd=d1<a.binddist                          # drop outliers (green edges/floaters)
Hm0,hq,Hs,ho,hc=Hm0[kd],hq[kd],Hs[kd],ho[kd],hc[kd]
if a.cavity:   # dark gaussians behind lips so open jaw shows interior (needs z-tuning)
    sys.path.insert(0,"/work"); from face.talk import lip_region
    lip=lip_region(model,betas[0])["idx"]; back=V0[lip].copy(); back[:,2]-=0.013
    cq=np.tile([1,0,0,0],(len(back),1)).astype(np.float32); cs=np.full((len(back),3),0.007,np.float32)
    co=np.ones(len(back),np.float32); cc=np.tile([0.03,0.015,0.015],(len(back),1)).astype(np.float32)
    Hm0=np.concatenate([Hm0,back]); hq=np.concatenate([hq,cq]); Hs=np.concatenate([Hs,cs])
    ho=np.concatenate([ho,co]); hc=np.concatenate([hc,cc])
KN=a.k; dK,NN=_tree.query(Hm0,k=KN)                              # (M,K) nearest triangles
W=(1.0/(dK+1e-6)).astype(np.float32); W/=W.sum(1,keepdims=True)  # inverse-distance blend weights
localK=np.einsum('mkij,mkj->mki',np.transpose(B0[NN],(0,1,3,2)),Hm0[:,None,:]-c0[NN]).astype(np.float32)  # (M,K,3)
print("bound",len(Hm0),"(dropped",int((~kd).sum()),"outliers) to",len(faces),"tris, k=",KN)

# --- animate ---
FRAMEDIR=os.path.join(os.path.dirname(a.out),"_cfframes"); os.makedirs(FRAMEDIR,exist_ok=True)
res=a.res; YFOV=0.85; fx=res/(2*np.tan(YFOV/2)); K=torch.tensor([[[fx,0,res/2],[0,fx,res/2],[0,0,1]]],dtype=torch.float32,device=dev)
bp=kb.reshape(F,-1,3).copy()
arm=[11,12,13,14,15,16,17,18,19,20]   # neck/head/collars/shoulders/elbows/wrists (keep torso+legs live)
bp[:,arm]=ap_pose[0,arm]+(bp[:,arm]-ap_pose[0,arm])*a.armdamp
modelF=smplx.create("/work/models",model_type="smplx",gender="male",num_betas=10,use_pca=False,flat_hand_mean=True,num_expression_coeffs=100,batch_size=F)
with torch.no_grad():
    O=modelF(global_orient=torch.zeros(F,3),betas=torch.from_numpy(np.tile(betas,(F,1))),
             body_pose=torch.from_numpy(bp.reshape(F,-1)),expression=torch.from_numpy(e),
             jaw_pose=torch.from_numpy(j),leye_pose=torch.from_numpy(l),reye_pose=torch.from_numpy(r))
VV=O.vertices.numpy().astype(np.float32)
Hq=torch.tensor(hq.astype(np.float32),device=dev); Hs_t=torch.tensor(Hs.astype(np.float32),device=dev)
ho_t=torch.tensor(ho.astype(np.float32),device=dev); hc_t=torch.tensor(hc.astype(np.float32),device=dev)
# --- true LBS skinning setup (per-splat SMPL-X skin weights + bone transforms) ---
if a.lbs:
    import smplx.lbs as LBS
    LW=model.lbs_weights.detach().numpy()[cKDTree(V0).query(Hm0)[1]].astype(np.float32)   # (M,55) nearest-vertex weights
    Hm0h=np.concatenate([Hm0,np.ones((len(Hm0),1),np.float32)],1)
    Jrest=LBS.vertices2joints(model.J_regressor, model.v_template.unsqueeze(0)+LBS.blend_shapes(torch.from_numpy(betas),model.shapedirs[:,:,:10]))
    def jointA(bp21,jaw3):                               # (B,21,3),(B,3) -> A (B,55,4,4) numpy
        Bn=len(bp21); fp=np.zeros((Bn,55,3),np.float32); fp[:,1:22]=bp21; fp[:,22]=jaw3
        rot=LBS.batch_rodrigues(torch.from_numpy(fp.reshape(-1,3))).reshape(Bn,55,3,3)
        _,A=LBS.batch_rigid_transform(rot,Jrest.expand(Bn,-1,-1),model.parents); return A.detach().numpy()
    Abind_inv=np.linalg.inv(jointA(ap_pose,np.zeros((1,3),np.float32))[0])                 # (55,4,4)
    deltaF=np.einsum('fjxy,jyz->fjxz',jointA(bp,j),Abind_inv).astype(np.float32)           # (F,55,4,4) bind->frame
angs=np.radians(np.linspace(-30,30,F)) if a.orbit else np.zeros(F)
for fi in range(F):
    if a.lbs:
        Tn=np.einsum('mj,jxy->mxy',LW,deltaF[fi])               # (M,4,4) blended bone transform per splat
        Hm=np.einsum('mxy,my->mx',Tn,Hm0h)[:,:3]
        HqF=quat_mul(mat2quat(Tn[:,:3,:3]),hq); HqF/=np.linalg.norm(HqF,axis=1,keepdims=True)
    else:
        V=VV[fi]; c1,B1=tri_frames(V,faces)
        posK=c1[NN]+np.einsum('mkij,mkj->mki',B1[NN],localK)
        Hm=(posK*W[:,:,None]).sum(1)
        n0=NN[:,0]; R=np.einsum('mij,mkj->mik',B1[n0],B0[n0]); HqF=quat_mul(mat2quat(R),hq); HqF/=np.linalg.norm(HqF,axis=1,keepdims=True)
    means=torch.tensor(Hm.astype(np.float32),device=dev); quats=torch.tensor(HqF.astype(np.float32),device=dev)
    if fi==0:
        mn,mx=Hm.min(0),Hm.max(0); ctr=((mn+mx)/2); bh=mx[1]-mn[1]; dist=bh/(2*np.tan(0.5*YFOV))*1.15
        if a.closeup: ctr=np.array([ctr[0],mx[1]-0.30,mx[2]],np.float32); dist=0.72/(2*np.tan(0.5*YFOV))*1.1
    ang=angs[fi]; eye=[ctr[0]+dist*np.sin(ang),ctr[1],ctr[2]+dist*np.cos(ang)]
    vm=torch.tensor(look_at(eye,[ctr[0],ctr[1],ctr[2]]),device=dev)[None]
    out,_,_=rasterization(means,quats,Hs_t,ho_t,hc_t,vm,K,res,res)
    Image.fromarray((out[0].clamp(0,1)*255).byte().cpu().numpy()).save(f"{FRAMEDIR}/f_{fi:04d}.png")
print("CHARFULL_OK",FRAMEDIR,"F",F)
