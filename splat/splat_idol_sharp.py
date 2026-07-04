#!/usr/bin/env python3
"""Bolt the SHARP head onto the IDOL body: register SHARP head to IDOL's canonical (vs_template) head,
pose per frame with IDOL's exported head bone transform, render with IDOL's exported camera (gsplat),
composite over the IDOL body frames. Uses IDOL's OWN transforms -> lands in IDOL's space.
  python splat_idol_sharp.py --export export.npz --ply head.ply --body bodyframes/ --out out/ [--res 1024 --one]"""
import os,sys,argparse,numpy as np,torch
from plyfile import PlyData
from PIL import Image
from gsplat import rasterization
dev="cuda"
ap=argparse.ArgumentParser()
ap.add_argument("--export",required=True); ap.add_argument("--ply",required=True)
ap.add_argument("--body",required=True); ap.add_argument("--out",required=True)
ap.add_argument("--res",type=int,default=1024); ap.add_argument("--one",action="store_true")
ap.add_argument("--hscale",type=float,default=1.0); ap.add_argument("--c2w",action="store_true"); ap.add_argument("--hy",type=float,default=0.0); ap.add_argument("--hcrop",type=float,default=0.62); ap.add_argument("--zback",type=float,default=100.0); ap.add_argument("--absscale",type=float,default=0.0)
a=ap.parse_args()
E=np.load(a.export); cam=E["cam"][0]; head_tfs=E["head_tfs"].astype(np.float32); vt=E["vs_template"].astype(np.float32)
K4=cam[:4]; ext=cam[4:20].reshape(4,4).astype(np.float32)
# IDOL canonical head region (vs_template, high Y)
ytop=vt[:,1].max(); headv=vt[vt[:,1]>ytop-0.25]; hc=headv.mean(0); hh=headv[:,1].max()-headv[:,1].min()
# SHARP head
p=PlyData.read(a.ply)["vertex"]; g=lambda k:np.asarray(p[k],np.float32)
hm=np.stack([g("x"),g("y"),g("z")],1); hs=np.exp(np.stack([g("scale_0"),g("scale_1"),g("scale_2")],1))
hq=np.stack([g("rot_0"),g("rot_1"),g("rot_2"),g("rot_3")],1); hq/=np.linalg.norm(hq,axis=1,keepdims=True)
ho=1/(1+np.exp(-g("opacity"))); hcol=np.clip(0.2820948*np.stack([g("f_dc_0"),g("f_dc_1"),g("f_dc_2")],1)+0.5,0,1)
green=hcol[:,1]-np.maximum(hcol[:,0],hcol[:,2]); ymin,ymax=hm[:,1].min(),hm[:,1].max()
keep=(green<0.02)&(ho>0.3)&(hm[:,1]<ymin+a.hcrop*(ymax-ymin))&(hm[:,2]<np.percentile(hm[:,2],a.zback))
hm,hs,hq,ho,hcol=hm[keep],hs[keep],hq[keep],ho[keep],hcol[keep]
# register SHARP -> IDOL canonical head: OpenCV y-down -> y-up (180-X); ABSOLUTE scale; align CROWN
Hm0=hm.copy(); Hm0-=hm.mean(0); Hm0[:,1]*=-1; Hm0[:,2]*=-1
s=a.absscale if a.absscale>0 else hh/max(Hm0[:,1].max()-Hm0[:,1].min(),1e-6)*a.hscale
Hm0*=s
crown=vt[:,1].max()
Hm0[:,0]+=hc[0]-Hm0[:,0].mean(); Hm0[:,2]+=hc[2]-Hm0[:,2].mean()
Hm0[:,1]+=crown-Hm0[:,1].max()+a.hy          # align head crown to body crown; hy nudges
hq=np.stack([-hq[:,1],hq[:,0],-hq[:,3],hq[:,2]],1); Hs=hs*s
Hm0h=np.concatenate([Hm0,np.ones((len(Hm0),1),np.float32)],1)
print("SHARP head",len(Hm0),"canon head center",np.round(hc,3),"hh %.3f"%hh,"scale %.2f"%s)
# camera
res=a.res; fx,fy,cx,cy=K4
K=torch.tensor([[[fx,0,cx],[0,fy,cy],[0,0,1]]],dtype=torch.float32,device=dev)
vm=np.linalg.inv(ext) if a.c2w else ext
vm_t=torch.tensor(vm,dtype=torch.float32,device=dev)[None]
def mat2quat(R):
    w=np.sqrt(np.clip(1+R[:,0,0]+R[:,1,1]+R[:,2,2],0,None))/2; w=np.clip(w,1e-8,None)
    return np.stack([w,(R[:,2,1]-R[:,1,2])/(4*w),(R[:,0,2]-R[:,2,0])/(4*w),(R[:,1,0]-R[:,0,1])/(4*w)],1)
def quat_mul(A,B):
    aw,ax,ay,az=A[:,0],A[:,1],A[:,2],A[:,3]; bw,bx,by,bz=B[:,0],B[:,1],B[:,2],B[:,3]
    return np.stack([aw*bw-ax*bx-ay*by-az*bz,aw*bx+ax*bw+ay*bz-az*by,aw*by-ax*bz+ay*bw+az*bx,aw*bz+ax*by-ay*bx+az*bw],1)
import glob
bodyfiles=sorted(glob.glob(a.body+"/*.png")); F=1 if a.one else min(len(head_tfs),len(bodyfiles))
os.makedirs(a.out,exist_ok=True)
Hs_t=torch.tensor(Hs.astype(np.float32),device=dev); ho_t=torch.tensor(ho.astype(np.float32),device=dev); hcol_t=torch.tensor(hcol.astype(np.float32),device=dev)
for fi in range(F):
    T=head_tfs[fi]  # (4,4) canonical->posed
    Hm=(T@Hm0h.T).T[:,:3]
    R=T[:3,:3]/ (np.linalg.det(T[:3,:3])**(1/3)+1e-9)
    HqF=quat_mul(mat2quat(R[None].repeat(len(hq),0)),hq); HqF/=np.linalg.norm(HqF,axis=1,keepdims=True)
    out,_,_=rasterization(torch.tensor(Hm.astype(np.float32),device=dev),torch.tensor(HqF.astype(np.float32),device=dev),Hs_t,ho_t,hcol_t,vm_t,K,res,res)
    head_img=(out[0].clamp(0,1)*255).byte().cpu().numpy()
    body=np.asarray(Image.open(bodyfiles[fi]).convert("RGB").resize((res,res)),np.uint8)
    from scipy.ndimage import gaussian_filter, binary_erosion
    m0=binary_erosion(head_img.sum(2)>20, iterations=3)          # drop dark crop fringe
    alpha=np.clip(gaussian_filter(m0.astype(np.float32), sigma=2.5),0,1)[...,None]  # feather head->body
    comp=(head_img*alpha+body*(1-alpha)).astype(np.uint8)
    Image.fromarray(comp).save(f"{a.out}/{fi:04d}.png")
    Image.fromarray(head_img).save(f"{a.out}/head_{fi:04d}.png") if a.one else None
print("IDOL_SHARP_OK",a.out,"F",F)
