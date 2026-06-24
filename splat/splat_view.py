#!/usr/bin/env python3
"""Load a 3DGS .ply (SHARP/inria format) and render with gsplat. SHARP = OpenCV convention, scene
center ~ (0,0,+z).  python splat_view.py --ply x.ply --out o_ --res 512 [--orbit]"""
import sys, argparse, numpy as np, torch
from plyfile import PlyData
from gsplat import rasterization
from PIL import Image
dev="cuda"
ap=argparse.ArgumentParser(); ap.add_argument("--ply",required=True); ap.add_argument("--out",required=True)
ap.add_argument("--res",type=int,default=512); ap.add_argument("--orbit",action="store_true")
a=ap.parse_args()
p=PlyData.read(a.ply)["vertex"]; g=lambda k: np.asarray(p[k],np.float32)
means=np.stack([g("x"),g("y"),g("z")],1)
scales=np.exp(np.stack([g("scale_0"),g("scale_1"),g("scale_2")],1))
quats=np.stack([g("rot_0"),g("rot_1"),g("rot_2"),g("rot_3")],1); quats/=np.linalg.norm(quats,axis=1,keepdims=True)
opac=1/(1+np.exp(-g("opacity")))
rgb=np.clip(0.2820948*np.stack([g("f_dc_0"),g("f_dc_1"),g("f_dc_2")],1)+0.5,0,1)
N=len(means); print("loaded",N,"gaussians; centroid",np.round(means.mean(0),2),"z-range",round(float(means[:,2].min()),2),round(float(means[:,2].max()),2))
M=[torch.tensor(x,device=dev) for x in (means,quats,scales,opac,rgb)]
res=a.res; fx=0.83*res; K=torch.tensor([[[fx,0,res/2],[0,fx,res/2],[0,0,1]]],dtype=torch.float32,device=dev)
c=means.mean(0)                                              # orbit target
def vm_at(ang):
    r=np.linalg.norm([c[0],c[2]])                            # radius in x-z plane from origin
    eye=np.array([c[0]+r*np.sin(ang)-c[0]*0, c[1]*0, 0],np.float32) if False else np.array([r*np.sin(ang),0,c[2]-r*np.cos(ang)+0],np.float32)
    # look at centroid c, OpenCV (z fwd, y down)
    z=c-eye; z/=np.linalg.norm(z); x=np.cross(z,np.array([0,-1,0],np.float32)); x/=np.linalg.norm(x); y=np.cross(z,x)
    vm=np.eye(4,dtype=np.float32); vm[:3,:3]=np.stack([x,y,z],0); vm[:3,3]=-np.stack([x,y,z],0)@eye
    return torch.tensor(vm,device=dev)[None]
angs=np.radians(np.linspace(-30,30,12)) if a.orbit else [0.0]
for i,ang in enumerate(angs):
    out,_,_=rasterization(M[0],M[1],M[2],M[3],M[4],vm_at(ang),K,res,res)
    Image.fromarray((out[0].clamp(0,1)*255).byte().cpu().numpy()).save(f"{a.out}{i:03d}.png")
print("VIEW_OK",len(angs))
