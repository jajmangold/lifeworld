#!/usr/bin/env python3
"""Bake greenman's real frontal face onto the SMPL-X head as per-face splat COLORS (static identity,
splat-native, stable, turns in 3D). Aligns the photo to the mesh eye/nose, warps to render space,
samples front-facing head triangles (skips green bg). -> facecolor.npy (Ftri,3).
  python splat_texbake.py --photo face0.png --lmk facelmk.json --out facecolor.npy [--res 512]"""
import os, sys, json, argparse, numpy as np, torch, smplx
from PIL import Image
sys.path.insert(0, "/lw/splat"); from splat_build import face_color
ap = argparse.ArgumentParser()
ap.add_argument("--photo", required=True); ap.add_argument("--lmk", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--res", type=int, default=512)
a = ap.parse_args(); res = a.res

def look_at(eye, tgt, up=(0,1,0)):
    eye=np.array(eye,np.float32);tgt=np.array(tgt,np.float32);up=np.array(up,np.float32)
    z=tgt-eye;z/=np.linalg.norm(z);x=np.cross(z,up);x/=np.linalg.norm(x);y=np.cross(z,x)
    vm=np.eye(4,dtype=np.float32);vm[:3,:3]=np.stack([x,y,z],0);vm[:3,3]=-np.stack([x,y,z],0)@eye;return vm
def affine(src,dst):                                   # 2x3 mapping src->dst (3 pts exact)
    S=np.hstack([src,np.ones((3,1))]); return np.linalg.solve(S,dst).T

uv=np.load("/work/assets/smplx_uv_2023.npz")["uv_coordinates"].astype(np.float32)
tex=np.asarray(Image.open("/work/assets/smplx_texture_m_alb.png").convert("RGB"),np.float32)/255.0
model=smplx.create("/work/models",model_type="smplx",gender="male",num_betas=10,use_pca=False,flat_hand_mean=True,num_expression_coeffs=100,batch_size=1)
faces=model.faces.astype(np.int64)
go=np.zeros((1,3),np.float32); go[:,1]=0.0
with torch.no_grad():
    o=model(global_orient=torch.from_numpy(go),betas=torch.zeros(1,10))
V=o.vertices.numpy()[0].astype(np.float32); J=o.joints.numpy()[0].astype(np.float32)
top=float(V[:,1].max()); headc=top-0.13; dist=0.42
eye=np.array([0,headc,dist],np.float32); vm=look_at(eye,[0,headc,0])
fx=res*1.2; K=np.array([[fx,0,res/2],[0,fx,res/2],[0,0,1]],np.float32)
def project(P):                                        # world (N,3)->(N,2) OpenCV
    pc=(vm@np.c_[P,np.ones(len(P))].T).T[:,:3]; z=np.clip(pc[:,2],1e-6,None)
    return np.stack([fx*pc[:,0]/z+res/2, fx*pc[:,1]/z+res/2],1), pc[:,2]
# mesh landmarks (eyeL85,eyeR76,nose89) -> render px
mlmk,_=project(J[[85,76,89]])
gl=json.load(open(a.lmk))["pts"]; glmk=np.array([gl["eyeL"],gl["eyeR"],gl["nose"]],np.float32)
# order eyes by image-x in both
if mlmk[0,0]>mlmk[1,0]: mlmk[[0,1]]=mlmk[[1,0]]
if glmk[0,0]>glmk[1,0]: glmk[[0,1]]=glmk[[1,0]]
Ainv=affine(mlmk,glmk)                                 # render-px -> photo-px (for PIL warp)
photo=Image.open(a.photo).convert("RGB")
warped=np.asarray(photo.transform((res,res),Image.AFFINE,Ainv.flatten().tolist(),resample=Image.BILINEAR),np.float32)/255.0
# per-face: bake front-facing head triangles whose warped sample is non-green
tri=V[faces]; cen=tri.mean(1); n=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]); n/=np.clip(np.linalg.norm(n,axis=1,keepdims=True),1e-9,None)
front=(n*(cen-eye)).sum(1)<0
cen2d,cz=project(cen); cx=np.clip(cen2d[:,0].astype(int),0,res-1); cy=np.clip(cen2d[:,1].astype(int),0,res-1)
samp=warped[cy,cx]; greenness=samp[:,1]-np.maximum(samp[:,0],samp[:,2])
headregion=cen[:,1]>(top-0.26)                         # only the head, not body
bake=front&headregion&(greenness<0.06)&(cz>0)
cols=face_color(faces,uv,tex); cols[bake]=samp[bake]
np.save(a.out,cols.astype(np.float32)); print("TEXBAKE_OK",a.out,"baked",int(bake.sum()),"of",len(faces))
