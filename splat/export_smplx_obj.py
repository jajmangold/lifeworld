#!/usr/bin/env python3
"""Export greenman's SMPL-X body as OBJ with the SMPL-X UV (for StableProjectorz texturing).
A-pose (limbs separated for clean projection), his SAM3D betas."""
import sys,json,numpy as np,torch,smplx
betas=np.zeros((1,10),np.float32)
betas[0]=np.array(json.load(open("/o/sam3d/greenman_skel.json"))["shape_params"][:10],np.float32)
m=smplx.create("/work/models",model_type="smplx",gender="male",num_betas=10,use_pca=False,flat_hand_mean=True,num_expression_coeffs=100,batch_size=1)
bp=np.zeros((1,21,3),np.float32); bp[0,15]=[0,0,-1.0]; bp[0,16]=[0,0,1.0]   # gentle A-pose
with torch.no_grad():
    V=m(global_orient=torch.zeros(1,3),betas=torch.from_numpy(betas),body_pose=torch.from_numpy(bp.reshape(1,-1))).vertices.numpy()[0]
F=m.faces.astype(np.int64)
uv=np.load("/work/assets/smplx_uv_2023.npz")["uv_coordinates"].astype(np.float32)   # per-loop (F*3,2)
out="/o/greenman_smplx_body.obj"
with open(out,"w") as f:
    f.write("# greenman SMPL-X body for StableProjectorz\n")
    for v in V: f.write("v %.6f %.6f %.6f\n"%(v[0],v[1],v[2]))
    for t in uv: f.write("vt %.6f %.6f\n"%(t[0],t[1]))
    for fi,tri in enumerate(F):
        a,b,c=tri+1; ua,ub,uc=fi*3+1,fi*3+2,fi*3+3
        f.write("f %d/%d %d/%d %d/%d\n"%(a,ua,b,ub,c,uc))
print("OBJ_OK",out,"verts",len(V),"faces",len(F),"uv",len(uv))
