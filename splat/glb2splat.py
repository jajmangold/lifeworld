#!/usr/bin/env python3
"""Sample a textured GLB mesh into colored gaussians -> SHARP-format .ply (x,y,z,f_dc,opacity,scale,
rot) so splat_charfull can load it. Complete mesh -> arms have geometry everywhere (no smear).
  python glb2splat.py in.glb out.ply [N=400000]"""
import sys, numpy as np, trimesh
from plyfile import PlyData, PlyElement
glb,out=sys.argv[1],sys.argv[2]; N=int(sys.argv[3]) if len(sys.argv)>3 else 400000
s=trimesh.load(glb,process=False)
mesh=s.to_geometry() if isinstance(s,trimesh.Scene) else s
cv=mesh.visual.to_color().vertex_colors[:,:3].astype(np.float32)/255.0
pts,fidx=trimesh.sample.sample_surface(mesh,N)
col=cv[mesh.faces[fidx]].mean(1)
pts=pts.astype(np.float32)
diag=float(np.linalg.norm(pts.max(0)-pts.min(0))); sc=diag/np.sqrt(N)*1.4   # surface spacing
fdc=((col-0.5)/0.2820948).astype(np.float32)
op=np.full(len(pts),6.0,np.float32)                       # sigmoid(6)~0.998
slog=np.full((len(pts),3),np.log(sc),np.float32)
rot=np.tile([1,0,0,0],(len(pts),1)).astype(np.float32)
arr=np.zeros(len(pts),dtype=[('x','f4'),('y','f4'),('z','f4'),('f_dc_0','f4'),('f_dc_1','f4'),('f_dc_2','f4'),
    ('opacity','f4'),('scale_0','f4'),('scale_1','f4'),('scale_2','f4'),('rot_0','f4'),('rot_1','f4'),('rot_2','f4'),('rot_3','f4')])
arr['x'],arr['y'],arr['z']=pts[:,0],pts[:,1],pts[:,2]
arr['f_dc_0'],arr['f_dc_1'],arr['f_dc_2']=fdc[:,0],fdc[:,1],fdc[:,2]
arr['opacity']=op
arr['scale_0'],arr['scale_1'],arr['scale_2']=slog[:,0],slog[:,1],slog[:,2]
arr['rot_0'],arr['rot_1'],arr['rot_2'],arr['rot_3']=rot[:,0],rot[:,1],rot[:,2],rot[:,3]
PlyData([PlyElement.describe(arr,'vertex')]).write(out)
print("GLB2SPLAT_OK",out,len(pts),"pts, spacing %.4f"%sc,"bbox",np.round(pts.min(0),2),np.round(pts.max(0),2))
