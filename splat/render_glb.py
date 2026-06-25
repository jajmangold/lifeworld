import sys, numpy as np, trimesh, pyrender
from PIL import Image
glb=sys.argv[1]; out=sys.argv[2]
s=trimesh.load(glb); 
mesh=s.dump(concatenate=True) if isinstance(s,trimesh.Scene) else s
mesh.vertices-=mesh.vertices.mean(0); mesh.vertices/=np.abs(mesh.vertices).max()
imgs=[]
for ang in [0,90,180,270]:
    sc=pyrender.Scene(bg_color=[0,0,0,0],ambient_light=[0.6,0.6,0.6])
    R=trimesh.transformations.rotation_matrix(np.radians(ang),[0,1,0])
    m2=mesh.copy(); m2.apply_transform(R)
    sc.add(pyrender.Mesh.from_trimesh(m2,smooth=False))
    cam=pyrender.PerspectiveCamera(yfov=0.9); cp=np.eye(4); cp[2,3]=2.4; sc.add(cam,pose=cp)
    sc.add(pyrender.DirectionalLight(intensity=3.0),pose=cp)
    r=pyrender.OffscreenRenderer(300,360); col,_=r.render(sc); r.delete(); imgs.append(col)
Image.fromarray(np.concatenate(imgs,1)).save(out); print("GLB_RENDER_OK",out)
