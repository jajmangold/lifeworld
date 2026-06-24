"""Surface-aligned Gaussian splats from an SMPL-X mesh: one flat disk per triangle, oriented to the
face normal, sized to the triangle, colored per-face from the texture. Sharp (tiles the surface)
vs blobby per-vertex splats. Returns means(F,3), quats(F,4 wxyz), scales(F,3), colors(F,3)."""
import numpy as np

def _quat_from_R(R):                                   # R (N,3,3) -> wxyz
    w = np.sqrt(np.clip(1 + R[:,0,0] + R[:,1,1] + R[:,2,2], 0, None)) / 2
    w = np.clip(w, 1e-8, None)
    x = (R[:,2,1] - R[:,1,2]) / (4*w); y = (R[:,0,2] - R[:,2,0]) / (4*w); z = (R[:,1,0] - R[:,0,1]) / (4*w)
    q = np.stack([w, x, y, z], 1); return (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float32)

def face_color(faces, uv, tex):                        # per-face mean texture color
    Ht, Wt = tex.shape[:2]; loops = uv[:len(faces)*3].reshape(-1,3,2)
    px = np.clip((loops[:,:,0]*Wt).astype(int), 0, Wt-1); py = np.clip(((1-loops[:,:,1])*Ht).astype(int), 0, Ht-1)
    return tex[py, px].mean(1).astype(np.float32)      # (Ftri,3)

def build_face_splats(V, faces, uv, tex, thin=0.18):
    """V (Nv,3) one frame. Returns surface splats for that frame."""
    tri = V[faces]                                     # (Ftri,3,3)
    cen = tri.mean(1)                                  # centroids = means
    e1 = tri[:,1]-tri[:,0]; e2 = tri[:,2]-tri[:,0]
    n = np.cross(e1, e2); area = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.clip(area, 1e-9, None)                  # face normals (unit)
    x = e1 / np.clip(np.linalg.norm(e1, axis=1, keepdims=True), 1e-9, None)
    y = np.cross(n, x)
    R = np.stack([x, y, n], 2)                         # cols x,y,n -> local z = normal
    quats = _quat_from_R(R)
    rad = np.sqrt(np.clip(area[:,0], 1e-9, None)) * 0.9     # in-plane size ~ triangle scale
    scales = np.stack([rad, rad, rad*thin], 1).astype(np.float32)  # flat disk along normal
    cols = face_color(faces, uv, tex)
    return cen.astype(np.float32), quats, scales, cols
