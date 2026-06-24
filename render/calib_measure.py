#!/usr/bin/env python3
"""Fast (no-render) head-track calibration: pose the SMPL-X head from a head_R sequence with given
signs/gain, project its face landmarks, measure roll/yaw, and correlate against the FLOAT face's
fa_pose -> objective signs. Eye-line roll resolves the camera/world mirror.
  python calib_measure.py --arkit head_R.json --fapose float_fapose.json [--signs 1,1,1 --gain 4]"""
import sys, argparse, json, numpy as np, torch, smplx
ap = argparse.ArgumentParser()
ap.add_argument("--arkit", required=True); ap.add_argument("--fapose", required=True)
ap.add_argument("--signs", default="1,1,1"); ap.add_argument("--gain", type=float, default=4.0)
a = ap.parse_args()
sx, sy, sz = [float(x) for x in a.signs.split(",")]
HR = np.asarray(json.load(open(a.arkit))["head_R"], np.float32).reshape(-1, 3, 3); F = len(HR)

def mat2aa(R):
    tr=np.clip((R[:,0,0]+R[:,1,1]+R[:,2,2]-1)/2,-1,1); ang=np.arccos(tr)
    v=np.stack([R[:,2,1]-R[:,1,2],R[:,0,2]-R[:,2,0],R[:,1,0]-R[:,0,1]],1); n=np.linalg.norm(v,axis=1,keepdims=True); n[n<1e-8]=1
    return (v/n)*ang[:,None]
pitch=sx*a.gain*np.arctan2(-HR[:,2,1],HR[:,2,2]); yaw=sy*a.gain*np.arctan2(HR[:,2,0],np.hypot(HR[:,2,1],HR[:,2,2])); roll=sz*a.gain*np.arctan2(-HR[:,1,0],HR[:,0,0])
def _Rx(t):o=np.zeros((F,3,3),np.float32);c,s=np.cos(t),np.sin(t);o[:,0,0]=1;o[:,1,1]=c;o[:,1,2]=-s;o[:,2,1]=s;o[:,2,2]=c;return o
def _Ry(t):o=np.zeros((F,3,3),np.float32);c,s=np.cos(t),np.sin(t);o[:,1,1]=1;o[:,0,0]=c;o[:,0,2]=s;o[:,2,0]=-s;o[:,2,2]=c;return o
def _Rz(t):o=np.zeros((F,3,3),np.float32);c,s=np.cos(t),np.sin(t);o[:,2,2]=1;o[:,0,0]=c;o[:,0,1]=-s;o[:,1,0]=s;o[:,1,1]=c;return o
aa=mat2aa(_Ry(yaw)@_Rx(pitch)@_Rz(roll))
bp=np.zeros((F,21,3),np.float32); bp[:,11]=aa*0.6; bp[:,14]=aa*0.4
go=np.zeros((F,3),np.float32); go[:,1]=np.radians(180.0)
m=smplx.create("/work/models",model_type="smplx",gender="male",num_betas=10,use_pca=False,flat_hand_mean=True,num_expression_coeffs=100,batch_size=F)
with torch.no_grad():
    J=m(global_orient=torch.from_numpy(go),body_pose=torch.from_numpy(bp.reshape(F,-1))).joints.numpy().astype(np.float32)
# camera identical to calib_head
allv=J.reshape(-1,3); headc=float(J[:,15,1].mean())  # head joint Y ~ frame; use head joint
YFOV=0.6; dist=0.34/np.tan(0.5*YFOV); W,H=288,360; FY=1/np.tan(YFOV/2); ASP=W/H
campos=np.array([0,headc,dist],np.float32)
def proj(P):  # (F,3)->(F,2)
    pc=P-campos; z=-pc[:,2]; u=(pc[:,0]*FY/ASP/z+1)/2*W; v=(1-pc[:,1]*FY/z)/2*H; return np.stack([u,v],1)
NOSE,EYL,EYR=89,85,76
nz=proj(J[:,NOSE]); el=proj(J[:,EYL]); er=proj(J[:,EYR])
mesh_roll=np.degrees(np.arctan2(er[:,1]-el[:,1], er[:,0]-el[:,0]))
eyemid=(el+er)/2; eyew=np.linalg.norm(er-el,axis=1)+1e-6
mesh_yaw=np.degrees(np.arctan2(nz[:,0]-eyemid[:,0], eyew))   # nose horizontal offset ~ yaw proxy
fe=np.array(json.load(open(a.fapose))["euler"]); n=min(F,len(fe))
fl_pitch,fl_yaw,fl_roll=fe[:n,0],fe[:n,1],fe[:n,2]
def corr(x,y):
    x=x[:n]-np.mean(x[:n]); y=y[:n]-np.mean(y[:n])
    return float(np.corrcoef(x,y)[0,1]) if x.std()>1e-6 and y.std()>1e-6 else 0.0
print("signs",a.signs,"gain",a.gain)
print("  roll  corr(mesh,float)=%+.2f  -> roll sign %s"%(corr(mesh_roll,fl_roll), "OK" if corr(mesh_roll,fl_roll)>0 else "FLIP sz"))
print("  yaw   corr(mesh,float)=%+.2f  -> yaw  sign %s"%(corr(mesh_yaw,fl_yaw),  "OK" if corr(mesh_yaw,fl_yaw)>0 else "FLIP sy"))
