#!/usr/bin/env python3
"""Run INSIDE the MDM container (cwd=/mdm). Extract per-joint LOCAL rotations + root from MDM's
raw 263-dim output -> npz for the SMPL-X bridge.  python mdm_extract_local.py <raw263.npy> <out.npz>"""
import numpy as np, torch, sys
from data_loaders.humanml.utils.paramUtil import t2m_kinematic_chain  # noqa
from data_loaders.humanml.scripts.motion_process import recover_root_rot_pos
from data_loaders.humanml.common.quaternion import quaternion_to_cont6d, cont6d_to_matrix

RAW, OUT = sys.argv[1], sys.argv[2]
J = 22
data = torch.from_numpy(np.load(RAW)[0, 0]).float()        # [F,263]
F = data.shape[0]
r_rot_quat, r_pos = recover_root_rot_pos(data)             # [F,4],[F,3]
r_rot_cont6d = quaternion_to_cont6d(r_rot_quat)            # [F,6]
start = 1 + 2 + 1 + (J - 1) * 3; end = start + (J - 1) * 6
joint_cont6d = data[..., start:end].view(F, J - 1, 6)      # [F,21,6] LOCAL rel-parent
cont6d = torch.cat([r_rot_cont6d.unsqueeze(1), joint_cont6d], dim=1)   # [F,22,6]
mats = cont6d_to_matrix(cont6d.reshape(-1, 6)).reshape(F, J, 3, 3)     # [F,22,3,3] LOCAL
np.savez(OUT, mats=mats.numpy(), root_pos=r_pos.numpy())
print("LOCAL_OK", OUT, "F", F)
