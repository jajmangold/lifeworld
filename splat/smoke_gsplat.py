import torch, numpy as np
from gsplat import rasterization
dev = "cuda"
# 9 colored gaussians in a 3x3 grid at z=3
xs = np.array([[i, j, 3.0] for i in (-1, 0, 1) for j in (-1, 0, 1)], np.float32)
N = len(xs)
means = torch.tensor(xs, device=dev)
quats = torch.tensor([[1, 0, 0, 0]] * N, dtype=torch.float32, device=dev)
scales = torch.full((N, 3), 0.25, device=dev)
opac = torch.ones(N, device=dev)
cols = torch.tensor(np.random.RandomState(0).rand(N, 3), dtype=torch.float32, device=dev)
viewmat = torch.eye(4, device=dev)[None]                     # camera at origin looking +z
K = torch.tensor([[[200, 0, 128], [0, 200, 128], [0, 0, 1]]], dtype=torch.float32, device=dev)
out, alpha, meta = rasterization(means, quats, scales, opac, cols, viewmat, K, 256, 256)
img = (out[0].clamp(0, 1) * 255).byte().cpu().numpy()
nonzero = int((img.sum(-1) > 0).sum())
print(f"SMOKE_OK render {img.shape} nonzero_px={nonzero} arch={torch.cuda.get_arch_list()}")
np.save("/o/smoke.npy", img)
