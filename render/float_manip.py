#!/usr/bin/env python3
"""Manipulate FLOAT's LIA motion latent directly.
  sweep: push each orthonormal motion axis Q[:,k] by +/- mags from a still ref -> labeled grid
         (to discover which axis = head pose / mouth / eyes / brows).
    python float_manip.py sweep <ref.png> <out_grid.png> [naxes] [maxmag]
  gen:   generate talking video with motion gain + a static bias along chosen axes.
    python float_manip.py gen <ref.png> <aud.wav> <out.mp4> --gain G --bias "k:mag,k:mag" [--nfe N]
"""
import sys, os
sys.argv_backup = sys.argv
mode = sys.argv[1]
sys.argv = ["float_manip", "--ckpt_path", os.environ.get("FLOAT_CKPT", "./checkpoints/float.pth")]
import torch, numpy as np
from PIL import Image
from generate import InferenceAgent, InferenceOptions

opt = InferenceOptions().parse(); opt.rank, opt.ngpus = 0, 1
agent = InferenceAgent(opt); G = agent.G; dp = agent.data_processor
rank = opt.rank


def load_img(p):
    s = dp.default_img_loader(p)
    s = dp.process_img(s)
    return dp.transform(image=s)["image"].unsqueeze(0).to(rank)


@torch.no_grad()
def encode(p):
    s = load_img(p)
    s_r, s_r_lambda, s_r_feats = G.encode_image_into_latent(s)
    return s_r, s_r_lambda, s_r_feats


@torch.no_grad()
def decode(s_r, s_r_feats, disp):
    img, _ = G.motion_autoencoder.dec(s_r + disp, alpha=None, feats=s_r_feats)
    return ((img.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8)[0].permute(1, 2, 0).cpu().numpy()


if mode == "sweep":
    ref, out = sys.argv_backup[2], sys.argv_backup[3]
    naxes = int(sys.argv_backup[4]) if len(sys.argv_backup) > 4 else 12
    maxmag = float(sys.argv_backup[5]) if len(sys.argv_backup) > 5 else 4.0
    s_r, s_r_lambda, s_r_feats = encode(ref)
    Q = G.motion_autoencoder.dec.direction(None)            # [512, motion_dim]
    md = Q.shape[1]
    print(f"motion_dim={md}  s_r {tuple(s_r.shape)}  lambda |abs| mean={s_r_lambda.abs().mean():.3f} max={s_r_lambda.abs().max():.3f}", flush=True)
    naxes = min(naxes, md)
    mags = [-maxmag, -maxmag / 2, 0.0, maxmag / 2, maxmag]
    cell = 160
    grid = Image.new("RGB", (cell * len(mags), cell * naxes), (20, 20, 20))
    for k in range(naxes):
        fp = fm = None
        for j, m in enumerate(mags):
            fr = decode(s_r, s_r_feats, m * Q[:, k])
            if j == 0: fm = fr.astype(np.int16)
            if j == len(mags) - 1: fp = fr.astype(np.int16)
            grid.paste(Image.fromarray(fr).resize((cell, cell)), (j * cell, k * cell))
        d = np.abs(fp - fm).astype(np.float32); H, W = d.shape[:2]
        top, mid, bot = d[:H//3].mean(), d[H//3:2*H//3].mean(), d[2*H//3:].mean()
        L, R = d[:, :W//2].mean(), d[:, W//2:].mean()
        # pose/yaw: brightness centroid shifts horizontally between -m and +m
        gp = fp.mean(2); gm = fm.mean(2); xs = np.arange(W)
        cx_p = (gp.sum(0) * xs).sum() / (gp.sum() + 1e-6)
        cx_m = (gm.sum(0) * xs).sum() / (gm.sum() + 1e-6)
        print(f"axis {k:2d}: total={d.mean():5.1f}  eyes(top)={top:5.1f} mid={mid:5.1f} mouth(bot)={bot:5.1f}  L={L:5.1f} R={R:5.1f}  yawshift={cx_p-cx_m:+5.1f}px", flush=True)
    grid.save(out)
    print(f"SWEEP_OK {out} axes={naxes} mags={mags}", flush=True)

elif mode == "gen":
    import argparse, subprocess, tempfile, torchvision
    ref, aud, out = sys.argv_backup[2], sys.argv_backup[3], sys.argv_backup[4]
    ap = argparse.ArgumentParser()
    ap.add_argument("--gain", type=float, default=1.0)
    ap.add_argument("--bias", type=str, default="")          # "k:mag,k:mag"
    ap.add_argument("--nfe", type=int, default=10)
    ap.add_argument("--a_cfg_scale", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=25)
    a = ap.parse_args(sys.argv_backup[5:])
    s_r, r_s_lambda, s_r_feats = G.encode_image_into_latent(load_img(ref))
    data = dp.preprocess(ref, aud, no_crop=False)
    data["r_s"] = G.motion_autoencoder.dec.direction(r_s_lambda)
    sample = G.sample(data, a_cfg_scale=a.a_cfg_scale, r_cfg_scale=1.0, e_cfg_scale=1.0,
                      emo=None, nfe=a.nfe, seed=a.seed)                  # [B,T,512]
    Q = G.motion_autoencoder.dec.direction(None)
    bias = torch.zeros(512, device=rank)
    for tok in [t for t in a.bias.split(",") if t.strip()]:
        k, m = tok.split(":"); bias = bias + float(m) * Q[:, int(k)]
    T = sample.shape[1]; fps = opt.fps
    frames = []
    for t in range(T):
        disp = a.gain * sample[:, t] + bias
        img, _ = G.motion_autoencoder.dec(s_r + disp, alpha=None, feats=s_r_feats)
        frames.append(((img.clamp(-1, 1) + 1) / 2 * 255).to(torch.uint8)[0].permute(1, 2, 0).cpu())
    vid = torch.stack(frames)
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tv:
        tmp = tv.name
    torchvision.io.write_video(tmp, vid, fps=fps)
    subprocess.call(f"ffmpeg -y -loglevel error -i {tmp} -i {aud} -c:v copy -c:a aac {out}", shell=True)
    os.remove(tmp)
    print(f"GEN_OK {out} gain={a.gain} bias='{a.bias}'", flush=True)
