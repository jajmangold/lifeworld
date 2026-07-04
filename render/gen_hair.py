"""Generate a 3D strand hair groom from a text prompt via the resident HAAR service (container `hairgen`).
Promoted from R&D: HAAR is now a first-class capability of the character pipeline. Grooms are cached in
assets/grooms/<name>.ply and reused (generation ~30s; don't regen per render). The groom is grafted onto a
character by render_character/graft_hair (NEWS_HAIR_GROOM).

  python3 render/gen_hair.py --prompt "a woman with a long straight center-parted hairstyle" --name long_straight
  python3 render/gen_hair.py --name bob_short         # just report the cached path (no regen)

Container: `hairgen` mounts /mnt/24tb/rnd/haar-spike -> /work (weights + HAAR env live there, off nvme).
Output .ply is copied into the studio's assets/grooms/ library. Pass --stage-rtx0 to also push it to the
render box for immediate use.
"""
import argparse, os, subprocess, sys

BOT = "/srv/nvme-data/containers/live/studio"
GROOMS = os.path.join(BOT, "assets", "grooms")
RTX0 = "josh@rtx0:/mnt/datadisk/containers/sampl"

def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, **kw)

def generate(prompt, name, n_samples=1, seed=42):
    os.makedirs(GROOMS, exist_ok=True)
    exp = f"gen_{name}"
    # run HAAR inference inside the resident container
    inner = (
        "source activate haar && cd /work/HAAR && "
        "python infer.py --config ./configs/infer.json "
        "--ckpt_path ./pretrained_models/haar_prior/haar_diffusion.pth "
        f"--exp_name {exp} --hairstyle_description \"{prompt}\" "
        f"--n_samples {n_samples} --seed {seed} --save_upsampled_hairstyle "
        "--save_path ./inference_results"
    )
    print(f"[gen_hair] generating '{name}': {prompt}", flush=True)
    r = sh(f"docker exec hairgen bash -lc '{inner}'", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if "For results see folder" not in r.stdout and r.returncode != 0:
        print(r.stdout[-1500:]); sys.exit("HAAR inference failed")
    # the groom lands on the host at the container's bind mount
    src = f"/mnt/24tb/rnd/haar-spike/HAAR/inference_results/{exp}/upsampled_hairstyle/pc_0.ply"
    if not os.path.exists(src):
        sys.exit(f"expected groom not found: {src}")
    dst = os.path.join(GROOMS, f"{name}.ply")
    sh(f"cp {src} {dst}")
    print(f"GROOM_OK -> {dst} ({os.path.getsize(dst)//1024//1024}MB)")
    return dst

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="")
    ap.add_argument("--name", required=True)
    ap.add_argument("--n-samples", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--stage-rtx0", action="store_true")
    a = ap.parse_args()
    cached = os.path.join(GROOMS, f"{a.name}.ply")
    if a.prompt:
        cached = generate(a.prompt, a.name, a.n_samples, a.seed)
    elif os.path.exists(cached):
        print(f"GROOM_CACHED -> {cached}")
    else:
        sys.exit(f"no cached groom '{a.name}' and no --prompt given")
    if a.stage_rtx0:
        sh(f"scp -q {cached} {RTX0}/groom_{a.name}.ply")
        print(f"staged -> {RTX0}/groom_{a.name}.ply")
