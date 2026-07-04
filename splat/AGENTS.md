# splat/ — AGENTS.md
> Experimental 3D-avatar research — single-image Gaussian-splatting reconstruction, SMPL-X rigging, and video face restoration. Mostly vendored upstream repos + local glue. NONE of it is wired into production.

## What it does
Parked R&D toward a fully-3D talking-head: reconstruct a photoreal head/body as 3D Gaussian splats from a single image (SHARP, FastAvatar, IDOL), texture a SMPL-X body (SMPLitex), and drive it with our Kimodo (body) + FLAME/ARKit (face) params rendered via `gsplat`. The idea was a seamless 3D character that turns/talks without the 2D warp jitter — the `jitter_metric.py` yardstick it had to beat. As of now this is all EXPERIMENTAL: the production anchor/news pipeline (`make_anchor.sh`, `make_character.sh`) does NOT reference anything here. Treat splat/ as a research sandbox, not a shipping path.

## Subprojects (vendored)
- `ml-sharp` — Apple SHARP: single-image → 3D Gaussian scene, real-time render. The head/body source most local glue builds on.
- `IDOL` — "Instant Photorealistic 3D Human from a Single Image": feed-forward full-body human reconstruction.
- `FastAvatar` — real-time 3DGS face reconstruction from one unconstrained-pose photo.
- `SMPLitex` — generative SMPL-X texture estimation from a single image (uses Automatic1111/SD).
- `KEEP` — Kalman-inspired video face super-resolution / restoration (ECCV 2024).

## Local glue (this dir, `splat_*.py` etc.)
Experiment scripts binding the vendored outputs together — e.g. `splat_charfull.py`, `splat_idol_sharp.py`, `splat_anim.py`, `splat_assemble.py` (SHARP head + SMPL-X/IDOL body → animated splat scene), `glb2splat.py`, `splat_view.py`, `splat_texbake.py`, `jitter_metric.py`. Each has a docstring with its own invocation; `Dockerfile.splat` (on `kdtalker:1.1` + gsplat/smplx) is the env.

## Run
No production entrypoint. Individual scripts are runnable ad-hoc per their header docstrings (paths to `--ply`/`--kimodo`/`--arkit`), inside the `Dockerfile.splat` env. Assume nothing here runs unattended.

## Gotchas & rules
- The subdirs are VENDORED upstream repos — treat as third-party, DO NOT rewrite; each has its own env, deps, licenses (MIT/Apple/S-Lab). Confine changes to the local `splat_*.py` glue.
- gsplat/pytorch3d/diff-gaussian-rasterization need CUDA builds under the sm_70 constraint — see studio/AGENTS.md.
- Before investing effort here, confirm the task actually targets the 3D path — most character work lives elsewhere.
- Global rules inherited from studio/AGENTS.md.

## Related
- `../viverse_avatar/`, `../humgen/` — other (production-adjacent) avatar approaches.
