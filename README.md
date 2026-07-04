<!-- ⚠️ ASPIRATIONAL / EARLY-VISION DOC (SMPL-X-centric "lifeworld"). This is NOT the current state.
     For what actually exists and how to work here, read studio/AGENTS.md (and the per-area AGENTS.md). -->

# lifeworld

Autonomous SMPL-X **humans that live lives** in a simulated world, remember
through a graph, and whose selected moments are rendered cinematically.

Agents perceive a real 3D world through virtual sensors, decide with an LLM
"neocortex", remember in a graph database, speak/hear, and act through embodied
SMPL-X humanoids. Selected story beats are re-rendered in film-quality Blender.

> Status: **core vertical working** (validated 2026-06-21 on the V100 fleet). The full
> loop runs end to end — see "What works" below.

## What works (validated, all on sm_70 V100, headless)

| Milestone | What runs | Artifact |
|-----------|-----------|----------|
| **M1** embodiment | `world/habitat_walk.py` — humanoid walks ReplicaCAD; RGB+depth+semantic sensors | `output/habitat_walk.mp4`, `M1_*.png` |
| **M2** mind loop | `world/habitat_mind.py` — perceive → DeepSeek decides → navigate → log Neo4j | `output/mind.mp4` |
| **M3 text** society | `mind/society.py` — personas + seeded conflict → DeepSeek storyline + FEELS graph | Neo4j storyline |
| **M3 embodied** | `world/habitat_social.py` — two humanoids co-present + logged conversation | `output/social_twoshot.png` |
| **Capstone** | `render/make_talk.sh "<line>"` — a Neo4j storyline line → lip-synced SMPL-X clip | `output/beat_mara_confront.mp4` |
| **Render core** | `render/render_smplx.py` — Blender-free GPU SMPL-X (pyrender/EGL) | `output/walk.mp4` |

Run the embodied/mind/social pieces in `lifeworld-habitat` with `--network host` +
`-e DEEPSEEK_API_KEY -e NEO4J_PASSWORD` and `/mnt/24tb/habitat` mounted. See `docs/DECISIONS.md`.

## Why this shape (hardware reality)

This runs on an **sm_70 fleet** (Tesla V100 / CMP 100-210, 16 GB, no RT cores).
That single fact drives the whole design:

- **NVIDIA Isaac Sim / Genie Sim / AgiBot World are out.** Isaac requires RTX RT
  cores; the V100 is explicitly unsupported. AgiBot's GO-1 is a real-robot
  *manipulation* policy anyway — orthogonal to living narrative lives.
- **Habitat 3.0 is in.** Standard-CUDA rasterization, runs headless on V100, and
  its humanoids are **SMPL-X-native** — the exact representation our working
  Blender renderer already consumes. Motion flows sim → render with **zero
  retargeting**, killing the cross-skeleton brittleness that sank the prior path.

## Architecture (layers)

| Layer | Tech | Role |
|-------|------|------|
| **Mind** (neocortex) | DeepSeek V4 Flash + local Qwen VLM (visual cortex) | perceive → remember → reflect → plan → act → speak |
| **World / embodiment** | **Habitat 3.0** (V100, headless) | SMPL-X humanoids, navigation, virtual sensors (RGB/depth/semantic/audio), real indoor scenes |
| **Memory / society** | **Neo4j** | persistent life-record, social graph, storylines, events |
| **Voice / ears** | Higgs-TTS (`:8055`) + Granite STT (amd0/amd1) | speech out / in |
| **Cinematic render** | `sampl` SMPL-X → Blender Cycles | film-quality render of *selected* moments; shares SMPL-X pose with Habitat |

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for detail and
[`docs/DECISIONS.md`](docs/DECISIONS.md) for the rationale log (mirrored to issues).

## Principle: stay in SMPL-X space

Body shape, pose, hands, and face are **always SMPL-X**. No second skeleton, ever
— that is the source of every coordinate-convention and rest-pose bug. The same
pose tensor drives the Habitat humanoid and the Blender render.

## Repo layout (planned)

```
docs/            architecture, decisions, runbooks
world/           Habitat integration (env, sensors, humanoid driver)
mind/            agent cognition (brain client, memory, perception, planner)
memory/          Neo4j schema + access layer
render/          bridge to the sampl SMPL-X→Blender cinematic pipeline
deploy/          docker-compose + per-service Dockerfiles
```

## Hardware

GPU fleet: 11× sm_70 16 GB (V100 + CMP), mostly occupied by other services; one
K620. A separate `rtx0` host has 4× RTX 3060 (Ampere) used as an OptiX render
host. Everything here targets sm_70 / CUDA, headless, in Docker.
