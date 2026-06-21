# lifeworld

Autonomous SMPL-X **humans that live lives** in a simulated world, remember
through a graph, and whose selected moments are rendered cinematically.

Agents perceive a real 3D world through virtual sensors, decide with an LLM
"neocortex", remember in a graph database, speak/hear, and act through embodied
SMPL-X humanoids. Selected story beats are re-rendered in film-quality Blender.

> Status: **bootstrap**. Architecture decided; first build = embodiment substrate
> + a single SMPL-X human that works end-to-end (sim pose → cinematic render).

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
