# Decision log

Lightweight ADRs. Each is mirrored to a GitHub issue for discussion.

## ADR-0001 — Embodiment substrate: Habitat 3.0 (not Isaac/Genie/AgiBot)
**Status:** accepted — **VALIDATED on sm_70 (2026-06-21)**: headless habitat-sim built
(micromamba, `world/Dockerfile.habitat`) and initialized its GPU EGL renderer on a
Tesla V100, producing sensor frames. Gotcha fixed: the cuda:runtime base lacks the
NVIDIA EGL vendor ICD (`10_nvidia.json`); without it glvnd loads software Mesa and
habitat errors "unable to find CUDA device 0 among 1 EGL devices". The Dockerfile now
writes the ICD.
**Context:** Need a world for agents to live in with virtual sensors, on a Volta
(sm_70, no RT cores) GPU fleet.
**Decision:** Use Habitat 3.0. It runs on standard CUDA / headless / V100, its
humanoids are SMPL-X-native (matches our render pipeline → zero retargeting), and
it ships RGB/depth/semantic/audio sensors and navigation.
**Rejected:** NVIDIA Isaac Sim / AgiBot Genie Sim — require RTX RT cores; V100
unsupported. AgiBot GO-1 is a manipulation policy, orthogonal to the goal.
**Genesis** (see ADR-0005) was compared and deferred: stronger physics/touch but
robot-centric (no first-class SMPL-X), and physics-accurate humans need an
articulated rig — reintroducing the rigging complexity ADR-0002 removes.
**Consequences:** Inherit robust SMPL-X locomotion; large scene-dataset download;
in-sim render is not cinematic (Blender handles that).

## ADR-0002 — Stay in SMPL-X space, retire cross-skeleton retargeting
**Status:** accepted
**Context:** The prior pipeline drove stylized VRoid/Mixamo rigs by retargeting
from SMPL; this is the root of coordinate-convention, rest-pose, topology, and
foot-sliding brittleness.
**Decision:** Characters are textured SMPL-X meshes driven by SMPL-X-native motion
(AMASS-SMPL-X, Motion-X, Inter-X). Foot contact handled by procedural foot-lock
IK in SMPL-X space. No second skeleton anywhere.
**Consequences:** Drops MotionMillion 272-dim (broken decoder) and the whole
retarget code path. Same pose tensor drives Habitat and Blender.

## ADR-0003 — Neocortex: DeepSeek V4 Flash (API)
**Status:** accepted
**Context:** Need strong long-horizon reasoning for agent cognition/narrative.
**Decision:** Route agent reasoning to DeepSeek V4 Flash. Local Qwen VLM serves as
the visual cortex (Habitat RGB → text). Local Qwen may later absorb high-volume
routine chatter if cost demands.
**Consequences:** Per-token cost scales with sim size; sim content leaves the
machine — accepted.

## ADR-0004 — Memory/society in Neo4j
**Status:** accepted
**Decision:** Persistent life-record, social graph, storylines, and episodic
memory in a dedicated Neo4j instance (separate from the existing `n4j_atlas`).

## ADR-0005 — Genesis as a later opt-in physics/touch layer (not the substrate)
**Status:** accepted (deferred to M5+)
**Context:** Genesis offers contact-rich differentiable physics and the only
physically-accurate tactile/touch sensors of the candidates — matching the
original interest in ROM, kinematics, advanced hands, and touch.
**Decision:** Do **not** make Genesis the runtime substrate (Habitat is, ADR-0001).
Bring Genesis in later as an opt-in physics module for selected contact-rich
interactions / tactile experiments, feeding SMPL-X poses back to the render path.
**Note:** the *Virtual Community* paper (arXiv 2508.14893) builds a humans+society
open world on Genesis — a reference if a full physics-substrate migration is ever
warranted.
**Open:** V100/sm_70 support for Genesis is unverified; validate before relying on it.

## ADR-0006 — Lip-sync: LAM Audio2Expression is the SMPL-X path (A2F-3D is NOT installed)
**Status:** accepted
**Context:** A full-drive review (2026-06-21) found **no NVIDIA Audio2Face-3D install** anywhere
(despite prior belief it had been tried). What IS present and live: **LAM_A2E** (:8202,
audio→ARKit-52, ~0.15s), **KDTalker** (:8200, portrait→2D talking-head video), **ACE-Step**
(music/TTS), **Higgs-TTS** (:8055). **TalkSHOW** exists but is code-only/stale (2023).
**Decision:** For SMPL-X (in-engine) lip-sync, use **LAM_A2E → `face.talk.arkit_to_face`**
(learned jawOpen visemes + real eye blinks), called from the host (the render container can't
reach :8202). The amplitude-jaw `audio_to_face` is the offline fallback. **KDTalker** is reserved
for 2D talking-head *video* shots. **Audio2Face-3D vertex-drive** stays the optional future upgrade
(better visemes; SMPL-X FLAME expression PCs are too weak alone).

## ADR-0007 — ProtoMotions as the physics-embodiment bridge (for ADR-0005)
**Status:** accepted (validate next)
**Context:** The review found **ProtoMotions** (`/mnt/24tb/containers-archive/charbrain/ProtoMotions`)
— a SMPL character-control harness with built-in adapters for **Genesis / MuJoCo / IsaacGym /
IsaacLab / Newton**. This is a ready-made SMPL→physics bridge.
**Decision:** When pursuing the ADR-0005 physics/touch track, build on ProtoMotions rather than
hand-rolling SMPL↔sim. Target Genesis or MuJoCo on the V100 fleet (no RT cores needed).
**Open:** underlying sims (Genesis/MuJoCo) are not installed; ProtoMotions is code-only.
**Spike result (2026-06-21):** current **Genesis requires torch≥2.8**, but the Volta-supporting
CUDA wheels top out at **torch 2.6** (cu124), and Genesis also needs X11 at import
(`XRenderFindVisualFormat`). So Genesis-on-sm_70 is NOT a quick win — it needs an older Genesis
pinned to torch 2.4–2.6 + headless-X handling (xvfb). **Recommendation: use MuJoCo (CPU, via
ProtoMotions `requirements_mujoco.txt`) as the physics path on this fleet**; revisit Genesis only
with a version-pinned build or on the RTX box. Tracked in issue #11.

## ADR-0008 — Habitat is a fresh install; Replica scene data already present
**Status:** noted (amends ADR-0001)
**Context:** Review confirms **habitat-lab / habitat-sim are NOT installed** — only Replica scene
data exists (`/srv/nvme-data/containers/live/homebuilder/replica_out`). No HM3D/HSSD/ReplicaCAD.
**Consequence:** M1 (issue #3) is a genuine fresh build. Also available as a *2D/video* render
alternative: the **`wan2gp` arsenal** (`/mnt/24tb/containers-archive/wan2gp`) — Wan2.1 InfiniteTalk
(audio→talking video), LTX-2.3-22B, Flux2-Klein, Z-Image-Turbo, wav2vec, depth-anything, RIFE.
