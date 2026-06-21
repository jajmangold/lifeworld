# Decision log

Lightweight ADRs. Each is mirrored to a GitHub issue for discussion.

## ADR-0001 — Embodiment substrate: Habitat 3.0 (not Isaac/Genie/AgiBot)
**Status:** accepted (pending build validation on sm_70)
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
