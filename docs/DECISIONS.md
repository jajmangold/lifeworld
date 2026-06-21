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

**A2F-3D-on-Volta validated (2026-06-21):** the official TensorRT path is out (TRT
removed Volta/sm_70 in 10.5), BUT the model ships as ONNX and **runs on a CMP 100-210
via ONNX Runtime CUDA EP** — full diffusion-v3 forward in ~909ms for 60 frames
(faster than real-time). Model at `/mnt/24tb/a2f/network.onnx`; smoke test
`world/a2f_smoke.py`. Inputs: window(B,16000)+identity(B,3)+emotion(B,30,10)+
input_latents(2,2,B,256)+noise(B,3,60,88831); output geometry(B,60,88831) for
identities Claire/James/Mark. **Integration TODO:** map A2F geometry output →
ARKit-52 (via the bs_skin blendshape solve) → reuse our `arkit_to_face` path, OR
use the Samples microservice ARKit output. Then A2F replaces LAM as the lip-sync
engine on the CMP/V100 fleet (no RTX box required).

**A2F-3D INTEGRATED (2026-06-21):** `render/a2f_lipsync.py` (in `lifeworld-a2f` image)
runs the A2F diffusion ONNX on a CMP/V100 via ORT and produces a LAM-compatible ARKit
JSON ({arkit_names, weights, fps}) — drop-in for `face.talk.arkit_to_face` → SMPL-X.
Bridge: the ONNX outputs per-vertex face-geometry DELTAS (network space); we project
each frame onto `model_data.lip_open_pose_delta` / `eye_close_pose_delta` → normalized
jawOpen + eyeBlink (no cross-space 52-blendshape solve needed). Wired into `scene_cine.py`
(batched, one container session). Validated: cinematic 3-char scene, lip-sync tracks the
active speaker. LAM remains the lightweight fallback. Mouth-shape detail beyond jaw
(pucker/funnel) is a future add (needs the geometry→bs-space alignment).

**FULL VISEMES working (2026-06-21):** confirmed `bs.neutral == model_data.neutral_skin`
(same coordinate space, max diff 0.25), so the network geometry deltas DO project onto
the named ARKit-52 bases. `a2f_lipsync.py` now does the full ridge-regularized,
frontal-masked, active-pose-gated least-squares solve -> all 52 ARKit weights
(jawOpen + mouthClose + mouthPucker + mouthFunnel + eyeBlink + ...). `bake_talk.py` /
`bake_scene.py` apply jaw (jaw_pose) + mesh-space mouth shaping (`face.talk.apply_lips`
with mouth_close/pucker) + blinks (`apply_blink`). Validated: distinct visemes
(open vowel vs closed bilabial) in single + cinematic multi-char renders.

**Sync + jaw fixes (2026-06-21):** two bugs found after first multi-char render —
(1) OUT OF SYNC: a2f_lipsync used a 1s hop / 30fps and prepended 1s of silence; the
real A2F diffusion protocol (per SDK docs) is **60fps output, 1s buffer, 0.5s hop
(50% overlap), keep center 30 frames**. Fixed windowing + a small constant lead-trim
→ jaw-vs-audio cross-correlation lag 500ms→**0ms** (corr 0.46). (2) "UNHINGED JAW":
arkit_to_face's LAM-era jaw range (jaw_max=0.52rad≈30°) saturated on A2F's strong
jawOpen; bake_talk/bake_scene now pass jaw_max=0.24 (~13°), jaw_gain=1.1 so the
mesh-space visemes shape the mouth instead of a giant drop.

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
