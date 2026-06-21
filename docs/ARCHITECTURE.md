# Architecture

## Goal

An auto-NPC / auto-sim: characters that can *live a whole life* (within reason).
The long arc — many agents living interlocking storylines tracked in a graph,
interacting first in text, then embodied and selectively rendered to film. The
near arc — **get one human working correctly**, end-to-end, on real hardware.

## The one infeasible idea, removed

The original framing (AgiBot World → map SMPL-X into a robot → derive kinematics
→ train VLA policies → Genie Sim) fails on two counts:

1. **Hardware.** Genie Sim is built on NVIDIA Isaac Sim / Omniverse, which
   *require* RTX RT cores. The V100 (sm_70) is explicitly unsupported. The whole
   fleet here is Volta. Non-starter.
2. **Goal mismatch.** GO-1 is a real-robot *manipulation* foundation policy.
   Living narrative lives is a cognition + world-sim + animation problem, not a
   grasping-policy problem. Training a VLA on sm_70 would be a multi-month detour
   that produces no narrative output.

AgiBot World is retained only as an *optional* manipulation-motion data source
(hand/object interaction clips) if a rendered scene ever needs fine manipulation.

## The substrate: Habitat 3.0

[Habitat 3.0](https://arxiv.org/abs/2310.13724) ("A Co-Habitat for Humans,
Avatars and Robots") is the embodiment layer.

- **Runs on V100, headless.** Standard-CUDA rasterization renderer; no RT cores,
  no Omniverse. The original Habitat agents were trained on V100 clusters.
- **SMPL-X-native humanoids.** Skinned-mesh avatars with SMPL-X pose/shape +
  pose-dependent blend shapes. The driving signal is the *same* SMPL-X tensor our
  Blender renderer consumes → sim ↔ render with **zero retargeting**.
- **Robust locomotion/navigation already solved.** Path-planned humanoid walking,
  obstacle avoidance, human-robot interaction benchmarks. We inherit "the human
  walks around the house without breaking" instead of rebuilding it.
- **Virtual sensors built in.** Per-agent RGB, depth, semantic segmentation,
  GPS/compass; audio via SoundSpaces. This *is* the vision/hearing/proprioception
  the project wanted.
- **Real worlds to live in.** HM3D / Matterport3D / Replica / Gibson scenes.

Performance (single-env, from the paper): Spot 245 FPS, humanoid 188 FPS,
robot+humanoid 136 FPS; >1000 FPS batched.

### Known costs / open risks
- Non-trivial install (habitat-sim build, large scene datasets, some NC licenses).
- In-sim render is real-time rasterized (functional, not cinematic) — cinematic
  output comes from the Blender path, not Habitat.
- Humanoid motion repertoire is locomotion-centric; rich gestures/manipulation
  still come from our motion library (AMASS/Motion-X/Inter-X) + MDM.
- sm_70 viability is historically true but must be re-validated on *this* build.

## Layers

### 1. Mind (neocortex)
Per-agent cognition loop: **perceive → remember → reflect → plan → act → speak**.

- **Reasoning brain:** DeepSeek V4 Flash (API). Drives goals, plans, dialogue,
  reflection. (Cost scales with sim size; content leaves the machine — accepted.)
- **Visual cortex:** local Qwen VLM turns Habitat RGB frames into text the brain
  reasons over (per the house rule: vision via the local Qwen VLM, not Claude's).
- **Memory stream:** importance-weighted observations with retrieval, in the
  lineage of Stanford's Generative Agents.

### 2. World / embodiment — Habitat 3.0
The live stage. Owns physics, navigation, sensors, and the SMPL-X humanoid
bodies. Exposes an action API (navigate-to, look-at, play-motion, speak-event)
and a sensor API (rgb/depth/semantic/audio) to the Mind layer.

### 3. Memory / society — Neo4j
Persistent life-record and social graph: agents, relationships, locations,
objects, events, storyline beats, and episodic memories. The graph *is* the life.
Runs as its own Neo4j instance (separate from the existing `n4j_atlas`).

### 4. Voice / ears
- **Higgs-TTS** (`:8055`, OpenAI `/v1/audio/speech`) for speech out.
- **Granite STT** (amd0/amd1) for live human speech in.
Needed only for spoken interaction / perception-closure; symbolic dialogue events
suffice otherwise.

### 5. Cinematic render — `sampl`
The existing SMPL-X → headless Blender (Cycles/CUDA) pipeline renders *selected*
beats at film quality. Because it shares SMPL-X with Habitat, a selected window of
agent poses re-renders directly. Outstanding `sampl` work (faces via LAM, clothing,
beta variety) is tracked separately.

## Data flow (one tick)

```
Habitat sensors (rgb/depth/semantic/audio)
   → Qwen VLM caption + symbolic world-state from Neo4j
      → DeepSeek V4 Flash: decide action + dialogue
         → Habitat action API (navigate / motion / speak)  ← motion clips from AMASS/Motion-X/Inter-X
            → write observations + events to Neo4j
   (selected windows) → sampl → Blender cinematic render
```

## Milestones

- **M0** Repo + decision log + runnable Neo4j. *(this commit)*
- **M1** Embodiment substrate: Habitat 3.0 in Docker on a V100; one SMPL-X
  humanoid spawns in a scene and navigates; export its SMPL-X pose stream and
  re-render the same motion in `sampl` Blender. **Proves "the human works."**
- **M2** Mind loop: that humanoid perceives (Qwen VLM over Habitat RGB), decides
  (DeepSeek V4 Flash), acts; observations logged to Neo4j.
- **M3** Two agents interacting (Inter-X / dialogue events) + voice (Higgs/Granite).
- **M4** Society/storylines at scale + a "director" that auto-selects beats to render.
- **M5 (opt)** AgiBot manipulation clips; richer manipulation in rendered scenes.
