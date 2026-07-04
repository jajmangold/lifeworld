# Changelog

All notable changes to the studio. Format: [Keep a Changelog](https://keepachangelog.com/). Every
behavior/structure change adds a bullet under **Unreleased** (see [pm/PROCESS.md §8](pm/PROCESS.md)).
Date-stamp a release when a roadmap milestone ships.

## [Unreleased]

### Added
- **Project-management system** in `pm/` — VISION (north star + quality bar), STRATEGY (money + YouTube/
  TikTok playbook + compliance survival rules, sourced), PROCESS (the operating system: ship>perfect>shiny,
  mandatory sequential-thinking + trailmark, DoD, WIP limits, experiment/parking-lot discipline), ROADMAP
  (P0 lock quality → P1 first channel → P2 portfolio → P3 AI showrunner), BACKLOG (prioritized, dependency-
  mapped work items). Wired into `studio/AGENTS.md` ("How we work — READ AND FOLLOW").
- `render/sdnq_ltx/a2v_talk.py` — native audio-driven talking anchor on the SDNQ farm (frozen audio latent
  + cross-modal attention); replaces the swap+muse lip-sync bolt-on.
- `render/sdnq_ltx/ic_union.py` — IC-LoRA depth/pose control + green subject-mask keying on the farm.
- AGENTS.md hierarchy: master `studio/AGENTS.md` + per-area (render, newscast, swap, docupipe, splat,
  viverse_avatar) + `render/sdnq_ltx/SDNQ_STACK.md` deep reference.
- Canonical shared-services section in `studio/AGENTS.md` (amd0 qwen9b, amd1 qwen3-tts), env-overridable.

### Changed
- **Honest reset of P0**: we do NOT have a production-quality anchor yet. Reframed `pm/` E0.1 from "freeze
  the spec" to "**ACHIEVE** a production-quality talking anchor" (unsolved R&D), marked the VISION quality
  bar **STATUS: NOT MET**, and **unparked** the multi-GPU crisp-lip-sync work into E0.1b as a candidate fix
  (per the shiny-object gate, fixing the unlocked base is P0, not a parking-lot item). Everything downstream
  is explicitly blocked on E0.1.
- **Endpoints consolidated** onto amd0/amd1 (env-configurable): qwen9b vision+chat →
  `amd0.python-bull.ts.net/v1` (requires `chat_template_kwargs:{enable_thinking:false}`); qwen3-tts →
  `amd1:8064`. Repointed newscast/{synth_voice,synth_anchor,review}, docupipe config + tools/review,
  podcastfy, `.env.example`.
- **Branding standardized to NNS** (National News Service) — removed FFNN / "The Feed Forward" drift
  (broadcast_gfx, broadcast_finish, segments.json, talent.json).
- **Paths**: `/containers/projects/bot` (symlink) → real `/containers/live/studio` across newscast + render.
- Bannered `README.md` / `STUDIO_ARCHITECTURE.md` as aspirational (not current state).

### Removed
- Decommissioned the retired local `qwen3dia` TTS container (superseded by amd1; ~5.9 GB VRAM freed).

### Archived
- SMPL-X / Kimodo / FLOAT / MDM / teeth / Audio2Face avatar cluster (18 files) →
  `/mnt/24tb/nvme-offload/bot-archive/render-experiments/` (superseded by the LTX-2.3 SDNQ farm + a2v_talk).

### Fixed
- `newscast/review.py` empty-output bug (amd0's qwen3.5 needs `enable_thinking:false` or content lands in
  `reasoning_content`).

<!-- next release: date-stamp when a ROADMAP milestone (e.g. P0 exit) ships. -->
