# studio/ — AGENTS.md

> Synthetic broadcast studio: generate photoreal AI **news anchors** and **docuseries** on a Volta mining
> rig (sm_70). This file orients you and sets the rules. **Read it before touching anything here.**

Parent: `/srv/nvme-data/containers/live/AGENTS.md` (the whole `live` container — services, host OOM safety,
sm_70 notes). Each major area below has its **own `AGENTS.md`** — the nearest one wins; read it before
working in that area.

---

## How we work — READ AND FOLLOW ([pm/](pm/) is the project-management system)

This is a **revenue-generating** operation with a project-management system that is **binding**. Before any
non-trivial work:

1. **Orient:** [pm/VISION.md](pm/VISION.md) (north star + the production quality bar) and
   [pm/STRATEGY.md](pm/STRATEGY.md) (how we make money + the compliance rules that keep the network alive).
2. **Pick work from [pm/BACKLOG.md](pm/BACKLOG.md)** — the highest-priority *unblocked* item. Respect the
   **WIP limit (≤2 in progress)**. If your task isn't an item, make it one first. Sequencing:
   [pm/ROADMAP.md](pm/ROADMAP.md).
3. **Follow [pm/PROCESS.md](pm/PROCESS.md) for every item** — it is the operating system:
   - **SHIP > perfect > shiny.** Lock production quality, then scale. New tech must clear the shiny-object
     gate (PROCESS §6) or it goes to the parking lot. **The system exists to keep us shipping — don't chase
     shiny.**
   - **Sequential-thinking is MANDATORY** to plan any non-trivial change.
   - **Trailmark is MANDATORY** before touching unfamiliar code (structure/complexity/deps), where it makes
     sense.
   - **Done = demonstrated + documented + logged:** meet the item's Done-when with evidence, update the
     nearest `AGENTS.md`, add a [CHANGELOG](CHANGELOG.md) entry, set the BACKLOG status. See PROCESS §2 (DoD).
   - **Experiments are time-boxed** with a kill criterion; graduate-or-archive. Retire superseded code to
     `bot-archive/` — no sprawl.

The golden rules below (sm_70, never `kill -9` CUDA, etc.) are the operational floor; `pm/` is *how we
decide what to build and prove it's done*.

---

## What studio is (current state, not the plan)

Two production lines share one rig and one finishing toolchain:

1. **News anchor / short clips** — the active, proven path:
   **LTX-2.3 SDNQ farm** (`render/sdnq_ltx/`, resident int4 on the Volta cards) generates photoreal anchor
   video → **face-swap** (`swap/`) locks identity → **finishing** (FlashVSR upscale on rtx0, graphics) →
   **newscast** (`newscast/`) adds TTS narration + broadcast graphics. Newer: native **audio-driven talking**
   and **IC-LoRA depth control** live in `render/sdnq_ltx/` (see its doc).
2. **Docuseries / long-form** — `docupipe/`: a LangGraph pipeline (research → script → media → assembly)
   that reuses the same service clients (tts, media, restore, swap).

**Legacy / experimental — do NOT assume these are live:**
- `README.md` ("lifeworld", SMPL-X-everything) and `STUDIO_ARCHITECTURE.md` ("unified LangGraph, *target*
  layout, migration") are **aspirational/planning docs**, not current state. Don't take their file layout or
  "one unified graph" as real. This `AGENTS.md` + the per-area ones are the source of truth for what exists.
- The Blender **SMPL-X / kimodo / FLOAT / teeth / calibration** cluster in `render/` is retired.
- `splat/` is mostly vendored 3D-avatar/restoration research, parked (not in the anchor pipeline).

---

## Area map (each has its own AGENTS.md unless noted)

| Area | Role | Status |
|---|---|---|
| `render/sdnq_ltx/` | **LTX-2.3 SDNQ generation farm** — the engine. Full doc: `render/sdnq_ltx/SDNQ_STACK.md` | **active core** |
| `render/` | Blender anchor/character renderers + FlashVSR + LTX finishing passes | active (Blender path) + legacy cluster |
| `swap/` | face-swap job-queue server (inswapper+GFPGAN), locks talent identity | active |
| `newscast/` | news production: TTS narration, broadcast graphics, segment/factory orchestration | active |
| `docupipe/` | docuseries LangGraph pipeline (research→script→media→assembly) | active (separate line) |
| `viverse_avatar/` | avatar meshes/rigs (Avatar_* bone GLB) + hair tools + head textures for the Blender path | supporting |
| `splat/` | vendored 3D Gaussian-splat / avatar-recon / restoration research | experimental/parked |
| `humgen/`, `catalog/`, `assets/`, `sampl/` | talent/wardrobe JSON, human-gen assets, seed images (data, no AGENTS.md) | data |
| `podcastfy/`, `bake/` | podcast gen; texture baking — see their own `README.md` | supporting |
| `output/` | **21 GB of generated artifacts + job queues** (`swap_jobs/`, `muse_jobs/`) | generated — see boundaries |
| `deploy/` | docker-compose for the studio services | infra |
| `world/`, `src/`, `mind/` | smaller/early modules — read the code before assuming a role | misc |

Root orchestrators (studio/): `make_anchor.sh` / `make_character.sh` (one-shot photoreal talking anchor:
proc_perf → render → swap → MuseTalk → finish), `render_anchor_anim.py` / `render_anchor_env.py` (Blender
anchor render), `proc_perf.py` (audio→ARKit performance), `flashvsr_face.sh`. Docs: `ANCHOR_PIPELINE.md`,
`FLASHVSR_FACE_UPSCALE.md`.

---

## Shared network services (canonical endpoints — the ONE place)

These are consolidated onto two Tailscale hosts. Code reads them from env vars (defaults below); never
hardcode a `localhost:port` for these again.

| Service | Endpoint (default) | Env var | Notes |
|---|---|---|---|
| **qwen9b** — ALL vision-QA + chat | `https://amd0.python-bull.ts.net/v1/chat/completions` | `QWEN_URL` / `QWEN_MODEL` | model `Qwen3.5-9B-UD-Q4_K_XL.gguf`. **MUST send `chat_template_kwargs:{enable_thinking:false}`** or `content` is empty (text lands in `reasoning_content`). Replaces all the old separate qwen-vis instances (`:8021`, `:8040`). |
| **qwen3-tts** — narration/voices | `http://amd1:8064` | `TTS_URL` | voxserver: `/mono` (POST `{text,ref1,seed}`), `/dialogue`, `/blend`, `/v1/audio/speech`, `/voices`. `--ref` paths are inside the server's `/work` (e.g. `/work/ref1.wav`). https tailscale 502s — use `http://amd1:8064`. Replaces the retired local `qwen3dia`/`crispasr-tts`. |

(DeepSeek for script drafting stays external: `https://api.deepseek.com`.)

## Golden rules (apply EVERYWHERE — inherited by every area)

1. **sm_70 / Volta reality.** CMP 100-210 & V100: 16 GB, **no bf16 tensor cores, no flash-attention, no
   NVENC, PCIe Gen1 ×1**. Quantize (SDNQ int4), keep heavy modules resident on their own card, encode video
   on CPU. Single-card attention caps ≈ 384×512 for the 22B DiT.
2. **NEVER `kill -9` a live CUDA process.** It corrupts the driver (GPUs → "Unknown Error", container wedges,
   needs a host reboot). Always **SIGTERM, then poll VRAM until <2 GB** before reusing a card.
3. **Don't rewrite locked, working code — WRAP it.** Servers (swap, muse, TTS, FlashVSR) and vendored trees
   are proven; add adapters/clients, don't refactor them.
4. **Prompt caches are byte-identical-keyed.** LTX embeds/connectors are cached by prompt md5
   (`output/embeds_<md5>.pt`, `conn_<md5>.pt`); a one-char prompt change silently re-streams the 6.35 GB
   connector module over the ×1 riser. Reuse the exact canonical prompt.
5. **LTX frame counts are 8k+1** (25, 57, 73, 89, …); use the distilled 8-step sigma schedule, not linspace.
6. **container GPU index ≠ `CUDA_VISIBLE_DEVICES`** (CUDA enumerates by PCI order). When freeing a card,
   confirm by which physical card actually drops to idle.
7. **Outputs go to `output/`.** Use it for generated media and job queues; keep the repo tree clean.

---

## Boundaries

**Never (without explicit ask):**
- `rm`/overwrite `output/` (21 GB of finished renders + live job queues) or any model cache
  (`~/.cache/huggingface`, checkpoints) — they're huge and slow/impossible to regenerate.
- Touch other containers' data under `/srv/nvme-data/containers/*` or `/mnt/*` archives.
- `docker rm`/recreate `sdnq-mgpu`, restart FlashVSR workers on rtx0, or reboot the host.
- Commit secrets / tokens.

**Ask first:** recreating the farm container, changing the canonical anchor prompt (invalidates caches),
deleting checkpoints, anything that stops the resident daemons mid-job.

**Safe by default:** reading code, `trailmark` analysis, writing to `output/` or a scratch dir, adding new
scripts/clients, the free-a-card test pattern (SIGTERM a daemon → poll <2 GB → test → restart).

---

## Conventions for editing these docs

- Use **AGENTS.md** (open standard) per major area; **nearest file wins**. Lead with **commands**, describe
  **capabilities** (not exhaustive file paths — paths rot and poison agent context), point to a few
  **canonical** files, flag **legacy** ones. Keep concise; update when the code moves.
- The deepest operational reference is `render/sdnq_ltx/SDNQ_STACK.md` (the farm) — model there.
