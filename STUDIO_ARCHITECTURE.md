<!-- ⚠️ ASPIRATIONAL / TARGET-ARCHITECTURE DOC. Describes a *planned* unified LangGraph system and a target
     layout that does NOT fully exist yet. For the current, working state (the SDNQ LTX farm, the news +
     docupipe paths as they actually are), read studio/AGENTS.md and the per-area AGENTS.md. docupipe/ is
     the real LangGraph pipeline today; the news path is script/server-based, not a unified graph. -->

# Studio — unified LangGraph production system (news + docuseries + future shows)

Target design to unify the two current systems into one langgraph "Studio," with a shared data model
that scales to new show types and scenes. Grounded in trailmark analysis of the current code and
prevailing langgraph/broadcast patterns.

## Where we are (trailmark)
- **docuseries = `docupipe/`** — a clean langgraph package (123 nodes, 1 entrypoint): `state.py`
  (DocuState + Segment), `graph.py` (research → deep-research fanout → story bible → write-act fanout →
  script editor → curate → render fanout → assemble), `cache.py` (node caching), `provenance.py`, `sql/`,
  RetryPolicy. **This is the target shape.**
- **news = `bot/` root + `newscast/`** — 55 loose functions, **0 entrypoints, no langgraph**, + heavy
  cruft (114 `.log`, superseded scripts). The *method* is locked (make_anchor → swap → muse → settle →
  flashvsr → broadcast_finish; formats anchor_wall/fullscreen_anchor/ots/vo_broll/title/reporter_pkg) —
  it just isn't packaged.

## The core insight
News and docuseries are the **same abstract pipeline**:
`brief → writers' room (research/script) → plan ordered segments → render each segment (fan-out) →
assemble episode + graphics + review.`
They differ only in (1) **segment kinds/renderers**, (2) **writers'-room prompts/pacing**, (3)
**talent/set/brand**. All three are *data*, not separate code. So we generalize docupipe, don't fork it.

## Data model (`studio/state.py`) — scales via polymorphism + registries
Industry hierarchy Show → Episode → Segment → Shot, with reusable talent/sets/graphics.

```
Production   { id, show_type: "news"|"docuseries"|..., brief, title, profile_ref, status,
               research, dossiers[], bible, segments[], rendered[+], cue_sheet[+], errors[+],
               episode_path }                       # generalizes DocuState (typed envelope)

Segment      { id, order, kind, spec{...}, talent_ref, set_ref,        # kind = the discriminator
               narration, wav_path, image_path, clip_path, dur,        # render-filled (paths-not-bytes)
               asset{source,license,attribution,rights_ok}, enhanced } # extends docupipe Segment
```
`kind ∈ {anchor_wall, fullscreen_anchor, ots, reporter_pkg, vo_broll, title, narration, cold_open, …}`
— open vocabulary; each maps to a renderer in the **registry**.

**Reusable catalogs** (`studio/catalog/`, JSON/SQL like docupipe) — *this is what makes new shows/scenes cheap*:
- **Talent** `{id, role: anchor|reporter|narrator, voice_ref, avatar_ref, lower_third}` — Graham Pierce,
  Elise Warren, docuseries narrator.
- **Set** `{id, kind: studio|field|videowall|broll_plate, env (HDRI pano), framing}` — newsroom, flood field.
- **Brand** `{id, logo, colors, lower_third_style, ticker, bug}` — NNS package.
- **ShowProfile** `{show_type, writers_room(prompts/pacing/target), allowed kinds, default talent/set/brand}`
  — the *only* thing that differs news from docuseries. Add a profile → new show type.

## Graph (`studio/graph.py`) — one spine, subgraph-composed, profile-driven
```
START → intake(brief→profile) → research → deep_research(fanout) → bible/rundown → write(fanout)
      → assemble_script → editor → curate → review/approve
      → render_segment (Send fan-out; DISPATCHER by segment.kind) → assemble_episode → finish(gfx+bed) → END
```
- Writers' room = docupipe's, unchanged; per-show differences come from **ShowProfile** (prompts/pacing).
- `render_segment` is a **dispatcher**: reads `segment.kind`, calls the registered renderer. Renderers are
  reusable **subgraphs/services** (research: subgraphs are the langgraph modularity pattern):
  - `renderers/anchor.py`, `reporter.py` → the make_anchor talking-head pipeline (a shared subgraph).
  - `renderers/vo_broll.py`, `title.py`, `narration.py` → the ffmpeg/Ken-Burns builders (exist today).
- GPU parallelism: factory.py's per-resource flocks become concurrency limits under the Send fan-out.

**New show type = new ShowProfile (+ maybe a renderer). New scene = new Set entry. No graph rewrite.**

## Service layer (`studio/services/`) — wrap the locked, working code (don't rewrite)
`llm` (deepseek/qwen) · `tts` (amd1 qwen3-tts voices) · `avatar` (calls make_anchor.sh) · `imagegen` (zimage/klein)
· `archival` · `upscale` (flashvsr) · `graphics` (broadcast_gfx brand package) · `vision_review` (qwen9b-vis).
Renderers call services; services are idempotent `spec → path`. The locked render scripts are wrapped,
not rewritten.

## Target layout (mirror docupipe src-layout)
```
studio/src/studio/  state.py  graph.py  profiles/{news,docuseries}.py  nodes/{writers_room,curate,review,assemble}.py
                    renderers/{registry,anchor,reporter,vo_broll,title,narration}.py
                    services/{llm,tts,avatar,imagegen,archival,upscale,graphics,vision_review}.py
                    catalog/{talent,sets,brands}.py  cache.py  provenance.py  config.py  run.py
studio/assets/      brand logos · HDRIs · avatars · voice refs
```

## News writers' room + sources
News gets the SAME writers' room as docuseries (research → rundown → write → editor → curate), driven by
the `news` ShowProfile. Its **research node reads Wikipedia `Portal:Current events`** (`newscast/wiki_news.py`):
the daily, curated, sourced feed across 10 categories → structured stories `{category, headline, summary,
links, sources}`. Story visuals come from **Wikimedia Commons lead images** (license + attribution, `rights_ok`
gate) with **zimage generation as fallback**. Attribution flows into the existing docupipe **provenance +
cue_sheet**. This replaces the single-shot `draft_news.py`.

## Delivery — one render, many deliverables (multi-aspect + multi-platform)
The expensive GPU work (Blender render → swap → MuseTalk → FlashVSR) is **aspect-agnostic** — it acts on the
character/face. So each segment renders **one high-res "clean master"**; delivery derives cheaply from it:
- **Master** (16:9, generous margins) → the shared, GPU-costly artifact per segment.
- **Reframes** (cheap ffmpeg + per-aspect graphics): **16:9** (YouTube long-form) and **9:16** (Shorts/TikTok),
  graphics laid out per aspect (vertical stacks lower-third/ticker).
- **Cuts**: per-story **shorts** sliced from the masters (hook + punchy vertical framing + burned captions).
- **Packages**: full episode + per-platform playlists (YouTube long-form, YouTube Shorts, TikTok).
Data model: `Segment.master_path` (shared) + `deliverables[]` `{platform, aspect, path}`. The `finish` node
runs the reframes/cuts; GPU cost paid once, deliverables multiply for free. Reframe happens AFTER the facial-
animation stage so it's all downstream of the shared cost.

## Migration — incremental, both systems stay working
- **Phase 0 (now, safe/reversible):** archive cruft — 114 `.log`, superseded scripts, stale output dirs →
  `/mnt/24tb/nvme-offload/bot-archive/` (still in git). Repo becomes legible.
- **Phase 1:** `studio/state.py` + `catalog/` seeds (talent/sets/brand). Schema only, no behavior change.
- **Phase 2:** service wrappers around existing working code; prove a news segment via renderer == via factory.
- **Phase 3:** generalize docupipe graph → `studio/graph.py` with the render dispatcher + `profiles/news.py`;
  run a full news rundown through the graph at parity; factory becomes the executor under `render_segment`.
- **Phase 4:** retire the ad-hoc top-level orchestration; make_anchor stays as the avatar service.

Risk controls: docupipe untouched until Phase 3; locked scripts wrapped not rewritten; caching/provenance
carried over so reruns are cheap; each phase leaves a shippable system.
