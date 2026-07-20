# docupipe/ — AGENTS.md
> LangGraph "writers' room" that turns a topic into a finished long-form documentary episode.

## What it does
Given a `--topic`, a LangGraph `StateGraph` runs a decomposed documentary pipeline: broad
research → per-entity deep-research dossiers (fan-out) → story bible → per-act script writing
(fan-out) → assembled + showrunner-edited script → a human/auto script-approval gate → per-segment
render (fan-out) → episode assembly. Each segment pairs narrated VO with a rights-cleared archival
image (or a generated period illustration / motion graphic), Ken-Burns'd and mastered by ffmpeg into
a `.mp4` with a credits sheet and an APA GenAI-disclosure cue sheet. It is topic-agnostic
(true-crime is the working example). The writers'-room + render spine is active; talking-head
researcher cutaways (`cast`, `anchor`) and premium FlashVSR superscale are heavier/opt-in.

## Run
```bash
cd docupipe/src
python -m docupipe.run --topic "The Axeman of New Orleans" --auto   # end-to-end, auto-approve script
python -m docupipe.run --resume <job_id>                            # inspect a paused job's script
python -m docupipe.run --resume <job_id> --approve                  # approve script & finish render
```
State + checkpoints live in a SQLite DB (`data/docupipe.sqlite`); jobs are resumable. Config is all
env-overridable (`config.py`) — LLM defaults to cloud DeepSeek, services default to on-box HTTP.
OpenCode Go credentials resolve from the tier-specific key, `OPENCODE_GO_API_KEY`, an explicitly
configured `DOCUPIPE_OPENCODE_GO_TOKEN_FILE`, then the OpenCode auth file. Runtime container trees
are never credential sources. The LLM client sends the declared `HTTP_UA`; OpenCode Go rejects
Python urllib's default user agent even when the bearer credential is valid.
DeepSeek V4 writing calls use explicit high-effort thinking. Treat `reasoning_content` as internal
model output, require nonempty final `content`, and budget `max_tokens` for both.

## Architecture
- **`state.py`** — `DocuState` TypedDict, flat and paths-not-bytes; fan-in keys (`dossiers`,
  `act_drafts`, `rendered`, `cue_sheet`, `errors`) use `operator.add` reducers.
- **`graph.py`** — the spine: wires nodes + conditional fan-out edges + a `RetryPolicy` for transient
  HTTP. Read this first to see the flow.
- **`nodes.py`** — every node's logic (research/write/curate/`render_segment`/assemble) + the fan-out
  `Send` functions + the `interrupt()` approval gate. `render_segment` and `curate_node` are the hot,
  complex nodes.
- **`clients/`** — one thin, resilient wrapper per external service (see below). They cache and
  degrade gracefully (return original/None on failure).
- **`config.py`** service URLs; **`provenance.py`** rights gate + credits + cue sheet; **`cache.py`**
  content-hash disk cache.

## Key files
- `src/docupipe/graph.py` — source of truth for pipeline shape/order.
- `src/docupipe/state.py` — the data contract every node reads/writes.
- `src/docupipe/nodes.py` — all node behavior + the script-approval interrupt.
- `src/docupipe/config.py` — every service endpoint, model, and craft default (FPS, LUFS, WPM…).
- `src/docupipe/provenance.py` — rights enforcement + APA disclosure output.

## clients/ (what each wraps)
`research` (SearXNG search+scrape), `archives` (rights-gated Wikimedia/LoC image fetch), `llm`
(OpenAI-compatible chat, DeepSeek/local), `vision` (Qwen3.5-9B image judge), `generate` (Z-Image
Turbo period illustrations), `docgfx`/`geo`/`slides` (motion graphics / OSM maps / Presenton decks),
`restore` (ESRGAN upscale) + `superscale` (FlashVSR premium), `tts` (amd1 qwen3-tts narration), `music`
(ACE-Step instrumental score), `media` (ffmpeg Ken-Burns + master), `cast`/`anchor` (talking-head
researcher cutaways via the news pipeline).

## Gotchas & rules
- Do NOT rewrite the `clients/` — they wrap locked, working on-box services (URLs/ports in
  `config.py`); keep the resilient "return original on failure + cache" contract intact.
- Provenance is load-bearing: never emit an image without a resolved `rights_ok` asset, and keep the
  cue-sheet / disclosure path — `provenance.gate()` gates assembly.
- Fan-in state keys must keep their `operator.add` reducers or parallel render writes clobber.
- Prefer editing nodes/graph over the state contract; a `DocuState` change ripples to every node.
- Distinct from the news-anchor path — this is the long-form docuseries generator. Global rules
  such as keeping generated media, credentials, and model weights out of Git still apply.

## Related
- Shares on-box services with the news path: `tts` (amd1 qwen3-tts), `media`/ffmpeg, `restore` (ESRGAN),
  and the face-swap + MuseTalk talking-head chain via `make_anchor.sh` (`clients/anchor.py`).
