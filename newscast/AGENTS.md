# newscast/ — AGENTS.md
> News-production layer: TTS narration, broadcast graphics, and segment/package orchestration on top of the LTX anchor render.

## What it does
Turns a written rundown into finished broadcast segments. **Active:** anchor/reporter TTS via the qwen3-tts on **amd1** (`/mono` on `amd1:8064`, env `TTS_URL`), the graphics package (bug / lower-third / ticker / on-set video wall), multi-segment pipelining through `make_anchor.sh`, VO-over-b-roll and title/vertical variants, and mouth/eye post-fixes on rendered clips. **Content helpers:** DeepSeek script drafting + Wikipedia research + Qwen-vision QA. **Legacy/misplaced:** `blenderkit*.py` (character-asset tooling, not news), `zgen.py` / `fetch_*.sh` / `gen_backgrounds.sh` / `fetch_assets.py` (scraping helpers, mostly unused).

## Run
```bash
# from studio/
python3 newscast/synth_anchor.py [--seed N]                 # news_package.json -> output/anchor_vo_raw.wav
python3 newscast/synth_voice.py --text "..." --ref /work/reporter_ref.wav --out output/vo.wav
python3 newscast/broadcast_gfx.py                            # -> output/gfx_{bug,lower_third,ticker_*}.png
bash    newscast/broadcast_finish.sh in.mp4 out.mp4 [music.mp3]   # composite gfx + music over a clip
python3 newscast/factory.py manifest.json [--max N] [--stagger S] # pipeline many segments
```

## Key files
- `synth_anchor.py` / `synth_voice.py` — canonical TTS entrypoints (anchor read vs. arbitrary voice). Both post to `$TTS_URL/mono` (default `amd1:8064`), split into ≤28-word turns, and run the normalizer backstop.
- `normalize_tts.py` — importable `normalize(text)`; deterministic digit/acronym/unit/currency expansion. The base clone has NO text frontend, so this is mandatory — always call before TTS.
- `voices.json` — voice registry (anchor/weather/sports; `clone` refs vs. `cv` speaker+instruct).
- `broadcast_gfx.py` — source of truth for the on-air graphics package; `broadcast_finish.sh` consumes its `output/gfx_*.png` outputs.
- `factory.py` — multi-segment orchestrator: flocks the 3 scarce resources so different segments occupy render / premium / swap+muse concurrently. `build_reporter_pkg.py`, `build_vo_broll.py`, `build_title.py`, `reframe_vertical.py` are the other segment builders (last three are pure PIL/ffmpeg, no GPU).
- `make_screen.py` / `ots_panel.py` / `reporter_lt.py` — story video-wall, OTS bubble, reporter lower-third PNGs.
- `mouth_settle.py`, `eye_restore.py` — post-render fixups (silence-pause lip smoothing; real-eye compositing after FlashVSR-face).
- `segments.json` — the daily rundown (per-segment text, emotion, speaker, lower-third, image).
- `draft_news.py` (DeepSeek), `wiki_news.py` (research), `review.py` (Qwen-vision QA) — content-side helpers.

## Gotchas & rules
- **TTS = qwen3-tts on amd1** (`http://amd1:8064`, override with `TTS_URL`); `/mono` POST `{text,ref1,seed}`. `--ref` paths are inside the TTS server's `/work` (e.g. `/work/ref1.wav`), not host paths.
- **Vision QA = qwen9b on amd0** (`https://amd0.python-bull.ts.net/v1`, override `QWEN_URL`); requests MUST set `chat_template_kwargs:{enable_thinking:false}` or `content` comes back empty (all in `reasoning_content`). See `review.py`.
- Run PIL/numpy/cv2 scripts (`broadcast_gfx.py`, `make_screen.py`, etc.) in the **mp-extract** container.
- `broadcast_gfx.py` must run before `broadcast_finish.sh` — finish reads `output/gfx_*.png` and errors otherwise.
- **Network name is `NNS` (National News Service)** — canonical, `catalog/brands.json`. Display = `NNS`; spoken/TTS text = `N N S` (spaced). Logo `newscast/assets/nns_logo.png` (the `ffnn_*` asset is dead).
- `factory.py`/`build_reporter_pkg.py` drive `make_anchor.sh` on the HOST (not in a container).

## Related
- Upstream render: `render/` + the LTX/SDNQ farm (`sdnq_ltx`), `make_anchor.sh`, and `swap/` (face swap + MuseTalk servers) — the anchor clips this layer narrates and dresses.
- TTS = qwen3-tts voxserver on **amd1** (`http://amd1:8064`, env `TTS_URL`). Old local impl in `podcastfy/qwen3dia/` is retired.
- Global rules (sm_70, never `kill -9` CUDA, never rewrite locked code, outputs→`output/`) inherited from studio/AGENTS.md.
