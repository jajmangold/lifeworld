# Podcastfy (on-prem)

> **TTS endpoint (2026-07):** canonical qwen3-tts is now `http://amd1:8064` (env `PODCASTFY_TTS_URL`/`PODCASTFY_DIALOGUE_URL`). The local `qwen3dia` container was DECOMMISSIONED (2026-07-04) — the docker instructions below are historical; use amd1. Recreate locally only if amd1 is unreachable. See `studio/AGENTS.md` → Shared network services.


[Podcastfy](https://github.com/souzatharsis/podcastfy) wired to this machine's
resident model services — no cloud TTS, no OpenAI/Gemini keys needed.

| Stage | Backend | Where |
|-------|---------|-------|
| Transcript (LLM) | **DeepSeek V4 Flash** | `api.deepseek.com`, key in `$DEEPSEEK_API_KEY` |
| Audio (TTS) | **Qwen3-TTS-1.7B** (`crispasr-tts`) | OpenAI-compatible server on `localhost:8062` |

`podcast.py` is a thin wrapper that points podcastfy's `openai` TTS provider at
the local crispasr-tts server and its LLM at DeepSeek via LiteLLM.

## Setup (already done)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv podcastfy playwright
.venv/bin/python -m playwright install chromium
```

Requires the `crispasr-tts` container running (`docker compose up -d` in
`/srv/nvme-data/containers/projects/CrispASR`) and `DEEPSEEK_API_KEY` in the env.

## Usage

```bash
# From raw text
.venv/bin/python podcast.py --text "..." --out my_episode

# From one or more web pages
.venv/bin/python podcast.py --url https://example.com/article --out my_episode

# From a topic (LLM uses its own knowledge)
.venv/bin/python podcast.py --topic "the history of espresso"

# Longer, multi-chunk conversation
.venv/bin/python podcast.py --url https://... --longform

# Just the script, no audio
.venv/bin/python podcast.py --url https://... --transcript-only

# Re-synthesize an edited transcript
.venv/bin/python podcast.py --transcript output/my_episode.txt
```

Outputs land in `output/`: `<name>.wav` (from the 24 kHz WAV-only TTS server)
and `<name>.mp3` (transcoded here). Transcripts also persist under
`output/transcripts/`.

### Voices

Person1/Person2 map to crispasr-tts voices via `--voice1` / `--voice2`
(defaults `cast_kai` / `cast_brooke`). List what the server has:

```bash
curl -s localhost:8062/v1/voices | python3 -m json.tool
```

Current cast voices: `cast_brooke`, `cast_kai`, `cast_hank`, `cast_vivian`,
`cast_marisol` (plus the `washer_*` / `ref_*` reference clones).

## How it's wired (notes)

- The LLM uses `ChatLiteLLM(model="deepseek/deepseek-v4-flash")` with
  `api_key_label="DEEPSEEK_API_KEY"`; LiteLLM's deepseek provider targets
  `api.deepseek.com` automatically.
- Podcastfy pulls prompt templates from LangChain Hub; newer `langsmith`
  blocks public-prompt pulls, so `podcast.py` disables that guard (the templates
  are podcastfy's own `souzatharsis/podcastfy_*`).
- `audio_format` is forced to **wav** — crispasr-tts only emits WAV, and
  podcastfy re-reads each segment using the configured format, so they must match.

### The TTS runaway saga (important)

Qwen3-TTS on this V100/SM70 box has a **no-end-of-speech "runaway"**: instead of
stopping, it generates until a frame cap, producing ~120s of garbage for a short
line. Getting podcastfy's audio clean required peeling back several layers:

1. **Connection reuse is the real bug.** The crispasr server leaks generation
   state across requests sent over the *same* keep-alive HTTP connection — the
   2nd+ request on a pooled connection runs away. Podcastfy's `openai` provider
   uses the OpenAI SDK, which pools connections (httpx), so it cascaded into
   ~70% runaways. **Fix:** `podcast.py` overrides `OpenAITTS.generate_audio` to
   POST each fragment over a **fresh connection** (`Connection: close`, raw
   urllib) instead of the SDK. This alone removes the cascade.
2. **One request per whole speaker turn** (not per sentence). Independent
   per-sentence requests make the cloned voice drift ("every sentence a slightly
   different person"); synthesizing the entire turn in one continuous generation
   keeps the timbre stable across it. If a whole turn runs away, it falls back to
   bounded per-sentence/clause synthesis for that turn only.
3. **Avoid `cast_brooke`.** Its reference clip *independently* triggers runaways
   even with fresh connections; `cast_kai`/`cast_vivian` are robust. (`reftest`
   and `washer_b` reference clips error out.)
4. **Smart-punctuation normalization.** Curly quotes / em-dashes / ellipses are
   ASCII-folded before synthesis.
5. **Backstops** (rarely needed now): a server-side frame cap
   (`QWEN3_TTS_MAX_FRAMES=220`, set on the container) bounds any stray runaway to
   ~17.6s instead of 120s; the shim retries a runaway fragment with new seeds and,
   as a last resort, restarts the `crispasr-tts` container to clear dirty state.

Tunables (env vars): `PODCASTFY_TTS_URL`, `PODCASTFY_TTS_TEMPERATURE`,
`PODCASTFY_TTS_SEED`, `PODCASTFY_TTS_MAX_WORDS`, `PODCASTFY_TTS_RETRIES`,
`PODCASTFY_TTS_RUNAWAY_S`, `PODCASTFY_TTS_CONTAINER`, `PODCASTFY_TTS_MAX_RESTARTS`.

---

## Consistent-voice path: Qwen3-TTS two-speaker DIALOGUE (recommended)

The crispasr per-line path drifts ("different person each sentence") because each
utterance is an independent zero-shot clone. The fix is to render the WHOLE
conversation in one continuous pass with the **official** Qwen3-TTS (Wan2GP impl),
which natively clones two speakers and keeps each voice consistent.

**Backend:** `qwen3dia` container (image `wan2gp-volta:1.0`, pinned to a real V100 —
NOT a CMP card; CMP PCIe is too slow). Loads `Qwen3-TTS-12Hz-1.7B-Base` once and
serves `POST /dialogue` on host port **8064**. Source in `qwen3dia/`
(`server.py` + `dialogue_tts.py`); weights in `qwen3dia/ckpts`.

  docker run ... wan2gp-volta:1.0 -c 'python3 /work/server.py'   # see qwen3dia/ for env

Key env: `QWEN3_FP16=1` (fp16 — V100 has fp16 tensor cores but NOT bf16, so bf16 is
~10x slower and CMP cards are unusably slow), `QWEN3_ENGINE=cg` (cudagraph decode).
Speed ~1.2-1.6x real-time on V100 (~3 min for a 2.5-min episode).

**Generate a podcast:**

  .venv/bin/python podcast_dialogue.py --topic "the history of espresso"
  .venv/bin/python podcast_dialogue.py --url https://... --out my_episode
  .venv/bin/python podcast_dialogue.py --transcript output/transcripts/xxx.txt --out my_episode

Pipeline: DeepSeek V4 Flash writes the transcript (podcastfy, transcript-only) ->
converted to `Speaker 1:/Speaker 2:` -> one-shot dialogue render on :8064 ->
`output/<name>.wav` + `.mp3`. Voices clone `qwen3dia/ref1.wav` (Speaker 1) and
`ref2.wav` (Speaker 2); override with `--ref1/--ref2` (paths inside the container).

`podcast.py` (crispasr per-line, fast but drifts) is kept as the fallback backend.

### Swapping the cloned voices (VCTK)

Reference clips for the dialogue path live in `qwen3dia/ref1.wav` (Speaker 1) and
`ref2.wav` (Speaker 2), each with a sibling `.txt` transcript the server auto-uses
for better clone alignment. Current voices: **p246** (M, Scottish) / **p294** (F, American),
pulled from VCTK with `qwen3dia/fetch_vctk_refs.py`.

```bash
cd qwen3dia
.venv/bin/python fetch_vctk_refs.py --list        # speakers available in the shard
.venv/bin/python fetch_vctk_refs.py p246 p302     # pick two (S1 S2); writes ref1/ref2 + .txt
SHARD=audio/train-03.tar .venv/bin/python fetch_vctk_refs.py   # other shards = other speakers
docker restart qwen3dia                            # (optional) clears warm state
```

(`.cast_bak` files are the original cast_kai/cast_vivian refs if you want them back.)
Or just drop your own `ref1.wav`/`ref2.wav` (+ optional `.txt`) in `qwen3dia/`.

### Output normalization (mastering)

Qwen3-TTS decodes each turn independently, so raw output drifts ~14 dB in loudness
turn-to-turn. `master_audio.py` fixes this: split on inter-turn pauses → level each
turn to a common RMS (capped gain) → `loudnorm` to -16 LUFS / -1.5 dBTP (+55 Hz
highpass). Measured: per-turn RMS spread 16.9 dB → 2.4 dB (std 0.5). `podcast_dialogue.py`
applies it by default; `--no-master` to skip. Standalone: `master_audio.py in.wav out.wav`.

### Prompt tiling (experimental, OFF — not recommended)

`--tile` / `QWEN3_TILE=1` / `{"tile":true}` extends each speaker's ICL prompt with
their prior generated turns (the base model already decodes `[ref_code | generated]`,
so this just grows the prefix). In testing this **destabilizes**: the base model
isn't trained for multi-turn ICL, so it loses the end-of-speech signal and runs away.
The single-reference dialogue mode + per-turn mastering already gives consistent
voices; leave tiling off. (Code in `dialogue_tts.generate_tiled`.)

### Emotion & expressiveness tuning

The Base clone model has **no emotion tags or instruction parameter** (maintainers
confirm — that's the unreleased VoiceEditing model; passing `instruct` to Base does
nothing, and inline `[happy]`-style tags get read literally). Emotion in a cloned
voice comes from three levers, in order of impact:

1. **The reference clip.** "A monotone reference produces monotone clones." Use a
   10–15 s clip with *varied* intonation (NOT flat elicitation). Refs are now 12–15 s
   with a 0.5 s tail silence (prevents first-token phoneme bleed). >15 s degrades and
   risks runaway/outbursts. Full ICL (audio + accurate `ref_text`, which we use via the
   sibling `.txt`) lifts speaker similarity ~0.75→0.89 vs x-vector-only.
   - For strong emotion, clone an *expressive* reference (e.g. an EARS emotional clip).
2. **The text.** The model performs what the words imply. `podcast_dialogue.py`
   instructs DeepSeek to write *performable* lines — emotion via word choice,
   interjections (oh/wow/hmm), em dashes (interruptions), ellipses (hesitation), and
   short emphatic sentences. `--flat` disables this.
3. **Sampling.** Defaults set to community-optimal: `temperature 0.85`, `top_p 0.9`,
   `repetition_penalty 1.08`, `max_new_tokens 1024` (caps infinite-loop "outbursts").
   - `--temperature` ↑ (→0.95) = more expressive/varied but less stable; ↓ (→0.7) =
     calmer/more consistent, can sound flat. `--top-p`, `--rep-penalty` also exposed.
   - Per request the server accepts `temperature/top_p/top_k/repetition_penalty`.

Pitfalls (from the community): >15 s or very long refs → generation hangs; high
`max_new_tokens` → random laughs/outbursts; specific accents (British/Australian)
drift toward American unless the reference itself has that accent; same sentence
twice can vary in timbre (zero-shot instability) — fixed seed helps.

For fully *directed* emotion ("say this excitedly") you'd need the **VoiceDesign**
variant (`Qwen3-TTS-...-VoiceDesign`, natural-language `instruct`, 7 emotion
dimensions) — but it doesn't clone a specific voice. A VoiceDesign→Clone pipeline is
the community trick for directed emotion in a chosen timbre; not wired up here yet.

### Designed voices (VoiceDesign → clone)

To set a voice's *character/baseline emotion* from a text description instead of a
VCTK clip, use the VoiceDesign→clone path: `qwen3dia/design_refs.py` renders a
reference clip from a natural-language description (via the Qwen3-TTS VoiceDesign
model), then the normal clone-dialogue pipeline reproduces that persona consistently.

```bash
# VoiceDesign needs the full V100 — it won't fit beside the running base server:
docker stop qwen3dia
docker run -d --name qwen3design --rm --runtime nvidia \
  -e NVIDIA_VISIBLE_DEVICES=<V100-uuid> -e TORCH_CUDA_ARCH_LIST=7.0 \
  -e HF_HOME=/work/ckpts/hf -e HOME=/root -e QWEN3_FP16=1 \
  -v /home/josh/archives/wan2gp:/workspace -v $PWD/qwen3dia:/work -w /work \
  --shm-size 8g --entrypoint bash wan2gp-volta:1.0 \
  -c 'python3 design_refs.py "A warm, upbeat young man, energetic, bright tone, medium-fast pace." "A calm, smooth woman in her thirties, articulate, warm low tone, unhurried."'
docker start qwen3dia          # base server reads the new ref1/ref2 on next request
```

Description tips (1–3 sentences): be specific — timbre (deep/crisp/bright/raspy/mellow),
pace (slow/fast), and baseline emotion (cheerful/calm/serious/lively/soothing). The
designed persona's baseline emotion is fixed by the description; line-level emotion
still comes from the dialogue text. Weights: `qwen3_tts_12hz_1b7_voicedesign_bf16.safetensors`
in `qwen3dia/ckpts`. (`.vctk_bak` files restore the previous VCTK refs.)

> True per-line directed emotion ("say *this line* angrily") needs the unreleased
> VoiceEditing model; designed personas + expressive text is the best available now.

#### Designing *from* an existing clip

`qwen3dia/design_from_clip.py` analyzes a wav (pitch / brightness / pace), writes a
VoiceDesign description of its character, optionally adds an emotion, and renders a
new ref. Use it to take a VCTK voice you like and give it directed emotion.

```bash
python3 design_from_clip.py clip.wav 1 --describe-only            # preview the description
python3 design_from_clip.py ref1.wav.vctk_bak 1 --emotion "warm and enthusiastic"
python3 design_from_clip.py clip.wav 2 --gender female --emotion "calm, thoughtful"
```

Caveat: VoiceDesign is text-only, so this clones the clip's *described character*
(gender/pitch/brightness/pace) + your emotion — NOT its exact timbre. For exact
timbre, clone the clip directly (no emotion design). Runs in the same VoiceDesign
one-shot container as `design_refs.py` (server stopped).

#### Current installed voices: newscasters

The active `ref1`/`ref2` are VoiceDesign-built news anchors:
- **ref1 (Speaker 1):** "deep, resonant male, mature/authoritative, calm and trustworthy, smooth broadcast tone, measured pace, prime-time news anchoring"
- **ref2 (Speaker 2):** "warm, clear female, polished/articulate, confident and trustworthy, smooth broadcast tone, even measured pace, prime-time news anchoring"

Read sentence for the design was news-style (set via `QWEN3_REF_SENTENCE`) so the
captured prosody is anchor-like. Previous designed/VCTK refs saved as `.prev`/`.vctk_bak`.

> **Input modes:** use `--text` (paste source content), `--url`, or `--transcript`.
> `--topic` is NOT supported here — podcastfy hardcodes Google Gemini for topic
> grounding (needs a Gemini key); our stack is DeepSeek-only. For a "topic", just
> paste a few sentences of source material as `--text`.

#### Long episodes (codec-decode OOM fix)

The codec decodes a whole request's audio at once, so a long episode (~15+ turns)
OOMs the 16GB V100 at decode. `podcast_dialogue.py` therefore renders in chunks of
`--chunk-turns` (default 6) turns and concatenates — the fixed refs keep both voices
consistent across chunks. The server also runs with `PYTORCH_CUDA_ALLOC_CONF=
expandable_segments:True`. Mastering pins output to 24kHz (loudnorm otherwise
upsamples to 192kHz).

### More voices: LibriTTS-R narrators

`qwen3dia/fetch_libritts_refs.py` pulls clone refs from LibriTTS-R (audiobook
narrators — more natural/professional delivery than VCTK read-prompts). Uses the
OpenSLR dev-clean tar (~40 speakers) cached at `qwen3dia/ckpts/libritts/dev_clean.tar.gz`.

```bash
python3 fetch_libritts_refs.py --list           # narrators + measured pitch (male?/female?)
python3 fetch_libritts_refs.py 1272 5338         # speaker1 -> ref1, speaker2 -> ref2
python3 fetch_libritts_refs.py --auto            # auto-pick a deep male + warm female
```
(For a bigger pool, download train_clean_100 from OpenSLR/141 into the same dir.)

### Combining clones (blend two voices into a new one)

The model encodes each voice as a speaker x-vector; in x-vector-only mode that
embedding alone defines identity. The server `/blend` endpoint interpolates two
clips' embeddings (`alpha*A + (1-alpha)*B`) and synthesizes a NEW hybrid voice —
a novel speaker, not an identifiable real person. Verified as a true interpolation
(e.g. 92 Hz male + 202 Hz female, alpha 0.5 -> 170 Hz).

```bash
# inside the qwen3dia container's reach (host curl to the server):
curl -s -X POST amd1:8064/blend -H 'Content-Type: application/json' \
  -d '{"ref_a":"/work/ref1.wav","ref_b":"/work/ref2.wav","alpha":0.5,"out":"/work/ref1.wav"}' \
  -o /tmp/blend_preview.wav
# alpha 1.0 = pure A, 0.0 = pure B, 0.5 = even blend. "out" installs it as a ref.
```
Tip: blend two *same-gender* voices for a natural new persona; the saved clip is
then full-ICL cloned by the dialogue pipeline like any ref. (`dialogue_tts.blend_voice`.)
