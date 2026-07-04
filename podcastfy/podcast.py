#!/usr/bin/env python3
"""
podcast.py — Podcastfy wired to this machine's resident services.

  LLM (transcript)  : DeepSeek V4 Flash   (api.deepseek.com, key in $DEEPSEEK_API_KEY)
  TTS (audio)       : resident Qwen3-TTS   (crispasr-tts, OpenAI-compatible @ :8062)

The crispasr-tts server speaks the OpenAI /v1/audio/speech API but only emits WAV,
so we force audio_format=wav inside podcastfy and transcode the final mix to mp3 here.

Usage:
  ./podcast.py --url https://example.com/article
  ./podcast.py --text "raw text to turn into a podcast"
  ./podcast.py --topic "the history of espresso"
  ./podcast.py --transcript-only --url ...        # skip audio, just write the script
  ./podcast.py --longform --url ...               # longer, multi-chunk conversation

Flags:
  --voice1 / --voice2   cast voice names (default cast_kai / cast_brooke)
  --out NAME            base name for outputs (default: podcast)
  --llm-model MODEL     litellm model id (default deepseek/deepseek-v4-flash)
"""
import argparse
import os
import shutil
import subprocess
import sys

# ── Resident TTS endpoint (must be set before podcastfy/openai import) ─────────
TTS_BASE_URL = os.environ.get("PODCASTFY_TTS_URL", "http://amd1:8064/v1")  # qwen3-tts voxserver on amd1 (OpenAI /v1)
os.environ.setdefault("OPENAI_BASE_URL", TTS_BASE_URL)
os.environ.setdefault("OPENAI_API_KEY", "local-crispasr")  # server ignores it

if "DEEPSEEK_API_KEY" not in os.environ:
    sys.exit("ERROR: DEEPSEEK_API_KEY is not set in the environment.")

# Podcastfy pulls its prompt templates from LangChain Hub via hub.pull(). Newer
# langsmith clients refuse public-prompt pulls unless dangerously_pull_public_prompt
# is set, which podcastfy never passes. These are podcastfy's own published
# templates (souzatharsis/podcastfy_*), so disable the guard.
import langsmith.client as _ls_client  # noqa: E402
_ls_client._validate_public_prompt_pull = lambda *a, **k: None

from podcastfy.client import generate_podcast  # noqa: E402

# ── Per-fragment TTS shim with runaway guard ──────────────────────────────────
# crispasr-tts (Qwen3-TTS, V100/SM70) sporadically fails to emit end-of-speech and
# "runs away" to its frame cap (120s of garbage) — sampling-dependent and more
# likely on long or casually-punctuated text. Podcastfy hands the provider whole
# multi-sentence speaker turns, so we override OpenAITTS.generate_audio to:
#   1. split each turn into sentences, and further into clauses when very long;
#   2. synthesize one fragment per request with a hard max_speech_tokens cap so a
#      runaway returns quickly (capped) instead of stalling for ~a minute;
#   3. detect an over-length result (>> the fragment's expected duration) and retry
#      with a different seed, keeping the shortest take;
#   4. concatenate the WAV fragments.
import io  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
import urllib.request  # noqa: E402
import wave  # noqa: E402
from podcastfy.tts.providers.openai import OpenAITTS  # noqa: E402

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CLAUSE_SPLIT = re.compile(r"(?<=[,;:—-])\s+")
TTS_TEMPERATURE = float(os.environ.get("PODCASTFY_TTS_TEMPERATURE", "0.7"))
TTS_SEED = int(os.environ.get("PODCASTFY_TTS_SEED", "42"))
TTS_MAX_WORDS = int(os.environ.get("PODCASTFY_TTS_MAX_WORDS", "22"))  # clause-split above this
TTS_RETRIES = int(os.environ.get("PODCASTFY_TTS_RETRIES", "3"))
# The server caps generation at QWEN3_TTS_MAX_FRAMES (set to 220 ≈ 17.6s on the
# container) so a runaway returns clipped at ~17.6s instead of grinding to 120s.
# Anything at/above this threshold is treated as a (rare) runaway and retried.
TTS_RUNAWAY_S = float(os.environ.get("PODCASTFY_TTS_RUNAWAY_S", "16.5"))
# Recovery: a persistent runaway means the server's CUDA/graph state has gone dirty
# (SM70) — seed retries can't escape it, but a container restart reliably clears it.
TTS_CONTAINER = os.environ.get("PODCASTFY_TTS_CONTAINER", "crispasr-tts")
TTS_MAX_RESTARTS = int(os.environ.get("PODCASTFY_TTS_MAX_RESTARTS", "6"))
_restarts_used = [0]


def _tts_healthy():
    base = TTS_BASE_URL.rsplit("/v1", 1)[0]
    try:
        with urllib.request.urlopen(base + "/health", timeout=3) as r:
            return b'"ok"' in r.read()
    except Exception:
        return False


def _restart_tts_server():
    """Restart the resident TTS container to clear dirty generation state."""
    if not TTS_CONTAINER or _restarts_used[0] >= TTS_MAX_RESTARTS:
        return False
    _restarts_used[0] += 1
    logger_print(f"    [recover] restarting {TTS_CONTAINER} "
                 f"({_restarts_used[0]}/{TTS_MAX_RESTARTS}) to clear runaway state...")
    subprocess.run(["docker", "restart", TTS_CONTAINER],
                   capture_output=True, text=True)
    deadline = time.time() + 300
    while time.time() < deadline:
        if _tts_healthy():
            time.sleep(2)  # let it settle past the first request
            logger_print("    [recover] server ready")
            return True
        time.sleep(5)
    logger_print("    [recover] server did NOT come back healthy")
    return False


def _normalize(text):
    """ASCII-fold smart punctuation that the TTS tokenizer handles poorly."""
    return (text.replace("’", "'").replace("‘", "'")
                .replace("“", '"').replace("”", '"')
                .replace("—", " - ").replace("–", " - ")
                .replace("…", "...").strip())


def _fragments(text):
    """Split a speaker turn into sentences, then clause-split overlong sentences."""
    text = _normalize(re.sub(r"\s+", " ", text.strip()))
    if not text:
        return []
    out = []
    for sent in _SENT_SPLIT.split(text):
        sent = sent.strip()
        if not sent:
            continue
        if len(sent.split()) <= TTS_MAX_WORDS:
            out.append(sent)
            continue
        # Too long → clause-split, then greedily re-merge tiny fragments.
        clauses = [c.strip() for c in _CLAUSE_SPLIT.split(sent) if c.strip()]
        buf = ""
        for c in clauses:
            cand = (buf + " " + c).strip() if buf else c
            if len(cand.split()) > TTS_MAX_WORDS and buf:
                out.append(buf)
                buf = c
            else:
                buf = cand
        if buf:
            out.append(buf)
    return out or [text]


def _wav_duration(blob):
    with wave.open(io.BytesIO(blob)) as w:
        return w.getnframes() / float(w.getframerate())


_SPEECH_URL = TTS_BASE_URL.rstrip("/") + "/audio/speech"


def _synth_fragment(voice, model, text, seed):
    # IMPORTANT: a fresh connection per request ("Connection: close"). The crispasr
    # server leaks generation state across requests on a reused (keep-alive) HTTP
    # connection, which makes Qwen3-TTS run away (no end-of-speech) on the 2nd and
    # later requests. The openai SDK pools connections, so we bypass it and post
    # directly, forcing the connection closed each time. This is THE fix for the
    # cascade of 17.6s runaways; everything else (cap, retry, restart) is a backstop.
    body = json.dumps({
        "model": model, "voice": voice, "input": text,
        "temperature": TTS_TEMPERATURE, "seed": seed, "spoken_disclaimer": False,
    }).encode()
    req = urllib.request.Request(
        _SPEECH_URL, data=body,
        headers={"Content-Type": "application/json", "Connection": "close"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def _synth_guarded(voice, model, text):
    # A fragment near the server's frame cap never emitted end-of-speech (runaway).
    best = None

    def attempt(seed):
        nonlocal best
        blob = _synth_fragment(voice, model, text, seed=seed)
        dur = _wav_duration(blob)
        if best is None or dur < best[0]:
            best = (dur, blob)
        return blob if dur < TTS_RUNAWAY_S else None

    # Round 1: seed retries — escapes transient runaways cheaply.
    for k in range(TTS_RETRIES + 1):
        blob = attempt(TTS_SEED + k * 101)
        if blob is not None:
            return blob
        logger_print(f"    runaway ({best[0]:.1f}s) retry {k+1}/{TTS_RETRIES}: {text[:50]!r}")

    # Round 2: a persistent runaway = dirty server state. Restart to clear it, then
    # retry on the fresh server (reliably clean), up to the restart budget.
    while _restart_tts_server():
        for k in range(2):
            blob = attempt(TTS_SEED + k * 101)
            if blob is not None:
                return blob
        logger_print(f"    still runaway after restart: {text[:50]!r}")

    logger_print(f"    GIVING UP (using {best[0]:.1f}s capped take): {text[:50]!r}")
    return best[1]  # least-bad take


def logger_print(msg):
    print(msg, flush=True)


def _concat_wavs(blobs):
    frames, params = [], None
    for b in blobs:
        with wave.open(io.BytesIO(b)) as w:
            if params is None:
                params = w.getparams()
            frames.append(w.readframes(w.getnframes()))
    out = io.BytesIO()
    with wave.open(out, "wb") as w:
        w.setnchannels(params.nchannels)
        w.setsampwidth(params.sampwidth)
        w.setframerate(params.framerate)
        for f in frames:
            w.writeframes(f)
    return out.getvalue()


def _turn_budget(text):
    """Generous upper bound (seconds) on a whole turn's plausible spoken length.
    A runaway inflates the turn by ~17.6s per stuck internal chunk, so anything
    well past this is treated as a runaway and re-done sentence-by-sentence."""
    return max(TTS_RUNAWAY_S, len(text.split()) / 1.8 + 6.0)


def _generate_audio(self, text, voice, model, voice2=None):
    # Synthesize the WHOLE speaker turn in ONE request so the cloned voice stays
    # continuous across the turn's sentences (independent per-sentence requests
    # drift in timbre — "every sentence a slightly different person"). Fall back to
    # bounded per-sentence synthesis only if the whole turn runs away.
    turn = _normalize(re.sub(r"\s+", " ", text.strip()))
    if not turn:
        raise ValueError("Text cannot be empty")

    blob = _synth_fragment(voice, model, turn, TTS_SEED)
    if _wav_duration(blob) <= _turn_budget(turn):
        return blob

    logger_print(f"  turn runaway; per-sentence fallback: {turn[:50]!r}")
    frags = _fragments(turn)
    return _concat_wavs([_synth_guarded(voice, model, f) for f in frags])


OpenAITTS.generate_audio = _generate_audio


def voices_listed():
    import urllib.request, json
    try:
        with urllib.request.urlopen(f"{TTS_BASE_URL}/voices", timeout=5) as r:
            return [v["name"] for v in json.load(r)["voices"]]
    except Exception:
        return []


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", action="append", dest="urls", help="source URL (repeatable)")
    src.add_argument("--text", help="raw text input")
    src.add_argument("--topic", help="generate about a topic (LLM uses its own knowledge)")
    src.add_argument("--transcript", help="path to an existing <Person1>/<Person2> transcript")

    # cast_kai (M) + cast_vivian (F): both robust against the Qwen3-TTS no-EOS
    # runaway in testing. Avoid cast_brooke — its reference clip reliably triggers it.
    ap.add_argument("--voice1", default="cast_kai", help="Person1 voice (default cast_kai)")
    ap.add_argument("--voice2", default="cast_vivian", help="Person2 voice (default cast_vivian)")
    ap.add_argument("--out", default="podcast", help="output base name (default: podcast)")
    ap.add_argument("--outdir", default="output", help="output directory")
    ap.add_argument("--llm-model", default="deepseek/deepseek-v4-flash")
    ap.add_argument("--transcript-only", action="store_true")
    ap.add_argument("--longform", action="store_true")
    ap.add_argument("--name", default="The Resident Podcast", help="podcast_name")
    ap.add_argument("--tagline", default="Generated on-prem with DeepSeek + Qwen3-TTS")
    args = ap.parse_args()

    avail = voices_listed()
    if avail:
        for v in (args.voice1, args.voice2):
            if v not in avail:
                sys.exit(f"ERROR: voice '{v}' not on the TTS server. Available: {', '.join(avail)}")

    os.makedirs(args.outdir, exist_ok=True)

    # Podcastfy conversation config. audio_format MUST be wav: the OpenAI provider
    # writes the server's raw bytes to a temp file then re-reads it with this same
    # format, and crispasr-tts only produces WAV.
    conversation_config = {
        "podcast_name": args.name,
        "podcast_tagline": args.tagline,
        "output_language": "English",
        "text_to_speech": {
            "default_tts_model": "openai",
            "audio_format": "wav",
            "ending_message": "Thanks for listening.",
            "output_directories": {
                "transcripts": os.path.join(args.outdir, "transcripts"),
                "audio": os.path.join(args.outdir, "audio"),
            },
            "openai": {
                "default_voices": {"question": args.voice1, "answer": args.voice2},
                "model": "tts-1",  # arbitrary; crispasr-tts uses its loaded backend
            },
        },
    }

    print(f"LLM : {args.llm_model}  (DeepSeek API)")
    print(f"TTS : {TTS_BASE_URL}  voices={args.voice1}/{args.voice2}")

    result = generate_podcast(
        urls=args.urls,
        text=args.text,
        topic=args.topic,
        transcript_file=args.transcript,
        tts_model="openai",
        transcript_only=args.transcript_only,
        longform=args.longform,
        conversation_config=conversation_config,
        llm_model_name=args.llm_model,
        api_key_label="DEEPSEEK_API_KEY",
    )

    if args.transcript_only:
        dest = os.path.join(args.outdir, f"{args.out}.txt")
        shutil.copy(result, dest)
        print(f"\nTranscript -> {dest}")
        return

    wav_dest = os.path.join(args.outdir, f"{args.out}.wav")
    shutil.copy(result, wav_dest)
    print(f"\nAudio (wav) -> {wav_dest}")

    # Transcode to a shareable mp3.
    mp3_dest = os.path.join(args.outdir, f"{args.out}.mp3")
    rc = subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", wav_dest,
         "-c:a", "libmp3lame", "-b:a", "192k", mp3_dest]
    ).returncode
    if rc == 0:
        print(f"Audio (mp3) -> {mp3_dest}")
    else:
        print("(mp3 transcode failed; wav is available)")


if __name__ == "__main__":
    main()
