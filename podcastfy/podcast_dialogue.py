#!/usr/bin/env python3
"""
podcast_dialogue.py — podcast generator with CONSISTENT cloned voices.

  Transcript (LLM) : DeepSeek V4 Flash via podcastfy
  Audio (TTS)      : resident Qwen3-TTS two-speaker DIALOGUE server (Wan2GP official
                     impl, container `qwen3dia`, :8064) — renders the whole
                     conversation in one continuous pass so each speaker's voice
                     stays consistent (no per-line clone drift).

Usage:
  ./podcast_dialogue.py --text "..."     --out my_episode
  ./podcast_dialogue.py --url https://...  --out my_episode
  ./podcast_dialogue.py --topic "the history of espresso"
  ./podcast_dialogue.py --transcript output/transcripts/xxx.txt --out my_episode

Voices clone two reference clips on the server side (defaults ref1.wav/ref2.wav in
the qwen3dia work dir = cast_kai / cast_vivian). Override with --ref1/--ref2 (paths
as seen *inside* the qwen3dia container, e.g. /work/ref1.wav).
"""
import argparse
import os
import re
import subprocess
import sys
import urllib.request

os.environ.setdefault("OPENAI_BASE_URL", "http://localhost:8062/v1")  # unused but keeps podcastfy happy
os.environ.setdefault("OPENAI_API_KEY", "unused")
if "DEEPSEEK_API_KEY" not in os.environ:
    sys.exit("ERROR: DEEPSEEK_API_KEY not set")

import langsmith.client as _ls  # noqa: E402
_ls._validate_public_prompt_pull = lambda *a, **k: None
from podcastfy.client import generate_podcast  # noqa: E402

DIALOGUE_URL = os.environ.get("PODCASTFY_DIALOGUE_URL", "http://amd1:8064/dialogue")  # qwen3-tts voxserver on amd1


def transcript_to_dialogue(tagged):
    """<Person1>..</Person1><Person2>..</Person2>  ->  Speaker 1: ..\nSpeaker 2: .."""
    lines = []
    # Bare stage-words the transcript LLM occasionally leaks (Qwen reads them literally).
    stage = re.compile(r"^\s*\(?(laughing|laughs|chuckles?|chuckling|sighs?|sighing|"
                       r"gasps?|whispering|whispers|giggles?|scoffs?)\)?[\s,:-]+", re.I)
    for who, body in re.findall(r"<Person([12])>(.*?)</Person[12]>", tagged, re.DOTALL):
        t = re.sub(r"<[^>]+>", " ", body)          # strip stray ssml
        t = re.sub(r"\[[^\]]*\]|\*[^*]*\*", " ", t)  # strip [stage] / *stage* directions
        t = stage.sub("", t)                          # strip a leading bare stage-word
        t = re.sub(r"\s+", " ", t).strip()
        if t:
            lines.append(f"Speaker {who}: {t}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", action="append", dest="urls")
    src.add_argument("--text")
    src.add_argument("--topic")
    src.add_argument("--transcript", help="existing <Person1>/<Person2> transcript file")
    ap.add_argument("--out", default="podcast")
    ap.add_argument("--outdir", default="output")
    ap.add_argument("--llm-model", default="deepseek/deepseek-v4-flash")
    ap.add_argument("--ref1", default="/work/ref1.wav", help="Speaker 1 ref (path inside qwen3dia)")
    ap.add_argument("--ref2", default="/work/ref2.wav", help="Speaker 2 ref (path inside qwen3dia)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--temperature", type=float, default=0.85,
                    help="TTS sampling temp: lower=flatter/robotic, higher=more expressive but less stable (0.85 default)")
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--rep-penalty", type=float, default=None)
    ap.add_argument("--flat", action="store_true",
                    help="don't add the expressive-writing instruction to the transcript LLM")
    ap.add_argument("--longform", action="store_true")
    ap.add_argument("--no-master", action="store_true", help="skip per-turn leveling + loudnorm")
    ap.add_argument("--chunk-turns", type=int, default=6,
                    help="turns per TTS request; smaller avoids codec-decode OOM on long episodes")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 1) transcript via DeepSeek (podcastfy, transcript-only)
    if args.transcript:
        tagged = open(args.transcript, encoding="utf-8").read()
        print(f"Using transcript: {args.transcript}")
    else:
        print(f"LLM : {args.llm_model} (DeepSeek) — generating transcript ...")
        # The Qwen3 clone has no emotion tags — emotion rides on word choice and
        # punctuation, so steer the transcript LLM to write *performable* lines.
        EXPRESSIVE = (
            "Write the dialogue to be performed aloud with genuine emotion. Convey "
            "feeling through word choice, rhythm, and punctuation — not stage directions "
            "or bracketed tags (a TTS reads those literally). Use natural interjections "
            "(oh, wow, hmm, look), contractions, em dashes for interruptions, ellipses "
            "for hesitation, and the occasional emphatic short sentence. Vary energy and "
            "pacing between lines; let the two speakers react to each other. Avoid ALL-CAPS "
            "and emojis."
        )
        cc = {"text_to_speech": {"default_tts_model": "edge"}}
        if not args.flat:
            cc["user_instructions"] = EXPRESSIVE
            cc["conversation_style"] = ["expressive", "warm", "natural", "dynamic"]
        tpath = generate_podcast(
            urls=args.urls, text=args.text, topic=args.topic,
            transcript_only=True, longform=args.longform,
            llm_model_name=args.llm_model, api_key_label="DEEPSEEK_API_KEY",
            conversation_config=cc,
        )
        tagged = open(tpath, encoding="utf-8").read()

    dialogue = transcript_to_dialogue(tagged)
    if not dialogue:
        sys.exit("ERROR: transcript produced no Speaker lines")
    dpath = os.path.join(args.outdir, f"{args.out}.dialogue.txt")
    open(dpath, "w", encoding="utf-8").write(dialogue)
    print(f"Dialogue ({dialogue.count(chr(10))+1} turns) -> {dpath}")

    # 2) render on the resident server. The codec decodes a whole request's audio at
    #    once, so a long episode OOMs the 16GB V100 — split into chunks of turns (the
    #    fixed refs keep both voices consistent across chunks) and concatenate.
    import json
    import io
    import numpy as np
    import soundfile as sf

    lines = dialogue.split("\n")

    def both_speakers(ls):
        return any(l.startswith("Speaker 1") for l in ls) and any(l.startswith("Speaker 2") for l in ls)

    chunks, cur = [], []
    for ln in lines:
        cur.append(ln)
        if len(cur) >= args.chunk_turns and both_speakers(cur):
            chunks.append(cur)
            cur = []
    if cur:
        if chunks and not both_speakers(cur):
            chunks[-1] += cur          # avoid a trailing single-speaker chunk
        else:
            chunks.append(cur)
    chunk_texts = ["\n".join(c) for c in chunks]
    print(f"TTS : {DIALOGUE_URL}  ({len(chunk_texts)} chunk(s), Qwen3-TTS two-speaker clone) — rendering ...")

    def render_chunk(text):
        payload = {"text": text, "ref1": args.ref1, "ref2": args.ref2,
                   "seed": args.seed, "temperature": args.temperature}
        if args.top_p is not None:
            payload["top_p"] = args.top_p
        if args.rep_penalty is not None:
            payload["repetition_penalty"] = args.rep_penalty
        req = urllib.request.Request(DIALOGUE_URL, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=1800) as r:
            return r.read()

    raw_dest = os.path.join(args.outdir, f"{args.out}.raw.wav")
    if len(chunk_texts) == 1:
        open(raw_dest, "wb").write(render_chunk(chunk_texts[0]))
    else:
        parts, sr = [], 24000
        for i, ct in enumerate(chunk_texts, 1):
            print(f"  chunk {i}/{len(chunk_texts)} ...", flush=True)
            a, sr = sf.read(io.BytesIO(render_chunk(ct)), dtype="float32")
            if a.ndim > 1:
                a = a.mean(1)
            parts.append(a)
            parts.append(np.zeros(int(sr * 0.4), dtype=np.float32))  # gap between chunks
        sf.write(raw_dest, np.concatenate(parts), sr)

    # Master: per-turn loudness leveling (Qwen3 decodes each turn independently, so
    # raw output swings ~14 dB turn-to-turn) + loudnorm to -16 LUFS / -1.5 dBTP.
    wav_dest = os.path.join(args.outdir, f"{args.out}.wav")
    if args.no_master:
        os.replace(raw_dest, wav_dest)
    else:
        from master_audio import mastered
        mastered(raw_dest, wav_dest)
        os.remove(raw_dest)
    print(f"Audio (wav) -> {wav_dest}")

    mp3_dest = os.path.join(args.outdir, f"{args.out}.mp3")
    rc = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", wav_dest,
                         "-c:a", "libmp3lame", "-b:a", "192k", mp3_dest]).returncode
    print(f"Audio (mp3) -> {mp3_dest}" if rc == 0 else "(mp3 transcode failed)")


if __name__ == "__main__":
    main()
