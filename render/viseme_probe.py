#!/usr/bin/env python3
"""Viseme probe: what lip shape SHOULD the mouth be making, when — vs what A2F is doing.

Runs a wav2vec2 PHONEME recognizer on a line's audio to get phonemes + timestamps, maps them
to viseme classes (open 'ah', round 'oo', wide 'ee', closed 'm/b/p', lip-teeth 'f/v'), then
compares against the A2F arkit channels (jawOpen / mouthFunnel / mouthPucker / mouthClose).
Outputs a timeline PNG + a vowel-based lag estimate (the objective 'is it in sync?' number).

  python3 viseme_probe.py --wav line.wav --arkit line.arkit.json --out probe.png
"""
import os
os.environ.setdefault("HF_HOME", "/ts/hf-cache")
import argparse
import json
import numpy as np
import torch
import soundfile as sf
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

MODEL = "vitouphy/wav2vec2-xls-r-300m-phoneme"

# IPA-ish phoneme -> viseme class (+ which A2F channel should be active)
VIS = {"open": ("aɑʌɒæ", "jawOpen"), "round": ("oɔuʊw", "mouthPucker/Funnel"),
       "wide": ("iɪeɛjy", "(spread)"), "closed": ("mbp", "mouthClose"),
       "lipteeth": ("fv", "(F/V)")}
COL = {"open": "#e24", "round": "#28e", "wide": "#2a2", "closed": "#a0a",
       "lipteeth": "#fa0", "other": "#888"}


def classify(ph):
    base = ph.strip().lower()[:1]
    for k, (chars, _) in VIS.items():
        if base and base in chars:
            return k
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav", required=True)
    ap.add_argument("--arkit", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    wav, sr = sf.read(a.wav)
    if wav.ndim > 1:
        wav = wav.mean(1)
    wav = librosa.resample(wav.astype(np.float32), orig_sr=sr, target_sr=16000) if sr != 16000 else wav.astype(np.float32)
    dur = len(wav) / 16000.0

    proc = Wav2Vec2Processor.from_pretrained(MODEL)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL).eval()
    with torch.no_grad():
        logits = model(proc(wav, sampling_rate=16000, return_tensors="pt").input_values).logits[0]
    ids = logits.argmax(-1).numpy()
    step = dur / len(ids)                          # ~20ms per frame
    vocab = {v: k for k, v in proc.tokenizer.get_vocab().items()}
    blank = model.config.pad_token_id

    # CTC collapse -> phoneme segments with times
    segs = []
    prev = blank
    for i, t in enumerate(ids):
        if t != prev and t != blank:
            ph = vocab.get(int(t), "?")
            segs.append([ph, i * step, (i + 1) * step, classify(ph)])
        elif segs and t == prev and t != blank:
            segs[-1][2] = (i + 1) * step
        prev = t

    # expected open-ness from vowels (open=1, round=0.7, wide=0.5, else 0) at 50 fps
    fps = 50; n = int(dur * fps)
    exp = np.zeros(n)
    w = {"open": 1.0, "round": 0.7, "wide": 0.5}
    for ph, t0, t1, c in segs:
        if c in w:
            exp[int(t0 * fps):int(t1 * fps)] = w[c]
    # A2F jawOpen resampled to 50 fps
    d = json.load(open(a.arkit)); W = np.asarray(d["weights"], np.float32); afps = d["fps"]
    def ch(name):
        j = W[:, d["arkit_names"].index(name)] if name in d["arkit_names"] else np.zeros(len(W))
        return np.interp(np.arange(n) / fps, np.arange(len(j)) / afps, j)
    jaw = ch("jawOpen"); funnel = ch("mouthFunnel"); puck = ch("mouthPucker"); mclose = ch("mouthClose")

    # sustain the vowel signal (CTC marks brief onsets) so it's comparable to the jaw curve
    k = 7
    exp = np.convolve(np.pad(exp, k // 2, "edge"), np.ones(k) / k, "valid")[:n]
    # vowel-based lag, searched only within +-200ms (real sync range; avoids spurious far peaks)
    e = exp - exp.mean(); jj = jaw - jaw.mean()
    xc = np.correlate(jj, e, "full"); c = n - 1; Wn = int(0.2 * fps)
    lag = (np.argmax(xc[c - Wn:c + Wn + 1]) - Wn) * (1000 / fps)
    print(f"VOWEL_LAG {lag:+.0f}ms  (jaw vs expected vowels, +=jaw late)  phonemes={len(segs)}")

    # ---- plot ----
    t = np.arange(n) / fps
    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(min(16, 3 + dur * 2.2), 5), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 2]})
    aw = np.abs(wav); aw = aw / (aw.max() + 1e-9)
    ax0.plot(np.arange(len(wav)) / 16000, aw, color="#999", lw=0.5); ax0.set_ylabel("audio")
    ax0.set_title(f"{os.path.basename(a.wav)}   vowel-lag {lag:+.0f}ms", fontsize=10)
    for ph, t0, t1, c in segs:
        ax0.axvspan(t0, t1, color=COL[c], alpha=0.18)
        ax0.text((t0 + t1) / 2, 1.02, ph, ha="center", va="bottom", fontsize=7, color=COL[c])
    ax1.plot(t, exp, color="#333", lw=2, label="expected open (vowels)")
    ax1.plot(t, jaw / (jaw.max() + 1e-9), color="#e24", lw=1.5, label="A2F jawOpen")
    ax1.plot(t, np.maximum(funnel, puck) / (max(funnel.max(), puck.max()) + 1e-9), color="#28e", lw=1, label="A2F round")
    ax1.plot(t, mclose / (mclose.max() + 1e-9), color="#a0a", lw=1, label="A2F mouthClose")
    for ph, t0, t1, c in segs:
        ax1.axvspan(t0, t1, color=COL[c], alpha=0.12)
    ax1.set_ylabel("normalized"); ax1.set_xlabel("time (s)"); ax1.legend(loc="upper right", fontsize=7)
    ax1.set_xlim(0, dur)
    plt.tight_layout(); plt.savefig(a.out, dpi=110); print(f"PROBE_OK -> {a.out}")


if __name__ == "__main__":
    main()
