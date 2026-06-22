#!/usr/bin/env python3
"""Pool FORCED-ALIGNED visemes vs A2F jaw across many lines -> systematic lip-sync offset.

Per line: torchaudio MMS forced-alignment (transcript + audio) gives accurate per-character
timing -> a clean 'expected jaw-open' signal (vowels open, consonants closed). Compare to A2F
jawOpen and sweep a global offset; pool over all lines for a robust systematic latency. Forced
alignment is the fix for the earlier ~0 correlations (free CTC / audio-envelope were too noisy).
"""
import os
os.environ.setdefault("TORCH_HOME", "/a2f/torch")
import argparse
import glob
import json
import re
import numpy as np
import torch
import soundfile as sf
import librosa
from torchaudio.pipelines import MMS_FA as BUNDLE

FPS = 50
# letter -> jaw open-ness (vowels open, consonants ~closed); accurate timing comes from FA
OPENW = {"a": 1.0, "o": 0.8, "e": 0.6, "u": 0.6, "i": 0.4, "y": 0.4}


def expected(wav, text, model, tok, aligner):
    w, sr = sf.read(wav)
    if w.ndim > 1:
        w = w.mean(1)
    if sr != 16000:
        w = librosa.resample(w.astype(np.float32), orig_sr=sr, target_sr=16000)
    wt = torch.tensor(w, dtype=torch.float32)[None]
    words = re.sub(r"[^a-z ]", " ", text.lower()).split()
    with torch.inference_mode():
        emission, _ = model(wt)
        spans = aligner(emission[0], tok(words))
    ratio = wt.shape[1] / emission.shape[1] / 16000.0     # seconds per emission frame
    n = int(len(w) / 16000 * FPS); exp = np.zeros(n)
    for word, sp in zip(words, spans):
        for ch, ts in zip(word, sp):
            v = OPENW.get(ch, 0.0)
            if v:
                f0 = int(ts.start * ratio * FPS); f1 = max(f0 + 1, int(ts.end * ratio * FPS))
                exp[f0:min(f1, n)] = np.maximum(exp[f0:min(f1, n)], v)
    k = 5
    return np.convolve(np.pad(exp, k // 2, "edge"), np.ones(k) / k, "valid")[:n]


def jaw_at(arkit, n):
    d = json.load(open(arkit)); W = np.asarray(d["weights"], np.float32)
    j = W[:, d["arkit_names"].index("jawOpen")]
    return np.interp(np.arange(n) / FPS, np.arange(len(j)) / d["fps"], j)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); a = ap.parse_args()
    tx = json.load(open(f"{a.dir}/transcripts.json"))
    model = BUNDLE.get_model(); tok = BUNDLE.get_tokenizer(); aligner = BUNDLE.get_aligner()
    pairs, perlag = [], []
    TAUS = np.arange(-15, 16)
    for wav in sorted(glob.glob(f"{a.dir}/*.wav")):
        idx = os.path.basename(wav)[:-4]; ark = wav[:-4] + ".arkit.json"
        if idx not in tx or not os.path.exists(ark):
            continue
        try:
            e = expected(wav, tx[idx], model, tok, aligner)
        except Exception as ex:
            print("skip", idx, ex); continue
        j = jaw_at(ark, len(e)); e = e - e.mean(); j = j - j.mean()
        if e.std() < 1e-6 or j.std() < 1e-6:
            continue
        pairs.append((e, j))
        sc = []
        for t in TAUS:
            aa = e[max(0, t):len(e) + min(0, t)]; bb = np.roll(j, t)[max(0, t):len(j) + min(0, t)]
            sc.append(np.corrcoef(aa, bb)[0, 1] if len(aa) > 5 else -1)
        perlag.append(TAUS[int(np.argmax(sc))] * 1000 / FPS)
    pooled = []
    for t in TAUS:
        num = de = dj = 0.0
        for e, j in pairs:
            jr = np.roll(j, t); s = slice(abs(t), len(j) - abs(t) if t else None)
            num += float((e[s] * jr[s]).sum()); de += float((e[s]**2).sum()); dj += float((jr[s]**2).sum())
        pooled.append(num / (np.sqrt(de * dj) + 1e-9))
    pooled = np.array(pooled); perlag = np.array(perlag)
    best = TAUS[int(pooled.argmax())] * 1000 / FPS
    print("=" * 58)
    print(f"samples: {len(pairs)}   (forced-aligned visemes vs A2F jawOpen)")
    print(f"PER-LINE lag: mean {perlag.mean():+.0f}  median {np.median(perlag):+.0f}  std {perlag.std():.0f} ms")
    print(f"POOLED best offset: {best:+.0f}ms   peak corr {pooled.max():.2f}   corr@0 {pooled[TAUS == 0][0]:.2f}")
    print(f"  (+ = A2F jaw LATE; shift jaw earlier by {best:+.0f}ms to correct)")
    print("  offset:corr  " + " ".join(f"{int(t*1000/FPS):+d}:{c:.2f}" for t, c in zip(TAUS[::2], pooled[::2])))
    print("=" * 58)


if __name__ == "__main__":
    main()
