#!/usr/bin/env python3
"""
master_audio.py — even out and loudness-normalize a dialogue render.

Qwen3-TTS decodes each turn independently, so per-turn loudness drifts (~14 dB
swing observed). This:
  1. splits on the inter-turn pause silences,
  2. levels each turn to a common RMS (capped gain so near-silent turns don't blow up),
  3. runs ffmpeg highpass + loudnorm (-16 LUFS / -1.5 dBTP) for a broadcast-clean master.

Usage: master_audio.py in.wav out.wav   (also call mastered() from other scripts)
"""
import subprocess
import sys
import numpy as np
import soundfile as sf

TARGET_RMS_DBFS = -20.0
MAX_GAIN_DB = 12.0
SIL_THRESH = 0.006      # ~ -44 dBFS
MIN_SILENCE_S = 0.30


def _level_turns(a, sr):
    win = int(sr * 0.02)
    env = np.array([np.sqrt((a[i:i + win] ** 2).mean()) for i in range(0, len(a) - win, win)])
    sil = env < SIL_THRESH
    minsil = int(MIN_SILENCE_S / 0.02)
    target = 10 ** (TARGET_RMS_DBFS / 20)
    maxg = 10 ** (MAX_GAIN_DB / 20)
    out = a.copy()
    j, n = 0, len(env)
    while j < n:
        if not sil[j]:
            k = j
            while k < n:
                if sil[k]:
                    m = k
                    while m < n and sil[m]:
                        m += 1
                    if m - k >= minsil:
                        break
                    k = m
                    continue
                k += 1
            s, e = j * win, min(len(a), k * win)
            seg = a[s:e]
            r = np.sqrt((seg ** 2).mean())
            if r > 1e-5:
                out[s:e] = seg * min(target / r, maxg)
            j = k
        else:
            j += 1
    peak = float(np.max(np.abs(out))) or 1.0
    if peak > 0.99:
        out *= 0.99 / peak
    return out


def mastered(in_wav, out_wav):
    a, sr = sf.read(in_wav, dtype="float32")
    if a.ndim > 1:
        a = a.mean(1)
    leveled = _level_turns(a, sr)
    tmp = out_wav + ".lvl.wav"
    sf.write(tmp, leveled, sr)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", tmp,
         "-af", "highpass=f=55,loudnorm=I=-16:TP=-1.5:LRA=11",
         "-ar", str(sr), out_wav],   # loudnorm upsamples to 192k by default; pin to source sr
        check=True,
    )
    import os
    os.remove(tmp)
    return out_wav


if __name__ == "__main__":
    mastered(sys.argv[1], sys.argv[2])
    print(f"mastered -> {sys.argv[2]}")
