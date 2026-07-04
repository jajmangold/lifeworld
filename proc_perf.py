#!/usr/bin/env python3
"""Procedural facial PERFORMANCE generator — replaces the FLOAT->MediaPipe path.
Authors blinks + micro-expressions + mood directly on the avatar's ARKit blendshapes, with
emphasis brow-raises/nods derived from the audio envelope (so it's speech-aware without a
talking-head model). Outputs the same json the renderer consumes (proc:true => applied directly).

  python proc_perf.py --audio anchor_audio.wav --mood warm --seed 7 --out anchor.perf.json [--fps 25]

Moods: neutral | warm | serious | alert   (tweak MOODS below)
"""
import sys, json, wave, math
import numpy as np

def arg(flag, d):
    a = sys.argv
    return a[a.index(flag)+1] if flag in a else d

AUDIO = arg("--audio", "/io/anchor_audio.wav")
OUT   = arg("--out",   "/io/anchor.perf.json")
FPS   = float(arg("--fps", "25"))
MOOD  = arg("--mood", "neutral")
SEED  = int(arg("--seed", "7"))
DUR   = arg("--dur", None)   # override duration (s); else from audio
NOD   = float(arg("--nod", "1.0"))    # emphasis head-nod scale (1.0 = default, <1 = subtler)
BROW  = float(arg("--brow", "1.0"))   # eyebrow/forehead expressiveness scale (0 = still, >1 = more)
BROWBASE = arg("--browbase", None)    # override resting brow lift (else mood default)
VISEMES  = "--visemes" in sys.argv     # drive jaw/viseme mouth from audio (fast-preview, no MuseTalk)
JAWGAIN  = float(arg("--jawgain", "1.0"))

rng = np.random.default_rng(SEED)

# ---- mood presets: baseline expression + dynamics --------------------------------
MOODS = {
    # browBase: constant browInnerUp lift | cheek: "smiling-eyes" cheekSquint | eyeWide | squint
    # blink_s: mean inter-blink (s) | head_amp: idle multiplier | emph: emphasis gain
    "neutral": dict(browBase=0.00, cheek=0.00, eyeWide=0.00, squint=0.00, blink_s=3.6, head_amp=1.0, emph=1.0),
    "warm":    dict(browBase=0.07, cheek=0.13, eyeWide=0.00, squint=0.05, blink_s=3.2, head_amp=1.0, emph=1.05),
    "serious": dict(browBase=0.00, cheek=0.00, eyeWide=0.00, squint=0.02, blink_s=4.2, head_amp=0.7, emph=0.8),
    "alert":   dict(browBase=0.03, cheek=0.02, eyeWide=0.06, squint=0.00, blink_s=3.0, head_amp=1.1, emph=1.25),
}
M = MOODS.get(MOOD, MOODS["neutral"])

# ---- duration / frame count ------------------------------------------------------
if DUR is not None:
    sr = 16000; dur = float(DUR); env = None
else:
    w = wave.open(AUDIO, "rb")
    sr = w.getframerate(); n = w.getnframes(); nch = w.getnchannels()
    raw = w.readframes(n); w.close()
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if nch > 1: a = a.reshape(-1, nch).mean(axis=1)
    dur = len(a) / sr
    # per-frame RMS envelope + spectral centroid (for viseme vowel selection)
    hop = int(sr / FPS)
    T0 = int(math.ceil(len(a) / hop))
    env = np.zeros(T0); cen = np.zeros(T0)
    freqs = np.fft.rfftfreq(hop, 1.0/sr) if hop > 1 else np.array([0.0])
    for i in range(T0):
        seg = a[i*hop:(i+1)*hop]
        env[i] = np.sqrt(np.mean(seg**2) + 1e-9)
        if len(seg) == hop:
            mag = np.abs(np.fft.rfft(seg * np.hanning(hop)))
            cen[i] = (freqs * mag).sum() / (mag.sum() + 1e-9)   # Hz
    env = np.clip(env / (np.percentile(env, 95) + 1e-6), 0, 1.5)
    # normalize centroid to 0..1 over a speech-ish band (~300-2500 Hz)
    cen = np.clip((cen - 300.0) / 2200.0, 0.0, 1.0)

T = int(round(dur * FPS))
if env is None: env = np.zeros(T)
try: cen
except NameError: cen = np.zeros(T)
def _fit(v):
    v = np.asarray(v, np.float32)
    return np.concatenate([v, np.zeros(T-len(v))]) if len(v) < T else v[:T]
env = _fit(env); cen = _fit(cen)
t_sec = np.arange(T) / FPS

# ---- blinks: Poisson-timed, realistic close/hold/open curve ----------------------
SHAPE = np.array([0.55, 1.0, 0.8, 0.3, 0.08])   # ~5f @25fps (~0.2s): snappy close, quick open (no slow hold/tail)
blink = np.zeros(T)
t = int(rng.uniform(0.5, 1.8) * FPS)
while t < T:
    for k, v in enumerate(SHAPE):
        if 0 <= t + k < T: blink[t+k] = max(blink[t+k], v)
    if rng.random() < 0.12:                      # occasional double-blink
        gap = rng.uniform(0.26, 0.42)
    else:
        gap = float(np.clip(rng.exponential(M["blink_s"]), 1.5, 9.0))
    t += int(gap * FPS) + len(SHAPE)

# ---- slow brow drift (liveliness) ------------------------------------------------
phs = rng.uniform(0, 2*math.pi, 3); frq = [0.05, 0.09, 0.13]
drift = sum(a*np.sin(2*math.pi*f*t_sec + p) for a, f, p in zip([0.05, 0.03, 0.02], frq, phs))
drift = np.clip(drift, -0.06, 0.10)

# ---- HEAD MOTION: layered natural idle (coherent drift + breathing + micro-saccades) + speech emphasis
# Real heads are never still. Replace the old dead-still + random sin(frame) jerk with procedural idle:
#   coherent fractal-noise drift (low freq = calm) + a breathing pitch-bob + occasional eased micro-saccades,
#   scaled up a touch while speaking and settling (never freezing) in pauses. All seeded -> reproducible.
IDLE = float(arg("--idle", "1.0"))     # idle-motion scale (0 = perfectly still)
D2R = math.pi / 180.0
emph_brow = np.zeros(T); head = np.zeros((T, 3))   # cols: pitch, yaw, roll (radians)

def _fractal(amps_deg, freqs):         # cheap 1D coherent noise = sum of incommensurate low sines
    ph = rng.uniform(0, 2*math.pi, len(freqs))
    return sum(a*np.sin(2*math.pi*f*t_sec + p) for a, f, p in zip(amps_deg, freqs, ph)) * D2R
yaw_idle   = _fractal([0.7, 0.4, 0.25], [0.06, 0.11, 0.17])
pitch_idle = _fractal([0.5, 0.3, 0.20], [0.05, 0.10, 0.16])
roll_idle  = _fractal([0.4, 0.25],      [0.06, 0.12])

# breathing: ~15 breaths/min pitch bob, slight slow period jitter so it isn't metronomic
breath_phase = 2*math.pi*0.25*t_sec + 0.5*np.sin(2*math.pi*0.03*t_sec + rng.uniform(0, 6))
breath = (0.35*D2R) * np.sin(breath_phase)

# micro-saccades: Poisson ~1/5.5s, eased small step that decays (direction coherent w/ the drift, not random)
sacc_yaw = np.zeros(T); sacc_pitch = np.zeros(T); i = int(1.5*FPS)
while i < T-1:
    i += max(int(0.8*FPS), int(rng.exponential(5.5) * FPS))
    if i >= T-1: break
    dy = math.copysign(rng.uniform(0.5, 1.1), yaw_idle[i] or 1.0) * D2R
    dp = rng.uniform(-0.4, 0.4) * D2R
    tt = np.clip(t_sec - t_sec[i], 0, None)
    ease = np.where(t_sec >= t_sec[i], (1 - np.exp(-tt/0.18)) * np.exp(-tt/3.0), 0.0)
    sacc_yaw += dy*ease; sacc_pitch += dp*ease

# speech activity -> idle gain (a little more motion while talking, settle in pauses but never freeze)
_wlen = max(1, int(0.5*FPS))
spk = np.convolve(np.clip(env, 0, 1), np.ones(_wlen)/_wlen, mode="same")
idle_gain = np.convolve(0.85 + 0.30*np.clip(spk, 0, 1), np.ones(_wlen)/_wlen, mode="same")

# emphasis nod on stressed peaks (chin-down) + a tiny COHERENT yaw (sign from the drift, not sin(frame))
emph_pitch = np.zeros(T); emph_yaw = np.zeros(T)
if env.any():
    thr = np.percentile(env, 68)
    win = np.array([0.2, 0.55, 0.9, 1.0, 0.85, 0.6, 0.35, 0.15])  # ease pulse
    refr = int(0.55 * FPS); last = -refr
    for i in range(1, T-1):
        if env[i] > thr and env[i] >= env[i-1] and env[i] > env[i+1] and (i - last) >= refr:
            g = M["emph"] * min(1.0, (env[i]-thr)/(1.0-thr+1e-6)) * rng.uniform(0.8, 1.1)
            ydir = math.copysign(1.0, yaw_idle[i] or 1.0)
            for k, wv in enumerate(win):
                j = i + k - 1
                if 0 <= j < T:
                    emph_brow[j]   = max(emph_brow[j], 0.32*g*wv)
                    emph_pitch[j] += -math.radians(1.5)*g*wv*NOD          # chin-down nod
                    emph_yaw[j]   +=  math.radians(0.6)*g*wv*NOD*ydir     # coherent micro-yaw
            last = i

head[:, 0] = IDLE*idle_gain*(pitch_idle + breath) + emph_pitch + IDLE*sacc_pitch
head[:, 1] = IDLE*idle_gain*yaw_idle + emph_yaw + IDLE*sacc_yaw
head[:, 2] = IDLE*idle_gain*roll_idle
head[:, 0] = np.clip(head[:, 0], -3.0*D2R, 3.0*D2R)   # safety envelope so layers never stack cartoonish
head[:, 1] = np.clip(head[:, 1], -3.5*D2R, 3.5*D2R)
head[:, 2] = np.clip(head[:, 2], -2.0*D2R, 2.0*D2R)

# ---- compose channels (clip 0..1) ------------------------------------------------
def C(x): return list(np.clip(x, 0.0, 1.0).astype(float))
bb = float(BROWBASE) if BROWBASE is not None else M["browBase"]
browInner = (bb + drift + emph_brow) * BROW
browOuter = (bb*0.5 + 0.4*emph_brow) * BROW
ch = {
    "eyeBlinkLeft":  C(blink),
    "eyeBlinkRight": C(blink),
    "browInnerUp":   C(browInner),
    "browOuterUpLeft":  C(browOuter),
    "browOuterUpRight": C(browOuter),
    # faint breathing-synced shimmer so the face isn't a frozen mask (tiny — a light garnish)
    "cheekSquintLeft":  C(np.full(T, M["cheek"]) + 0.02*IDLE*(0.5+0.5*np.sin(breath_phase))),
    "cheekSquintRight": C(np.full(T, M["cheek"]) + 0.02*IDLE*(0.5+0.5*np.sin(breath_phase))),
    "eyeSquintLeft":  C(np.full(T, M["squint"])),
    "eyeSquintRight": C(np.full(T, M["squint"])),
    "eyeWideLeft":  C(np.full(T, M["eyeWide"])),
    "eyeWideRight": C(np.full(T, M["eyeWide"])),
}
# ---- visemes (fast-preview mouth, no MuseTalk): jaw from loudness, vowel from spectral centroid ----
if VISEMES:
    jaw = np.convolve(env, np.ones(3)/3, mode="same")
    jaw = np.clip(jaw * 0.75 * JAWGAIN, 0.0, 1.0)
    speak = np.convolve((env > 0.12).astype(np.float32), np.ones(3)/3, mode="same")
    lo = 1.0 - cen   # roundness (low centroid -> ou/oo)
    hi = cen         # frontness (high centroid -> ee/E)
    ch.update({
        "jawOpen":     C(jaw),
        "aa":          C(jaw * 0.6 * speak),
        "E":           C(jaw * 0.5 * hi * speak),
        "ou":          C((0.4*lo + 0.3*jaw*lo) * speak),
        "mouthFunnel": C(0.35 * lo * speak),
        "mouthClose":  C(0.5 * (1-speak) * (1-jaw)),
    })
else:
    # GENTLE jaw under MuseTalk: even though muse owns the 2D mouth, a subtle 3D jaw/chin drop synced to
    # loudness makes the head look like it's actually speaking (the mouth+jaw move together). Low gain so it
    # never fights the muse inpaint. Renderer drives 'jawOpen' at a further-reduced weight.
    _jaw = np.convolve(env, np.ones(5)/5, mode="same")
    ch["jawOpen"] = C(np.clip(_jaw * 0.45 * JAWGAIN, 0.0, 0.6))
nvis = sum(k in ch for k in ("jawOpen","aa","E","ou"))
names = list(ch.keys())
weights = [[ch[nm][fi] for nm in names] for fi in range(T)]
json.dump({"proc": True, "arkit_names": names, "weights": weights, "fps": FPS,
           "num_frames": T, "head": head.tolist(), "head_amp": M["head_amp"], "mood": MOOD},
          open(OUT, "w"))
nblink = int(((blink > 0.5)[:-1] < (blink > 0.5)[1:]).sum())
print(f"PERF_OK mood={MOOD} frames={T} dur={dur:.1f}s blinks={nblink} visemes={'on('+str(nvis)+')' if VISEMES else 'off'} -> {OUT}")
