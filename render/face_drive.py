#!/usr/bin/env python3
"""Single coherent face driver: ARKit weights -> SMPL-X face channels.

Consolidates what used to be spread across (and double-applied by) face.talk's
arkit_to_face + the bakers: ONE resample, ONE smoothing stage, ONE gaze source,
NO inert FLAME expression, and a SINGLE coordinated mouth model where the jaw and
the mesh-space lip-press can't contradict each other (lip-press gates the jaw).

Returns a dict consumed by the bakers:
  jaw (F,3) -> jaw_pose ;  leye/reye (F,3) -> eye gaze ;
  mouth_close/mouth_pucker (F,) -> apply_lips ;  blink_l/blink_r (F,) -> apply_blink
The caller does NOT set `expression` (it displaces the SMPL-X face <1mm — dead weight).
"""
import numpy as np
from face.talk import _gaze_channels      # the one gaze generator we keep


def _smooth(v, k=3):
    if k < 2:
        return v
    return np.convolve(np.pad(v, k // 2, mode="edge"), np.ones(k) / k, "valid")[:len(v)]


def drive(arkit, F, fps, *, jaw_max=0.30, lip_gate=0.5, smooth=3, seed=0):
    """ARKit-52 weights (a2f or LAM json) -> SMPL-X face channels for F frames @ fps."""
    names = arkit["arkit_names"]
    W = np.asarray(arkit["weights"], np.float32)
    src = float(arkit.get("fps", 30.0))
    T = len(W)
    st = np.arange(T) / src
    dt = np.clip(np.arange(F) / float(fps), 0, st[-1] if T > 1 else 0.0)

    def ch(name):
        if name in names:
            return np.interp(dt, st, W[:, names.index(name)]).astype(np.float32)
        return np.zeros(F, np.float32)

    # ONE smoothing stage (a2f_lipsync no longer smooths; arkit_to_face is bypassed)
    jaw_open = _smooth(ch("jawOpen"), smooth)
    mouth_close = _smooth(ch("mouthClose"), smooth)
    mouth_pucker = _smooth(ch("mouthPucker") + 0.5 * ch("mouthFunnel"), smooth)
    blink_l, blink_r = ch("eyeBlinkLeft"), ch("eyeBlinkRight")

    # normalize jawOpen so typical speech peaks use the full range (the raw weights
    # rarely exceed ~0.6, which would otherwise waste jaw_max and look barely-open).
    jaw_norm = np.clip(jaw_open / (np.percentile(jaw_open, 95) + 1e-6), 0.0, 1.0)
    # ONE coordinated mouth model: a lip-press (mouthClose) suppresses jaw opening so
    # the two systems can't pull the lips apart and together at once.
    jaw = np.zeros((F, 3), np.float32)
    jaw[:, 0] = jaw_norm * (1.0 - lip_gate * mouth_close) * jaw_max

    leye, reye = _gaze_channels(F, seed)            # ONE continuous gaze source
    return {"jaw": jaw, "leye": leye, "reye": reye,
            "mouth_close": mouth_close, "mouth_pucker": mouth_pucker,
            "blink_l": blink_l, "blink_r": blink_r}
