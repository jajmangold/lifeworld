#!/usr/bin/env python3
"""LAM/A2F ARKit-52 -> FLAME (SMPL-X) expression + jaw + eye, via PeizhiYan/mediapipe-blendshapes
-to-flame linear maps. Drives the FULL face (brows, smiles, squints, ...) through SMPL-X's own
FLAME-2020 blendshapes — not just jaw. Mappings are research/educational-use only.

  drv = FlameDriver(mappings_dir)
  expr(F,100), jaw(F,3), leye(F,3), reye(F,3) = drv.drive(arkit_json, F, fps)
"""
import numpy as np

# MediaPipe FaceLandmarker blendshape order (52, index 0 = _neutral). The mapping matrices are
# in THIS order; LAM/A2F emit ARKit names (no _neutral, +tongueOut) so we look up by name.
MP_ORDER = [
    "_neutral", "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft",
    "browOuterUpRight", "cheekPuff", "cheekSquintLeft", "cheekSquintRight", "eyeBlinkLeft",
    "eyeBlinkRight", "eyeLookDownLeft", "eyeLookDownRight", "eyeLookInLeft", "eyeLookInRight",
    "eyeLookOutLeft", "eyeLookOutRight", "eyeLookUpLeft", "eyeLookUpRight", "eyeSquintLeft",
    "eyeSquintRight", "eyeWideLeft", "eyeWideRight", "jawForward", "jawLeft", "jawOpen",
    "jawRight", "mouthClose", "mouthDimpleLeft", "mouthDimpleRight", "mouthFrownLeft",
    "mouthFrownRight", "mouthFunnel", "mouthLeft", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthPressLeft", "mouthPressRight", "mouthPucker", "mouthRight", "mouthRollLower",
    "mouthRollUpper", "mouthShrugLower", "mouthShrugUpper", "mouthSmileLeft", "mouthSmileRight",
    "mouthStretchLeft", "mouthStretchRight", "mouthUpperUpLeft", "mouthUpperUpRight",
    "noseSneerLeft", "noseSneerRight",
]


class FlameDriver:
    def __init__(self, mdir):
        self.bs2exp = np.load(f"{mdir}/bs2exp.npy")   # (52,100)
        self.bs2jaw = np.load(f"{mdir}/bs2jaw.npy")   # (52,3)
        self.bs2eye = np.load(f"{mdir}/bs2eye.npy")   # (52,6) leye(3)+reye(3)

    def drive(self, arkit, F, fps, smooth=3, gain=0.4):
        names = arkit["arkit_names"]
        W = np.asarray(arkit["weights"], np.float32)
        src = float(arkit.get("fps", 30.0)); T = len(W)
        st = np.arange(T) / src
        dt = np.clip(np.arange(F) / float(fps), 0, st[-1] if T > 1 else 0.0)
        idx = {n: i for i, n in enumerate(names)}
        mp = np.zeros((F, 52), np.float32)
        for j, nm in enumerate(MP_ORDER):                 # reorder ARKit -> MediaPipe order
            if nm in idx:
                mp[:, j] = np.interp(dt, st, W[:, idx[nm]])
        if smooth and smooth > 1:                          # light temporal smoothing
            k = np.ones(smooth) / smooth
            mp = np.stack([np.convolve(np.pad(mp[:, c], smooth // 2, "edge"), k, "valid")[:F]
                           for c in range(52)], 1)
        # gain<1 tames over-driven coeffs (MP_2_FLAME peaks ~|8|; SMPL-X distorts above ~|3|,
        # worst on the high-order components during speech). jaw kept full-strength.
        exp = (mp @ self.bs2exp).astype(np.float32) * gain  # (F,100) FLAME expression
        jaw = (mp @ self.bs2jaw).astype(np.float32)        # (F,3)  jaw pose
        eye = (mp @ self.bs2eye).astype(np.float32)        # (F,6)  eye pose
        return exp, jaw, eye[:, :3].copy(), eye[:, 3:6].copy()
