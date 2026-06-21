#!/usr/bin/env python3
"""Audio2Face-3D lip-sync on Volta (sm_70) via ONNX Runtime -> ARKit-style JSON.

Drop-in for the LAM arkit json: same schema ({arkit_names, weights, fps}) so the
existing face.talk.arkit_to_face -> SMPL-X path is unchanged.

A2F-v3's ONNX outputs per-vertex face-geometry DELTAS (network space). model_data
ships the exact motion directions for mouth-open (lip_open_pose_delta) and blink
(eye_close_pose_delta) in that same space, so we get clean jawOpen / eyeBlink by
projecting each frame's delta onto those directions — no cross-space blendshape
solve needed. Runs on a CMP/V100 (CUDA EP; TensorRT dropped Volta).

  python3 a2f_lipsync.py --wav speech.wav --a2f /work/a2f --identity Claire --out out.arkit.json
"""
import argparse
import json
import sys
import wave
import numpy as np

FPS = 30
WIN = 16000
CENTER0, CENTER1 = 15, 45      # keep center 30 of the 60 predicted frames
SKIN_DIMS = 24002 * 3


def load_wav_16k(path):
    w = wave.open(path, "rb")
    n, sr, ch, sw = w.getnframes(), w.getframerate(), w.getnchannels(), w.getsampwidth()
    raw = w.readframes(n); w.close()
    dt = {1: np.int8, 2: np.int16, 4: np.int32}.get(sw, np.int16)
    x = np.frombuffer(raw, dtype=dt).astype(np.float32)
    if ch > 1:
        x = x.reshape(-1, ch).mean(1)
    x = x / (np.abs(x).max() + 1e-9)
    if sr != 16000:
        t = np.linspace(0, len(x) / sr, int(len(x) / sr * 16000), endpoint=False)
        x = np.interp(t, np.arange(len(x)) / sr, x).astype(np.float32)
    return x


def run_one(sess, ins, a2f_dir, wav, out, identity):
    idx = ["Claire", "James", "Mark"].index(identity)

    def shp(name):
        return [d if isinstance(d, int) and d > 0 else 1 for d in ins[name]]

    md = np.load(f"{a2f_dir}/model_data_{identity}.npz", allow_pickle=True)
    lip = md["lip_open_pose_delta"].astype(np.float32).reshape(-1)
    eye = md["eye_close_pose_delta"].astype(np.float32).reshape(-1)
    lip_n, eye_n = float(lip @ lip) + 1e-9, float(eye @ eye) + 1e-9

    audio = load_wav_16k(wav)
    audio = np.concatenate([np.zeros(WIN, np.float32), audio, np.zeros(WIN, np.float32)])
    nwin = max(1, int(np.ceil((len(audio) - WIN) / WIN)) + 1)

    lat = np.zeros(shp("input_latents"), np.float32)
    rng = np.random.RandomState(0)
    opens, blinks = [], []
    for wi in range(nwin):
        seg = audio[wi * WIN:wi * WIN + WIN]
        if len(seg) < WIN:
            seg = np.concatenate([seg, np.zeros(WIN - len(seg), np.float32)])
        feeds = {
            "window": seg[None, :].astype(np.float32),
            "identity": np.eye(3, dtype=np.float32)[idx][None, :],
            "emotion": np.zeros(shp("emotion"), np.float32),
            "input_latents": lat,
            "noise": rng.randn(*shp("noise")).astype(np.float32),
        }
        pred, lat = sess.run(["prediction", "output_latents"], feeds)
        skin = pred[0, CENTER0:CENTER1, :SKIN_DIMS]          # (30, 72006) deltas
        opens.append(skin @ lip / lip_n)
        blinks.append(skin @ eye / eye_n)

    o = np.concatenate(opens); b = np.concatenate(blinks)
    o = np.clip(o, 0, None); b = np.clip(b, 0, None)
    o = o / (np.percentile(o, 97) + 1e-6)                    # normalize to ~[0,1]
    b = b / (np.percentile(b, 99) + 1e-6)
    o = np.clip(o, 0, 1).astype(np.float32); b = np.clip(b, 0, 1).astype(np.float32)
    W = np.stack([o, b, b], 1)                               # jawOpen, blinkL, blinkR
    json.dump({"arkit_names": ["jawOpen", "eyeBlinkLeft", "eyeBlinkRight"],
               "weights": W.tolist(), "fps": FPS, "engine": "a2f-3d"}, open(out, "w"))
    print(f"A2F_LIPSYNC_OK frames={len(W)} jawOpen[mean/max]={o.mean():.3f}/{o.max():.3f} "
          f"blink[max]={b.max():.3f} -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wav")
    ap.add_argument("--a2f", default="/work/a2f")
    ap.add_argument("--identity", default="Claire")
    ap.add_argument("--out")
    ap.add_argument("--manifest", help="json list [{wav,out,identity}] processed in one session")
    a = ap.parse_args()

    import onnxruntime as ort
    sess = ort.InferenceSession(f"{a.a2f}/network.onnx",
                                providers=[("CUDAExecutionProvider", {"device_id": 0}),
                                           "CPUExecutionProvider"])
    if "CUDAExecutionProvider" not in sess.get_providers():
        print("CUDA_EP_NOT_ACTIVE", sess.get_providers()); sys.exit(3)
    ins = {i.name: i.shape for i in sess.get_inputs()}

    if a.manifest:
        for job in json.load(open(a.manifest)):
            run_one(sess, ins, a.a2f, job["wav"], job["out"], job.get("identity", "Claire"))
    else:
        run_one(sess, ins, a.a2f, a.wav, a.out, a.identity)


if __name__ == "__main__":
    main()
