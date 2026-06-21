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

FPS = 60                       # diffusion model outputs 60fps
WIN = 16000                    # 1s audio buffer
HOP = 8000                     # 0.5s hop (50% overlap) — per the A2F SDK protocol
CENTER0, CENTER1 = 15, 45      # keep center 30 (=0.5s @60fps) of the 60 predicted frames
LEAD_TRIM = 10                 # drop the constant lead so output frame 0 ~ audio 0 (tuned ~0 lag)
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


def load_solver(a2f_dir, identity, ridge=0.05):
    """Build the frontal-masked ARKit-52 least-squares solver for an identity.
    The network output and bs_skin bases share a coordinate space (bs.neutral ==
    model_data.neutral_skin), so geometry deltas project onto the named blendshape
    deltas. Bases are zero outside the face -> restrict to frontalMask."""
    import json as _json
    bs = np.load(f"{a2f_dir}/bs_skin_{identity}.npz", allow_pickle=True)
    pose_names = [p.decode() if isinstance(p, bytes) else str(p) for p in bs["poseNames"]]
    names = [n for n in pose_names if n != "neutral"]
    neutral = bs["neutral"].astype(np.float32)
    vmask = bs["frontalMask"].astype(np.int64)
    dmask = (vmask[:, None] * 3 + np.arange(3)).reshape(-1)          # vertex->xyz dim indices
    A = np.stack([(bs[n].astype(np.float32) - neutral).reshape(-1)[dmask] for n in names], 1)  # (M,52)
    pinv = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1], dtype=np.float32), A.T).astype(np.float32)  # (52,M)
    active = np.ones(len(names), np.float32)
    try:
        cfg = _json.load(open(f"{a2f_dir}/bs_skin_config_{identity}.json"))
        ap = cfg["blendshape_params"]["bsSolveActivePoses"]
        active = np.array(ap[:len(names)], np.float32)
    except Exception:
        pass
    return {"names": names, "dmask": dmask, "pinv": pinv, "active": active}


def run_one(sess, ins, a2f_dir, wav, out, identity, cache):
    idx = ["Claire", "James", "Mark"].index(identity)
    if identity not in cache:
        cache[identity] = load_solver(a2f_dir, identity)
    slv = cache[identity]

    def shp(name):
        return [d if isinstance(d, int) and d > 0 else 1 for d in ins[name]]

    audio = load_wav_16k(wav)
    dur = len(audio) / 16000.0
    # 1s buffer, 0.5s hop (50% overlap); keep center 30 frames @60fps per call (A2F
    # SDK protocol). NO leading pad. Sequential, carrying latents.
    nwin = max(1, int(np.ceil(len(audio) / HOP)))

    lat = np.zeros(shp("input_latents"), np.float32)
    rng = np.random.RandomState(0)
    dmask, pinv, active, names = slv["dmask"], slv["pinv"], slv["active"], slv["names"]
    chunks = []
    for wi in range(nwin):
        seg = audio[wi * HOP:wi * HOP + WIN]
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
        chunks.append((pinv @ skin[:, dmask].T).T)           # (30, 52) ARKit weights

    W = np.concatenate(chunks, 0) * active[None, :]          # gate inactive poses
    W = W[LEAD_TRIM:LEAD_TRIM + max(1, round(dur * FPS))]    # drop lead + trim to audio (A/V sync)
    # remove resting bias per channel so the face rests NEUTRAL (mouth closed) and only
    # deviates during speech. NOTE: smoothing is intentionally NOT done here — the single
    # smoothing stage lives in face_drive.drive() (avoids double low-pass / lag).
    W = W - np.percentile(W, 25, axis=0, keepdims=True)
    W = np.clip(W, 0.0, 1.0).astype(np.float32)
    jo = W[:, names.index("jawOpen")]
    mc = W[:, names.index("mouthClose")]
    mp = W[:, names.index("mouthPucker")]
    json.dump({"arkit_names": names, "weights": W.tolist(), "fps": FPS, "engine": "a2f-3d"},
              open(out, "w"))
    print(f"A2F_LIPSYNC_OK frames={len(W)} jawOpen[mean/max]={jo.mean():.2f}/{jo.max():.2f} "
          f"mouthClose[max]={mc.max():.2f} mouthPucker[max]={mp.max():.2f} -> {out}")


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

    cache = {}
    if a.manifest:
        for job in json.load(open(a.manifest)):
            run_one(sess, ins, a.a2f, job["wav"], job["out"], job.get("identity", "Claire"), cache)
    else:
        run_one(sess, ins, a.a2f, a.wav, a.out, a.identity, cache)


if __name__ == "__main__":
    main()
