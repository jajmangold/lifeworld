#!/usr/bin/env python3
"""
server.py — resident Qwen3-TTS two-speaker dialogue server (Wan2GP official impl).
Loads the base model ONCE (cudagraph engine), then serves dialogue render jobs over
HTTP so each podcast skips the ~60s model load.

POST /dialogue   JSON: {
    "text": "Speaker 1: ...\nSpeaker 2: ...",   # required
    "ref1": "/work/ref1.wav", "ref2": "/work/ref2.wav",
    "ref1_text": "", "ref2_text": "",
    "seed": 42, "temperature": 0.9, "pause_seconds": 0.45
}
-> audio/wav (24kHz mono)
GET /health -> {"status":"ok"}
"""
import io
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, "/workspace")
sys.path.insert(0, "/work")

import numpy as np
import soundfile as sf
import dialogue_tts as d   # reuses build_pipeline (cg + move-to-cuda + fp opt)

PORT = int(os.environ.get("PORT", "8080"))
_pipe = None
_lock = threading.Lock()   # model is single-GPU; serialize requests


def get_pipe():
    global _pipe
    if _pipe is None:
        print("[server] loading pipeline ...", flush=True)
        _pipe = d.build_pipeline(os.environ.get("QWEN3_ENGINE", "cg"))
        print(f"[server] ready (sr={_pipe.sample_rate})", flush=True)
    return _pipe


def _ref_text(req, key, ref_path):
    """Use request-provided ref text, else the sibling <ref>.txt (VCTK transcript)."""
    t = req.get(key, "").strip()
    if t:
        return t
    sib = os.path.splitext(ref_path)[0] + ".txt"
    if os.path.isfile(sib):
        return open(sib, encoding="utf-8").read().strip()
    return ""


def render(req):
    pipe = get_pipe()
    text = req["text"]
    ref1 = req.get("ref1", "/work/ref1.wav")
    ref2 = req.get("ref2", "/work/ref2.wav")
    rt1 = _ref_text(req, "ref1_text", ref1)
    rt2 = _ref_text(req, "ref2_text", ref2)
    tile = req.get("tile", os.environ.get("QWEN3_TILE", "0") == "1")
    with _lock:
        # Per-request sampling overrides (temperature/top_p/top_k/repetition_penalty).
        gd = dict(pipe.tts.generate_defaults)
        for k_req, k_cfg in (("top_p", "top_p"), ("top_k", "top_k"),
                             ("repetition_penalty", "repetition_penalty"),
                             ("temperature", "temperature")):
            if k_req in req and req[k_req] is not None:
                v = req[k_req]
                gd[k_cfg] = v
                gd["subtalker_" + k_cfg] = v if k_cfg != "repetition_penalty" else gd.get("subtalker_" + k_cfg)
        pipe.tts.generate_defaults = gd
        if tile:
            result = d.generate_tiled(
                pipe, text, ref1, ref2, rt1, rt2,
                language=req.get("language", "english"),
                temperature=float(req.get("temperature", 0.9)),
                seed=int(req.get("seed", 42)),
                pause_seconds=float(req.get("pause_seconds", 0.45)),
            )
        else:
            result = pipe.generate(
                input_prompt=text,
                model_mode=req.get("language", "english"),
                audio_guide=ref1,
                audio_guide2=ref2,
                alt_prompt=(rt1 + "\n" + rt2).strip("\n") or None,
                temperature=float(req.get("temperature", 0.9)),
                seed=int(req.get("seed", 42)),
                audio_prompt_type="AB",
                pause_seconds=float(req.get("pause_seconds", 0.45)),
            )
    if result is None:
        raise RuntimeError("generation returned None")
    audio = result["x"] if isinstance(result, dict) else result
    sr = int(result.get("audio_sampling_rate", pipe.sample_rate)) if isinstance(result, dict) else pipe.sample_rate
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().float().numpy()
    audio = np.asarray(audio, dtype=np.float32).squeeze()
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


def render_mono(req):
    # Single-speaker clone (audio_prompt_type="A"): one ref voice, plain text (no Speaker N: prefixes).
    pipe = get_pipe()
    text = req["text"]
    ref1 = req.get("ref1", "/work/ref1.wav")
    rt1 = _ref_text(req, "ref1_text", ref1)
    with _lock:
        result = pipe.generate(
            input_prompt=text,
            model_mode=req.get("language", "english"),
            audio_guide=ref1,
            alt_prompt=(rt1 or None),
            audio_prompt_type="A",
            temperature=float(req.get("temperature", 0.85)),
            seed=int(req.get("seed", 42)),
        )
    if result is None:
        raise RuntimeError("generation returned None")
    audio = result["x"] if isinstance(result, dict) else result
    sr = int(result.get("audio_sampling_rate", pipe.sample_rate)) if isinstance(result, dict) else pipe.sample_rate
    if hasattr(audio, "detach"):
        audio = audio.detach().cpu().float().numpy()
    audio = np.asarray(audio, dtype=np.float32).squeeze()
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path == "/health":
            body = json.dumps({"status": "ok"}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self):
        if self.path not in ("/dialogue", "/blend", "/mono"):
            self.send_error(404)
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/blend":
                # combine two voices via x-vector interpolation -> new hybrid ref wav
                import soundfile as _sf
                pipe = get_pipe()
                with _lock:
                    audio, sr = d.blend_voice(
                        pipe, req["ref_a"], req["ref_b"],
                        alpha=float(req.get("alpha", 0.5)),
                        sentence=req.get("sentence"),
                        temperature=float(req.get("temperature", 0.85)))
                out = req.get("out")
                if out:                       # save in-container (e.g. /work/ref1.wav)
                    _sf.write(out, audio, sr)
                buf = io.BytesIO()
                _sf.write(buf, audio, sr, format="WAV")
                wav = buf.getvalue()
            elif self.path == "/mono":
                if not req.get("text", "").strip():
                    raise ValueError("missing 'text'")
                wav = render_mono(req)
            else:
                if not req.get("text", "").strip():
                    raise ValueError("missing 'text'")
                wav = render(req)
            self.send_response(200)
            self.send_header("Content-Type", "audio/wav")
            self.send_header("Content-Length", str(len(wav)))
            self.end_headers()
            self.wfile.write(wav)
        except Exception as e:
            import traceback
            traceback.print_exc()
            body = json.dumps({"error": str(e)}).encode()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)


if __name__ == "__main__":
    get_pipe()  # load before accepting connections
    print(f"[server] listening on :{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
