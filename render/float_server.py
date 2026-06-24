#!/usr/bin/env python3
"""Resident FLOAT talking-head server: loads the model ONCE (float.pth + wav2vec + face_alignment),
then serves talking-head generation over HTTP so each request skips the ~cold-start load.
  GET  /health                      -> {"status":"ready","busy":bool}
  POST /gen     {"ref_path","audio_path","out_path", nfe?, a_cfg_scale?, e_cfg_scale?, seed?, no_crop?, emo?}
                -> renders a full mp4 to out_path (blocking, serialized)
  POST /render  {"audio_path", ref_path?, nfe?, a_cfg_scale?, seed?, emo?}
                -> streams frames into per-job HLS (non-blocking); returns playlist + player URLs
  GET  /hls/<job>/...               -> HLS playlist + segments (served live while rendering)
  GET  /player.html?src=/hls/<job>/index.m3u8
Paths are container-side (mount a shared dir). Inference is serialized (one GPU)."""
import sys, os, json, threading, traceback, mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

BASE = os.environ.get("FLOAT_BASE", "/workspace")
HLS = os.path.join(BASE, "hls")
os.makedirs(HLS, exist_ok=True)
CKPT = os.environ.get("FLOAT_CKPT", "./checkpoints/float.pth")
PORT = int(os.environ.get("FLOAT_PORT", "8210"))
DEFAULT_PORTRAIT = os.environ.get("FLOAT_PORTRAIT", "pastor_portrait.png")
sys.argv = ["float_server", "--ckpt_path", CKPT]          # feed InferenceOptions a clean argv
from generate import InferenceAgent, InferenceOptions

opt = InferenceOptions().parse()
opt.rank, opt.ngpus = 0, 1
agent = InferenceAgent(opt)                               # loads model once (~cold start here)
MOTION_DIM = int(agent.G.motion_autoencoder.dec.direction(None).shape[1])   # 20 motion axes
LOCK = threading.Lock()
STATE = {"job": 0, "busy": False}
print("FLOAT_SERVER_READY", flush=True)


def _resolve(p):
    if not p:
        return None
    p = p if os.path.isabs(p) else os.path.join(BASE, p)
    return p if os.path.exists(p) else None


def _render_stream(job, audio, portrait, nfe, a_cfg, seed, emo):
    outdir = os.path.join(HLS, str(job))
    os.makedirs(outdir, exist_ok=True)
    try:
        with LOCK:
            STATE["busy"] = True
            agent.run_inference_stream(portrait, audio, outdir, a_cfg_scale=float(a_cfg),
                                       nfe=int(nfe), seed=int(seed), emo=emo)
    except Exception:
        traceback.print_exc()
        with open(os.path.join(outdir, "error.txt"), "w") as f:
            f.write(traceback.format_exc())
    finally:
        STATE["busy"] = False


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers(); self.wfile.write(b)

    def _serve_static(self, rel):
        path = os.path.normpath(os.path.join(BASE, rel.lstrip("/")))
        if not path.startswith(BASE) or not os.path.isfile(path):
            return self._send(404, {"error": "not found"})
        ctype = mimetypes.guess_type(path)[0] or "application/octet-stream"
        if path.endswith(".m3u8"): ctype = "application/vnd.apple.mpegurl"
        elif path.endswith(".ts"): ctype = "video/mp2t"
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers(); self.wfile.write(data)

    def do_GET(self):
        p = self.path.split("?", 1)[0]
        if p == "/health":
            return self._send(200, {"status": "ready", "busy": STATE["busy"]})
        if p == "/player.html" or p.startswith("/hls/"):
            return self._serve_static(p)
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            return self._send(400, {"error": f"bad json: {e}"})

        if self.path == "/gen":
            # required
            for key in ("ref_path", "audio_path", "out_path"):
                if not req.get(key):
                    return self._send(400, {"error": f"missing '{key}'"})
            ref = _resolve(req["ref_path"]); aud = _resolve(req["audio_path"])
            if not ref:
                return self._send(400, {"error": f"ref_path not found: {req['ref_path']}"})
            if not aud:
                return self._send(400, {"error": f"audio_path not found: {req['audio_path']}"})
            out = req["out_path"]
            out = out if os.path.isabs(out) else os.path.join(BASE, out)
            if not os.path.isdir(os.path.dirname(out)):
                return self._send(400, {"error": f"out_path dir missing: {os.path.dirname(out)}"})
            gain = float(req.get("gain", 1.0)); bias = str(req.get("bias", ""))
            for tok in [t for t in bias.split(",") if t.strip()]:
                if ":" not in tok or not (0 <= int(tok.split(":")[0]) < MOTION_DIM):
                    return self._send(400, {"error": f"bad bias '{tok}'; use 'axis:mag' with 0<=axis<{MOTION_DIM}"})
            common = dict(a_cfg_scale=float(req.get("a_cfg_scale", opt.a_cfg_scale)),
                          r_cfg_scale=float(req.get("r_cfg_scale", opt.r_cfg_scale)),
                          e_cfg_scale=float(req.get("e_cfg_scale", opt.e_cfg_scale)),
                          emo=req.get("emo", None), nfe=int(req.get("nfe", opt.nfe)),
                          no_crop=bool(req.get("no_crop", False)), seed=int(req.get("seed", opt.seed)))
            try:
                import time; t0 = time.time()
                with LOCK:
                    STATE["busy"] = True
                    if gain != 1.0 or bias.strip():          # manipulated path
                        agent.run_inference_manip(out, ref, aud, gain=gain, bias=bias, **common)
                    else:
                        agent.run_inference(out, ref, aud, **common)
                STATE["busy"] = False
                return self._send(200, {"out_path": out, "seconds": round(time.time() - t0, 2),
                                        "gain": gain, "bias": bias, "emo": common["emo"]})
            except Exception as e:
                STATE["busy"] = False
                traceback.print_exc()
                return self._send(500, {"error": str(e)})

        if self.path == "/render":
            audio = _resolve(req.get("audio_path") or req.get("audio"))
            portrait = _resolve(req.get("ref_path") or req.get("portrait") or DEFAULT_PORTRAIT)
            if not audio:
                return self._send(400, {"error": f"audio not found: {req.get('audio_path')}"})
            if not portrait:
                return self._send(400, {"error": f"portrait not found: {req.get('ref_path')}"})
            STATE["job"] += 1
            job = STATE["job"]
            threading.Thread(target=_render_stream, args=(
                job, audio, portrait, req.get("nfe", opt.nfe), req.get("a_cfg_scale", opt.a_cfg_scale),
                req.get("seed", opt.seed), req.get("emo")), daemon=True).start()
            src = f"/hls/{job}/index.m3u8"
            return self._send(200, {"job": job, "playlist": src, "player": f"/player.html?src={src}"})

        return self._send(404, {"error": "not found"})


if __name__ == "__main__":
    print(f"serving on :{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
