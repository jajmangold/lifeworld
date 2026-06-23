#!/usr/bin/env python3
"""Resident FLOAT talking-head server: loads the model ONCE (float.pth + wav2vec + face_alignment),
then serves talking-head generation over HTTP so each request skips the ~cold-start load.
  GET  /health                      -> {"status":"ready"}
  POST /gen  {"ref_path","audio_path","out_path", nfe?, a_cfg_scale?, e_cfg_scale?, seed?, no_crop?, emo?}
Paths are container-side (mount a shared dir). Inference is serialized (one GPU)."""
import sys, os, json, threading, traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CKPT = os.environ.get("FLOAT_CKPT", "./checkpoints/float.pth")
PORT = int(os.environ.get("FLOAT_PORT", "8210"))
sys.argv = ["float_server", "--ckpt_path", CKPT]          # feed InferenceOptions a clean argv
from generate import InferenceAgent, InferenceOptions

opt = InferenceOptions().parse()
opt.rank, opt.ngpus = 0, 1
agent = InferenceAgent(opt)                               # loads model once (~cold start here)
LOCK = threading.Lock()
print("FLOAT_SERVER_READY", flush=True)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        self._send(200, {"status": "ready"}) if self.path == "/health" else self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/gen":
            return self._send(404, {"error": "not found"})
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
            ref, aud, out = req["ref_path"], req["audio_path"], req["out_path"]
            import time; t0 = time.time()
            with LOCK:                                    # one GPU -> serialize
                agent.run_inference(
                    out, ref, aud,
                    a_cfg_scale=float(req.get("a_cfg_scale", opt.a_cfg_scale)),
                    r_cfg_scale=float(req.get("r_cfg_scale", opt.r_cfg_scale)),
                    e_cfg_scale=float(req.get("e_cfg_scale", opt.e_cfg_scale)),
                    emo=req.get("emo", None),
                    nfe=int(req.get("nfe", opt.nfe)),
                    no_crop=bool(req.get("no_crop", False)),
                    seed=int(req.get("seed", opt.seed)),
                )
            self._send(200, {"out_path": out, "seconds": round(time.time() - t0, 2)})
        except Exception as e:
            traceback.print_exc()
            self._send(500, {"error": str(e)})


ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()
