#!/usr/bin/env python3
"""Real-time teeth tuner. Runs INSIDE the GPU container (lifeworld-preview): loads the body
once, keeps the textured mesh + pyrender scene warm, and on each change rebuilds ONLY the
tiny mouth mesh and redraws -> ~0.1s per update (vs ~13s for the docker-per-render version).

Launch via render/teeth_ui.sh, then open http://<host>:8771. Writes render/mouth_params.json
(also drives the cinematic: scene_cine.py --reuse).
"""
import os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import io
import json
import sys
import time
import numpy as np
import torch
import smplx
import pyrender
import trimesh
from PIL import Image, ImageDraw
from http.server import BaseHTTPRequestHandler, HTTPServer   # single-thread: GL ctx is thread-affine

sys.path.insert(0, "/work"); sys.path.insert(0, "/lw/render")
from face_drive import drive
from face.talk import eyelid_upper_indices, apply_blink, lip_region, apply_lips
from render_smplx import frame_camera, _aim, build_mesh
from mouth_parts import build_mouth, load_params

PARAMS = "/lw/render/mouth_params.json"
ARKIT = "/a2f/test.arkit.json"
PORT = 8771
FPS = 24
VIEWS = [(0, "front"), (16, "3/4")]
TEX = {"female": "/work/assets/smplx_texture_f_alb_eyefix.png",
       "male": "/work/assets/smplx_texture_m_alb_eyefix.png"}
UV = "/work/assets/smplx_uv_2023.npz"

_r = pyrender.OffscreenRenderer(512, 512)
_uv = np.load(UV)["uv_coordinates"]
_state = {}   # gender -> warm per-view state


def build_state(gender):
    """One-time per gender: bake the body (peak frame) for each view; warm scene + body mesh."""
    arkit = json.load(open(ARKIT))
    F = max(2, round(len(arkit["weights"]) / float(arkit["fps"]) * FPS))
    model = smplx.create("/work/models", model_type="smplx", gender=gender, num_betas=10,
                         use_pca=False, flat_hand_mean=True, batch_size=F)
    betas = np.zeros((1, 10), np.float32)
    face = drive(arkit, F, FPS)
    peak = int(face["jaw"][:, 0].argmax())
    tex = Image.open(TEX[gender]).convert("RGB").resize((1024, 1024), Image.LANCZOS)
    rest = np.zeros((1, 21, 3), np.float32); rest[0, 15] = [0, 0, -1.0]; rest[0, 16] = [0, 0, 1.0]
    lip_idx = lip_region(model, betas[0])["idx"]
    views = []
    for yaw_deg, label in VIEWS:
        yaw = np.radians(yaw_deg)
        out = model(betas=torch.from_numpy(np.tile(betas, (F, 1))),
                    global_orient=torch.from_numpy(np.tile([[0.0, yaw, 0.0]], (F, 1)).astype(np.float32)),
                    body_pose=torch.from_numpy(np.tile(rest.reshape(1, 63), (F, 1))),
                    jaw_pose=torch.from_numpy(face["jaw"]),
                    leye_pose=torch.from_numpy(face["leye"]), reye_pose=torch.from_numpy(face["reye"]))
        v = out.vertices.detach().numpy().astype(np.float32)
        li, ri = eyelid_upper_indices(model, betas[0])
        apply_blink(v, li, ri, face["blink_l"], face["blink_r"])
        apply_lips(v, lip_region(model, betas[0]), face["mouth_close"], face["mouth_pucker"])
        yfov, cam_pose, center = frame_camera(v, "head", 1.0)
        scene = pyrender.Scene(bg_color=[0.05, 0.05, 0.07, 1.0], ambient_light=[0.4, 0.4, 0.42])
        scene.add(build_mesh(v[peak], model.faces.astype(np.int64), _uv, tex))   # body: texture uploaded once
        scene.add(pyrender.PerspectiveCamera(yfov=yfov, aspectRatio=1.0), pose=cam_pose)
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=3.2), pose=_aim([1, 1, 2], center))
        scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.4), pose=_aim([-2, 1, 1], center))
        views.append(dict(yaw=yaw, label=label, verts=v, scene=scene, mouth_node=None))
    _state[gender] = dict(F=F, peak=peak, jaw=face["jaw"][:, 0], lip_idx=lip_idx, views=views)
    return _state[gender]


def render(params, gender):
    st = _state.get(gender) or build_state(gender)
    gate = params.get("gate", 0.12)
    tiles = []
    for vw in st["views"]:
        mv, mf, mcol = build_mouth(vw["verts"], st["lip_idx"], st["jaw"],
                                   yaw_rad=vw["yaw"], params=params)
        if vw["mouth_node"] is not None:
            vw["scene"].remove_node(vw["mouth_node"]); vw["mouth_node"] = None
        if st["jaw"][st["peak"]] > gate:
            mt = trimesh.Trimesh(mv[st["peak"]], mf, vertex_colors=mcol, process=False)
            vw["mouth_node"] = vw["scene"].add(pyrender.Mesh.from_trimesh(mt, smooth=False))
        im = Image.fromarray(_r.render(vw["scene"])[0])
        W, H = im.size
        crop = im.crop((int(W * 0.20), int(H * 0.52), int(W * 0.80), int(H * 0.92))).resize((420, 400))
        ImageDraw.Draw(crop).text((6, 6), vw["label"], fill=(255, 255, 0))
        tiles.append(crop)
    sheet = Image.new("RGB", (420 * 2, 400))
    sheet.paste(tiles[0], (0, 0)); sheet.paste(tiles[1], (420, 0))
    buf = io.BytesIO(); sheet.save(buf, "PNG"); return buf.getvalue()


PAGE = open("/lw/render/teeth_ui_page.html").read()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, ctype, body):
        self.send_response(code); self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store"); self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            cur = load_params()
            page = (PAGE.replace("__PARAMS__", json.dumps(cur))
                        .replace("__GLOBALS__", json.dumps(GLOBALS))
                        .replace("__FIELDS__", json.dumps(FIELDS))
                        .replace("__PARTS__", json.dumps(PARTS)))
            self._send(200, "text/html; charset=utf-8", page.encode())
        else:
            self._send(404, "text/plain", b"404")

    def do_POST(self):
        if self.path != "/render":
            return self._send(404, "text/plain", b"404")
        n = int(self.headers.get("Content-Length", 0))
        try:
            p = json.loads(self.rfile.read(n))
            gender = p.pop("_gender", "female")
            cur = load_params()
            for k, v in p.items():
                if isinstance(v, dict) and isinstance(cur.get(k), dict):
                    cur[k].update(v)
                else:
                    cur[k] = v
            json.dump({k: cur[k] for k in cur}, open(PARAMS, "w"), indent=2)
            t = time.time()
            png = render(cur, gender)
            self.send_response(200); self.send_header("Content-Type", "image/png")
            self.send_header("X-Render-ms", str(round((time.time() - t) * 1000)))
            self.send_header("Content-Length", str(len(png)))
            self.send_header("Cache-Control", "no-store"); self.end_headers()
            self.wfile.write(png)
        except Exception as e:
            self._send(200, "text/plain", f"ERR {e}".encode())


GLOBALS = [("width", 0, 1, 0.01), ("y_drop", -0.03, 0.03, 0.001),
           ("anchor_up", -0.02, 0.02, 0.001), ("drop_scale", 0, 0.15, 0.005),
           ("gate", 0, 0.5, 0.01)]
FIELDS = [("y0", -0.03, 0.03, 0.0005), ("y1", -0.03, 0.03, 0.0005),
          ("z", -0.05, 0.01, 0.001), ("w", 0.2, 1.6, 0.01)]
PARTS = ["upper", "lower", "tongue", "cavity"]

if __name__ == "__main__":
    print("warming up (female)...", flush=True); build_state("female")
    print(f"teeth tuner LIVE -> http://0.0.0.0:{PORT}", flush=True)
    HTTPServer(("0.0.0.0", PORT), H).serve_forever()
