#!/usr/bin/env python3
"""Live 3D teeth tuner (viser). Theo's talking head renders in the BROWSER (three.js); the
teeth mesh updates instantly as you drag sliders (no server re-render), and you can orbit/
zoom. Scrub or play the talking clip. Writes render/mouth_params.json -> scene_cine --reuse.

  render/teeth_viser.sh    # then open http://<host>:8772
"""
import os
import sys
import json
import threading
import time
import numpy as np
import torch
import smplx
import trimesh
import viser

sys.path.insert(0, "/work"); sys.path.insert(0, "/lw/render")
from face_drive import drive
from face.talk import eyelid_upper_indices, apply_blink, lip_region, apply_lips
from mouth_parts import build_mouth, load_params

ARKIT = "/work/output/scene/theo_0.arkit.json"
PARAMS_OUT = "/lw/render/mouth_params.json"
FPS = 24
SKIN = (150, 110, 86)

print("baking Theo...", flush=True)
arkit = json.load(open(ARKIT))
F = max(2, round(len(arkit["weights"]) / float(arkit["fps"]) * FPS))
model = smplx.create("/work/models", model_type="smplx", gender="male", num_betas=10,
                     use_pca=False, flat_hand_mean=True, batch_size=F)
betas = np.zeros((1, 10), np.float32)
face = drive(arkit, F, FPS)
rest = np.zeros((1, 21, 3), np.float32); rest[0, 15] = [0, 0, -1.0]; rest[0, 16] = [0, 0, 1.0]
out = model(betas=torch.from_numpy(np.tile(betas, (F, 1))), global_orient=torch.zeros((F, 3)),
            body_pose=torch.from_numpy(np.tile(rest.reshape(1, 63), (F, 1))),
            jaw_pose=torch.from_numpy(face["jaw"]),
            leye_pose=torch.from_numpy(face["leye"]), reye_pose=torch.from_numpy(face["reye"]))
VERTS = out.vertices.detach().numpy().astype(np.float32)
li, ri = eyelid_upper_indices(model, betas[0])
apply_blink(VERTS, li, ri, face["blink_l"], face["blink_r"])
apply_lips(VERTS, lip_region(model, betas[0]), face["mouth_close"], face["mouth_pucker"])
FACES = model.faces.astype(np.int32)
LIP = lip_region(model, betas[0])["idx"]
JAW = face["jaw"][:, 0]
PEAK = int(JAW.argmax())
# centre + scale so the head sits nicely / coords in metres are small -> fine for viser
CTR = VERTS[PEAK].mean(0)

srv = viser.ViserServer(host="0.0.0.0", port=8772)
srv.scene.set_up_direction("+y")

GUI = {}
PARTS = ["upper", "lower", "tongue", "cavity"]
GFIELDS = [("width", 0.0, 1.0, 0.01), ("y_drop", -0.03, 0.03, 0.001),
           ("anchor_up", -0.02, 0.02, 0.001), ("drop_scale", 0.0, 0.15, 0.005),
           ("gate", 0.0, 0.5, 0.01)]
PFIELDS = [("y0", -0.03, 0.03, 0.0005), ("y1", -0.03, 0.03, 0.0005),
           ("z", -0.05, 0.01, 0.001), ("w", 0.2, 1.6, 0.01)]
init = load_params()

state = {"frame": PEAK, "play": False, "mvs": None, "mf": None, "mcol": None}


def collect():
    p = {k: GUI[k].value for k, *_ in GFIELDS}
    for part in PARTS:
        p[part] = {f: GUI[f"{part}.{f}"].value for f, *_ in PFIELDS}
        p[part]["color"] = list(GUI[f"{part}.color"].value)
    return p


def rebuild_teeth():
    p = collect()
    json.dump({**load_params(), **p}, open(PARAMS_OUT, "w"), indent=2)
    state["mvs"], state["mf"], state["mcol"] = build_mouth(VERTS, LIP, JAW, yaw_rad=0.0, params=p)
    show(state["frame"])


def show(fi):
    state["frame"] = fi
    srv.scene.add_mesh_simple("/theo", VERTS[fi] - CTR, FACES, color=SKIN,
                              flat_shading=False, side="double")
    if state["mvs"] is not None and JAW[fi] > GUI["gate"].value:
        mt = trimesh.Trimesh(state["mvs"][fi] - CTR, state["mf"],
                             vertex_colors=np.asarray(state["mcol"], np.uint8), process=False)
        state["teeth_h"] = srv.scene.add_mesh_trimesh("/teeth", mt)
    elif state.get("teeth_h") is not None:
        state["teeth_h"].remove(); state["teeth_h"] = None


# ---- GUI ----
with srv.gui.add_folder("global"):
    for k, lo, hi, st in GFIELDS:
        GUI[k] = srv.gui.add_slider(k, lo, hi, st, float(init[k]))
        GUI[k].on_update(lambda _: rebuild_teeth())
for part in PARTS:
    with srv.gui.add_folder(part):
        for f, lo, hi, st in PFIELDS:
            GUI[f"{part}.{f}"] = srv.gui.add_slider(f, lo, hi, st, float(init[part][f]))
            GUI[f"{part}.{f}"].on_update(lambda _: rebuild_teeth())
        GUI[f"{part}.color"] = srv.gui.add_rgb("color", tuple(int(c) for c in init[part]["color"]))
        GUI[f"{part}.color"].on_update(lambda _: rebuild_teeth())
with srv.gui.add_folder("playback"):
    g_frame = srv.gui.add_slider("frame", 0, F - 1, 1, PEAK)
    g_play = srv.gui.add_checkbox("play", False)
    g_frame.on_update(lambda _: show(g_frame.value))

rebuild_teeth()


def loop():
    while True:
        if g_play.value:
            nf = (state["frame"] + 1) % F
            g_frame.value = nf
            show(nf)
            time.sleep(1.0 / FPS)
        else:
            time.sleep(0.05)


threading.Thread(target=loop, daemon=True).start()
print(f"teeth viser LIVE -> http://0.0.0.0:8772  (F={F}, peak={PEAK})", flush=True)
while True:
    time.sleep(1.0)
