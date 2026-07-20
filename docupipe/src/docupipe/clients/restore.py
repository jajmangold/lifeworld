"""Archival restoration via the resident ESRGAN ComfyUI server (:8197).
Structure-preserving upscale of small/soft stills BEFORE Ken-Burns (the pan magnifies
softness). Gated by size; resilient (returns the original on any failure)."""
import json
import os
import struct
import time
import urllib.request
from .. import config, cache

UPSCALE_MODEL = "4x-UltraSharp.pth"
_CONTAINER_OUT = "/root/ComfyUI/output"   # we read results via `docker cp`


def _png_dims(path):
    try:
        with open(path, "rb") as f:
            head = f.read(33)
        if head[:8] == b"\x89PNG\r\n\x1a\n":
            return struct.unpack(">II", head[16:24])
    except Exception:
        pass
    return None


def _img_min_dim(path):
    # cheap: use ffprobe for any format
    import subprocess
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", path],
                           capture_output=True, text=True, timeout=20)
        w, h = (int(x) for x in r.stdout.strip().split("x")[:2])
        return min(w, h)
    except Exception:
        return 9999


def _post(graph):
    data = json.dumps({"prompt": graph}).encode()
    req = urllib.request.Request(config.ESRGAN_URL.rstrip("/") + "/prompt", data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)["prompt_id"]


def _wait(pid, timeout=180):
    t = time.time()
    while time.time() - t < timeout:
        try:
            with urllib.request.urlopen(config.ESRGAN_URL.rstrip("/") + f"/history/{pid}",
                                        timeout=15) as r:
                h = json.load(r)
            if pid in h:
                outs = h[pid].get("outputs", {})
                for node in outs.values():
                    for im in node.get("images", []):
                        return im["filename"], im.get("subfolder", "")
        except Exception:
            pass
        time.sleep(2)
    return None, None


def upscale(image: str, min_dim_threshold=1100) -> str:
    """4x upscale `image` if it's small; else return as-is. Cached. ESRGAN runs INSIDE
    ComfyUI's container, so we copy the image in and the result out via docker cp."""
    if _img_min_dim(image) >= min_dim_threshold:
        return image
    out = cache.path("upscaled", cache.key("up", image, os.path.getsize(image)), "png")
    if cache.have(out):
        return out
    base = os.path.basename(image)
    try:
        # stage the input into the esrgan container's input dir
        os.system(f"docker cp {image} esrgan:/root/ComfyUI/input/{base} >/dev/null 2>&1")
        graph = {
            "1": {"class_type": "LoadImage", "inputs": {"image": base}},
            "2": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": UPSCALE_MODEL}},
            "3": {"class_type": "ImageUpscaleWithModel",
                  "inputs": {"upscale_model": ["2", 0], "image": ["1", 0]}},
            "4": {"class_type": "SaveImage", "inputs": {"images": ["3", 0],
                                                        "filename_prefix": "docupipe_up"}},
        }
        pid = _post(graph)
        fn, sub = _wait(pid)
        if not fn:
            return image
        src = f"{_CONTAINER_OUT}/{sub + '/' if sub else ''}{fn}"
        rc = os.system(f"docker cp esrgan:{src} {out} >/dev/null 2>&1")
        return out if rc == 0 and cache.have(out) else image
    except Exception:
        return image
