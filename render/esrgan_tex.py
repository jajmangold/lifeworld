"""ESRGAN 4x-UltraSharp upscale of character BASE-COLOR textures (albedo only) via the resident esrgan
ComfyUI container (:8197). Adds real fabric/skin micro-detail (unlike a plain resize). Writes up_<name>.png.
  python3 esrgan_tex.py <texdir> <name1.png> [name2.png ...]
"""
import os, sys, json, time, urllib.request, subprocess
ESRGAN = "http://127.0.0.1:8197"; MODEL = "4x-UltraSharp.pth"

def esrgan4x(local_png, out_png):
    base = os.path.basename(local_png)
    subprocess.run(f"docker cp {local_png} esrgan:/root/ComfyUI/input/{base}", shell=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    graph = {"1": {"class_type": "LoadImage", "inputs": {"image": base}},
             "2": {"class_type": "UpscaleModelLoader", "inputs": {"model_name": MODEL}},
             "3": {"class_type": "ImageUpscaleWithModel", "inputs": {"upscale_model": ["2", 0], "image": ["1", 0]}},
             "4": {"class_type": "SaveImage", "inputs": {"images": ["3", 0], "filename_prefix": "tex_up"}}}
    req = urllib.request.Request(ESRGAN + "/prompt", data=json.dumps({"prompt": graph}).encode(),
                                 headers={"Content-Type": "application/json"})
    pid = json.load(urllib.request.urlopen(req, timeout=30))["prompt_id"]
    for _ in range(150):
        try:
            h = json.load(urllib.request.urlopen(ESRGAN + f"/history/{pid}", timeout=15))
            if pid in h:
                for node in h[pid].get("outputs", {}).values():
                    for im in node.get("images", []):
                        sub = (im.get("subfolder") or "")
                        src = f"/root/ComfyUI/output/{sub + '/' if sub else ''}{im['filename']}"
                        subprocess.run(f"docker cp esrgan:{src} {out_png}", shell=True,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        return os.path.exists(out_png)
        except Exception:
            pass
        time.sleep(2)
    return False

texdir = sys.argv[1]
for name in sys.argv[2:]:
    src = os.path.join(texdir, name); out = os.path.join(texdir, "up_" + name)
    print(f"esrgan 4x {name} ...", flush=True)
    print("  OK ->", out if esrgan4x(src, out) else "FAILED")
