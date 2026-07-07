"""Qwen-Image-Edit instruction client — the replacement for viverse_avatar/klein_edit.py.

Posts image + instruction to the resident qwen-edit sd.cpp server (Qwen-Image-Edit-2511, 4-step
lightning baked, ~25s/edit) and writes the edited PNG. Hits the server DIRECTLY (not the :9013
qwen9b proxy) — callers here already phrase a subject-preserving instruction, so no rewrite needed.
Run from a container on the lm-stack_ai network (any image with httpx, e.g. qwen-edit-proxy:1.0):

  docker run --rm --network lm-stack_ai -v $DIR:/s --entrypoint python3 qwen-edit-proxy:1.0 \
      /s/qwen_edit_client.py <in.png> "<instruction>" <out.png>
"""
import os
import sys
import base64
import httpx

inp, prompt, out = sys.argv[1], sys.argv[2], sys.argv[3]
URL = os.environ.get("QWEN_EDIT_URL", "http://qwen-edit:9000/v1/images/edits")
SIZE = os.environ.get("QWEN_EDIT_SIZE", "512x512")

with open(inp, "rb") as f:
    files = {"image": (os.path.basename(inp), f.read(), "image/png")}
r = httpx.post(URL, files=files, data={"prompt": prompt, "size": SIZE}, timeout=600)
r.raise_for_status()
with open(out, "wb") as f:
    f.write(base64.b64decode(r.json()["data"][0]["b64_json"]))
print("QWEN_OK ->", out)
