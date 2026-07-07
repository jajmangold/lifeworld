"""Klein-4B instruction image-edit client. Posts an image + instruction to the resident klein-proxy
(FLUX.2 Klein-4B sd.cpp) and writes the edited PNG. Run from a container on the lm-stack_ai network:
  docker run --rm --network lm-stack_ai -v $DIR:/s --entrypoint python3 klein-proxy:1.0 \
      /s/klein_edit.py <in.png> "<instruction>" <out.png>
Inputs are auto-fit to 768 by the proxy. ~50s/edit on Volta.
"""
import sys, base64, httpx

inp, prompt, out = sys.argv[1], sys.argv[2], sys.argv[3]
with open(inp, "rb") as f:
    files = {"image": (inp.split("/")[-1], f.read(), "image/png")}
data = {"prompt": prompt, "n": "1", "size": "auto"}
r = httpx.post("http://klein-proxy:9000/v1/images/edits", files=files, data=data, timeout=600)
r.raise_for_status()
j = r.json()
b64 = j["data"][0]["b64_json"]
with open(out, "wb") as f:
    f.write(base64.b64decode(b64))
print("KLEIN_OK ->", out)
