"""Z-Image txt2img client (A1111 /sdapi). Run in a container on lm-stack_ai.
  python3 zgen.py "<prompt>" "<neg>" <w> <h> <out.png> <seed>"""
import sys, base64, json, urllib.request
p, neg, w, h, out, seed = sys.argv[1:7]
body = json.dumps({"prompt": p, "negative_prompt": neg, "width": int(w), "height": int(h),
                   "steps": 8, "cfg_scale": 1.0, "seed": int(seed)}).encode()
req = urllib.request.Request("http://zimage:9000/sdapi/v1/txt2img", data=body,
                             headers={"Content-Type": "application/json"})
img = json.load(urllib.request.urlopen(req, timeout=240))["images"][0]
open(out, "wb").write(base64.b64decode(img.split(",",1)[-1] if "," in img else img))
print("ZGEN_OK", out)
