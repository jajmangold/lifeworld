"""Vision review: send an image (+ context) to the resident Qwen vision model (qwen9b-vis, llama.cpp
:8021, OpenAI-compatible) and print its critique. Used as an automated QA eye on our renders/frames.
  python3 review.py <image.png> "<what this is / what to check>"
"""
import sys, base64, json, urllib.request
img_p, ctx = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "Critique this broadcast news frame.")
b64 = base64.b64encode(open(img_p, "rb").read()).decode()
body = json.dumps({
    "model": "qwen", "temperature": 0.2, "max_tokens": 400,
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": ctx + "\nBe specific and concise. List concrete problems (artifacts, "
         "framing, text, realism) and a 1-10 broadcast-quality score. If it looks good, say so briefly."},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}]}]}).encode()
req = urllib.request.Request("http://localhost:8021/v1/chat/completions", data=body,
                             headers={"Content-Type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=180))
print(r["choices"][0]["message"]["content"])
