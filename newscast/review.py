"""Vision QA review via the qwen9b vision model on amd0 (https://amd0.python-bull.ts.net/v1, OpenAI-compatible).
Sends a SPECIFIC per-artifact checklist (not a vague "critique") and Qwen's recommended sampling for
instruct VLMs (temp 0.7 / top_p 0.8 / top_k 20 / presence_penalty 1.5 / repeat 1.0 — never greedy, which
degenerates into repetition loops). Prints a structured PASS/FAIL verdict + score + top fix.
  python3 review.py <image.png> [--type anchor|face|backdrop|screen|title|vo|generic] ["extra context"]
"""
import sys, os, base64, json, urllib.request
QWEN_URL   = os.environ.get("QWEN_URL", "https://amd0.python-bull.ts.net/v1/chat/completions")  # qwen9b, all vision-QA+chat
QWEN_MODEL = os.environ.get("QWEN_MODEL", "Qwen3.5-9B-UD-Q4_K_XL.gguf")

CHECKS = {
 "face": ("This image is an AI-generated headshot used ONLY as a face-swap SOURCE for a TV reporter.",
   ["Exactly ONE face, fully visible (forehead + chin not cropped)",
    "FRONT-facing — looking straight at camera, not turned or in profile",
    "Eyes open, gaze forward; mouth closed/neutral expression",
    "Evenly lit, no harsh shadow crossing the face",
    "Nothing (hair/hand/glasses) covering eyes, nose or mouth",
    "No AI artifacts on eyes/teeth/ear/hairline that would break a swap"]),
 "anchor": ("This is a finished broadcast news ANCHOR frame (anchor person + on-set graphics).",
   ["Head fully in frame with natural HEADROOM (top of hair not cropped)",
    "Face is PHOTOREAL — not waxy/plastic/melted; eyes and mouth look real",
    "No artifacts/smudge/checker pattern on the shirt, collar or shoulders",
    "Clean silhouette edge — no halo/outline/fringe around hair or shoulders",
    "Any on-screen wall/graphic text is legible AND correctly spelled",
    "Lower-third + ticker readable, aligned, not overlapping each other"]),
 "backdrop": ("This is a DEFOCUSED background plate for compositing (no subject in it yet).",
   ["No readable or garbled text/signage", "No people or faces present",
    "Coherent, plausible real-world scene", "No AI melt/duplication/warping artifacts",
    "Appropriately soft/defocused for a background layer"]),
 "screen": ("This is an on-set news video-wall / story graphic.",
   ["Headline correctly spelled and legible", "Kicker/label reads clearly",
    "Layout balanced, nothing clipped at the edges", "Brand colours consistent (navy/red)",
    "No garbled AI text anywhere"]),
 "title": ("This is a broadcast TITLE / bumper card.",
   ["Network logo clean (no checkerboard/box artifact around it)",
    "Title text spelled correctly and legible", "Balanced composition", "Brand-consistent colours"]),
 "vo": ("This is a full-screen b-roll + voiceover news frame with graphics.",
   ["B-roll reads as real footage, no bad AI artifacts", "Headline bar spelled correctly + legible",
    "Headline bar does NOT overlap the ticker", "Bug + ticker readable"]),
 "generic": ("This is a broadcast news frame.",
   ["Looks broadcast-professional", "No visible artifacts", "Any text is legible and correct"]),
}

img_p = sys.argv[1]
typ, extra = "generic", ""
rest = sys.argv[2:]
i = 0
while i < len(rest):
    if rest[i] == "--type" and i + 1 < len(rest):
        typ = rest[i + 1]; i += 2
    else:
        extra = rest[i]; i += 1
desc, items = CHECKS.get(typ, CHECKS["generic"])
checklist = "\n".join(f"{n+1}. {c}" for n, c in enumerate(items))
prompt = (f"{desc} {extra}\n\nInspect EACH numbered item and answer PASS or FAIL with a <=12-word reason:\n"
          f"{checklist}\n\nThen output exactly two final lines:\nSCORE: <n>/10\n"
          "FIX: <the single most important thing to fix, or 'none'>\n\n"
          "Judge only what you can actually see in the image; do not invent problems.")
b64 = base64.b64encode(open(img_p, "rb").read()).decode()
body = json.dumps({
    "model": QWEN_MODEL, "chat_template_kwargs": {"enable_thinking": False},
    "temperature": 0.7, "top_p": 0.8, "top_k": 20,
    "presence_penalty": 1.5, "repeat_penalty": 1.0, "max_tokens": 500,
    "messages": [{"role": "user", "content": [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64," + b64}}]}]}).encode()
req = urllib.request.Request(QWEN_URL, data=body,
                             headers={"Content-Type": "application/json"})
print(json.load(urllib.request.urlopen(req, timeout=180))["choices"][0]["message"]["content"])
