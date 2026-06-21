#!/usr/bin/env python3
"""The neocortex: DeepSeek V4 Flash decision client (ADR-0003).

Stdlib-only (urllib) so it runs inside any container without extra deps. Reads
DEEPSEEK_API_KEY from env (sourced from the shared /srv/nvme-data/containers/.env).
"""
import json
import os
import urllib.request

BASE = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")


def decide(system, user, *, temperature=0.7, max_tokens=400, as_json=True, timeout=60):
    """One DeepSeek chat turn. Returns parsed JSON dict if as_json, else text."""
    key = os.environ["DEEPSEEK_API_KEY"]
    body = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if as_json:
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(
        f"{BASE}/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    txt = json.load(urllib.request.urlopen(req, timeout=timeout))["choices"][0]["message"]["content"]
    if not as_json:
        return txt
    try:
        return json.loads(txt)
    except Exception:
        s, e = txt.find("{"), txt.rfind("}")
        return json.loads(txt[s:e + 1])


if __name__ == "__main__":
    out = decide(
        "You are an NPC living in an apartment. Reply ONLY JSON.",
        "You can see: fridge, kitchen counter, sofa, bookshelf, potted plant. "
        "You are a little hungry. Choose ONE visible thing to walk to next and why. "
        'Respond {"target": "<one item>", "reason": "<short>"}.')
    print("BRAIN_OK:", json.dumps(out))
