"""Minimal OpenAI-compatible chat client (works for DeepSeek cloud and local vLLM/llama.cpp).
Stdlib only. JSON-mode helper with one retry + cache."""
import json
import time
import urllib.request
from .. import config, cache


class CompletionLengthError(RuntimeError):
    """The model exhausted its output budget before producing a complete answer."""


def chat(tier: str, system: str, user: str, *, temperature=0.4, max_tokens=4000,
         json_mode=False, use_cache=True, retries=3):
    cfg = config.LLM[tier]
    ck = cache.key(
        "llm", tier, cfg["model"], system, user, temperature, max_tokens, json_mode
    )
    cp = cache.path("llm", ck, "json")
    if use_cache and cache.have(cp):
        return cache.load_json(cp)["content"]

    body = {
        "model": cfg["model"],
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if cfg["model"] in {"deepseek-v4-flash", "deepseek-v4-pro"}:
        # DeepSeek V4 thinking ignores sampling controls and counts reasoning
        # against max_tokens. Only final content is a Docupipe artifact.
        body.pop("temperature")
        body["thinking"] = {"type": "enabled"}
        body["reasoning_effort"] = "high"
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    data = json.dumps(body).encode()
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                cfg["base_url"].rstrip("/") + "/chat/completions", data=data,
                headers={"Content-Type": "application/json",
                         "User-Agent": config.HTTP_UA,
                         "Authorization": "Bearer " + (cfg["key"] or "x")})
            with urllib.request.urlopen(req, timeout=180) as r:
                out = json.load(r)
            choice = out["choices"][0]
            reason = choice.get("finish_reason", "unknown")
            if reason == "length":
                raise CompletionLengthError(
                    f"completion exhausted max_tokens={max_tokens} before finishing"
                )
            content = choice["message"]["content"]
            if not (content or "").strip():
                raise RuntimeError(f"empty completion (finish_reason={reason})")
            if use_cache:
                cache.save_json(cp, {"content": content})
            return content
        except CompletionLengthError:
            raise
        except Exception as e:  # noqa
            last = e
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"LLM {tier} failed after {retries}: {last}")


def _parse_json(txt: str):
    txt = (txt or "").strip()
    if txt.startswith("```"):
        txt = txt.split("```", 2)[1]
        txt = txt[4:] if txt.lower().startswith("json") else txt
    try:
        return json.loads(txt)
    except Exception:
        s, e = txt.find("{"), txt.rfind("}")
        if s >= 0 and e > s:
            return json.loads(txt[s:e + 1])
        raise


def chat_json(tier: str, system: str, user: str, **kw):
    """Return parsed JSON; tolerates ```fences; retries a bad/empty parse without cache."""
    for attempt in range(3):
        try:
            return _parse_json(chat(tier, system, user, json_mode=True,
                                    use_cache=(kw.get("use_cache", True) and attempt == 0),
                                    **{k: v for k, v in kw.items() if k != "use_cache"}))
        except CompletionLengthError:
            raise
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)
