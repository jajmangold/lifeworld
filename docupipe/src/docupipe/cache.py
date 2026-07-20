"""Content-addressed disk cache — makes every expensive node idempotent + resumable."""
import hashlib
import json
import os
from . import config


def key(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(repr(p).encode())
    return h.hexdigest()[:20]


def path(kind: str, k: str, ext: str) -> str:
    d = os.path.join(config.CACHE_DIR, kind)
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{k}.{ext}")


def have(p: str) -> bool:
    return os.path.exists(p) and os.path.getsize(p) > 0


def atomic_write_bytes(p: str, data: bytes):
    tmp = p + ".tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, p)   # atomic: a crash never leaves a half-written file the existence check trusts


def load_json(p: str):
    return json.load(open(p, encoding="utf-8")) if have(p) else None


def save_json(p: str, obj):
    atomic_write_bytes(p, json.dumps(obj, ensure_ascii=False, indent=2).encode())
