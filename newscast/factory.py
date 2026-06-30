"""Clip factory — produce many anchor segments fast by PIPELINING across our 3 scarce resources
(rtx0 GPU0 = render, rtx0 GPU1 = FlashVSR premium, V100 box = swap+muse servers). Each segment still
runs make_anchor.sh end-to-end, but make_anchor wraps its 4 GPU/server stages in per-resource flocks
(FLOCK_DIR), so the kernel serializes each resource while DIFFERENT segments occupy DIFFERENT resources
at once. Per-segment frame dirs (ANCHOR_OUT) keep concurrent renders from clobbering each other.
Steady-state throughput ~= the bottleneck stage (~render or ~premium) per clip, not the 12-min sum.

Manifest (JSON):
  { "defaults": {"premium": true, "screen": "output/screen_graphic_f.png", "mood": "serious"},
    "segments": [ {"name":"seg01","audio":"output/seg01.wav"},
                  {"name":"seg02","audio":"output/seg02.wav","screen":"output/seg02_screen.png"} ] }

  python3 newscast/factory.py <manifest.json> [--max N] [--stagger S]
"""
import sys, json, os, time, subprocess, threading
from concurrent.futures import ThreadPoolExecutor, as_completed

BOT = "/srv/nvme-data/containers/projects/bot"
FLOCK_DIR = "/tmp/clipfactory"
man = json.load(open(sys.argv[1]))
defaults = man.get("defaults", {})
segs = man["segments"]
MAXN = int(sys.argv[sys.argv.index("--max")+1]) if "--max" in sys.argv else 5
STAGGER = float(sys.argv[sys.argv.index("--stagger")+1]) if "--stagger" in sys.argv else 20.0

os.makedirs(FLOCK_DIR, exist_ok=True)
for r in ("render", "premium", "swap", "muse"):
    open(os.path.join(FLOCK_DIR, r + ".lock"), "a").close()
# clean slate: kill any stray FlashVSR head_up left on rtx0 GPU1 by a crashed/manual run — a leftover
# squatting on GPU1 OOMs the next premium. Safe at factory start (no legit head_up should be running).
subprocess.call(["ssh", "-o", "BatchMode=yes", "josh@rtx0", "pkill -9 -f head_up 2>/dev/null; true"])

_lt = threading.Lock(); _next = [0.0]
def staggered_start():
    with _lt:
        now = time.time()
        wait = max(0.0, _next[0] - now)
        _next[0] = max(now, _next[0]) + STAGGER
    if wait: time.sleep(wait)

def build_cmd(seg):
    g = dict(defaults); g.update(seg)
    name = g["name"]; out = g.get("out", f"output/{name}_final.mp4")
    cmd = ["bash", "make_anchor.sh", "--audio", g["audio"], "--out", out]
    if g.get("premium"): cmd += ["--premium"]
    if g.get("screen"):  cmd += ["--screen", g["screen"]]
    for k in ("mood", "brow", "nod", "seed", "restore", "face", "headtex", "ots"):
        if k in g: cmd += [f"--{k}", str(g[k])]
    return name, out, cmd

def run_segment(seg):
    name, out, cmd = build_cmd(seg)
    staggered_start()
    env = dict(os.environ, FLOCK_DIR=FLOCK_DIR, ANCHOR_OUT=f"/work/output/anchor_anim_{name}/")
    t0 = time.time()
    log = open(os.path.join(BOT, f"output/factory_{name}.log"), "w")
    print(f"[factory] start {name}", flush=True)
    rc = subprocess.call(cmd, cwd=BOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    print(f"[factory] {'OK' if rc==0 else 'FAIL(%d)'%rc} {name} in {dt/60:.1f}min -> {out}", flush=True)
    return name, out, rc, dt

if __name__ == "__main__":
    T0 = time.time()
    results = []
    with ThreadPoolExecutor(max_workers=MAXN) as ex:
        futs = [ex.submit(run_segment, s) for s in segs]
        for f in as_completed(futs):
            results.append(f.result())
    wall = time.time() - T0
    serial = sum(r[3] for r in results)
    ok = sum(1 for r in results if r[2] == 0)
    print(f"\n[factory] {ok}/{len(results)} ok | wall {wall/60:.1f}min | "
          f"sum-of-segments {serial/60:.1f}min | speedup {serial/max(wall,1):.2f}x")
    for n, o, rc, dt in sorted(results):
        print(f"  {'ok ' if rc==0 else 'ERR'} {n:16s} {dt/60:5.1f}min  {o}")
