"""CLI: run a docuseries job end-to-end with SQLite checkpointing + script-approval gate.

  python -m docupipe.run --topic "The Axeman of New Orleans" --auto
  python -m docupipe.run --resume <job_id>            # continue an interrupted job
  python -m docupipe.run --resume <job_id> --approve  # approve the script & finish
"""
import argparse
import json
import os
import re
import sys
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command
from . import config
from .graph import build

DB = os.path.join(config.DATA, "docupipe.sqlite")


def _slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")[:48]


def _drain(graph, payload, cfg):
    """Stream node updates, return True if paused on an interrupt.
    (Per-node progress is printed directly by nodes._progress; the multi-mode
    stream_mode list hangs in langgraph 1.2.7, so we use the single 'updates' mode.)"""
    interrupted = False
    for chunk in graph.stream(payload, cfg, stream_mode="updates", durability="sync"):
        for node in chunk:
            if node == "__interrupt__":
                interrupted = True
            else:
                print(f"[node] {node} done", flush=True)
    return interrupted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic")
    ap.add_argument("--job")
    ap.add_argument("--resume")
    ap.add_argument("--auto", action="store_true", help="auto-approve the script")
    ap.add_argument("--approve", action="store_true", help="approve a paused job and finish")
    args = ap.parse_args()

    with SqliteSaver.from_conn_string(DB) as cp:
        graph = build().compile(checkpointer=cp)

        if args.resume:
            job = args.resume
            cfg = {"configurable": {"thread_id": job}, "max_concurrency": 1}
            st = graph.get_state(cfg)
            if args.approve or args.auto:
                print(f"[resume] approving script for {job}")
                _drain(graph, Command(resume={"status": "approved"}), cfg)
            else:
                segs = st.values.get("segments", [])
                print(json.dumps({"job": job, "title": st.values.get("title"),
                                  "segments": segs}, indent=2)[:4000])
                print("\n→ re-run with --resume", job, "--approve  to finish")
                return
        else:
            if not args.topic:
                ap.error("--topic required")
            job = args.job or _slug(args.topic)
            cfg = {"configurable": {"thread_id": job}, "max_concurrency": 1}
            print(f"[job] {job}  topic={args.topic!r}")
            paused = _drain(graph, {"job_id": job, "topic": args.topic}, cfg)
            if paused:
                if args.auto:
                    print("[hitl] auto-approving script")
                    _drain(graph, Command(resume={"status": "approved"}), cfg)
                else:
                    print(f"\n[hitl] paused for script approval. Review & finish with:\n"
                          f"   python -m docupipe.run --resume {job} --approve")
                    return

        final = graph.get_state(cfg).values
        ep = final.get("episode_path")
        print("\n=== DONE ===")
        print("episode:", ep)
        print("credits:", final.get("credits_path"))
        if final.get("errors"):
            print("errors:", final["errors"])
        sys.exit(0 if ep else 2)


if __name__ == "__main__":
    main()
