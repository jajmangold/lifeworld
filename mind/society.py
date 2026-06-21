#!/usr/bin/env python3
"""M3 (text-first): a small NPC society lives a scene in TEXT, tracked in Neo4j.

The user's plan: bots interact in text first, then embody/render selected beats.
Each agent has a persona + goal; round by round they speak/act (DeepSeek V4 Flash)
given the unfolding transcript. Utterances + evolving feelings are written to the
graph, which becomes the storyline record that the embodied layer can later render.
"""
import argparse
import sys

sys.path.insert(0, "/work")
sys.path.insert(0, ".")
from mind.brain import decide
from memory.graph import Memory

AGENTS = [
    {"name": "Mara", "persona": "tidy, a bit anxious, works from home; hates when her food goes missing"},
    {"name": "Theo", "persona": "easygoing musician, forgetful, conflict-averse, often borrows things"},
    {"name": "Priya", "persona": "blunt, funny, the peacemaker who says what everyone's thinking"},
]
SCENE = ("Three roommates in their shared apartment, Saturday morning. Mara just "
         "discovered her labeled leftovers are gone from the fridge.")


def turn(agent, others, transcript):
    sys_p = (f"You are {agent['name']}: {agent['persona']}. You are in a scene with "
             f"{', '.join(others)}. Stay in character, be natural and brief (1-2 sentences). "
             "Reply ONLY JSON.")
    convo = "\n".join(f"{t['who']}: \"{t['say']}\" [{t['do']}]" for t in transcript[-8:])
    user = (f"SCENE: {SCENE}\n\nSo far:\n{convo or '(nobody has spoken yet)'}\n\n"
            f"It's your turn. What do you say and do? Respond "
            '{"say":"<your line>","do":"<a brief physical action>",'
            '"toward":"<the person you address or empty>",'
            '"feeling":"<one word: how you feel toward them>",'
            '"sentiment":"<positive|neutral|negative>"}')
    return decide(sys_p, user, temperature=0.9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args()

    mem = Memory()
    for ag in AGENTS:
        mem.ensure_agent(ag["name"])
    # fresh storyline
    with mem.drv.session() as s:
        s.run("MATCH (u:Utterance) DETACH DELETE u")
        s.run("MATCH (:Agent)-[r:FEELS]->(:Agent) DELETE r")

    names = {ag["name"] for ag in AGENTS}
    transcript = []
    t = 0
    for _ in range(a.rounds):
        for ag in AGENTS:
            others = [x["name"] for x in AGENTS if x["name"] != ag["name"]]
            try:
                out = turn(ag, others, transcript)
            except Exception as e:
                print(f"[{ag['name']}] turn failed: {e}"); continue
            say, do = out.get("say", ""), out.get("do", "")
            transcript.append({"who": ag["name"], "turn": t, "say": say, "do": do})
            mem.add_utterance(ag["name"], t, say, do, scene="apt")
            toward = str(out.get("toward", "")).strip()
            if toward in names:          # only record feelings toward a real single agent
                mem.set_feeling(ag["name"], toward, out.get("feeling", ""),
                                out.get("sentiment", "neutral"))
            print(f"  {ag['name']}: \"{say}\"  [{do}]")
            t += 1

    print("\n=== STORYLINE (from Neo4j) ===")
    for u in mem.transcript():
        print(f"  t{u['turn']} {u['who']}: {u['say']}")
    print("\n=== RELATIONSHIPS (from Neo4j) ===")
    for r in mem.relationships():
        print(f"  {r['a']} -> {r['b']}: {r['feeling']} ({r['sentiment']})")
    mem.close()
    print(f"\nSOCIETY_OK agents={len(AGENTS)} turns={t}")


if __name__ == "__main__":
    main()
