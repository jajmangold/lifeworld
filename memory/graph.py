#!/usr/bin/env python3
"""Memory/society graph access (ADR-0004) — the life-record in Neo4j.

Schema (minimal, M2): (:Agent {name})-[:DID]->(:Step {t, seen, thought, target})
plus (:Step)-[:AT]->(:Place {name}). The graph IS the life-record; later milestones
add relationships, episodic memory, storyline beats.
"""
import os
from neo4j import GraphDatabase

URI = os.environ.get("NEO4J_URI", "bolt://127.0.0.1:7688")
USER = os.environ.get("NEO4J_USER", "neo4j")
PWD = os.environ.get("NEO4J_PASSWORD", "lifeworld-dev")


class Memory:
    def __init__(self, uri=URI, user=USER, pwd=PWD):
        self.drv = GraphDatabase.driver(uri, auth=(user, pwd))

    def close(self):
        self.drv.close()

    def ensure_agent(self, name):
        with self.drv.session() as s:
            s.run("MERGE (a:Agent {name:$n})", n=name)

    def reset_agent(self, name):
        """Wipe an agent's prior steps for a clean life-record run."""
        with self.drv.session() as s:
            s.run("MATCH (a:Agent {name:$n})-[:DID]->(st:Step) DETACH DELETE st", n=name)
            s.run("MERGE (a:Agent {name:$n})", n=name)

    def log_step(self, agent, t, seen, thought, target, place=None):
        """Append one perceive->decide->act step to the agent's life-record."""
        with self.drv.session() as s:
            s.run(
                """
                MATCH (a:Agent {name:$agent})
                CREATE (st:Step {t:$t, seen:$seen, thought:$thought, target:$target})
                CREATE (a)-[:DID]->(st)
                WITH st WHERE $place IS NOT NULL
                MERGE (p:Place {name:$place})
                CREATE (st)-[:AT]->(p)
                """,
                agent=agent, t=t, seen=seen, thought=thought,
                target=target, place=place)

    def life_story(self, agent):
        with self.drv.session() as s:
            r = s.run(
                "MATCH (a:Agent {name:$n})-[:DID]->(st:Step) "
                "RETURN st.t AS t, st.target AS target, st.thought AS thought "
                "ORDER BY st.t", n=agent)
            return [dict(x) for x in r]


if __name__ == "__main__":
    m = Memory()
    m.ensure_agent("Mara")
    m.log_step("Mara", 0, "fridge, counter, sofa", "I'm hungry", "fridge", "kitchen")
    print("GRAPH_OK life:", m.life_story("Mara"))
    m.close()
