"""Wire the docuseries StateGraph — a WRITERS' ROOM feeding the render pipeline:
research -> deep-research (fan-out) -> story bible -> write acts (fan-out) -> assemble script
-> showrunner edit -> curate -> [approve] -> render (fan-out) -> assemble episode."""
from langgraph.graph import StateGraph, START, END
from langgraph.types import RetryPolicy
from .state import DocuState
from . import nodes

_NET = RetryPolicy(max_attempts=4)   # transient HTTP to LLM / archives / TTS


def build():
    g = StateGraph(DocuState)
    # writers' room
    g.add_node("research", nodes.research_node, retry_policy=_NET)
    g.add_node("deep_research", nodes.deep_research_node, retry_policy=_NET)
    g.add_node("story_bible", nodes.story_bible_node, retry_policy=_NET)
    g.add_node("write_act", nodes.write_act_node, retry_policy=_NET)
    g.add_node("assemble_script", nodes.assemble_script_node)
    g.add_node("script_editor", nodes.script_editor_node, retry_policy=_NET)
    # render pipeline (unchanged)
    g.add_node("curate", nodes.curate_node, retry_policy=_NET)
    g.add_node("review", nodes.review_script)
    g.add_node("render_segment", nodes.render_segment, retry_policy=_NET)
    g.add_node("assemble", nodes.assemble_node)

    g.add_edge(START, "research")
    g.add_conditional_edges("research", nodes.deep_research_fanout, ["deep_research"])
    g.add_edge("deep_research", "story_bible")
    g.add_conditional_edges("story_bible", nodes.write_act_fanout, ["write_act"])
    g.add_edge("write_act", "assemble_script")
    g.add_edge("assemble_script", "script_editor")
    g.add_edge("script_editor", "curate")
    g.add_edge("curate", "review")
    g.add_conditional_edges("review", nodes.fan_out, ["render_segment"])
    g.add_edge("render_segment", "assemble")
    g.add_edge("assemble", END)
    return g
