"""DocuState — flat, paths-not-bytes. Fan-in keys use operator.add reducers."""
import operator
from typing import Optional
from typing_extensions import Annotated, TypedDict


class Segment(TypedDict, total=False):
    id: int
    kind: str               # "cold_open" | "narration" | "title"
    narration: str          # VO text ("" for silent title)
    image_query: str        # archive search query for the visual
    # filled during render:
    asset: Optional[dict]   # {source, license, attribution, page_url, rights_ok}
    image_path: Optional[str]
    wav_path: Optional[str]
    dur: Optional[float]
    clip_path: Optional[str]
    enhanced: bool          # AI restoration applied? (for cue sheet)


class DocuState(TypedDict, total=False):
    job_id: str
    topic: str
    title: str
    research: dict                              # {overview, key_facts[], entities[], threads[], sources[]}
    dossiers: Annotated[list, operator.add]     # deep-research fan-in: one rich dossier per entity/thread
    story_bible: dict                           # {logline, target_minutes, cast[], threads[], acts[]}
    act_drafts: Annotated[list, operator.add]   # write-act fan-in: {act, segments[]}
    editor_notes: dict                          # showrunner critique of the assembled draft
    segments: list                              # planned by script node (ordered)
    script_approved: bool
    # fan-in (parallel render writes here) -> MUST have reducer:
    rendered: Annotated[list, operator.add]     # rendered Segment dicts
    cue_sheet: Annotated[list, operator.add]    # AI-op log (APA)
    errors: Annotated[list, operator.add]
    episode_path: Optional[str]
    credits_path: Optional[str]
