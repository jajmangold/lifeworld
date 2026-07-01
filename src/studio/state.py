"""Studio state — the shared production data model for ALL show types (news, docuseries, + future).
Generalizes docupipe's DocuState into a typed Production envelope with POLYMORPHIC, MULTI-DELIVERABLE
segments. Flat, paths-not-bytes; fan-in keys use operator.add reducers (langgraph map-reduce). See
STUDIO_ARCHITECTURE.md.
"""
import operator
from typing import Optional
from typing_extensions import Annotated, TypedDict


class Deliverable(TypedDict, total=False):
    platform: str            # "youtube" | "youtube_shorts" | "tiktok"
    aspect: str              # "16:9" | "9:16"
    path: str
    duration: Optional[float]


class Segment(TypedDict, total=False):
    id: int
    order: int
    kind: str                # renderer discriminator: anchor_wall | fullscreen_anchor | ots | reporter_pkg
                             #   | vo_broll | title | narration | cold_open | ...  (open vocab -> registry)
    spec: dict               # kind-specific params (screen img, ots panels, broll queries, headline, ...)
    talent_ref: Optional[str]  # -> catalog/talent  (anchor/reporter/narrator: voice + avatar)
    set_ref: Optional[str]     # -> catalog/sets    (studio HDRI / field pano / video wall)
    # writers' room fills:
    narration: str           # VO/read text ("" for silent title)
    image_query: str         # archive/gen query for the visual
    sources: list            # [{name, url}] source citations -> cue sheet
    # render fills (paths, not bytes):
    asset: Optional[dict]    # {source, license, attribution, page_url, rights_ok}  (Wikimedia/zimage)
    image_path: Optional[str]
    wav_path: Optional[str]
    dur: Optional[float]
    master_path: Optional[str]   # the shared HIGH-RES gpu render (aspect-agnostic; the costly artifact)
    deliverables: list           # [Deliverable] — 16:9 / 9:16 / shorts derived CHEAPLY from master
    enhanced: bool               # AI restoration/upscale applied? (cue sheet)


class Production(TypedDict, total=False):
    job_id: str
    show_type: str           # "news" | "docuseries" | ...
    profile: str             # ShowProfile ref (prompts/pacing/allowed kinds/defaults)
    brief: str               # topic / date / brief
    title: str
    brand_ref: str           # -> catalog/brands
    # writers' room (shared spine; differences come from the profile):
    research: dict           # {overview, key_facts[], entities[], threads[], sources[]} (news: from Portal)
    dossiers: Annotated[list, operator.add]       # deep-research fan-in
    bible: dict              # story bible (docuseries) / rundown (news): {logline, target_minutes, acts/blocks[]}
    act_drafts: Annotated[list, operator.add]     # write-act/block fan-in
    editor_notes: dict       # showrunner/EP critique
    segments: list           # planned, ordered
    script_approved: bool
    # render fan-in (parallel writes -> MUST have reducers):
    rendered: Annotated[list, operator.add]       # rendered Segment dicts
    cue_sheet: Annotated[list, operator.add]      # AI-op + attribution log (APA)
    errors: Annotated[list, operator.add]
    # delivery:
    episode_path: Optional[str]                   # full 16:9 episode master
    deliverables: Annotated[list, operator.add]   # per-platform packages/playlists (16:9, 9:16, shorts)
    credits_path: Optional[str]
