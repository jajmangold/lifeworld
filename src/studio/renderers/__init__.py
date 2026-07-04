"""Renderer registry for the unified Studio render_segment dispatcher.
Importing this package registers all renderers (anchor/reporter + non-talking builders)."""
from .registry import render_segment, register, REGISTRY
from . import anchor    # noqa: F401  registers fullscreen_anchor/anchor_wall/ots/reporter_pkg
from . import builders  # noqa: F401  registers vo_broll/title/cold_open/narration

__all__ = ["render_segment", "register", "REGISTRY"]
