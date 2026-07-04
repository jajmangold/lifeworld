"""render_segment dispatcher — the connective tissue of the unified Studio graph.

The graph's render_segment node reads Segment.kind and calls the registered renderer, which produces
segment['clip_path']. Renderers WRAP the locked, working scripts (make_anchor/make_character.sh, the
newscast ffmpeg builders) — they are NOT rewrites. New segment kind => register a renderer; no graph change.
This is the 'one spine, dispatcher' design from STUDIO_ARCHITECTURE.md.
"""
REGISTRY = {}

def register(*kinds):
    def deco(fn):
        for k in kinds:
            REGISTRY[k] = fn
        return fn
    return deco

def render_segment(seg, ctx):
    """seg: Segment dict (kind, talent_ref, set_ref, spec, wav_path, ...). ctx: {talent, sets, brands, ...}.
    Returns the segment with clip_path filled."""
    kind = seg.get("kind")
    fn = REGISTRY.get(kind)
    if not fn:
        raise ValueError(f"render_segment: no renderer registered for kind '{kind}' (have {sorted(REGISTRY)})")
    return fn(seg, ctx)
