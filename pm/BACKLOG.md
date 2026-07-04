# BACKLOG — the single source of truth for work

> Every piece of work is an item here. Pick the highest-priority **unblocked** item; follow
> [PROCESS.md](PROCESS.md). Update status as part of Done. Don't do work that isn't an item — make the item
> first. Decompose later phases only when you reach them (avoid over-planning).
>
> **Item schema:** `[ID] Title · priority · status` + **Ships** (revenue/shippable outcome) · **Depends-on**
> · **Done-when** (acceptance, includes docs + CHANGELOG). Priority: **P0** ships revenue / unblocks a P0;
> **P1** important; **P2** nice. Status: `todo · wip · blocked · done · parked`. `[exp]` = time-boxed experiment.
> **WIP limit: ≤2 `wip` items, ideally 1.**

Legend of current reality (so items are grounded): the resident **SDNQ LTX farm** generates the anchor;
**a2v_talk.py** does native audio-driven talking (lips move but subtle at 256–384px; 384×512 is the single-
card ceiling); **green-screen subject-mask + newsroom composite** works (~7/10); **swap/muse/FlashVSR** are the
finishing path; **newscast/** does script+TTS+graphics; **NNS** brand. See `render/sdnq_ltx/SDNQ_STACK.md`.

---

## PHASE 0 — Lock production quality  (ACTIVE · all P0)

### E0.1 — ACHIEVE a production-quality talking anchor  ← THE gate; all of Phase 1+ is blocked on it
> Honest status: **we do not have this yet.** a2v_talk lips move but are subtle + res-capped (384×512 on one
> card). This epic is genuinely unsolved R&D, not polish. Nothing else ships until it clears the bar.
- **[T0.1a] Write the measurable quality bar** · P0 · todo
  - Ships: an objective target so "good enough" isn't a vibe.
  - Depends-on: —
  - Done-when: `pm/QUALITY_BAR.md` defines pass/fail thresholds — lip-sync (audio-driven, verified),
    identity consistency, motion/no-glitch, resolution/sharpness, throughput/segment — with the qwen-QA
    method to score each. (Defines the bar; does not assume we meet it.)
- **[T0.1b] SPIKE: what does it take to hit the lip-sync/motion bar?** · P0 · done · `[exp]` time-box ~2–3 days
  - Ships: a decision on the anchor pipeline of record + evidence it can hit the bar.
  - Depends-on: T0.1a
  - Done-when: the real options are evaluated against the bar × complexity × runs-on-farm, and ONE is chosen
    with evidence. Options to weigh (don't pre-decide):
    - **(a) PRIMARY — wire a2v into the two-stage SPATIAL UPSCALER.** The *official* path past the ceiling,
      already working: `sdnq_two_stage.py` denoises 256×384 → LTX2 spatial x2 latent upsampler → refines to
      **512×768** (sharp — qwen "high resolution" vs single-stage "moderate") **on one card**, the res
      single-stage a2v OOMs at. Gap: a2v (audio-driven motion) is NOT wired to it. Task: a2v at 256/384 →
      spatial-upscale latents → 3-step refine → decode; verify **lip motion survives the refine** + crisp
      mouth (qwen). Also test the **temporal upscaler** (`temporal_upsample=True`, unused today) for motion
      smoothness. Upscaler weights live in the full Lightricks LTX-2.3 cache, not the SDNQ checkpoint.
    - **(b) Attention/VAE tiling** to fit 512×768 single-stage — only if (a) can't preserve lip motion.
    - **(c) Multi-GPU** tensor/sequence-parallel — last resort (bigger spend).
    - **(d) Tighter MCU framing** where current res already suffices (cheapest fallback).
    Kill criterion: if none clears the bar in the time-box, escalate. First steps: prove a2v is genuinely
    audio-driven (silent-vs-speech differential); QA all comparisons with qwen9b.
  - NOTE (measured this session): both a2v AND IC-LoRA work on the **production `use_quantized_matmul=True`**
    path with **no quality loss and ~1.7–2× speedup** (a2v 256×384 = 58.6s vs eager 97s; ic_union = 44s).
    The 384×512 OOM ceiling is unchanged by quantized-matmul (it's attention memory) — hence option (a).
  - EVIDENCE (T0.1b, 2026-07-04): built `a2v_two_stage.py` (a2v audio-driven → LTX2 spatial x2 upsampler →
    3-step refine) — **512×768 FITS one card (offload, ~12.8GB)** and is **SHARP (qwen 9/10)**. Resolution
    is SOLVED via the spatial upscaler. BUT **lip-MOTION amplitude is still minimal** (qwen: mouth "same
    shape" across frames) — the refine sharpened the subtle a2v motion, didn't increase it. So the crisp-
    lips problem splits: **sharpness = solved (two-stage); motion amplitude = the real open gap** (likely
    levers: modality_scale, audio↔video length match, confirm audio-driven). Investigating next.
  - DECISION (evidence): the production anchor pipeline = **a2v two-stage (256→512×768 spatial upscaler)
    + modality_scale≈6 + audio-matched frame count.** Two levers, both proven this session: (1) sharpness →
    spatial upscaler (qwen 9/10 @512×768, fits one card); (2) lip-MOTION → modality_scale 6 (vs 3) +
    NPX=audio-frames gives clear articulation (qwen 10/10 "actively speaking"; mouth-strip shows open/teeth/
    jaw variation vs the old subtle motion). Next: T0.1c confirms the COMBINED pipeline on ≥3 clips.
- **[T0.1c] Implement the chosen approach to the bar (two-stage + modality≈6 + matched audio)** · P0 · todo
  - Ships: the production-quality anchor itself.
  - Depends-on: T0.1b
  - Done-when: the anchor passes `pm/QUALITY_BAR.md` on ≥3 varied clips (qwen-scored, lip-sync verified);
    becomes the documented pipeline of record; SDNQ_STACK.md + CHANGELOG updated.

### E0.2 — Reliable one-command segment
- **[T0.2a] One command → finished segment, unattended** · P0 · todo
  - Ships: repeatable production (the thing we scale).
  - Depends-on: T0.1a
  - Done-when: a single documented command produces a finished segment end-to-end on the resident farm with
    no manual steps; failure modes handled; runbook in `render/sdnq_ltx/` or newscast AGENTS.md.
- **[T0.2b] 5 unattended segments, zero manual fixes** · P0 · todo · (Phase-0 exit proof)
  - Ships: proof the pipeline is production-locked.
  - Depends-on: T0.2a, T0.3*, T0.4*
  - Done-when: 5 consecutive segments produced unattended, each passing E0.3/E0.4 gates, logged.

### E0.3 — Original-analysis script + compliance gates (SURVIVAL — see STRATEGY §3)
- **[T0.3a] Script adds original analysis (no verbatim)** · P0 · todo
  - Ships: monetization-safe scripts (verbatim wire-reading = channel demonetization).
  - Depends-on: —
  - Done-when: script gen (newscast `draft_news`/`news_room`) produces commentary/analysis/context, not
    narrated wire copy; a lint/check flags near-verbatim/near-duplicate; finance/tech framing enforced.
- **[T0.3b] Fact / defamation gate** · P0 · todo
  - Ships: legal safety (publisher liability on named real people/companies).
  - Depends-on: —
  - Done-when: any claim about a named real person/company requires 2-source verification + a human sign-off
    step before publish; the check is logged per asset.
- **[T0.3c] AI-disclosure metadata per asset** · P0 · todo
  - Ships: keeps the channel un-flagged (non-disclosure is the risk).
  - Depends-on: —
  - Done-when: every rendered asset carries disclosure metadata; the publish step sets YouTube "altered
    content" + TikTok "AI-generated" toggles; C2PA where feasible; state logged.

### E0.4 — Packaging + per-platform variants
- **[T0.4a] Engineered hook + CTR title/thumbnail** · P0 · todo
  - Ships: reach (CTR×AVD is the ranking signal).
  - Depends-on: T0.1a
  - Done-when: long-form opens hit the 3-phase hook (<10s intro, result-first); title/thumbnail generated to
    the CTR 6–8% / AVD 40–55% intent; TikTok first-3s >70%-completion hook.
- **[T0.4b] Watermark-free master → 16:9 + 9:16 variants** · P0 · todo
  - Ships: multi-platform distribution without the watermark/reused-content penalty.
  - Depends-on: T0.2a
  - Done-when: pipeline emits a clean master + per-platform variants (distinct first frame/audio/caption);
    no TikTok watermark on YT/IG; captions + NNS graphics correct per aspect.

---

## PHASE 1 — First channel to money  (blocked by P0 · P0→P1) — epics, decompose at phase start
- **[E1.1]** Channel setup + brand + first identity/isolation-map entry · P1 · blocked
- **[E1.2]** Compliance layer productionized (disclosure + human review gate + fact gate as hard steps) · P0 · blocked
- **[E1.3]** Publishing pipeline: YouTube Data API + TikTok Direct Post (within quotas; **start TikTok audit ~5–10d early**) · P1 · blocked
- **[E1.4]** Cadence engine (1–3 long-form/wk + 3–5 Shorts + TikTok 1–3/day jittered) · P1 · blocked
- **[E1.5]** Revenue-stream wiring (memberships/affiliate/sponsorship slot + clip-licensing export) · P1 · blocked
- **[E1.6]** Analytics ingest (CTR/AVD/retention/RPM per video → scoreboard) · P1 · blocked

## PHASE 2 — Portfolio  (blocked by P1 cash-flow-positive · P1) — epic stubs
- **[E2.1]** Repeatable channel-launch playbook (differentiated, not cloned) · blocked
- **[E2.2]** Multi-identity infra (GCP-project sharding + AdSense isolation; TikTok profiles+proxies+14-day warm-up) · blocked
- **[E2.3]** Per-identity health/strike dashboard · blocked

## PHASE 3 — AI showrunner  (blocked by P2 + analytics loop · P2) — epic stubs
- **[E3.1]** Orchestrator + specialists (qwen9b scoring/draft on amd0, DeepSeek planning/fact-review) · blocked
- **[E3.2]** Closed feedback loop (ingest→score→produce→measure→learn) + decision audit trail · blocked
- **[E3.3]** Bounded-autonomy charter + escalation gates · blocked

---

## PARKING LOT (captured, NOT built — shiny-object gate, PROCESS §6)
Revisit only at phase boundaries. Being here means "good idea, wrong time."
- ~~Crisp lip-sync via multi-GPU~~ — **UNPARKED into E0.1b option (a):** it's not shiny, it's a candidate
  fix for the P0 anchor-quality blocker. Evaluate it there against the alternatives.
- **True 3D/2.5D newsroom grounding** beyond the green-screen composite.
- **Two-stage LTX (half-res→upscaler→refine)** for sharper output — only if the bar demands it.
- **IC-LoRA depth/pose control in production** — works (`ic_union.py`) but not needed to ship a talking anchor.
- **New base models / anchors / voices** — evaluate only against a shipped baseline, time-boxed `[exp]`.
- **Fold a2v audio-driven into the windowed long-clip path** (`sdnq_llong.py`) for >3s talking.
