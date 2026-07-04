# VISION — the north star

> One page. Read it first. If a decision doesn't move us toward this, don't do it.

## Mission

Run a **high-quality, fully-automated news studio** that produces segments and shows for **YouTube and
TikTok**, and **maximize revenue** from them — a portfolio of monetized channels operated by software,
increasingly steered by an AI showrunner. Quality first, then scale; ship, don't tinker.

## What "good" looks like (the product)

Broadcast-credible AI news video: a photoreal anchor, accurate scripted copy with original analysis,
on-air graphics, correct pacing, delivered in the right format per platform (long-form + Shorts/TikTok),
consistently, on a schedule, at low marginal cost — generated on our own resident GPU farm.

## The business in one line

**Cost ≈ fixed (our own hardware) → every additional monetized minute is near-pure margin.** So the game is:
(1) hit the monetization/quality bar, (2) publish consistently across a **portfolio** of niche channels,
(3) let an AI showrunner compound it by reacting to trends + analytics. Detailed money model, niches, and
platform playbook: [STRATEGY.md](STRATEGY.md).

## The production quality bar (what "locked in" means — DoD for the pipeline)

**STATUS: NOT MET.** We do **not** have a production-quality anchor yet. The native audio-driven anchor
(`a2v_talk.py`) is a promising prototype but the lips move only subtly and are resolution-capped (384×512
on one 16GB sm_70 card) — below a credible talking-news-anchor bar. **Achieving this bar is the current P0
work (E0.1), and it is genuinely unsolved R&D, not a formality.** Everything downstream (channels, revenue)
is blocked on it — you cannot run a news studio without a believable anchor.

We do not scale or add capability until the core pipeline reliably hits this bar (measured, not vibes):

- **Anchor**: photoreal, identity-consistent, **lips actually move / audio-synced**, natural head motion,
  no glitching. (qwen9b QA ≥ target; the a2v audio-driven path is the direction.)
- **Script**: factually grounded, adds **original analysis** (not just narrated wire copy — required to
  stay monetizable), correct broadcast structure.
- **Package**: correct graphics (NNS brand), captions, thumbnail/title, per-platform aspect (16:9 + 9:16).
- **Reliability**: one command → finished segment; repeatable; runs on the resident farm unattended.
- **Throughput**: enough segments/day to sustain a publishing cadence (see STRATEGY targets).

## Success metrics (how we know it's working)

- **P0 — Ship**: a monetizable segment produced end-to-end, unattended, at the quality bar. Then: N
  segments/week published on ≥1 live channel.
- **Growth**: reach the platform monetization thresholds (YPP / TikTok Creator Rewards) → first revenue.
- **Scale**: a repeatable channel-launch playbook → portfolio of niches; revenue per channel; total MRR.
- **Autonomy**: the AI showrunner owns topic selection + cadence + strategy tweaks from analytics, with
  human go/no-go shrinking over time.

## Principles (non-negotiable)

1. **Ship > perfect > shiny.** Lock production quality, then scale. New tech must pass the shiny-object
   gate ([PROCESS.md §6](PROCESS.md)).
2. **Boring, resident, reliable.** Prefer the proven farm + proven tools. Fewer moving parts.
3. **Portfolio, not hero channel.** Repeatable launches beat one perfect channel.
4. **Compliant by construction.** AI-disclosure + accuracy + platform policy are build-time requirements,
   not afterthoughts — a policy strike kills revenue (see STRATEGY risk section).
5. **The system keeps us honest.** WIP limits, time-boxed experiments, DoD-includes-docs, evidence over
   claims. The process exists so we ship.

## Non-goals (what we deliberately will NOT chase)

- Chasing every new model/paper. A better anchor we can't ship beats nothing; a shipped anchor beats a
  better one we're still tuning.
- A grand unified framework before we have revenue. Generalize only after the second channel proves it.
- Perfect photorealism at the cost of throughput. "Broadcast-credible + consistent" wins over "flawless
  once."
- Building the AI showrunner before the manual pipeline ships. Autonomy is a scale lever, not a
  prerequisite.
