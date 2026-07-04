# ROADMAP — phases (sequence the epics)

> Phases are gates, not calendar. **You do not start a phase until the prior phase's exit criteria are
> met** (ship discipline — [PROCESS.md §0](PROCESS.md)). One phase in flight. Epics map to
> [BACKLOG.md](BACKLOG.md); re-prioritize the backlog only at phase boundaries.

```
P0 Lock production quality ──▶ P1 First channel to money ──▶ P2 Portfolio ──▶ P3 AI showrunner
     (SHIP the pipeline)          (prove the business)         (scale it)       (automate the ops)
```

---

## Phase 0 — Lock production quality  ·  status: ACTIVE  ·  priority: P0

**Goal:** one command → a finished, compliant, monetizable **finance/tech-news** segment that hits the
[VISION quality bar](VISION.md), reliably, unattended, on the resident farm. Nothing scales until this holds.

**Epics:** E0.1 Anchor quality bar · E0.2 Reliable one-command segment · E0.3 Original-analysis script +
fact-gate · E0.4 Packaging (graphics/captions/hook) + per-platform variants.

**Exit criteria (all measured, not vibes):**
- 5 consecutive segments produced unattended with **zero manual fixes**.
- Anchor: qwen QA ≥ bar, lips audio-synced (verified), no glitching.
- Script passes the original-analysis + fact-gate checks; AI-disclosure metadata attached.
- Master + 16:9 + 9:16 watermark-free variants emitted; hook engineered.
- **The quality spec is FROZEN and written down** (E0.1 output). After this, we add volume/channels, not fidelity.

## Phase 1 — First channel to money  ·  P0→P1  ·  (blocked by P0)

**Goal:** one flagship **finance/business/tech-news** channel live on YouTube (+ TikTok), publishing on
cadence, compliant, wired for revenue, driving toward YPP / Creator-Rewards thresholds.

**Epics:** E1.1 Channel setup + brand + identity/isolation map (first identity) · E1.2 Compliance layer
(disclosure + human editorial review gate + fact/defamation gate as hard pipeline steps) · E1.3
Publishing pipeline (YouTube Data API + TikTok Direct Post, within quotas; TikTok audit started) · E1.4
Cadence engine (1–3 long-form/wk + 3–5 Shorts + TikTok 1–3/day jittered) · E1.5 Revenue-stream wiring
(memberships/affiliate/sponsorship slot + clip-licensing export) · E1.6 Analytics ingest (read CTR/AVD/
retention/RPM per video).

**Exit criteria:** channel live + publishing on schedule; every publish passes the compliance gates;
approaching (or hit) a monetization threshold; per-video analytics flowing into a scoreboard.

## Phase 2 — Portfolio  ·  P1  ·  (blocked by P1 being cash-flow positive)

**Goal:** clone the proven workflow to 2–3 channels across finance/tech/business-news verticals with
**isolated identities** so no strike cascades.

**Epics:** E2.1 Channel-launch playbook (repeatable, differentiated-not-cloned) · E2.2 Multi-identity infra
(GCP-project sharding, AdSense isolation; TikTok profiles+proxies+14-day warm-up) · E2.3 Per-identity
health/strike dashboard.

**Exit criteria:** 2–3 channels live on isolated identities; combined revenue trending to the $5–10k/mo target.

## Phase 3 — AI showrunner  ·  P2  ·  (blocked by P2 + analytics loop)

**Goal:** qwen9b/DeepSeek orchestrator owns the content-ops loop (topic selection, scheduling, A/B,
kill-underperformers, strategy updates from analytics) under **bounded autonomy** with human escalation on
sensitive/legal/launch/spend.

**Epics:** E3.1 Orchestrator + specialist agents (qwen9b scoring/draft, DeepSeek planning/fact-review) ·
E3.2 Closed feedback loop (ingest→score→produce→measure→learn) with decision audit trail · E3.3 Bounded-
autonomy charter + escalation gates.

**Exit criteria:** the loop proposes+schedules+iterates content autonomously within limits; human touch
per segment trending toward zero; measured lift over manual selection.

---

### Parking lot → see [BACKLOG.md#parking-lot](BACKLOG.md). Shiny-object ideas are captured there, not built.
