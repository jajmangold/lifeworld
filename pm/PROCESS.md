# PROCESS — how we work (the operating system)

> This is binding for **every** contributor, human or AI. `studio/AGENTS.md` points here and every task
> must follow it. If a rule here conflicts with what you feel like doing, the rule wins. Keep it lean —
> this file is the whole process; don't add ceremony.

North star and priorities live in [VISION.md](VISION.md); the money/platform playbook in
[STRATEGY.md](STRATEGY.md); the ordered work in [ROADMAP.md](ROADMAP.md) → [BACKLOG.md](BACKLOG.md).

---

## 0. The prime directive: SHIP > perfect > shiny

We are building a **revenue-generating** automated news studio. The goal is shipped, monetized output —
not the most advanced pipeline. In priority order, always:

1. **Ship** a working, monetizable segment/show at the agreed quality bar.
2. **Lock** production quality (make the current pipeline reliable/repeatable) before adding capability.
3. Only then, **shiny** — new tech/features — and only if it clears the gate in §5.

If you are ever unsure what to do, do the thing that ships the next segment.

---

## 1. Every task follows this loop (Spec → Plan → Do → Validate → Log)

1. **Pick** the highest-priority *unblocked* item from [BACKLOG.md](BACKLOG.md) (respect WIP limit, §4).
   Don't invent work that isn't an item — if it's worth doing, make it an item first.
2. **Sequential-think it (MANDATORY).** Start real work with `sequentialthinking`: restate the goal,
   the acceptance criteria, the plan, the risks. No non-trivial change without it. (This models the bar
   we hold everyone to — it's not optional.)
3. **Trailmark before touching unfamiliar code (MANDATORY where it makes sense).** Run `trailmark analyze`
   (structure), `--complexity`, `entrypoints`, or `diff` to understand what you're changing before you
   change it — never hand-grep a module you don't already know. (Skip only for pure-docs or a one-line
   edit to a file you just wrote.)
4. **Do** the smallest change that satisfies the item's *Done-when*. Don't gold-plate.
5. **Validate**: run it / test it. Prove it works with evidence, not a claim. **Any image/video output MUST
   be QA'd with qwen9b (amd0) AS YOU GO** — grounded VQA (temp 0, structured describe-then-judge, no
   priming) via `/tmp/claude-1000/proof/qa.py` (`compare`/`rate`); requests need
   `chat_template_kwargs:{enable_thinking:false}`. Visual quality is a **qwen score, never a vibe** — QA
   every important frame/clip, keep the evidence. "Done" means demonstrated, not written.
6. **Log**: update the item's status in BACKLOG, add a [CHANGELOG](../CHANGELOG.md) entry, and update the
   relevant `AGENTS.md`/doc if behavior or structure changed (§3).

## 2. Definition of Done (DoD) — an item is done ONLY when ALL are true

- [ ] Acceptance criteria (the item's **Done-when**) demonstrably met, with evidence.
- [ ] **Any visual output was QA'd with qwen9b** (score/compare) and the evidence is kept — no shipping images on vibes.
- [ ] Sequential-thinking was used to plan it; trailmark was used if code structure was touched.
- [ ] Docs updated: the nearest `AGENTS.md` reflects reality; capabilities not stale paths.
- [ ] A CHANGELOG entry exists under **Unreleased**.
- [ ] No new sprawl: no orphan scripts, no stale endpoints/paths/branding, outputs in `output/`.
- [ ] BACKLOG status set to `done` (and any unblocked items noted).

## 3. Documentation is part of Done (not optional)

- Reality lives in `AGENTS.md` (nearest-wins) + the deep refs (`render/sdnq_ltx/SDNQ_STACK.md`). Describe
  **capabilities**, not exhaustive file paths (paths rot and poison agent context).
- If you changed how something runs, the doc changes in the **same** change. A PR/commit that alters
  behavior without touching docs is not done.
- Aspirational/planning docs must be **bannered** as such (see `README.md`, `STUDIO_ARCHITECTURE.md`).
- Endpoints/config are env-overridable and documented in ONE place (`studio/AGENTS.md` → Shared services).

## 4. Anti-sprawl + WIP limits (keeps us honest, keeps the tree clean)

- **WIP limit: at most 2 items `in-progress` at once** (ideally 1). Finish before you start.
- **One epic in flight** at a time unless the roadmap explicitly parallelizes.
- **Retire as you go**: superseded code → `/mnt/24tb/nvme-offload/bot-archive/<category>/` (the archival
  convention). Flag legacy in `AGENTS.md`; don't leave dead code in the active tree.
- **No new top-level areas** without a BACKLOG item that justifies it. New area ⇒ new `AGENTS.md`.

## 5. Experiments — allowed, but time-boxed and must graduate-or-die

Experimentation is how we improve, but it is the #2 threat to shipping (after scope creep). So:

- Experiments are BACKLOG items tagged `[exp]` with an explicit **time-box** (hours/days) and a **kill
  criterion** ("if it doesn't beat the current pipeline on <metric> by <date>, archive it").
- Run them in a sandbox/scratch, not the production path. They touch production only after graduating.
- **Graduate** = beats the current approach on a measured metric AND has a DoD-complete item. Otherwise
  **archive** it (bot-archive) and write one line in CHANGELOG about what we learned. No zombie experiments.

## 6. The shiny-object gate (this is the "keep me honest" rule)

Before adding ANY new model / framework / capability, it must pass ALL of:

1. **Does it ship or unblock shipping** a monetizable segment/show? (Not "is it cool.")
2. **Does the current pipeline already hit the quality bar** in [VISION.md](VISION.md)? If NOT, fix that
   first — you may not add capability on top of an unlocked base.
3. **Is it the boring/simplest option** that works? Prefer the resident farm + proven tools over the
   newest thing.

If it fails any, it goes to the **Parking Lot** in [BACKLOG.md](BACKLOG.md#parking-lot) (captured, not
built) and we move on. Revisit the parking lot only at roadmap-phase boundaries.

## 7. Prioritization + dependencies

- Every item has a **priority** (P0 = ships revenue / unblocks a P0; P1 = important; P2 = nice) and an
  explicit **Depends-on** list. An item is only pickable when its deps are `done`.
- [ROADMAP.md](ROADMAP.md) sequences epics into phases; [BACKLOG.md](BACKLOG.md) is the single source of
  truth for item status. Keep them in sync (updating an item is part of DoD).
- Re-prioritize at phase boundaries, not mid-flight (avoid thrash).

## 8. Changelog discipline

- `CHANGELOG.md` (repo root) follows **Keep a Changelog** + SemVer-ish. Every behavior/structure change
  adds a bullet under `## [Unreleased]` (Added / Changed / Fixed / Removed / Archived).
- Tag a release (date-stamped) when a roadmap milestone ships. The changelog is how we — and the AI
  showrunner later — know what changed and why.

## 9. Roles (now: human + Claude Code; later: the AI showrunner)

- **You (human):** own the vision, the quality bar, and go/no-go. The system exists to keep you shipping.
- **Claude Code:** executes items under this process. Must sequential-think, trailmark, verify, and doc.
- **AI showrunner (roadmap):** qwen9b/DeepSeek will own content decisions (topics, cadence, strategy
  updates from analytics) under the same DoD — see [STRATEGY.md](STRATEGY.md) and the roadmap epic.
