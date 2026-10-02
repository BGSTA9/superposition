# SPEC.md: `<project name>`

<!--
Skeleton of a project specification, following the section structure of this repository's
SPEC.md. Copy it to your new repository's root and fill every field. Guidance sits in comments
like this one; delete them as you go. Write it before any code, after reading the paper in full
(cookbook 10_next_project.md, stages 1-7). Terms: ../../GLOSSARY.md.
-->

A `<replication | extension | new study>` of *{paper title}* ({authors}, {year}).

Version {x.y} · {YYYY-MM-DD}

> **How to use this document.** The single source of truth for this project. Page numbers refer
> to `<references/file.pdf>`, a {N}-page print of {URL}. Re-read the sections a phase touches at
> the start of that phase.

---

## 1. Mission

<!-- What the project does, in 3-4 numbered items. One must be the question you could be wrong
about. -->

1. ____
2. ____

### Who this is for

<!-- Your background and what you want to learn. It sets how much the code explains itself. -->

## 2. Research integrity rules (non-negotiable)

<!-- Keep these headings. Adapt the wording only to tighten it. -->

**Provenance.** Results come only from code in this repo; every number traces to a logged run
(config, seed, commit, dirty flag).

**Honest reporting.** A result that does not reproduce: check the implementation → more restarts
or steps → document → report PARTIAL or FAIL. Never weaken a criterion without a DECISIONS entry.
Tests verify correctness; they do not define it.

**Decisions and defaults.** Every under-specified choice: a principled, configurable default and
a DECISIONS entry (decision, rationale, page, alternatives). Variants never change a default.

**Independence.** Reference implementations: ____. Consulted only to diagnose a failure after an
honest attempt, and logged.

**Licensing.** ____ (e.g. never commit the paper PDF).

## 3. Paper facts and conventions (implement exactly)

<!-- Only what the paper states, each with its page. Choices go to DECISIONS.md, not here. -->

### Notation

| Symbol | Meaning | Name in code | Page |
|---|---|---|---|
| | | | |

### Data

- ____ (p. __)

### Models

| Model | Page | Equations |
|---|---|---|
| | | |

### Loss (p. __)

<!-- Write the reduction order explicitly: which axes are summed, which averaged. -->

### Metrics

- ____ (p. __). Edge cases (zero vectors, empty sets): ____

### Theory and closed forms

<!-- Any quantity you can derive exactly. Each becomes a unit test. -->

### Optimization notes from the paper

<!-- Restarts, selection rules, things the paper says are hard. -->

### Initialization

### Visualization conventions

## 4. Repository layout

<!-- A tree with one comment per entry. Keep `results/` git-ignored. -->

## 5. Engineering standards

**Environment and dependencies.** ____

**Compute.** Reference hardware: ____. Budget per `quick` profile: ____ minutes.

**Code.** Type hints; named tensor shapes; no magic numbers (thresholds live in configs).

**Run artifacts.** `results/<experiment>/<run_id>/` with config, manifest (commit, dirty flag,
versions, device, seed, times, wall-clock), metrics, acceptance record, checkpoints, figures, log.

**CLI.**

| Command | What it does |
|---|---|
| | |

**Testing tiers.** Unit (fast, CI) · property · replication (`slow`) · ____

## 6. Core library requirements

<!-- One subsection per module: interface, responsibilities, and the tests to write first. -->

### `<module>.py`

- Interface: ____
- Required known-answer tests:

  | Configuration | Expected result | Source (page or derivation) |
  |---|---|---|
  | | | |

- Required property tests: ____

## 7. Experiments

<!-- The thresholds below are initial proposals; §2 governs any change. Mark each criterion's
source: (paper p. __) or (proposal). One EXPERIMENT_CARD.md per experiment. -->

### E00 `<name>` (p. __)

**Setup:** ____

**Accept:**

| Condition | Criterion | Source |
|---|---|---|
| | | paper p. __ / proposal |

**Reference values** (from the verification run; sanity checks, never criteria): ____
Evaluation samples: ____ · restarts: ____ · how the best was selected: ____

### Open questions (pp. __)

<!-- A menu of extensions. Do not implement them in the replication. -->

## 8. Interactive tools (optional)

## 9. Notebooks (optional)

## 10. Documentation

| File | Contents |
|---|---|
| `docs/REPRODUCTION.md` | per experiment: claim with page, figure, status, evidence, discrepancies with hypotheses, run IDs |
| `docs/DECISIONS.md` | dated entries: decision, rationale, page, alternatives |
| `docs/REVIEW.md` | review findings and their resolutions |
| `docs/progress/phase-N.md` | one report per phase |

## 11. Execution plan

### Closing every phase

1. Run lint, tests, and the quick profile of every experiment the phase touched.
2. Publish figures; update REPRODUCTION.md, CHANGELOG.md and PLAN.md.
3. Commit.
4. Write `docs/progress/phase-N.md` (template: PHASE_REPORT.md).

### Phases

<!-- Phase 0 bootstrap; Phase 1 engine, alone; experiment phases; a final review phase. Each
with exit criteria that can be run. The next phase starts only when the previous report exists
and its exit criteria are ticked. -->

**Phase 0: ____.** Exit when: ____

### State across interruptions

<!-- What is the source of truth after a break: PLAN.md checkboxes, phase reports, git log,
tests. -->

## 12. Definition of done

- [ ] Every phase complete, exit criteria verified
- [ ] Every experiment: a cited run, published figures, an honest REPRODUCTION.md entry
- [ ] Every review finding resolved or listed as open
- [ ] A fresh clone passes: ____
