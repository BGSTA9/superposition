# Phase N report: {phase title}

<!--
One report per phase, written after the phase's exit criteria have actually been run (SPEC §11
"Closing every phase"). Save as docs/progress/phase-N.md. The real reports in
../../progress/ are worked examples: phase-3.md and phase-6.md show "Before changing anything";
phase-2.md and phase-4.md show commit notes. Delete these comments.
-->

Date: YYYY-MM-DD · Commits `<first>`..`<last>` · SPEC §__ · Decisions D-___–D-___

> **Commit note** (only if needed). {Commits made from outside your working session, drafts
} committed early, history that was not rewritten, and what each contains.>

## Before changing anything

<!-- What you checked before the first edit. -->

- [ ] The previous phase report exists and its exit criteria are ticked: ____
- [ ] Tree clean at `<hash>`; lint and tests green (____ passed)
- [ ] Nothing else is writing to the repository (other sessions, editors with auto-commit,
      scheduled jobs, leftover servers): ____
- [ ] Commits since the last report that you did not make: ____

## What was built

- **Library:** ____
- **Experiments:** ____
- **Docs, notebooks, tools:** ____

## Runs

| Exp. | Run ID | Profile | Seed | Commit | Dirty | Wall-clock | Status | Cited as |
|---|---|---|---|---|---|---|---|---|
| | | | | | false | s | | |

Not cited (exploratory, superseded, outside `results/`): ____

## Test status

| Tier | Command | Result |
|---|---|---|
| Lint | | |
| Unit + property | | ____ passed in ____ s |
| Slow / replication | | |

Tests added or changed, and why (disclose any relaxed tolerance or changed seed): ____

## Acceptance table

| Exp. | Criterion | Source (page / proposal) | Value | Threshold | Status |
|---|---|---|---|---|---|
| | | | | | |

## Decisions made

- D-___: {one line each, linking DECISIONS.md}

## Open issues

1. **{title}.** {what is unresolved, the evidence, what would settle it, whose decision it is}

## What to look at and learn from this phase

<!-- A short path through the phase's results: which figure next to which paper page, which
file to read, which page or notebook to try. -->

Three questions:

1. ____
2. ____
3. ____
