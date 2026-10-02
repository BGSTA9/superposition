# Cookbook report

Date: 2026-10-02 · Commits `7f127ca`..`f0f78c1` (5), plus this report · COOKBOOK_SPEC.md v1.0

> **Scope.** A documentation task, not a build phase. It added the research cookbook
> (`docs/cookbook/`) and the rebuild kit (`rebuild/`), and changed nothing in the replication:
> no file under `src/`, `configs/`, `tests/`, `results/`, `notebooks/` or `app/`, and no
> existing doc except one linking sentence each in README.md and docs/STUDY_GUIDE.md.

## Before changing anything

- `main` was clean at `2603068`, and no commit had come from outside a session since then.
- VS Code was running. It has committed and pushed half-finished work in this repository before
  (docs/cookbook/01_timeline.md, "Commits made from outside a session"). So all writing happened
  in a separate git worktree on a branch, `cookbook`. `main` was fast-forwarded only at the end,
  after checking that it was still clean at `2603068`.
- The transcripts of the eight Claude Code sessions were in `.cookbook-evidence/transcripts/`,
  which is git-ignored. They were read through a private extraction script,
  `.cookbook-evidence/extract.py`, whose output is in `.cookbook-evidence/extracted/`. While
  gathering evidence I also read a few facts straight from the raw transcripts with small
  scripts. I then extended the extraction script, so that every transcript fact the cookbook
  cites appears in its output. That includes the paper reads, the full owner prompts and each
  session's report to the owner.

## What was produced

**The cookbook** (`docs/cookbook/`, 31 files):
- **README.md:** two tracks (replay this project by hand; run your next project),
  prerequisites, the labels, compute measured from manifests, and **Known gaps**. Known gaps
  includes 15 inconsistencies found in the existing record.
- **The retrospective:** 01_timeline.md, the eight sessions, every phase, 131 commits, the
  statuses over time, and what went wrong.
- **Chapters:** 02 the research method, 03 from paper to spec, 04 scaffolding, 05 the engine,
  06 the debugging playbook, 07 reporting and review, 08 project management, 09 skills and
  drills, 10 the runbook for the next project (with a dry run on three candidates).
- **experiments/E00.md–E12.md:** one recipe card per experiment, in COOKBOOK_SPEC §6's ten
  sections.
- **templates/:** SPEC, experiment card, decision entry, phase report, reproduction entry, lab
  log.
- **VERIFICATION.md:** the verification record, below.

Every claim about the past cites a commit, file, decision, run, phase report or transcript
reference, and carries one label: [Documented], [Reconstructed] or [Recommended].

**The rebuild kit** (`rebuild/`, 11 files):
- the skeleton of `data`, `models`, `train` and `metrics`, generated from the real modules, so
  the signatures and docstrings are verbatim and every body raises `NotImplementedError`;
- the repository's 64 unit tests for those modules, with only their imports changed;
- a README with graded milestones: M1–M4, M5 (E00 by hand, compared with the cited run and
  SPEC's reference values) and M6 (E01, optional);
- `make rebuild-test`.

The kit imports two infrastructure modules from the real package, `superposition.utils` (seed
streams) and `superposition.config`. One test imports `superposition.theory.linear_optimum_loss`,
whose formula the owner derives in M5.

**How it was written.** As COOKBOOK_SPEC §9 asks:
- The evidence tables and 01_timeline.md came first.
- Seven writer agents then worked in parallel, each owning only its own files.
- I integrated the shared files, built the rebuild kit, and ran the verification.
- A separate verifier agent with fresh context checked the claims.

Two usage limits interrupted the writers; each one was resumed with its context intact.

## Verification results (details in docs/cookbook/VERIFICATION.md)

| Check | Result |
|---|---|
| §8.1 fidelity | A fresh-context verifier checked 88 [Documented] claims: 83 confirmed, 3 wrong, 1 unsupported, 1 mislabeled. All 5 were fixed or relabeled. It found no raw transcript text, no email address and no invented number. I fixed five more errors while integrating, among them when CI first passed (at `b50d062`, not at Phase 6) and a uv version that no run had used. |
| §8.2 commands | 226 distinct commands. Every unmarked one ran: in a fresh clone (143 runs over two batches), in a new empty folder (chapter 04 part B, 28), or as the two worktree flows with their placeholders filled in. All exit codes were as the text expects. 59 carry [Not verified: reason]. The 21 marked [Verified against manifest] match real manifests. All 72 `superposition` commands pass the `list` and `--help` checks. |
| §8.2 E00 and E01 | The quick profiles in the fresh clone gave PASS 11/11 (69 s) and PASS 12/12 (521 s). Every criterion value was identical to the cited runs (18/18, 14/14). |
| §8.3 rebuild kit | As shipped: 64 failed, all `NotImplementedError`. Real implementations substituted in place: 64 passed. Reverted with `git checkout`: clean tree, 64 failed again. |
| §8.4 integrity | Outside the new files, `git diff --stat 2603068` shows only README.md, docs/STUDY_GUIDE.md, the Makefile (`rebuild-test`) and pyproject.toml (ruff excludes `rebuild/`). `make lint test` is clean, with 261 passed. `.cookbook-evidence/` is ignored and nothing in it is tracked. `results/` still holds its 35 runs. |

## Decisions made

- **The kit's package is named `superposition_rebuild`.** Its tests run through their own
  `rebuild/pytest.ini`, so the kit stays out of the default `pytest`, out of CI, and out of the
  root conftest.
- **ruff skips `rebuild/` with `force-exclude = true`.** That keeps the skeleton's unused imports
  and the owner's half-finished work from failing pre-commit. Pyright's include list and
  pytest's `testpaths` already left it out, so neither changed.
- **The skeleton is generated from the real modules,** not typed by hand. This keeps every
  signature and docstring exact, and the same generator substituted the real code for the
  two-way check.
- **Commands that read the git-ignored PDF were checked with the owner's copy** placed in the
  clone's `references/`, as a reader would have it. Their text says they need the PDF.

## Known gaps

- Everything listed in docs/cookbook/README.md, "Known gaps":
  - the planning conversation (only COOKBOOK_SPEC §4 is used);
  - what the scratchpad prototypes and development runs produced;
  - runs no longer on disk;
  - the owner's work outside Claude Code;
  - human time (no effort estimates are given);
  - workflow internals and the reviewers' own scripts;
  - the full profiles and the paper tier, which were never run.
- The old sessions' scratchpad folders still exist under `/private/tmp` on this machine. They are
  outside the evidence COOKBOOK_SPEC §3 allows, so they were not examined.
- Not run, for the reasons given beside each command:
  - `figures --publish` and `report --pin` (they modify tracked docs);
  - the never-run full profiles and the paper tier;
  - `superposition app` (covered by the slow tier's launch test, which passed);
  - `git push` and `gh run list`;
  - `brew install poppler`.
- **A side effect on this machine.** Chapter 04's `uv python install 3.12` downloaded a
  uv-managed Python 3.12.14 into uv's own store. No project file changed.
- **Existing docs that disagree with the evidence.** They are listed in the cookbook README but
  deliberately not fixed, because this task could not edit existing docs. Among them:
  - D-R02 still gives the pp. 23–24 justification that review R2-03 withdrew;
  - D-R04 still says TopK L0 = k "by construction";
  - the Makefile's `report` help still says "latest runs";
  - PLAN.md still quotes E03's superseded PASS;
  - README and CITATION.cff do not disclose that Claude Code wrote the code or that the reviewers
    were AI agents.

  Each is a decision for you.
- **The two decisions Phase 6 left open are still open:** E02's seed question (R1-01) and E03's
  edge threshold (R2-01).

## What the owner should look at

Read docs/cookbook/README.md first, then 01_timeline.md's "What went wrong" section. The cookbook
is built on that section. Then run `make rebuild-test`, open `rebuild/README.md`, and start M1
without opening `src/superposition/data.py`.

Three questions:

1. 01_timeline.md shows the Phase 4 session starting while Phase 3 was unfinished, because its
   only stop condition was a red `make lint test`. What stop condition would have caught it, and
   how would you check it by hand at the start of a working session?
2. VERIFICATION.md's verifier found that a commit was dated 24 minutes before the event it was
   said to follow. Which kind of claim in a retrospective is easiest to get wrong this way, and
   what evidence settles it?
3. The fresh-clone E00 run matched the cited run in every criterion value. What would you
   conclude if it had matched in status but not in values, and what would you check first?
