# COOKBOOK_SPEC.md: the `superposition` research cookbook

Version 1.0 · 2026-10-01

> **How to use this document.** This is the complete specification for one documentation task in the `superposition` repository: writing a cookbook that teaches the owner to run the next research project by hand. Read it in full before doing anything. It sits beside SPEC.md and does not replace it.

---

## 1. Goal

Write a cookbook that does two jobs.

1. **Retrospective.** An evidence-based, step-by-step account of how this repository was actually built: the real sequence, the commands, the decisions, the failures and how each was handled.
2. **Playbook.** A generalized, inch-by-inch recipe the owner can follow alone on a new project, with templates, checklists, drills and a rebuild kit.

**The owner** is strong in Python, PyTorch and linear algebra, and new to mechanistic interpretability. In this repo:
- Claude Code wrote the code, working from SPEC.md.
- SPEC.md itself was drafted with Claude in a separate planning conversation.
- The owner directed the work, made the decisions and reviewed the results.

The cookbook must leave the owner able to write this kind of code and run this kind of project with no AI writing it. Optimize every chapter for that transfer of skill, not for documenting the AI.

## 2. Integrity rules (non-negotiable)

**Evidence for every claim about the past.** Every statement about what was done cites its evidence:
- a commit hash;
- a file path and section;
- a DECISIONS ID;
- a run ID;
- a phase report;
- or a transcript reference (session ID plus timestamp).

**Label every step** with exactly one of these:

| Label | Meaning |
|---|---|
| [Documented] | Directly shown by the cited evidence. |
| [Reconstructed] | Inferred from evidence. Say from what, and how. |
| [Recommended] | Advice for doing it by hand. Not something that happened. |

Never present a reconstruction or a recommendation as documented.

**Keep the mess.** Record what actually happened and how each problem was resolved, including:
- Phase 3 left unfinished across three sessions;
- exploratory runs outside the provenance system;
- commits from outside a session;
- grids refined after failed runs;
- acceptance criteria that turned out to be wrong;
- selection by ground truth.

A polished, linear story would teach the owner the wrong process.

**Say "unknown" when evidence is missing.** This covers pruned transcripts, workflow internals and scratchpad runs. State what is missing. Never invent commands, numbers, timings or reasons.

**Transcripts stay private.** They may contain secrets or personal data. Extract facts from them, but never paste raw transcript text into tracked files, and never commit transcripts.

**The replication stays untouched.** Change nothing in:
- `src/`, `configs/` or the existing `tests/`;
- `results/`;
- notebooks or the app;
- existing docs.

The only allowed edits outside new files:
- a link to the cookbook in README.md and in docs/STUDY_GUIDE.md;
- `.gitignore` entries;
- a `rebuild-test` Makefile target;
- lint, test and CI configuration that excludes `rebuild/`.

**Every command in the cookbook is verified** as §8 specifies, or marked [Not verified] with the reason.

## 3. Evidence sources

Read these in order. Use scripts rather than loading large files into context.

1. SPEC.md, CLAUDE.md, PLAN.md and README.md.
2. Everything in `docs/`:
   - the progress reports, for both phases and rectification sessions;
   - DECISIONS.md, REPRODUCTION.md and REVIEW.md;
   - PAPER_MAP.md, DESIGN.md and STUDY_GUIDE.md.
3. **Git history.** This is the authoritative timeline. Read it with `git log --reverse --stat`, including dates and authors. It shows what was built, in what order, and in which phase.
4. **Run records** under `results/`. Each run's `manifest.json`, `config.yaml` and `acceptance.json` show what ran, when, for how long and with what outcome. `results/` is git-ignored, so state what exists and what doesn't.
5. Configs, tests, the Makefile, the CI workflow and pyproject.toml. These give the exact commands and checks.
6. **Claude Code session transcripts**, if present in `.cookbook-evidence/transcripts/` (git-ignored).
   - Write a small extraction script inside `.cookbook-evidence/`.
   - Per session, have it pull:
     - start and end times;
     - which phase the session served;
     - the shell commands in order (setup, tests, runs, git);
     - files created or edited;
     - errors hit and how they were resolved.
   - Subagent transcripts may sit in per-session `subagents/` folders.
   - Work from the script's output, not from raw transcripts.
   - If transcripts are absent or partial, say so and continue from the repo evidence.
7. **§4 below.** This is the owner's account of the planning conversation, which you cannot see.

## 4. Context you can't see: the planning conversation

Cite this section as "[Planning conversation, per owner]". The items below were recommendations made in that conversation. DECISIONS.md and the progress reports are the record of what was actually done.

### Reading and understanding before building

- **Reading the paper.** The paper was read in full from the PDF's text layer. Pages whose equations or figures the text layer misses (pp. 2, 13, 14, 17, 19) were rendered as images and read visually. A page map of sections was built by searching each page's text.
- **Study aids.** These came before any code:
  - a seven-stage reading plan;
  - a hand-built interactive pentagon: five features in two dimensions, with ReLU and a negative bias, showing interference;
  - "explain it back" questions;
  - a Johnson–Lindenstrauss warm-up.
- **Warm-up result.** Random unit vectors with N = 2d gave a largest |cos| of:

  | d | Largest \|cos\| |
  |---|---|
  | 2 | 0.98 |
  | 10 | 0.79 |
  | 100 | 0.41 |
  | 1,000 | 0.15 |

### Verification before specification

- **The intro experiment was run before its criteria were written.** It ran in JAX, for both the ReLU and linear models, before E00's acceptance criteria existed. Its settings:
  - n = 5, m = 2, Iᵢ = 0.9ⁱ, S ∈ {0, 0.8, 0.9};
  - 8 restarts;
  - Adam at learning rate 1e-3, batch size 1,024, 10,000 steps;
  - Xavier initialization with std √(2/(n + m)).
- **The linear model's optimum was derived in closed form.** It is Σ_{i≥m} Iᵢ(p/3 − p²/4), and it matched those runs.
- **E03's thresholds came from the paper's figure.** The D* curve was read off the p. 17 figure at 300 dpi.
- **Page references were checked.** Every page reference in SPEC.md was checked against per-page text.
- **External state was checked before the execution plan was written:**
  - Anthropic's reference repo (archived June 2025);
  - the contents of ARENA §1.5.4;
  - Claude Code's documentation for ultracode, workflows, CLAUDE.md and model settings.

### Course corrections after Phases 3–5

1. Don't start Phase 6 while Phase 3 is open. Run two rectification sessions first.
2. E08's "more than m/2 features computed well" criterion came from the spec, not the paper. Re-derive it from p. 37 and footnotes 16–17, and keep the original FAIL visible.
3. E05 stage B: treat the different trajectory as a documented variant, per pp. 23–24 and 27–28.
4. E11 pentagon: mark it PARTIAL and add a fresh-seed held-out check. Its pass rested on one λ, from a grid refined after a failed run.
5. E10: choose the headline result by a rule that uses no ground truth, and report the oracle as an upper bound.
6. E09: make the density pilot a cited run.
7. Split the slow test tier.

### Publication guidance

- Disclose AI assistance precisely.
- Present the website honestly, as one early-career researcher's work.
- Write results in the owner's own words.
- Publish only findings the owner can explain.

## 5. Deliverables

Everything goes in `docs/cookbook/` unless stated otherwise.

### Chapters

- **`README.md`**
  - How to use the cookbook.
  - Two tracks: **A**, replay this project by hand; **B**, run your next project.
  - Prerequisites.
  - Compute times measured from manifests. Give no invented human-effort estimates.
  - What the three labels mean.

- **`01_timeline.md`**
  - The actual build, phase by phase, as a table: date · session or phase · commits · key outputs · issues.
  - A narrative of what went wrong and how it was fixed.

- **`02_the_research_method.md`**
  - The loop this project followed:
    1. question
    2. paper
    3. facts with pages
    4. under-specification log
    5. pre-declared acceptance criteria
    6. verification run
    7. plan
    8. engine
    9. experiments
    10. honest report
    11. independent review
  - For each step: purpose, inputs, outputs, how it was done here (with evidence), how to do it by hand, and its failure modes.

- **`03_from_paper_to_spec.md`**: an inch-by-inch recipe for turning a paper into a SPEC.
  - Reading strategy: the text layer plus rendered pages.
  - Building the page map.
  - Notation and conventions.
  - Finding what's under-specified.
  - Designing acceptance criteria, paper-grounded versus proposed (the E08 lesson).
  - Verification runs before committing to numbers (the E00 lesson).
  - Reference values versus criteria.
  - Closed-form checks.
  - It ends with one paper section worked end to end.

- **`04_scaffolding.md`**
  - The environment and repo setup exactly as done, taken from history, the Makefile and CI.
  - In order, each step with its purpose and a check.

- **`05_engine.md`**
  - First, the order the core library was actually built in, from git history.
  - Then, the recommended order for building it by hand.
  - For each module:
    - its responsibilities and interface;
    - the key math, with page references;
    - the tests to write first, drawn from the existing tests;
    - pitfalls hit here, with evidence;
    - check commands;
    - a pointer to the repo implementation as the answer key, to open only after trying.

- **`experiments/E00.md` … `E12.md`**
  - One recipe card per experiment, following the template in §6.

- **`06_debugging_playbook.md`**: decision procedures learned here.

  | Problem | Example from this repo |
  |---|---|
  | Training failure versus model limitation | E08's dead output ReLUs |
  | Restarts for hard optimization | p. 23 |
  | Forking paths, and held-out confirmation | E11 |
  | Selection by ground truth | E10 |
  | Provenance gaps | the E09 pilot; scratchpad runs |
  | Trajectory variance versus failure | E05 stage B |
  | When and how to revise a criterion honestly | E08 |

- **`07_reporting_and_review.md`**
  - How REPRODUCTION.md, DECISIONS.md, the phase reports and the independent review were done.
  - What a reviewer looks for.
  - How to publish honestly, following §4.

- **`08_project_management.md`**
  - Phases, sessions, state files, test tiers and cost.
  - The process failures, and the rules adopted because of them:
    - the unfinished Phase 3;
    - concurrent writers;
    - runs outside the provenance system.

- **`09_skills_and_drills.md`**
  - Every skill the owner needs to work alone, for example:
    - batched einsum training;
    - metrics with known-answer tests;
    - criterion design;
    - reading weight plots;
    - ablations;
    - SAE training;
    - any others found in the repo.
  - For each: where it appears in this repo, plus two or three drills with self-checks.

- **`10_next_project.md`**: a fill-in-the-blanks runbook for the owner's next project, done entirely by hand.
  - One checklist per stage.
  - Links to the templates.
  - A dry run on two or three candidate projects sized for one person.
  - Candidates come from STUDY_GUIDE's open-question menu, or from the owner's interest in pattern discovery in number theory (for example, reverse-engineering a small network trained on modular arithmetic).
  - For each candidate, give:
    - the question;
    - the paper pages it builds on;
    - the smallest useful experiment;
    - draft acceptance criteria, labeled as proposals;
    - a compute estimate, labeled as an estimate;
    - the risks.
  - Claim no novelty. The owner's first step on any candidate is a literature check.

- **`templates/`**
  - SPEC_TEMPLATE.md
  - EXPERIMENT_CARD.md
  - DECISION_ENTRY.md
  - PHASE_REPORT.md
  - REPRODUCTION_ENTRY.md
  - LAB_LOG.md

### The rebuild kit, at `rebuild/` in the repo root

This is where the skill transfer happens.

**Contents**
- A stripped skeleton of the core engine: `data`, `models`, `train`, `metrics`. Keep the real signatures and docstrings; every body raises `NotImplementedError`.
- The existing unit tests for those modules, adapted to import from the skeleton.
- A README with graded milestones:
  1. M1: data
  2. M2: models
  3. M3: training
  4. M4: metrics
  5. M5: reproduce E00 by hand, and compare against the reference values
  6. M6 (optional): reproduce E01

**Isolation**
- Exclude `rebuild/` from the main package, lint, type checks, default pytest collection and CI.
- `make rebuild-test` runs its tests.

**Verify it both ways**
1. As shipped, every rebuild test fails with `NotImplementedError`.
2. With the real implementations temporarily substituted, every test passes.
3. Revert the substitution.

### Links

Link the cookbook from README.md and from docs/STUDY_GUIDE.md.

## 6. Recipe card template (one per experiment)

1. **Claim and pages.** What the paper says, and where.
2. **Prerequisites.** The modules and earlier experiments it depends on.
3. **Config and commands.** The quick and full profiles, and the exact commands.
4. **Procedure.**
   - The steps as executed, numbered [Documented].
   - Then how to do it by hand [Recommended].
   - Each step gives: action → command → expected result → how to check → what to do if it fails.
5. **Acceptance.** For each criterion:
   - where it came from (paper or spec proposal);
   - its final status;
   - the evidence (run IDs).
6. **What actually happened.**
   - Issues, diagnosis, and decisions (DECISIONS IDs).
   - Anything changed after results were seen, flagged as such.
7. **Compute.** Measured wall-clock time from manifests. Label estimates as estimates.
8. **What to look at.** The figures, the app page and the notebook.
9. **Explain it back.** Three questions and one drill.
10. **Answer key.** The files and functions to open, after attempting.

## 7. Style

- **Write for the owner.** Be plain and direct. Explain *why* at every step. Define each term on first use, and link GLOSSARY.md.
- **Go inch by inch.** Use numbered steps, each with an action, command, expected result, check and failure response.
- **Don't dump code.** Show interfaces, equations, tests and short excerpts only. Full implementations stay in the repo as the answer key.
- **Format.** Markdown, one H1 per file, relative links. Reference figures from `docs/figures/`.

## 8. Verification

1. **Fidelity.**
   - A verifier with fresh context samples at least 30 [Documented] claims across the chapters.
   - It checks each claim against its cited evidence.
   - Fix or relabel every claim that fails.
   - Log the results in `docs/cookbook/VERIFICATION.md`.
2. **Commands.** Work in a fresh clone in a temporary directory.
   - Run every command in `04_scaffolding.md` and `05_engine.md`, plus the E00 and E01 quick profiles.
   - For the other experiments, check each command three ways: against `superposition list`, against the CLI's `--help`, and against an existing run manifest it would have produced. Label it [Verified against manifest].
   - When a command fails, fix the text, never the code.
3. **Rebuild kit.** Run the two-way check from §5.
4. **Integrity.**
   - `git diff --stat` against the starting commit shows changes only where §2 allows.
   - `make lint test` passes.
   - `.cookbook-evidence/` is git-ignored, and nothing under it is tracked.

## 9. Execution

1. **Sequential: evidence.** Build the commit table, the run table and the transcript facts. Write `01_timeline.md`.
2. **Parallel: chapters and recipe cards.**
   - Each agent owns only its own files.
   - Shared files (`README.md`, cross-links) are integrated in one step at the end.
3. **Sequential: the rebuild kit.**
4. **Verification.** Run §8 with a verifier that has fresh context.
5. **Close.**
   1. Commit, using conventional commits.
   2. Write `docs/progress/cookbook.md`: what was produced, the verification results, and the known gaps.
   3. Stop.

Don't ask the owner questions mid-run. Record every gap in a "Known gaps" section of the cookbook's README.
