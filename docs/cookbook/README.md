# The superposition research cookbook

This cookbook does two jobs.

1. **Retrospective.** It records how this repository was actually built: the real order of
   work, the commands, the decisions, and every failure and how it was handled.
2. **Playbook.** It turns that record into a recipe you can follow alone on your next project,
   with templates, checklists, drills and a rebuild kit.

It is written for the owner of this repository: strong in Python, PyTorch and linear algebra,
and new to mechanistic interpretability. In this repository Claude Code wrote the code from
SPEC.md; SPEC.md itself was drafted with Claude in a separate planning conversation; the owner
directed the work, made the decisions and reviewed the results. The aim of this cookbook is
that you can write this kind of code and run this kind of project with no AI writing it. Every
chapter is written for that transfer of skill, not to document the AI.

The specification of this cookbook is [COOKBOOK_SPEC.md](../../COOKBOOK_SPEC.md). Terms are
defined in the [glossary](../GLOSSARY.md).

## The three labels

Every claim about the past, and every step, carries exactly one label.

| Label | Meaning |
|---|---|
| **[Documented]** | The cited evidence shows it directly. |
| **[Reconstructed]** | Inferred from evidence. The text says from what, and how. |
| **[Recommended]** | Advice for doing it by hand. It is not something that happened. |

Evidence is cited as a commit (`` `340f825` ``), a file and section, a decision ID (D-309, in
[DECISIONS.md](../DECISIONS.md)), a run ID (`20260930T090540Z-4e3325bf`, whose manifest is under
`results/`, git-ignored), a phase report in [docs/progress/](../progress/), or a Claude Code
transcript as `[S <session prefix> @ <UTC time>]`. The session prefixes are listed in
[01_timeline.md](01_timeline.md). The transcripts themselves stay private on the owner's machine
and are never committed. What the owner reports about the planning conversation is cited as
[Planning conversation, per owner].

Commands carry their own marks. A command with no mark was run when this cookbook was written:
in a fresh clone, or, for chapter 04's part B, in a new empty folder
([VERIFICATION.md](VERIFICATION.md)). **[Verified against manifest]** means
an existing run's `manifest.json` records exactly that command. **[Not verified: reason]** says
why a command was not run.

## Two tracks

### Track A · Replay this project by hand

Rebuild the engine yourself, then reproduce the experiments one by one and compare your numbers
with the repository's cited runs.

1. Skim [01_timeline.md](01_timeline.md) to see what really happened, mess included.
2. Read [03_from_paper_to_spec.md](03_from_paper_to_spec.md) with the paper open. Write your own
   one-page spec for E00 before reading SPEC.md's.
3. Follow [04_scaffolding.md](04_scaffolding.md), part B, to set up a repository of your own, or
   work inside [`rebuild/`](../../rebuild/README.md) in this one.
4. Build the engine with [05_engine.md](05_engine.md) and the
   [rebuild kit](../../rebuild/README.md): milestones M1 data, M2 models, M3 training, M4
   metrics.
5. Reproduce E00 (kit milestone M5) and E01 (M6) with the recipe cards
   [experiments/E00.md](experiments/E00.md) and [experiments/E01.md](experiments/E01.md).
6. Continue through [E02](experiments/E02.md) to [E12](experiments/E12.md). Each card gives the
   claim and pages, what was run, how to do it by hand, the acceptance criteria and where they
   came from, what went wrong, and an "explain it back" drill.
7. When something does not reproduce, use [06_debugging_playbook.md](06_debugging_playbook.md).
8. Write up what you found with [07_reporting_and_review.md](07_reporting_and_review.md) and the
   [templates](templates/).
9. Practise the underlying skills with [09_skills_and_drills.md](09_skills_and_drills.md).

### Track B · Run your next project

1. Learn the loop: [02_the_research_method.md](02_the_research_method.md).
2. Turn a paper into a spec: [03_from_paper_to_spec.md](03_from_paper_to_spec.md) and
   [templates/SPEC_TEMPLATE.md](templates/SPEC_TEMPLATE.md).
3. Plan phases, sessions and state files: [08_project_management.md](08_project_management.md).
4. Follow the runbook, stage by stage: [10_next_project.md](10_next_project.md). It ends with a
   dry run on candidate projects sized for one person.
5. Report and review honestly: [07_reporting_and_review.md](07_reporting_and_review.md).
6. Keep [06_debugging_playbook.md](06_debugging_playbook.md) open while you run experiments.

## Chapters

| File | What it gives you |
|---|---|
| [01_timeline.md](01_timeline.md) | the actual build, phase by phase and session by session, and what went wrong |
| [02_the_research_method.md](02_the_research_method.md) | the eleven-step loop this project followed, step by step |
| [03_from_paper_to_spec.md](03_from_paper_to_spec.md) | turning a paper into a specification, with one section worked end to end |
| [04_scaffolding.md](04_scaffolding.md) | the environment and repository setup, as done and by hand |
| [05_engine.md](05_engine.md) | the core library, module by module, in the order to build it |
| [experiments/](experiments/) | one recipe card per experiment, E00 to E12 |
| [06_debugging_playbook.md](06_debugging_playbook.md) | decision procedures for failures met here |
| [07_reporting_and_review.md](07_reporting_and_review.md) | how the reports and the independent review were done; publishing honestly |
| [08_project_management.md](08_project_management.md) | phases, sessions, state files, test tiers, cost, and the process failures |
| [09_skills_and_drills.md](09_skills_and_drills.md) | every skill you need to work alone, with drills |
| [10_next_project.md](10_next_project.md) | a fill-in-the-blanks runbook for your next project |
| [templates/](templates/) | SPEC, experiment card, decision entry, phase report, reproduction entry, lab log |
| [VERIFICATION.md](VERIFICATION.md) | how this cookbook was checked: claims, commands, the rebuild kit |
| [../../rebuild/](../../rebuild/README.md) | the rebuild kit: the engine's skeleton and its tests |

## Prerequisites

- Python, PyTorch and linear algebra at the level of writing a training loop and reading an
  einsum without help.
- [uv](https://docs.astral.sh/uv/) and git. Everything runs on a laptop CPU; every timing below
  is from an Apple M2.
- The paper, *Toy Models of Superposition* (Elhage et al., 2022). Page numbers everywhere refer
  to the 62-page PDF print described in [references/README.md](../../references/README.md). The
  PDF is not in the repository; download it yourself and keep it git-ignored.
- For reading figures and equations the PDF's text layer misses: poppler's `pdftotext` and
  `pdftoppm` ([03_from_paper_to_spec.md](03_from_paper_to_spec.md)).

## Compute, measured

All numbers are wall-clock seconds from the `manifest.json` of the repository's runs (Apple M2;
CPU unless noted). This cookbook gives no estimates of human effort: the evidence does not
contain them.

| Exp. | Quick profile, cited run | Other measured runs |
|---|---|---|
| E00 intro_pentagon | 68 | full profile 159 (MPS) |
| E01 basic_results | 472 | full profile 1,673 (MPS) |
| E02 phase_change | 481 | seeds 1 and 2: 498 each; batch1024 variant 1,597 |
| E03 uniform_geometry | 346 | batch1024 variant 1,472 |
| E04 nonuniform_geometry | 287 | batch1024 variant 952 |
| E05 learning_dynamics | 518 | — |
| E06 adversarial | 401 | — |
| E07 privileged_basis | 419 | — |
| E08 abs_computation | 493 | full profile 981 (MPS) |
| E09 mech_interp_abs | 255 | pilot profile 200 |
| E10 sae_recovery | 415 | — |
| E11 l1_removes_superposition | 548 | held-out seed 546 |
| E12 activation_functions | 507 | — |

The 13 cited quick runs take 5,210 s together, about 87 minutes. Every run directory under
`results/` (35 runs, including superseded and variant runs) adds up to 18,292 s, about 5.1
hours. Full profiles of E02–E07 and E09–E12 were never run; their runtimes in `configs/` are
labeled as estimates (D-R06). The repository's test tiers: `make test` takes under a minute in
parallel, `make test-slow` 429 s, and `make test-paper` has never been timed as a whole
(docs/progress/phase-6.md, "Test status").

## Known gaps

What this cookbook could not establish, and why.

- **The planning conversation.** SPEC.md was drafted in a conversation that is not available.
  Only the owner's summary in COOKBOOK_SPEC §4 is used, labeled [Planning conversation, per
  owner]. The JAX verification code behind SPEC's E00 reference values, what was tried and
  discarded, and how long it took are unknown.
- **Scratchpad runs.** Prototypes and development runs were run with scratchpad scripts or into
  scratchpad results folders, outside `results/` ([01_timeline.md](01_timeline.md), "Runs
  outside the provenance system"). Their configurations and full outputs are unknown; only the
  numbers quoted in DECISIONS.md are known. The scratchpad folders may still exist under
  `/private/tmp` on the owner's machine; they are outside the evidence this cookbook was allowed
  to use and were not examined.
- **Runs no longer on disk.** The dirty-tree runs moved aside in the first session, and at least
  one E11 development run (docs/REVIEW.md R4-07), are not in `results/`.
- **The owner's own work.** Commits made from VS Code have no transcript. What the owner read,
  checked or decided between sessions is known only where a later prompt or report says so.
- **Human time.** Session start and end times are known; the time the owner spent reading,
  reviewing and deciding is not. No human-effort estimates are given.
- **Workflow internals.** How Claude Code ran its subagents, and the five reviewers' own analysis
  scripts (docs/REVIEW.md), are not part of the evidence. Numbers that only a reviewer's script
  produced are labeled as such in REVIEW.md.
- **Full profiles and the paper tier.** Most full profiles, and `make test-paper` as a whole,
  were never run, so their runtimes are estimates.

### Inconsistencies found in the existing record

Writing this cookbook turned up places where the repository's own documents disagree with each
other or with the evidence. COOKBOOK_SPEC §2 forbids editing the existing docs in this task, so
they are listed here for the owner to decide on. Each is **[Documented]** by the evidence named,
except item 1, which is **[Reconstructed]**.

1. D-105 says E01's criteria were fixed before the first E01 run. The criteria were in code
   before the run (08:45Z), but the DECISIONS text was most likely appended at 08:49Z, while
   the run that started at 08:46Z was training. **[Reconstructed]**: from timestamps alone; the
   08:49Z append is the only write to DECISIONS.md between 08:20Z and 09:02Z, and the extraction
   does not show its content. [S 35cab3f0 @ 2026-09-29T08:45–08:49Z]
2. D-R02 and rectification.md still justify E05's stage B with pp. 23–24; review R2-03 says
   that justification is wrong, and REPRODUCTION.md was corrected, but D-R02 was not.
3. D-R04 still says TopK SAEs have L0 = k "by construction"; review R4-05 says that is wrong,
   and only REPRODUCTION.md was corrected.
4. Review R3-04 says E07's sidestepping row is now labeled by the sparsity actually used; the
   cited run's `acceptance.json` still keys it `1-S=0.1` (the fix is in the docs only).
5. D-603 says the ±0.02 windows of 2/5 and 3/8 "overlap on [0.355, 0.42]"; that interval is
   their union. They overlap on [0.38, 0.395].
6. D-505 lists 0.15 and 1 as the values dropped from E11's λ grid; 0.001, in the first
   development grid, is gone from `configs/l1_removes_superposition.yaml` too.
7. D-606 says every cited run after E00–E02 was launched from a clean detached worktree; the
   cited runs of E06 and E07 (`5a825d0`) ran from the main tree while it was clean (their
   manifests' `command` fields).
8. phase-4.md gives the bias-init variant's loss reduction on the full profile as 29%; the
   run's record (0.000675 → 0.000484) gives 28.3%.
9. phase-6.md reports 260 unit tests; at `2603068` the fast tier has 261, because `2564cf1`
   added one.
10. The Makefile's help for `make report` still says "from the latest runs", the phrase review
    R1-09 corrected in the README; per D-202 the report uses the cited runs.
11. PLAN.md's Phase 3 entry still quotes E03 as PASS 4/4 with the superseded run
    `20260930T091350Z-2cb2dc61`; the cited run is PARTIAL 3/4.
12. The runtime comments in `configs/` quote the first cited runs (e.g. E00 62 s), not the
    current cited runs (E00 68 s).
13. An L1 unit test's margin was relaxed from 0.5 to 0.8 with a comment giving the observed
    ratio (≈ 0.62) [S 35cab3f0 @ 2026-09-29T08:33Z]; phase-1.md, which reports the density
    test's change, does not mention it.
14. references/README.md says to print the paper on US Letter; the local PDF has 62 pages but a
    576 × 792 pt page size, not Letter's 612 × 792 (`pdfinfo`, run while writing chapter 03).
15. README.md and CITATION.cff do not say that Claude Code wrote the code or that the
    "independent review" was done by AI agents. [07_reporting_and_review.md](07_reporting_and_review.md)
    gives a disclosure paragraph to adapt.
