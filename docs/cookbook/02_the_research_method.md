# 02 · The research method: the loop this project followed

This chapter takes the project apart into eleven steps. For each step it gives the purpose,
what goes in, what comes out, how it was done in this repository (with evidence), how to do it
by hand, and how it fails. The failures are real ones from this repository, not hypothetical
ones.

The eleven steps:

1. question → 2. paper → 3. facts with pages → 4. under-specification log →
5. pre-declared acceptance criteria → 6. verification run → 7. plan → 8. engine →
9. experiments → 10. honest report → 11. independent review → (back to 4 or 5)

The numbering is the ideal order. The real order was messier: for E00 the verification run came
before the criteria, Phase 3 was left open while Phases 4 and 5 went ahead, and several
criteria were written after exploratory numbers had been seen. Each step says where that
happened. The last section lists every place the loop was not followed.

Dates and sessions are in [01_timeline.md](01_timeline.md); this chapter cites them rather than
repeating them. Terms are defined in the [glossary](../GLOSSARY.md). Decision IDs (D-…) refer to
[docs/DECISIONS.md](../DECISIONS.md); review IDs (R…) to [docs/REVIEW.md](../REVIEW.md).

**Labels.** **[Documented]**: the cited evidence shows it directly. **[Reconstructed]**:
inferred, and the text says from what. **[Recommended]**: advice for doing it by hand, not
something that happened.

## The loop at a glance

| # | Step | Artifact in this repository | When it was made |
|---|---|---|---|
| 1 | Question | [SPEC.md](../../SPEC.md) §1 | planning conversation, before 09-29 |
| 2 | Paper | the 62-page PDF (git-ignored), [references/README.md](../../references/README.md) | planning; re-read 09-29 |
| 3 | Facts with pages | SPEC §3, [CLAUDE.md](../../CLAUDE.md) pitfalls, [PAPER_MAP.md](../PAPER_MAP.md), `Paper: …, p. N` docstrings | SPEC v1.0; Phase 0 |
| 4 | Under-specification log | [DECISIONS.md](../DECISIONS.md) | every phase |
| 5 | Acceptance criteria | SPEC §7, operationalized in DECISIONS, graded in each run's `acceptance.json` | SPEC v1.0; each phase |
| 6 | Verification run | SPEC §7 E00 reference values | planning conversation |
| 7 | Plan | SPEC §11, [PLAN.md](../../PLAN.md) | SPEC v1.0; Phase 0 |
| 8 | Engine | `src/superposition/{data,models,train,metrics}.py` and tests | Phase 1 |
| 9 | Experiments | `experiments/`, `configs/`, `results/`, [cited_runs.yaml](../cited_runs.yaml) | Phases 1, 3–6 |
| 10 | Honest report | [REPRODUCTION.md](../REPRODUCTION.md), [progress reports](../progress/) | every phase |
| 11 | Independent review | [REVIEW.md](../REVIEW.md) | Phase 6 |

**[Documented]** (SPEC §1–§11; git history in 01_timeline.md; COOKBOOK_SPEC §4 for the
planning conversation.)

---

## Step 1. Question

**Purpose.** Fix what would count as an answer before any work starts, and fix what is out of
scope.

**Inputs.** The paper; your reason for doing the project (learning, replication, extension).

**Outputs.** A short mission statement; the list of claims you will test; what you will not do.

**How it was done here.**
- SPEC §1 states four aims: replicate the experiments and report each as PASS, PARTIAL or FAIL;
  build an interactive lab; build interpretability tooling; serve as the owner's learning
  instrument. **[Documented]**
- SPEC §7 then breaks "replicate the paper" into thirteen experiments, E00–E12, each a claim
  with page numbers. E12 is the only extension, and it has no pass/fail. **[Documented]**
- The paper's open questions (pp. 45–46) were put out of scope: a menu in STUDY_GUIDE, "do not
  implement them" (SPEC §7, "Open questions"). **[Documented]**
- SPEC.md was drafted with Claude in a separate planning conversation that is not visible here.
  **[Documented]** (COOKBOOK_SPEC §1.)

**By hand.** **[Recommended]**
1. Write one sentence per claim: "On page P the paper says X; I will check whether X holds when
   I implement the setup from the text alone."
2. Write down what PASS, PARTIAL and FAIL mean before you start. This repository only defined a
   *graded* PARTIAL for a single criterion on 09-30 (D-R05; REPRODUCTION.md preamble).
3. Write down what is out of scope, and keep extensions in separate experiments.

**Failure modes.**
- Scope creep: the lab, notebooks and tooling multiplied the work per experiment (SPEC §7 asks
  each experiment for a module, config, figures, a REPRODUCTION section, a notebook and an app
  page). **[Documented]** For a solo project, cut these first. **[Recommended]**
- Mixing replication with extension. SPEC §2 forbids an extension from changing a replication
  default; E08's bias-init and init-scale variants (D-409, D-R01) stay INFO rows for that
  reason. **[Documented]**

## Step 2. Paper

**Purpose.** A fixed, citable source with stable page numbers.

**Inputs.** The web article or PDF.

**Outputs.** A local copy whose page numbers everyone uses; a references file that says how to
get the same numbering; reading notes.

**How it was done here.**
- Page numbers everywhere refer to a 62-page print of the web article, saved as
  `references/toy_models_of_superposition.pdf`. The file is git-ignored for licensing.
  references/README.md says how to print it to get the same numbering. **[Documented]**
- In the planning conversation the paper was read in full from the PDF's text layer, and pages
  whose equations or figures the text layer misses (pp. 2, 13, 14, 17, 19) were rendered as
  images and read visually. [Planning conversation, per owner]
- Study aids came before any code: a seven-stage reading plan, a hand-built interactive pentagon,
  "explain it back" questions and a Johnson–Lindenstrauss warm-up. [Planning conversation, per
  owner]
- The build session read the paper again. Its PDF reader failed because poppler was missing; it
  installed poppler with Homebrew, extracted pp. 41–62 with `pdftotext`, and mapped section
  headings to pages with a per-page `pdftotext` loop. **[Documented]** [S 35cab3f0 @
  08:06–08:19Z]

**By hand.** Follow [03_from_paper_to_spec.md](03_from_paper_to_spec.md), "Reading strategy".
**[Recommended]**

**Failure modes.**
- Reading figures at low resolution. D-113 claimed E01 represented fewer features than
  pp. 11–12 show. A 300-dpi render of those pages showed no shortfall, and D-114 withdrew the
  premise (phase-1.md open issue 2). **[Documented]** [S 35cab3f0 @ 09:46–09:47Z]
- A different print gives different page numbers, and every docstring, criterion and DECISIONS
  entry would then point to the wrong page. **[Recommended]**: check the page count and a few
  headings against PAPER_MAP.md before trusting your copy.

## Step 3. Facts with pages

**Purpose.** Separate what the paper says from what you choose. Everything later depends on
this line being sharp.

**Inputs.** Your reading notes.

**Outputs.** A facts section (here SPEC §3: notation, data, models, loss, metrics, theory,
optimization notes, initialization, importance curves, plotting conventions), a short pitfalls
list, a page map, and a page reference in every docstring.

**How it was done here.**
- SPEC §3 is titled "Paper facts and conventions (implement exactly)", and every item carries a
  page. **[Documented]**
- Every page reference in SPEC.md was checked against per-page text. [Planning conversation, per
  owner]
- CLAUDE.md's "§3 pitfalls" and PAPER_MAP.md (fully mapped with page ranges) were written in
  Phase 0 (`f31595b`; phase-0.md). **[Documented]**
- Docstrings name section and page, e.g. `Paper: Basic Results, p. 10` (SPEC §1). metrics.py
  alone has 17 such references. **[Documented]** (`grep -c "Paper:"
  src/superposition/metrics.py`, run while writing this chapter.)
- Each graded criterion in a run's `acceptance.json` carries its page; REPRODUCTION.md shows it
  in a "Paper" column. **[Documented]**

**By hand.** **[Recommended]** Keep a fact table with columns: fact · page · your one-line
paraphrase · where it goes in code · the test that checks it. A fact without a page is a
decision, and goes to step 4.

**Failure modes.**
- A reading presented as a fact. The paper's phase-change section writes ReLU(WᵀWx − b) (p. 14);
  the rest of the paper writes +b. SPEC §3 picks +b everywhere and says so. **[Documented]**
- A value that is not in the paper. The intro figure's importance is "not stated"; SPEC §3 uses
  Iᵢ = 0.9ⁱ and marks it as verified by a run, not as a paper fact. **[Documented]**
- A justification from the wrong sentence: E05's stage B was first justified with a p. 23
  sentence about correlated features, but the two sets are independent (R2-03). **[Documented]**

## Step 4. Under-specification log

**Purpose.** Make every choice the paper does not dictate visible, dated and revisable.

**Inputs.** The fact table, and the list of everything the code needs to run.

**Outputs.** DECISIONS.md: dated, append-only entries with the decision, rationale, paper page
and alternatives. A superseded decision gets a new entry pointing back.

**How it was done here.**
- SPEC §2 requires it: a principled default, configurable, recorded with decision, rationale,
  page and alternatives. **[Documented]**
- Typical entries: D-101 (the paper never states its optimizer settings), D-109 (the paper
  canonicalizes 2D solutions without a rule), D-110 (no numeric thresholds in the paper),
  D-113 (no step or restart counts), D-307 (no density for E04 parts b–d), D-408 (E08a: neither
  density nor importance), D-502 (E09: no density or importance). **[Documented]**
- Corrections are new entries, not edits: D-114 corrects D-113; D-603 tightens D-306; D-R01
  supersedes part of D-408. **[Documented]**

**By hand.** **[Recommended]**
1. Walk the pipeline: data → model → initialization → optimizer → batch → steps → restarts →
   evaluation samples → restart selection → thresholds. For each value ask "which page?".
2. No page → write a DECISIONS entry *before* the run that uses it
   ([templates/DECISION_ENTRY.md](templates/DECISION_ENTRY.md)).
3. If you choose by looking at data (a pilot), log the pilot as a run, not as a script.

**Failure modes.**
- Entries that do not exist. The Phase 3 draft's code cited D-301 to D-307, which were not
  written until the closing session the next day (DECISIONS.md, Phase 3 preamble).
  **[Documented]**
- A decision resting on an uncited run. E09's graded densities came from a scratchpad pilot
  (D-502); the owner later had it re-run as a cited profile (D-R03). **[Documented]**
- "Provisional" values that are not revisited in time. D-110 called its graph-edge threshold
  (0.05) provisional. E03's graded factor rule reads factors at an edge threshold of 0.05
  (D-308, D-603), and whether it should be 0.2 is an open owner decision, now that the outcome
  is known (phase-6.md open issue 2). **[Documented]**

## Step 5. Pre-declared acceptance criteria

**Purpose.** Decide what PASS means before the result can be seen, so the result cannot shape
the bar.

**Inputs.** Facts, decisions, and sometimes a verification run (step 6).

**Outputs.** Criteria with a source tag (paper or proposal), a threshold and its reason,
written and committed before the graded run; in this repository, code that writes each
criterion's value, threshold and status to `acceptance.json`.

**How it was done here.**
- SPEC §7 lists criteria per experiment and calls the thresholds "initial proposals" that §2
  governs: any change needs a DECISIONS entry. **[Documented]**
- SPEC.md itself was never edited after `7ac9cef`; every criterion change went through
  DECISIONS.md. **[Documented]** (`git log -- SPEC.md` shows one commit.)
- Words became tests in DECISIONS entries, e.g. D-105 for E01 ("count never decreases,
  tolerance 1" means against the running maximum). **[Documented]**
- D-105 says it fixed E01's meaning before the first E01 run. The transcript shows the E01
  module (with its criteria) written at 08:45Z, the first E01 run started at 08:46Z, and the
  Phase 1 DECISIONS entries appended at 08:49Z. **[Reconstructed]**: from the order of the Write
  and Bash events in the extraction of [S 35cab3f0]; the criteria existed in code before the
  run, and the written entry followed minutes later.
- Phase 3 checked every E02–E04 criterion on hand-built ideal weights (must pass) and broken
  ones (must fail) (phase-3.md, "Tests"). **[Documented]**

**By hand.** **[Recommended]**
1. Tag each criterion **paper** (quote the page) or **proposal** (say who proposed it and why).
2. For each, compute what a model *without* the effect would score. If that baseline passes, the
   criterion is wrong.
3. Write unit tests on hand-built weights: an ideal case passes, a broken case fails.
4. Commit the criteria, then run. Write down what you had already seen.

**Failure modes, all from this repository.**

| Failure | Where | What was done | Final |
|---|---|---|---|
| Criteria written after exploratory numbers were seen | E02 (D-305: 87.5% and 81.9% seen), E03 (D-306: a D* curve seen), E06 (D-406: a prototype) | disclosed in each entry; nothing re-tuned | E02 PASS on the cited seed; E03 PARTIAL; E06 PARTIAL |
| A criterion from the SPEC, not the paper | E08 "more than m/2 computed well" (D-408) | re-derived from p. 37 and footnotes 16–17 (D-R01); the original kept as INFO | 23 features by the new rule (needs > 40): FAIL; the original gives 3; E08 PARTIAL 2/4 |
| Selection by ground truth | E10 graded the oracle-best SAE (D-504) | a rule using only reconstruction and sparsity (D-R04); the oracle kept as an upper bound | PASS 3/3 → PARTIAL 1/3 |
| A grid refined after a failed run | E11 λ grid (D-505) | PARTIAL, plus a held-out seed (D-R05); the seed did not confirm the window (R4-06) | PASS 8/8 → PARTIAL 7/8 |
| A criterion that double-counts | E03 Dᵢ clusters by value alone (R2-01) | counted only inside the matching polytope (D-603) | PASS 4/4 → PARTIAL 3/4 |
| A criterion that matches a pattern, not a mechanism | E09 motif detector (R4-01) | motifs must carry their roles (D-604) | still PASS 3/3 |
| A criterion that passes on one seed only | E02 n = 3 agreement 0.819 vs 0.80 (R1-01) | seeds 1, 2 under a rule fixed first: 0.764, 0.792 (D-605) | open, the owner's decision |

**[Documented]** (the cited DECISIONS and REVIEW entries; rectification.md; phase-6.md.)

The E08 case is the clearest lesson: the planning conversation itself later recommended
re-deriving the criterion from p. 37 and keeping the original FAIL visible. [Planning
conversation, per owner] [03_from_paper_to_spec.md](03_from_paper_to_spec.md) works through it.

## Step 6. Verification run

**Purpose.** Before committing numbers to a spec, check that the setup as you understood it
produces the paper's phenomenon at all, and get sanity values.

**Inputs.** The fact table and a draft setup.

**Outputs.** Confirmed defaults, reference values labeled as sanity checks, a closed-form check.

**How it was done here.**
- E00 was run in JAX, for the ReLU and linear models, before E00's criteria were written:
  n = 5, m = 2, Iᵢ = 0.9ⁱ, S ∈ {0, 0.8, 0.9}, 8 restarts, Adam at 1e-3, batch 1,024, 10,000
  steps, Xavier init with std √(2/(n + m)). The linear optimum Σ_{i≥m} Iᵢ(p/3 − p²/4) was
  derived in closed form and matched those runs. [Planning conversation, per owner]
- Its results are SPEC §7's E00 reference table, with the instruction to use them for sanity
  checks, not as criteria, and notes such as "3 of 8 ReLU restarts settled for only 4
  features" at S = 0.9. D-101 cites the run as the source of the optimizer defaults.
  **[Documented]**
- In the build, the S = 0.9 pentagon's loss (≈ 0.0214) sat about 9% above the reference 0.0196.
  The session first probed this with scratchpad scripts [S 35cab3f0 @ 08:39–08:41Z], then made
  the probes logged INFO rows of the run (D-107). They show 3.7% noise in a 20,000-sample
  estimate, 1 of 200 such estimates at or below 0.0196, and reference *linear* losses below the
  provable optimum. **[Documented]** (REPRODUCTION.md E00; phase-1.md open issue 1.)
- The explanation is labeled a hypothesis, not a finding (R1-05). No criterion depends on it.
  **[Documented]**

**By hand.** **[Recommended]**
1. Write the smallest independent implementation you can (another framework, or a 50-line
   script), with the same settings.
2. Evaluate final losses on ≥ 100,000 samples; record every restart, not just the best.
3. Save the code, seeds and outputs where you can re-run them.
4. Copy the numbers into the spec as **reference values**, explicitly not criteria.

**Failure modes.**
- A reference value that is itself a best-of-8 noisy draw. Selecting the lowest of several noisy
  estimates biases it low (a "winner's curse"); the reference linear losses show it.
  **[Documented]** (REPRODUCTION.md E00, item 4.)
- A verification run that cannot be re-examined. Its code is not in the repository, so the
  reference model could not be re-evaluated on more samples (phase-1.md open issue 1).
  **[Documented]**
- Criteria fitted to one run. E00's criteria were written after the JAX run. They held on the
  quick and full profiles (11/11 each), on CPU and MPS, which is the check that matters.
  **[Documented]** (phase-1.md acceptance table.)

## Step 7. Plan

**Purpose.** An order of work, with exit criteria per phase, that survives interruptions.

**Inputs.** The spec.

**Outputs.** Phases with exit criteria; a checklist; one report per phase.

**How it was done here.**
- SPEC §11 defines Phases 0–6, a closing ritual for each phase (lint and tests, quick profiles,
  figures, docs, commit, report), and rules for parallel work. **[Documented]**
- PLAN.md was written in Phase 0 (`f31595b`), each task mapped to a SPEC section. **[Documented]**
- Phases 0 and 1 shared one prompt: read SPEC.md and the paper, execute Phases 0 and 1, verify
  every exit criterion, commit, write the reports, stop [S 35cab3f0 @ 2026-09-29T08:06Z].
  Phase 2's prompt added re-reading CLAUDE.md, PLAN.md and the last report [S 35cab3f0 @
  12:32Z]. From Phase 3 on, every phase prompt also required a green `make lint test` baseline
  before starting, and stopping if it failed [S 2df87f77 @ 13:23Z; S 92884087 @ 16:29Z;
  S 57d571bd @ 19:58Z; S 1584c849 @ 2026-09-30T20:13Z]. **[Documented]**
- Before the plan was written, the state of the reference implementations and of Claude Code's
  documentation was checked. [Planning conversation, per owner]

**By hand.** **[Recommended]** One phase per sitting. Begin each sitting by reading the
previous phase report and checking its exit criteria are ticked. End it by writing the report
([templates/PHASE_REPORT.md](templates/PHASE_REPORT.md)), even if the phase is unfinished.

**Failure modes.**
- An unfinished phase nobody noticed. Phase 3 stopped at a usage limit; Phase 4 began anyway,
  because its only stop condition was a red `make lint test`, and unregistered code has no
  tests to fail (D-401). Phases 4 and 5 ran beside it until a rectification session closed it.
  **[Documented]** (01_timeline.md §1; phase-3.md.)
- A planned safeguard not used. SPEC §11 pairs every experiment with an independent verifier.
  Phases 3–5 had none (phase-3.md open issue 11); their sessions started no subagents.
  **[Documented]** (transcript extraction index.) The Phase 6 review was the only independent
  check.
- External commits swept up half-done work during phases (01_timeline.md §2). **[Documented]**

## Step 8. Engine

**Purpose.** A small, tested core that every experiment shares.

**Inputs.** SPEC §3 (math) and §6 (interfaces and required tests).

**Outputs.** data, models, train, metrics (and later theory), each with known-answer tests.

**How it was done here.**
- Built sequentially in Phase 1, as SPEC §11 requires. Authoring order in the session: utils,
  config, data, models, metrics, train, theory, then tests, runs, plotting, experiments, CLI
  [S 35cab3f0 @ 08:25–08:46Z]; committed as separate logical commits `ed4b82f`..`162bd1a`.
  **[Documented]**
- 87 unit tests at the end of Phase 1, including every known-answer and property test SPEC §6
  lists, and the closed-form linear optimum as a test of the loss and evaluation code
  (`4558329`). **[Documented]** (phase-1.md.)

**By hand.** Follow [05_engine.md](05_engine.md) and the [rebuild kit](../../rebuild/README.md):
write the tests first, then the bodies. **[Recommended]**

**Failure modes.**
- A statistical test that fails by chance. The density test failed at one seed by 3.45σ; a
  3,000-draw study showed the sampler unbiased; the seed was changed with a comment, and a
  64-seed calibration test was added (phase-1.md, "Test status"). **[Documented]**
- A test bar moved to the observed value. An L1 test's margin went from 0.5 to 0.8 with a
  comment giving the observed ratio (≈ 0.62) [S 35cab3f0 @ 08:33Z]. **[Documented]** It is a
  direction check, so this is defensible, but it is the pattern to watch for. **[Recommended]**

## Step 9. Experiments

**Purpose.** Turn each claim into a run with a manifest and an acceptance record.

**Inputs.** Engine, criteria, configs.

**Outputs.** A module and a config (quick and full profiles) per experiment; run directories
with `manifest.json` (git commit and dirty flag), `config.yaml` and `acceptance.json`; cited
runs pinned in `docs/cited_runs.yaml`.

**How it was done here.**
- 35 run directories exist; every manifest records `dirty: false`. **[Documented]** (manifests;
  01_timeline.md.)
- The first run launched from a clean detached git worktree, with `SUPERPOSITION_RESULTS`
  pointing at the main `results/`, was E08's full profile in Phase 4 [S 92884087 @ 17:58Z].
  From rectification session 1 on, every cited run was launched that way. Three runs that are
  still cited ran from the main tree while it was clean: E01 (`340f825`), E06 and E07
  (`5a825d0`). **[Documented]** (the `command` and `git` fields of those runs' manifests;
  phase-3.md, rectification.md and phase-6.md for the worktree launches.)
- A deviation from SPEC (batch 256 instead of 1,024 for E02–E04 quick profiles) was settled by
  six logged CLI runs, not by scratchpad numbers (D-309). **[Documented]**

**By hand.** **[Recommended]** Check the tree is clean, run, then read the manifest.

```bash
git status --short
uv run superposition list
uv run superposition run intro_pentagon --profile quick --device cpu
```

Expected: `git status --short` prints nothing; the run ends with a line like
`E00 intro_pentagon: PASS (11/11)` and the run directory (the cited run,
`20261001T071529Z-5eeb804c`, passed 11/11 in 68 s on an M2 CPU). Check that the run's
`manifest.json` has `"git": {"commit": …, "dirty": false}`. If the tree is dirty, commit or
stash first; `superposition report --pin` refuses a dirty run.

**Failure modes.**
- Runs outside the provenance system: scratchpad scripts and development runs behind D-305,
  D-306, D-406, D-502 and D-505. Their outputs are not in the repository. **[Documented]**
  (01_timeline.md §3.)
- Dirty-tree runs: the Phase 1 session moved its dirty-tree runs aside before launching the
  cited ones [S 35cab3f0 @ 09:05Z]. **[Documented]**
- An untracked new module did not mark a run dirty until D-606 fixed it (R1-10). **[Documented]**

## Step 10. Honest report

**Purpose.** A record a skeptic can check line by line.

**Inputs.** Cited runs, their `acceptance.json`, DECISIONS.

**Outputs.** REPRODUCTION.md (generated tables plus a hand-written narrative per experiment:
claim with page, figure, status, evidence, discrepancies with hypotheses, run IDs); a phase
report per phase with open issues.

**How it was done here.**
- `superposition report` regenerates the tables from the runs pinned in `docs/cited_runs.yaml`
  (D-202); the narrative is written by hand (REPRODUCTION.md preamble). **[Documented]**
- Mistakes were corrected in the open: D-114 withdrew D-113's premise; the E00 explanation was
  retitled a hypothesis (R1-05); unmeasured runtimes were labeled estimates (D-R06).
  **[Documented]**

**By hand.** Write the narrative only after the runs; give every number a run ID; keep findings
and hypotheses in separate sentences ([07_reporting_and_review.md](07_reporting_and_review.md)).
**[Recommended]**

**Failure modes.**
- Overclaims that survived until review: README's E03 caption (R2-02), "the neuron picture of
  p. 37 reproduces" (R3-01), E06's failure called "what the claim predicts" (R3-03).
  **[Documented]**
- Citing "the latest run", which may be dirty or from another device (phase-1.md open issue 3);
  fixed by pinning (D-202). **[Documented]**
- A draft report with placeholders pushed by an external commit (`7bde9b1`; phase-3.md).
  **[Documented]**

## Step 11. Independent review

**Purpose.** Someone without your history re-derives each claim from the pages and audits the
code and numbers.

**Inputs.** Paper pages, the repository, the cited runs.

**Outputs.** Numbered findings with severity and evidence; a resolution for each: Fixed,
Corrected, or Open.

**How it was done here.**
- Four reviewers with fresh context, one per group (engine and E00–E02, E03–E05, E06–E08,
  E09–E11), started at 20:17–20:18Z on 09-30, and a fifth for E12 at 06:50Z on 10-01. Each
  re-derived its experiments from the pages first, then audited, read-only, without the
  reference implementations (D-607). **[Documented]** [S 1584c849]
- 59 findings, none critical. The review moved E03 to PARTIAL, opened E02's seed question, and
  tightened E09. **[Documented]** (REVIEW.md tables.)
- The reviewers' own scripts are not in the repository; numbers only they produced are labeled
  as theirs (D-607 caveat). **[Documented]**

**By hand.** **[Recommended]** Without a second person, use time as the fresh context: wait a
few days, then re-derive each claim from the pages *before* opening your code, and check every
number in your report against its run. Write findings in the REVIEW.md format.

**Failure modes.**
- Review only at the end. E03's cluster rule (D-306, 09-30) and E09's motif detector stood until
  the review on 10-01. **[Documented]**
- Resolving a finding by tuning. The review showed pentagons separate at an edge threshold of
  0.2; changing the threshold after seeing that would be tuning, so it was left to the owner
  (phase-6.md open issue 2). **[Documented]**

---

## The second pass: owner decisions

The loop is a loop. After Phases 3–5 the planning conversation recommended seven course
corrections [Planning conversation, per owner]. The owner put two of them (closing Phase 3,
splitting the slow test tier) into the brief of rectification session 1
**[Documented]** [S 5ab2965c @ 2026-09-30T07:33Z], and the other five into owner decisions
D-R01–D-R05, recorded before the runs that apply them **[Documented]** (DECISIONS.md,
"Rectification session 2").

| Course correction | Decision | Outcome |
|---|---|---|
| Close Phase 3 before Phase 6 | rectification session 1 | E02–E04 registered, run, reported (phase-3.md) |
| E08 criterion came from the spec | D-R01 | re-derived; FAIL kept visible |
| E05 stage B is a variant | D-R02 | reported, not a discrepancy |
| E11 rests on one λ | D-R05 | PARTIAL; held-out seed run |
| E10 selection by ground truth | D-R04 | rule without ground truth; PARTIAL 1/3 |
| E09 pilot uncited | D-R03 | pilot re-run as a cited profile |
| Slow test tier too long | D-310 | slow < 15 min; opt-in paper tier |

**[Documented]** for the decisions and outcomes.

## Where this project left the loop

| Where | What happened | Evidence |
|---|---|---|
| Step 6 before step 5 | E00's verification run came before its criteria | [Planning conversation, per owner] |
| Step 5 after exploration | E02, E03, E06 criteria written after exploratory numbers; E10's rule after the previous run's summary | D-305, D-306, D-406, D-R04 |
| Step 7 | Phase 3 left open across three sessions | D-401; phase-3.md |
| Step 7 | No per-experiment verifiers in Phases 3–5 | phase-3.md open issue 11 |
| Step 9 | Decisions resting on runs outside provenance | D-305, D-306, D-406, D-502, D-505 |
| Step 5 / 9 | A grid refined after failed development runs | D-505, D-R05 |
| Step 5 | Selection by ground truth | D-504, D-R04 |
| Step 11 | One review, at the very end | D-607 |

**[Documented]** for every row.

What is unknown: how long the planning conversation took, what it tried and discarded, and the
JAX code (01_timeline.md, "Gaps").

For a checklist per step for your next project, see [10_next_project.md](10_next_project.md)
and the [templates](templates/). **[Recommended]**
