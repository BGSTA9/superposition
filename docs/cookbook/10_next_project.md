# 10 · Your next project, by hand

This chapter is a runbook for your next project, done with no AI writing the code or the
documents. It has one checklist per stage of the loop this repository followed (see
[02_the_research_method.md](02_the_research_method.md) for the loop itself, with its evidence),
then a dry run on three candidate projects sized for one person.

**Labels.** A sentence about how this repository did something ends with **[Documented]** and
cites its evidence. The checklists, the candidates, their criteria and their compute numbers are
advice and end with **[Recommended]**. Draft acceptance criteria are marked *proposal*, and
compute numbers *estimate*. Terms in *italics* are in [the glossary](../GLOSSARY.md).

**How to use it.** Copy the checklists into your project's lab log
([LAB_LOG.md](templates/LAB_LOG.md)) and fill every blank (`____`). A stage is done when every
box is ticked and its output file exists. Do not start a stage while the previous one is open:
this repository started Phase 4 while Phase 3 was unfinished, and closing it took a separate
session the next day ([01_timeline.md](01_timeline.md), "1. Phase 3 was left unfinished across
three sessions"). **[Documented]**

| Stage | Output | Template |
|---|---|---|
| 1 Question | one paragraph in SPEC §1 | [SPEC_TEMPLATE](templates/SPEC_TEMPLATE.md) |
| 2 Paper | page map, reading notes | lab log |
| 3 Facts with pages | SPEC §3 | SPEC_TEMPLATE |
| 4 Under-specification log | DECISIONS.md entries | [DECISION_ENTRY](templates/DECISION_ENTRY.md) |
| 5 Pre-declared criteria | SPEC §7, one card per experiment | [EXPERIMENT_CARD](templates/EXPERIMENT_CARD.md) |
| 6 Verification run | reference values, a closed form | lab log |
| 7 Plan | PLAN.md with phases and exit criteria | SPEC_TEMPLATE §11 |
| 8 Engine | tested core library | [PHASE_REPORT](templates/PHASE_REPORT.md) |
| 9 Experiments | cited runs | EXPERIMENT_CARD, lab log |
| 10 Honest report | REPRODUCTION.md entries | [REPRODUCTION_ENTRY](templates/REPRODUCTION_ENTRY.md) |
| 11 Independent review | REVIEW.md | PHASE_REPORT |

## Stage 1 · Question

**Purpose.** Fix what you are trying to find out, in one sentence you could be wrong about.

**Here.** SPEC §1 states the mission: replicate the paper, build a lab, build interpretability
tooling, and serve as the owner's learning instrument. **[Documented]**

Checklist **[Recommended]**:
- [ ] The question, in one sentence: ____
- [ ] It is a replication / an extension / new work (circle one). If an extension, the
      replication it extends: ____
- [ ] What result would make you say "no": ____
- [ ] Literature check done before anything else: search terms ____, papers found ____,
      what they already answer ____. Record it in the lab log with the date.
- [ ] Scope fits one person: the largest run you expect, in hours on your machine (an
      estimate, reasoned as in "Compute baseline" below): ____

## Stage 2 · Paper

**Purpose.** Read the source completely before building, with every page addressable.

**Here.** The paper was read in full from the PDF's text layer; pages whose equations or figures
the text layer misses (pp. 2, 13, 14, 17, 19) were rendered as images [Planning conversation,
per owner]. In session 35cab3f0 the PDF reader first failed because poppler was missing, and the
session installed it [S 35cab3f0 @ 2026-09-29T08:06–08:07Z]. The section headings of pp. 38–62
were mapped to pages with a `pdftotext` loop [S 35cab3f0 @ 2026-09-29T08:19Z]. **[Documented]**

Checklist **[Recommended]**:
- [ ] The exact PDF you cite, saved locally and never committed: ____ (pages: ____)
- [ ] Text layer extracted per page; pages with missing equations or figures listed: ____
- [ ] Those pages rendered at 300 dpi and read (skill 16 of
      [09_skills_and_drills.md](09_skills_and_drills.md))
- [ ] A page map: section → pages → what you will build (like [PAPER_MAP](../PAPER_MAP.md))
- [ ] Study aids before code: one hand-built example you can compute on paper ____, three
      "explain it back" questions ____

## Stage 3 · Facts with pages

**Purpose.** Separate what the paper states from what you will choose.

**Here.** SPEC §3, "Paper facts and conventions (implement exactly)", lists notation, data,
models, loss, metrics and initialization, each with a page; CLAUDE.md repeats the pitfalls.
Every page reference in SPEC.md was checked against per-page text [Planning conversation, per
owner]. **[Documented]**

Checklist **[Recommended]**:
- [ ] Notation table: symbol, meaning, name in code
- [ ] Each fact with its page; each fact checked against that page's text: ____ of ____ checked
- [ ] Sign conventions and reductions written out (here: `+b` everywhere although p. 14 writes
      −b; loss summed over features, averaged over the batch, summed over instances)
- [ ] The pitfalls list: things a careful reader still gets wrong ____

## Stage 4 · Under-specification log

**Purpose.** Every choice the paper leaves open becomes a dated, reasoned entry before it
affects a result.

**Here.** DECISIONS.md: the paper never states its optimizer settings (D-101); p. 14 says
"sparsity from 1.0 to 0.01" while the axes read density, and the figures decided (D-303).
Entries are append-only; a superseded decision gets a new entry pointing back (D-114 corrects
D-113). **[Documented]**

Checklist **[Recommended]**:
- [ ] Every "not stated" in your Stage 3 notes has an entry
      ([DECISION_ENTRY](templates/DECISION_ENTRY.md)): ____ entries
- [ ] Each entry has: decision, rationale, page, alternatives, and a disclosure of anything you
      had already seen
- [ ] Defaults are configurable; variants live in their own profile, never in the default

## Stage 5 · Pre-declared acceptance criteria

**Purpose.** Decide what counts as success before seeing results, and say where each threshold
comes from: the paper, or your proposal.

**Here.** SPEC §7's thresholds were "initial proposals" governed by SPEC §2. D-105 fixed E01's
exact meaning before its first run. Three criteria were written after exploratory numbers had
been seen, and say so (D-305, D-306, D-406). E08's "> m/2" came from the spec, not the paper,
and was re-derived from p. 37 with the original result kept visible (D-R01). **[Documented]**

Checklist **[Recommended]**:
- [ ] For each claim: the paper's words and page ____, your operational test ____, its source
      (paper / proposal) ____
- [ ] Each threshold traces to a page, a known answer, or another criterion's threshold
- [ ] The criteria are committed (hash ____) before the first graded run
- [ ] What you have already seen is disclosed in the DECISIONS entry
- [ ] Measurement-validity checks (convergence, evaluation noise, held-out seeds) are separate
      from the scientific answer, which is reported whatever it is

## Stage 6 · Verification run

**Purpose.** Before writing numbers into a spec, run the core experiment once, independently,
and derive one quantity in closed form.

**Here.** The intro experiment was run in JAX before E00's criteria existed, and the linear
optimum Σ_{i≥m} Iᵢ(p/3 − p²/4) was derived and matched [Planning conversation, per owner]. SPEC
§7 lists those numbers as reference values, "not as acceptance criteria". The S = 0.9 reference
(0.0196) later turned out to sit below this repository's converged 0.0214, with selection bias
as the leading hypothesis (REPRODUCTION, E00). **[Documented]**

Checklist **[Recommended]**:
- [ ] A small independent implementation (another framework, or a 50-line script) ran: ____
- [ ] One closed-form or exact value derived by hand: ____ = ____; matched by the run: yes / no
- [ ] Reference values recorded with their evaluation sample size and number of restarts
- [ ] Noted: reference values are sanity checks; criteria come from Stage 5

## Stage 7 · Plan

**Purpose.** Phases with exit criteria, in an order that builds shared code first.

**Here.** SPEC §11: phases 0–6, a "Closing every phase" list, and "State across context resets"
(PLAN.md checkboxes, `docs/progress/`, the git log and the tests are the source of truth).
**[Documented]**

Checklist **[Recommended]**:
- [ ] Phases, each with exit criteria that can be run: ____
- [ ] The engine phase comes first and alone; experiments only after it
- [ ] A rule: the previous phase report exists and its exit criteria are ticked before the next
      phase starts
- [ ] Test tiers with time budgets (here: fast unit tier; `slow` under 15 minutes; an opt-in
      `paper` tier, D-310)
- [ ] Where runs go, and how a run becomes citable (Stage 9)

## Stage 8 · Engine

**Purpose.** A small, tested core: data, models, training, metrics.

**Here.** Session 35cab3f0 wrote the core in the order utils, config, data, models, metrics,
train, theory, then the tests, and committed it as separate logical commits (`ed4b82f`
config/seeding, `36bacc6` data, `e1abf9b` models, `4558329` train, `e80cdf2` metrics)
[S 35cab3f0 @ 2026-09-29T08:25–09:05Z]. 87 unit tests passed at the end of Phase 1 (phase-1.md).
**[Documented]**

Checklist **[Recommended]**:
- [ ] Practice first: the rebuild kit's M1–M4 ([../../rebuild/README.md](../../rebuild/README.md))
- [ ] Known-answer tests written before each function: ____
- [ ] Property tests for invariants (rotation, permutation, scale): ____
- [ ] Shapes named and checked; a leading instance dimension if you train many models
- [ ] Seeding per instance; data and evaluation on separate streams
- [ ] `make lint test` (or your equivalent) green; one commit per logical change
- [ ] Phase report written ([PHASE_REPORT](templates/PHASE_REPORT.md))

## Stage 9 · Experiments

**Purpose.** Each experiment runs from a config, writes a run record, and is graded by the
Stage 5 criteria.

**Here.** Each experiment has a module, a config with `quick` and `full` profiles, an
`acceptance.json` per run, and a pinned citation (D-202). Cited runs come from a clean tree;
later ones from a clean detached worktree (D-606). **[Documented]**

Checklist **[Recommended]**:
- [ ] Experiment card filled before the run ([EXPERIMENT_CARD](templates/EXPERIMENT_CARD.md))
- [ ] Commit → run → narrative → pin, in that order; the run's dirty flag is false
- [ ] Every exploratory run is logged too, even if it will never be cited (lab log: run id,
      commit, dirty)
- [ ] Restarts and evaluation samples sized for the loss gaps you must resolve (skill 9)
- [ ] A held-out seed for any result that passes near its threshold (here: D-605, D-R05)
- [ ] If a result fails: implementation → more restarts or steps → document → PARTIAL/FAIL
      (SPEC §2). Never change the criterion without a DECISIONS entry.

## Stage 10 · Honest report

**Purpose.** Per experiment: claim and page, figure, status, evidence, discrepancies with
hypotheses, run IDs.

**Here.** REPRODUCTION.md has exactly these parts; its tables are regenerated from cited runs
and the narrative is written by hand (SPEC §10). Statuses changed after first being reported:
E10 and E11 became PARTIAL through owner decisions, E03 through review finding R2-01
([01_timeline.md](01_timeline.md), "Statuses over time"). **[Documented]**

Checklist **[Recommended]**:
- [ ] One [REPRODUCTION_ENTRY](templates/REPRODUCTION_ENTRY.md) per experiment
- [ ] Every number appears in a run record you can name
- [ ] Each hypothesis is labeled as one, with what would test it
- [ ] Failures and changed criteria stay visible (INFO rows, DECISIONS links)
- [ ] Results written in your own words; publish only what you can explain

## Stage 11 · Independent review, by hand

**Purpose.** Someone who did not build it re-derives each claim from the paper, then audits the
code and checks every number.

**Here.** Five reviewer agents with fresh context, one per experiment group, re-derived from the
cited pages first, audited the code, checked numbers against the cited runs, and returned 59
numbered findings (D-607, REVIEW.md). **[Documented]**

Checklist, alone **[Recommended]**:
- [ ] Reviewer chosen: a fresh reader (a colleague, a study partner) ____, or yourself after a
      delay of ____ days with the code closed
- [ ] The reviewer first answers, from the paper only: what does page ____ claim, and what
      would a faithful test be?
- [ ] Then compares that with your criteria and code
- [ ] Every number in the report checked against its run record: ____ of ____
- [ ] Findings numbered with severity (major / minor / note) and evidence, in REVIEW.md
- [ ] Each finding resolved as fixed (re-run, cited), corrected (text), or open

## Compute baseline for estimates

Every runtime below is an **estimate**, reasoned from this repository's measured runs on an Apple
M2 CPU. Only the matrix products are counted; backward is taken as twice the forward cost.
**[Recommended]**

| Measured run | Models × size | Batch × steps | FLOP (estimate) | Wall-clock | FLOP/s |
|---|---|---|---|---|---|
| E03 quick `20261001T071640Z-2cb2dc61` | 40 × (n 400, m 30) | 256 × 10,000 | 1.47 × 10¹³ | 346 s | ≈ 4.3 × 10¹⁰ |
| E03 batch1024 `20260930T082510Z-c635c83a` | 40 × (n 400, m 30) | 1,024 × 10,000 | 5.9 × 10¹³ | 1,472 s | ≈ 4.0 × 10¹⁰ |
| E08 quick `20260930T191445Z-49fd30ed` | 40 × (n 100, m 40), two matrices | 1,024 × 10,000 | 2.0 × 10¹³ | 493 s | ≈ 4.0 × 10¹⁰ |

Arithmetic for the first row: two products per step (Wx and Wᵀh), 2 × 256 × 30 × 400 =
6.1 M multiply-adds = 12.3 MFLOP forward; × 3 for forward plus backward = 36.9 MFLOP per model per
step; × 40 models × 10,000 steps = 1.47 × 10¹³ FLOP. The model counts come from the configs
(E03: 20 sparsities × 2 restarts; E08: 28 sweep models plus 4 and 8 in its two diagnostic
variants, assumed to train for the same steps). The wall-clock times are each run's
`manifest.json` field `wall_clock_s`. **[Recommended]**

Plan with **about 4 × 10¹⁰ FLOP/s** for small dense models on this CPU. The wall-clock includes
data generation, evaluation and figures, so this is an effective rate, not a peak. Very small
models (E00, E09) are dominated by per-step overhead instead: scale them by step count, not by
FLOP. **[Recommended]**

## Dry run: three candidate projects

**No novelty is claimed for any candidate.** Each may already be answered in the literature.
Your first step on any of them is a literature check (Stage 1); the names below are things to
search, not summaries vouched for here. If the check finds the question answered, the project
can still be worth doing as a replication, labeled as one. **[Recommended]**

### Candidate A · Reverse-engineering a small network trained on modular addition

- **Question.** What features does a one-hidden-layer network use to compute (a + b) mod P, and
  does it store more of them than it has hidden dimensions (superposition in the sense of p. 8)?
- **Literature check first.** Search "grokking", "modular addition mechanistic
  interpretability", "Fourier features modular arithmetic". Titles to look up and read
  yourself: Power et al. 2022, *Grokking: Generalization Beyond Overfitting on Small Algorithmic
  Datasets*; Nanda et al. 2023, *Progress measures for grokking via mechanistic
  interpretability*; Zhong et al. 2023, *The Clock and the Pizza*; Gromov 2023, *Grokking
  modular arithmetic*. Verify each citation; narrow the question to what they leave open.
- **Paper pages it builds on.** p. 8 (formal superposition: WᵀW not invertible), pp. 31–34
  (privileged basis, polysemantic neurons), pp. 34–39 (computation in superposition; footnote
  15's rescaling), pp. 45–46 (open questions; p. 46 asks to what extent networks can "do useful
  computation" on features in superposition).
- **Smallest useful experiment.** P = 59; input the one-hot a and b (118 dimensions); a ReLU
  hidden layer of width m; P output logits; cross-entropy; full-batch training on 40% of the
  P² = 3,481 pairs (about 1,400), held-out accuracy on the rest. Eight seeds as instances (skill
  1). Widths m ∈ {16, 32, 64, 128}. Then take the discrete Fourier transform over residues of
  each neuron's input weights from a, and of its output weights.
- **Draft criteria (proposal; fix the numbers, with reasons, before the first graded run).**
  - A1, the network works: held-out accuracy ≥ 99% on every seed at the widths analyzed. If
    not, report that and make no mechanism claim.
  - A2, structure measure, defined before looking: the share of each live neuron's input-weight
    power at its top frequency (k and P − k merged); report the distribution. Threshold for
    "frequency-specific": ____ (proposal).
  - A3, rebuild: keep only the frequencies named in A2 (project the weights onto them); held-out
    accuracy stays ≥ 95% (proposal).
  - A4, superposition count, reported not graded: 2 × (frequencies kept) against m per width.
- **Compute (estimate).** Dense forward per example: 118 × 128 + 128 × 59 = 22,656
  multiply-adds = 45 kFLOP; × 3 = 136 kFLOP; × 1,400 examples = 0.19 GFLOP per model per step;
  × 8 seeds = 1.5 GFLOP per step. The number of steps is unknown until the literature check;
  budget 50,000: 7.6 × 10¹³ FLOP ÷ 4 × 10¹⁰ FLOP/s ≈ 1,900 s ≈ 32 min at m = 128. All four widths
  cost (16 + 32 + 64 + 128)/128 = 1.9× that: about 1 hour.
- **Risks.** No ground truth: unlike the toy models, the features are inferred, so the paper's
  Dᵢ and WᵀW tools apply only to directions you define. Training outcome depends on weight decay,
  data fraction and step count, which the literature will set; a failure to generalize is an
  optimization result, not a finding about computation. Cross-entropy changes the loss geometry
  from the paper's squared error. Choosing the width after seeing the Fourier picture is the E11
  trap (D-505, D-R05): fix the widths and seeds first. **[Recommended]**

### Candidate B · Computation in superposition beyond |x|

- **Question.** [STUDY_GUIDE](../STUDY_GUIDE.md) extension project 12: if E08's |x| target is replaced by ReLU(x)
  (or later x²), does the asymmetric-superposition-with-inhibition motif of p. 39 reappear?
- **Literature check first.** Search "computation in superposition", and papers citing *Toy
  Models of Superposition* that study nonlinear targets.
- **Paper pages.** pp. 34–39 (the absolute-value model, side neurons, the motif), footnote 15,
  pp. 45–46.
- **Smallest useful experiment.** E09's design (D-502) with the target changed: n ∈ {4, 5, 6}
  features, 1 − S ∈ {0.3, 0.2} graded and {0.5, 0.1} reported, 20 restarts, 10,000 steps, batch
  1,024. ReLU(x) needs one neuron per feature without superposition, so use m = n − 2 (E09 uses
  2n − 2 for |x|, two fewer than it needs). Build it as a new model class and experiment; the
  replication's `AbsModel` stays untouched (SPEC §2).
- **Draft criteria (proposal).**
  - B1, sanity: at 1 − S = 1 with m = n, every per-feature loss is below 5% of the
    mean-prediction loss (E08(a)'s bar).
  - B2: at the graded densities, more than m features usefully computed, with "usefully"
    re-derived for this target as D-R01 did for |x|.
  - B3: a role-consistent motif (D-604's rule, adapted) in at least one model; frequency reported
    per configuration, with the share of roles assigned at initialization as a baseline.
- **Compute (estimate).** E09's quick profile trained 3 sizes × 4 densities × 20 restarts =
  240 tiny models in 255 s (`20261001T072551Z-3fdd1d10`). These models are overhead-bound, so the
  same design with a new target costs about the same: about 4–5 minutes per target, about
  15 minutes for three targets, plus a pilot like D-502's (200 s measured for E09's).
- **Risks.** The role taxonomy is built for |x| (D-501) and must be redesigned and fixed before
  the graded run. Weight-pattern detectors over-count (R4-01, R4-02). Densities chosen from a
  pilot must be disclosed (D-502). **[Recommended]**

### Candidate C · Does the 1/2 plateau move with scale?

- **Question.** STUDY_GUIDE extension project 6 (p. 46 asks whether superposition goes away
  with enough scale): at fixed n/m, does the D\* = 1/2 plateau of p. 17 move as m grows?
- **Literature check first.** Search "superposition scaling", "feature packing", and phase
  transitions in compressed sensing (STUDY_GUIDE project 10 links the two).
- **Paper pages.** pp. 16–21 (uniform superposition, D\*, Dᵢ), pp. 45–46.
- **Smallest useful experiment.** E03's quick design (20 points of 1/(1 − S) over [1, 10], 2
  restarts, 10,000 steps, batch 256) at m = 60, n = 800, compared with E03's cited m = 30 run
  (plateau 2.34–3.79). Then m = 120, n = 1,600 if the first is informative. A new profile, never a
  change to E03's.
- **Draft criteria (proposal).** Grade the measurement, report the answer.
  - C1, convergence: on 4 of the 20 points, D\* changes by less than 0.02 between 10,000 and
    20,000 steps. D-309 found E03's transition region not converged at 10,000 steps, so this
    must pass before any "moved" is claimed.
  - C2, plateau found with D-306's rule unchanged (0.5 ± 0.03, span ≥ 1.25×).
  - Reported, not graded: the plateau's endpoints at each m against m = 30's.
- **Compute (estimate).** Cost per model scales with n·m. m = 60, n = 800 is 4× E03's: 4 × 346 s
  ≈ 23 min (the batch1024 run, also 4× the FLOP, took 1,472 s, which agrees). m = 120, n = 1,600 is
  16×: ≈ 1.5 hours. C1's longer runs on 4 points add about 4/20 of each run.
- **Risks.** An unconverged transition can look like a moved plateau. Restart choice at some
  sparsities is near a coin flip (R2-10); use more evaluation samples, as E04 does (D-307). Larger
  models may need more steps, which changes the comparison. **[Recommended]**

### Choosing

Pick the candidate whose Stage 1 literature check leaves the clearest open question, then run
Stages 2–11 on it with the checklists above. If you want practice first, Candidate C reuses this
repository's code most directly; Candidate A matches your interest in number theory but has no
ground truth, which makes Stages 5 and 11 the hard part. **[Recommended]**
