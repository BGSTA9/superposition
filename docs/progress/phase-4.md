# Phase 4 report: dynamics, robustness, neurons

Date: 2026-09-29 · Commits `fd60e7c`..(this report) · SPEC §11 Phase 4, §7 E05–E08, §8 pages
6–8, §9 notebooks 06–08

> **Commit note.** An unfinished draft of this report, with two unfilled placeholders, was
> committed as `7bde9b1` under the owner's git identity from outside this session while I was
> writing it. It was left in place; the commit that follows it holds the finished report.

> **Phase 3 is not finished, and this phase was built beside it.** At the start,
> `docs/progress/phase-3.md` did not exist and PLAN.md's Phase 3 items were unchecked. The
> repository held committed but unintegrated Phase 3 code (`theory.py` additions, E02–E04
> modules and two plotting modules from `4355d5d` and `acec078`) with no configs, registry
> entries, tests, notebooks, app pages, runs or DECISIONS entries (their docstrings cite a
> D-306 that does not exist). `make lint test` passed, so per your stop condition I went ahead.
> SPEC §11 lets experiments from Phase 3 on be built in parallel, each owning its files.
> Phase 4 uses none of the unfinished Phase 3 code and left it untouched and unregistered;
> Home, PAPER_MAP and STUDY_GUIDE mark Phase 3's parts as still to come (D-401).

## What was built

**Experiments** (each: module, `configs/<exp>.yaml` with quick and full profiles, figures,
acceptance criteria, REPRODUCTION.md section, notebook, app page):

- **E05 `learning_dynamics`** (pp. 26–28).
  - (a) The uniform setup (n = 400, m = 30) at 1/(1 − S) = 3, 4 restarts × 80,000 steps, a
    snapshot every 100 steps, and every snapshot's loss on one fixed batch. A jump detector
    requires a feature to leave a level it held for 5,000 steps (|ΔDᵢ| > 0.15 within 2,000
    steps) while the loss falls ≥ 3× faster than on the plateau and by more than its
    fluctuations.
  - (b) Two correlated sets of three features in m = 3, from 0.1 × Xavier init (p. 27: "close
    to zero"), 3 densities × 16 restarts. Every snapshot is classified into a geometric stage
    (small, collapsed, transition, prism, octahedron) and each run gets a trajectory type.
- **E06 `adversarial`** (pp. 29–30), with the library module `adversarial.py`: the analytic
  per-feature L2 attack (ε = 0.1 × mean input norm, direction (WᵀW)ᵢ/‖(WᵀW)ᵢ‖, per-example
  worst case, clean target), the no-superposition reference model, and vulnerability
  (L′ − L)/(L′₀ − L₀). Settings: the uniform setup the p. 29 axes imply (graded), SPEC's
  n = 20, m = 5 example over the same range (graded), and n = 20, m = 5 extended to
  1/(1 − S) = 1,000 (reported).
- **E07 `privileged_basis`** (pp. 31–34): the ReLU hidden layer model (n = 10, m = 5, 0.75ⁱ)
  at 8 sparsities chosen to hit each regime p. 33 names, best of 100 restarts. It also trains a
  hidden-bias variant, never the default, to measure how the model sidesteps its ReLU (p. 34).
- **E08 `abs_computation`** (pp. 34–39): (a) n = 3, m = 6 side neurons; (b) n = 100, m = 40,
  0.8ⁱ at the p. 37 sparsities, with features computed well, neuron counts, primary +
  secondary neurons, a dead-side diagnostic, and a bias-init diagnostic variant.

**Library:** `adversarial.py`; `metrics.per_feature_loss` (an exact conditional estimator that
stays precise at 1 − S = 0.001) and `metrics.normalize_hidden_units` (footnote 15);
`plotting/dynamics.py`, `plotting/adversarial.py`, `plotting/neurons.py` (p. 26, 27, 29, 30,
32–33, 36–37 figures; stack plots; plotly 3D trajectories).

**Lab:** three new pages that read the latest run and show the CLI command when it is missing.
- **Dynamics:** training-step scrubbers over Dᵢ and the loss, a Dᵢ histogram, detected jumps,
  and rotatable 3D trajectories with the stage at the scrubbed step.
- **Neurons:** E07 and E08 W matrices and stack plots per sparsity, neuron counts, the
  sidestepping table, and the E08 side neurons and capacity chart. An E09 tab notes Phase 5.
- **Adversarial:** E06 curves per setting and per-sparsity tables.

Loaders live in `superposition.lab`.

**Notebooks:** `06_learning_dynamics`, `07_adversarial` (an attack computed by hand first),
`08_neurons_and_computation`. Each reuses your latest quick run if there is one, asks for
predictions, and ends with 4 exercises. They also compute the tables behind the reproduction
report's narrative that go beyond the acceptance records: the plateau counts and
collapsed-stage statistics (06) and the extended sweep's biases (07).

**Docs:** DECISIONS D-401–D-409; REPRODUCTION sections E05–E08; STUDY_GUIDE stops 6–8;
GLOSSARY (4 terms); PAPER_MAP, DESIGN, README (status, gallery, lab), CHANGELOG, PLAN,
CLAUDE.md (a new gotcha, below).

## Test status

| Tier | Command | Result |
|---|---|---|
| Lint | `make lint` (ruff check, ruff format --check, pyright basic) | clean; pyright 0 errors |
| Unit + property (CI tier) | `make test` | **143 passed** in 12–18 s (103 before + 40 new) |
| Slow: new replication tests | `pytest -m slow tests/replication/test_{learning_dynamics,adversarial,privileged_basis,abs_computation}.py` | SLOW_RESULT |
| Slow: notebooks 06–08 | `pytest --nbmake notebooks/0{6,7,8}_*.ipynb` with `SUPERPOSITION_RESULTS=results/` | **3 passed** (they reuse the cited quick runs; without runs they train quick profiles, ≈ 25 min) |
| Slow: app launch | `tests/replication/test_lab.py::test_superposition_app_launches` | passed |

New unit tests:
- `test_adversarial.py`: budget; attack directions; reference damage = max Iᵢ ε² exactly; an
  antipodal pair is exactly 2× as vulnerable on the zero input; attacked ≥ clean loss; E06
  criteria on synthetic sweeps.
- `test_learning_dynamics.py`: the jump detector on synthetic runs (a jump with a drop, a jump
  without one, initial organization rejected); the stage classifier on ideal geometries
  (octahedron, antiprism, prism, collapsed, small), including within-set antipodes and rotation
  invariance; trajectory types; the display rotation.
- `test_neurons.py`: footnote-15 normalization preserves the function; per-feature loss is
  exact for an ideal |x| model, equals p/3 for the zero model, and agrees with plain Monte
  Carlo; side roles; dead neurons; primary + secondary; E07 and E08 criteria; the hidden-bias
  variant and clipping.
- `test_app.py`: each new page without runs shows the command, renders against tiny real runs,
  and the Dynamics scrubber moves the 3D view.

I also rendered all three pages headlessly against the real cited runs and moved their
sliders; the numbers match the runs. The in-app browser refused to open `localhost:8501` in
this session, so there is no screenshot of the live pages.

## Acceptance table

Cited runs: quick profile, CPU, seed 0, clean commit `5a825d0`.

| Exp. | Criterion | Value | Status |
|---|---|---|---|
| E05 | (a) a jump (\|ΔDᵢ\| > 0.15 in ≤ 2,000 steps, from a stable level) coinciding with a loss drop | 8 jumps, all with a drop; every restart forms its 30th digon late (29–52k steps), loss −0.043 to −0.055 vs plateau fluctuation ≈ 0.011 | PASS |
| E05 | (b) octahedron final state in ≥ 1 restart | 48/48 runs (16/16 at each density) | PASS |
| E05 | (b) trajectory types reported | all runs pass through *collapsed*; 38/48 sit in the aligned *prism* ≥ 1,000 steps before the antiprism | INFO |
| E06 | uniform n = 400, m = 30: rise > 2× (paper > 3×) | 8.47× (1.03 → 8.70) | PASS |
| E06 | uniform: Spearman(V, 1/D*) ≥ 0.8 | 1.00 | PASS |
| E06 | n = 20, m = 5: rise > 2× | 1.53× (1/D* ≤ 1.82 in the range) | FAIL |
| E06 | n = 20, m = 5: Spearman ≥ 0.8 | 0.78 | FAIL |
| E07 | dense: every neuron monosemantic | 5/5 | PASS |
| E07 | polysemantic count non-decreasing (tol. 1) | 0, 0, 2, 2, 3, 3, 5, 5 | PASS |
| E07 | polysemantic > 0 at sparsest | 5 | PASS |
| E07 | ReLU sidestepping documented | with a hidden bias: 5/5 biases positive, ReLU clips 0.06–0.21% (vs 34.6–53.8% without) | INFO |
| E08 | (a) one positive- and one negative-side neuron per feature | exact (±1 / 1, b = 0) | PASS |
| E08 | (a) per-feature loss < 5% of mean baseline | 0, 0, 0 | PASS |
| E08 | (b) dense: every neuron monosemantic | 39/40 (all 40 at 1 − S = 0.3; full profile: 40/40) | FAIL |
| E08 | (b) sparse: > m/2 = 20 features computed well | 3 at 1 − S = 0.001 (max over sparsities: 19; full profile: 11) | FAIL |

Overall: **E05 PASS (2/2), E06 PARTIAL (2/4), E07 PASS (3/3), E08 PARTIAL (2/4).** Quick-profile
wall-clock on an Apple M2 CPU: 516 s, 401 s, 419 s, 369 s (SPEC §5: ≤ 10 minutes each).

**Phase 4 exit criteria** (SPEC §11: E05–E08 with their notebooks and app pages):

| Criterion | Evidence | Status |
|---|---|---|
| E05–E08 implemented, run, reported | four experiments registered; quick runs from a clean tree, pinned in `docs/cited_runs.yaml`; REPRODUCTION.md sections with honest statuses | done |
| Notebooks 06–08 | committed without outputs; execute top to bottom (nbmake) | done |
| App pages Dynamics, Neurons, Adversarial | AppTest with and without runs; rendered against the cited runs; `superposition app` launch test passes | done |
| Closing steps (§11) | `make lint test` green; quick profiles run; figures published (≤ 150 KB each); REPRODUCTION, CHANGELOG, PLAN updated; committed | done |

## Decisions made

- **D-401** Phase 4 built beside the unfinished Phase 3, without touching or registering its
  code.
- **D-402** E05(a): setup, the fixed-batch loss curve, and what counts as a jump and a
  coinciding loss drop.
- **D-403** E05(b): densities 0.01–0.1, 0.1 × Xavier init (p. 27), 30,000 steps.
- **D-404** E05(b): the rotation-invariant stage tests and trajectory types.
- **D-405** E06: budget, candidates, per-example worst case, clean target, the
  no-superposition reference, rise and correlation.
- **D-406** E06: grade the uniform setup (the p. 29 axes) and SPEC's n = 20, m = 5 over the
  paper's range; report the extended range. It discloses that a prototype had shown the fall at
  extreme sparsity.
- **D-407** E07: grid, 100 restarts at batch 256, the 25% polysemanticity rule, and the
  hidden-bias variant.
- **D-408** E08: setups, the exact per-feature loss, "computed well", neuron counting.
- **D-409** E08(b): what was tried, the dead-side diagnosis, and the bias-init variant.

## Open issues

1. **E08 computation in superposition does not reproduce.** The capacity criterion fails badly
   (3 features computed well at 1 − S = 0.001), and even the dense models compute only 13 (the
   no-superposition optimum is 20). The run's diagnostics point to optimization, not capacity.
   With zero-initialized output biases, a feature side whose initial gain is negative never
   receives gradient when features are sparse: 0 of 89 such sides are learned at 0.001, against
   23 of 111 alive ones. Starting the bias at +0.1 halves the loss and lifts the count to 14,
   still short of 20. Before fixing the profile I also tried 4 restarts, 30,000 steps,
   learning rate 3e-3 and batch 4,096 (8–16 features). The full profile (8 restarts × 30,000
   steps, MPS, clean tree, run `20260929T180008Z-db58449f`, 981 s) fixes the dense criterion
   (40/40 monosemantic; PARTIAL 3/4) but still computes only 11 features well at
   1 − S = 0.001, with the same dead-side pattern (1 of 90 dead sides learned); there the
   bias-init variant helps less (loss −29%, 11 → 13 features). So dead sides explain part of
   the gap, not all of it. **Your decision:** SPEC §2 now allows consulting the reference
   implementations to diagnose this; I did not, so that the choice stays yours. Another next
   step is a bias-init or warm-up variant.
2. **E06 n = 20, m = 5 fails**, because this model barely enters superposition in the paper's
   range (1/D* ≤ 1.8). Grading it was my choice (SPEC names it as an example), made before
   seeing results, so it stays graded. The paper's own setup passes strongly (8.5×, ρ = 1.0).
3. **E05 stage B differs from the paper's picture.** Our sets collapse onto nearly orthogonal
   directions (the linear, PCA-like solution) rather than onto opposite ends of one axis. They
   become opposed only in the prism stage. The paper's density, init and importance for this
   figure are unstated.
4. **E07 at 1 − S = 0.12** shows two monosemantic neurons where p. 33 shows one; only 5–8% of
   restarts reach the best configuration there. The paper's 1,000 restarts (the full profile)
   may change it.
5. **Full profiles of E05–E07 were not run** (their runtimes in the configs are estimates
   scaled from the quick runs). E08's was run on MPS from a clean worktree (issue 1). E07's
   full profile (1,000 restarts × 15 sparsities) needs about 2 hours on a CPU; use a GPU.
6. **The slow tier is now long.** Replication tests for E05–E08 add ≈ 30 minutes, and notebooks
   06–08 add ≈ 25 minutes when no runs exist. Consider a nightly slow workflow only.
7. **Phase 3** (D-401) still needs finishing: registry, configs, tests, notebooks 03–05, app
   pages, runs, its DECISIONS entries and its phase report.
8. **Workflow gotcha, now in CLAUDE.md:** a run records `git.dirty` when it *starts*, so
   editing tracked files while queued runs start makes the later ones uncitable.
9. Carried over: owner placeholders (LICENSE, CITATION.cff, notebooks' `<OWNER>`); CI not yet
   run on GitHub.

## What the owner should look at and learn from this phase

Open `docs/figures/learning_dynamics/energy_levels.png` next to p. 26, then scrub one restart
on the lab's **Dynamics** page: the loss sits flat for tens of thousands of steps with one
dimension badly used, until two features take it over together. Then run notebook 07's first
section: a two-feature antipodal pair is exactly twice as vulnerable as a model without
superposition, and the uniform sweep turns that into 8.7× at 1/(1 − S) = 10 (compare the p. 29
curve and the p. 30 scatter). Finally, read E08's side diagnostic in notebook 08. It is a
small, concrete example of how an optimization detail (a zero bias under a ReLU) can look like
a limit of what a representation can do, and why the repo reports E08 as PARTIAL instead of
tuning the setup until the criterion passes.

Three questions:

1. In E05(a), why does the late jump always involve *two* features at once? The loss drops by
   about 0.05, much less than twice p/3 − p²/4 ≈ 0.083 (the loss of an unlearned feature at
   1/(1 − S) = 3). Why is it smaller than storing two features from scratch would save?
2. E06 measures vulnerability relative to a model without superposition at the *same*
   sparsity and budget. What would go wrong if you compared raw damages L′ − L across
   sparsities instead? (Footnote 17 has a similar warning about losses.)
3. In E07, giving the hidden layer a bias makes the model *better*, while E08's model needs its
   ReLU. What property of the task decides whether a hidden ReLU is used or sidestepped?
