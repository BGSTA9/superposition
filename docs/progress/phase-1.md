# Phase 1 report: engine

Date: 2026-09-29 · Commits `529f724`..(this report) · SPEC §11 Phase 1

## What was built

**Library** (`src/superposition/`, every public function documented with its paper page):

| Module | What it does |
|---|---|
| `utils` | `derive_seed(seed, *keys)` independent random streams; `seed_everything`; `resolve_device` (auto = CUDA → MPS → CPU) |
| `config` | typed dataclass configs, strict YAML loading (unknown keys and wrong types are errors), `quick`/`full` profiles, config hash |
| `data` | `IndependentSparse` (per-instance / per-feature p, any value range), `CorrelatedSets`, `AnticorrelatedSets`, `Composite`, `geometric_importance` |
| `models` | `TiedModel` (linear, ReLU output, ReLU hidden; no hidden bias), `AbsModel`, activation registry; per-instance Xavier-normal init that depends only on (seed, instance) |
| `train` | vectorized multi-instance Adam trainer (sum over features → mean over batch → sum over instances), schedules, L1 on h, snapshots, fresh-sample final loss; `evaluate_loss`; `select_best` |
| `metrics` | ‖Wᵢ‖, superposition score, Dᵢ, D*, represented mask, formal superposition test, 2D angles and canonicalization, geometry graph + tegum components, fraction classification, neuron polysemanticity, k-sparse loss terms |
| `theory` | `linear_optimum_loss` (rest of the module is Phase 3) |
| `runs`, `acceptance` | run directories with manifests (git commit + dirty flag, versions, device, seed, timings), parquet tables, checkpoints, `acceptance.json` |
| `plotting/` | shared style; WᵀW + bias heatmaps and ‖Wᵢ‖ bars colored by superposition in the p. 11 grid; canonicalized 2D arrows (p. 2) |
| `experiments/` | registry, `Experiment` type, sparsity-grid trainer, runner; **E00** `intro_pentagon`, **E01** `basic_results` |
| `report`, `cli` | `superposition list · run · figures [--publish] · report · app` |

**Experiments:** E00 and E01 with configs (`configs/*.yaml`, quick + full, measured runtimes
in comments), acceptance criteria, figures (published to `docs/figures/`), and slow
replication tests.

**Notebooks:** `00_orientation`, `01_features_directions_superposition` (pp. 1–8, hand-built
weights, nothing trained), `02_demonstrating_superposition` (runs E00, trains E01's n = 20
model). Learning goals, predict-first prompts, 4–5 exercises each with `<details>` solutions,
a Colab install cell with `<OWNER>` left as a placeholder. Committed without outputs.

**Docs:** REPRODUCTION.md (E00, E01 narrative + generated tables), DECISIONS.md D-101–D-114,
DESIGN.md (layers, data flow, extension points, how to add an experiment), STUDY_GUIDE.md
stops 0–2, CHANGELOG.md, PLAN.md, README gallery, CLAUDE.md additions.

## Test status

| Tier | Command | Result |
|---|---|---|
| Lint | `make lint` (ruff check, ruff format --check, pyright basic) | clean; pyright 0 errors |
| Unit + property (CI tier) | `make test` | **87 passed in ≈ 6 s** |
| Replication + notebooks (slow tier) | `make test-slow` | **5 passed in 12.5 min**: E00 and E01 quick profiles pass on CPU; notebooks 00–02 execute top to bottom (nbmake) |

The unit tier covers every known-answer and property test SPEC §6 requires (pentagon,
antipodal pair, triangle, tetrahedron, orthonormal basis, zero column, regular simplex for
n = 2…13; unit-norm tight frames; rotation invariance), data statistics, model equations and
initialization, instance independence, the closed-form linear optimum, schedules, snapshots,
config strictness, run lifecycle, report generation, the CLI, and the acceptance logic of E00
and E01 on hand-built geometries.

One statistical test was adjusted, openly: the per-instance density check (3 standard errors)
failed at seed 2 by 3.45σ. A 3,000-draw study (mean z −0.03, std 1.001, tail rate 0.33% vs
0.27% expected) showed the sampler is unbiased, so it was a fixed-seed false positive. The test
now uses seed 12 with a comment saying why, and a new 64-seed calibration test checks the
stronger property (z-scores have mean ≈ 0 and std ≈ 1).

## Acceptance table

Cited runs are quick profile, CPU, seed 0, commit `340f825`, clean tree. The full profiles were
also run (on Apple MPS) as robustness checks.

| Exp | Criterion (SPEC §7) | Value | Status |
|---|---|---|---|
| E00 | ReLU S = 0: exactly {0, 1}, \|cos\| < 0.1 | {0, 1}, \|cos\| = 0.0014 | PASS |
| E00 | ReLU S = 0.8: {0, 1, 2, 3} as two antipodal pairs | pairs (0, 2), (1, 3), cos = −1.00 | PASS |
| E00 | ReLU S = 0.9: pentagon; gaps 72° ± 10°; norms > 1; biases < 0 | gaps 69.8–74.0°, norms 1.08–1.11, biases −0.198 to −0.211 | PASS |
| E00 | Linear: exactly {0, 1}, orthogonal, at S = 0, 0.8, 0.9 | \|cos\| ≤ 0.0017 | PASS ×3 |
| E00 | Linear loss within 3% of the closed-form optimum | 0.13%, 0.38%, 1.01% | PASS ×3 |
| E00 | Unrepresented features' bias within 0.02 of (1 − S)/2 (both models) | max deviation 0.0016 / 0.0034 | PASS ×2 |
| **E00** | **overall** (run `20260929T093651Z-5eeb804c`, 62 s) | **11/11** | **PASS** |
| E01 | Linear: exactly top m, score < 0.05, all 7 sparsities (n = 20; n = 80) | max score 0.0001; 0.0011 | PASS ×2 |
| E01 | ReLU matches linear at 1 − S = 1 (set, score, loss within 5%) | loss diff 0.00%; 0.03% | PASS ×2 |
| E01 | ReLU count non-decreasing (tolerance 1) | 5→14 of 20; 20→67 of 80 | PASS ×2 |
| E01 | ReLU count > m at 1 − S = 0.001 | 14 > 5; 67 > 20 | PASS ×2 |
| E01 | Superposition first among less important features | at 0.3: {3–6} vs orthogonal {0–2}; {13–26} vs {0–12} | PASS ×2 |
| E01 | Antipodal pairs at intermediate sparsity (report where) | at 0.3, 0.1, 0.03, 0.01, 0.003 (both sizes) | PASS ×2 |
| **E01** | **overall** (run `20260929T093755Z-2cb0bbc4`, 472 s) | **12/12** | **PASS** |
| E00 full | 32 restarts, MPS (run `20260929T090615Z-eb7b9e3b`, 159 s) | 11/11 | PASS |
| E01 full | 8 restarts, 30k steps, MPS (run `20260929T090856Z-6d287488`, 1,673 s) | 12/12 | PASS |

**Exit criteria:** unit tier green ✔ · E00 passes ✔ · E01 reported ✔ (it also passes).
Quick-profile budget (≤ 10 min on a laptop CPU): E00 62 s, E01 472 s ✔.

## Decisions made

All in `docs/DECISIONS.md`:
- **D-101** optimizer defaults (Adam 1e-3, batch 1024, 10k steps, constant): unstated in the paper.
- **D-102** one uniform draw per entry in `IndependentSparse` (exact same distribution, half
  the RNG cost, which dominated CPU time).
- **D-103** seed streams; linear and ReLU grids share seeds (paired comparison).
- **D-104** cited runs use `--device cpu`; `auto` still prefers accelerators.
- **D-105** exact operationalization of E01's criteria, written before the first E01 run.
- **D-106** E00 details (orthogonality threshold for the linear model, gap computation,
  vacuous bias check, one 100k-sample evaluation for selection and comparison).
- **D-107** E00 pentagon diagnostics as INFO rows (to make the reference-loss investigation
  traceable to a logged run).
- **D-108** `report` and `figures --publish` pulled forward from Phase 2.
- **D-109** canonical orientation for 2D solutions.
- **D-110** provisional metric thresholds (rank tolerance, graph edges, fraction tolerance,
  polysemanticity).
- **D-111** results location, test isolation, configs shipped in the wheel.
- **D-112** ruff E501 off (formatter owns line length).
- **D-113 / D-114** full profiles; D-114 corrects D-113's premise (see open issues).

## Open issues

1. **E00 reference loss at S = 0.9.** Our pentagon's loss is 0.0214 (1M samples) against the
   SPEC reference 0.0196. The logged diagnostics point to noise and selection bias in the
   reference: a 20k-sample estimate has 3.7% noise; all 12 feature arrangements converge to
   0.0214–0.0217; and the reference *linear* losses sit below the provable optimum. It affects
   no criterion. If the reference run's evaluation code is available, re-evaluating its model
   on 1M samples would settle it.
2. **A mistake I made and corrected.** I first believed E01 represented fewer features at
   1 − S = 0.001 than pp. 11–12 show, and wrote D-113 on that premise. A 300-dpi reading of the
   figures shows no shortfall (D-114); the full profile independently shows the counts do not
   depend on training length. Figure readings are approximate by nature.
3. **`superposition report` cites the latest run.** Running notebook 02 (or any E00 run) creates
   a newer E00 run, and the next `report` would cite it, possibly from a dirty tree or another
   device. Phase 2 should let `report` pin cited runs or prefer clean-tree quick runs.
4. **Full-profile runtime on CPU is unmeasured.** E01 full took 28 min on MPS; on a CPU expect
   well over an hour. Documented in the config comment as an expectation, not a measurement.
5. **Owner placeholders.** LICENSE / CITATION.cff author ("superposition contributors", D-006)
   and the notebooks' `<OWNER>` GitHub placeholder.
6. **CI has not run on GitHub.** Nothing was pushed; the workflows were validated locally
   (Phase 0, D-007) and every CI command passes locally.
7. **`superposition app`** prints a Phase 2 message; the lab is Phase 2.

## What the owner should look at and learn from this phase

Start with `docs/figures/intro_pentagon/pentagon_arrows.png` next to the paper's p. 2 figure,
then open `notebooks/02_demonstrating_superposition.ipynb` and do its exercises. Read
`src/superposition/train.py` (≈ 250 lines): it is the whole training story, including why
instances never interact. Then read the E00 section of `docs/REPRODUCTION.md` as a worked
example of the SPEC §2 discipline: a sanity number disagreed, and the investigation separated
"our model is worse" from "the reference estimate is noisy", with evidence logged in the run.

Three questions:

1. In the S = 0.9 pentagon, every bias is ≈ −0.21 and every norm ≈ 1.1. Using only the 1-sparse
   readout ReLU(‖Wᵢ‖² x + bᵢ), explain why ‖Wᵢ‖² + bᵢ ≈ 1, and why positive interference into a
   neighbour survives only for x > ≈ 0.56.
2. E00's best-of-8 linear losses are all slightly *below* the exact optimum. Why is that not a
   bug, and what does it imply about reporting "the best restart's loss" when the loss is
   estimated from samples?
3. In E01 at 1 − S = 0.3, features 0–2 keep dedicated dimensions while 3–6 form antipodal pairs.
   Using the benefit/interference picture of p. 13, argue why the *least* important represented
   features are the first to share a dimension.
