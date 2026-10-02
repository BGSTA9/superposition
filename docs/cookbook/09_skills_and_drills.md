# 09 · Skills and drills

This chapter lists the skills you need to run a project like this one alone, and gives drills
to build each of them. Every skill names where it appears in this repository: the files,
functions, tests and decisions to study. Every drill has a concrete task and a self-check: an
exact value or a test that must pass, taken from the repository's own tests or from known
answers.

**How to read the labels.** A bullet under "Where in this repo" describes the repository or
its history and ends with **[Documented]**, backed by the file, test or decision it names. A
drill is advice for practice and ends with **[Recommended]**. Terms in *italics* are defined in
[the glossary](../GLOSSARY.md).

**Two kinds of test command.** Commands under `tests/unit/` run the repository's own
implementation, which is the answer key: they pass now, and they show you the check you are
aiming at. Commands under `rebuild/` run *your* implementation from the rebuild kit
([`../../rebuild/README.md`](../../rebuild/README.md)): as shipped, they fail with
`NotImplementedError` until you write the module.

| # | Skill | Rebuild milestone |
|---|---|---|
| 1 | Batched multi-instance training | M2, M3 |
| 2 | Per-instance seeding and initialization | M2 |
| 3 | Metrics with known answers and property tests | M4 |
| 4 | Statistical tests: tolerances and calibration | M1 |
| 5 | Closed-form and exact-expectation checks | M3, M5 |
| 6 | Criterion design and operationalization | M5, M6 |
| 7 | Reading weight plots | M5, M6 |
| 8 | Canonicalizing 2D solutions | M4 |
| 9 | Restarts and best-of selection | M3, M5 |
| 10 | Phase diagrams: family theory against the vote | — |
| 11 | Analytic worst cases (adversarial attacks) | — |
| 12 | Path gains and ablations | — |
| 13 | Rebuild tests | — |
| 14 | SAE training and ground-truth evaluation | — |
| 15 | Provenance discipline | M5 |
| 16 | Reading a paper figure at 300 dpi | M6 |
| 17 | Writing honest reports | M5 |

## 1. Batched multi-instance training

**What it is.** Every parameter carries a leading *instance* dimension, so one model object
holds many independent models: restarts, sparsities or importance settings. One optimizer step
trains them all. Shapes are named, e.g. `W: (inst, hidden, feat)`, and products are written as
`einops.einsum` over named dimensions, which makes a shape mistake fail loudly.

**Where in this repo.**
- `TiedModel.hidden`, `readout` and `gram` in [`models.py`](../../src/superposition/models.py)
  are one `einops.einsum` each, e.g. `"inst hidden feat, batch inst feat -> batch inst hidden"`.
  **[Documented]**
- The loss order is fixed by SPEC §3 ("Loss (p. 10)"): sum over features, mean over the batch,
  sum over instances for `backward()`. The docstring of
  [`train.py`](../../src/superposition/train.py) explains why: summing keeps each instance's
  gradient what it would be alone, and Adam's update is elementwise. **[Documented]**
- `test_importance_weighted_loss_reduction_order` checks the order against a Python loop;
  `test_instances_are_independent` perturbs instance 1 and checks that instance 0's gradient is
  bit-identical. **[Documented]**

**Drill 1.1.** Implement `importance_weighted_loss` and `train` in the rebuild kit from their
docstrings (milestone M3). **[Recommended]**
- Self-check: all 13 tests pass.

```bash
uv run pytest -c rebuild/pytest.ini rebuild/tests/test_train.py -q
```

**Drill 1.2.** On paper, write WᵀW for a `(inst, hidden, feat)` tensor as an einsum pattern,
then check your pattern against the repository's test of the same property. **[Recommended]**
- Self-check: your pattern equals `"inst hidden f1, inst hidden f2 -> inst f1 f2"`, the one
  `tests/unit/test_models.py::test_gram_is_wt_w` checks against `W[k].T @ W[k]`.

**Drill 1.3.** Explain in two sentences why one Adam step on the summed loss of k instances
equals k separate Adam steps, and what would break if each instance's loss were divided by the
total over all instances (a reduction that mixes instances). **[Recommended]**
- Self-check: your answer must mention that each instance's parameters appear only in its own
  loss term. Then run the independence test:

```bash
uv run pytest tests/unit/test_train.py::test_instances_are_independent -q
```

## 2. Per-instance seeding and initialization

**What it is.** Each instance's starting weights depend only on (global seed, instance index).
Adding instances never changes existing ones, and named random streams keep initialization,
training data and evaluation data apart.

**Where in this repo.**
- D-103 ([DECISIONS](../DECISIONS.md)): one global seed; `derive_seed(seed, *keys)` gives the
  streams `INIT_STREAM`, `DATA_STREAM`, `EVAL_STREAM`, `PROBE_STREAM`. The linear and ReLU grids
  of E00 and E01 share a seed, so each ReLU instance starts from its linear twin's W (a paired
  comparison). **[Documented]**
- `xavier_normal_per_instance` in [`models.py`](../../src/superposition/models.py) draws each
  instance with its own generator at std √(2/(n + m)); its docstring explains why
  `nn.init.xavier_normal_` on a 3-D tensor is wrong. **[Documented]**
- D-R01 records that ARENA's absolute-value model initializes the whole 3-D tensor at once, which
  gave a weight scale 5.3× smaller than SPEC §3's. **[Documented]**

**Drill 2.1.** Predict the std that `nn.init.xavier_normal_` gives a `(200, 10, 40)` tensor.
PyTorch treats dimension 1 as fan-in and dimension 0 as fan-out, and multiplies both by the
product of the remaining dimensions. Then measure it. **[Recommended]**
- Self-check: about √(2/(400 + 8,000)) ≈ 0.0154, not the intended √(2/(10 + 40)) = 0.2. The
  printed value varies in the third decimal because the draw is unseeded.

```bash
uv run python -c "import torch; t = torch.empty(200, 10, 40); torch.nn.init.xavier_normal_(t); print(round(t.std().item(), 4))"
```

**Drill 2.2.** Implement `xavier_normal_per_instance` and `TiedModel.__init__` in the rebuild
kit (milestone M2). **[Recommended]**
- Self-check: `test_xavier_scale_and_independence_from_instance_count` passes; it checks the std
  within 2% and that 3 instances equal the first 3 of 9.

```bash
uv run pytest -c rebuild/pytest.ini rebuild/tests/test_models.py -q
```

**Drill 2.3.** Read D-103's rejected alternative (`seed + k`) and write one sentence on how two
runs with seeds 0 and 1 could then share an instance's weights. **[Recommended]**
- Self-check: run 0's instance k + 1 and run 1's instance k get the same seed, k + 1.

## 3. Metrics with known answers and property tests

**What it is.** A *metric* is a pure function of the weights. You trust it only after it gives
the right answer on geometries whose answer you know by hand, and after a *property test*
(random inputs from `hypothesis`) shows an invariant holds everywhere.

**Where in this repo.**
- [`metrics.py`](../../src/superposition/metrics.py): `feature_norms`, `superposition_score`,
  `feature_dimensionality` (*Dᵢ*), `dims_per_feature` (*D\**), `is_superposition`. **[Documented]**
- [`tests/unit/test_metrics.py`](../../tests/unit/test_metrics.py) implements SPEC §6's table:
  pentagon Dᵢ = 2/5 and D* = 0.4, antipodal pair 1/2, triangle 2/3, tetrahedron 3/4,
  orthonormal basis 1, zero column 0, regular simplex (n − 1)/n; plus two hypothesis tests
  (unit-norm tight frames give Dᵢ = m/n and Σᵢ Dᵢ = m; rotations leave every metric unchanged).
  **[Documented]**
- The tight-frame strategy hit a hypothesis error while it was being written
  [S 35cab3f0 @ 2026-09-29T08:31Z]. **[Documented]**

**Drill 3.1.** By hand, compute Dᵢ for a regular pentagon of unit vectors: the denominator is
Σⱼ (Ŵᵢ·Wⱼ)² = 1 + 2cos²72° + 2cos²144°. **[Recommended]**
- Self-check: the denominator is exactly 2.5, so Dᵢ = 0.4; D* = 2/5 = 0.4. Then make your
  rebuild metric pass:

```bash
uv run pytest -c rebuild/pytest.ini rebuild/tests/test_metrics.py::test_pentagon -q
```

**Drill 3.2.** E00's trained pentagon has norms 1.111, 1.101, 1.100, 1.090 and 1.076
([REPRODUCTION](../REPRODUCTION.md), E00). Predict its D* before computing it, and explain
why it is not 0.4. **[Recommended]**
- Self-check: D* = 2/Σ‖Wᵢ‖² = 0.333, the value in the cited run's INFO row. D* counts squared
  norm, and these norms exceed 1. D* depends on W only through ‖W‖_F, so any placement works:

```bash
uv run python -c "import torch; from superposition import metrics; print(round(metrics.dims_per_feature(torch.tensor([[[1.111, 0, 0, 0, 0], [0, 1.101, 1.1, 1.09, 1.076]]])).item(), 3))"
```

**Drill 3.3.** Prove that Σᵢ Dᵢ ≤ m for every W. Hint: with M = WWᵀ, Dᵢ = ‖Wᵢ‖⁴/(WᵢᵀMWᵢ) ≤
WᵢᵀM⁺Wᵢ by Cauchy–Schwarz, and Σᵢ WᵢᵀM⁺Wᵢ = tr(M⁺M) = rank(M). Then write a hypothesis test of
the inequality for random W. **[Recommended]**
- Self-check: your test passes on random W, and equality holds on tight frames:

```bash
uv run pytest tests/unit/test_metrics.py::test_unit_norm_tight_frames_have_uniform_dimensionality -q
```

## 4. Statistical tests: tolerances and calibration

**What it is.** A test of random code compares an estimate with a known value within a band of
standard errors (SE). A correct sampler still fails a 3-SE band sometimes. A calibration test
checks the stronger property that many z-scores have mean ≈ 0 and std ≈ 1.

**Where in this repo.**
- phase-1.md ("Test status", [link](../progress/phase-1.md)): the density test failed at seed 2
  by 3.45σ. A 3,000-draw study (mean z −0.03, std 1.001) showed no bias, so the seed was moved
  to 12 with a comment, and a 64-seed calibration test was added. **[Documented]**
- The L1 test's margin was relaxed from 0.5 to 0.8 [S 35cab3f0 @ 2026-09-29T08:33Z]; the test
  carries the comment "observed ratio ≈ 0.62 after 600 steps"
  ([`test_train.py`](../../tests/unit/test_train.py)). **[Documented]**
- Theory tests compare exact losses with Monte Carlo within a fixed number of SEs (D-301).
  **[Documented]**

**Drill 4.1.** Compute the probability that a correct two-sided 3-SE check fails. **[Recommended]**
- Self-check: 2·(1 − Φ(3)) ≈ 0.0027, the "0.27% expected" of phase-1.md.

**Drill 4.2.** Implement `IndependentSparse` (milestone M1), then run the calibration test before
any single-seed density test. **[Recommended]**
- Self-check: the calibration test passes; then all 14 data tests pass.

```bash
uv run pytest -c rebuild/pytest.ini rebuild/tests/test_data.py -q
```

## 5. Closed-form and exact-expectation checks

**What it is.** Before trusting a training loop, compute a loss you can derive exactly and check
that the code reproduces it. Here the *linear model* optimum is known in closed form, and the
ReLU output model's expected loss is a sum over sparsity patterns of piecewise integrals.

**Where in this repo.**
- `theory.linear_optimum_loss` ([`theory.py`](../../src/superposition/theory.py)):
  L* = Σ_{i≥m} Iᵢ(p/3 − p²/4), with a proof sketch in the docstring. SPEC §7 E00 lists 0.183,
  0.124 and 0.068 for p = 1, 0.2 and 0.1. **[Documented]**
- The planning conversation derived this optimum before E00's criteria were written and checked
  it against a JAX run [Planning conversation, per owner]. **[Documented]**
- D-301: every expectation in `theory.py` is closed form; `relu_square_integral` agrees with
  scipy's adaptive quadrature to 1e-11. **[Documented]**

**Drill 5.1.** Derive L* (hint: the optimal bias is b = (I − WᵀW)μ, and the best rank-m
approximation of diag(√Iᵢ) keeps the m largest entries). Evaluate it for n = 5, m = 2,
Iᵢ = 0.9ⁱ. **[Recommended]**
- Self-check: 0.18292, 0.12439 and 0.06768 (REPRODUCTION, E00). Then:

```bash
uv run pytest tests/unit/test_train.py::test_evaluation_reproduces_linear_optimum -q
```

**Drill 5.2.** Two features, unit importance, W = [[1, 0.3], [0, 0.954]], all biases −0.35. On
1-sparse inputs xᵢ ∈ [0, 1], is any interference passed by the output ReLU? **[Recommended]**
- Self-check: no. The largest interference is 0.3·x ≤ 0.3 < 0.35, so ReLU(0.3x − 0.35) = 0
  (p. 13). The repository test makes the same point:

```bash
uv run pytest tests/unit/test_theory.py::test_negative_interference_is_free_on_one_sparse_inputs -q
```

## 6. Criterion design and operationalization

**What it is.** A criterion in words ("never decreases", "a plateau", "a cluster") must become a
test with numbers before the first graded run. Each number gets a source (paper page, SPEC, or
an earlier criterion) and a DECISIONS entry.

**Where in this repo.**
- D-105 fixed E01's words before its first run; "never decreases (tolerance 1)" compares each
  count with the running maximum of denser settings, not only the previous one. **[Documented]**
- D-305, D-306 and D-307 operationalize E02–E04. D-305 and D-306 disclose that exploratory runs
  had been seen before they were written. **[Documented]**
- Wrong criteria corrected openly: E08's "> m/2" came from SPEC, not the paper, and was
  re-derived from p. 37 (D-R01); E03's cluster rule counted one band twice (R2-01, D-603).
  **[Documented]**

**Drill 6.1.** Counts at rising sparsity are 5, 4, 3, 2. Does "never decreases, tolerance 1" pass
under a consecutive-only rule? Under D-105's running-maximum rule? **[Recommended]**
- Self-check: consecutive-only passes (each step drops by 1); running-maximum fails at 3 < 5 − 1.
  That is D-105's stated reason for the stricter rule.

**Drill 6.2.** Re-derive D-R01's 75% threshold. A feature computed on one side only, ReLU(xᵢ),
misses E[xᵢ²; xᵢ < 0] = p/6 of the mean-prediction loss p/3 − p²/4. **[Recommended]**
- Self-check: at p = 0.001 the ratio is 0.5004 ≈ 1/2; the midpoint between 1/2 (one exact side)
  and 1 (nothing computed) is 0.75.

**Drill 6.3.** D-306 counts a feature toward the nearest characteristic fraction within ±0.02.
Dᵢ values spread around 0.3875 ± 0.01: how many "clusters" does that rule find, and what extra
condition stops the double count? **[Recommended]**
- Self-check: two (2/5 and 3/8, which are 0.025 apart); D-603 also requires the feature's
  *tegum factor* to be that fraction's polytope:

```bash
uv run pytest tests/unit/test_phase3_experiments.py::test_e03_a_band_between_two_fractions_is_not_two_clusters -q
```

## 7. Reading weight plots

**What it is.** Five plot types carry most results here. A **WᵀW heatmap** (diverging colors,
0 at the center, the bias beside it) shows which features interfere. **Feature-norm bars** show
‖Wᵢ‖, colored by superposition score from dark (orthogonal) to bright (≥ 1). **2D arrows** show
each Wᵢ when m = 2. A *neuron stack plot* shows each neuron's weights stacked by feature, negatives
below the axis. **D\* and Dᵢ plots** summarize geometry against 1/(1 − S).

**Where in this repo.**
- SPEC §3 "Visualization conventions" and `plotting/`. Published figures:
  [E01 weights](../figures/basic_results/weights_n20_m5.png),
  [E00 arrows](../figures/intro_pentagon/pentagon_arrows.png),
  [E07 stack plots](../figures/privileged_basis/neurons.png),
  [E03 D\*](../figures/uniform_geometry/dstar.png),
  [E03 Dᵢ](../figures/uniform_geometry/dimensionality.png). **[Documented]**

**Drill 7.1.** In the E01 figure, at 1 − S = 0.3 (ReLU), cover the color bar and say which
features have a dedicated dimension and which share one. **[Recommended]**
- Self-check: features 0–2 dedicated, 3–6 in antipodal pairs (phase-1.md, question 3).

**Drill 7.2.** In the E07 figure, cover each panel's title and count monosemantic and
polysemantic neurons at 1 − S = 0.3 and 0.07. **[Recommended]**
- Self-check: 3 and 2 at 0.3; 0 and 5 at 0.07 (the panel titles).

## 8. Canonicalizing 2D solutions

**What it is.** The hidden space of the linear and ReLU output models has no privileged basis
(p. 31), so a 2D solution is defined only up to rotation and reflection. Canonicalizing picks one
representative so restarts and sparsities can be compared (p. 22).

**Where in this repo.**
- D-109: `metrics.canonicalize_2d` rotates the most important represented feature to +y, then
  reflects so the first off-axis feature has x < 0. **[Documented]**
- `test_angles_and_canonicalize_2d` checks invariance under random rotations and reflections.
  **[Documented]**

**Drill 8.1.** By hand, canonicalize W with columns (1, 0), (−1, 0), (0, 1). **[Recommended]**
- Self-check: (0, 1), (0, −1), (−1, 0). Feature 1 lies on the axis, so feature 2 decides the
  reflection:

```bash
uv run python -c "import torch; from superposition import metrics; print(metrics.canonicalize_2d(torch.tensor([[[1.0, -1.0, 0.0], [0.0, 0.0, 1.0]]])).round(decimals=3))"
```

**Drill 8.2.** Implement `canonicalize_2d` in the rebuild kit from D-109's rule. **[Recommended]**
- Self-check:

```bash
uv run pytest -c rebuild/pytest.ini rebuild/tests/test_metrics.py::test_angles_and_canonicalize_2d -q
```

## 9. Restarts and best-of selection

**What it is.** Gradient descent in 2D gets stuck, so the paper fits several times and keeps the
lowest loss (p. 23). The selection is only as good as the loss estimate: picking the minimum of
noisy estimates favors lucky draws (the winner's curse).

**Where in this repo.**
- `train.select_best` picks the lowest-loss instance per group, and never picks a NaN loss
  unless a whole group is NaN. **[Documented]** (its docstring in `src/superposition/train.py`.)
- E00 grades the linear loss on 100,000 samples; its best-of-8 linear losses still sit below the
  exact optimum (0.12392 against 0.12439 at S = 0.8), and a 20,000-sample estimate of the
  pentagon loss has 3.7% relative noise (REPRODUCTION, E00). **[Documented]**
- D-307 uses 200,000 evaluation samples in E04, because a set at density 0.01–0.02 is active only
  a few hundred times in 20,000. Review finding R1-06 ([REVIEW](../REVIEW.md)): in E01 another
  restart wins in 7 of 14 sparse conditions when re-evaluated. **[Documented]**

**Drill 9.1.** Losses [0.5, 0.2, 0.9, 0.1, 0.3, NaN, 0.4], groups [2, 2, 0, 0, 1, 1, 1]: which
instance wins each group? **[Recommended]**
- Self-check: [3, 4, 1] for groups 0, 1, 2.

```bash
uv run pytest tests/unit/test_train.py::test_select_best -q
```

**Drill 9.2.** Simulate the winner's curse: eight unbiased estimates with relative noise σ, keep
the minimum. How far below the truth is it on average? **[Recommended]**
- Self-check: about 1.42σ, the mean maximum of 8 standard normals. At σ = 3.7% that is about 5%,
  which is why E00 re-evaluates its chosen pentagon on 1,000,000 samples (D-107).

## 10. Phase diagrams: family theory against the vote

**What it is.** A *phase diagram* labels each cell of an (importance, density) grid with the
solution the model adopts. The theory minimizes the exact loss of a few *weight families*; the
empirical diagram trains 10 models per cell, drops the worst and takes the majority of 9 (p. 14).

**Where in this repo.**
- D-302 (families and their free scalars), D-304 (labels and the vote), D-305 (≥ 80% agreement);
  `theory.theoretical_phase_diagram` and `experiments/phase_change.py`. **[Documented]**
- The cited n = 2 diagram disagrees with the theory in 18 of 144 cells
  ([figure](../figures/phase_change/phase_n2_m1.png)). Under D-605's rule, fixed before the
  held-out runs, the n = 3 agreement was 0.819, 0.764 and 0.792 on seeds 0, 1 and 2, against a
  0.80 bar (REVIEW R1-01). **[Documented]**

**Drill 10.1.** Apply D-304's vote to two cells of five models (drop the highest loss, majority
of the rest, ties to the lowest-loss model among the tied labels). Cell A: labels a, a, b, b, b
with losses 1–5. Cell B: labels a, b, b, a, c with losses 5, 1, 2, 3, 0.5. **[Recommended]**
- Self-check: A → "a" (a 2–2 tie broken by loss 1); B → "b".

```bash
uv run pytest tests/unit/test_phase3_experiments.py::test_e02_cell_majority_discards_the_worst_model -q
```

**Drill 10.2.** From the figure's disagreement count, compute the agreement. **[Recommended]**
- Self-check: 126/144 = 0.875 (D-309).

## 11. Analytic worst cases (adversarial attacks)

**What it is.** For these models the most harmful small L2 perturbation of each feature is known
in closed form (p. 29). *Adversarial vulnerability* compares the damage with that of a model
without superposition, attacked with the same budget ε.

**Where in this repo.**
- D-405 and [`adversarial.py`](../../src/superposition/adversarial.py): candidates
  ε(WᵀW)ᵢ/‖(WᵀW)ᵢ‖, the per-input worst, and a reference whose damage is maxᵢ Iᵢε².
  D-406 discloses a prototype seen before the graded range was chosen. **[Documented]**

**Drill 11.1.** W = [1, −1] in one dimension, zero bias, zero input, ε = 0.1. Compute the attacked
loss and the vulnerability. **[Recommended]**
- Self-check: the push along (1, −1)/√2 raises output 0 by ε√2, so the loss is 2ε² = 0.02; the
  reference's is ε²; vulnerability exactly 2.

```bash
uv run pytest tests/unit/test_adversarial.py::test_antipodal_pair_on_zero_inputs_is_twice_as_vulnerable -q
```

**Drill 11.2.** Importances 1.0, 0.8, …, and ε = 0.3 and 0.05: the reference model's damage?
**[Recommended]**
- Self-check: I₀ε² = 0.09 and 0.0025.

```bash
uv run pytest tests/unit/test_adversarial.py::test_reference_damage_is_top_importance_times_budget_squared -q
```

## 12. Path gains and ablations

**What it is.** In the absolute-value model h = ReLU(W₁x), y′ = ReLU(W₂h + b), a *path gain*
G_h[i, j] = |W₁[h, i]|·W₂[j, h] is what neuron h passes from feature i to output j on 1-sparse
inputs. An *ablation* zeroes a neuron (or replaces it by its mean) and measures each feature's
loss change.

**Where in this repo.**
- D-501 and [`interp.py`](../../src/superposition/interp.py): roles from path gains, which are
  invariant to footnote 15's rescaling; `ablation_matrix`. **[Documented]**
- Review findings R4-01 and R4-02: the motif detector matched weight patterns on unexplained
  neurons, and about half the neurons of random weights got a role. D-604 tightened both.
  **[Documented]**

**Drill 12.1.** One neuron with W₁ row (−1, 0.25) and W₂ column (1, 4). Compute the 2 × 2 path
gains and say which feature interferes with which. **[Recommended]**
- Self-check: G = [[1, 4], [0.25, 1]]: both features computed exactly (diagonal 1), feature 0
  pushes 4 onto output 1, feature 1 only 0.25 onto output 0. That is *asymmetric
  superposition* (p. 39).

```bash
uv run python -c "import torch; from superposition import interp; W1 = torch.tensor([[-1.0, 0.25]]); W2 = torch.tensor([[1.0], [4.0]]); print(interp.path_gains(W1, W2)[0])"
```

**Drill 12.2.** In the exact |x| model (one positive- and one negative-side neuron per feature),
derive the loss rise from zeroing a side neuron, and from mean-ablating it (E[h] = c = p/4). Use
p = 0.3. **[Recommended]**
- Self-check: p/6 = 0.05 and p/6 − pc/2 + c² = 0.0444.

```bash
uv run pytest tests/unit/test_interp.py::test_ablations_of_the_exact_model_match_closed_forms -q
```

## 13. Rebuild tests

**What it is.** A *rebuild test* turns an explanation into a model: build idealized weights by
hand from the inferred roles, and report the share of the trained model's loss reduction (over
predicting the mean) that the rebuild recovers: (L_mean − L_rebuild)/(L_mean − L_trained).

**Where in this repo.**
- `interp.rebuild_weights` and `interp.rebuild_test` (D-501). D-501 records that, before
  inhibitor *triggers* were added after an uncited pilot, the rebuild recovered 51–84% instead of
  94–97%. **[Documented]**

**Drill 13.1.** L_mean = 0.10, L_trained = 0.02, L_rebuild = 0.03. Compute the recovered share.
**[Recommended]**
- Self-check: (0.10 − 0.03)/(0.10 − 0.02) = 0.875.

**Drill 13.2.** Explain why a model that already is its idealization must recover exactly 1, and
why an unexplained neuron is zeroed rather than copied. **[Recommended]**
- Self-check:

```bash
uv run pytest tests/unit/test_interp.py::test_rebuild_reproduces_idealized_models tests/unit/test_interp.py::test_rebuild_drops_unexplained_neurons_and_idealizes_the_rest -q
```

## 14. SAE training and ground-truth evaluation

**What it is.** A *sparse autoencoder* (SAE) learns an overcomplete dictionary of the hidden
activations h. In a toy model the true feature directions are known, so you can check whether
the dictionary found them: max cosine per true feature, and the fraction above 0.9 or 0.95.

**Where in this repo.**
- D-503 and [`sae.py`](../../src/superposition/sae.py): ReLU encoder, unit-norm decoder columns,
  MSE + λ·L1, TopK, dead-latent resampling. **[Documented]**
- E10 first graded the SAE chosen by recovery itself (D-504). D-R04 replaced it with a rule a
  practitioner could apply without ground truth, and E10 went from PASS 3/3 (phase-5.md) to
  PARTIAL 1/3 (REPRODUCTION status table). **[Documented]**

**Drill 14.1.** An SAE with latents +e₁, −e₁, +e₂, −e₂ and zero biases reconstructs every 2D
vector exactly. For h = (0.5, −0.25), (0, 0), (−1, 0), compute L0 and the dead fraction.
**[Recommended]**
- Self-check: L0 = (2 + 0 + 1)/3 = 1; +e₂ never fires, so the dead fraction is 0.25. Perfect
  reconstruction says nothing about recovering a pentagon's five directions.

```bash
uv run pytest tests/unit/test_sae.py::test_signed_basis_sae_reconstructs_exactly -q
```

**Drill 14.2.** Apply D-R04's rule (among SAEs with held-out FVU ≤ 0.1, take the lowest L0;
ties to the lower FVU; if none qualifies, the lowest FVU) to five SAEs with (FVU, L0) = (0.02,
0.9), (0.4, 0.3), (0.08, 0.6), (0.01, 1.0), (0.05, 0.6). **[Recommended]**
- Self-check: four qualify; the fifth (FVU 0.05, L0 0.6) wins. With a bar of 0.005 none qualifies
  and the fourth wins.

```bash
uv run pytest tests/unit/test_phase5_experiments.py::test_e10_selection_without_ground_truth -q
```

## 15. Provenance discipline

**What it is.** Every number traces to a logged run: config, seed, commit and a *dirty* flag
recorded when the run starts. A *cited run* is pinned in `docs/cited_runs.yaml` and must come
from a clean tree.

**Where in this repo.**
- [`runs.py`](../../src/superposition/runs.py) writes `manifest.json` (commit, dirty, command,
  times, wall-clock). D-202 adds `report --pin`, which refuses dirty runs; D-R07 adds labeled
  supplementary citations. **[Documented]**
- D-606 (review R1-10): untracked files under `src/` and `configs/` now mark a run dirty.
  **[Documented]**
- A clean detached worktree was first used for a run in session 92884087, for E08's full
  profile [S 92884087 @ 2026-09-29T17:58Z]. From rectification 1 on, cited runs were launched
  that way. Three runs still cited from before then (E01 at `340f825`, E06 and E07 at `5a825d0`)
  ran from the main tree while it was clean (each run's `manifest.json`: `command` and
  `git.dirty`). **[Documented]**
- Runs outside `results/` (scratchpad scripts and development runs) are disclosed in D-305,
  D-306, D-406, D-502 and D-505; see [01_timeline.md](01_timeline.md), "3. Runs outside the
  provenance system". **[Documented]**

**Drill 15.1.** Run E00's quick profile, then open the new run's
`results/intro_pentagon/<run_id>/manifest.json` and find the commit, the dirty flag, the
command and the wall-clock. **[Recommended]**
- Self-check: overall PASS 11/11 in `acceptance.json`; `git.dirty` false on a clean tree. The cited
  run took 68 s on an Apple M2 CPU (`20261001T071529Z-5eeb804c`).

```bash
uv run superposition run intro_pentagon --profile quick --device cpu
```

**Drill 15.2.** Predict whether a new untracked file in `src/` and a new untracked note in
`docs/` each make a run dirty. **[Recommended]**
- Self-check: yes and no (D-606):

```bash
uv run pytest tests/unit/test_runs_and_report.py::test_untracked_code_makes_the_tree_dirty tests/unit/test_runs_and_report.py::test_pinning_refuses_dirty_runs -q
```

**Drill 15.3.** Practice the clean-worktree flow once on a scratch clone: create a detached
worktree, sync it, run E00 with `SUPERPOSITION_RESULTS` pointing at the main `results/`, and
check the manifest. **[Recommended]**

```bash
git worktree add --detach /tmp/superposition-runs HEAD
uv sync --directory /tmp/superposition-runs
SUPERPOSITION_RESULTS=/path/to/superposition/results uv run --directory /tmp/superposition-runs superposition run intro_pentagon --profile quick --device cpu
```

Fill in the results path. Checked in a scratch clone with the path filled in
([VERIFICATION.md](VERIFICATION.md)); afterwards remove the worktree with
`git worktree remove /tmp/superposition-runs`.

## 16. Reading a paper figure at 300 dpi

**What it is.** Thresholds read from a figure are only as good as the rendering. Render the page
at 300 dpi, crop the chart, and record each reading as approximate, with its resolution.

**Where in this repo.**
- D-113 claimed E01 represented fewer features than pp. 11–12 show, from low-resolution
  renders. At 300 dpi there was no shortfall: about 12 features at full norm plus about 4 near
  0.5 on p. 11 (ours 14), about 50 plus a partial tail on p. 12 (ours 67). D-114 corrected it.
  **[Documented]**
- The Phase 3 draft rendered p. 17 at 300 dpi before writing E03's thresholds
  [S 2df87f77 @ 2026-09-29T13:34Z]; D-306 records the readings. The planning conversation read
  the same curve at 300 dpi [Planning conversation, per owner]. **[Documented]**

**Drill 16.1.** Render p. 17 and read the 1/2 plateau's span. **[Recommended]**
- Self-check: roughly [2.6, 3.7] (SPEC §7 E03, D-306). Then compare with the repository's
  2.34–3.79, and say whether the difference is a reading error or a finding.

```bash
pdftoppm -r 300 -f 17 -l 17 -png references/toy_models_of_superposition.pdf p17
```

Needs the git-ignored paper PDF and poppler's `pdftoppm`; checked with the owner's copy
([VERIFICATION.md](VERIFICATION.md)).

**Drill 16.2.** Count p. 11's features at 1 − S = 0.001 at screen resolution, then at 300 dpi.
**[Recommended]**
- Self-check: D-114's readings above. Write both counts and the resolution in your lab log.

## 17. Writing honest reports

**What it is.** A report states the claim with its page, the status, the evidence (run IDs), and
every discrepancy with a hypothesis labeled as a hypothesis. It keeps failures visible.

**Where in this repo.**
- REPRODUCTION.md's E00 section separates "our model is worse" from "the reference estimate is
  noisy", and after review R1-05 calls its explanation a hypothesis with a probability.
  **[Documented]**
- E08 keeps SPEC's original criterion as an INFO row beside the re-derived one (D-R01); E11's
  pentagon was downgraded to PARTIAL by the owner (D-R05). **[Documented]**

**Drill 17.1.** Fill a [REPRODUCTION_ENTRY](templates/REPRODUCTION_ENTRY.md) for E00 from your
own M5 run (rebuild kit). **[Recommended]**
- Self-check: compare with [REPRODUCTION](../REPRODUCTION.md) § E00; every number you write
  must appear in your run's records.

**Drill 17.2.** Rewrite this overclaim: "The 0.0196 reference was a lucky draw." **[Recommended]**
- Self-check: compare with REPRODUCTION § E00, "Hypothesis": it says hypothesis, gives 2.2
  standard deviations and "a probability of a few percent", and notes that no criterion depends
  on it.
