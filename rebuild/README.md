# Rebuild kit: write the core engine yourself

This kit is where the cookbook's skill transfer happens. It holds a stripped copy of the
repository's core engine: the feature distributions (`data`), the toy models (`models`), the
multi-instance trainer (`train`) and the superposition metrics (`metrics`). Every signature and
docstring is the repository's own. Every body raises `NotImplementedError`. The repository's
unit tests for those four modules come with it, changed only to import from the skeleton.

Your job is to make the tests pass, milestone by milestone, without opening
`src/superposition/`. Then reproduce the paper's first figure (E00) with your own code, and
check your numbers against the repository's cited run.

Read [docs/cookbook/05_engine.md](../docs/cookbook/05_engine.md) alongside this kit. It explains
each module's job, the math with paper pages, and the pitfalls this repository actually hit.
Terms are defined in [docs/GLOSSARY.md](../docs/GLOSSARY.md). Labels follow the cookbook
([README](../docs/cookbook/README.md)): **[Documented]** facts about this repository's runs,
**[Reconstructed]** inferences, **[Recommended]** advice. The milestones themselves are
recommendations.

## What is here

| Path | What it is |
|---|---|
| `superposition_rebuild/data.py` | M1: `geometric_importance`, `IndependentSparse`, `CorrelatedSets`, `AnticorrelatedSets`, `Composite` |
| `superposition_rebuild/models.py` | M2: the activation registry, `xavier_normal_per_instance`, `TiedModel`, `AbsModel` |
| `superposition_rebuild/train.py` | M3: `importance_weighted_loss`, `lr_multiplier`, `evaluate_loss`, `train`, `select_best` |
| `superposition_rebuild/metrics.py` | M4: norms, the superposition score, Dᵢ, D*, the formal superposition test, 2D canonicalization, the geometry graph and tegum factors, k-sparse loss terms, per-feature loss |
| `tests/test_{data,models,train,metrics}.py` | the repository's tests (14, 12, 13 and 25 of them) |
| `pytest.ini` | the kit's own test settings |

The kit is kept out of the main package, `make lint`, pyright, the default `pytest` run and CI,
because its tests fail until you implement it. Only `make rebuild-test` runs them.

## Rules

**[Recommended]**

1. **The tests are the specification for M1–M4.** Do not edit them. If you think a test is
   wrong, write down why in your lab log ([template](../docs/cookbook/templates/LAB_LOG.md)) and
   keep going. The repository has examples of tests changed openly, with a recorded reason
   (docs/progress/phase-1.md, "Test status").
2. **Do not open the answer key until you have tried.** The answer key for each milestone is the
   matching file in `src/superposition/` and its test in `tests/unit/`. Try for real first.
3. **You may import infrastructure from the real package.** The skeleton already imports
   `superposition.utils` (seed streams such as `derive_seed`, `make_generator`) and
   `superposition.config` (`TrainConfig`, `Schedule`). These are plumbing, not the exercise.
   `tests/test_train.py` imports `superposition.theory.linear_optimum_loss`; you derive that
   formula yourself in M5.
4. **Use the libraries the repository uses:** torch, einops, jaxtyping, numpy, networkx.
5. **No Python loops over instances or features in hot paths.** Loops at initialization are
   fine (CLAUDE.md, "Conventions").

## Start

Run every command from the repository root.

1. Create the environment.

   ```bash
   uv sync
   ```

   Expected: it finishes without errors. Check: `uv run python -c "import torch"` prints
   nothing.

2. Run the kit's tests as shipped.

   ```bash
   make rebuild-test
   ```

   Expected: `64 failed`, every one with `NotImplementedError`, and `make` reports an error.
   That is the starting line. If anything fails with another error, your environment is not the
   repository's (check `uv sync` again).

3. Work on one milestone at a time with its own test file, for example:

   ```bash
   uv run pytest -c rebuild/pytest.ini rebuild/tests/test_data.py -x -q
   ```

   `-x` stops at the first failure, so you fix one thing at a time.

## Graded milestones

Each milestone has three grades. **Pass:** the tests are green. **Explain:** you can answer the
questions aloud without notes. **Stretch:** optional, for depth.

### M1 · Data (pp. 9, 23, 35)

- **Implement:** `superposition_rebuild/data.py`.
- **Paper:** p. 9 (each feature is 0 with probability S, else uniform on [0, 1]); p. 23
  (correlated and anticorrelated sets); p. 35 (values on [−1, 1] for |x|).
- **Pass:**

  ```bash
  uv run pytest -c rebuild/pytest.ini rebuild/tests/test_data.py -q
  ```

  Expected: `14 passed`.
- **Watch for:**
  - Every batch is shaped `(batch, inst, feat)`. The instance dimension is what lets one call
    train a whole sparsity sweep.
  - A probability can be a scalar, one per instance, one per feature, or a full
    `(inst, feat)` table. Out-of-range values and wrong shapes must raise `ValueError`.
  - The density tests use a 3-standard-error band at a fixed seed. Any correct sampler fails
    such a test about 0.3% of the time per check. If one fails, run
    `test_density_z_scores_are_calibrated` first: it checks the stronger property across 64
    seeds. Do not change a seed to make a test pass without that study.
- **Explain:** (1) Why is the repository's one-uniform-per-entry trick (u < p, value u/p) exactly
  the same distribution as two independent draws (D-102)? (2) What does "a correlated set"
  share, and what does it not share?
- **Stretch:** time your sampler for 1024 × 64 × 400 on CPU, then implement the one-uniform
  trick and time it again.

### M2 · Models (pp. 10, 31, 35)

- **Implement:** `superposition_rebuild/models.py`.
- **Paper:** p. 10 (x′ = ReLU(WᵀWx + b)); p. 31 (ReLU on the hidden layer, no hidden bias, p. 34);
  p. 35 (the absolute-value model with untied W₁, W₂).
- **Pass:**

  ```bash
  uv run pytest -c rebuild/pytest.ini rebuild/tests/test_models.py -q
  ```

  Expected: `12 passed`.
- **Watch for:**
  - `W` is `(inst, hidden, feat)`, so feature i's direction in instance k is `W[k, :, i]`.
  - Xavier-normal init has std √(2/(n + m)) **per instance**. `nn.init.xavier_normal_` on the
    3-D tensor would count the instance dimension in the fan and change the scale with the
    number of instances (SPEC §3).
  - Instance k's weights must depend only on (seed, k): adding instances never changes
    existing ones. Use `derive_seed(seed, INIT_STREAM, param_key, k)` and a fresh generator per
    instance.
  - `bias=False` means a fixed zero buffer, not a parameter.
- **Explain:** (1) Why does the ReLU hidden layer model have no hidden bias? (2) Why can W be read
  directly in the ReLU hidden model, but only WᵀW in the ReLU output model?
- **Stretch:** add `register_activation("tanh", torch.tanh)` and train one model with it after M3.

### M3 · Training (pp. 10, 14, 23)

- **Implement:** `superposition_rebuild/train.py`.
- **Paper:** p. 10 (L = Σₓ Σᵢ Iᵢ (xᵢ − x′ᵢ)²); p. 23 (fit several times and keep the lowest
  loss); p. 42 (λ‖h‖₁ for `l1_hidden`).
- **Pass:**

  ```bash
  uv run pytest -c rebuild/pytest.ini rebuild/tests/test_train.py -q
  ```

  Expected: `13 passed`. This file also exercises M1 and M2.
- **Watch for:**
  - The loss in three steps: sum over features, mean over the batch (one value per instance),
    **sum** over instances for `backward()`. Summing keeps every instance's gradient exactly
    what it would be alone (`test_instances_are_independent`).
  - A fresh batch every step, from a generator seeded by `derive_seed(seed, DATA_STREAM)`;
    evaluation uses its own stream (`EVAL_STREAM`).
  - The logged steps include the last step, and snapshots include step 0 and the final step.
  - `select_best` never picks a NaN loss unless a whole group is NaN.
- **Explain:** (1) Why does it not matter for Adam whether instances are summed or averaged?
  (2) Why does `test_evaluation_reproduces_linear_optimum` use 200,000 samples?
- **Stretch:** add a `warmup_cosine` schedule run to a small model and plot its loss history.

### M4 · Metrics (pp. 8, 11, 13, 17–20, 22)

- **Implement:** `superposition_rebuild/metrics.py`.
- **Paper:** p. 8 (formal superposition: WᵀW not invertible, on represented features only); p. 11
  (‖Wᵢ‖, superposition score); p. 13 (k-sparse loss terms); p. 17 (D* = m/‖W‖²_F); pp. 18–20
  (Dᵢ, characteristic fractions, the geometry graph, tegum factors); p. 22 (canonicalizing 2D
  solutions).
- **Pass:**

  ```bash
  uv run pytest -c rebuild/pytest.ini rebuild/tests/test_metrics.py -q
  ```

  Expected: `25 passed`, including two hypothesis property tests.
- **Known answers the tests check:** a regular pentagon gives every Dᵢ = 2/5 and D* = 0.4; an
  antipodal pair 1/2; a triangle 2/3; a tetrahedron 3/4; an orthonormal basis 1; a zero column
  0; an n-point regular simplex (n − 1)/n; any unit-norm tight frame m/n with Σᵢ Dᵢ = m.
- **Watch for:** Dᵢ = 0 when Wᵢ = 0 (no division by zero); the rank test uses represented
  features only; every metric except `angles_2d` must be unchanged by rotating the hidden
  space (the property test does exactly that).
- **Explain:** (1) Why can Dᵢ alone not tell a pentagon (2/5) from a square antiprism (3/8), and
  what does `component_geometry` use instead (D-308)? (2) Why does the formal superposition test
  apply only to represented features?
- **Stretch:** compute Dᵢ for a triangular bipyramid by hand before running
  `test_geometry_census_splits_tegum_products`.

When M1–M4 are green:

```bash
make rebuild-test
```

Expected: `64 passed`. Now open the answer key and compare, one module at a time:

```bash
git diff --no-index src/superposition/data.py rebuild/superposition_rebuild/data.py
```

The command exits with status 1 whenever the files differ, which they will. Read the
differences as a code review of your own work: what did the repository handle that you did not,
and what did you do more simply?

### M5 · Reproduce E00 by hand (p. 2)

The paper's first figure: five features in two dimensions become two dedicated dimensions,
then two antipodal pairs, then a pentagon as sparsity rises. Recipe card:
[docs/cookbook/experiments/E00.md](../docs/cookbook/experiments/E00.md).

1. **Derive the closed form first.** For the linear model with n = 5, m = 2 and Iᵢ = 0.9ⁱ, the
   optimum represents the top m features exactly and predicts the mean p/2 for the rest.
   Show that its loss is Σ_{i≥m} Iᵢ (p/3 − p²/4), and evaluate it at p = 1, 0.2 and 0.1.
   Check: 0.183, 0.124 and 0.068 (SPEC §7 E00).
2. **Write your own script**, for example `rebuild/e00_by_hand.py`, using only your four
   modules (plus `superposition.utils`). Setup (SPEC §7 E00, `configs/intro_pentagon.yaml`):
   S ∈ {0, 0.8, 0.9}; 8 restarts per S; the ReLU output and the linear model; Adam, learning
   rate 1e-3, batch 1024, 10,000 steps, constant schedule; final losses on 100,000 fresh
   samples; keep the lowest-loss restart per S and model. Put all 24 runs of one model in one
   multi-instance model: 3 sparsities × 8 restarts.
3. **Run it.**

   ```bash
   uv run python rebuild/e00_by_hand.py
   ```

   [Not verified: you write this script.] For scale, the repository's own E00 quick profile
   took 68 s on an Apple M2 CPU (run `20261001T071529Z-5eeb804c`). **[Documented]**
4. **Grade it against E00's acceptance criteria** (SPEC §7 E00, best restart per S):
   - ReLU, S = 0: exactly features {0, 1} represented (‖Wᵢ‖ > 0.5), |cos| < 0.1.
   - ReLU, S = 0.8: exactly {0, 1, 2, 3}, each with a partner at cos < −0.95.
   - ReLU, S = 0.9: all five; sorted angular gaps within 72° ± 10°; all norms > 1; all biases
     < 0.
   - Linear, every S: exactly {0, 1}, orthogonal.
   - Linear, every S: best loss within 3% of the closed form.
   - Both models: every feature with ‖Wᵢ‖ < 0.05 has a bias within 0.02 of (1 − S)/2.
5. **Compare with the repository.** Its cited run gives:

   | S | ReLU best loss | Linear best loss | Closed form | SPEC's JAX reference (ReLU) |
   |---|---|---|---|---|
   | 0 | 0.18269 | 0.18269 | 0.18292 | 0.182 |
   | 0.8 | 0.05932 | 0.12392 | 0.12439 | 0.059 |
   | 0.9 | 0.02117 | 0.06700 | 0.06768 | 0.0196 |

   The pentagon had angular gaps 69.8°–74.0°, norms 1.076–1.111 and biases −0.198 to −0.211.
   **[Documented]** (Source: `results/intro_pentagon/20261001T071529Z-5eeb804c/acceptance.json`,
   git-ignored; docs/REPRODUCTION.md, E00.)

   Your losses will not match to the last digit, because your random streams differ. The run's
   own noise study puts a 20,000-sample estimate of the pentagon's loss at 3.7% relative
   standard deviation **[Documented]**, so expect differences of a percent or two at 100,000
   samples **[Reconstructed: 3.7% divided by √5]**. Your restarts may also land differently: in
   the cited run, 2 of 8 restarts at S = 0.9 kept only four features **[Documented]**.
6. **Investigate the one disagreement.** At S = 0.9 the repository's pentagon loss (0.0214 on
   1,000,000 samples) sits above SPEC's reference 0.0196 **[Documented]**. Read
   docs/progress/phase-1.md open issue 1 and the E00 section of docs/REPRODUCTION.md, then decide what your own numbers say.

- **Pass:** all eleven criteria hold for your models.
- **Explain:** why are the pentagon's biases negative and its norms above 1? (Use the 1-sparse
  readout ReLU(‖Wᵢ‖²x + bᵢ).)
- **Stretch:** plot your three best ReLU solutions as 2D arrows after canonicalizing them with
  your own `canonicalize_2d`, and compare with `docs/figures/intro_pentagon/pentagon_arrows.png`.

### M6 (optional) · Reproduce E01 (pp. 10–12)

- **Setup** (SPEC §7 E01, `configs/basic_results.yaml`): n = 20, m = 5, Iᵢ = 0.7ⁱ, and n = 80,
  m = 20, Iᵢ = 0.9ⁱ; 1 − S ∈ {1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001}; linear and ReLU output;
  4 restarts; 10,000 steps; 20,000 evaluation samples.
- **Criteria:** SPEC §7 E01 and their exact operationalization in docs/DECISIONS.md D-105.
- **Compare** with the cited run `20260929T093755Z-2cb0bbc4` (472 s on an Apple M2 CPU). Its ReLU
  represented-feature counts were 5, 7, 9, 11, 12, 12, 14 for n = 20 and 20, 27, 33, 41, 52, 61,
  67 for n = 80, densest first. The review found that the best restart at sparse settings is
  chosen by evaluation noise, so counts can move by about ±2 (docs/REVIEW.md R1-06).
  **[Documented]**
- **Pass:** every E01 criterion holds; your counts are within that noise of the repository's.

## Starting over

Your work lives in `rebuild/superposition_rebuild/`. To restore one module to its skeleton,
for example after a failed experiment, check it out from git. This discards your changes to
that file:

```bash
git checkout -- rebuild/superposition_rebuild/data.py
```

[Not verified: discards your own work in that file.]
