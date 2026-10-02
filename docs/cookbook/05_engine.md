# 05 · The engine: building the core library

The *engine* is the library every experiment stands on. This chapter records how it was
actually built, gives an order for building it yourself, then walks through each module: job,
interface, math with paper pages, tests to write first, pitfalls hit here, a check, and the
answer key.

Use it together with the [rebuild kit](../../rebuild/README.md). Its milestones M1–M4 are the
data, models, train and metrics sections below; the kit ships the same signatures and the same
unit tests (14, 12, 13 and 25 tests) with every body raising `NotImplementedError`.

Labels follow [README.md](README.md). Terms (instance, sparsity, feature probability,
superposition score, Dᵢ, tegum factor) are defined in the [glossary](../GLOSSARY.md). One term
used everywhere: an **instance** is one independent copy of a model. Every parameter carries a
leading instance dimension, so a whole sweep of restarts and sparsities trains as one batched
model.

## 1. How it was actually built

### Authored in one pass, committed later as logical commits

Session 35cab3f0 wrote the core library in about twenty minutes, modules first and their unit
tests a few minutes after. **[Documented: Write timestamps in S 35cab3f0, 09-29 08:25–08:46Z]**

| UTC (09-29) | Written | Then |
|---|---|---|
| 08:25 | `utils`, `config` | |
| 08:26 | `data`, `models` | |
| 08:27 | `metrics` | |
| 08:28 | `train` | |
| 08:29 | `theory` (the linear optimum); `test_data`, `test_models` | 08:29 closed-form optimum values checked with a short script; 08:29:51 first test run: a density test fails |
| 08:30 | | z-score studies over 200 and more seeds; test re-run |
| 08:31 | `test_metrics` | a hypothesis strategy error in the tight-frame property test |
| 08:32 | `test_train`, `test_config` | 08:32:54 all unit tests with timings |
| 08:33 | `runs`, `acceptance` | 08:33:06 L1 test margin relaxed from 0.5 to 0.8 |
| 08:34–08:35 | `plotting/style`, `weights`, `arrows` | palette checked with a validator script |
| 08:36–08:37 | `experiments/base`, `runner`, `common` | |
| 08:38 | `experiments/intro_pentagon` (E00) | E00 run; pentagon loss gap studied in scratchpad scripts 08:39–08:41 |
| 08:41–08:42 | | ruff `E501` ignored; F821/NPY002 and pyright errors fixed; E00 re-run with `uv run python -c` after `rm -rf results` |
| 08:45–08:46 | `basic_results` (E01), `report`, `cli` | E01 launched in the background; CLI smoke test |

**[Documented: S 35cab3f0 @ 08:25–08:46Z: the transcript extraction's Write times and command list]**

The code was then committed between 09:01 and 09:05Z as separate logical commits, after E00
and E01 had run and the first notebooks had been drafted. **[Documented: git; S 35cab3f0 @
09:01–09:05Z]**

| Commit | UTC | What |
|---|---|---|
| `529f724` | 09:01:26 | ruff format owns line length (D-112) |
| `ed4b82f` | 09:01:38 | `utils`, `config`, `test_config` |
| `36bacc6` | 09:01:39 | `data`, `test_data` |
| `e1abf9b` | 09:01:40 | `models`, `test_models` |
| `4558329` | 09:01:40 | `train`, `theory.linear_optimum_loss`, `test_train` |
| `e80cdf2` | 09:01:41 | `metrics`, `test_metrics` |
| `0583d5d` | 09:01:56 | `runs`, `acceptance` |
| `54bca87` | 09:01:57 | `plotting/style`, `weights`, `arrows` |
| `3194d04` | 09:01:58 | experiments framework, E00, E01, configs, criteria tests |
| `d23643d` | 09:01:59 | `cli`, `report`, `test_runs_and_report` |
| `162bd1a` | 09:01:59 | root `conftest.py` (test isolation, notebook kernel) |
| `4d34edc`, `94e31be`, `340f825` | 09:05:14–15 | plain tqdm; notebooks 00–02; Phase 1 decisions and DESIGN |

```bash
git log --reverse --format='%h %ad %s' --date=iso 1815b59..340f825
```

Expected: the fourteen commits above, from `529f724` to `340f825`, dated 12:01–12:05 +0300.

**The two orders differ, and the difference matters.** The commit log reads as a clean
bottom-up, test-alongside build. The authoring was bottom-up too, but tests came after their
modules, `metrics` was written before `train`, and the commits were cut from a finished working
tree. **[Documented: the two tables]** Read git history as a record of logical changes, not of
the order of thought. **[Recommended]** Before committing, the session moved the runs made from
a dirty tree out of `results/` into its scratchpad (09:05:37Z) and launched the four Phase 1
runs (E00 and E01, quick on CPU and full on MPS) from a script at commit `340f825`.
**[Documented: S 35cab3f0 @ 09:05Z; manifests of `20260929T090615Z-eb7b9e3b`,
`20260929T090856Z-6d287488`, `20260929T093651Z-5eeb804c`, `20260929T093755Z-2cb0bbc4`]**

### The engine kept growing after Phase 1

| Commit | When (UTC) | Change | Phase |
|---|---|---|---|
| `4355d5d` | 09-29 13:44 | `theory`: linear and 1-sparse decompositions, exact ReLU-output losses, weight families, phase diagrams (committed from outside the session) | 3 draft |
| `acec078` | 09-29 14:37 | `metrics`: `component_geometry`, `geometry_census`; small `theory` helpers (from outside the session) | 3 draft |
| `27c0f82` | 09-29 17:28 | `metrics`: `normalize_hidden_units`, `per_feature_loss` | 4 |
| `81ad83c` | 09-30 18:25 | `acceptance.graded`: PARTIAL as a graded status | rectification 2 |
| `ad8ad78` | 10-01 06:39 | `theory`: exact losses for any output activation (D-601) | 6 |
| `fb808e6` | 10-01 06:50 | `metrics`: `feature_factors`, `polytope_of_fraction` (D-603) | 6 |
| `450938e` | 10-01 06:50 | `runs.git_info`: untracked code makes a run dirty (D-606) | 6 |

**[Documented: `git log` of `theory.py`, `metrics.py`, `runs.py` and `acceptance.py`;
[01_timeline.md](01_timeline.md) for the external commits]** D-301 to D-307, which the Phase 3
draft's code already cited, were written only on 09-30 when Phase 3 was closed.
**[Documented: [DECISIONS.md](../DECISIONS.md) "Phase 3" preamble]**

## 2. The recommended order for building it by hand

**[Recommended]** throughout this section. Build bottom-up, and for each module run this loop:

1. Read the paper pages the module cites. Write down the equations in your own notation.
2. Write the interface: signatures, docstrings with the paper page, jaxtyping shapes.
3. Write the known-answer tests first (section 3 names them). Run them and watch them fail.
4. Implement until the tests pass. No Python loops over instances or features in hot paths.
5. `make lint`. Commit with a conventional message.
6. Only then open the answer key and compare.

| Step | Module | Why here | Rebuild kit |
|---|---|---|---|
| 1 | `utils` | seed streams and devices; everything else uses them | given |
| 2 | `config` | typed settings; no magic numbers anywhere after this | given |
| 3 | `data` | needs only torch | M1 |
| 4 | `models` | needs `utils` for per-instance seeding | M2 |
| 5 | `train` + `theory.linear_optimum_loss` | `test_train` imports `data`, `models` and the closed form | M3 |
| 6 | `metrics` | `test_metrics` imports `data`, `models` and `train.evaluate_loss` | M4 |
| 7 | E00 by hand, as a script | proves the four modules reproduce p. 2 before any framework exists | M5 |
| 8 | `runs` + `acceptance` | now a result deserves a provenance record | — |
| 9 | `experiments/` framework | turn the M5 script into a registered experiment | — |
| 10 | `report` + `cli` | the commands you and a reviewer will run | — |
| 11 | `theory` decompositions | needed from E02 on; validated against M3's Monte Carlo | — |
| — | `plotting` | whenever a figure is needed; keep it separate from computation | — |

Step 7 before step 8 differs from this repository, which wrote the experiment framework
(08:36–08:37Z) before the E00 module (08:38Z). **[Documented: S 35cab3f0 @ 08:36–08:38Z]** By
hand, a plain script first keeps the first success small, and a framework built afterwards has
a working example to serve.

## 3. Module by module

Every check command runs from the repository root. The answer key is the module in
`src/superposition/`: open it after you have tried.

### 3.1 `utils` and `config`: seeds, devices, typed settings

**Responsibilities.** One global seed fans out into independent random streams; device
resolution; dataclass configs loaded strictly from YAML (unknown keys and wrong types are
errors), with a `base` block plus `profiles: {quick, full, …}` and a stable hash for run IDs.

**Interface.**

- `derive_seed(seed: int, *keys: int) -> int` (depends only on its arguments);
  `make_generator(seed, device="cpu") -> torch.Generator`; `resolve_device("auto")` picks CUDA,
  then MPS, then CPU; `seed_everything(seed)` only for scripts and notebooks. Stream keys:
  `INIT_STREAM = 0`, `DATA_STREAM = 1`, `EVAL_STREAM = 2`, `PROBE_STREAM = 3`.
- `TrainConfig(steps=10_000, batch_size=1024, lr=1e-3, schedule="constant", warmup_frac=0.05,
  l1_hidden=0.0, log_every=100, snapshot_every=None, eval_samples=20_000)`;
  `ExperimentConfig(experiment, profile, seed, device, train)`, which experiments subclass;
  `from_dict`, `to_dict`, `deep_merge`, `load_config(cls, path, profile="quick", overrides=None)`,
  `config_hash(config, length=8)`, `ConfigError`.

**Key ideas.** NumPy's `SeedSequence(entropy=seed, spawn_key=keys)` gives statistically
independent child streams; instance k's initialization uses `(seed, INIT_STREAM, param, k)`, so
adding instances never changes existing ones (SPEC §3). The paper never states its optimizer:
Adam, lr 1e-3, batch 1024, 10,000 steps were fixed in SPEC §6 after the planning conversation's
verification run of E00, and recorded as D-101. **[Documented: D-101; Planning conversation,
per owner]**

**Tests first.** `test_derive_seed`, `test_resolve_device`,
`test_from_dict_builds_nested_and_coerces`, `test_from_dict_rejects_bad_input`,
`test_deep_merge`, `test_load_config_profiles_and_round_trip`,
`test_config_hash_is_stable_and_content_sensitive`, `test_named_variant_profiles`,
`test_missing_profile_is_an_error` (all in `tests/unit/test_config.py`).

**Pitfalls.** D-103 rejected `seed + k` arithmetic: streams of runs with nearby seeds can
collide. **[Documented: D-103]** Seeding NumPy's legacy global RNG tripped ruff `NPY002`; it is
intentional in `seed_everything`, so it got a `noqa` with the reason. **[Documented: S 35cab3f0
@ 08:42Z; utils.py]** Named profiles beyond quick and full came later: `batch1024` (D-309, run
`20260930T075829Z-609cc6fb`) and `pilot` (D-R03, run `20261001T072229Z-1e8bafbd`).
**[Documented: those manifests]** Strict loading matters because a misspelled key that falls
back to a default is a wrong experiment that still runs. **[Recommended]**

**Check.**

```bash
uv run pytest tests/unit/test_config.py -q
```

Expected: `15 passed`. Answer keys: `src/superposition/utils.py`, `src/superposition/config.py`.

### 3.2 `data`: feature distributions (rebuild kit M1)

**Responsibilities.** Sample sparse feature vectors for every instance at once.

**Interface.**

- Protocol `FeatureDistribution`: attributes `n_features`, `n_inst`;
  `sample(batch_size, generator) -> Float[Tensor, "batch inst feat"]`.
- `IndependentSparse(feature_probability, n_inst, n_features, low=0.0, high=1.0)`; the
  probability is a scalar, `(inst,)`, `(1, feat)` or `(inst, feat)`.
- `CorrelatedSets`, `AnticorrelatedSets`, `Composite(parts)` (concatenates along features).
- `geometric_importance(n_features, decay) -> Float[Tensor, " feat"]`, Iᵢ = decayⁱ.

**Math (p. 9; p. 23; p. 35).** xᵢ = 0 with probability S, otherwise U[0, 1]. The code uses the
feature probability p = 1 − S everywhere. The absolute-value task uses values on [−1, 1].

**The one-uniform trick (D-102).** Draw u ~ U[0, 1) once per entry. The feature is active iff
u < p, and its value is low + (high − low)·u/p. Conditioned on u < p, u/p is exactly U[0, 1), so
the joint distribution equals two independent draws, at half the random-number cost. Phase 0's
scratchpad benchmark found random-number generation dominated a CPU training step (about 11 ms
per 4.6 M uniforms, single-threaded). **[Documented: D-102; phase-0.md "Open issues";
S 35cab3f0 @ 08:14–08:16Z, outside the provenance system]**

**Tests first.** `test_independent_density_matches_p` (four values of p),
`test_density_z_scores_are_calibrated`, `test_independent_per_instance_and_per_feature_probabilities`,
`test_independent_nonzero_range_and_mean`, `test_independent_rejects_bad_probabilities`,
`test_correlated_sets_switch_together`, `test_anticorrelated_sets_never_two_active`,
`test_composite_concatenates_parts`, `test_sampling_is_deterministic_given_generator`,
`test_geometric_importance`.

**Pitfall: a correct sampler failed a statistical test.** The per-instance density check
(3 standard errors at a fixed seed) failed at seed 2 by 3.45σ at the first test run. The session
ran z-score studies; a 3,000-draw study gave mean z −0.03, std 1.001 and a tail rate of 0.33%
against 0.27% expected, so the sampler was unbiased and the failure was a fixed-seed false
positive. The test moved to seed 12 with a comment saying why, and a new 64-seed calibration
test checks the stronger property. **[Documented: [phase-1.md](../progress/phase-1.md) "Test
status"; S 35cab3f0 @ 08:29–08:30Z; tests/unit/test_data.py]** A 3-SE check fails about 0.27% of
the time by design. Changing a seed is honest only after a study like this, and only with the
reason written next to it. **[Recommended]**

**Check.**

```bash
uv run pytest tests/unit/test_data.py -q
uv run pytest -c rebuild/pytest.ini rebuild/tests/test_data.py -q
```

Expected: `14 passed` for the repository; for the kit, `14 failed` until you implement M1, then
`14 passed`. Answer key: `src/superposition/data.py`.

### 3.3 `models`: the toy models (rebuild kit M2)

**Responsibilities.** The three tied-weight models and the absolute-value model, with a leading
instance dimension, and per-instance initialization.

**Interface.**

- `TiedModel(n_features, n_hidden, n_inst, hidden_act=None, out_act=None, bias=True, seed=0)`;
  `TiedModel.from_kind("linear" | "relu_output" | "relu_hidden", n_features, n_hidden, n_inst, seed)`.
- Parameters `W: (inst, hidden, feat)` (column `W[k, :, i]` is Wᵢ in instance k), `b: (inst, feat)`.
  With `bias=False`, `b` is a fixed zero buffer.
- Methods: `hidden(x) -> Float[Tensor, "batch inst hidden"]`, `readout(h)`,
  `forward(x) -> Float[Tensor, "batch inst feat"]`, `forward_with_hidden(x)`, `target(x)`,
  `gram() -> Float[Tensor, "inst feat feat"]`.
- `AbsModel(n_features, n_hidden, n_inst, seed)`: `W1: (inst, hidden, feat)`,
  `W2: (inst, feat, hidden)`, target |x|.
- `xavier_normal_per_instance(n_inst, fan_out, fan_in, seed, param_key=0) -> Float[Tensor, "inst out in"]`.
- Activation registry: `get_activation(name)`, `register_activation(name, fn)`.

**Math.**

| Model | Paper | Hidden | Output |
|---|---|---|---|
| linear | p. 10 | h = Wx | x′ = WᵀWx + b |
| ReLU output | p. 10 | h = Wx | x′ = ReLU(WᵀWx + b) |
| ReLU hidden | p. 31 | h = ReLU(Wx), **no hidden bias** (p. 34) | x′ = ReLU(Wᵀh + b) |
| absolute value | p. 35 | h = ReLU(W₁x) | y′ = ReLU(W₂h + b), target \|x\| |

Use +b everywhere. The phase-change section writes ReLU(WᵀWx − b) (p. 14), so its b is our −b.
Initialization: Xavier normal per instance, std √(2/(n + m)), biases zero (SPEC §3). Every hot
path is one `einops.einsum`, for example
`"inst hidden feat, batch inst feat -> batch inst hidden"` for h.

**Tests first.** `test_tied_model_matches_reference` (against a per-instance loop; the loop is
fine in a test), `test_linear_model_is_wtw_x_plus_b`, `test_gram_is_wt_w`,
`test_relu_hidden_has_no_hidden_bias`, `test_bias_false_is_a_fixed_zero_buffer`,
`test_xavier_scale_and_independence_from_instance_count`, `test_biases_initialize_to_zero`,
`test_abs_model_shapes_target_and_forward`, `test_activation_registry`,
`test_from_kind_rejects_unknown_kind`.

**Pitfall: Xavier on a 3-D tensor.** `nn.init.xavier_normal_` on an `(inst, m, n)` tensor counts
the instance dimension in the fan, so the scale silently changes with the number of instances.
SPEC §3 warned about it before any code existed; the repository draws each instance with its own
generator seeded by `derive_seed(seed, INIT_STREAM, param_key, k)`, and the Xavier test checks
both the scale (200 instances) and that the first 3 instances of a 9-instance model equal a
3-instance model. **[Documented: SPEC §3; models.py docstring; test_models.py]** No evidence
shows this bug was ever hit here: it was prevented, not fixed. **[Documented: absence from the
phase reports and DECISIONS]**

**Check.**

```bash
uv run pytest tests/unit/test_models.py -q
uv run pytest -c rebuild/pytest.ini rebuild/tests/test_models.py -q
```

Expected: `12 passed` (the kit: once M2 is done). Answer key: `src/superposition/models.py`.

### 3.4 `train`: the multi-instance trainer (rebuild kit M3)

**Responsibilities.** Train every instance with Adam on fresh batches; log; snapshot; evaluate
on fresh samples; pick the best restart per condition.

**Interface.**

- `importance_weighted_loss(output, target, importance) -> Float[Tensor, " inst"]`.
- `train(model, dist, importance, steps=10_000, batch_size=1024, lr=1e-3, schedule="constant",
  l1_hidden=0.0, snapshot_every=None, log_every=100, *, eval_samples=20_000, warmup_frac=0.05,
  seed=0, progress=False) -> TrainResult`.
- `TrainResult`: `log_steps`, `loss_history: (log, inst)`, `final_loss: (inst,)`, `steps`,
  `eval_samples`, `wall_time_s`, `snapshot_steps`, `snapshots`.
- `evaluate_loss(model, dist, importance, n_samples, generator) -> Float[Tensor, " inst"]`
  (chunked to bound memory).
- `select_best(result_or_losses, groups) -> Int[Tensor, " group"]`: lowest loss per condition
  (p. 23), never NaN unless a whole group is NaN.
- `lr_multiplier(schedule, steps, warmup_frac)`; `train_from_config(model, dist, importance, config, seed)`.

**Math (p. 10).** L = Σₓ Σᵢ Iᵢ (xᵢ − x′ᵢ)², in three steps (SPEC §3):

1. sum the importance-weighted squared error over features;
2. mean over the batch, giving one loss per instance;
3. **sum** over instances for `backward()`.

An instance's parameters appear only in its own loss term, so summing leaves each instance's
gradient exactly what it would be alone; Adam's update is elementwise, so instances never
interact. With an L1 penalty (Approach 1, p. 42) the objective adds λ·mean over the batch of
Σ|hⱼ|. Data come from `derive_seed(seed, DATA_STREAM)`, evaluation from `EVAL_STREAM`.

**The closed-form check (p. 13, footnote 12).** The linear model's optimum represents the m most
important features exactly and predicts the mean p/2 for the rest:
L* = Σ_{i≥m} Iᵢ (p/3 − p²/4). For n = 5, m = 2, Iᵢ = 0.9ⁱ it gives 0.183, 0.124, 0.068 at
p = 1, 0.2, 0.1. It was derived in the planning conversation and matched its JAX runs, and the
session checked the values again at 08:29Z. **[Documented: SPEC §7 E00; Planning conversation,
per owner; S 35cab3f0 @ 08:29Z]** `theory.linear_optimum_loss` implements it; derive it
yourself (rebuild kit M5).

**Tests first.** `test_importance_weighted_loss_reduction_order`,
`test_instances_are_independent`, `test_training_reduces_loss_and_logs`,
`test_training_is_deterministic_given_seed`, `test_snapshots`,
`test_l1_hidden_shrinks_hidden_activations`, `test_abs_model_trains`, `test_lr_schedules`,
`test_select_best`, `test_evaluation_reproduces_linear_optimum` (three values of p),
`test_evaluate_loss_handles_partial_chunks`.

**Pitfalls hit here.**

- **An arbitrary test margin.** `test_l1_hidden_shrinks_hidden_activations` first asserted
  that λ = 0.5 cuts mean hidden activity below 0.5× the unpenalized run. The margin was relaxed
  to 0.8× with an inline comment recording the observed ratio (≈ 0.62 after 600 steps).
  **[Documented: S 35cab3f0 @ 08:33Z; tests/unit/test_train.py]** That the 0.5 version failed is
  inferred: an observed 0.62 cannot pass `< 0.5`. **[Reconstructed: from the comment's observed
  ratio]** The 0.5 came from no paper page; the session called it arbitrary. **[Documented:
  S 35cab3f0 @ 08:33Z]** Relaxing such a margin openly, with the observed value written next to
  it, is fine; a paper-derived criterion would need a DECISIONS entry instead (CLAUDE.md,
  "Research integrity"). **[Recommended]**
- **Selecting restarts on a noisy loss.** `select_best` trusts `final_loss`, a Monte-Carlo
  estimate. With 20,000 samples its noise can match the spread between restarts. The review
  found it in E01 (R1-06: on more samples another restart wins in 7 of 14 conditions; counts
  shift by about ±2), E03 (R2-10) and E08 (R3-06); the first two are corrected in the docs and
  re-runs remain open. **[Documented: [REVIEW.md](../REVIEW.md) R1-06, R2-10, R3-06]** E00 picks
  and compares on the same 100,000 samples (D-106); a 20,000-sample estimate of the pentagon's
  loss has about 3.7% noise. **[Documented: D-106; phase-1.md open issue 1]** Best-of-8 linear
  losses even land slightly *below* the exact optimum: selection on noise.
  **[Documented: phase-1.md "What the owner should look at", question 2;
  run `20261001T071529Z-5eeb804c`]**
- **Paired seeds.** E00 and E01 train linear and ReLU grids from the same seed, so each ReLU
  instance starts from its linear twin's W and sees the same data. **[Documented: D-103]**

**Check.**

```bash
uv run pytest tests/unit/test_train.py -q
uv run pytest -c rebuild/pytest.ini rebuild/tests/test_train.py -q
```

Expected: `13 passed` (the kit: once M1–M3 are done). Answer key: `src/superposition/train.py`.

### 3.5 `metrics`: superposition and geometry (rebuild kit M4)

**Responsibilities.** Pure functions of W (and, for two functions, of a model run on sampled
inputs from an explicit generator) that turn weights into the paper's quantities.

**Interface.** Every function takes `W: Float[Tensor, "inst hidden feat"]`.

| Function | Returns | Paper |
|---|---|---|
| `gram(W)`, `feature_norms(W)` | `(inst, feat, feat)`, `(inst, feat)` | p. 11 |
| `superposition_score(W)` | `(inst, feat)` | p. 11 |
| `feature_dimensionality(W)` | Dᵢ, `(inst, feat)` | p. 18 |
| `dims_per_feature(W)` | D*, `(inst,)` | p. 17 |
| `represented_mask(W, threshold=0.5)` | `Bool (inst, feat)` | p. 11 |
| `is_superposition(W, threshold=0.5, rtol=1e-2)` | `Bool (inst,)` | p. 8 |
| `angles_2d(W)`, `canonicalize_2d(W, threshold=0.5)` | `(inst, feat)`, `(inst, 2, feat)` | p. 22 |
| `geometry_graph`, `component_geometry`, `geometry_census`, `classify_fraction(D, tol=0.02)` | graph, tegum factors, nearest fraction | pp. 18–20 |
| `neuron_polysemanticity` | features per neuron | pp. 31–33 |
| `loss_by_num_active(model, k, importance, n_samples, generator, low=0.0, high=1.0)` | Lₖ, `(inst,)` | p. 13 |
| `per_feature_loss` | per-feature loss | pp. 35–38 |

**Math.**

- Represented: ‖Wᵢ‖ > 0.5 (SPEC §3, configurable).
- Superposition score: Σ_{j≠i} (Ŵᵢ·Wⱼ)², with Ŵᵢ = Wᵢ/‖Wᵢ‖. It is 0 for a feature orthogonal
  to all others and ≥ 1 when others activate it as strongly as itself (p. 11).
- Feature dimensionality: Dᵢ = ‖Wᵢ‖² / Σⱼ (Ŵᵢ·Wⱼ)² (p. 18). The j = i term equals ‖Wᵢ‖², so
  Dᵢ ≤ 1. Define Dᵢ = 0 when Wᵢ = 0 (SPEC §3).
- Dimensions per feature: D* = m / ‖W‖²_F (p. 17).
- Characteristic fractions (p. 19): 1 dedicated, 3/4 tetrahedron, 2/3 triangle, 1/2 antipodal
  pair, 2/5 pentagon, 3/8 square antiprism, 0 not learned.
- Formal superposition (p. 8): WᵀW is not invertible. Apply it to the represented set R only:
  rank(W_R) < |R|, with numerical rank at `rtol` × the largest singular value (D-110). Why
  restricted: for any n > m, the full n × n WᵀW has rank ≤ m, so the unrestricted test would
  call a linear model that simply drops n − m features "in superposition" (phase-0.md,
  question 3).
- Canonical 2D orientation (p. 22, D-109): rotate the most important represented feature to +y;
  reflect so the first off-axis feature (represented first, by index) has x < 0.
- k-sparse terms (p. 13): L = Σₖ pᵏ (1 − p)ⁿ⁻ᵏ Lₖ, where Lₖ sums the expected loss over all sets
  of exactly k active features. `loss_by_num_active` estimates Lₖ = C(n, k)·E_A[E[loss | A]];
  L₀ = Σᵢ Iᵢ ReLU(bᵢ)² exactly for the ReLU output model.

**Tests first.** Known answers (SPEC §6 table): `test_pentagon` (every Dᵢ = 2/5, D* = 0.4),
`test_antipodal_pair` (1/2), `test_equilateral_triangle` (2/3), `test_regular_tetrahedron`
(3/4), `test_orthonormal_basis` (1), `test_zero_column` (0), `test_regular_simplex` ((n − 1)/n
for n = 2 … 13). Definitions: `test_feature_norms_and_gram`,
`test_superposition_score_matches_definition`, `test_represented_mask_threshold`,
`test_is_superposition`, `test_classify_fraction`, `test_angles_and_canonicalize_2d`. Graph:
`test_geometry_graph_components`, `test_component_geometry_names_the_papers_polytopes`,
`test_geometry_census_splits_tegum_products`. Neurons: `test_neuron_polysemanticity`. p. 13:
`test_loss_by_num_active_recombines_to_total_loss`. Properties (hypothesis):
`test_unit_norm_tight_frames_have_uniform_dimensionality` (every Dᵢ = m/n, Σ Dᵢ = m) and
`test_metrics_are_rotation_invariant`.

Build each known answer by hand as a `(1, hidden, feat)` tensor (a regular k-gon, the
tetrahedron's vertices over √3, the standard basis projected off (1, …, 1)). Writing these
fixtures is half the exercise: it forces you to know what each geometry looks like.
**[Recommended]**

**Pitfalls hit here.**

- **A broken test generator, not a broken metric.** The tight-frame property test first failed
  with a hypothesis strategy error at 08:31Z; the failure was in the strategy that builds the
  frames. **[Documented: S 35cab3f0 @ 08:31–08:32Z]** What exactly was changed is unknown. When a
  property test fails, check the generator before the code under test. **[Recommended]**
- **Typing a sum of tensors.** pyright rejected the built-in `sum` over tensors in the
  recombination test; it became `torch.stack([...]).sum(0)`. **[Documented: S 35cab3f0 @
  08:42Z]**
- **Dᵢ alone cannot name a geometry.** 2/5 and 3/8 are 0.025 apart, so with a ±0.02 tolerance
  one band of Dᵢ near 0.39 counted as both a pentagon and a square antiprism cluster in E03. The
  review caught it (R2-01) and D-603 counts a feature only if its tegum factor is that
  fraction's polytope; E03 went from PASS to PARTIAL. **[Documented: D-603; D-206 caveat;
  REVIEW R2-01]** The thresholds of D-110 were provisional by design. **[Documented: D-110]**

**Check.**

```bash
uv run pytest tests/unit/test_metrics.py -q
uv run pytest -c rebuild/pytest.ini rebuild/tests/test_metrics.py -q
```

Expected: `25 passed` (the kit: once M1–M4 are done). Then run the whole kit:

```bash
make rebuild-test
```

Expected as shipped: `64 failed`, all `NotImplementedError`, and `make` reports an error. After
M1–M4: `64 passed`, and you are ready for M5, E00 by hand ([E00 card](experiments/E00.md)).
**[Documented: rebuild/README.md]** Answer key: `src/superposition/metrics.py`.

### 3.6 `theory`: the linear optimum and the loss decompositions

**Responsibilities.** Exact expected losses, with nothing sampled, to check training against
and to build E02's theoretical phase diagrams.

**Interface (the parts to build first).**

- `uniform_feature_variance(p)` = p/3 − p²/4; `linear_optimum_loss(importance, n_hidden, p)`.
- `linear_loss_decomposition(W, b, importance, p) -> LinearLossTerms` (`feature_benefit`,
  `interference`, `bias`, each `(inst,)`; `interference_matrix` `(inst, feat, feat)`; `.total`).
- `one_sparse_decomposition(W, b, importance) -> OneSparseTerms` (`zero` = L₀, `benefit`
  `(inst, feat)`, `interference` `(inst, feat, feat)`, `.one` = L₁); `k_sparse_weights(n, p)`.
- Later, for E02: `relu_square_integral`, `relu_output_expected_loss`, `optimize_family`,
  `theoretical_phase_diagram`.

**Math (p. 13).** For the linear model with independent features of variance vⱼ and mean μⱼ,
the error of output i has

E[eᵢ²] = vᵢ (1 − ‖Wᵢ‖²)² + Σ_{j≠i} vⱼ (Wᵢ·Wⱼ)² + (μᵢ − Σⱼ (WᵀW)ᵢⱼ μⱼ − bᵢ)².

The first term is the paper's *feature benefit*, the second its *interference*; the paper
states them for unit-variance Gaussian inputs (footnote 10), and the variances here are
p/3 − p²/4. The optimal b zeroes the third term. For the ReLU output model,
L₀ = Σᵢ Iᵢ ReLU(bᵢ)² and

L₁ = Σᵢ ∫ Iᵢ (xᵢ − ReLU(‖Wᵢ‖² xᵢ + bᵢ))² dxᵢ + Σ_{i≠j} ∫ Iⱼ ReLU(Wⱼ·Wᵢ xᵢ + bⱼ)² dxᵢ.

Negative interference is free on 1-sparse inputs, and a negative bias turns small positive
interference into negative (p. 13). Every ReLU term reduces to one double integral
F(γ, α, β, c) = ∫₀¹∫₀¹ (γx − ReLU(αx + βy + c))² dy dx, computed in closed form: y analytically,
then a piecewise cubic in x by 3-point Gauss–Legendre, exact to degree 5 (D-301).

**Tests first.** `test_evaluation_reproduces_linear_optimum` (in `test_train.py`),
`test_linear_decomposition_matches_monte_carlo`,
`test_linear_decomposition_terms_on_known_geometries`, `test_one_sparse_terms_match_monte_carlo`,
`test_negative_interference_is_free_on_one_sparse_inputs`,
`test_k_sparse_weights_cover_every_sparsity_pattern`,
`test_relu_square_integral_matches_adaptive_quadrature`,
`test_exact_relu_output_loss_matches_monte_carlo`,
`test_exact_loss_recombines_from_k_sparse_terms`. The pattern: every exact formula is tested
against an independent Monte-Carlo estimate from `train.evaluate_loss` or
`metrics.loss_by_num_active`, and the integral against scipy's adaptive `quad`.

**What happened.** Only the linear optimum was built in Phase 1 (`4558329`). The decompositions
and phase-diagram machinery were drafted in Phase 3, committed from outside the session in
`4355d5d` and `acec078`, and their decisions (D-301, D-302) written up on 09-30. **[Documented:
git; DECISIONS "Phase 3" preamble; [01_timeline.md](01_timeline.md)]** The closed form was
chosen over quadrature because it is exact, vectorizes over a whole phase diagram and has no
tolerance to tune; during development it agreed with Monte Carlo in a 20-seed check.
**[Documented: D-301]**

**Check.**

```bash
uv run pytest tests/unit/test_theory.py -q
```

Expected: `40 passed` (the slowest file in this chapter's checks). Answer key: the functions
named above in `src/superposition/theory.py`.

### 3.7 `runs` and `acceptance`: provenance and criteria

**Responsibilities.** Give every run a directory and a manifest that say exactly what ran; give
every acceptance criterion a value, a threshold, a page and a status.

**Interface.**

- `results_root()`: `$SUPERPOSITION_RESULTS`, else `<project root>/results` (D-111).
- `create_run(experiment, config, device, seed) -> RunDir` (ID = UTC timestamp + config hash;
  writes `config.yaml` and a `running` manifest: `git_info()`, packages, platform, command,
  start time); `finish_run(run, status="complete", error=None)` adds end time and `wall_clock_s`.
- `git_info(root=None) -> {"commit", "dirty"}`; `list_runs`, `latest_run`, `resolve_run`.
- `acceptance.Criterion(id, description, paper_page, status, value, threshold, group, notes)`;
  constructors `check(...)` (PASS/FAIL), `graded(...)` (PASS/PARTIAL/FAIL), `info(...)` (INFO,
  never graded); `overall_status(criteria)`; `acceptance_document(...)`.

**Pitfalls hit here.**

- **`git.dirty` is recorded when a run starts.** `create_run` calls `git_info()` before training,
  so an edit to any tracked file at that moment makes the run uncitable, even if the edit is
  undone a second later; phase-3.md warns that VS Code's autosave can do exactly that.
  **[Documented: runs.py; CLAUDE.md "Conventions"; phase-3.md open issue 8]** The response was to
  launch cited runs from a clean detached `git worktree` with its own `uv sync` and
  `SUPERPOSITION_RESULTS` pointing at the main `results/`, first in session 92884087 at 17:58Z.
  **[Documented: S 92884087 @ 17:58Z; manifests' `command` fields]**
- **Untracked code did not count as dirty.** `git_info` ignored untracked files, so a run that
  imported a module not yet committed was recorded as clean (review R1-10). D-606 (`450938e`)
  counts untracked files under `src/` and `configs/`. No cited run was affected.
  **[Documented: D-606; REVIEW R1-10]**
- **Dirty runs in Phase 1.** Before citing anything, session 35cab3f0 moved its dirty-tree runs
  out of `results/` into its scratchpad and re-ran from a clean commit; the earliest manifest
  left in `results/` starts at 09:06Z. **[Documented: S 35cab3f0 @ 09:05Z; manifest of
  `20260929T090615Z-eb7b9e3b`]** What those dirty runs showed is unknown.
- **PASS/FAIL was not enough.** `graded(...)` with PARTIAL was added in `81ad83c` when the
  owner's decision on E11's pentagon needed partial credit (D-R05). **[Documented: `81ad83c`;
  [rectification.md](../progress/rectification.md) item 5; `test_graded_partial_criteria`]**

**Tests first.** `test_run_directory_lifecycle`, `test_resolve_run_names_the_command`,
`test_overall_status_and_jsonable`, `test_graded_partial_criteria`,
`test_untracked_code_makes_the_tree_dirty`, `test_pinning_refuses_dirty_runs`.

**Check.**

```bash
uv run pytest tests/unit/test_runs_and_report.py -q -k "dirty or lifecycle or overall"
```

Expected: `4 passed, 7 deselected`. Answer keys: `src/superposition/runs.py`,
`src/superposition/acceptance.py`.

### 3.8 `experiments/`: base, common, runner, registry

**Responsibilities.** A uniform shape for every experiment, a shared sweep trainer, and one
function that runs an experiment end to end.

**Interface.**

- `base.Experiment(name, code, title, paper_pages, config_cls, run, figures)` (frozen), where
  `run(config, run_dir, device, logger) -> list[Criterion]` must save checkpoints and tables so
  that `figures(run_dir) -> list[Path]` works from the saved run alone.
- `REGISTRY: dict[str, Experiment]` in `experiments/__init__.py`, in E00 … E12 order.
- `runner.run_experiment(name, profile="quick", seed=None, device=None, overrides=None, echo=True,
  figures=True) -> RunDir`: load config, resolve device, `seed_everything`, `create_run`, call
  `exp.run`, write `acceptance.json`, draw figures, `finish_run` (marked `failed` on any
  exception). `publish_figures` refuses files over 500 KB.
- `common.train_sparsity_grid(kind, n_features, n_hidden, importance, feature_probabilities,
  n_restarts, train_config, seed, device) -> GridResult`: conditions × restarts instances in one
  model, each with its own feature probability; best restart via `select_best`.

**Pitfalls hit here.** E00's re-run at 08:42Z went through `uv run python -c`, not the CLI,
after `results/` was wiped with `rm -rf`. **[Documented: S 35cab3f0 @ 08:42Z]** The
S = 0.9 pentagon loss sat about 9% above SPEC's reference; it was first studied with scratchpad
scripts, then rebuilt as report-only diagnostics inside E00 (D-107), so the evidence lives in a
logged run. **[Documented: S 35cab3f0 @ 08:39–08:41Z; D-107; phase-1.md open issue 1]** Configs
ship inside the wheel so a pip install can run experiments (D-111).

**Tests first.** Acceptance logic on hand-built geometries, with no training:
`test_e00_ideal_solutions_pass`, `test_e00_broken_solutions_fail`, `test_pentagon_helpers`,
`test_e01_superposition_order_criterion`. A criterion that passes on an ideal pentagon and fails
on a broken one is tested before any run can flatter it. **[Recommended]**

**Check.**

```bash
uv run pytest tests/unit/test_experiment_criteria.py -q
uv run superposition list
```

Expected: `4 passed`; thirteen experiments. The E00 run itself is in its recipe card
([experiments/E00.md](experiments/E00.md)). Answer keys: `src/superposition/experiments/`
(`base.py`, `common.py`, `runner.py`, `__init__.py`, then `intro_pentagon.py`).

### 3.9 `report` and `cli`

**Responsibilities.** The commands you run, and the tables in docs/REPRODUCTION.md generated
from cited runs rather than typed by hand.

**Interface.** Typer commands `list [--names]`, `run <exp> [--profile] [--seed] [--device]
[--no-figures]`, `figures <exp> [--run latest|ID] [--publish]`, `report [--pin EXP | --pin-all]
[--run ID --label L] [--allow-dirty]`, `app`. In `report`: `pin_run`, `load_citations`,
`save_citations` (`docs/cited_runs.yaml`), `status_table`, `criteria_table`, `replace_block`,
`update_reproduction`.

**What happened.** `report` and `figures --publish` moved from Phase 2 into Phase 1 so Phase 1's
docs came from the real commands (D-108). Phase 1's `report` cited the latest run, so a later
notebook run could silently swap the tables (phase-1.md open issue 3); Phase 2 added pins in
`docs/cited_runs.yaml` that refuse dirty runs (D-202), and Typer's option defaults tripped ruff
`B008` on the way. **[Documented: D-108; D-202; phase-1.md; S 35cab3f0 @ 12:36Z]**

**Tests first.** `test_report_blocks`, `test_cli_list_figures_and_app`,
`test_pinning_and_regenerating_reproduction`, `test_supplementary_citations`,
`test_cli_report_label_needs_one_pin`, `test_pinning_refuses_dirty_runs`.

**Check.**

```bash
uv run pytest tests/unit/test_runs_and_report.py -q
uv run superposition report --help
```

Expected: `11 passed`; the five `report` options. Never run `report --pin` as a check: it rewrites
tracked docs. Answer keys: `src/superposition/cli.py`, `src/superposition/report.py`.

### 3.10 `plotting/` (lighter)

**Responsibilities.** Paper-style figures from arrays. Plotting never computes metrics, and
metrics never plot (SPEC §5).

**Interface.** `style.paper_style()`, `style.save_figure(...)`, `weight_norm`,
`superposition_norm`; `weights.weights_grid`, `plot_gram`, `plot_bias`, `plot_feature_bars`
(WᵀW and b heatmaps, ‖Wᵢ‖ bars colored by superposition score, p. 11);
`arrows.arrows_grid`, `plot_feature_arrows`, `importance_colors` (canonicalized 2D arrows, p. 2).

**What happened.** The palette was checked with a palette-validation script; pyright errors
were fixed by importing `Line2D` directly; layout constants were adjusted after looking at E00's
first figures. **[Documented: S 35cab3f0 @ 08:34Z, 08:42Z, 08:44Z]** No file in `tests/unit`
imports `superposition.plotting`, and the replication tests call `run_experiment(...,
figures=False)`. In the test suite, figures are drawn only by the lab pages that AppTest runs,
which is how the Agg problem surfaced (D-205; [04_scaffolding.md](04_scaffolding.md) A7).
**[Documented: grep of tests/unit and tests/replication; D-205]** Give your own plotting
functions at least a smoke test that draws and saves one figure. **[Recommended]**

**Check.** Draw your M5 pentagon with your own `canonicalize_2d` and compare it with
[the published E00 arrows](../figures/intro_pentagon/pentagon_arrows.png). Answer key:
`src/superposition/plotting/`.
