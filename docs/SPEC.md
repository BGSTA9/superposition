# SPEC.md: `superposition`

A research-grade, independent replication of *Toy Models of Superposition* (Elhage et al., 2022), with an interactive lab and mechanistic-interpretability tooling.

Version 1.0 · 2026-09-29

> **How to use this document.** This is the complete specification and the single source of truth for building the `superposition` repository. Read it in full before writing any code, then read the paper at `references/toy_models_of_superposition.pdf`. Every page number below refers to that PDF, a 62-page print of the web article at https://transformer-circuits.pub/2022/toy_model/index.html. Keep this file in the repo root. At the start of every phase, re-read the sections that phase touches.

---

## 1. Mission

Build `superposition`, a Python repository that does four things.

1. **Replicates** the paper's experiments faithfully and reproducibly. Every claim is checked against the paper and reported honestly as PASS, PARTIAL or FAIL.
2. **Provides an interactive lab**, a Streamlit app for exploring the toy models: train them live, probe them, and browse precomputed paper-scale results.
3. **Provides mechanistic-interpretability tooling** on these models: reverse-engineering trained networks, ablations, and sparse autoencoders evaluated against known ground truth.
4. **Serves as the owner's learning instrument** and the base for their own extensions.

### Who this is for

The owner is strong in Python, PyTorch and linear algebra, and new to mechanistic interpretability. They will learn the paper by reading this code, running the notebooks and using the app, then extend it. So:

- Optimize for clarity over cleverness. Write explicit code that mirrors the paper's math and notation.
- Every module, class and public function docstring names the paper section and page it implements, e.g. `Paper: Demonstrating Superposition, p. 10`.
- Explain *why* as well as *what*, in docstrings and notebooks.

## 2. Research integrity rules (non-negotiable)

**Provenance**
- Results come only from code in this repo. Never fabricate, estimate or hand-edit numbers or figures.
- Every number or figure in the docs traces to a logged run: config, seed and git commit.

**Honest reporting**
- If a result does not reproduce, work through it in this order:
  1. Check the implementation against the paper.
  2. Increase restarts or steps within reason.
  3. Document what you tried.
  4. Report PARTIAL or FAIL with evidence and hypotheses.
- Do not weaken an acceptance criterion to make it pass. Any change to a criterion needs a written justification in `docs/DECISIONS.md`.
- Tests verify correctness; they don't define it. Never special-case code to satisfy a test. Never hard-code expected outputs into implementation code.

**Decisions and defaults**
- Where the paper under-specifies something (step counts, grids, schedules, sizes):
  - choose a principled default and make it configurable;
  - record it in `docs/DECISIONS.md` with the decision, rationale, paper page and alternatives considered.
- Keep paper-faithful defaults separate from variants. Extensions never change a replication's default configuration.

**Independence**
- Implement from the paper. Two reference implementations exist:
  - Anthropic's archived notebook, https://github.com/anthropics/toy-models-of-superposition
  - ARENA §1.5.4, https://learn.arena.education/chapter1_transformer_interp/34_superposition/
- Do not copy code from either. You may consult them only to diagnose a failed replication after an honest independent attempt, and you must record that consultation in `docs/DECISIONS.md`.

**Completion**
- A phase is done only when its exit criteria have actually been run and verified.

**Licensing**
- Never commit the paper PDF: git-ignore `references/*.pdf` and link to the web version instead.

## 3. Paper facts and conventions (implement exactly)

### Notation

| Symbol | Meaning | Name in code |
|---|---|---|
| n | number of features | `n_features` |
| m | number of hidden dimensions | `n_hidden` |
| W ∈ ℝ^{m×n} | weights; column Wᵢ is feature i's direction, and Ŵᵢ = Wᵢ/‖Wᵢ‖ | `W` |
| x ∈ ℝⁿ | input features | `x` |
| h ∈ ℝ^m | hidden activations | `h` |
| b ∈ ℝⁿ | output bias | `b` |
| I | importance | `importance` |
| 1 − S | feature probability | `feature_probability` |

### Data

- **Independent features (p. 9).** Each feature xᵢ is 0 with probability Sᵢ, otherwise uniform on [0, 1]. The default is Sᵢ = S for all i.
- **Parameterization.** Code uses `feature_probability` p = 1 − S. Plots use 1 − S or 1/(1 − S), as the paper does.
- **Importance.** Iᵢ multiplies each feature's squared error (footnote 1).
- **Correlated feature set (p. 23).** One shared draw decides whether the whole set is zero. Nonzero values are drawn independently.
- **Anticorrelated feature set (p. 23).** The whole set is zero with probability S. Otherwise exactly one uniformly chosen member is active, with a U[0, 1] value.
- **Absolute-value task (p. 35).** Nonzero values are uniform on [−1, 1], and the target is y = |x|.

### Models

| Model | Page | Hidden layer | Output |
|---|---|---|---|
| Linear | p. 10 | h = Wx | x′ = Wᵀh + b = WᵀWx + b |
| ReLU output (tied: the decoder is Wᵀ) | p. 10 | h = Wx | x′ = ReLU(WᵀWx + b) |
| ReLU hidden layer | p. 31 | h = ReLU(Wx), no hidden bias | x′ = ReLU(Wᵀh + b) |
| Absolute value (W₁, W₂ untied) | p. 35 | h = ReLU(W₁x) | y′ = ReLU(W₂h + b) |

- The ReLU hidden layer model has no hidden bias because, given one, the model shifts its neurons into the linear regime and sidesteps the ReLU (p. 34).
- The phase-change section writes ReLU(WᵀWx − b) (p. 14). Use +b everywhere and note the sign convention where it matters.

### Loss (p. 10)

L = Σₓ Σᵢ Iᵢ (xᵢ − x′ᵢ)². Implement it in three steps:
1. Sum over features.
2. Take the mean over the batch, giving one value per instance.
3. Sum over instances for `backward()`.

### Metrics

- **Feature norm ‖Wᵢ‖** (p. 11). It is about 1 if the feature is represented and about 0 if not. The default "represented" threshold is ‖Wᵢ‖ > 0.5 (configurable).
- **Superposition score** Σ_{j≠i} (Ŵᵢ·Wⱼ)² (p. 11). It is 0 when feature i is orthogonal to all others, and ≥ 1 when other features can activate i as strongly as i itself.
- **Dimensions per feature** D* = m/‖W‖²_F (p. 17).
- **Feature dimensionality** Dᵢ = ‖Wᵢ‖² / Σⱼ (Ŵᵢ·Wⱼ)² (p. 18). Define Dᵢ = 0 when Wᵢ = 0.
- **Characteristic fractions** (p. 19):

  | Dᵢ | Geometry |
  |---|---|
  | 3/4 | tetrahedron |
  | 2/3 | triangle |
  | 1/2 | antipodal pair (digon) |
  | 2/5 | pentagon |
  | 3/8 | square antiprism |
  | 0 | not learned |

  In the dense regime features are not basis-aligned, so no clean line at 1 appears (p. 19).
- **Formal superposition** (p. 8): WᵀW is not invertible. Apply this test to represented features only.
- **Feature geometry graph** (p. 18): nodes are features, and edge weights are |Wᵢ·Wⱼ|. Connected components correspond to tegum factors (p. 20).

### Theory

- **Linear model loss** (p. 13): L ~ Σᵢ Iᵢ(1 − ‖Wᵢ‖²)² + Σ_{i≠j} Iⱼ(Wⱼ·Wᵢ)². The first sum is the feature benefit; the second is the interference.
- **Sparsity decomposition** (p. 13): the loss splits into terms Lₖ for k-sparse inputs.
  - L₀ penalizes positive biases.
  - The 1-sparse term is L₁ = Σᵢ ∫ Iᵢ(xᵢ − ReLU(‖Wᵢ‖²xᵢ + bᵢ))² + Σ_{i≠j} ∫ Iⱼ ReLU(Wⱼ·Wᵢ xᵢ + bⱼ)².
- For 1-sparse inputs, negative interference is free, and a negative bias converts small positive interference into negative interference (p. 13).

### Optimization notes from the paper

- 2D hidden spaces are hard for gradient descent. The authors fit each model several times and kept the lowest loss (p. 23).
- ReLU hidden layer figures use 1,000 models per setting and show the lowest-loss one (p. 32).
- Phase diagrams use 10 models per grid point, discard the highest-loss model, and average the rest (p. 14).
- Canonicalize 2D solutions by rotation and reflection before comparing them (p. 22).
- Losses of models trained at different sparsities are not comparable, because they measure different tasks (footnote 17, p. 56).

### Initialization

- Use Xavier normal per instance, with std √(2/(n + m)). Initialize biases to zero.
- Do **not** call `nn.init.xavier_normal_` on a 3-D `(n_inst, m, n)` tensor. PyTorch would treat the instance dimension as fan-out, so the scale would silently change with the number of instances.
- Make each instance's initialization depend only on (global seed, instance index), so adding instances never changes existing ones.

### Importance curves in the paper

| Experiment | Importance | Page |
|---|---|---|
| Intro figure | not stated; use Iᵢ = 0.9ⁱ (verified to reproduce p. 2, see E00) | p. 2 |
| n = 20, m = 5 | 0.7ⁱ | p. 11 |
| n = 80, m = 20 | 0.9ⁱ | p. 12 |
| Uniform geometry, n = 400, m = 30 | Iᵢ = 1 | p. 17 |
| ReLU hidden layer, n = 10, m = 5 | 0.75ⁱ | p. 32 |
| Absolute value, n = 100, m = 40 | 0.8ⁱ | p. 36 |

### Visualization conventions

- **WᵀW heatmap** with a diverging colormap centered at 0, with the bias vector alongside (p. 10).
- **Per-feature bars** of ‖Wᵢ‖, colored by superposition score from dark (orthogonal) to bright (≥ 1) (p. 11).
- Features are ordered from most to least important.
- **Absolute-value model** (footnote 15, p. 56): rescale each hidden unit so its largest-magnitude W₁ weight is 1, and apply the inverse scale to W₂. The function is unchanged.

## 4. Repository layout

```
superposition/
├── SPEC.md                     # this file
├── CLAUDE.md                   # concise agent guide: commands, layout, conventions, gotchas
├── README.md
├── pyproject.toml, uv.lock
├── LICENSE (MIT), CITATION.cff, CHANGELOG.md
├── Makefile                    # setup, lint, test, test-slow, run-quick, figures, report, app
├── .pre-commit-config.yaml, .github/workflows/ci.yml, .gitignore
├── references/                 # README.md with URL and BibTeX (p. 55); PDF is git-ignored
├── configs/                    # one YAML per experiment, each with `quick` and `full` profiles
├── src/superposition/
│   ├── config.py               # typed dataclass configs, YAML IO, profiles
│   ├── data.py                 # feature distributions
│   ├── models.py               # model variants, all with a leading instance dimension
│   ├── train.py                # vectorized multi-instance trainer, restarts, snapshots
│   ├── metrics.py              # superposition and geometry metrics (pure functions)
│   ├── theory.py               # loss decompositions, analytic phase diagrams
│   ├── adversarial.py
│   ├── interp.py               # neuron roles, ablations, motif detection, rebuild tests
│   ├── sae.py                  # sparse autoencoders, ground-truth recovery metrics
│   ├── runs.py                 # run directories, manifests, checkpoints, latest-run lookup
│   ├── report.py               # builds docs/REPRODUCTION.md tables from acceptance results
│   ├── plotting/               # paper-style figures, one module per figure family
│   ├── experiments/            # one module per experiment, plus a registry
│   └── cli.py                  # the `superposition` command (Typer)
├── app/                        # Streamlit multipage lab
├── notebooks/                  # the learning path, one notebook per paper section
├── tests/
│   ├── unit/                   # fast and deterministic (CI)
│   └── replication/            # slow, marked @pytest.mark.slow
├── docs/
│   ├── REPRODUCTION.md, PAPER_MAP.md, DESIGN.md, DECISIONS.md
│   ├── STUDY_GUIDE.md, GLOSSARY.md, REVIEW.md
│   ├── progress/               # one report per phase
│   └── figures/                # small published figures referenced by the docs
└── results/                    # run outputs (git-ignored)
```

## 5. Engineering standards

**Environment and dependencies**
- Python ≥ 3.11, managed with `uv`; commit the lockfile.
- Core dependencies: torch, numpy, einops, jaxtyping, matplotlib, plotly, pandas, pyarrow, networkx, scipy, pyyaml, typer, tqdm, streamlit.
- Dev dependencies: pytest, pytest-xdist, hypothesis, ruff, pyright, pre-commit, nbmake, nbstripout.
- Add nothing heavier (no Hydra, no Lightning) without a DECISIONS.md entry.

**Compute**
- CPU-first. `--device auto` picks CUDA, then MPS, then CPU.
- Every `quick` profile finishes in ≤ 10 minutes on a laptop CPU.
- Every `full` profile documents its expected runtime and hardware.

**Code**
- Type hints on all public functions.
- Annotate tensor shapes with jaxtyping, e.g. `Float[Tensor, "inst hidden feat"]`.
- Do all multi-instance math with `einsum`/`einops` over named dimensions. No Python loops over instances or features in hot paths.
- Keep computation and plotting separate. Metrics are pure functions of tensors; plotting functions take arrays and return figures.
- Configs are typed dataclasses serialized to YAML, with no magic numbers in code.
- One `seed_everything(seed)` utility, plus per-instance initialization seeding as in §3.

**Run artifacts** go in `results/<experiment>/<run_id>/`, where `run_id` is a UTC timestamp plus a short config hash. Each run directory contains:
- `config.yaml`: the resolved config
- `manifest.json`: git commit and dirty flag, package versions, device, seed, start and end times, wall-clock
- `metrics.parquet`: per-instance metrics
- `acceptance.json`: each criterion with its value, threshold and status
- checkpoints (`.pt`)
- figures (PNG and PDF)
- `log.txt`

**CLI**

| Command | What it does |
|---|---|
| `superposition list` | list experiments |
| `superposition run <exp> [--profile quick\|full] [--seed N] [--device auto]` | run an experiment |
| `superposition figures <exp> [--run latest] [--publish]` | make figures; `--publish` copies them to `docs/figures/<exp>/` |
| `superposition report` | regenerate the tables in `docs/REPRODUCTION.md` from the latest runs |
| `superposition app` | launch the lab |

**Testing tiers**
1. Unit tests: fast (the whole tier under 60 s on CPU), run in CI.
2. Property-based tests (hypothesis) for metrics.
3. Replication tests, marked `slow`: they train small models and check acceptance criteria.
4. Notebook smoke tests with nbmake on the quick profile, marked `slow`.

**CI (GitHub Actions)**
- `ruff check` and `ruff format --check`.
- pyright (basic mode).
- `pytest -m "not slow"` on CPU, using CPU-only torch wheels.
- Optionally, a scheduled workflow for the slow tests.

**Repository hygiene**
- Commits: conventional commits (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:`), one logical change per commit.
- Large files:
  - `results/` and checkpoints are git-ignored;
  - only small published figures (≤ 500 KB each) go in `docs/figures/`.

## 6. Core library requirements

### `data.py`
- A `FeatureDistribution` protocol: `sample(batch_size, generator) -> Float[Tensor, "batch inst feat"]`.
- Implementations:
  - `IndependentSparse`: per-instance and optionally per-feature probabilities. The value range is configurable, so [0, 1] and [−1, 1] share code.
  - `CorrelatedSets`.
  - `AnticorrelatedSets`.
  - `Composite`: concatenates groups along the feature axis, to mix independent, correlated and anticorrelated features.
- Unit tests:
  - Empirical density matches p within 3 standard errors.
  - Nonzero values have the right range and mean.
  - Correlated sets switch on and off together.
  - Anticorrelated sets never have two members active at once.

### `models.py`
- `TiedModel(n_features, n_hidden, n_inst, hidden_act=None|"relu", out_act=None|"relu", bias=True)` covers the linear, ReLU output and ReLU hidden layer models.
  - Parameters: `W: (inst, hidden, feat)` and `b: (inst, feat)`.
  - Methods: `forward`, `hidden(x)`, and `gram()`, which returns `(inst, feat, feat)`.
- `AbsModel(n_features, n_hidden, n_inst)`: `W1: (inst, hidden, feat)`, `W2: (inst, feat, hidden)`, `b: (inst, feat)`.
- Activations are pluggable through a small registry (relu, gelu, silu, softplus, leaky_relu, sigmoid, identity), so extensions can swap them.

### `train.py`
- Signature: `train(model, dist, importance, steps, batch_size, lr, schedule="constant"|"cosine"|"warmup_cosine", l1_hidden=0.0, snapshot_every=None, log_every) -> TrainResult`.
- Defaults (used for E00, and verified): Adam, lr 1e-3, batch 1024, 10,000 steps, constant schedule, a fresh batch every step.
- `TrainResult` holds:
  - the per-instance loss history;
  - optional parameter snapshots, for dynamics;
  - the final per-instance loss, evaluated on a fresh batch of ≥ 20,000 samples.
- `select_best(result, groups)` picks the best restart per experimental condition.
- Instances are independent: each instance's parameters appear only in its own loss term.

### `metrics.py` (pure functions, fully unit-tested)

**Functions**
- `feature_norms`, `superposition_score`, `feature_dimensionality`, `dims_per_feature`, `represented_mask`, `gram`.
- `is_superposition`: rank test on represented features.
- `angles_2d`, `canonicalize_2d`.
- `geometry_graph`, which also returns the tegum-factor components.
- `classify_fraction`: nearest characteristic fraction within a tolerance.
- `neuron_polysemanticity`: features per neuron above a relative-weight threshold.
- `loss_by_num_active(model, k)`: the empirical k-sparse loss terms (p. 13).

**Required unit tests with known answers**

| Configuration | Expected result |
|---|---|
| Regular pentagon of unit vectors in 2D | every Dᵢ = 2/5, and D* = 0.4 |
| Antipodal pair | Dᵢ = 1/2 |
| Equilateral triangle in 2D | Dᵢ = 2/3 |
| Regular tetrahedron in 3D | Dᵢ = 3/4 |
| Orthonormal basis | Dᵢ = 1 |
| Zero column | Dᵢ = 0 |
| n-point regular simplex, from projecting the standard basis onto the complement of (1, …, 1) (p. 21) | Dᵢ = (n − 1)/n |

**Required property tests**
- For any unit-norm tight frame of n vectors in m dimensions, every Dᵢ = m/n and Σᵢ Dᵢ = m.
- Rotating W by any orthogonal matrix leaves every metric unchanged.

### `theory.py`
- Implement the linear-model loss decomposition and the 1-sparse ReLU decomposition (p. 13). Validate each against Monte-Carlo estimates in unit tests.
- Theoretical phase diagrams (pp. 14–15):
  - **n = 2, m = 1:** candidate families W ∝ [1, 0], [0, 1] and [1, −1]. Add McGrath's "confused feature" family W₁ ≈ W₂ (pp. 51–53) as an optional fourth.
  - **n = 3, m = 2:** the four families W ⊥ [0,0,1], [1,0,0], [0,1,1] and [1,1,0] (p. 15). Build each by projecting the standard basis onto the complement of the ignored direction.
  - For each family, minimize the exact expected loss over its free scalars (weight scale, biases). Compute the expectation exactly: sum over sparsity patterns, and integrate each piecewise-quadratic term either in closed form or with adaptive quadrature at tolerance ≤ 1e-8. Record which in DECISIONS.md.
  - Report each cell's winning family and the loss gap.

### `adversarial.py` (p. 29)
- L2 attacks with a norm budget of 0.1 × the average input norm.
- Use analytic optimal attacks per feature, λ(WᵀW)ᵢ/‖(WᵀW)ᵢ‖₂, and take the most harmful.
- Optionally, gradient attacks for comparison, with a note on gradient masking at high sparsity.
- Optionally, adversarial training (p. 30).

### `sae.py`
- A multi-instance sparse autoencoder on the hidden activations h of a frozen trained toy model:
  - ReLU encoder;
  - linear decoder with unit-norm columns, renormalized after every step;
  - loss = MSE + λ·L1;
  - a TopK variant;
  - dead-latent resampling.
- Ground-truth evaluation, which is the reason toy models are valuable (p. 44):
  - for each represented true direction Wᵢ, the max cosine similarity with any decoder column;
  - mean max cosine similarity;
  - fraction recovered at cos > 0.9 and at cos > 0.95;
  - L0, reconstruction MSE, and dead-latent fraction.

### `interp.py`
- **Neuron role table** for `AbsModel`, computed after footnote-15 normalization. Roles:
  - positive-side(i);
  - negative-side(i);
  - asymmetric-superposition(i, j);
  - inhibitor(i);
  - unexplained.
- **Ablation matrix:** for each neuron, the change in each feature's loss when the neuron is zeroed, and when it is mean-ablated.
- **Motif detector** for asymmetric superposition plus inhibition (pp. 38–39). One neuron stores two features with unequal weight magnitudes and has reciprocal output weights. A second neuron inhibits the feature that would otherwise receive positive interference.
- **Rebuild test,** the gold standard of the method:
  1. Construct an idealized model by hand from the inferred roles.
  2. Compare its per-feature loss with the trained model's.
  3. Report the fraction of the trained model's loss reduction (relative to predicting the mean) that the rebuild recovers.

### `plotting/`
Paper-style figures, saved as PNG and PDF, sharing one style module:

| Figure | Page |
|---|---|
| WᵀW heatmap with bias | p. 10 |
| Feature-norm bars colored by superposition score | p. 11 |
| 2D feature arrows colored by importance, canonicalized | p. 2 |
| D* versus 1/(1 − S) on a log axis, with reference lines at 1 and 1/2 | p. 17 |
| Dᵢ scatter versus 1/(1 − S), with lines at the characteristic fractions | p. 19 |
| Feature geometry graphs | p. 19 |
| Empirical and theoretical phase diagrams, side by side | pp. 14–15 |
| Neuron stack plots: each column is a neuron, each block is one weight colored by feature, negative weights sit below the axis | p. 32 |
| Dimensionality over training, with the loss curve | p. 26 |
| 3D trajectories (plotly) | p. 27 |
| Adversarial vulnerability | p. 29 |

## 7. Experiments

Each experiment gets:
- a module in `experiments/`;
- a config with `quick` and `full` profiles;
- published figures;
- an `acceptance.json` from each run;
- a section in `docs/REPRODUCTION.md`;
- a notebook and an app page.

The thresholds below are initial proposals. The integrity rules in §2 govern any change to them.

### E00 intro_pentagon (p. 2)

**Setup:** n = 5, m = 2, Iᵢ = 0.9ⁱ. Run S ∈ {0, 0.8, 0.9} with 8 restarts each, for both the ReLU output and linear models, using the §6 defaults.

**Accept** (ReLU output model, best restart per S):

| S | Criterion |
|---|---|
| 0 | exactly features {0, 1} represented, with \|cos\| < 0.1 |
| 0.8 | exactly features {0, 1, 2, 3} represented, forming two antipodal pairs: each has a partner with cos < −0.95 |
| 0.9 | all 5 represented; sorted angular gaps within 72° ± 10°; all norms > 1; all biases < 0 |

**Accept** (linear model): exactly features {0, 1} represented, and orthogonal, at every S.

**Accept** (both models, best restart): every feature with ‖Wᵢ‖ < 0.05 has a bias within 0.02 of its mean, (1 − S)/2 (p. 10).

**Accept** (linear model, exact): the best linear loss is within 3% of the closed-form optimum.
- The linear optimum is known exactly: represent the top-m features perfectly and predict the mean for the rest.
- Its loss is Σ_{i≥m} Iᵢ (p/3 − p²/4), which is 0.183, 0.124 and 0.068 for p = 1, 0.2 and 0.1.
- Evaluate these final losses on ≥ 100,000 fresh samples.
- Also add this as a unit test of the loss and evaluation code.

**Reference values.** These come from an independent verification run (JAX, identical math and hyperparameters). Use them for sanity checks, not as acceptance criteria. Losses are the per-sample importance-weighted squared error, summed over features.

| S | ReLU best loss | Linear best loss | Linear exact optimum |
|---|---|---|---|
| 0 | 0.182 | 0.182 | 0.183 |
| 0.8 | 0.059 | 0.122 | 0.124 |
| 0.9 | 0.0196 | 0.067 | 0.068 |

Measured losses used 20,000 evaluation samples, so they carry about 1–2% sampling noise.

- Pentagon norms were ≈ 1.08–1.11, and its biases ≈ −0.20 to −0.21.
- At S = 0.9, 3 of 8 ReLU restarts settled for only 4 features.
- At S = 0, 2 of 8 linear restarts had not finished separating features 1 and 2 after 10,000 steps. Best-of-restarts handles this.

### E01 basic_results (pp. 10–12)

**Setup:**
- n = 20, m = 5, Iᵢ = 0.7ⁱ, over 1 − S ∈ {1, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001}.
- Linear versus ReLU output model, with ≥ 4 restarts.
- Also run n = 80, m = 20, Iᵢ = 0.9ⁱ (p. 12).

**Figures:** WᵀW with bias, plus the feature bars, for each sparsity, laid out as on p. 11.

**Accept:**
- Linear model: exactly the top-m features represented, with superposition score < 0.05, at every sparsity. The sparsest settings may need more steps or larger batches; document any change.
- ReLU model:
  - At 1 − S = 1, it matches the linear model.
  - Its represented-feature count never decreases as sparsity rises (tolerance 1), and it exceeds m at the sparsest setting.
  - Superposition appears first among the less important features (p. 11).
  - Antipodal pairs appear at intermediate sparsity; report where.

### E02 phase_change (pp. 14–15)

**Empirical setup** (n = 2, m = 1):
- The extra feature's importance on a log grid over [0.1, 10].
- Feature density 1 − S on a log grid over [0.01, 1]. The paper words this as "sparsity from 1.0 to 0.01"; document your reading.
- 10 models per cell. Discard the highest-loss model and report the majority label of the remaining 9 (p. 14).
- Label each model as one of:
  - extra-feature-not-learned;
  - superposition (antipodal);
  - dedicated-dimension-for-extra-feature;
  - confused (same-sign W₁ ≈ W₂; pp. 51–53).

**Theoretical:** from `theory.py`, for both n = 2, m = 1 and n = 3, m = 2.

**Accept:**
- All three main regions appear in both diagrams.
- Empirical and theoretical labels agree on ≥ 80% of cells.
- Report the disagreements with a map, and relate them to McGrath's blurry triple point and confused-feature regime.

### E03 uniform_geometry (pp. 16–21)

**Setup:** n = 400, m = 30, Iᵢ = 1, with 1/(1 − S) log-spaced over [1, 10] (quick: 20 points; full: ≥ 60) and ≥ 2 restarts.

**Figures:** the D* curve (p. 17), the Dᵢ scatter with fraction lines (p. 19), and geometry graphs at selected sparsities.

**Accept.** Approximate readings from the p. 17 figure:
- D* ≈ 1 up to about 1/(1 − S) = 1.2.
- A plateau at 0.5 ± 0.03 over a contiguous range; the paper's spans roughly [2.6, 3.7]. Report yours.
- D* falls beyond the plateau, to about 0.25 at 1/(1 − S) = 8.
- Nonzero Dᵢ values cluster within ±0.02 of at least three characteristic fractions.
- Report Σᵢ Dᵢ against m.

### E04 nonuniform_geometry (pp. 21–25)

**(a) Perturbing one feature (p. 22).** n = 5, m = 2, I = 1, with 1 − S = 0.05 for four features; sweep the fifth feature's density. Use many restarts, and plot every restart's loss to reveal the crossing loss curves of the competing geometries.
- **Accept:**
  - Denser → the neighbors repel, so neighbor angles grow.
  - Sparser → the neighbors close in.
  - Beyond a threshold → collapse to two digons, with the perturbed feature at about 0.

**(b) Correlated and anticorrelated sets in m = 2 (p. 23).**
- **Accept:**
  - Anticorrelated features show more negative interference (lower mean cosine) than correlated features.
  - Correlated features are orthogonal when capacity allows, and otherwise sit adjacent.

**(c) Local almost-orthogonal bases (p. 24).** Two correlated sets in a larger model, with a WᵀW figure.
- **Accept:** mean |cos| within a set < mean |cos| across sets.

**(d) Collapse to principal components (p. 25).** Six features in three correlated pairs, m = 2, sweeping sparsity.
- **Accept:**
  - Very sparse → a hexagon with pair members adjacent.
  - As density rises → the cosine within each pair rises toward 1: the collapse to (a+b)/√2.

### E05 learning_dynamics (pp. 26–28)

**(a) Energy-level jumps (p. 26).** Use the uniform setup at a sparsity inside the 1/2 plateau, and snapshot every feature's Dᵢ every k steps.
- **Accept:** at least one jump (|ΔDᵢ| > 0.15 within a short window) that coincides with a drop in the loss curve.

**(b) Geometric transformations (p. 27).** n = 6, m = 3, with two sets of 3 correlated features. Record the trajectories and plot them interactively in 3D.
- **Accept:** in at least one restart, the octahedron final state: three orthogonal antipodal pairs, each pairing features from different sets.
- **Report** the trajectory types observed.

### E06 adversarial (pp. 29–30)

**Setup:** sweep sparsity on a setup prone to superposition, e.g. n = 20, m = 5.

**Accept:**
- Vulnerability rises as superposition forms. The paper reports more than 3×; accept more than 2× and report the actual value.
- Spearman correlation ≥ 0.8 between vulnerability and features per dimension, 1/D*.

**Optional:** show that adversarial training removes superposition only at large budgets (p. 30).

### E07 privileged_basis (pp. 31–34)

**Setup:** ReLU hidden layer model, n = 10, m = 5, Iᵢ = 0.75ⁱ, sweeping sparsity, with 1,000 restarts per setting (quick: 100). Visualize the best model as a W heatmap and a neuron stack plot.

**Accept:**
- Dense → every neuron is monosemantic.
- The polysemantic-neuron count never decreases as sparsity rises (tolerance 1), and is > 0 at the sparsest setting.
- Document the model's tendency to sidestep the ReLU (p. 34).

### E08 abs_computation (pp. 34–39)

**(a) n = 3, m = 6, no superposition (p. 36).**
- **Accept:**
  - Each feature gets one positive-side and one negative-side neuron.
  - Every per-feature loss is below 5% of the loss of predicting the mean.

**(b) n = 100, m = 40, Iᵢ = 0.8ⁱ, sweeping sparsity.** Make stack plots of |W| per neuron, with neurons sorted by the importance of their largest feature (p. 36).
- **Accept:**
  - Dense → monosemantic neurons.
  - Sparse → the number of features computed well exceeds m/2, which is the capacity without superposition. A feature counts as computed well when its per-feature loss is below 50% of the mean-prediction baseline.
- **Report** the pattern of neurons with one primary feature plus secondary features (p. 38).

### E09 mech_interp_abs (pp. 38–39)

**Setup:** train absolute-value models in the regime that produces limited superposition, then run `interp.py`.

**Accept:**
- At least 90% of neurons receive a named role.
- Ablation effects agree with the assigned roles.
- The asymmetric-superposition motif is found in at least one model; report the configurations.
- The rebuild test is reported.

### E10 sae_recovery (pp. 42–44)

**Setup:**
- Train SAEs on the hidden activations of trained models in superposition: the E00 pentagon, the sparse E01 models, and one E03 model.
- Sweep dictionary size and the L1 coefficient, with and without resampling.

**Accept:**
- The pentagon's 5 features are recovered at cos > 0.95.
- For the n = 20, m = 5 model, at least 90% of represented features are recovered at cos > 0.9.

### E11 l1_removes_superposition (p. 42)

**Setup:** add λ‖h‖₁ to the loss and sweep λ.

**Accept:** as λ grows, superposition falls to none while the loss rises. Report the trade-off curve.

### E12 activation_functions (extension; p. 51)

Redwood reported that phase diagrams change with the activation function (p. 51). Recompute E02 for relu, gelu, silu, softplus, leaky_relu and sigmoid.

There is no pass/fail; report the differences.

### Open questions (pp. 45–46)

Put these in `docs/STUDY_GUIDE.md` as a menu of extension projects, each with a suggested first experiment. Do not implement them.

## 8. The interactive lab (Streamlit, `superposition app`)

**Rules:**
- The app never runs paper-scale experiments. It trains small models live and loads everything else from `results/` through `runs.py`.
- When an artifact is missing, it shows the exact CLI command that generates it.
- Reuse the plotting module, so app figures match paper figures.
- Use `st.cache_resource` and `st.cache_data`.

**Pages:**

1. **Home.** The paper map (section → pages → code → experiment → notebook) and the live PASS/PARTIAL/FAIL table.
2. **Playground.**
   - Controls:
     - n (2–40) and m (1–10);
     - importance decay (0.5–1.0);
     - feature density 1 − S, on a log slider;
     - model type: linear, ReLU output, or ReLU hidden layer;
     - steps, restarts and seed.
   - Outputs:
     - the loss;
     - WᵀW with bias, the feature bars, and a Dᵢ table;
     - 2D arrows when m = 2, or a 3D plot when m = 3.
   - **Probe panel.** Choose which features are active and set their values. The panel shows h, each feature's readout before and after bias and ReLU, and the reconstruction next to the truth.
   - Target: n = 5, m = 2 trains in ≤ 10 s on CPU. Cache results by config.
3. **Phase diagrams.** E02 and E12, with the theory overlay. Clicking a cell shows representative weights.
4. **Geometry.** The E03 curves. Selecting a sparsity shows its geometry graph and Dᵢ histogram.
5. **Non-uniform.** E04, with a density slider for the perturbed feature.
6. **Dynamics.** E05, with a training-step scrubber and the 3D trajectories.
7. **Neurons.** E07 and E08 stack plots across sparsities, plus the E09 role tables and ablations.
8. **Adversarial.** E06.
9. **SAE lab.**
   - Pick a trained toy model.
   - Train a small SAE live, or load one.
   - Compare decoder directions with the true features, in a 2D view when m = 2.
   - Show the recovery metrics.

## 9. Notebooks: the learning path

| Notebook | Covers |
|---|---|
| `00_orientation` | the repo and how to use it |
| `01_features_directions_superposition` | pp. 1–8 |
| `02_demonstrating_superposition` | E00, E01 |
| `03_the_math` | loss decompositions, `theory.py` |
| `04_phase_change` | E02 |
| `05_geometry` | E03, E04 |
| `06_learning_dynamics` | E05 |
| `07_adversarial` | E06 |
| `08_neurons_and_computation` | E07, E08 |
| `09_mech_interp` | E09 |
| `10_sparse_autoencoders` | E10, E11 |
| `11_extensions` | E12 and the open questions |

Each notebook:
- Opens with its learning goals and the paper pages it covers.
- Asks the reader to predict the result before each run.
- Calls the library instead of re-implementing it, and uses the `quick` profile by default.
- Ends with 3–5 exercises, with solutions in collapsible `<details>` blocks.
- Runs top to bottom both locally and in Colab. A first cell installs the package from GitHub if it's missing; leave the GitHub owner as a placeholder.
- Is committed without outputs (nbstripout).

## 10. Documentation

| File | Contents |
|---|---|
| `README.md` | purpose; quickstart (`uv sync`, `uv run superposition run intro_pentagon`, `uv run superposition app`); results gallery; reproduction status table; how to cite the paper (BibTeX from p. 55) and this repo (CITATION.cff); related implementations |
| `docs/REPRODUCTION.md` | per experiment: the paper's claim with page, our figure, status, evidence, discrepancies with hypotheses, run IDs. `superposition report` regenerates the tables; the narrative is written by hand from actual results |
| `docs/PAPER_MAP.md` | every paper section mapped to code, experiments, notebooks and app pages |
| `docs/DESIGN.md` | architecture, data flow, extension points, how to add an experiment |
| `docs/DECISIONS.md` | dated entries: decision, rationale, paper page, alternatives |
| `docs/STUDY_GUIDE.md` | an ordered path for the owner (read → notebook → app page → "explain it back" questions → exercises), plus the menu of extension projects |
| `docs/GLOSSARY.md` | feature, linear representation, privileged basis, superposition, polysemanticity, sparsity, importance, interference, antipodal pair, feature dimensionality, tegum product, uniform polytope, SAE, dictionary learning, each with paper pages |
| `docs/REVIEW.md` | findings from the Phase 6 independent review, and their resolutions |
| `CLAUDE.md` | at most about 150 lines: commands, layout, conventions, the §3 pitfalls, and a pointer to SPEC.md |

## 11. Execution plan

Work in phases. A phase ends only when its exit criteria pass.

### Closing every phase

1. Run `make lint test`, plus the quick profile of every experiment the phase touched.
2. Publish figures, and update `docs/REPRODUCTION.md`, `CHANGELOG.md` and the PLAN.md checkboxes.
3. Commit.
4. Write `docs/progress/phase-N.md`, covering:
   - what was built;
   - the test status;
   - an acceptance table;
   - decisions made;
   - open issues;
   - a short "what the owner should look at and learn from this phase" section, with three questions.

Do not ask the owner questions mid-phase. Decide, log the decision in DECISIONS.md, and surface it in the phase report.

### Phases

**Phase 0: bootstrap.**
- Read SPEC.md and the paper.
- Write PLAN.md, with tasks mapped to spec sections and acceptance criteria.
- Set up the skeleton, tooling, CI, pre-commit, CLAUDE.md and `references/README.md`.
- Exit when: `uv sync` works; lint and the placeholder test suite pass; the CI config is valid.

**Phase 1: engine.**
- config, data, models, train and metrics, with every unit and property test from §6.
- runs, cli, and base plotting.
- E00 and E01, and notebooks 00–02.
- Exit when: the unit tier is green, E00 passes, and E01 is reported.

**Phase 2: lab v1.**
- App Home and Playground, including the probe panel.
- The `report` and `--publish` commands.
- Exit when: the app launches and the Playground meets its speed target.

**Phase 3: theory and geometry.**
- `theory.py`; E02, E03 and E04; their notebooks and app pages.

**Phase 4: dynamics, robustness, neurons.**
- E05, E06, E07 and E08; their notebooks and app pages.

**Phase 5: mechanistic interpretability.**
- `interp.py` and `sae.py`; E09, E10 and E11; their notebooks and app pages.

**Phase 6: extensions, review, polish.**
- E12.
- An independent review: a reviewer with fresh context re-derives each experiment from the cited paper pages, then audits the implementation and the evidence, looking adversarially for bugs and overclaims. Log findings and fixes in REVIEW.md.
- Complete every doc and the README gallery.
- Pass the fresh-clone test:
  1. `git clone`
  2. `uv sync`
  3. `uv run pytest -m "not slow"`
  4. `uv run superposition run intro_pentagon --profile quick`
  5. `uv run superposition app`

### Parallelism

- Build Phase 1 sequentially; it establishes the shared APIs.
- From Phase 3 on, experiments are independent units of work and may be built in parallel.
  - Each unit owns only its own files: experiment module, config, tests, notebook and app page.
  - Shared files (the registry, docs, REPRODUCTION.md, PAPER_MAP.md, CHANGELOG.md) are updated in a single integration step at the end of the phase, so parallel edits never collide.
  - Each unit runs its own tests and quick profile before reporting done.
  - Pair every experiment with an independent verifier that checks it against the cited pages.

### State across context resets

The PLAN.md checkboxes, `docs/progress/`, the git log and the test suite are the source of truth. After any reset, re-read CLAUDE.md, PLAN.md and the latest phase report before continuing.

## 12. Definition of done

- All phases are complete, with their exit criteria verified.
- CI is green, and the fresh-clone test passes.
- Every experiment has a quick-profile run, published figures, and a REPRODUCTION.md entry with an honest status.
- The app and all notebooks run.
- The docs are complete, and every item in REVIEW.md is either resolved or listed as open.
- The final message to the owner covers:
  - what was built and how to run it;
  - the reproduction status table;
  - known gaps and discrepancies;
  - the three most interesting findings;
  - suggested first steps from the study guide.
