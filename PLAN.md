# PLAN.md

The working plan for building `superposition`. Every task maps to a section of
[SPEC.md](SPEC.md); experiment tasks carry their acceptance criteria (SPEC §7, thresholds are
initial proposals governed by the integrity rules in SPEC §2). Checkboxes, `docs/progress/`,
the git log and the test suite are the source of truth across context resets (SPEC §11).

Status legend: `[x]` done and verified · `[ ]` not done · `[~]` done with an open issue
(see the phase report).

---

## Phase 0: bootstrap (SPEC §11)

- [x] Read SPEC.md and the paper (all 62 pages).
- [x] PLAN.md (this file): tasks mapped to spec sections and acceptance criteria.
- [x] Skeleton: `src/superposition/` package with module stubs, `configs/`, `app/`,
      `notebooks/`, `tests/{unit,replication}/`, `docs/{progress,figures}/` (SPEC §4).
- [x] Tooling: `pyproject.toml` + `uv.lock` (SPEC §5), ruff, pyright (basic), pytest markers,
      Makefile targets `setup lint test test-slow run-quick figures report app` (SPEC §4).
- [x] CI: `.github/workflows/ci.yml` (ruff check, ruff format --check, pyright, pytest -m "not
      slow" on CPU-only torch) and an optional scheduled `slow.yml` (SPEC §5).
- [x] pre-commit: hygiene hooks, ruff, nbstripout (SPEC §5).
- [x] `CLAUDE.md`, `references/README.md` (URL + BibTeX from p. 55), README, LICENSE (MIT),
      CITATION.cff, CHANGELOG.md, docs skeletons, `docs/DECISIONS.md` (SPEC §10).
- [x] `.gitignore`: `results/`, checkpoints, `references/*.pdf` (SPEC §2 licensing).
- [x] **Exit:** `uv sync` works; lint and the placeholder test suite pass; CI config is valid.

## Phase 1: engine (SPEC §6, §7 E00–E01, §9 notebooks 00–02)

Library (SPEC §6):
- [ ] `config.py`: typed dataclass configs, strict YAML IO, `quick`/`full` profiles (SPEC §5).
- [ ] `utils.py`: `seed_everything`, per-instance seed derivation, device resolution
      (`auto` = CUDA → MPS → CPU) (SPEC §3 initialization, §5 compute).
- [ ] `data.py`: `FeatureDistribution` protocol; `IndependentSparse` (per-instance and
      per-feature probabilities, configurable value range), `CorrelatedSets`,
      `AnticorrelatedSets`, `Composite` (SPEC §3 data, §6).
  - [ ] Tests: density within 3 SE; nonzero range and mean; correlated sets switch together;
        anticorrelated sets never have two members active.
- [ ] `models.py`: `TiedModel` (linear / ReLU output / ReLU hidden, `W: (inst, hidden, feat)`,
      `b: (inst, feat)`, `forward`, `hidden`, `gram`), `AbsModel`, activation registry
      (relu, gelu, silu, softplus, leaky_relu, sigmoid, identity); per-instance Xavier-normal
      init depending only on (seed, instance index) (SPEC §3 models/initialization, §6).
- [ ] `train.py`: `train(...) -> TrainResult` (Adam, lr 1e-3, batch 1024, 10k steps, constant /
      cosine / warmup_cosine, `l1_hidden`, snapshots, final loss on ≥ 20,000 fresh samples);
      loss = sum over features → mean over batch → sum over instances; `select_best`
      (SPEC §3 loss, §6).
  - [ ] Tests: instances are independent; loss/eval code reproduces the closed-form linear
        optimum Σ_{i≥m} Iᵢ(p/3 − p²/4) (SPEC §7 E00).
- [ ] `metrics.py`: `feature_norms`, `superposition_score`, `feature_dimensionality`,
      `dims_per_feature`, `represented_mask`, `gram`, `is_superposition`, `angles_2d`,
      `canonicalize_2d`, `geometry_graph` (+ tegum components), `classify_fraction`,
      `neuron_polysemanticity`, `loss_by_num_active` (SPEC §3 metrics, §6).
  - [ ] Known-answer tests: pentagon (Dᵢ = 2/5, D* = 0.4), antipodal pair (1/2), triangle
        (2/3), tetrahedron (3/4), orthonormal basis (1), zero column (0), regular simplex
        ((n − 1)/n).
  - [ ] Property tests (hypothesis): unit-norm tight frames give Dᵢ = m/n and ΣDᵢ = m;
        every metric is invariant under orthogonal rotation of W.
- [ ] `runs.py`: `results/<exp>/<run_id>/` with `config.yaml`, `manifest.json`,
      `metrics.parquet`, `acceptance.json`, checkpoints, figures, `log.txt`; latest-run lookup
      (SPEC §5 run artifacts).
- [ ] `cli.py`: `list`, `run <exp> --profile --seed --device`, `figures <exp> --run --publish`
      (SPEC §5 CLI). `report` pulled forward from Phase 2 (see DECISIONS.md).
- [ ] `plotting/`: shared style; WᵀW heatmap with bias (p. 10); feature-norm bars colored by
      superposition score (p. 11); canonicalized 2D arrows colored by importance (p. 2).
- [ ] `experiments/`: registry + base experiment protocol.

Experiments (SPEC §7):
- [ ] **E00 intro_pentagon (p. 2)**: n = 5, m = 2, Iᵢ = 0.9ⁱ, S ∈ {0, 0.8, 0.9}, 8 restarts,
      ReLU output and linear. Accept (ReLU, best restart per S):
  - S = 0: exactly {0, 1} represented, |cos| < 0.1.
  - S = 0.8: exactly {0, 1, 2, 3} represented as two antipodal pairs (partner cos < −0.95).
  - S = 0.9: all 5 represented; sorted angular gaps 72° ± 10°; all norms > 1; all biases < 0.
  - Linear: exactly {0, 1} represented and orthogonal at every S.
  - Both: every feature with ‖Wᵢ‖ < 0.05 has bias within 0.02 of (1 − S)/2 (p. 10).
  - Linear exact: best loss within 3% of Σ_{i≥m} Iᵢ(p/3 − p²/4) on ≥ 100,000 samples.
- [ ] **E01 basic_results (pp. 10–12)**: n = 20, m = 5, Iᵢ = 0.7ⁱ, 1 − S ∈ {1, 0.3, 0.1, 0.03,
      0.01, 0.003, 0.001}, linear vs ReLU output, ≥ 4 restarts; plus n = 80, m = 20, Iᵢ = 0.9ⁱ.
      Figures: WᵀW with bias + feature bars per sparsity (p. 11 layout). Accept:
  - Linear: exactly the top-m features represented, superposition score < 0.05, every S.
  - ReLU: matches linear at 1 − S = 1; represented count non-decreasing with sparsity
    (tolerance 1) and > m at the sparsest setting; superposition appears first among the less
    important features; antipodal pairs at intermediate sparsity (report where).
- [ ] Replication tests (`tests/replication/`, `@pytest.mark.slow`) for E00 and E01.

Notebooks (SPEC §9):
- [ ] `00_orientation`, `01_features_directions_superposition` (pp. 1–8),
      `02_demonstrating_superposition` (E00, E01); nbmake smoke tests marked slow.

Close (SPEC §11):
- [ ] `make lint test`; quick profiles of E00 and E01; publish figures; REPRODUCTION.md,
      CHANGELOG.md, this file; commit; `docs/progress/phase-1.md`.
- [ ] **Exit:** unit tier green, E00 passes, E01 reported.

## Phase 2: lab v1 (SPEC §8, §11)

- [ ] App Home: paper map and live PASS/PARTIAL/FAIL table.
- [ ] Playground: n (2–40), m (1–10), importance decay (0.5–1.0), log density slider, model
      type, steps/restarts/seed; loss, WᵀW + bias, feature bars, Dᵢ table, 2D arrows / 3D plot.
- [ ] Probe panel: choose active features and values; show h, readouts before/after bias and
      ReLU, reconstruction vs truth.
- [ ] `report` and `figures --publish` commands (if not already complete from Phase 1).
- [ ] **Exit:** the app launches; Playground n = 5, m = 2 trains in ≤ 10 s on CPU.

## Phase 3: theory and geometry (SPEC §6 theory.py, §7 E02–E04)

- [ ] `theory.py`: linear-model loss decomposition, 1-sparse ReLU decomposition (validated
      against Monte-Carlo), exact expected-loss phase diagrams for n = 2, m = 1 (3 families +
      optional confused family) and n = 3, m = 2 (4 families). Record closed-form vs quadrature.
- [ ] **E02 phase_change (pp. 14–15)**: 10 models per cell, drop highest-loss, majority label;
      all three regions in both diagrams; ≥ 80% empirical/theory agreement; disagreement map.
- [ ] **E03 uniform_geometry (pp. 16–21)**: n = 400, m = 30; D* ≈ 1 to 1/(1 − S) ≈ 1.2;
      plateau 0.5 ± 0.03 (report span); ≈ 0.25 at 8; Dᵢ clusters at ≥ 3 fractions; ΣDᵢ vs m.
- [ ] **E04 nonuniform_geometry (pp. 21–25)**: (a) perturbed pentagon; (b) correlated vs
      anticorrelated in m = 2; (c) local almost-orthogonal bases; (d) collapse to principal
      components.
- [ ] Notebooks 03–05, app pages Phase diagrams, Geometry, Non-uniform.

## Phase 4: dynamics, robustness, neurons (SPEC §7 E05–E08)

- [ ] **E05 learning_dynamics (pp. 26–28)**: energy-level jump coinciding with a loss drop;
      n = 6, m = 3 octahedron final state in ≥ 1 restart; trajectory types reported.
- [ ] `adversarial.py` + **E06 adversarial (pp. 29–30)**: vulnerability rises > 2× (report
      actual); Spearman ≥ 0.8 with 1/D*.
- [ ] **E07 privileged_basis (pp. 31–34)**: dense → monosemantic; polysemantic count
      non-decreasing (tol. 1), > 0 at sparsest; document ReLU sidestepping.
- [ ] **E08 abs_computation (pp. 34–39)**: (a) n = 3, m = 6 positive/negative-side neurons,
      per-feature loss < 5% of mean baseline; (b) n = 100, m = 40 stack plots, features
      computed well > m/2 when sparse.
- [ ] Notebooks 06–08, app pages Dynamics, Neurons, Adversarial.

## Phase 5: mechanistic interpretability (SPEC §6 interp.py, sae.py, §7 E09–E11)

- [ ] `interp.py`: neuron roles, ablation matrix, motif detector, rebuild test.
- [ ] `sae.py`: ReLU and TopK SAEs, unit-norm decoder, resampling, ground-truth metrics.
- [ ] **E09 mech_interp_abs**: ≥ 90% neurons named; ablations agree; motif found; rebuild
      reported.
- [ ] **E10 sae_recovery**: pentagon 5/5 at cos > 0.95; n = 20, m = 5 ≥ 90% at cos > 0.9.
- [ ] **E11 l1_removes_superposition**: superposition → none as λ grows; trade-off curve.
- [ ] Notebooks 09–10, app pages Neurons (roles), SAE lab.

## Phase 6: extensions, review, polish (SPEC §7 E12, §10, §11, §12)

- [ ] **E12 activation_functions**: E02 for relu, gelu, silu, softplus, leaky_relu, sigmoid.
- [ ] Independent review → `docs/REVIEW.md`.
- [ ] Complete every doc; README gallery; notebook 11.
- [ ] Fresh-clone test: clone → `uv sync` → `pytest -m "not slow"` → quick E00 → app.
