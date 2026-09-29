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
- [x] `config.py`: typed dataclass configs, strict YAML IO, `quick`/`full` profiles (SPEC §5).
- [x] `utils.py`: `seed_everything`, per-instance seed derivation, device resolution
      (`auto` = CUDA → MPS → CPU) (SPEC §3 initialization, §5 compute).
- [x] `data.py`: `FeatureDistribution` protocol; `IndependentSparse` (per-instance and
      per-feature probabilities, configurable value range), `CorrelatedSets`,
      `AnticorrelatedSets`, `Composite` (SPEC §3 data, §6).
  - [x] Tests: density within 3 SE; nonzero range and mean; correlated sets switch together;
        anticorrelated sets never have two members active.
- [x] `models.py`: `TiedModel` (linear / ReLU output / ReLU hidden, `W: (inst, hidden, feat)`,
      `b: (inst, feat)`, `forward`, `hidden`, `gram`), `AbsModel`, activation registry
      (relu, gelu, silu, softplus, leaky_relu, sigmoid, identity); per-instance Xavier-normal
      init depending only on (seed, instance index) (SPEC §3 models/initialization, §6).
- [x] `train.py`: `train(...) -> TrainResult` (Adam, lr 1e-3, batch 1024, 10k steps, constant /
      cosine / warmup_cosine, `l1_hidden`, snapshots, final loss on ≥ 20,000 fresh samples);
      loss = sum over features → mean over batch → sum over instances; `select_best`
      (SPEC §3 loss, §6).
  - [x] Tests: instances are independent; loss/eval code reproduces the closed-form linear
        optimum Σ_{i≥m} Iᵢ(p/3 − p²/4) (SPEC §7 E00).
- [x] `metrics.py`: `feature_norms`, `superposition_score`, `feature_dimensionality`,
      `dims_per_feature`, `represented_mask`, `gram`, `is_superposition`, `angles_2d`,
      `canonicalize_2d`, `geometry_graph` (+ tegum components), `classify_fraction`,
      `neuron_polysemanticity`, `loss_by_num_active` (SPEC §3 metrics, §6).
  - [x] Known-answer tests: pentagon (Dᵢ = 2/5, D* = 0.4), antipodal pair (1/2), triangle
        (2/3), tetrahedron (3/4), orthonormal basis (1), zero column (0), regular simplex
        ((n − 1)/n).
  - [x] Property tests (hypothesis): unit-norm tight frames give Dᵢ = m/n and ΣDᵢ = m;
        every metric is invariant under orthogonal rotation of W.
- [x] `runs.py`: `results/<exp>/<run_id>/` with `config.yaml`, `manifest.json`,
      `metrics.parquet`, `acceptance.json`, checkpoints, figures, `log.txt`; latest-run lookup
      (SPEC §5 run artifacts).
- [x] `cli.py`: `list`, `run <exp> --profile --seed --device`, `figures <exp> --run --publish`
      (SPEC §5 CLI). `report` and `--publish` pulled forward from Phase 2 (D-108).
- [x] `plotting/`: shared style; WᵀW heatmap with bias (p. 10); feature-norm bars colored by
      superposition score (p. 11); canonicalized 2D arrows colored by importance (p. 2).
- [x] `experiments/`: registry, `base.Experiment`, `common` (sparsity-grid trainer, tables),
      `runner` (run, figures, publish).
- [x] `acceptance.py`: criteria with value/threshold/status; overall PASS/PARTIAL/FAIL.

Experiments (SPEC §7):
- [x] **E00 intro_pentagon (p. 2)**: n = 5, m = 2, Iᵢ = 0.9ⁱ, S ∈ {0, 0.8, 0.9}, 8 restarts,
      ReLU output and linear. Accept (ReLU, best restart per S):
  - S = 0: exactly {0, 1} represented, |cos| < 0.1.
  - S = 0.8: exactly {0, 1, 2, 3} represented as two antipodal pairs (partner cos < −0.95).
  - S = 0.9: all 5 represented; sorted angular gaps 72° ± 10°; all norms > 1; all biases < 0.
  - Linear: exactly {0, 1} represented and orthogonal at every S.
  - Both: every feature with ‖Wᵢ‖ < 0.05 has bias within 0.02 of (1 − S)/2 (p. 10).
  - Linear exact: best loss within 3% of Σ_{i≥m} Iᵢ(p/3 − p²/4) on ≥ 100,000 samples.
  - **Result: PASS 11/11** (quick, CPU, run `20260929T093651Z-5eeb804c`; full/MPS also 11/11).
- [x] **E01 basic_results (pp. 10–12)**: n = 20, m = 5, Iᵢ = 0.7ⁱ, 1 − S ∈ {1, 0.3, 0.1, 0.03,
      0.01, 0.003, 0.001}, linear vs ReLU output, ≥ 4 restarts; plus n = 80, m = 20, Iᵢ = 0.9ⁱ.
      Figures: WᵀW with bias + feature bars per sparsity (p. 11 layout). Accept:
  - Linear: exactly the top-m features represented, superposition score < 0.05, every S.
  - ReLU: matches linear at 1 − S = 1; represented count non-decreasing with sparsity
    (tolerance 1) and > m at the sparsest setting; superposition appears first among the less
    important features; antipodal pairs at intermediate sparsity (report where).
  - **Result: PASS 12/12** (quick, CPU, run `20260929T093755Z-2cb0bbc4`; full/MPS also 12/12).
- [x] Replication tests (`tests/replication/`, `@pytest.mark.slow`) for E00 and E01.

Notebooks (SPEC §9):
- [x] `00_orientation`, `01_features_directions_superposition` (pp. 1–8),
      `02_demonstrating_superposition` (E00, E01); nbmake smoke tests marked slow.

Close (SPEC §11):
- [x] `make lint test`; quick profiles of E00 and E01; publish figures; REPRODUCTION.md,
      CHANGELOG.md, this file; commit; `docs/progress/phase-1.md`.
- [x] **Exit:** unit tier green, E00 passes, E01 reported.

## Phase 2: lab v1 (SPEC §8, §11)

- [x] App Home: paper map (parsed from docs/PAPER_MAP.md) and live PASS/PARTIAL/FAIL table
      (cited run vs latest local run), latest figures, lab page list.
- [x] Playground: n (2–40), m (1–10), importance decay (0.5–1.0), log density slider, model
      type, steps/restarts/seed (+ device, paper presets); loss, WᵀW + bias, feature bars, Dᵢ
      table, 2D arrows / 3D plot (+ W matrix for ReLU hidden, training curves).
- [x] Probe panel: choose active features and values; show h, readouts before/after bias and
      ReLU, reconstruction vs truth.
- [x] `report` and `figures --publish` commands (Phase 1, D-108); `report --pin` cites fixed
      runs via docs/cited_runs.yaml (D-202, resolves Phase 1 open issue 3).
- [x] `superposition app` launches the lab (`--port`, `--headless`); `st.cache_data` /
      `st.cache_resource`; missing artifacts show the CLI command.
- [x] Tests: `lab` unit tests; AppTest for both pages (fast tier); slow tests for app launch
      and the speed target.
- [x] **Exit:** the app launches ✔ (server healthy, pages render; checked in a browser and by
      `tests/replication/test_lab.py`); Playground n = 5, m = 2 trains in ≤ 10 s on CPU ✔
      (3.6 s training, 3.6 s whole uncached page).

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

- [x] **E05 learning_dynamics (pp. 26–28)**: energy-level jump coinciding with a loss drop;
      n = 6, m = 3 octahedron final state in ≥ 1 restart; trajectory types reported.
  - **Result: PASS 2/2** (quick, CPU, run `20260929T172920Z-2fae0fc5`): every restart jumps
    late (a 30th digon forms) with a 0.04–0.05 loss drop; 48/48 runs end as octahedra.
- [~] `adversarial.py` + **E06 adversarial (pp. 29–30)**: vulnerability rises > 2× (report
      actual); Spearman ≥ 0.8 with 1/D*.
  - **Result: PARTIAL 2/4** (run `20260929T173800Z-e84cec26`): uniform setup 8.5×, ρ = 1.0;
    n = 20, m = 5 1.5×, ρ = 0.78 (little superposition in the paper's range).
- [x] **E07 privileged_basis (pp. 31–34)**: dense → monosemantic; polysemantic count
      non-decreasing (tol. 1), > 0 at sparsest; document ReLU sidestepping.
  - **Result: PASS 3/3** (run `20260929T174443Z-d54df07e`); with a hidden bias all biases go
    positive and the ReLU clips < 0.3% of the time.
- [~] **E08 abs_computation (pp. 34–39)**: (a) n = 3, m = 6 positive/negative-side neurons,
      per-feature loss < 5% of mean baseline; (b) n = 100, m = 40 stack plots, features
      computed well > m/2 when sparse.
  - **Result: PARTIAL 2/4** (run `20260929T175145Z-5a0a35cf`): (a) exact; (b) 39/40
    monosemantic when dense; 3 features computed well at 1 − S = 0.001 (dead output sides at
    initialization; D-409).
- [x] Notebooks 06–08, app pages Dynamics, Neurons, Adversarial.
- [x] **Exit:** E05–E08 implemented with notebooks and app pages; quick profiles run from a
      clean tree (each ≤ 10 min on CPU) and reported honestly; lint and unit tier green.

## Phase 5: mechanistic interpretability (SPEC §6 interp.py, sae.py, §7 E09–E11)

- [x] `interp.py`: path gains, neuron roles, ablation matrix (zero and mean), motif detector,
      rebuild test (D-501).
- [x] `sae.py`: ReLU and TopK SAEs, unit-norm decoder, resampling, ground-truth metrics (D-503).
- [x] **E09 mech_interp_abs**: ≥ 90% neurons named; ablations agree; motif found; rebuild
      reported.
  - **Result: PASS 3/3** (quick, CPU, run `20260929T211826Z-3fdd1d10`): 48/48 live neurons
    named, 48/48 ablations agree, the motif in 134 of 240 models; rebuild recovers 90–97%.
- [x] **E10 sae_recovery**: pentagon 5/5 at cos > 0.95; n = 20, m = 5 ≥ 90% at cos > 0.9.
  - **Result: PASS 3/3** (run `20260929T212259Z-fa564b17`): best SAEs recover every
    represented feature; 10/27, 16/18 and 6/18 settings pass; on the E03 model TopK 8
    recovers 100% and TopK 16 2% (INFO).
- [x] **E11 l1_removes_superposition**: superposition → none as λ grows; trade-off curve.
  - **Result: PASS 8/8** (run `20260929T212948Z-027b0327`): n = 20 keeps 4 orthogonal
    features at λ = 0.2 (8.4× the loss); the pentagon is free of superposition only with one
    feature left (λ = 0.65).
- [x] Notebooks 09–10, app pages Neurons (roles), SAE lab.
- [x] **Exit:** interp.py, sae.py and E09–E11 implemented with notebooks and app pages; quick
      profiles run from a clean tree (each ≤ 10 min on CPU) and reported; lint and unit tier
      green.

## Phase 6: extensions, review, polish (SPEC §7 E12, §10, §11, §12)

- [ ] **E12 activation_functions**: E02 for relu, gelu, silu, softplus, leaky_relu, sigmoid.
- [ ] Independent review → `docs/REVIEW.md`.
- [ ] Complete every doc; README gallery; notebook 11.
- [ ] Fresh-clone test: clone → `uv sync` → `pytest -m "not slow"` → quick E00 → app.
