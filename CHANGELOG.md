# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow the build phases
in SPEC.md §11 until a first release.

## [Unreleased]

### Phase 3: theory and geometry (closed 2026-09-30)

Drafted on 2026-09-29 and finished in a rectification session after Phases 4 and 5.

#### Added
- `superposition.theory` (pp. 13–16): the linear decomposition into feature benefit,
  interference and a bias term; the exact 0- and 1-sparse terms; the exact expected loss of a
  ReLU output model whose outputs each read at most one other feature (closed form, D-301);
  weight families W ⊥ v, their exact optima over scale and biases (D-302), and theoretical phase
  diagrams for n = 2, m = 1 and n = 3, m = 2, with McGrath's confused family and a zero-bias
  variant.
- `metrics.component_geometry` and `metrics.geometry_census`: tegum factors named by features,
  dimensions spanned and uniformity (D-308; resolves Phase 2 open issue 1).
- **E02 phase_change** (pp. 14–16). PASS 6/6: all three regions in both diagrams, agreement
  0.875 (n = 2) and 0.819 (n = 3); every disagreement is a majority of stuck models.
- **E03 uniform_geometry** (pp. 16–21). PASS 4/4: D* plateau at 1/2 over 2.34–3.79, 0.264 at
  1/(1 − S) = 7.85, Dᵢ clusters at 2/3, 1/2, 2/5 and 3/8.
- **E04 nonuniform_geometry** (pp. 21–25). PASS 9/9: the perturbed pentagon, correlated and
  anticorrelated sets, local bases, collapse to principal components. The quick profiles take
  487, 365 and 287 s on a laptop CPU.
- Named variant profiles in config files (`superposition run <exp> --profile batch1024`), used
  for the logged batch-size comparison behind D-309.
- Plotting: `plotting/phase.py` (phase diagrams, family loss curves), `plotting/geometry.py`
  (D* and Dᵢ figures, geometry graphs, factor census); `plotly` phase map for the lab.
- Lab: **Phase diagrams** (click a cell for its models and every family's optimum),
  **Geometry** (geometry graph and Dᵢ histogram per sparsity) and **Non-uniform** (the
  perturbed-feature density slider, correlated sets, local bases, collapse);
  `lab.phase_cell_shared_losses` compares a cell's models with the families exactly where
  possible and on shared samples otherwise.
- Notebooks `03_the_math`, `04_phase_change`, `05_geometry`.
- Tests: `theory.py` against Monte Carlo and adaptive quadrature; E02–E04 criteria on
  hand-built weights; polytope naming; reduced replication tests in the slow tier and full
  quick-profile tests in the paper tier.

#### Changed
- Test tiers (D-310): `slow` keeps E00, the lab tests, the reduced E02–E04 tests and notebooks
  00–03 under 15 minutes; replication tests that re-run a whole quick profile, and notebooks
  04–10, move to a new opt-in `paper` marker (`make test-paper`).
- E04 picks its best restarts on 200,000 evaluation samples in every part.

#### Fixed
- E04's lint suppressions replaced by real fixes; its Spearman helper returns NaN on constant
  input instead of dividing by zero.
- E02 figures: the two "dedicated" families of n = 3 are told apart, and the empirical panel
  counts its voters.
- The lab launch test no longer leaks a Streamlit server per run (it terminated only the CLI,
  and the server's output pipe lost its reader).

### Phase 5: mechanistic interpretability (2026-09-30)

#### Added
- `superposition.interp` (pp. 38–39): path gains |W₁[h,i]|·W₂[j,h] (invariant to footnote 15's
  rescaling); neuron roles (positive/negative side, superposition, asymmetric superposition,
  inhibitor with triggers, dead, unexplained); the asymmetric superposition + inhibition motif
  detector; zero- and mean-ablation matrices on shared samples; the rebuild test.
- `superposition.sae` (pp. 42–44): multi-instance sparse autoencoders (ReLU + L1, unit-norm
  decoder renormalized every step, TopK, per-instance latent masks for several dictionary
  sizes, dead-latent resampling) and ground-truth metrics (max cosine with live latents,
  recovered fractions, L0, MSE, FVU, dead fraction).
- **E09 mech_interp_abs** (pp. 38–39): m = 2n − 2 absolute-value models reverse-engineered
  neuron by neuron. PASS 3/3: every live neuron named (48/48), ablations agree (48/48), the
  p. 39 motif in 134 of 240 models, the rebuild recovers 90–97% of the loss reduction.
- **E10 sae_recovery** (pp. 42–44): SAE sweeps on the E00 pentagon, sparse E01 models and an
  E03 model, scored against the true features. PASS 3/3 (best SAE of each sweep); reports how many
  settings pass and that low reconstruction error need not mean recovery (TopK 8 vs 16 on the
  E03 model).
- **E11 l1_removes_superposition** (p. 42): λ‖h‖₁ on the ReLU output model swept over λ. PASS
  8/8; the pentagon loses superposition only when one feature is left (antipodal pairs survive
  the penalty). The E09–E11 quick profiles take 271, 406 and 533 s on a laptop CPU.
- Plotting: `plotting/interp.py` (the p. 39 view with roles, ablation matrices, rebuild, role
  census), `plotting/sae.py` (2D dictionaries, recovery sweeps, reconstruction vs recovery),
  `plotting/l1.py` (loss and superposition against λ, the trade-off curve);
  `style.CATEGORICAL` (the validated eight-slot palette order).
- Lab: the **Neurons** page's E09 tab (roles, ablations, motifs, rebuild) and a new **SAE lab**
  page (train an SAE live on the pentagon or a sparse E01 model; E10 sweeps; E11 trade-off).
- Notebooks `09_mech_interp` and `10_sparse_autoencoders`.
- Tests: known-answer tests for roles, motifs, ablations (p/6 and the mean-ablation closed
  form) and the rebuild; SAE tests (unit-norm decoder, TopK, masks, signed-cosine recovery,
  resampling, instance independence); E09–E11 criteria on synthetic inputs; AppTest for the new
  tab and page; slow replication tests for E09–E11.
- Docs: DECISIONS D-501–D-505; REPRODUCTION sections E09–E11; STUDY_GUIDE stops 9–10;
  GLOSSARY (side neuron, path gain, asymmetric superposition, inhibition, ablation, rebuild
  test, dead latent, TopK SAE); PAPER_MAP and DESIGN rows.

#### Changed
- `metrics.per_feature_loss(..., shared_samples=True)` scores every instance on the same
  inputs (common random numbers); the default path is unchanged (same random draws).

### Phase 4: dynamics, robustness, neurons (2026-09-29)

#### Added
- **E05 learning_dynamics** (pp. 26–28): energy-level jumps in the uniform n = 400, m = 30
  setup, with a jump detector (stable level → jump → coinciding loss drop on a fixed batch);
  two correlated sets in m = 3 classified into geometric stages (small, collapsed, transition,
  prism, octahedron) with trajectory types. PASS 2/2.
- **E06 adversarial** (pp. 29–30) and `superposition.adversarial`: the analytic per-feature L2
  attack (budget 0.1 × mean input norm, per-example worst case), a no-superposition reference
  and vulnerability (L′ − L)/(L′₀ − L₀). PARTIAL 2/4: passes on the uniform setup of the
  p. 29 figure (8.5× rise, ρ = 1.0), fails on n = 20, m = 5.
- **E07 privileged_basis** (pp. 31–34): the ReLU hidden layer model across sparsity (best of
  100 restarts), neuron polysemanticity, and a hidden-bias variant documenting how the model
  sidesteps its ReLU (p. 34). PASS 3/3.
- **E08 abs_computation** (pp. 34–39): side neurons for n = 3, m = 6; |W₁| stack plots,
  features computed well and primary + secondary neurons for n = 100, m = 40; dead-side and
  bias-init diagnostics. PARTIAL 2/4 (computation in superposition not reproduced; see
  REPRODUCTION.md).
- `metrics.per_feature_loss` (exact conditional estimator for very sparse features) and
  `metrics.normalize_hidden_units` (footnote 15).
- Plotting: `plotting/dynamics.py` (Dᵢ over training, stage rasters, 3D trajectories in
  matplotlib and plotly), `plotting/adversarial.py` (p. 29, p. 30), `plotting/neurons.py`
  (W heatmaps, neuron stack plots, p. 36 weight bars, capacity).
- Lab pages **Dynamics** (training-step scrubbers, 3D trajectories), **Neurons** (E07/E08;
  E09 in Phase 5) and **Adversarial**; loaders in `superposition.lab`.
- Notebooks `06_learning_dynamics`, `07_adversarial`, `08_neurons_and_computation`.
- Tests: unit tests for the attack, jump detector, stage classifier, neuron roles and
  per-feature loss; AppTest for the new pages (with and without runs); slow replication tests
  for E05–E08.
- Docs: DECISIONS D-401–D-409; REPRODUCTION sections E05–E08; STUDY_GUIDE stops 6–8;
  GLOSSARY (monosemantic/polysemantic neuron, stack plot, energy-level jump, adversarial
  vulnerability); PAPER_MAP and DESIGN rows.

#### Changed
- `superposition list` and the Home status table include E05–E08; the CLI test compares the
  listing with the registry instead of a fixed list.

#### Fixed
- E05 computes feature dimensionalities in chunks of snapshots (the full profile would have
  built a 7.7 GB intermediate); results are bitwise identical.
- Figures: the energy-level loss panel zooms to the plateau (as on p. 26); W heatmaps saturate
  at a robust maximum and |W₁| stack plots scale each row on its own.

### Phase 2: lab v1 (2026-09-29)

#### Added
- Streamlit lab (`superposition app [--port] [--headless]`):
  - **Home:** reproduction status (cited run vs latest local run), latest figures, lab pages,
    and a searchable paper map parsed from docs/PAPER_MAP.md.
  - **Playground:** live training of the linear / ReLU output / ReLU hidden models with the
    SPEC §8 controls and paper presets; loss, WᵀW + bias, feature bars, Dᵢ table, 2D arrows /
    3D plot / W matrix, training curves, and a probe panel (h, readout, + bias, output vs
    truth). Default n = 5, m = 2 trains in ≈ 3.6 s on a laptop CPU.
- `superposition.lab`: Playground jobs, per-feature table, probe, paper map, status table.
- `plotting/interactive.py` (plotly): 3D feature vectors, loss curves, probe and hidden bars;
  `arrows_figure`, `weight_matrix_figure`.
- `docs/cited_runs.yaml` and `superposition report --pin EXP | --pin-all [--allow-dirty]`:
  the docs' tables come from pinned runs (D-202).
- Tests: AppTest page tests (fast tier); slow tests for app launch and the ≤ 10 s target.
- Docs: DECISIONS D-201–D-206, DESIGN (lab, cited runs), README lab section, STUDY_GUIDE lab
  steps, PAPER_MAP lab pages, CLAUDE.md.

#### Changed
- `superposition report` regenerates tables from cited runs, not the latest run; unpinned
  experiments are marked "(uncited)". The status table gains a profile · device column.
- `weights_grid` has a minimum width (single-model figures); lab colors follow importance
  values. Published experiment figures regenerate byte-identical.
- Test sessions use the Agg matplotlib backend.

### Phase 1: engine (2026-09-29)

#### Added
- `utils`: seed streams (`derive_seed`), `seed_everything`, device resolution (CUDA → MPS → CPU).
- `config`: typed dataclass configs, strict YAML loading with `quick`/`full` profiles, config hash.
- `data`: `IndependentSparse` (per-instance and per-feature probabilities, any value range),
  `CorrelatedSets`, `AnticorrelatedSets`, `Composite`, `geometric_importance`.
- `models`: `TiedModel` (linear, ReLU output, ReLU hidden) and `AbsModel`, per-instance
  Xavier-normal init seeded by (seed, instance), activation registry.
- `train`: vectorized multi-instance Adam trainer (constant / cosine / warmup-cosine, L1 on
  hidden activations, snapshots, fresh-sample final loss), `evaluate_loss`, `select_best`.
- `metrics`: feature norms, superposition score, Dᵢ, D*, represented mask, formal
  superposition test, 2D angles and canonicalization, geometry graph with tegum components,
  fraction classification, neuron polysemanticity, k-sparse loss terms.
- `theory.linear_optimum_loss`: the exact optimum of the linear model.
- `runs`, `acceptance`: run directories with manifests (git commit and dirty flag, versions,
  device, seed, timings), parquet tables, checkpoints, acceptance records.
- `plotting`: shared style; WᵀW + bias heatmaps, feature-norm bars colored by superposition
  (p. 11 grid), canonicalized 2D arrows (p. 2).
- Experiments E00 `intro_pentagon` and E01 `basic_results` with configs, criteria, figures,
  and slow replication tests.
- CLI: `list`, `run`, `figures [--publish]`, `report`, `app` (placeholder until Phase 2).
- Notebooks `00_orientation`, `01_features_directions_superposition`,
  `02_demonstrating_superposition`.
- Docs: REPRODUCTION (E00, E01), DECISIONS D-101–D-114, DESIGN, STUDY_GUIDE stops 0–2,
  published figures under `docs/figures/`.

#### Changed
- ruff ignores E501; `ruff format` owns line length (D-112).

### Phase 0: bootstrap (2026-09-29)

#### Added
- uv project (`pyproject.toml`, `uv.lock`, Python 3.12 pin) with the SPEC §5 dependencies;
  CPU-only torch wheels on Linux via a uv index source.
- Package skeleton `src/superposition/` with module stubs naming the phase that implements
  each; minimal Typer CLI (`superposition list`).
- Tooling: ruff (lint + format), pyright (basic), pytest with a `slow` marker, Makefile
  (`setup lint format test test-slow run-quick figures report app`), pre-commit
  (hygiene hooks, ruff, nbstripout).
- GitHub Actions: `ci.yml` (lint, type-check, fast tests on CPU) and scheduled `slow.yml`.
- Docs: PLAN.md, CLAUDE.md, README, references/README.md (URL + BibTeX), LICENSE (MIT),
  CITATION.cff, docs skeletons, DECISIONS.md.
- Placeholder test suite (package import, CLI help).
