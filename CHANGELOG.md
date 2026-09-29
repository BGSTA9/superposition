# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow the build phases
in SPEC.md §11 until a first release.

## [Unreleased]

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
