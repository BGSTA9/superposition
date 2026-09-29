# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow the build phases
in SPEC.md §11 until a first release.

## [Unreleased]

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
