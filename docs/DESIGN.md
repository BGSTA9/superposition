# Design

Architecture, data flow, extension points, and how to add an experiment. SPEC.md §4–§6 is the
normative description; this file explains how the code realizes it. It grows with each phase
(PLAN.md).

## Layers

```
configs/<exp>.yaml ──load_config(profile)──▶ typed dataclass (config.py)
                                                   │
superposition run <exp> ─▶ experiments/runner.run_experiment
        │                      │  create_run() → results/<exp>/<run_id>/ (runs.py)
        │                      ▼
        │            experiments/<exp>.run(config, run_dir, device, logger)
        │                      │
        │     ┌────────────────┼──────────────────────────────┐
        │     ▼                ▼                              ▼
        │  data.py          models.py                     train.py
        │  (batch,inst,feat) (params: inst, …)            Adam, loss per instance,
        │     └────────────────┴──────────────▶ TrainResult (history, snapshots, final loss)
        │                                           │
        │                     metrics.py (pure functions of W, b) ─▶ criteria (acceptance.py)
        │                                           │
        │              checkpoints/*.pt, metrics.parquet, features.parquet, acceptance.json
        ▼
superposition figures <exp> [--publish] ─▶ experiments/<exp>.figures(run_dir) ─▶ plotting/
superposition report [--pin] ─▶ report.py ─▶ docs/cited_runs.yaml + AUTO blocks in REPRODUCTION.md
superposition app ─▶ streamlit run app/Home.py ─▶ app/pages/* ─▶ lab.py (+ runs, plotting)
```

| Module | Responsibility | Depends on |
|---|---|---|
| `utils` | seeds (`derive_seed` streams), devices | numpy, torch |
| `config` | dataclass configs, strict YAML loading, profiles, config hash | pyyaml |
| `data` | feature distributions returning `(batch, inst, feat)` | torch |
| `models` | `TiedModel`, `AbsModel`, activation registry, per-instance init | utils |
| `train` | trainer, evaluation, `select_best` | config, data |
| `metrics` | pure metric functions | torch, networkx |
| `theory` | closed forms (Phase 1: linear optimum; Phase 3: decompositions, phase diagrams) | torch |
| `acceptance` | criteria, statuses, `acceptance.json` payload | — |
| `runs` | run directories, manifests, latest-run lookup | config |
| `plotting/` | figures from arrays; `style` holds colors and rc settings | matplotlib |
| `experiments/` | one module per experiment + `common`, `base`, `runner`, registry | all of the above |
| `report` | cited runs (`docs/cited_runs.yaml`), REPRODUCTION.md tables | experiments, runs |
| `lab` | Playground jobs, per-feature table, probe, paper map, status table (no Streamlit) | experiments.common, metrics, report |
| `cli` | Typer commands | runner, report |
| `app/` | Streamlit pages (layout only) and `lab_ui.py` helpers | lab, plotting |

## Key design choices

- **The instance dimension.** Every parameter and batch carries a leading `inst` axis, and
  instances never interact: each instance's parameters appear only in its own loss term, the
  loss is summed over instances, and Adam is elementwise. One model object therefore trains a
  whole sweep (sparsities × restarts) in one vectorized step. Condition and restart indices
  are bookkeeping (`experiments.common.GridResult`).
- **Seeds are streams.** `derive_seed(seed, INIT_STREAM, param, k)` initializes instance k;
  `DATA_STREAM`, `EVAL_STREAM` and `PROBE_STREAM` feed the trainer, the final evaluation and
  diagnostics. Nothing reads global RNG state.
- **Computation vs presentation.** Experiments save checkpoints and tables; `figures()` redraws
  everything from a saved run, so figures can be regenerated (or restyled) without retraining.
- **Criteria are data.** An experiment returns a list of `Criterion` (value, threshold,
  status, paper page). Thresholds live in config dataclasses, never inline.
- **Provenance.** A run's manifest records the git commit and whether the tree was dirty;
  published figures carry a `SOURCE.txt` with the run ID and commit.

## The lab (`app/`, Phase 2+)

- **Pages are thin.** `app/Home.py` and `app/pages/N_Name.py` arrange widgets; every number
  comes from `superposition.lab` and every figure from `superposition.plotting`, so the lab's
  figures match the experiments' and the logic is unit-tested without a browser.
- **What is live and what is loaded.** The Playground trains small models live
  (`lab.train_playground`, the experiments' own grid trainer). Everything else is loaded from
  `results/` through `runs.py`; when something is missing the page shows the exact CLI
  command that creates it (`lab_ui.missing_artifact`).
- **Caching.** `st.cache_data` caches training by (config, device), so revisiting a setting is
  instant; `st.cache_resource` caches the rebuilt model the probe panel uses.
- **Adding a page** (Phases 3–5): write the computation as functions in `superposition.lab`
  (or the experiment module) with tests, add `app/pages/N_Name.py`, reuse plotting functions,
  and add an AppTest in `tests/unit/test_app.py`.

## Cited runs

`docs/cited_runs.yaml` records, per experiment, the run REPRODUCTION.md is written from, with
its profile, device, seed, commit, dirty flag and status summary. `superposition report`
rebuilds the tables from those runs (never from whatever ran last), and the lab's Home page
shows the cited status next to the latest local run. The workflow for a new result is:
commit → run → write the narrative → `superposition report --pin <exp>` → commit the docs.

## Extension points

- **A new activation:** `models.register_activation("name", fn)`; `TiedModel(hidden_act=...,
  out_act=...)` picks it up.
- **A new feature distribution:** any object with `n_features`, `n_inst` and
  `sample(batch_size, generator) -> (batch, inst, feat)` satisfies `data.FeatureDistribution`.
  `Composite` concatenates distributions along the feature axis.
- **A new metric:** add a pure function of `W` (and `b`) to `metrics.py`, with a known-answer
  test.
- **Variants of an experiment:** add fields to its config dataclass with the paper-faithful
  value as default, and override them in a new YAML profile or via `overrides` — never by
  changing a replication default (SPEC §2).

## How to add an experiment

1. Write `src/superposition/experiments/<name>.py` with:
   - a config dataclass subclassing `ExperimentConfig` (thresholds in a nested dataclass);
   - `run(config, run_dir, device, logger) -> list[Criterion]` that trains, saves checkpoints
     and tables into `run_dir`, and returns criteria built with `acceptance.check` / `info`;
   - `figures(run_dir) -> list[Path]` that loads the checkpoints and saves figures with
     `plotting.style.save_figure`;
   - `EXPERIMENT = Experiment(name, code, title, paper_pages, config_cls, run, figures)`.
2. Add `configs/<name>.yaml` with `experiment`, `base`, and `profiles: {quick, full}`; the quick
   profile must finish in ≤ 10 minutes on a laptop CPU. Comment the measured runtime.
3. Register it in `experiments/__init__.py`.
4. Test the criteria logic on hand-built weights (`tests/unit/test_experiment_criteria.py`)
   and add a slow replication test (`tests/replication/`).
5. Record every under-specified choice in `docs/DECISIONS.md`, run the quick profile, publish
   figures (`superposition figures <name> --publish`), run `superposition report`, and write
   the narrative in `docs/REPRODUCTION.md`.
