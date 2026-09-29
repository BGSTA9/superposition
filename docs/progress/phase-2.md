# Phase 2 report: lab v1

Date: 2026-09-29 · Commits `af93f25`..(this report) · SPEC §11 Phase 2, §8

> **Commit note.** The Phase 2 documentation changes (DECISIONS, DESIGN, PAPER_MAP,
> STUDY_GUIDE, README, CLAUDE.md, CHANGELOG, PLAN, `docs/cited_runs.yaml`, the regenerated
> REPRODUCTION.md tables) landed in `e3d80a3`, a commit made under the owner's git identity
> from outside this session while those files were staged. Its content is exactly what was
> intended. It also includes `.claude/launch.json`, a local preview config used to run the lab
> in the in-app browser; it is harmless, and the owner may delete it if unwanted.

## What was built

**The lab** (`uv run superposition app [--port 8501] [--headless]`, Streamlit multipage app in
`app/`):

- **Home** (`app/Home.py`):
  - *Reproduction status*: one row per experiment with the **cited** run (what
    docs/REPRODUCTION.md is written from) next to the **latest local run** read live from
    `results/`, plus its profile, device, clean-tree flag, and the command to (re)run it.
  - *Latest figures* of each experiment's newest run; if a run or its figures are missing, the
    exact CLI command is shown instead.
  - *Lab pages*: which pages exist now and which phase brings the rest.
  - *Paper map*, searchable, parsed from docs/PAPER_MAP.md so it has one source of truth.
- **Playground** (`app/pages/1_Playground.py`):
  - *Controls* (sidebar form, SPEC §8 ranges): n 2–40, m 1–10, importance decay 0.5–1.0,
    feature density 1 − S on a log select slider (0.001–1), model (linear / ReLU output / ReLU
    hidden), steps, restarts, seed, device (cpu / auto). Seven one-click **paper presets** (the
    four p. 2 settings, p. 11, p. 32, and a 3D example).
  - *Outputs*: best loss, represented features, D*, formal superposition flag, and either the
    closed-form optimum (linear model) or the loss spread across restarts; WᵀW with bias and
    ‖Wᵢ‖ bars colored by superposition (the experiments' own plotting code); canonicalized 2D
    arrows (m = 2), a rotatable 3D plot (m = 3), or a signed bar chart (m = 1); W itself for the
    ReLU hidden model; a per-feature table (importance, norm, bias, superposition score, Dᵢ,
    nearest characteristic fraction); training curves of every restart.
  - *Probe panel*: pick active features and their values; see h (and Wx before the ReLU for
    the ReLU hidden model) and, for every feature, the truth, the readout Wᵀh, the readout + b,
    and the output, as grouped bars and as a table, with the input's weighted loss.
  - *Caching*: training is cached by (config, device) with `st.cache_data`, and the probe's
    model with `st.cache_resource`.

**Library:** `superposition.lab` holds all lab computation (Playground jobs via the
experiments' grid trainer, per-feature table, probe, paper-map parser, status table);
`plotting/interactive.py` holds the plotly figures (3D vectors, loss curves, probe and hidden
bars); `arrows_figure` and `weight_matrix_figure` draw single models.

**Cited runs** (resolves Phase 1 open issue 3): `superposition report --pin <exp> | --pin-all`
writes `docs/cited_runs.yaml` (run ID, profile, device, seed, commit, dirty flag, status
summary). Tables are rebuilt from cited runs, never from whatever ran last; dirty-tree runs are
refused unless `--allow-dirty`. E00 and E01 are pinned to their Phase 1 quick CPU runs.

**Docs:** DECISIONS D-201–D-206; DESIGN (lab layer, cited-run workflow); README lab section;
STUDY_GUIDE lab steps for stops 0–2; PAPER_MAP lab pages; CLAUDE.md; CHANGELOG; PLAN.

## Test status

| Tier | Command | Result |
|---|---|---|
| Lint | `make lint` (ruff check, ruff format --check, pyright basic, now including `app/`) | clean; pyright 0 errors |
| Unit + property (CI tier) | `make test` | **103 passed** (87 before + 16 new: `lab`, AppTest page tests, citation pinning) |
| Slow tier | `make test-slow` | **8 passed in 12.7 min**: E00 and E01 quick profiles pass on CPU; the three lab tests (launch, ≤ 10 s training, ≤ 10 s page); notebooks 00–02 execute (nbmake) |

New tests: `tests/unit/test_lab.py` (config validation, training shapes/determinism, the
per-feature table on a pentagon, probe stages against the model equations for all three model
kinds, paper-map parsing, status without runs); `tests/unit/test_app.py` (Home renders its four
sections and shows run commands when nothing has run; the Playground trains, probes, applies a
preset through the Train button, and shows the closed-form optimum for the linear model);
pinning tests in `tests/unit/test_runs_and_report.py`; and `tests/replication/test_lab.py`
(slow: launch and speed).

**A test-infrastructure bug found and fixed.** AppTest runs page scripts in a worker thread.
When another test module imported matplotlib first, macOS's GUI backend was selected, and the
page tests failed with "Cannot create a GUI FigureManager outside the main thread". The test
session now sets `MPLBACKEND=Agg` (D-205). The live app was never affected, because
`streamlit run` uses Agg.

**Regression check.** The phase changed shared plotting code, so I regenerated E00's and E01's
published figures from their cited runs: all six PNGs are byte-identical to `docs/figures/`.

## Acceptance table (Phase 2 exit criteria)

| Criterion | Evidence | Status |
|---|---|---|
| The app launches | `superposition app` served Home and Playground in the in-app browser; clicked through presets, Train, the ReLU-hidden and 3D views, the probe panel and the Home figure expanders without errors. `tests/replication/test_lab.py::test_superposition_app_launches` starts the server on a free port, gets `ok` from `/_stcore/health` and HTTP 200 from `/` | PASS |
| Playground n = 5, m = 2 trains in ≤ 10 s on CPU | `test_default_playground_trains_within_target_on_cpu`: default config (5,000 steps × 4 restarts) trained in **3.6 s** on an Apple M2 CPU and reached the pentagon (5/5 features represented); `test_playground_page_first_run_within_target`: the whole uncached page ran in **3.6 s**; the live app reported 4.0–4.1 s | PASS |
| SPEC §8 rules: small models only; everything else from `results/` via `runs.py`; missing artifact → exact CLI command; plotting module reused; `st.cache_resource` and `st.cache_data` used | code review of `app/`; AppTest with an empty results folder shows `uv run superposition run intro_pentagon` | PASS |

Phase 2 touched no experiment, so no experiment was re-run. The quick-profile runs cited in
REPRODUCTION.md are unchanged (E00 PASS 11/11, E01 PASS 12/12, commit `340f825`).

## Decisions made

- **D-201** Playground defaults (the p. 2 pentagon; 5,000 steps × 4 restarts ≈ 3.6 s on CPU)
  and control ranges; a form with a Train button rather than retraining on every change.
- **D-202** cited runs pinned in `docs/cited_runs.yaml`; `report --pin`; dirty runs refused.
- **D-203** lab architecture: computation in `superposition.lab`, thin pages, classic
  `pages/` layout, paper map parsed from docs.
- **D-204** Streamlit settings: no usage statistics, light theme, no run-on-save.
- **D-205** lab tests: AppTest in the fast tier, launch and speed in the slow tier, Agg backend.
- **D-206** probe-panel stages and Playground views; importance-valued colors in the lab only.

## Open issues

1. **"Nearest fraction" is a hint, not a classifier.** The lab surfaced this: in the importance-
   weighted pentagon the least important feature has Dᵢ ≈ 0.387, nearer 3/8 (square antiprism)
   than 2/5, because the two fractions are only 0.025 apart. The Playground now says so under
   its table (D-206). E03 (Phase 3) should classify geometries with more than Dᵢ alone (e.g.
   tegum components plus Dᵢ).
2. **No screenshot of the lab is published.** The README describes the lab in words; the
   in-app browser's screenshots are not saved as files. A PNG could be added to
   `docs/figures/lab/` later.
3. **Streamlit prints a Watchdog hint** at startup ("install the Watchdog module"). It is only
   about faster file-change detection during development and is harmless.
4. Carried over from Phase 1: owner placeholders (LICENSE / CITATION.cff author, notebooks'
   `<OWNER>`); CI not yet run on GitHub; E01 full profile on CPU unmeasured.

## What the owner should look at and learn from this phase

Run `uv run superposition app`, open the Playground, and work through the presets in order:
p. 2 dense → antipodal pairs → pentagon → the linear model at S = 0.9. At each one, predict
before pressing Train, then read the per-feature table. Use the probe panel on the pentagon:
activate feature 0 alone, note which two features get a positive readout (its neighbours on
the circle), then activate those two together. In the default model feature 0, which is off,
comes out at ≈ 0.5, and the two active features are almost erased. That is interference
compounding (pp. 7, 13), and it is why superposition only pays when features are sparse. Then
read `src/superposition/lab.py` (≈ 330 lines) to see how little code sits between the sliders
and the paper's equations.

Three questions:

1. In the probe panel, the linear model's "output x′ (no ReLU)" equals its "+ bias" bars. Which
   of the paper's two ReLU benefits (p. 13) is the linear model missing, and how does the probe
   show it when you activate one feature of a model trained at S = 0.9?
2. Set m = 1, n = 2, decay 1.0 and sweep 1 − S from 1 down to 0.01 with the ReLU output model.
   At which density do the two features become an antipodal pair? Compare with the phase diagram
   on p. 15 (relative importance 1.0): does the Playground agree?
3. Choose the "p. 32 · ReLU hidden layer" preset. Why can W be read directly here (the W matrix
   view), while for the ReLU output model only WᵀW is meaningful? (Hint: what does rotating the
   hidden space do to each model? Notebook 01, section 2.)
