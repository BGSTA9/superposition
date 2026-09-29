# Decisions

Dated log of every choice the paper or SPEC.md leaves open, plus any change to an acceptance
criterion (SPEC §2). Each entry records the decision, the rationale, the paper page where
relevant, and the alternatives considered. Entries are append-only; a superseded decision gets
a new entry that points back to the old one.

Page numbers refer to the 62-page PDF print of the web article (see `references/README.md`).

---

## Phase 0: bootstrap

### D-001 · 2026-09-29 · Python 3.12, `requires-python >= 3.11`
- **Decision:** Pin the development interpreter to 3.12 (`.python-version`) while declaring
  support for ≥ 3.11 as SPEC §5 requires.
- **Rationale:** 3.12 has wheels for every dependency (torch, pyarrow, streamlit) on macOS
  arm64 and Linux x86-64, and is the newest version with a long, stable track record for
  scientific stacks. pyright checks against 3.11 syntax so the floor stays honest.
- **Paper:** n/a.
- **Alternatives:** 3.11 pin (older); 3.13/3.14 (fewer battle-tested scientific wheels).

### D-002 · 2026-09-29 · CPU-only torch wheels on Linux through a uv index source
- **Decision:** `[tool.uv.sources] torch = { index = "pytorch-cpu", marker = "sys_platform ==
  'linux'" }` with the explicit index `https://download.pytorch.org/whl/cpu`. macOS and
  Windows use PyPI wheels (already CPU/MPS builds). The universal `uv.lock` records both.
- **Rationale:** SPEC §5 asks CI to use CPU-only torch wheels; the default Linux wheel on PyPI
  pulls several GB of CUDA libraries. Doing it in `pyproject.toml` (not in the CI script) means
  `uv sync --locked` gives the same environment locally and in CI.
- **Consequence:** a Linux user with a GPU who wants CUDA must override the source
  (e.g. `uv pip install torch --index-url https://download.pytorch.org/whl/cu128`). This fits
  the CPU-first stance of SPEC §5.
- **Alternatives:** install torch separately in CI (diverges from the lockfile); optional
  `cpu`/`cuda` extras with conflicting groups (more machinery than the project needs).

### D-003 · 2026-09-29 · Build backend and layout
- **Decision:** hatchling with the `src/` layout from SPEC §4; `superposition` console script
  points at `superposition.cli:app`.
- **Rationale:** the `src/` layout prevents tests from accidentally importing the working tree
  instead of the installed package; hatchling needs no configuration beyond one line.
- **Alternatives:** setuptools (more configuration), uv_build (newer, fewer users).

### D-004 · 2026-09-29 · jaxtyping annotations and ruff
- **Decision:** ignore ruff `F722` (the ignore jaxtyping documents) and write single-dimension
  shape strings with a leading space, e.g. `Float[Tensor, " inst"]`, so ruff does not report
  `F821` for them. `RUF001–003` are ignored because docstrings deliberately use the paper's
  Unicode math (Wᵢ, ‖·‖, ≥).
- **Rationale:** SPEC §5 requires jaxtyping shape annotations; this keeps lint strict elsewhere.
- **Alternatives:** ignoring `F821` globally (would hide real undefined names).

### D-005 · 2026-09-29 · pre-commit runs ruff and nbstripout from the uv environment
- **Decision:** ruff and nbstripout are `repo: local` hooks invoking `uv run …`; only the
  generic hygiene hooks (`pre-commit-hooks` v6.0.0) are fetched from GitHub.
- **Rationale:** one version source (`uv.lock`) for tools that CI also runs, so pre-commit and
  CI can never disagree about formatting.
- **Alternatives:** upstream `ruff-pre-commit` / `nbstripout` hook repos (a second version pin
  to keep in sync).

### D-006 · 2026-09-29 · Author placeholder in LICENSE and CITATION.cff
- **Decision:** Use "superposition contributors" as the copyright holder and citation author.
- **Rationale:** SPEC §11 forbids asking the owner questions mid-phase, and the owner's
  preferred attribution is not stated anywhere in the repo. The owner should replace it
  (flagged in `docs/progress/phase-0.md`). Notebooks likewise keep `<OWNER>` as the GitHub
  owner placeholder (SPEC §9).
- **Alternatives:** guessing a name from the environment (could be wrong).

### D-007 · 2026-09-29 · How "the CI config is valid" is verified
- **Decision:** Phase 0 checks the workflows three ways: (1) `actionlint` (the GitHub Actions
  linter, run through the `actionlint-py` wheel with `uvx`); (2) `check-jsonschema` against the
  SchemaStore GitHub-workflow schema; (3) running every CI step locally in the same order
  (`uv sync --locked`, ruff check, ruff format --check, pyright, pytest -m "not slow").
- **Rationale:** running the workflow on GitHub would require pushing, which is the owner's
  call; these three checks cover syntax, schema and the commands themselves.
- **Alternatives:** `act` (needs Docker, heavier than warranted).

---

## Phase 1: engine

### D-101 · 2026-09-29 · Default optimizer settings
- **Decision:** Adam, lr 1e-3, batch 1024, 10,000 steps, constant schedule, a fresh batch
  every step, final loss on ≥ 20,000 fresh samples (`TrainConfig` defaults).
- **Rationale:** the paper never states its optimizer settings (p. 10 gives only the loss).
  SPEC §6 fixes these as the E00 defaults, verified by an independent run; E00 passes with
  them (docs/REPRODUCTION.md).
- **Paper:** p. 10.
- **Alternatives:** larger batches / longer runs for sparse settings (kept as per-experiment
  overrides, used only with a documented reason); SGD (slower, more tuning).

### D-102 · 2026-09-29 · One uniform draw per entry in `IndependentSparse`
- **Decision:** draw u ~ U[0, 1) once per entry; the feature is active iff u < p and its value
  is low + (high − low)·u/p.
- **Rationale:** conditioned on u < p, u/p is exactly U[0, 1), so the joint distribution equals
  the paper's (independent zero/nonzero choice and uniform value, p. 9). On CPU, random-number
  generation dominated a training step (Phase 0 benchmark: ≈ 11 ms per 4.6 M uniforms,
  single-threaded), so this halves the dominant cost. The only side effect is value
  granularity: float32 uniforms are multiples of 2⁻²⁴, so at p = 0.001 active values take
  ≈ 16,800 distinct levels instead of ≈ 16.8 M — irrelevant to any statistic we measure.
  Unit tests check density (3 SE), range, mean and variance, plus a 64-seed z-score
  calibration test.
- **Paper:** p. 9.
- **Alternatives:** two draws per entry (2× RNG cost); Bernoulli + uniform (same cost as two).
  `CorrelatedSets`/`AnticorrelatedSets` keep explicit draws (they are cheap and need
  independent values).

### D-103 · 2026-09-29 · Seeding scheme
- **Decision:** one global `seed` per run. `derive_seed(seed, *keys)` (NumPy `SeedSequence`
  spawn keys) gives independent streams: `INIT_STREAM` (per parameter, per instance),
  `DATA_STREAM`, `EVAL_STREAM`, `PROBE_STREAM`. The linear and ReLU grids of E00/E01 use the
  same seed, so each ReLU instance starts from the same W as its linear twin and sees the same
  data stream (a paired comparison).
- **Rationale:** SPEC §3 requires instance k's initialization to depend only on (seed, k);
  independent named streams make every other random choice reproducible and isolated.
- **Alternatives:** `seed + k` arithmetic (streams can collide across runs with nearby seeds).

### D-104 · 2026-09-29 · Devices for published runs
- **Decision:** `--device auto` resolves CUDA → MPS → CPU as SPEC §5 says. The runs cited in
  docs/REPRODUCTION.md for Phase 1 were made with `--device cpu`.
- **Rationale:** CPU is the SPEC's reference hardware (every quick profile must fit 10 minutes
  on a laptop CPU), and CPU results do not depend on an accelerator's kernels. On the Apple M2
  used here, MPS is 3–4× faster for these models; results on MPS differ numerically (different
  random streams) but must satisfy the same criteria.
- **Alternatives:** publishing MPS runs (faster, less portable).

### D-105 · 2026-09-29 · How E01's acceptance criteria are operationalized
SPEC §7 states E01's criteria in words; this entry fixes their exact meaning *before* the
first E01 run. All thresholds live in `BasicResultsThresholds`. Every criterion is evaluated
on the best (lowest-loss) restart per sparsity, for both model sizes (n = 20, m = 5 and
n = 80, m = 20).
- **Represented:** ‖Wᵢ‖ > 0.5 (SPEC §3). **In superposition:** represented and superposition
  score > 0.05 (the same 0.05 SPEC uses for the linear model).
- **Linear, "exactly the top-m, score < 0.05, at every sparsity":** literal; one criterion per
  model size that passes only if every sparsity passes.
- **ReLU "matches the linear model at 1 − S = 1":** the same represented set (top m), max
  superposition score < 0.05, and best losses within 5% of each other. (The loss clause is our
  addition: "matches" should include the function it computes.)
- **"Count never decreases as sparsity rises (tolerance 1)":** at every sparsity, the count is
  ≥ (maximum count at any denser setting) − 1. Comparing against the running maximum rather
  than only the previous setting stops a slow drift of −1 per step from passing.
- **"Exceeds m at the sparsest setting":** count at 1 − S = 0.001 is > m.
- **"Superposition appears first among the less important features" (p. 11):** at the densest
  sparsity where any feature is in superposition, feature 0 is not, and every superposed feature
  is less important than every represented feature that is not (strict ordering).
- **"Antipodal pairs appear at intermediate sparsity; report where":** at least one pair of
  represented features with cos < −0.95 at some 1 − S strictly between 1 and 0.001; the
  sparsities with pairs are reported.
- **Alternatives considered:** consecutive-only monotonicity (weaker); a mean-importance
  comparison for the ordering claim (weaker, harder to interpret).

### D-106 · 2026-09-29 · E00 details the SPEC leaves implicit
- **Decision:** (1) "orthogonal" for the linear model uses the same |cos| < 0.1 as the ReLU
  S = 0 criterion; (2) the pentagon's angular gaps are taken between all five feature
  directions sorted by angle (wrap-around included); (3) the bias criterion is vacuously true
  for a model with no feature below ‖Wᵢ‖ = 0.05, and lists every checked feature otherwise;
  (4) all E00 losses — the one used to pick the best restart and the one compared with the
  closed-form optimum — use the same 100,000 fresh samples; (5) the reference losses of SPEC
  §7 are INFO rows, never criteria.
- **Rationale:** each is the most literal reading of SPEC §7 E00.

### D-107 · 2026-09-29 · E00 pentagon-loss diagnostics
- **Decision:** E00 runs report-only diagnostics of the S = 0.9 pentagon (`PentagonDiagnostics`,
  on by default): the best restart re-evaluated on 1,000,000 samples; the spread of 200
  independent 20,000-sample estimates; and the loss of every cyclic arrangement of the five
  features (12 up to rotation and reflection), each trained from an ideal pentagon.
- **Rationale:** the best pentagon's loss (≈ 0.0214) sits ≈ 9% above the SPEC's reference value
  (0.0196). The diagnostics show whether that gap is a convergence problem, sampling noise or a
  worse local optimum, and they make the evidence traceable to a logged run (SPEC §2) instead
  of an ad hoc script. Findings are in docs/REPRODUCTION.md (E00).
- **Alternatives:** leave the gap unexplained (violates SPEC §2 "Honest reporting").

### D-108 · 2026-09-29 · `report` and `figures --publish` pulled forward into Phase 1
- **Decision:** implement `superposition report` and `figures --publish` in Phase 1 (SPEC §11
  lists them under Phase 2).
- **Rationale:** closing Phase 1 requires publishing figures and updating
  docs/REPRODUCTION.md; doing that with the real commands (instead of by hand) is what SPEC §2
  provenance asks for. Phase 2 keeps them on its list to extend (e.g. app integration).

### D-109 · 2026-09-29 · Canonical orientation of 2D solutions
- **Decision:** `canonicalize_2d` rotates the most important represented feature to +y, then
  reflects so the first off-axis feature (represented features first, by index) has x < 0.
- **Rationale:** the paper canonicalizes "by rotating and flipping" without a rule (p. 22,
  p. 24). Anchoring on importance makes panels at different sparsities comparable; the
  off-axis rule makes the reflection well defined even for antipodal or collinear anchors. A
  property test checks invariance under every rotation and reflection.

### D-110 · 2026-09-29 · Metric defaults that later phases may tune
- **Decision:** `is_superposition` counts singular values above `rtol = 1e-2` × the largest;
  `geometry_graph` joins features with |Wᵢ·Wⱼ| > 0.05; `classify_fraction` tolerance 0.02;
  `neuron_polysemanticity` counts weights ≥ 0.25 × the neuron's largest, with neurons whose
  largest |weight| < 0.05 counted as dead. All are keyword arguments.
- **Rationale:** the paper gives no numeric thresholds for these. The values are provisional
  until the experiments that depend on them (E03, E07) are built; any change will be logged.

### D-111 · 2026-09-29 · Run storage and test isolation
- **Decision:** results live in `$SUPERPOSITION_RESULTS` or `<project root>/results`, where the
  project root is found by walking up from the CWD (so notebooks in `notebooks/` write to the
  same place as the CLI). The root `conftest.py` points `SUPERPOSITION_RESULTS` at a temporary
  directory for every pytest session, so replication and notebook tests never add runs to the
  owner's `results/`. Configs are shipped in the wheel (`superposition/configs/`) so a pip
  install (Colab) can run experiments.

### D-112 · 2026-09-29 · Lint: ruff format owns line length
- **Decision:** ignore `E501`; `ruff format` wraps code at 100 columns, and long string
  literals (criterion descriptions, messages) are allowed. Amends D-004.

### D-113 · 2026-09-29 · Full profiles for E00 and E01
- **Decision:** E00 `full` = 32 restarts (quick: 8, the SPEC setup). E01 `full` = 8 restarts,
  30,000 steps and 100,000 evaluation samples (quick: 4 restarts, 10,000 steps, 20,000).
- **Rationale:** the paper does not give step counts or restart counts for these figures. More
  restarts sharpen restart statistics (E00) and the best-of-restarts selection. The longer E01
  run also tests a specific hypothesis: at 1 − S = 0.001 the quick profile's best models
  represent 14/20 and 67/80 features, fewer than the p. 11–12 figures suggest, and at that
  density each feature is active about once per batch, so 10,000 steps may be too few. Quick
  profiles stay paper-faithful to SPEC §7 and are what the acceptance status is based on.
- **Alternatives:** larger batches instead of more steps (same number of feature activations,
  but fewer optimizer updates); a learning-rate schedule (changes the E00-verified defaults).

### D-114 · 2026-09-29 · Correction to D-113's rationale (no sparse-count discrepancy)
- **Decision:** keep the E01 full profile as defined in D-113, but withdraw its motivating
  premise.
- **What changed:** D-113 said the quick profile represents fewer features at 1 − S = 0.001
  than pp. 11–12 suggest. That came from reading low-resolution renders of the figures. At
  300 dpi (approximate readings by eye), the p. 11 panel shows about 12 features at full norm
  plus about 4 near ‖Wᵢ‖ ≈ 0.5 (ours: 14 with ‖Wᵢ‖ > 0.5), and the p. 12 panel about 50 at full
  norm plus a partial tail (ours: 67). There is no shortfall. The full-profile run
  (30,000 steps, 8 restarts) independently shows the counts are stable in training length
  (12/20 and 68/80). See docs/REPRODUCTION.md (E01).
- **Paper:** pp. 11–12.

---

## Phase 2: lab v1

### D-201 · 2026-09-29 · Playground defaults and control ranges
- **Decision:** defaults n = 5, m = 2, Iᵢ = 0.9ⁱ, 1 − S = 0.1, ReLU output, **5,000 steps ×
  4 restarts**, seed 0, batch 1024, lr 1e-3, 20,000 evaluation samples, CPU. Controls span the
  SPEC §8 ranges (n 2–40, m 1–10, decay 0.5–1.0); density is a log-scale select slider over
  the 1-2-3-5-7 series from 0.001 to 1 (so S = 0, 0.8 and 0.9 are all reachable); steps
  500–20,000; restarts 1–16; device `cpu` (default) or `auto`.
- **Rationale:** the defaults are E00's pentagon setting, so the first thing a learner sees is
  the p. 2 result. On an Apple M2 CPU, 4 restarts × 5,000 steps train in 3.6 s and already
  reach the pentagon (both measured by `tests/replication/test_lab.py`), leaving a wide margin
  under the SPEC's 10-second target on slower laptops; training time is linear in steps, so
  10,000 steps take about twice as long. Paper presets (p. 2 ×4, p. 11, p. 32, a 3D example)
  set the controls in one click.
- **Paper:** pp. 2, 10–11, 31–32.
- **Alternatives:** 10,000 steps by default (tighter margin); training on every slider change
  (a form with a Train button avoids surprise multi-second retrains).

### D-202 · 2026-09-29 · Cited runs are pinned in `docs/cited_runs.yaml`
- **Decision:** `superposition report --pin <exp>` (or `--pin-all`) records the run that
  docs/REPRODUCTION.md is written from, with its profile, device, seed, commit, dirty flag and
  status summary. `report` regenerates tables from cited runs; an experiment without a pin
  falls back to its latest local run, marked "(uncited)". Pinning refuses runs from a dirty
  tree unless `--allow-dirty`.
- **Rationale:** resolves Phase 1 open issue 3. The narrative quotes numbers from specific runs,
  so a later notebook run (other device, maybe a dirty tree) must not silently swap the tables
  underneath it. Storing the summary in the pin keeps the status table correct on a fresh clone
  with no `results/`, where criteria blocks are left untouched.
- **Alternatives:** "latest clean quick-profile run" as a selection rule (still changes under a
  new notebook run); a manual run-ID list in the markdown (not machine-checked).

### D-203 · 2026-09-29 · Lab architecture
- **Decision:** all lab computation lives in `superposition.lab` (training jobs, per-feature
  table, probe, paper-map parsing, status table) and in `superposition.plotting`; Streamlit
  pages in `app/` only arrange widgets and figures. Pages use the classic `app/pages/`
  multipage layout; shared UI helpers are in `app/lab_ui.py` (Streamlit puts `app/` on
  `sys.path`; pyright gets `extraPaths = ["app"]`). The Home page's paper map is parsed from
  docs/PAPER_MAP.md (one source of truth). Results are cached with `st.cache_data` (training,
  keyed by config and device; status with a 10 s TTL) and `st.cache_resource` (the best model
  for the probe).
- **Rationale:** SPEC §5 separates computation from plotting and §8 requires reusing the
  plotting module; keeping pages thin makes the logic unit-testable without a browser.
- **Alternatives:** `st.navigation` with page objects (more code for two pages); computing in
  the pages (untestable).

### D-204 · 2026-09-29 · Streamlit settings
- **Decision:** `.streamlit/config.toml` turns off usage statistics, uses the light theme with
  the palette's blue as the primary color, and disables run-on-save. `superposition app` runs
  `streamlit run app/Home.py` from the repository root with `--port` (8501) and `--headless`
  options.
- **Rationale:** privacy by default; the paper-style figures are drawn on a light surface, so
  a light app keeps them from appearing as bright cards on a dark page.

### D-205 · 2026-09-29 · Testing the lab
- **Decision:** `streamlit.testing.v1.AppTest` runs both pages headlessly in the fast tier (tiny
  training via pre-set widget state). The slow tier checks the Phase 2 exit criteria directly:
  the default n = 5, m = 2 Playground trains in ≤ 10 s on CPU; the whole uncached page runs in
  ≤ 10 s; and `superposition app` starts a server that answers `/_stcore/health` and serves
  `/`. The test session sets `MPLBACKEND=Agg`.
- **Rationale:** AppTest runs page scripts in a worker thread; on macOS the default GUI
  backend cannot create figures off the main thread, so once any test imported matplotlib first
  the page tests failed. `streamlit run` already uses Agg, so this only affects tests.

### D-206 · 2026-09-29 · Probe panel and Playground views
- **Decision:** the probe takes a multiselect of active features plus a 0–1 slider per active
  feature, and shows h (with Wx before the ReLU for the ReLU hidden model), then per feature:
  truth x, readout Wᵀh, readout + b, and the output after the activation (labelled "no ReLU"
  for the linear model), as grouped bars and as a table. The geometry view is canonicalized
  2D arrows for m = 2, a rotatable 3D plot for m = 3, a signed bar chart for m = 1, and a note
  for m ≥ 4; the ReLU hidden model also shows W directly (p. 31). Lab colors follow the actual
  importance values (equal importances share a color); the experiments' published figures keep
  their index-based colors and regenerate byte-identical.
- **Rationale:** these are exactly the stages of x′ = act(Wᵀ act(Wx) + b), so a learner can see
  where interference enters (readout), where the bias trims it, and where the ReLU removes it
  (pp. 7, 13).
- **Caveat surfaced by the lab:** "nearest fraction" labels are hints only: 2/5 and 3/8 are
  0.025 apart, and in the importance-weighted pentagon the least important feature has
  Dᵢ ≈ 0.387, nearer 3/8. The Playground says so under its table.

## Phase 4: dynamics, robustness, neurons

### D-401 · 2026-09-29 · Phase 4 built while Phase 3 is unfinished
- **Decision:** Phase 4 was executed although `docs/progress/phase-3.md` does not exist and
  PLAN.md's Phase 3 items are unchecked. The E02–E04 modules, `theory.py` additions and
  `plotting/{geometry,phase}.py` committed in `4355d5d` and `acec078` were left untouched and
  unregistered (they have no configs, tests, notebooks, app pages, runs or DECISIONS entries;
  their docstrings cite D-306, which does not exist yet). Phase 4 uses none of that code: E05(a)
  and E06 train their own uniform models and take the D* = 1/2 plateau from the paper's p. 17
  reading (SPEC §7 E03), not from an E03 run.
- **Rationale:** SPEC §11 allows experiments from Phase 3 on to be built in parallel as
  independent units, each owning its files; shared files (registry, docs) are integrated at the
  end of the phase. Phase 4's registry and doc edits add rows without changing Phase 3's.
- **Alternatives:** stop and report (the request's stop condition was a failing
  `make lint test`, which passed); finish Phase 3 first (outside the request).

### D-402 · 2026-09-29 · E05(a): setup, loss curve and what counts as a jump
- **Setup:** uniform setup of p. 17 (n = 400, m = 30, Iᵢ = 1, ReLU output) at 1/(1 − S) = 3,
  inside the paper's 1/2 plateau (roughly [2.6, 3.7], p. 17); 4 restarts × 80,000 steps, batch
  256, Adam 1e-3, a snapshot every 100 steps. The p. 26 figure runs to 100,000 steps; 80,000
  keeps the quick profile under 10 minutes on a laptop CPU and, in a prototype of the same
  setup, every restart's late jump came between 33,000 and 52,000 steps. Batch 256 (not 1024)
  is for run time: 4 × 80,000 steps at batch 1024 would take about 20 minutes.
- **Loss curve:** every snapshot's loss on one fixed batch of 8,192 inputs, so that differences
  between snapshots come from the weights only. The paper's loss drops are small (≈ 0.8% on
  p. 26; ours ≈ 0.2%), well below the training-batch noise.
- **Jump:** the first 2,000-step window in which a feature's Dᵢ changes by more than 0.15
  (SPEC), starting from a level held within ±0.05 for the preceding 5,000 steps. The
  stability condition separates jumps *between levels* (p. 26: features "jump between different
  feature dimensionalities") from the initial organization of the weights, when every Dᵢ moves
  and the loss falls steeply.
- **Coinciding loss drop:** L(t) − L(t + 2w) > 0 (the drop lags the start of the rise, so we
  measure over the jump window and one more), at least 3× the background decrease over an
  equally long stretch of the stable window, and larger than the loss's max − min during the
  stable window (its fluctuation level).
- **Provenance of the thresholds:** fixed from the paper's description and SPEC's 0.15 before
  the graded run, then checked against the prototype above (not cited).
- **Alternatives:** training-batch loss (too noisy); comparing with the window after the jump
  (penalizes drops that last longer than one window); relative-drop thresholds (arbitrary).

### D-403 · 2026-09-29 · E05(b): densities, near-zero initialization, training length
- **Decision:** two correlated sets of three features in m = 3, Iᵢ = 1, set densities
  1 − S ∈ {0.01, 0.03, 0.1} (the paper gives none), 16 restarts each, 30,000 steps, a snapshot
  every 50 steps. The graded runs start from 0.1 × the Xavier init; the full profile adds the
  plain Xavier init (scale 1) as a variant.
- **Rationale:** p. 27 describes this figure's runs as "initially, weights are initialized
  randomly close to zero", and its stages A → B start from there. With SPEC §3's Xavier init
  (std √(2/9) ≈ 0.47, norms ≈ 0.75) the weights never pass through a near-zero stage. The
  scaling keeps SPEC §3's other guarantees: instance k's init still depends only on
  (seed, k), and biases start at 0. In prototypes, densities 0.01–0.1 end as octahedra and 0.3
  in a different, PCA-like solution; 0.01 shows the paper's long aligned-triangle stage (C)
  before the rotation into the antiprism (D), which took up to ~20,000 steps.
- **Alternatives:** one density (loses the density dependence of the trajectory types);
  Xavier init only (misses stage A).

### D-404 · 2026-09-29 · E05(b): geometric stages and trajectory types
- **Decision:** each snapshot is classified by rotation-invariant tests, in this order:
  *small* (mean ‖Wᵢ‖ < 0.3); *octahedron* (all ‖Wᵢ‖ > 0.5, every feature has a partner in the
  other set with cos < −0.95, and all within-set |cos| < 0.1); *collapsed* (mean within-set
  cos ≥ 0.8 in both sets); *prism* (set centroids opposed, cos ≤ −0.9); otherwise
  *transition*. A trajectory type is the ordered list of stages lasting ≥ 1,000 steps (the
  final stage always counts).
- **Rationale:** these are the paper's A–D stages (p. 27) written as tests. In an aligned pair
  of triangles on opposite sides, cross-set partners sit at cos −1/3 or −2/3; after the 60°
  rotation into the antiprism they are exactly antipodal, which is the octahedron test. The
  1,000-step minimum keeps transient flickers out of the type while keeping the paper's
  plateaus (hundreds to thousands of steps).
- **Alternatives:** clustering trajectories (opaque); classifying by loss plateaus (the loss
  alone cannot name the geometry).

### D-405 · 2026-09-29 · E06: the attack and the vulnerability measure
- **Budget:** ε = 0.1 × the mean L2 norm of inputs drawn at that sparsity, zeros included
  (p. 29: "0.1 of the average input norm").
- **Candidates:** δᵢ = ε (WᵀW)ᵢ/‖(WᵀW)ᵢ‖₂ for every feature with a nonzero row, in the positive
  direction as written on p. 29 (it raises output i's pre-activation by the most any budget-ε
  perturbation can). δ = 0 is also allowed, since the budget is a maximum.
- **Choice:** per input, the most harmful candidate (a per-example worst case: an adversarial
  example is chosen for its input). The paper's "taking the one of these attacks which most
  harms model performance" could also mean one attack per model; per-example is the standard
  notion and never weaker.
- **Loss:** the output on x + δ against the *clean* target x.
- **Reference:** the no-superposition model W = [Iₘ | 0], b = 0, attacked with the same inputs
  and budget; its damage is exactly maxᵢ<ₘ Iᵢ ε² (unit-tested). Vulnerability = (L′ − L) /
  (L′₀ − L₀), as on the p. 29 axis. The paper does not say which model it used; any model
  without superposition that represents the top features perfectly gives the same damage.
- **Rise:** V(sparsest)/V(densest) > 2 (SPEC; paper > 3×); Spearman ρ(V, 1/D*) ≥ 0.8.
- **Not done:** gradient attacks and adversarial training (SPEC: optional).
- **Alternatives:** a single universal attack per model; the linear model or the densest
  trained model as reference (both ≈ the same damage; the explicit model makes the ratio exactly
  1 without superposition).

### D-406 · 2026-09-29 · E06: which models to attack
- **Decision:** the graded sweep is the uniform setup of p. 17 (n = 400, m = 30, Iᵢ = 1) over
  1/(1 − S) ∈ [1, 10], 12 log-spaced points × 2 restarts, 10,000 steps at batch 256. SPEC's
  example n = 20, m = 5 (Iᵢ = 0.7ⁱ) is graded over the same range (10 points × 4 restarts,
  batch 1024) and also swept over [10, 1000] (7 points), reported without grading.
- **Rationale:** the p. 29 figure plots 1/(1 − S) from 1 to 10 with 1/D* rising to about 5,
  which only the uniform setup does (n = 20, m = 5 cannot exceed 1/D* = 4 and is near 1.5 at
  1/(1 − S) = 10). Disclosure: a prototype of n = 20, m = 5 over [1, 1000] showed vulnerability
  rising to ≈ 2 by 1/(1 − S) ≈ 12 and then falling, because ε shrinks with the input norm until
  the models' negative biases absorb the whole push. The graded range follows the paper's
  figure; the extended range is reported so that this behavior stays visible.
- **Alternatives:** n = 20, m = 5 only (does not match the figure); grading the extended range
  (outside the paper's claim).

### D-407 · 2026-09-29 · E07: grid, restarts, polysemanticity, sidestepping
- **Decision:** 1 − S ∈ {1, 0.5, 0.3, 0.2, 0.15, 0.12, 0.07, 0.03}, one or more points in each
  regime p. 33 names; best of 100 restarts per setting (SPEC quick; the paper used 1,000),
  10,000 steps at batch 256. A neuron carries feature i when |W[h, i]| ≥ 25% of its largest
  |weight| (the D-110 default of `neuron_polysemanticity`); dead below 0.05; monosemantic = one
  feature. "Dense" is 1 − S = 1. The count criterion uses E01's tolerance rule.
- **Sidestepping (p. 34):** a variant with a hidden bias, h = ReLU(Wx + c), trained at
  1 − S ∈ {1, 0.1, 0.03} (20 restarts); we report the share of positive cᵢ and how often the
  hidden ReLU clips (negative pre-activation on inputs with an active feature), against the
  replication model at the same sparsity. Never the replication default.
- **Rationale:** batch 256 because 800 instances at batch 1024 would take ≈ 25 minutes on a
  CPU; restarts compensate for the noisier optimization. The paper's second sidestepping route
  ("simulate a bias by averaging over many features") is not tested.

### D-408 · 2026-09-29 · E08: setups and how the criteria are measured
- **(a)** n = 3, m = 6, dense (1 − S = 1), Iᵢ = 1, 8 restarts (the paper gives neither
  density nor importance). A neuron is *positive-side(i)* / *negative-side(i)* when, after
  footnote-15 normalization, its W₁ row reads only feature i (other weights < 25% of the
  largest) and its W₂ column writes positively, mainly to output i.
- **(b)** the p. 37 sparsities, 4 restarts × 10,000 steps, batch 1024.
- **Per-feature loss:** E[(|xᵢ| − y′ᵢ)²] from `metrics.per_feature_loss`, which conditions on
  feature i's own activity (exact decomposition; 20,000 samples per term), divided by the loss
  of predicting the mean, Var(|xᵢ|) = p/3 − p²/4. Plain Monte Carlo would see a feature active
  only once per thousand samples at 1 − S = 0.001.
- **Computed well:** ratio < 0.5 (SPEC). A feature computed on one side only (ReLU(xᵢ) but not
  ReLU(−xᵢ)) has a ratio just above 0.5, so it sits on this boundary; the precise estimator
  keeps it from crossing by noise, and the side diagnostic reports one-sided features
  separately.
- **Neurons:** counted on footnote-15-normalized W₁ with the 25% rule; a neuron whose
  normalized W₂ column is below 0.05 everywhere is dead. *Primary + secondary* (p. 38): a
  polysemantic neuron whose largest |W₁| is ≥ 2× the next.

### D-409 · 2026-09-29 · E08(b): sparse training attempts and the dead-side diagnostic
- **What was tried** (prototypes before fixing the quick profile, not cited; features computed
  well at 1 − S = 0.001, best restart): 2 restarts × 10,000 steps: 16 (with a noisier
  4,000-sample estimate); 4 restarts × 10,000: 11; 2 × 30,000 steps: 8; learning rate 3e-3: 8;
  batch 4,096: 11. More restarts or steps did not approach m/2 = 20, and longer training did
  not help at all.
- **Diagnosis:** at very sparse settings nearly every input has one active feature, so each
  *side* of feature i (xᵢ > 0 or < 0) is trained only by its own examples. With b = 0 at
  initialization (SPEC §3), a side whose gain Σₕ W₂[i,h] ReLU(±W₁[h,i]) is negative gives a
  negative pre-activation, the output ReLU clips it, and no gradient reaches it. The run reports
  (INFO `b_side_diagnostics`) how many sides were alive at initialization and how many of the
  alive and of the dead ones end up computed.
- **Variant:** INFO `b_bias_init_variant` retrains 1 − S ∈ {0.01, 0.001} with the output bias
  starting at +0.1 (no side starts dead). It is a diagnostic; SPEC §3's zero init stays the
  replication default.
- **Not done:** the reference implementations were not consulted. SPEC §2 allows it now (an
  honest attempt failed); it is left to the owner as the next diagnostic step (open issue).

---

## Phase 5: mechanistic interpretability

### D-501 · 2026-09-30 · Neuron roles from path gains, and the role taxonomy
- **Decision:** `interp.py` reads each neuron of the absolute-value model through its path
  gains G_h[i, j] = |W₁[h, i]|·W₂[j, h]: the output j receives per unit of feature i, on the
  side of feature i the neuron fires for. Roles: *dead* (every |G| < 0.05); *inhibitor(j)*
  (−minᵢ G[i, j] ≥ 0.1 and at least the largest positive gain; its *triggers* are the features
  it reads only to inhibit j, and the rest of the neuron is its secondary role);
  *positive-/negative-side(i)* (computes only i, i.e. G[i, i] ≥ 0.1, every other |G| < 25% of
  it); *asymmetric-superposition(i, j)* (computes exactly i and j, nothing else ≥ 25%, and
  "sizes flipped": |W₁[h, i]| ≥ 2 |W₁[h, j]| while W₂[j, h] ≥ 2 W₂[i, h]); *superposition(i, j)*
  (the same without the flip); *unexplained* otherwise. The motif detector checks the flip on
  the weights directly and pairs it with a neuron that reads i with the same sign and has
  G[i, j] ≤ −0.5 G_asym[i, j] (cancels at least half the interference).
- **Rationale:** path gains are exactly what the loss sees on 1-sparse inputs (which dominate
  when sparse, p. 13), are invariant to the footnote-15 rescaling, and turn the p. 39 panels
  ("orange × red ≫ 1") into numbers. SPEC §6 lists positive-side, negative-side,
  asymmetric-superposition, inhibitor and unexplained; *superposition* is added because p. 39
  names normal superposition (W₁ = [1, −1], W₂ = [1, 1]) as the contrast, and *dead* because a
  neuron with no effect is neither named nor unexplained. The 25% rule is D-408's; the factor 2
  flip is weaker than the paper's example (4 each way), and trained motifs have ≈ 3–4.
  *Triggers* were added after a pilot (uncited): without them an inhibitor's small read of the
  heavy feature (p. 39's "small amount of orange") leaked into its main side's output, so the
  secondary side was labeled unexplained and dropped from the rebuild (which then recovered
  51–84% instead of 94–97%). This does not affect naming: the inhibitor role was assigned
  either way.
- **Grading choices (fixed before the cited run):** "named" is graded over *live* neurons with
  any role other than unexplained (the stricter reading: dead neurons count in neither
  numerator nor denominator), and the share using SPEC's four roles only is reported next to
  it. Ablation agreement: for each named live neuron, zeroing it and mean-ablating it must each
  raise the loss of every feature it computes or inhibits by ≥ 5% of that feature's
  mean-prediction loss, with the largest relative rise on one of those features; threshold
  ≥ 90% of named neurons (the same bar as naming).
- **Rebuild:** side neurons ±1 in and 1 out; superposition ±1 in, 1 out; asymmetric neurons
  keep their trained input magnitudes with reciprocal outputs (own gains exactly 1); inhibitors
  keep their trained inhibition weight and trigger weights plus their idealized secondary role;
  everything else zero. Biases: 0 where a neuron writes, the mean p/2 elsewhere (a variant
  keeps the trained biases). Reported as the share of the trained model's loss reduction over
  predicting the mean (importance-weighted), overall and per feature.
- **Paper:** pp. 36, 38–39; footnote 15.
- **Alternatives:** thresholds on raw W₁/W₂ (not rescaling-invariant); roles from activation
  statistics (needs a dataset-dependent definition of "fires"); fitting the rebuild's free
  numbers (would hide what the roles fail to explain).

### D-502 · 2026-09-30 · E09: the models analyzed
- **Decision:** absolute-value models with m = 2n − 2 neurons, (n, m) ∈ {(4, 6), (5, 8),
  (6, 10)}, Iᵢ = 1, graded at 1 − S ∈ {0.3, 0.2}; 1 − S ∈ {0.5, 0.1} are trained and analyzed
  the same way but only reported. 20 restarts per setting, 10,000 steps (E08's settings); the
  best restart of each setting is analyzed and the motif is searched in every restart. The
  figure shows the best n = 6, m = 10 graded model with the most motifs.
- **Rationale:** the p. 39 model has 6 features and 10 neurons (counted from the figure), two
  fewer than computing |x| without superposition needs, and the paper calls it "only doing a
  limited amount of superposition"; it gives no density or importance. A pilot (uncited,
  m = 2n − 2 at 1 − S ∈ {0.5, 0.3, 0.2, 0.1}, 16 restarts) showed clean side neurons and no
  superposition at 0.5, the motif in most restarts with every neuron named at 0.3 and 0.2, and
  many unexplained neurons at 0.1. SPEC asks for "the regime that produces limited
  superposition", so the graded densities are the middle ones, chosen from that pilot; the
  edges are run and reported so that the choice stays visible.
- **Alternatives:** the p. 37 model (n = 100, m = 40; too large to name by hand, and E08 shows
  it computes few features); a single configuration (the motif's frequency across sizes would
  not be visible).

### D-503 · 2026-09-30 · SAEs: architecture and training
- **Decision:** f = ReLU(W_enc(h − b_dec) + b_enc), ĥ = W_dec f + b_dec, decoder columns
  renormalized to unit norm after every step, loss = ‖h − ĥ‖² (summed over dims, mean over
  batch) + λ‖f‖₁; TopK keeps the k largest activations (λ = 0). Initialization: W_dec with
  random unit columns, W_enc = W_decᵀ, biases 0. Adam, lr 1e-3, batch 1024, 10,000 steps. Dead
  latents are resampled every 2,000 steps until 60% of training: a latent that fired on fewer
  than 1e-5 of the samples since the last check gets the direction of an input drawn with
  probability ∝ its squared reconstruction error (decoder column), the same direction scaled
  to 0.2 × the mean live encoder-row norm (encoder row), encoder bias 0 and fresh Adam moments.
  SAEs train on h = Wx of the frozen toy model, with x from the toy model's own distribution.
- **Rationale:** the standard SAE of the dictionary-learning literature, which realizes the
  paper's Approach 2 (footnote 20: H ≈ AB with A sparse). The frequency rule: a development
  run (uncited) with the strict rule "never fired since the last check" resampled nothing for
  any ReLU + L1 SAE, although 13–28% of latents then fired on none of 100,000 held-out samples;
  such latents fire once in millions of inputs and are useless. 1e-5 is 50× rarer than the
  rarest true feature fires at 1 − S = 0.001, so no feature latent qualifies.
- **Paper:** pp. 42–44, footnote 20.
- **Alternatives:** removing the gradient parallel to decoder columns (common, not needed with
  renormalization for these sizes); ghost grads; resampling at a fixed schedule regardless of
  firing.

### D-504 · 2026-09-30 · E10: toy models, sweeps and how recovery is graded
- **Decision:** toy models (ReLU output model, best restart, their experiments' settings): the
  E00 pentagon (1 − S = 0.1, 8 restarts); E01's n = 20, m = 5 at 1 − S ∈ {0.01, 0.001} (4
  restarts; 12 and 14 features represented in E01's cited run); one E03 model, n = 400,
  m = 30, 1 − S = 0.1 (2 restarts, batch 256), reported only. Sweeps: dictionary sizes
  {5, 10, 20} (pentagon) and {20, 40} (n = 20), λ ∈ {0.03, 0.1, 0.3, 1}, resampling off/on,
  plus TopK k = 1 for each size; the E03 model gets 256 latents with λ ∈ {0.3, 1} and TopK
  k ∈ {8, 16}. A feature is recovered when a live decoder column has cosine > 0.95 (pentagon)
  or > 0.9 (others) with +Wᵢ; only represented features (‖Wᵢ‖ > 0.5) are scored.
- **Grading:** the best SAE of each sweep, chosen *by ground truth*, must recover 5/5 pentagon
  features at cos > 0.95 (and the toy model must represent all 5) and ≥ 90% of each n = 20
  model's represented features at cos > 0.9. Both sparse n = 20 models are graded. Because the
  selection uses the ground truth, the run also reports how many settings pass, and how many
  SAEs reconstruct well (FVU < 0.1) without recovering the features.
- **Rationale:** SPEC asks whether SAEs recover the features in a sweep; a practitioner could
  not pick by ground truth, so the robustness counts are reported next to the best. The E03
  grid is small because a 256-latent SAE on 30 dimensions costs ≈ 2 minutes of the 10-minute
  quick budget; a pilot (uncited, 5,000 steps) found ReLU + L1 SAEs at λ ∈ {0.1, 0.3, 1}
  recovering 0–9%, TopK 8 recovering 100% and TopK 16 0% (with lower reconstruction error),
  so λ ∈ {0.3, 1} and k ∈ {8, 16} are the informative settings.
- **Alternatives:** an unsupervised selection rule (e.g. the sparsest SAE below a
  reconstruction-error target; the dense ±e₁, ±e₂ code makes such rules pick non-feature
  dictionaries in 2D); loading the toy models from E00/E01 runs (couples experiments).

### D-505 · 2026-09-30 · E11: setups, λ grid and criteria
- **Decision:** the ReLU output model with λ E‖h‖₁ added to the loss, on the E00 pentagon
  (1 − S = 0.1, 16 restarts) and E01's n = 20, m = 5 at 1 − S = 0.01 (8 restarts), λ ∈ {0,
  0.003, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.8}. Per λ the
  restart with the lowest training objective (loss + λ E‖h‖₁) is kept, and every instance is
  scored on the same 200,000 samples (common random numbers). Superposed feature: represented
  (‖Wᵢ‖ > 0.5) with superposition score > 0.05 (E01's orthogonality bar). Criteria per setup:
  (1) at λ = 0, more than m features represented and WᵀW singular on them; (2) the number of
  superposed features never rises by more than 1 between consecutive λ (E01's tolerance);
  (3) some λ > 0 gives a model with at least one represented feature, none superposed and WᵀW
  invertible on them; (4) no loss drop larger than 2% between consecutive λ, and the loss at
  the first superposition-free λ above the loss at λ = 0.
- **History (disclosed):** criteria (1)–(4) were fixed before any run and not changed. A first
  development run (uncited; 8/4 restarts, λ grid {0, 0.001, 0.003, 0.01, 0.03, 0.05, 0.1,
  0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1}, losses on independent samples) gave PARTIAL 5/8:
  a pentagon restart stuck in a worse state at λ = 0.4 (a 30% loss "drop" to λ = 0.5), 2–3%
  loss wobble at small λ between restarts, and a pair of n = 20 features reappearing at
  λ = 0.2. Following SPEC §2 (implementation, then restarts): restarts were doubled and
  evaluation moved to shared samples. A second development run passed every n = 20 criterion
  but not the pentagon's "falls to none": its last two survivors formed an antipodal pair up
  to λ = 0.6 and all features were gone at 0.8. Diagnosis from its checkpoints: at λ = 0.6 the
  antipodal pair, the orthogonal pair and feature 0 alone have training objectives within
  0.4% (0.1234, 0.1236, 0.1238); the pair wins because the penalty is *smaller* for it
  (|x₀ − x₁| ≤ x₀ + x₁ when both are active), and from those numbers the single-feature state
  should be optimal only in a window near λ ≈ 0.61–0.67 that the grid skipped. The grid was
  therefore refined where the last features die (0.55, 0.65, 0.7 added; 0.15 and 1 dropped to
  keep the quick profile under 10 minutes). The cited run uses this final grid.
- **Paper:** p. 42; footnote 17 (losses at one sparsity are comparable across λ); footnote 21.
- **Alternatives:** selecting restarts by reconstruction loss alone (not what training
  minimizes); a much finer grid everywhere (over the time budget).
