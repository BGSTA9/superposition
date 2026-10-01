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

## Phase 3: theory and geometry

Phase 3 was drafted on 2026-09-29 (commits `4355d5d`, `acec078`) and left unfinished; Phases 4
and 5 were built beside it (D-401). The draft's code already cited D-301 to D-307, which were
never written. They are recorded here on 2026-09-30, when Phase 3 was closed, together with
what the draft session had seen before each choice (its transcript is the evidence).

### D-301 · 2026-09-30 · Exact expectations in closed form, not by quadrature
- **Decision:** every expected loss in `theory.py` is computed in closed form. With independent
  features that are 0 w.p. 1 − p and U[0, 1] otherwise, each ReLU-output term that depends on
  at most one other feature is a mixture, over the four activity patterns of the two features,
  of F(γ, α, β, c) = ∫₀¹∫₀¹ (γx − ReLU(αx + βy + c))² dy dx. `relu_square_integral` integrates y
  analytically, which leaves a piecewise cubic in x with known breakpoints (αx + c = 0 and
  αx + c = −β), and integrates each piece with a 3-point Gauss–Legendre rule, exact for
  polynomials up to degree 5. The 1D special cases (`own_error_1d`, `relu_square_1d`) have
  their own closed forms.
- **Rationale:** SPEC §6 allows closed form or adaptive quadrature at tolerance ≤ 1e-8 and asks
  which. Closed form is exact up to rounding, vectorizes over a whole phase diagram, and has no
  tolerance to tune. It is validated in `tests/unit/test_theory.py`: F agrees with nested
  adaptive quadrature (scipy `quad`, kinks as breakpoints, tolerance 1e-13) to 1e-11 on random
  arguments, and the exact losses agree with Monte-Carlo estimates within 4 standard errors
  (a 20-seed check during development: mean z −0.25 and +0.04, spread ≈ 1).
- **Paper:** p. 13 (the integral over x decomposes by sparsity pattern); pp. 14–15.
- **Alternatives:** adaptive quadrature per cell (slower, needs tolerance bookkeeping);
  Monte-Carlo losses (noisy exactly where families nearly tie).

### D-302 · 2026-09-30 · Theoretical families: free scalars and the optimizer
- **Decision:** a family is "W ⊥ v": the standard basis projected onto the complement of v and
  each nonzero column normalized (the p. 21 construction). Its free scalars are one weight
  scale per tegum factor (a connected component of the template's represented features) and
  one bias per feature; an unrepresented feature's output is ReLU(bᵢ). Given the scales, each
  output's bias is optimized separately (outputs are separable); the loss is the sum over
  factors. The search is a coarse grid (41 scales in [0, 2] × 121 biases in [−1.5, 1.5], one
  vectorized call), then 40 golden-section iterations on the scale around the best grid point,
  re-optimizing every bias (grid, then golden section) at each trial scale. Ties on the grid go
  to the largest index, because a ReLU output is exactly flat in its bias where it is always
  off. Families: n = 2, m = 1: W ⊥ [0, 1], [1, 0], [1, 1] (p. 14) plus McGrath's confused
  W ⊥ [1, −1] (pp. 51–53) as an optional fourth; n = 3, m = 2: W ⊥ [0, 0, 1], [1, 0, 0],
  [0, 1, 1], [1, 1, 0] (p. 15) plus the analogous confused W ⊥ [0, 1, −1]. The paper's
  diagrams use the paper's families only; the confused family is reported beside them, with
  fitted biases and with every bias fixed at 0 (McGrath's analysis "ignoring bias terms",
  p. 51).
- **Rationale:** the paper gives "closed form solutions for the loss of different weight
  configurations" but not how their scale and biases were set; optimizing them is the only
  choice that does not favor one family. A scale per factor lets an antipodal pair and a
  dedicated feature in the same model take different lengths. The unit tests check that the
  optimum is not beaten by a local search, that the dropped-feature family recovers the known
  optimum I·(p/3 − p²/4) with b = p/2, and that an optimal family's loss matches Monte Carlo.
- **Alternatives:** fixed unit norms and zero biases (ignores the bias's role, p. 13); a
  general optimizer such as L-BFGS over all scalars (local minima; slower per cell); the
  paper's notebook formulas (not consulted, SPEC §2 independence).

### D-303 · 2026-09-30 · E02 grids and the reading of "sparsity from 1.0 to 0.01"
- **Decision:** the y axis is the feature density 1 − S, log-spaced over [0.01, 1]; the x axis
  is the extra feature's relative importance, log-spaced over [0.1, 10]; 12 × 12 cells in the
  quick profile, 24 × 24 in the full one. Both counts are even, so importance 1 (an exact tie
  between "drop the extra feature" and "drop the other one") never lies on the grid. Ten
  models per cell, ReLU output model, SPEC §6 optimizer settings except the batch size (D-309).
- **Rationale:** p. 14 says "we vary the sparsity of all features from 1.0 to 0.01", while the
  axes of pp. 15–16 are labeled "Feature Density (1−S)" from 0.01 to 1; the figures decide.
  SPEC §7 asks for this reading to be documented.
- **Paper:** pp. 14–16.

### D-304 · 2026-09-30 · E02 labels and the vote
- **Decision:** a model's extra feature e is *not learned* if ‖W_e‖ ≤ 0.5; *dedicated* if it is
  represented and Σ_{j≠e} (Ŵ_e·Wⱼ)² ≤ 0.05 over the other *represented* features; otherwise
  *superposition* when its most-overlapping represented partner points the opposite way
  (cos < 0) and *confused* when it points the same way (McGrath's W₁ ≈ W₂, pp. 51–53). Per cell
  the highest-loss model is discarded (p. 14) and the other nine vote; a tie between labels
  goes to the label of the lowest-loss model among the tied ones.
- **Rationale:** thresholds 0.5 and 0.05 are SPEC §3's represented threshold and E01's
  orthogonality bar (D-105). Counting only represented partners keeps a weakly embedded,
  unlearned feature from making a dedicated feature look superposed. The unit tests label the
  theory's own optimal weights with this function and recover the theory's label in every
  cell, so both diagrams use one definition of each regime.
- **Alternatives:** the paper's continuous coloring by norm × superposition (drawn as a panel
  but not graded, because agreement needs discrete labels); averaging the nine models' norms
  and scores before labeling (blurs bimodal cells).

### D-305 · 2026-09-30 · E02 acceptance criteria
- **Decision:** for each problem size (n = 2, m = 1 and n = 3, m = 2): the three main regions
  appear in the empirical diagram and in the theoretical diagram (two criteria), and the
  empirical majority label equals the theoretical label (paper families) on ≥ 80% of cells.
  Six graded criteria in all. Reported (INFO): agreement when each cell is labeled by its
  lowest-loss model; every disagreeing cell, whether it lies on or next to a theoretical
  boundary (a different label in its 8-neighbourhood, or the two best families within 5% of
  each other); and the confused regime (empirical cells, and where the confused family wins in
  theory with and without biases).
- **Rationale:** SPEC §7 E02 verbatim. SPEC's empirical setup names n = 2, m = 1, but the paper
  draws an empirical diagram for n = 3, m = 2 too (p. 16), so both are graded, the stricter
  reading.
- **Disclosure:** the draft session ran uncited exploratory grids (CPU, 10,000 steps, batch 256)
  before this entry was written: 87.5% agreement for n = 2, m = 1 and 81.9% for n = 3, m = 2.
  The ≥ 80% bar is SPEC's; nothing was tuned to those numbers.

### D-306 · 2026-09-30 · E03: readings of the p. 17 figure and what counts as a cluster
- **Decision:** thresholds (all in `UniformGeometryThresholds`):
  - “D* ≈ 1 up to 1/(1 − S) ≈ 1.2”: D* ≥ 0.95 at every grid point with 1/(1 − S) ≤ 1.2.
  - “A plateau at 0.5 ± 0.03 over a contiguous range”: the longest run (by log width) of grid
    points within the band, with end points at least 1.25× apart in 1/(1 − S) (three points of
    the 20-point quick grid); the span is reported. The paper's span, roughly [2.6, 3.7], is
    1.42× wide.
  - “Falls beyond the plateau, to ≈ 0.25 at 8”: every point after the plateau is below 0.47,
    and D* at the grid point nearest 1/(1 − S) = 8 is 0.25 ± 0.05.
  - “Dᵢ cluster within ±0.02 of at least three characteristic fractions”: a fraction among 3/4,
    2/3, 1/2, 2/5 and 3/8 counts as a cluster when, in at least one best model, at least 20% and
    at least 8 of the represented features have it as their nearest fraction within ±0.02.
  - Σᵢ Dᵢ against m, and the tegum-factor census (D-308), are reported.
- **Rationale:** SPEC §7 E03 gives the anchor values (1 up to 1.2; 0.5 ± 0.03; ≈ 0.25 at 8;
  ±0.02, three fractions); the tolerances turn "≈", "contiguous range" and "cluster" into
  tests that the paper's own curve passes with margin.
- **Disclosure:** the draft session read p. 17 at 300 dpi and had also seen the D* curve of an
  uncited exploratory run of this setup (CPU, 10,000 steps, batch 256: plateau 2.34–3.79, D*
  0.264 at 7.85) before writing these thresholds; the cluster rule was written before any Dᵢ
  of that run were analyzed. No threshold was changed after the cited run.
- **Paper:** pp. 17–19.

### D-307 · 2026-09-30 · E04 setups and criteria
- **(a) Perturbed pentagon (p. 22):** n = 5, m = 2, Iᵢ = 1, four features at 1 − S = 0.05 and
  feature 0 at 0.1× … 10× that (21 log-spaced ratios), 32 restarts. Feature 0 is the perturbed
  one so that `canonicalize_2d` puts it at the top, as on p. 22. *Denser → neighbours repel:*
  over ratios ≥ 1, the best pentagon's mean angle between feature 0 and its two neighbours has
  Spearman ρ ≥ 0.8 with the ratio, and the densest pentagon's angle exceeds 72° + 1°. *Sparser
  → neighbours close in:* the same over ratios ≤ 1, with the sparsest pentagon below 72° − 1°.
  *Collapse:* at the sparsest ratio the best restart is two digons with feature 0 dropped
  (‖W₀‖ < 0.1); the largest ratio where that holds is reported (paper: ≈ 1/2.5 = 0.4). The
  trends use the best *pentagon* at each ratio, because p. 22 describes how the pentagon
  deforms before it snaps; every restart's loss is plotted to show the crossing curves.
- **(b) Correlated and anticorrelated sets in m = 2 (pp. 23–24):** 2 correlated pairs,
  2 anticorrelated pairs and 3 correlated pairs, set density 0.02, 16 restarts each, the best
  by loss. *Anticorrelated interfere more negatively:* mean within-set cosine lower than the
  correlated pairs'. *Orthogonal when there is room:* each of the 2 correlated pairs has
  |cos| < 0.1. *Side by side when there is not:* with 3 pairs, all 6 represented and each
  pair's members angular neighbours.
- **(c) Local almost-orthogonal bases (pp. 24–25):** 2 correlated sets of 10 in m = 10, set
  density 0.02, 8 restarts; mean |cos| within a set < mean |cos| across sets.
- **(d) Collapse (p. 25):** 3 correlated pairs in m = 2 at 12 set densities, log-spaced over
  [0.01, 0.5], 16 restarts. *Very sparse → hexagon:* at the sparsest density all 6 represented,
  pairs adjacent, no pair with cos > 0.95. *Denser → collapse:* Spearman ρ ≥ 0.8 between density
  and the mean within-pair cosine, and that cosine > 0.95 at the densest setting.
- **Best-restart selection:** every part's final losses are estimated on 200,000 fresh
  samples. At set densities of 0.01–0.02 a set is active only a few hundred times in the
  default 20,000, too few to rank geometries whose losses differ by a few percent. Part (a)
  had this in the draft; parts (b)–(d) got it on 2026-09-30 (commit `230a51d`), before any
  cited run.
- **Rationale:** the paper states no density for (b)–(d). Its p. 25 figure puts "no features
  collapsed" at 1 − S ≤ 0.05 and "all sets collapsed" at 0.25–0.5, so 0.02 sits in the
  uncollapsed regime for (b) and (c), and (d)'s sweep spans every regime it names. Thresholds
  0.1, −0.95 and 0.5 are E00's and SPEC §3's; 0.95 marks a collapsed pair; ρ ≥ 0.8 is E06's
  correlation bar.
- **Alternatives:** grading every restart rather than the best (the paper itself keeps the
  best, p. 22); a fixed angle change instead of a rank trend (arbitrary size).

### D-308 · 2026-09-30 · Naming tegum factors
- **Decision:** `metrics.component_geometry` names a tegum factor (a connected component of
  the geometry graph, edges |Wᵢ·Wⱼ| > 0.05) by its number of features k, the dimensions r it
  spans (singular values above 0.1 × the largest) and whether it is uniform (every Dᵢ within
  0.05 of r/k): (1, 1) dedicated dimension, (2, 1) digon, (3, 2) triangle, (4, 3) tetrahedron,
  (5, 2) pentagon, (8, 3) square antiprism; anything else is "k in r (uniform)" or "k in r
  (non-uniform)". `metrics.geometry_census` counts represented features per name.
- **Rationale:** resolves Phase 2 open issue 1: Dᵢ alone cannot tell 2/5 from 3/8 (0.025
  apart), but (k, r) can, and p. 20 says Dᵢ follows the factor, not the whole polytope (a
  triangular bipyramid shows up as a triangle plus a digon). Unit tests check every named
  polytope, including the tight square antiprism (Dᵢ = 3/8), and the bipyramid split.
- **Paper:** pp. 19–21.

### D-309 · 2026-09-30 · E02–E04 quick profiles train at batch 256, not SPEC §6's 1024
- **Decision:** the quick profiles of E02, E03 and E04 use batch 256 (10,000 steps, everything
  else SPEC §6). Their full profiles use batch 1024. Each config keeps a `batch1024` profile,
  identical to `quick` except for the batch size, so the comparison below can be repeated with
  `superposition run <exp> --profile batch1024` (config files may define named variant
  profiles; the CLI accepts them).
- **Rationale:** at batch 1024 the quick grids do not fit SPEC §5's 10 minutes on a laptop CPU.
  Measured on an Apple M2 CPU, from clean commit `230a51d`, seed 0, through the CLI:

  | Exp. | batch 256 (quick, cited) | batch 1024 |
  |---|---|---|
  | E02 | `20260930T090540Z-4e3325bf`, 487 s, PASS 6/6 | `20260930T075829Z-609cc6fb`, 1,597 s, PASS 6/6 |
  | E03 | `20260930T091350Z-2cb2dc61`, 365 s, PASS 4/4 | `20260930T082510Z-c635c83a`, 1,472 s, PASS 4/4 |
  | E04 | `20260930T091958Z-43c63e78`, 287 s, PASS 9/9 | `20260930T084945Z-5297207f`, 952 s, PASS 9/9 |

  The batch size changes no criterion's outcome and little else:
  - *E02:* the majority labels are identical in all 144 cells for n = 2, m = 1 and in 140 of
    144 for n = 3, m = 2; the 4 that differ are votes of 5 to 4 that flip. Agreement with the
    theory: 0.875 and 0.875; 0.819 and 0.833.
  - *E03:* the same plateau span (2.34–3.79), D* at 1/(1 − S) ≈ 7.85 of 0.2637 and 0.2633, the
    same four Dᵢ clusters (2/3, 1/2, 2/5, 3/8). D* differs by less than 0.013 everywhere except
    at the two transitions: from 1 to 1/2 (1/(1 − S) ≈ 1.4–2.1), where batch 1024 falls earlier
    by up to 0.085 (at 1.83: 0.737 against 0.652), and just after the plateau (at 4.28: 0.410
    against 0.447). Neither run is converged in the first: the draft's uncited 50,000-step,
    batch-1024 run gave 0.613 at 1.83. No criterion reads these points, and
    docs/REPRODUCTION.md reports the first as a discrepancy with the paper's curve.
  - *E04:* the same collapse threshold (a ratio of 0.251), the same order of pair collapses in
    (d), neighbour angles within 0.8° at every ratio, and the same geometries in (b) and (c).
- **Exploratory runs (uncited, from the draft session's scratchpad), consistent with this:** E02
  n = 2, m = 1 on CPU at 10,000 steps × batch 256 and on Apple MPS at 50,000 × 1024 gave
  identical majority labels in all 144 cells (agreement 0.875 both); E03 at 50,000 × 1024 on MPS
  passed the same four criteria with the same plateau span, but its triangle (2/3) cluster was
  gone, leaving exactly the three clusters the criterion needs.
- **Scope:** E05(a), E06 and E07 chose batch 256 separately (D-402, D-406, D-407); those choices
  are not re-examined here.
- **Alternatives:** batch 1024 with fewer steps (the same number of samples, fewer updates, and
  the E03 transition is already the least converged part); smaller grids (coarser diagrams
  than the paper's); an accelerator for the quick profile (SPEC §5 fixes the budget on a CPU).

### D-310 · 2026-09-30 · Test tiers: slow under 15 minutes, and an opt-in paper tier
- **Decision:** three tiers. Unmarked tests are the fast unit and property tier (CI,
  `make test`). `slow` trains small models or runs short notebooks and must stay under 15
  minutes in total on a laptop CPU (`make test-slow`): E00's replication test, reduced
  replication tests of E02–E04 (about a minute each; each checks the claims that do not depend
  on a fine sweep), the lab tests and notebooks 00–03. `paper` re-runs an experiment's whole
  quick profile, or a notebook that does (`make test-paper`): the replication tests of E01–E11
  and notebooks 04–10. Paper tests are deselected unless the `-m` expression names `paper`
  (`conftest.py`).
- **Rationale:** the slow tier had grown to more than an hour (Phase 4 open issue 6), which
  made it useless as a routine check. Opt-in keeps SPEC's commands meaning what they meant:
  `pytest -m "not slow"` (CI, and the fresh-clone test of SPEC §11) and a bare `pytest` never
  start an hour of training, while the markers stay disjoint.
- **Alternatives:** marking paper tests `slow` as well and selecting `slow and not paper`
  (then `-m slow` alone runs everything); a `--runpaper` command-line flag (a second mechanism
  besides markers).

### D-311 · 2026-09-30 · Phase diagrams page: sliders pick the cell, not clicks
- **Decision:** the lab's Phase diagrams page picks a cell with two sliders (density,
  importance), outlines it on the map, and shows every cell's labels on hover. SPEC §8 asks for
  "clicking a cell"; the page does not use clicks.
- **Rationale:** click selection through `st.plotly_chart(on_select="rerun",
  selection_mode="points")` was built and checked in a browser against the cited E02 run
  (Streamlit 1.64.0, Plotly 7.1.0). Plotly fired `plotly_selected` with the right point, but the
  page received each click one click late: the first click after loading did nothing, and every
  later click applied the previous one, whether or not the figure changed between reruns. A
  selection that shows the wrong cell is worse than none, and the fault is outside this
  repository. Sliders are exact, and AppTest can drive them.
- **Alternatives:** box selection (a drag instead of a click); a grid of 144 buttons (no
  colors); revisiting clicks when a later Streamlit release fixes the ordering.

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
  is for run time: 4 × 80,000 steps at batch 1024 would take about 20 minutes (estimated, not
  measured).
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
  CPU (estimated, not measured); restarts compensate for the noisier optimization. The paper's second sidestepping route
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
  quick budget (estimated, not measured); a pilot (uncited, 5,000 steps) found ReLU + L1 SAEs at λ ∈ {0.1, 0.3, 1}
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

---

## Rectification session 2: owner decisions

The owner's brief for the second rectification session (2026-09-30, before Phase 6) made six
decisions. Each is recorded below as an owner decision (D-R01–D-R06), followed by how it was
put into practice; choices that the brief left to me are marked *mine*. D-R07 is an
implementation decision the others needed. All of these entries were committed before the runs
that apply them.

### D-R01 · 2026-09-30 · Owner decision: E08, consult the references on dead sides; re-derive the sparse criterion
- **Owner decision:** (1) The Anthropic and ARENA reference implementations may be consulted to
  diagnose the dead-feature-side problem (D-409). This is for diagnosis only: no code is copied,
  and what was consulted and learned is recorded here. (2) The sparse criterion is re-derived
  from the paper, because SPEC's "more than m/2 features computed well" came from the spec,
  not the paper. The original criterion's result is reported next to the revised one.
- **Consulted** (2026-09-30, pages read with a web-fetch tool; nothing was copied into the
  repository):
  - *Anthropic's archived notebook* `toy_models.ipynb`
    (github.com/anthropics/toy-models-of-superposition). It has three cells and only the tied
    intro model (ReLU output, no hidden layer ReLU):
    - Xavier-normal W, zero bias, features U[0, 1] kept with probability p;
    - AdamW at lr 1e-3 with no weight decay given (so PyTorch's default, 0.01);
    - batch 1024, 10,000 steps, constant learning rate by default.
    - Its loss compares the output with `batch.abs()`, which for its non-negative inputs is the
      input itself. There is no absolute-value model in it.
  - *ARENA §1.5.4*, page 2 ("superposition in a privileged basis", the neuron-computation
    exercise), and page 1 for the base optimizer defaults:
    - the model is h = ReLU(W₁x), y′ = ReLU(W₂h + b), with W₁ `(inst, m, n)` and W₂
      `(inst, n, m)`;
    - b starts at zero;
    - W₁ and W₂ use PyTorch's Kaiming-uniform init applied to the whole 3-D tensor, whose
      fan-in is then m·n = 4,000 for both matrices. That gives weights uniform in
      ±√(6/4000) ≈ ±0.039 (std ≈ 0.022; SPEC §3's Xavier-normal std is √(2/140) ≈ 0.120,
      5.3× larger);
    - inputs are U[−1, 1] kept with probability p, and the loss is importance-weighted squared
      error against |x|;
    - the n = 100, m = 40 sweep at the p. 37 densities trains one instance per density, with
      Adam at lr 1e-3, batch 1024 and 5,000 steps by default.

    ARENA's remark that Anthropic "trained 1000 instances and chose the ones which achieved
    lowest loss" is about the ReLU hidden layer model (p. 33, our E07), not the |x| model.
- **What I learned:**
  - Neither reference contains a remedy for dead sides. Both start the output bias at 0, as
    SPEC §3 does, so D-409's mechanism applies to them too.
  - The one substantive difference is the weight scale. ARENA's |x| model starts about 5.3×
    smaller per weight, so every side's initial gain Σₕ W₂[i,h]·ReLU(±W₁[h,i]) is about 28×
    smaller in magnitude. Which sides start dead depends only on the sign of the gain, not its
    scale. But a tiny negative gain can be outweighed by a small positive output bias, which the
    feature's live side pushes up, so a dead side might revive. This is a hypothesis to test,
    not a finding.
  - Neither reference trains restarts for the |x| sweep, and the paper gives no number for it
    (p. 33's 1,000 models concern E07).
  - No other training difference (optimizer, learning rate, batch, steps, data) looks large
    enough to explain the gap. The Anthropic notebook's AdamW default weight decay shrinks
    weights by about 10% over 10,000 steps and cannot change a gain's sign.
- **Diagnostic added** (mine, fixed before the run): INFO `b_init_scale_variant`.
  - It retrains 1 − S ∈ {0.01, 0.001} with both weight matrices at 0.187 × the Xavier init
    (√((n + m)/(n·m)), which gives the reference's std), with 4 restarts, the default sweep's
    number.
  - It reports the loss, the features computed, and how many sides that were dead at
    initialization end up computed, against the default runs.
  - It is never the replication default: SPEC §3's init stays.
- **Revised criterion** (my operationalization of the owner's decision; fixed before the run,
  with the previous cited run's numbers known). The paper (p. 37) says "once the features become
  sufficiently sparse, this model, too, uses superposition to represent more features than it
  has neurons", and "gradient descent manages to find solutions that usefully approximate the
  computation even when each neuron encodes a mix of multiple features".
  - *Graded:* at the sparsest setting (1 − S = 0.001, as before), more than m = 40 features
    are *usefully computed*, meaning their per-feature loss is below 75% of the loss of
    predicting the mean.
  - *Why 75%:* it is the midpoint between computing one side of |x| exactly and not computing
    the feature at all. With ReLU(xᵢ) alone the model misses E[xᵢ²; xᵢ < 0] = p/6 of the
    baseline p/3 − p²/4, a ratio of ≈ 50%; a feature that is not computed has a ratio of 100%.
    A feature counts when it is closer to one exact side than to nothing.
  - *Why more than m:* without superposition each neuron serves one side of one feature, so
    more than m usefully computed features needs neurons shared between features. That is the
    paper's "more features than it has neurons".
  - *SPEC's original criterion* (more than m/2 features below 50%, which needs both sides) is
    reported as INFO `b_sparse_original_criterion`, with the value it would have had.
- **Footnotes 16 and 17.**
  - Footnote 16: the paper chose n = 100, m = 40 for visualization, although the model "learns
    more easily when there are more neurons". The criterion therefore stays at the paper's
    sizes.
  - Footnote 17: losses at different sparsities are not comparable. The new INFO
    `b_within_sparsity_references` therefore compares each model's loss only with references
    at its own sparsity: predicting every mean, and no superposition (both sides of the m/2 most
    important features exact, the rest at their mean).
- **Alternatives:** keep more than m/2 at 50% (SPEC's criterion, not the paper's); count
  one-sided features on 1-sparse probes (the side diagnostic, kept as INFO); compare losses
  across sparsities (ruled out by footnote 17); grade at the sparsity where superposition is
  strongest (would pick the setting after seeing results).
- **Supersedes:** the sparse criterion of D-408 (SPEC §7 E08).

### D-R02 · 2026-09-30 · Owner decision: E05, stage B as a documented variant
- **Owner decision:** The paper says several trajectories occur (pp. 27–28), and correlated
  features preferring near-orthogonal placement matches pp. 23–24. Report the fraction of
  restarts reaching the octahedron and the trajectory types seen. Treat the stage-B difference
  (our sets collapse onto near-orthogonal directions rather than opposite ends of one axis) as a
  documented variant, unless the octahedron criterion itself fails.
- **Implementation (mine):**
  - The octahedron criterion's value now includes how many restarts reach it, in total and as a
    fraction.
  - INFO `stage_b_placement` reports, per set density:
    - the runs that visit the collapsed stage, and for how long;
    - the cosine between the two sets' centroids over all collapsed snapshots (median and
      quartiles), and at each run's last collapsed snapshot (p. 28: the trajectory "depends on
      which feature geometry the model is closest to when step B ends");
    - the share of collapsed snapshots whose centroids are opposed (cos ≤ −0.9, the existing
      prism threshold).
  - No new threshold and no criterion change.
- **Paper:** pp. 23–24 ("The model ideally represents correlated features orthogonally"),
  pp. 27–28 ("one of a few different learning trajectories"; "the toy network learns a linear
  PCA solution before moving to a better nonlinear solution").
- **Supersedes:** nothing; D-403 and D-404 stand.

### D-R03 · 2026-09-30 · Owner decision: E09, the density pilot as a cited run; the inhibitor sub-claim as a discrepancy
- **Owner decision:** Re-run the density pilot (D-502) through the CLI so that it becomes a
  cited run. Report the inhibitor result (net interference near zero rather than negative) as a
  discrepancy on that sub-claim. The paper says (p. 39) that inhibition "converts positive
  interference … into negative interference".
- **Implementation (mine):**
  - Profile `pilot` in configs/mech_interp_abs.yaml reproduces the pilot's design, taken from
    the pilot script in the Phase 5 scratchpad: the three sizes, 1 − S ∈ {0.5, 0.3, 0.2, 0.1},
    16 restarts, 10,000 steps, seed 0, every density analyzed the same way.
  - It is not bit-identical to the uncited original, because the E09 module orders the graded
    densities first, which changes which instance seed each model gets.
  - It is cited as the supplementary run `mech_interp_abs:pilot` (D-R07).
  - The quick profile is re-run at the same commit so that the lab and the notebooks, which
    show the latest run, show a quick run.
  - `motif_details` now reports the spread of the net interference (best models and all
    restarts), and of the net interference divided by the interference it cancels.
  - E09's status is not changed by this decision; the discrepancy is reported on the sub-claim.

### D-R04 · 2026-09-30 · Owner decision: E10, grade an SAE selected without ground truth
- **Owner decision:** The headline result is a selection rule a practitioner could apply
  without ground truth, fixed before looking at recovery. The oracle-best SAE is reported as an
  upper bound, together with the pass rates across the sweep and the TopK k = 8 versus k = 16
  contrast.
- **Rule** (mine; fixed and committed before the run that grades it):
  - Among the SAEs of a toy model's sweep whose held-out FVU is at most 0.1 (at least 90% of the
    activation variance explained on the 100,000 held-out samples), pick the one with the lowest
    held-out L0 (mean active latents per input). Break ties by lower FVU. If no SAE reaches 0.1,
    pick the lowest FVU.
  - It uses only reconstruction and sparsity, never recovery.
  - 0.1 is the reconstruction bar that D-504's reconstruction-vs-recovery row already used.
    "The sparsest dictionary that reconstructs well enough" is the standard trade-off
    practitioners tune SAEs by.
  - TopK SAEs have L0 = k by construction, because they also fire on all-zero inputs. Counting
    that as measured penalizes TopK against L1 codes; the rule is kept as stated rather than
    adjusted.
- **Disclosure:** the previous cited run (`20260929T212259Z-fa564b17`) was known when the rule
  was chosen. That includes its summary that 7 of 20, 10 of 12 and 2 of 10 low-FVU SAEs
  recovered the pentagon's and the two n = 20 models' features, and that TopK k = 16
  reconstructed the E03 model better than k = 8 while recovering almost nothing. The rule was not
  tuned against these numbers, and its result is reported whatever it is.
- **Criteria:**
  - *Graded,* per graded toy model: the selected SAE must recover the share of represented
    features SPEC asks for, unchanged: all five pentagon features at cos > 0.95 (all five
    represented), and ≥ 90% of each n = 20 model's at cos > 0.9.
  - *INFO:* `oracle_upper_bound` (the quantity graded before), `pass_rates` (all settings, and
    by kind), `ungraded_models` (the rule applied to the E03 model), and `topk_k_contrast`.
- **Supersedes:** D-504's grading by ground truth.

### D-R05 · 2026-09-30 · Owner decision: E11, the pentagon is PARTIAL; a held-out check on fresh seeds
- **Owner decision:** Mark the pentagon PARTIAL, not PASS. Confirm the refined λ grid (D-505) on
  fresh seeds as a held-out check, and report the result either way. Frame the antipodal-pair
  finding against p. 42, where the paper says L1 works by killing features below an importance
  threshold.
- **Implementation (mine):**
  - "Falls to none" now has three outcomes:
    - PASS if some λ > 0 gives a superposition-free model with at least 2 represented
      features;
    - PARTIAL if the only superposition-free models keep one feature: a single feature cannot
      be in superposition with anything, so it is free of superposition only trivially;
    - FAIL if there is no superposition-free model.
  - This is how the owner's PARTIAL enters the acceptance record: a graded PARTIAL status
    (`acceptance.graded`), which does not count as passed. It is stricter than D-505 and
    applies to both setups alike.
  - The held-out check is the unchanged quick profile run with `--seed 1`, which gives fresh
    initializations, training batches and evaluation samples. It is cited as the supplementary
    run `l1_removes_superposition:heldout` (D-R07). It is launched before the seed-0 run, so
    that the lab, which shows the latest run, shows the cited one.
- **Paper:** p. 42.
- **Supersedes:** criterion (3) of D-505.

### D-R06 · 2026-09-30 · Owner decision: full profiles are not run; unmeasured runtimes are labeled estimates
- **Owner decision:** Do not run the full profiles now. Label every runtime that was not
  measured as an estimate.
- **Implementation:**
  - Configs, README, CLAUDE.md, the Makefile, notebooks and the progress reports now say
    "estimated" or "not measured" wherever a runtime does not come from a logged run.
  - Three runtimes in earlier DECISIONS entries (D-402, D-407, D-504) were annotated in place
    with "(estimated, not measured)"; the decisions themselves are unchanged.
  - Notebook 07's "about 8 minutes" now gives the measured 401 s.

### D-R07 · 2026-09-30 · Supplementary cited runs
- **Decision:** docs/cited_runs.yaml accepts keys `<experiment>:<label>` for a second run that
  the narrative rests on (E09's pilot, E11's held-out seed).
  - Such a run is pinned with `superposition report --pin EXP --run ID --label L`, which applies
    the same dirty-tree refusal.
  - The experiment's main citation and the status table are unchanged.
  - The run's criteria render into an `AUTO:<code>:<label>` block of docs/REPRODUCTION.md.
- **Alternatives:** a separate registered experiment for each run (SPEC §7 would then ask for
  its own notebook and app page); citing the run in prose only (it would have no status summary
  in the record).

## Phase 6: extensions, review, polish

### D-601 · 2026-10-01 · Exact losses for other output activations: split Gauss–Legendre quadrature
- **Decision:** for an output x′ = σ(WᵀWx + b) with σ ≠ ReLU, `theory.activation_square_integral`
  integrates (γx − σ(αx + βy + c))² over [0, 1]² with Gauss–Legendre rules on pieces where σ's
  argument keeps one sign. x is split where αx + c = 0 or αx + c + β = 0; for each x node, y is
  split where αx + βy + c = 0. The 1-sparse terms use the same rule in 1D. σ is the torch
  function the model trains with, evaluated in float64. Every sparsity pattern is summed exactly
  as for the ReLU (p. 13).
- **Accuracy:**
  - The pieces are polynomials for the ReLU, leaky ReLU and identity, so the rule is exact
    there; the unit test matches the ReLU closed form to 1e-12.
  - For gelu, silu, softplus and sigmoid, 24 nodes per piece agree with nested adaptive
    quadrature (scipy, 1e-13) to 1e-9 across E12's search box: scale ≤ 4, so |α|, |β| ≤ 16, and
    bias in [−8, 4].
  - 12 nodes, which E12's search uses for speed, stay within 1e-4 there; the largest errors
    (about 1e-5, GELU) sit at the box's far corners, far from any optimum.
  - Every family optimum E12 reports is re-evaluated with 24 nodes. The run records the largest
    change and whether any cell's winner changes (`quadrature_check`).
- **Rationale:** no closed form exists for these σ. SPEC §6 asks for closed forms or adaptive
  quadrature at ≤ 1e-8 for E02; E12 is an extension, and it reports its accuracy check at the
  optima instead of assuming it.
- **Alternatives:** adaptive quadrature per evaluation (orders of magnitude slower in the
  family search); Monte Carlo (noisy at the 1e-4 relative gaps between families).
- **Paper:** p. 51 (Redwood), with p. 13's decomposition.

### D-602 · 2026-10-01 · E12: what is varied, profiles, and what is reported
- **Decision:**
  - E12 is E02 with the output ReLU replaced by relu, gelu, silu, softplus, leaky_relu
    (PyTorch's default negative slope 0.01) and sigmoid.
  - Everything else is E02's: the labels and vote of D-304, the thresholds, ten models per cell,
    seed 0, the same starting weights for every activation (initialization depends only on seed
    and instance), Adam at lr 1e-3, batch 256 and 10,000 steps (D-309).
  - Each activation gets its own theoretical diagram: the same families as E02, with the exact
    loss under that σ (D-601).
  - The family search uses one box for every activation, wider than E02's: scale ≤ 4, bias in
    [−8, 4], 30 golden-section steps. Sigmoid and softplus optima need larger weights and more
    negative biases; the optima found reach a scale of about 3.1 and biases of about −5.3. A
    unit test checks that this box gives E02's ReLU labels on E02's grid.
  - **quick:** n = 2, m = 1 on an 8 × 8 grid, so that six activations fit in SPEC §5's
    10 minutes.
  - **full:** both E02 problems on E02's quick 12 × 12 grid, so the ReLU column repeats E02's
    cited run. Not run; estimated at about an hour on an Apple M2 CPU (D-R06).
  - E12 has no pass/fail (SPEC §7). It reports per activation:
    - the regions, empirical and theoretical;
    - agreement with its own theory;
    - the cells whose label changes from the ReLU's, and how;
    - losses compared within a cell, which footnote 17 allows: the same task. These are the
      best trained model against the theory, and both against the ReLU. For n = 2 the trained
      models' losses are exact.
    - the share of cells in superposition;
    - outcome by starting sign (D-605);
    - the quadrature check.
- **Rationale:** p. 51 gives only the qualitative remark; varying one thing at a time makes
  differences attributable to the activation. The theory per activation separates what the loss
  landscape prefers from what training reaches.
- **Alternatives:** reusing the ReLU theory for every activation (it would not say whether a
  difference is optimization or optimum); a finer grid in quick (over the time budget).
- **Paper:** p. 51; pp. 14–16.

### D-603 · 2026-10-01 · E03: a Dᵢ cluster counts only in its polytope (tightens D-306)
- **Finding (review R2-01):** D-306 counts a feature toward a fraction when its Dᵢ is nearest to
  that fraction within ±0.02. 2/5 and 3/8 are 0.025 apart, so their windows overlap on
  [0.355, 0.42]. One band of Dᵢ near 0.39 then counts as a pentagon cluster and a square-antiprism
  cluster at once, and the upper tail of a band below 2/3 counts as triangles. The cited run's
  "triangle" and "square antiprism" clusters are of this kind.
- **Decision:** a feature counts toward a fraction only if, in addition, its tegum factor is that
  fraction's polytope as named by D-308. For example, a feature at 2/5 must sit in a pentagon,
  and one at 1/2 in a digon. The edge threshold is E03's graded 0.05.
  - Everything else in D-306 is unchanged: ±0.02, ≥ 20% and ≥ 8 of one model's represented
    features, at least three fractions.
  - The old count is still reported, as INFO `fraction_clusters_by_value`.
- **Rationale:** p. 19 says each fraction "corresponds to a specific weight geometry"; the
  criterion should test that geometry, not a value that two geometries share. This tightens
  the criterion; nothing is loosened.
- **Known before the change:** the reviewer's analysis of the cited run suggested that only 1/2
  and 2/5 survive, so E03 is expected to become PARTIAL. The rule was fixed and committed before
  the run that grades it.
- **Supersedes:** D-306's cluster rule.

### D-604 · 2026-10-01 · E09: motifs must carry their roles; "named" means nothing unexplained
- **Findings (review R4-01, R4-02):**
  - `find_motifs` checks a weight pattern only. It does not require the asymmetric neuron to
    store just the two features, nor the partner to be an inhibitor, and it reported motifs on
    neurons the role table calls unexplained.
  - The inhibitor role names almost any neuron whose most negative gain dominates. On random
    weights about half of the live neurons get a role, most of them inhibitors.
- **Decision:**
  - A motif counts only if its asymmetric neuron carries asymmetric-superposition(heavy, victim),
    as its role or its secondary role, and its partner is inhibitor(victim)
    (`interp.role_consistent_motifs`). The graded `motif_found` and the motif statistics use
    these. The weight-pattern counts are reported next to them.
  - A neuron is "named" for the graded 90% only if nothing in it is unexplained
    (`NeuronRole.fully_named`): an inhibitor whose other content fits no role does not count.
    Ablation agreement is measured over these neurons.
  - New INFO `named_at_initialization`: the same rules on every restart's untrained weights, as
    a baseline for the 90% bar.
- **Rationale:** the paper's motif (p. 39) is one neuron storing two features plus an inhibitor;
  a role the rules give to random weights is weak evidence. Both changes tighten the criteria.
- **Supersedes:** the motif counting and the "named" count of D-501.

### D-605 · 2026-10-01 · E02: held-out seeds for the n = 3 agreement; starting sign; exact losses
- **Findings (review R1-01–R1-03):**
  - The n = 3 agreement (0.819 against the 0.80 bar) rests on one seed. A resampling of each
    cell's models puts it near 0.80 on average. The batch-1024 comparison of D-309 shares seed 0
    and therefore the same starting weights.
  - In n = 2 cells where the theory has superposition, whether a model gets there seems to
    depend on the starting sign of its two weights.
  - Trained n = 2 models beat the restricted families in more cells than the docs said.
- **Decision (fixed before the runs):**
  - **Held-out seeds.** The unchanged quick profile is run with `--seed 1` and `--seed 2`, and
    pinned as `phase_change:seed1` and `phase_change:seed2` (D-R07). REPRODUCTION.md reports
    the n = 2 and n = 3 agreement of all three seeds and their mean.
  - **If the n = 3 mean over the three seeds is below 0.80:**
    - E02's section and README say the n = 3 agreement criterion does not hold across seeds;
    - review finding R1-01 stays open, with the status left to the owner as D-R05 did for E11;
    - the cited seed-0 run's grade is not edited.
  - **New INFO rows:**
    - `initial_sign`: in the cells where the theory has superposition, how many models reach it,
      split by the starting sign of their two weights (n = 2; models keep their starting
      weights in the checkpoint);
    - `exact_loss_vs_theory`: per cell, the best trained model's exact loss against the
      cheapest paper family's, for n = 2.
  - E12 reports `initial_sign` per activation.
- **Paper:** pp. 14–16.

### D-606 · 2026-10-01 · Untracked code makes a run dirty
- **Finding (review R1-10):** `runs.git_info` ignored untracked files, so a run that imported a
  new module or read a new config not yet committed was recorded as `dirty: false`.
- **Decision:** untracked files under `src/` and `configs/` now make the tree dirty. Other
  untracked files, such as notes, do not.
- **Effect on cited runs:** no cited run used an untracked module (review R1-10 checked
  E00–E02; every later cited run was launched from a clean detached worktree).

### D-607 · 2026-10-01 · How the Phase 6 review was run
- **Decision:**
  - Four reviewers with fresh context, each a separate agent that saw none of this
    repository's history, took one group each:
    - engine and E00–E02;
    - E03–E05;
    - E06–E08;
    - E09–E11.
  - Each re-derived its experiments from the cited paper pages first, then audited the code
    and checked every number in docs/REPRODUCTION.md against the cited runs.
  - They worked read-only on the repository, without the reference implementations, and
    returned numbered findings with evidence.
  - A fifth review of E12 ran after E12's code was written.
  - docs/REVIEW.md lists every finding and its resolution.
- **Rationale:** SPEC §11 asks for a reviewer with fresh context; splitting by experiment
  group keeps each review within one reading of the relevant pages.
- **Caveat:** the reviewers' own analysis scripts are not part of the repository. A number in
  REVIEW.md that comes from them is labeled as the reviewer's analysis. A number that the docs
  rely on was first re-computed by code in this repository and a logged run.
