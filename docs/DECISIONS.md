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
