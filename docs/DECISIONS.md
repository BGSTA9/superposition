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
