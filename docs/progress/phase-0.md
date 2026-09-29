# Phase 0 report: bootstrap

Date: 2026-09-29 · Commits: `7673c23`..`f31595b` (4 commits) · SPEC §11 Phase 0

## What was built

- **Environment.** A uv project (`pyproject.toml`, universal `uv.lock` with 119 packages, Python
  3.12 pin, `requires-python >= 3.11`) with every SPEC §5 core and dev dependency. torch
  resolves to 2.14.0 from PyPI on macOS and to `2.14.0+cpu` from the PyTorch CPU index on Linux
  (D-002), so CI never downloads CUDA wheels.
- **Skeleton.** `src/superposition/` with stubs for every module in SPEC §4; each stub's
  docstring names the paper pages and the phase that implements it. `configs/`, `app/`,
  `notebooks/`, `tests/unit/`, `tests/replication/`, `docs/progress/`, `docs/figures/`.
- **CLI.** A Typer skeleton (`superposition --help`, `superposition list`); the real commands
  arrive in Phase 1.
- **Tooling.** ruff (lint + format, jaxtyping-aware ignores, D-004), pyright basic, pytest with a
  registered `slow` marker, Makefile targets `setup lint format test test-slow run-quick figures
  report app clean-results`.
- **CI.** `.github/workflows/ci.yml` (lint job: ruff check, ruff format --check, pyright; test
  job: asserts CPU-only torch, then `pytest -m "not slow" -n auto`) and `slow.yml` (weekly +
  manual: replication and notebook tests).
- **pre-commit.** Hygiene hooks (pre-commit-hooks v6.0.0: yaml, toml, EOF, whitespace, merge
  conflicts, files > 500 KB except `uv.lock`) plus ruff and nbstripout run from the uv env
  (D-005). Installed into `.git/hooks`; it ran on every Phase 0 commit.
- **Docs.** PLAN.md (every task mapped to SPEC sections and acceptance criteria), CLAUDE.md
  (≈100 lines), README, `references/README.md` (URL, arXiv link, BibTeX from p. 55, how to get
  the same page numbering, related implementations), LICENSE (MIT), CITATION.cff, CHANGELOG.md,
  DECISIONS.md (D-001 to D-007), and skeletons of REPRODUCTION, PAPER_MAP (fully mapped with
  page ranges), DESIGN, STUDY_GUIDE, GLOSSARY (all SPEC §10 terms, with pages), REVIEW.

## Test status

| Check | Command | Result |
|---|---|---|
| Lockfile consistent | `uv lock --check` | pass |
| Environment from lockfile | `uv sync --locked` | pass (109 packages installed) |
| Lint | `make lint` (ruff check, ruff format --check, pyright) | pass; pyright 0 errors |
| Placeholder tests | `make test` | 2 passed |
| pre-commit | `pre-commit run --all-files` | all hooks pass |

## Acceptance table (Phase 0 exit criteria)

| Criterion | Evidence | Status |
|---|---|---|
| `uv sync` works | `uv sync --locked` succeeds from the committed `uv.lock`; `uv lock --check` passes | PASS |
| Lint and the placeholder test suite pass | `make lint test`: ruff clean, 0 pyright errors, 2/2 tests | PASS |
| The CI config is valid | `actionlint` (via `uvx --from actionlint-py`): 0 errors in 2 files; `check-jsonschema --builtin-schema vendor.github-workflows`: ok; every CI step run locally in order: pass (D-007) | PASS |

Also verified: `CITATION.cff` validates against CFF schema 1.2.0 (`cffconvert --validate`).

## Decisions made

All in `docs/DECISIONS.md`:
- D-001 Python 3.12 pin with a ≥ 3.11 floor.
- D-002 CPU-only torch on Linux through a uv index source (so CI and local share one lockfile).
- D-003 hatchling + `src/` layout.
- D-004 ruff ignores F722 for jaxtyping; single-dim shape strings get a leading space.
- D-005 pre-commit runs ruff/nbstripout from the uv env (one version source).
- D-006 "superposition contributors" as the LICENSE/CITATION author placeholder.
- D-007 how CI validity is verified without pushing.

## Open issues

- **Owner attribution.** LICENSE and CITATION.cff say "superposition contributors". Replace with
  your name if you want personal attribution (D-006).
- **Workflows have not run on GitHub.** The repo has an `origin` remote
  (`Argo-Navis-Lab/superposition`), but nothing was pushed; validity rests on actionlint, the
  JSON schema and a local run of every step. The first push will be the real test.
- **Environment note.** `uv` and `poppler` (for reading the PDF) were installed with Homebrew on
  this machine; neither is a project dependency.
- **CPU speed observation for Phase 1.** A quick benchmark on this M2 showed CPU training steps
  dominated by random-number generation (~11 ms per 4.6 M uniforms, single-threaded), and MPS
  3–4× faster than CPU even for the smallest models. Phase 1 designs the data sampler and the
  quick profiles with this in mind.

## What the owner should look at and learn from this phase

Look at `PLAN.md` first: it is the whole project on two pages, and each experiment line carries
the claim you will later check against the paper. Then skim `CLAUDE.md`'s "§3 pitfalls" list,
which is the short version of everything that can silently go wrong in a replication like this.
`docs/PAPER_MAP.md` and `docs/GLOSSARY.md` are the two documents to keep open while reading the
paper.

Three questions to check your understanding:

1. The loss is summed over features, averaged over the batch, then **summed** over instances
   before `backward()`. Why sum rather than average over instances, and why does it not matter
   for Adam whether the instances are summed or averaged? (Hint: what does Adam do to a
   gradient's scale?)
2. Why would `nn.init.xavier_normal_` on a `(n_inst, m, n)` tensor silently give the wrong scale,
   and why does seeding each instance from (seed, instance index) matter for experiments that
   add restarts later?
3. The paper's formal definition of superposition is "WᵀW is not invertible" (p. 8). Why does
   this repository apply that test to *represented* features only? What would the unrestricted
   test say about a linear model with n = 20 and m = 5?
