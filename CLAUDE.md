# CLAUDE.md

Agent guide for `superposition`, an independent replication of *Toy Models of Superposition*
(Elhage et al., 2022). **SPEC.md is the single source of truth**; this file is a summary.
After any context reset, re-read this file, `PLAN.md` and the latest `docs/progress/phase-N.md`.

## Commands

```bash
uv sync                                    # create/refresh the env from uv.lock
make lint                                  # ruff check + ruff format --check + pyright (basic)
make test                                  # fast tier: pytest -m "not slow" (CI)
make test-slow                             # slow tier (< 15 min): E00, reduced E02–E04, lab, notebooks 00–03
make test-paper                            # paper tier (hours, estimated): every quick profile + notebooks 04–10
uv run superposition list                  # registered experiments
uv run superposition run intro_pentagon --profile quick [--seed N] [--device auto|cpu|mps|cuda]
uv run superposition figures intro_pentagon --run latest --publish
uv run superposition report [--pin EXP | --pin-all]   # cite runs; regenerate REPRODUCTION tables
uv run superposition app [--port 8501] [--headless]   # Streamlit lab (Home, Playground, Phase diagrams, Geometry, Non-uniform, Dynamics, Neurons, Adversarial, SAE lab)
```

## Layout

- `src/superposition/`: `config` (dataclass configs + YAML profiles), `data` (feature
  distributions), `models` (all params have a leading instance dim), `train` (vectorized
  multi-instance trainer), `metrics` (pure functions), `theory`, `adversarial`, `interp`, `sae`,
  `runs` (run dirs/manifests), `report`, `plotting/`, `experiments/` (one module per experiment
  + registry), `cli`.
- `configs/<experiment>.yaml`: `base` block + `profiles: {quick, full}` overrides.
- `results/<experiment>/<run_id>/`: run outputs (git-ignored). `docs/figures/<exp>/`: published.
- `tests/unit/` (fast, CI) · `tests/replication/` (`slow`, or `paper` for a whole quick
  profile; `paper` is opt-in: deselected unless `-m` names it, see `conftest.py`).
- `notebooks/`: learning path, committed without outputs (nbstripout).
- `docs/`: REPRODUCTION, PAPER_MAP, DESIGN, DECISIONS, STUDY_GUIDE, GLOSSARY, REVIEW,
  `progress/phase-N.md`.

## Conventions

- Docstrings name the paper section and page: `Paper: Demonstrating Superposition, p. 10`.
  Pages refer to the 62-page PDF in `references/` (git-ignored; never commit it).
- Shapes via jaxtyping: `Float[Tensor, "inst hidden feat"]`. Single-dim strings take a leading
  space (`Float[Tensor, " inst"]`) so ruff does not read them as names.
- Multi-instance math with `einops.einsum` over named dims; no Python loops over instances or
  features in hot paths (init loops over instances are fine).
- Metrics are pure functions of tensors; plotting functions take arrays and return figures.
- No magic numbers: thresholds and hyperparameters live in config dataclasses / YAML.
- Paper-faithful defaults stay separate from variants; extensions never change a replication
  default. Under-specified choices go in `docs/DECISIONS.md` (decision, rationale, page,
  alternatives).
- Conventional commits (`feat:`, `fix:`, `test:`, `docs:`, `refactor:`, `chore:`), one logical
  change each. Published figures ≤ 500 KB.
- Cite only runs made from a clean, committed tree (`manifest.json` → `git.dirty: false`):
  commit code → run → write the narrative → `superposition report --pin <exp>` (records the run
  in `docs/cited_runs.yaml`; refuses dirty runs) → commit docs. Unpinned experiments fall back
  to their latest run, marked "(uncited)".
- A run records `git.dirty` when it *starts*: while runs are queued back to back, do not edit
  tracked files (new untracked files are fine), or the later runs become uncitable.
- Tests never write to `results/`: the root `conftest.py` points `SUPERPOSITION_RESULTS` at a temp
  dir and registers a `python3` kernel for this interpreter (a user-level `python3` kernelspec
  would otherwise shadow the project env in nbmake). It also sets `MPLBACKEND=Agg`: AppTest runs
  pages off the main thread, where macOS's GUI backend cannot draw.

## Research integrity (SPEC §2, non-negotiable)

- Numbers and figures come only from code in this repo and trace to a logged run.
- Never weaken an acceptance criterion to make it pass; changes need a DECISIONS.md entry.
- Never special-case code to satisfy a test or hard-code expected outputs.
- Implement from the paper. The Anthropic notebook and ARENA may be consulted only to diagnose
  a failed replication after an honest attempt, and the consultation is logged.
- A result that does not reproduce: check the implementation → more restarts/steps → document
  → report PARTIAL/FAIL with evidence.

## §3 pitfalls (implement exactly)

- Data: xᵢ = 0 with probability S, else U[0, 1]. Code uses `feature_probability` p = 1 − S.
  Absolute-value task: nonzero values U[−1, 1], target |x|.
- Loss: Σᵢ Iᵢ(xᵢ − x′ᵢ)² → sum over features, **mean over batch** (one value per instance),
  **sum over instances** for `backward()`.
- Use `+b` everywhere. The phase-change section writes ReLU(WᵀWx − b) (p. 14); note the sign
  where it matters.
- ReLU hidden layer model has **no hidden bias** (p. 34).
- Init: Xavier normal with std √(2/(n + m)) per instance, biases zero. **Never** call
  `nn.init.xavier_normal_` on a 3-D `(inst, m, n)` tensor (fan-out would include the instance
  dim). Instance k's init depends only on (seed, k).
- Dᵢ = ‖Wᵢ‖² / Σⱼ(Ŵᵢ·Wⱼ)², with Dᵢ = 0 when Wᵢ = 0. D* = m/‖W‖²_F.
- Formal superposition test (WᵀW not invertible) applies to represented features only.
- 2D hidden spaces are hard to optimize: use restarts and keep the lowest loss (p. 23).
  Canonicalize 2D solutions by rotation/reflection before comparing (p. 22).
- Losses at different sparsities are not comparable (footnote 17).
- Feature order is most → least important in every figure.
