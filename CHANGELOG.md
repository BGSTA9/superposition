# Changelog

All notable changes to this project are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow the build phases
in SPEC.md §11 until a first release.

## [Unreleased]

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
