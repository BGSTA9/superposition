# 04 · Scaffolding: the environment and repository setup

This chapter covers Phase 0: the Python environment, the lockfile, lint, tests, pre-commit, CI
and git hygiene that every later phase relied on. **Part A** is the real sequence, with
evidence; each of its commands is a check you can re-run in a fresh clone of this repository.
**Part B** is the same setup by hand, from an empty directory, for your next project.

Labels follow [README.md](README.md). Terms are in the [glossary](../GLOSSARY.md). Three tooling
terms: a **lockfile** (`uv.lock`) pins every package version; **CI** (continuous integration)
re-runs lint and tests on GitHub for every push; a **pre-commit hook** runs checks before git
accepts a commit.

## How Phase 0 actually ran

Session 35cab3f0 got one prompt for Phases 0 and 1 at 08:06Z on 09-29 and committed the Phase 0
report at 08:22:57Z. **[Documented: S 35cab3f0 @ 08:06Z; `1815b59`]**

| UTC (09-29) | What happened | Evidence |
|---|---|---|
| 08:06–08:07 | Reading the PDF failed (`pdftoppm` missing); `uv` and `poppler` installed with Homebrew, outside the project | [S 35cab3f0 @ 08:06–08:07Z]; [phase-0.md](../progress/phase-0.md) "Open issues" |
| 08:08–08:09 | Paper read; `pdftotext` for pp. 41–62 | [S 35cab3f0 @ 08:08Z] |
| 08:13–08:14 | Latest release tags of `actions/checkout`, `astral-sh/setup-uv`, `pre-commit-hooks` looked up via the GitHub API; skeleton folders; `.python-version` = 3.12; LICENSE, package `__init__`, first `uv sync`; torch entries in the lockfile inspected | [S 35cab3f0 @ 08:13–08:14Z] |
| 08:14–08:16 | CPU vs MPS benchmark and RNG profile in the scratchpad (`bench.py`, `prof.py`), outside the provenance system | [S 35cab3f0 @ 08:14–08:16Z] |
| 08:17–08:18 | CI workflows; module stubs, CLI skeleton, placeholder tests; one ruff fix | [S 35cab3f0 @ 08:17–08:18Z] |
| 08:19 | Section headings of pp. 38–62 mapped to pages with a `pdftotext` loop | [S 35cab3f0 @ 08:19Z] |
| 08:21–08:22 | `uv lock --check`, `uv sync --locked`, `make lint test`; actionlint, workflow schema, `cffconvert --validate`; `.DS_Store` untracked; pre-commit installed and run, then run again | [S 35cab3f0 @ 08:21–08:22Z] |
| 08:22 | Four logical commits in one command (`7673c23`, `9e13751`, `d6bd81f`, `f31595b`), then `1815b59` | git |

**[Documented: the rows' evidence.]** Unknown: what the 08:13Z tag lookup returned, why
pre-commit was run twice, and when `pyproject.toml`, the Makefile, `.gitignore` and
`.pre-commit-config.yaml` were written (the extraction lists those writes without times; they
existed before the 08:21Z check). All Phase 0 files were written first and split into commits
at the end, within three seconds; Phase 1 did the same ([05_engine.md](05_engine.md)).
**[Documented: commit times 11:22:25–11:22:27 +03:00]** Besides the setup files (and LICENSE,
in `9e13751`), `f31595b` added PLAN.md, CLAUDE.md, references/README.md, CITATION.cff, CHANGELOG,
DECISIONS (D-001–D-007) and doc skeletons; the author is "superposition contributors" because
the owner's attribution was never stated (D-006), still open. **[Documented: phase-0.md; D-006; phase-6.md
open issue 8]**

```bash
git show --stat 7673c23 9e13751 d6bd81f f31595b 1815b59
```

Expected: five commits dated 2026-09-29 11:22 +0300 with the files named above.

## Part A · As done here: check the state in a fresh clone

Run from the repository root of a fresh clone.

### A1. Python: a 3.12 pin with a 3.11 floor (D-001)

`.python-version` pins 3.12; `requires-python = ">=3.11"`; pyright checks 3.11 syntax so the
floor stays honest. **[Documented: [D-001](../DECISIONS.md); `9e13751`]**

```bash
cat .python-version
grep -n 'requires-python\|pythonVersion' pyproject.toml
```

Expected: `3.12`; `>=3.11` and `"3.11"`. If uv cannot find 3.12: `uv python install 3.12`.

### A2. The lockfile and CPU-only torch on Linux (D-002)

`[tool.uv.sources]` routes torch to an explicit `pytorch-cpu` index with
`marker = "sys_platform == 'linux'"`. The universal lock records both builds, so
`uv sync --locked` gives CI and a laptop one environment and CI never downloads CUDA wheels.
Phase 0 reported 119 locked packages, 109 installed on macOS. **[Documented: D-002; phase-0.md]**

```bash
uv lock --check
uv sync --locked
grep -n 'version = "2.14.0+cpu"' uv.lock
grep -q nvidia uv.lock || echo "no CUDA packages in uv.lock"
uv run python -c "import torch; print(torch.__version__); assert not torch.cuda.is_available()"
```

Expected: `Resolved 119 packages`; the sync finishes; at least one `2.14.0+cpu` line; the
message; a torch version and no assertion error (on a CUDA machine the last check fails by
design: it is CI's guard). If `uv lock --check` fails, `pyproject.toml` changed without a
re-lock: run `uv lock` and commit both files together.

### A3. Build backend, layout and CLI (D-003, D-111)

hatchling with the `src/` layout, which stops tests importing the working tree by accident; the
console script points at `superposition.cli:app`; configs are force-included in the wheel so a
pip install can run experiments. **[Documented: D-003; D-111; `3194d04`]**

```bash
grep -n -A2 '^\[build-system\]' pyproject.toml
uv run superposition --help
uv run superposition list
```

Expected: `hatchling.build`; the commands `list run figures report app`; thirteen experiments,
each `not run` in a fresh clone.

### A4. What git ignores (`7673c23`)

The paper PDF (licensing), `results/`, `*.pt`, caches and `.DS_Store`; the commit also untracked
a `.DS_Store` that the owner's `7ac9cef` had added. `references/README.md` explains how to get
the same 62-page numbering. **[Documented: `7673c23`; references/README.md]**

```bash
git check-ignore -v references/toy_models_of_superposition.pdf results/ .DS_Store
git log --oneline -- .DS_Store
```

Expected: three lines naming the `.gitignore` rules; commits `7673c23` and `7ac9cef`.

### A5. Lint and types (D-004, D-112)

ruff and pyright (basic). jaxtyping strings like `Float[Tensor, "inst hidden feat"]` are not
Python expressions, so ruff reports `F722`; it is ignored, and single-dimension strings get a
leading space (`" inst"`) because ruff reads `"inst"` as an undefined name (`F821`). `RUF001–003`
are ignored for the paper's Unicode math. **[Documented: D-004]** `E501` joined the ignores at
08:41Z in Phase 1, committed alone as `529f724`. **[Documented: D-112; S 35cab3f0 @ 08:41Z]**

```bash
grep -n '^ignore = ' pyproject.toml
make lint
```

Expected: `["F722", "E501", "RUF001", "RUF002", "RUF003"]`; `All checks passed!`,
`files already formatted`, `0 errors`. On a format failure, `make format` and review the diff.

### A6. Test tiers (D-310)

Phase 0 had one `slow` marker and two placeholder tests. **[Documented: phase-0.md]** The slow
tier grew past an hour (Phase 4 open issue 6), so rectification session 1 split it: unmarked =
fast tier (`make test`, CI); `slow` under 15 minutes (`make test-slow`); `paper` re-runs whole
quick profiles (`make test-paper`) and is deselected unless `-m` names it. **[Documented: D-310;
`8e6a429`]**

```bash
make test
uv run pytest --collect-only -q tests/replication
uv run pytest --collect-only -q -m paper tests/replication
```

Expected: no failures (260 tests at the Phase 6 head per phase-6.md, plus
`test_sidebar_shows_the_lab_credit` from `2564cf1`); `8/20 tests collected (12 deselected)`;
`12/20 tests collected (8 deselected)`. SPEC §5's "under 60 s" holds only with `-n auto`
(phase-6.md open issue 7). Do not use `make test-slow` or `make test-paper` as setup checks.

### A7. The root `conftest.py`: isolation, kernel, Agg

Each piece was added for a concrete reason. **[Documented: the evidence per item]**

1. **Results isolation** (D-111, `162bd1a`): unless `SUPERPOSITION_RESULTS` is set, a test
   session points it at a fresh temporary directory, so tests never add runs to `results/`.
2. **Notebook kernel** (`162bd1a`): nbmake starts a kernel named `python3`; a user-level
   kernelspec pointing at another Python would win, so conftest puts this interpreter first on
   `JUPYTER_PATH`. Added after inspecting the kernelspecs [S 35cab3f0 @ 08:55Z].
3. **Agg backend** (D-205, `338280c`): AppTest runs page scripts in a worker thread; macOS's GUI
   backend cannot draw off the main thread, so the page tests failed [S 35cab3f0 @ 12:46Z].
   conftest sets `MPLBACKEND=Agg` before matplotlib is imported.

Right after `make test` in a fresh clone, before any experiment run:

```bash
test ! -e results && echo "the test run wrote nothing to results/"
```

Expected: the message. Otherwise some code bypasses `results_root()`.

### A8. The Makefile and the rebuild kit target

Phase 0 targets: `setup lint format test test-slow run-quick figures report app clean-results`;
`test-paper` came with D-310; `rebuild-test` was added for this cookbook's
[rebuild kit](../../rebuild/README.md). **[Documented: phase-0.md; `8e6a429`; Makefile]**

```bash
make help
make rebuild-test
```

Expected: one line per target. `make rebuild-test` reports `64 failed`, all `NotImplementedError`,
and `make` exits with an error: the kit's starting line, not a broken repository.

### A9. pre-commit from the uv environment (D-005)

Hygiene hooks come from `pre-commit-hooks` v6.0.0 (YAML, TOML, end of file, whitespace, merge
conflicts, files over 500 KB except `uv.lock`); ruff and nbstripout are `repo: local` hooks
calling `uv run`, so their versions come from `uv.lock` and cannot disagree with CI. The hooks
ran on every Phase 0 commit and later blocked an owner commit of the Phase 3 draft on two ruff
errors. **[Documented: D-005; phase-0.md; S 2df87f77 @ 14:33–14:35Z]**

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

Expected: nine hooks `Passed` (the first run fetches `pre-commit-hooks` from GitHub). "Files
were modified" means a hook fixed something: read `git diff`.

### A10. CI, checked without pushing (D-007), and the tag that did not exist

`ci.yml`: a lint job and a test job (assert CPU-only torch, then `pytest -m "not slow" -n auto`);
`slow.yml`: the slow tier weekly and on demand. Pushing was the owner's call, so Phase 0 checked
the workflows with actionlint, the SchemaStore schema, and a local run of every CI step.
**[Documented: `d6bd81f`; D-007]**

```bash
uvx --from actionlint-py actionlint .github/workflows/ci.yml .github/workflows/slow.yml
uvx check-jsonschema --builtin-schema vendor.github-workflows .github/workflows/ci.yml .github/workflows/slow.yml
uvx cffconvert --validate
```

Expected: actionlint silent, exit 0; `ok -- validation done`; `Citation metadata are valid`.
The local CI steps are A2's sync and torch check, A5 and A6.

**All three checks passed, and CI was still broken.** `d6bd81f` pinned `astral-sh/setup-uv@v10`,
a tag that does not exist (setup-uv publishes no floating major tags after v7). Every CI run
from the first push on 09-29 failed at job setup; rectification session 1 counted 14. At the
owner's request on 09-30, session 522eb237 read the failed log (`gh run view --log-failed`),
listed setup-uv's releases through the GitHub API, pinned `v10.2.0` (`b50d062`), pushed, and
watched the first run on `b50d062` pass both jobs. Phase 6 later recorded run 36859923829 green
at `41aad4b`. **[Documented: `b50d062`; phase-3.md open issue 7; S 522eb237 @ 19:42–19:54Z;
phase-6.md]** Who pushed first is unknown. Why `@v10` was written is not recorded; most likely
the 08:13Z lookup returned a v10.x release and a floating major tag was assumed.
**[Reconstructed: from the 08:13Z lookup and the pin in `d6bd81f`]**

```bash
git show d6bd81f:.github/workflows/ci.yml | uvx --from actionlint-py actionlint -
git ls-remote --tags https://github.com/astral-sh/setup-uv refs/tags/v10 refs/tags/v10.2.0
```

Expected: actionlint accepts the broken file (exit 0); `ls-remote` prints one line, for
`v10.2.0` only. Lesson: linters check syntax, not that an action version exists; check each
`uses:` tag with `git ls-remote` and watch the first CI run. **[Recommended]**

## Part B · By hand in a new repository

**[Recommended]** throughout. Run in a new, empty directory that is not inside a git repository.
Checked end to end on 2026-10-02 with uv 0.12.19 on macOS ([VERIFICATION.md](VERIFICATION.md));
packages resolved today will be newer than this repository's (torch 2.14.1 instead of 2.14.0).

### B1. Create the package (D-001, D-003)

```bash
uv init --package --build-backend hatch --python 3.11 toyrepro
cd toyrepro
uv python pin 3.12
```

Check: `cat .python-version` prints `3.12`; `grep requires-python pyproject.toml` shows
`>=3.11`; `.git` and `.gitignore` exist (`uv init` ran `git init`). `uv init` copies your git
name and email into `authors`: edit them if you will publish.

### B2. Dependencies

```bash
uv add "torch>=2.4" "numpy>=1.26" "einops>=0.8" "jaxtyping>=0.2.34" "matplotlib>=3.8" "pyyaml>=6.0" "typer>=0.12" "tqdm>=4.66"
uv add --dev "pytest>=8.0" "pytest-xdist>=3.5" "hypothesis>=6.100" "ruff>=0.6" "pyright>=1.1.380" "pre-commit>=3.7"
uv run python -c "import torch, einops, jaxtyping; print(torch.__version__)"
```

Check: a torch version prints. Add `nbmake nbstripout ipykernel` with your first notebook, and
other libraries (SPEC §5's list) only when code needs them.

### B3. CPU-only torch on Linux (D-002)

Append to `pyproject.toml`:

```toml
[tool.uv.sources]
torch = [{ index = "pytorch-cpu", marker = "sys_platform == 'linux'" }]

[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true
```

```bash
uv lock
grep -n '+cpu"' uv.lock
grep -q nvidia uv.lock || echo "no CUDA packages in uv.lock"
uv lock --check
```

Check: a `+cpu` torch appears, no `nvidia` packages, the lock check passes.

### B4. Ignore the PDF, results and caches, before the first commit

```bash
printf 'references/*.pdf\nresults/\n*.pt\n.pytest_cache/\n.ruff_cache/\n.hypothesis/\n.DS_Store\n' >> .gitignore
mkdir -p references
touch references/paper.pdf
git check-ignore -v references/paper.pdf
```

Check: the `references/*.pdf` rule is named.

### B5. Lint, type and test settings (D-004, D-112, D-310)

Append to `pyproject.toml`:

```toml
[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "W", "I", "UP", "B", "SIM", "NPY", "RUF"]
ignore = ["F722", "E501", "RUF001", "RUF002", "RUF003"]

[tool.pyright]
typeCheckingMode = "basic"
pythonVersion = "3.11"
include = ["src", "tests"]
venvPath = "."
venv = ".venv"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra --strict-markers"
markers = ["slow: trains small models; the whole tier stays under 15 minutes"]
```

Write `src/toyrepro/shapes.py` with one function,
`feature_norms(W: Float[Tensor, "inst hidden feat"]) -> Float[Tensor, "inst feat"]` returning
`W.norm(dim=1)`, and `tests/test_shapes.py` checking that the norms of `torch.eye(3)[None]` are
all 1.

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest -q
```

Check: all pass. Drill: annotate a parameter `Float[Tensor, "inst"]`; ruff reports `F821`;
`" inst"` fixes it.

### B6. A root `conftest.py` (D-111, D-205)

Set `os.environ.setdefault("MPLBACKEND", "Agg")` at import time, and in `pytest_configure` point
your results variable at `tempfile.mkdtemp()` unless it is set. Your run code must read that
variable (answer key: `results_root()` in `src/superposition/runs.py`).

```bash
uv run pytest -q
test ! -e results && echo "tests wrote nothing to results/"
```

### B7. A Makefile

Recipe lines start with a tab. Start with the two targets you will run most (copy `setup`,
`format` and the tier targets from this repository's Makefile when you need them):

```make
lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run pyright

test:
	uv run pytest -m "not slow" -n auto
```

```bash
make lint test
```

Check: both pass. `missing separator` means spaces instead of a tab.

### B8. pre-commit from the uv environment (D-005)

Copy this repository's `.pre-commit-config.yaml`; until you have notebooks, drop the nbstripout
hook and `jupyter` from `types_or`.

```bash
git add -A
uv run pre-commit install
uv run pre-commit run --all-files
```

Check: every hook `Passed`. pre-commit only sees files git knows, hence `git add -A`.

### B9. CI, checked before the first push (D-007 and the `b50d062` lesson)

Copy this repository's `ci.yml` into `.github/workflows/` (one job is enough: checkout,
setup-uv, `uv sync --locked`, ruff, pyright, the CPU-only torch assertion, the fast tests).

```bash
uvx --from actionlint-py actionlint .github/workflows/ci.yml
git ls-remote --tags https://github.com/astral-sh/setup-uv refs/tags/v10.2.0
git ls-remote --tags https://github.com/actions/checkout refs/tags/v7
```

Check: actionlint silent; each `ls-remote` prints exactly one line. Empty output means the tag
does not exist and CI will fail at "Set up job".

### B10. Commit, push, watch CI

```bash
git commit -m "chore: bootstrap the project"
```

Check: the hooks pass and `git status --short` is empty.

```bash
git push -u origin main
gh run list --limit 1
```

[Not verified: needs a GitHub remote and credentials.] Check: the run's conclusion is `success`.

## What to take from this chapter

- SPEC §5 had already fixed the tools; Phase 0 had only D-001–D-007 left to decide.
  **[Documented: SPEC §5; phase-0.md "Decisions made"]**
- The one setup failure that reached GitHub passed every local check and broke every CI run
  from the first push until `b50d062`. **[Documented: phase-3.md open issue 7; `b50d062`]**
- Add results isolation on day one, not after the first stray run. **[Recommended]**
