# 08 · Project management

This chapter covers the parts of the project that were not research: how the work was cut into
phases and sessions, where the state lived between sessions, how tests were tiered, what it
cost, and which process failures happened and what rules they produced. It ends with a version
of all of this for one person working by hand. Labels follow [README.md](README.md); dates and
commits are in [01_timeline.md](01_timeline.md).

## 1. Phases

SPEC §11 cut the build into seven phases. A phase ends only when its exit criteria pass.
**[Documented: SPEC.md §11 "Phases".]**

| Phase | Scope | Exit criteria (SPEC §11) |
|---|---|---|
| 0 bootstrap | read SPEC and paper; PLAN.md; skeleton, tooling, CI, pre-commit, CLAUDE.md | `uv sync` works; lint and placeholder tests pass; CI config valid |
| 1 engine | config, data, models, train, metrics; runs, CLI, plotting; E00, E01; notebooks 00–02 | unit tier green; E00 passes; E01 reported |
| 2 lab v1 | app Home and Playground; `report` and `--publish` | app launches; Playground meets its speed target |
| 3 theory and geometry | `theory.py`; E02–E04; notebooks and app pages | (no separate list; the closing steps apply) |
| 4 dynamics, robustness, neurons | E05–E08; notebooks and app pages | (closing steps) |
| 5 mechanistic interpretability | `interp.py`, `sae.py`; E09–E11; notebooks and app pages | (closing steps) |
| 6 extensions, review, polish | E12; independent review; docs; README gallery | fresh-clone test passes |

Every phase also closes with the same steps: `make lint test` plus the quick profile of every
experiment touched, figures published, REPRODUCTION.md, CHANGELOG.md and PLAN.md updated, a
commit, and a phase report (chapter [07](07_reporting_and_review.md) §3). **[Documented: SPEC.md
§11 "Closing every phase".]**

**How it actually ran.** Phases 0–2 ran in order in one session. Phase 3 was drafted and left
unfinished at a usage limit; Phases 4 and 5 were built beside it; two rectification sessions,
which SPEC does not define, closed Phase 3 and applied six owner decisions; then Phase 6 ran.
**[Documented: 01_timeline.md "Phase by phase"; D-401; phase-3.md; rectification.md.]** SPEC
§11 allows experiments from Phase 3 on to be built in parallel as independent units, which is
the clause Phase 4 relied on (D-401). **[Documented: SPEC.md §11 "Parallelism"; D-401.]**

## 2. Sessions

Eight Claude Code sessions built the repository; the full table is in
[01_timeline.md](01_timeline.md#the-sessions). **[Documented]**

| Session | Served | Interrupted by a limit |
|---|---|---|
| 35cab3f0 | Phases 0, 1, 2 | no |
| 2df87f77 | Phase 3 draft | yes, 14:08Z; never resumed |
| 92884087 | Phase 4 | no |
| 57d571bd | Phase 5 | no |
| 5ab2965c | rectification 1 (closes Phase 3) | yes, 09:40Z; resumed 12:58Z |
| 522eb237 | rectification 2 (D-R01–D-R07, CI fix) | no |
| 1584c849 | Phase 6 and the review | yes, three times; resumed 06:24Z and 11:21Z |
| e7e666c7 | docs/TOUR.md | no |

**[Documented: S 2df87f77 @ 14:08Z; S 5ab2965c @ 09:40Z, 12:58Z; S 1584c849 @
2026-09-30T20:27Z, 2026-10-01T06:24Z, 07:16Z, 07:38Z, 11:21Z.]**

**The phase prompt.** Each phase got a prompt of the same shape: read CLAUDE.md, the SPEC
sections the phase touches, PLAN.md and the previous phase report; run `make lint test` and stop
if it fails; execute Phase N, consulting the PDF for cited pages; verify every exit criterion;
commit; write the phase report; stop. **[Documented: S 92884087 @ 16:29Z; S 57d571bd @ 19:58Z;
S 1584c849 @ 2026-09-30T20:13Z.]** Its only stop condition was a failing `make lint test`.
That is why Phase 4 went ahead when the Phase 3 report it was told to read did not exist.
**[Documented: D-401; phase-4.md banner.]**

## 3. State files

SPEC §11: "The PLAN.md checkboxes, `docs/progress/`, the git log and the test suite are the
source of truth. After any reset, re-read CLAUDE.md, PLAN.md and the latest phase report before
continuing." **[Documented: SPEC.md §11 "State across context resets".]**

| State file | Holds | Updated |
|---|---|---|
| PLAN.md | tasks mapped to SPEC sections; `[x]` done and verified, `[ ]` not done, `[~]` done with an open issue; one-line results with run IDs | at each phase close |
| `docs/progress/phase-N.md` | what happened, tests, acceptance, open issues | at each phase close |
| CLAUDE.md | commands, layout, conventions, pitfalls, workflow gotchas (at most about 150 lines) | when a new gotcha is learned |
| DECISIONS.md | every choice and why | when the choice is made |
| `docs/cited_runs.yaml` | which run each REPRODUCTION section is written from | by `report --pin` only |
| git log | what changed, in which commit, by whom | every commit |
| the test suite | what still works | every change |

**[Documented: PLAN.md legend; SPEC.md §10 (CLAUDE.md row); D-202.]**

- **Gotchas became conventions.** Phase 4 found that a run records `git.dirty` when it starts,
  so editing tracked files while queued runs start makes the later ones uncitable; it went into
  CLAUDE.md. **[Documented: phase-4.md open issue 8; CLAUDE.md "Conventions".]**
- **State outside the repository.** Rectification 1 also wrote notes to Claude Code's
  per-project memory, outside the repository, at its end. **[Documented: S 5ab2965c "Files
  created or edited" (three memory files).]** They are not part of the record and are not used
  in this cookbook.
- **State drifts.** PLAN.md's Phase 3 entry still shows E03 PASS 4/4 with the superseded run;
  the re-grade appears only under Phase 6. **[Documented: PLAN.md.]** phase-6.md reports 260
  unit tests; at `2603068` the fast tier collects 261, because `2564cf1` (an external commit
  after the session) added `test_sidebar_shows_the_lab_credit`. **[Documented: phase-6.md;
  `git show --stat 2564cf1`; collection at `2603068`.]**

To read the state at the start of a working day: **[Recommended]**

```bash
git log --oneline -10
git status --short
grep -n "\[~\]" PLAN.md
ls docs/progress
```

Expected: a clean status, the last commits you remember making, and the open `[~]` items. If a
commit is not yours, find out what it changed before anything else (section 6.2).
**[Recommended]**

## 4. Test tiers

D-310 (rectification 1) defined three tiers with pytest markers. **[Documented: D-310;
`conftest.py`; `pyproject.toml` `[tool.pytest.ini_options]`.]**

| Tier | Marker | Make target | Contents | Budget |
|---|---|---|---|---|
| fast | unmarked | `make test` (`pytest -m "not slow" -n auto`) | unit and property tests | CI; SPEC §5 says under 60 s |
| slow | `slow` | `make test-slow` | E00 replication, reduced E02–E04 (and later E12), lab tests, notebooks 00–03 | under 15 minutes |
| paper | `paper` | `make test-paper` | every whole quick profile; notebooks 04–11 | hours (estimated; never timed as a whole) |

**[Documented: Makefile; D-310; README "Quickstart".]** The paper tier is opt-in: `conftest.py`
deselects paper tests unless the `-m` expression names `paper`, so `pytest -m "not slow"` and a
bare `pytest` stay fast. **[Documented: `conftest.py` `pytest_collection_modifyitems`.]**

**Why the split.** By Phase 5 the slow tier re-ran whole quick profiles and had grown to about
an hour (estimated), which made it useless as a routine check. **[Documented: phase-4.md open
issue 6; phase-5.md open issue 6; D-310 "Rationale".]** The planning conversation recommended
splitting it. **[Planning conversation, per owner]**

**Measured times, from the phase reports.**

| When | Fast tier | Slow tier |
|---|---|---|
| Phase 0 | 2 passed | — |
| Phase 1 | 87 passed in ≈ 6 s | 5 passed in 12.5 min |
| Phase 2 | 103 passed | 8 passed in 12.7 min |
| Phase 4 | 143 passed in 12–18 s | new replication tests alone: 4 passed in 32 min |
| Phase 5 | 174 passed in 25 s | new replication tests alone: 3 passed in 20 min |
| Rectification 1 (after D-310) | 226 passed in 41 s | 11 passed in 5 min 56 s |
| Rectification 2 | 233 passed in 33 s | not run |
| Phase 6 | 260 passed in 49 s with `-n auto`; 108 s serially | 12 passed in 429 s |

**[Documented: "Test status" tables of phase-0.md … phase-6.md and rectification.md.]** Paper-tier
tests were timed one by one, never as a whole: E02–E04's took 494, 373 and 293 s.
**[Documented: phase-3.md "Test status"; phase-6.md.]** The fast tier meets SPEC's 60 s only in
parallel. **[Documented: phase-6.md open issue 7.]**

**Check the tiers yourself.** Action: list what each tier would run, without running it.
**[Recommended]**

```bash
uv run pytest --collect-only -q -m slow --nbmake tests/replication notebooks
uv run pytest --collect-only -q -m paper --nbmake tests/replication notebooks
```

Expected at `2603068`: 12 items for `slow` and 20 for `paper`, out of 32. Check: no item appears
in both. If `-m "not slow"` ever collects a replication test, the opt-in rule in `conftest.py`
is broken. **[Documented: collected at `2603068` in this cookbook's worktree.]**

```bash
make test
make test-slow
```

Expected: all pass; the slow tier took 429 s in Phase 6. **[Documented: phase-6.md.]**

```bash
make test-paper
```

[Not verified: re-runs every quick profile; hours, estimated, never timed as a whole]

## 5. Cost

### Compute

The 35 run directories under `results/` add up to 18,292 s of wall-clock: 15,479 s on the
Apple M2's CPU and 2,813 s on its GPU through MPS. **[Documented: `wall_clock_s` in every
`manifest.json`; 01_timeline.md "Compute actually used".]**

| Exp. | Runs on disk | Total s | Cited quick run s |
|---|---|---|---|
| E00 intro_pentagon | 3 | 289 | 68 |
| E01 basic_results | 2 | 2,146 | 472 |
| E02 phase_change | 5 | 3,561 | 481 (seeds 1, 2: 498 each) |
| E03 uniform_geometry | 3 | 2,183 | 346 |
| E04 nonuniform_geometry | 2 | 1,240 | 287 |
| E05 learning_dynamics | 2 | 1,034 | 518 |
| E06 adversarial | 1 | 401 | 401 |
| E07 privileged_basis | 1 | 419 | 419 |
| E08 abs_computation | 3 | 1,843 | 493 |
| E09 mech_interp_abs | 5 | 1,196 | 255 (pilot 200) |
| E10 sae_recovery | 2 | 821 | 415 |
| E11 l1_removes_superposition | 3 | 1,628 | 548 (held-out 546) |
| E12 activation_functions | 3 | 1,532 | 507 |

**[Documented: the manifests, summed per experiment directory; `docs/cited_runs.yaml`.]**
Re-running the 13 cited quick profiles once costs about 5,210 s of M2 CPU time, and the four
supplementary runs about 1,742 s more. **[Documented: sums of the cited runs' `wall_clock_s`.]**

- **Full profiles** were measured only for E00 (159 s, MPS), E01 (1,673 s, MPS) and E08 (981 s,
  MPS). The owner decided not to run the rest, and every unmeasured runtime is labeled an
  estimate. **[Documented: D-R06; runs `20260929T090615Z-eb7b9e3b`, `20260929T090856Z-6d287488`,
  `20260929T180008Z-db58449f`.]**
- **Unknown compute:** the scratchpad prototypes and development runs, the dirty-tree runs moved
  aside in Phase 1, the reviewers' own analyses, and the planning conversation's JAX run.
  **[Documented: their absence from `results/`; D-607 "Caveat".]**

### Usage limits

Usage limits were a real constraint. They stopped the Phase 3 draft for good, which is the
root of section 6.1, and split rectification 1 and Phase 6 across several sittings.
**[Documented: section 2's table.]** Background jobs kept going through a limit: Phase 6's
reviewer agents continued, and two that failed at the limit were resumed with their context.
**[Documented: phase-6.md "Before changing anything"; S 1584c849 @ 20:28Z.]**

### Human effort

Unknown. The transcripts give session start and end times, but a session's span includes idle
waiting (rectification 1 was idle 3.3 hours across its limit; Phase 6 9.9 hours overnight), and
nothing records the owner's reading, reviewing and deciding between sessions. This cookbook
gives no human-effort estimates. **[Documented: S 5ab2965c @ 09:40Z–12:58Z; S 1584c849 @
2026-09-30T20:30Z–2026-10-01T06:24Z.]**

## 6. Process failures and the rules adopted

### 6.1 The unfinished Phase 3

- **What happened.** The Phase 3 session stopped at a usage limit with code written but not
  registered, tested, run or documented. The draft reached git in two commits from outside the
  session (`4355d5d`, `acec078`), and the owner started Phase 4.
  Phases 4 and 5 left the draft untouched and said so in a banner. **[Documented: phase-3.md
  "How this phase ran"; D-401; phase-4.md and phase-5.md banners.]**
- **How it was closed.** Rectification 1 had one job: close Phase 3. It began by checking who
  else was writing to the repository and what the external commits had changed. **[Documented:
  phase-3.md "Before changing anything"; S 5ab2965c @ 07:33–07:35Z.]** The planning
  conversation had recommended not starting Phase 6 while Phase 3 was open. **[Planning
  conversation, per owner]**
- **Rule** **[Recommended]**: before starting a phase, check that the previous phase report
  exists and that PLAN.md's exit box for it is ticked. Green tests cannot catch unregistered
  code, because it has no tests to fail.

### 6.2 Concurrent writers

- **VS Code.** The editor had the repository open with `files.autoSave: afterDelay`,
  `git.enableSmartCommit: true` and `git.confirmSync: false`, which explains commits under the
  owner's identity that swept up work in progress: `e3d80a3`, `7bde9b1`, `7ac6491`, `92525fc`
  and `5e50b46`. **[Documented: phase-3.md "Before changing anything" and open issue 8.]** The
  Phase 3 draft itself reached git from outside the session too: `4355d5d` at 13:44, while the
  session was still working, and `acec078` at 14:37, after the owner's commit attempt at 14:33.
  **[Documented: git (no Co-Authored-By trailer; the session ran no `git commit`); S 2df87f77 @
  14:33–14:35Z.]** Whether the owner or VS Code's automatic sync made `4355d5d` is unknown. Two later ones, `2564cf1` and `5719ef9`, carry the generic
  subject "Refactor code structure for improved readability and maintainability" while adding
  a logo footer and docs/TOUR.md. **[Documented: their subjects in `git log`; content attribution
  Reconstructed in 01_timeline.md.]** Whether the owner changed the settings after phase-3.md
  suggested it is unknown.
- **`index.lock` collisions.** A commit failed because another git process held the lock, once
  at the end of rectification 1 and twice in Phase 6. Each time the session checked the tree
  and retried. **[Documented: S 5ab2965c @ 13:41Z; S 1584c849 @ 2026-10-01T06:39Z, 11:41Z;
  phase-6.md.]**
- **Leaked servers.** Four Streamlit servers, 10–19 hours old, were left by a test that killed
  the CLI but not its child. Fixed in `8a02181`; by rectification 2 none were running.
  **[Documented: phase-3.md open issue 9; rectification.md "Before changing anything".]**
- **Rules adopted** (from rectification 2 on): draft reports and long docs in a scratch folder
  and copy them in when finished; commit code in small pieces; launch cited runs from a clean
  detached `git worktree`; at the start, check for other writers and running processes; before
  writing a report, check that every commit since the phase began is your own.
  **[Documented: rectification.md "Before changing anything"; phase-6.md "Before changing
  anything"; S 522eb237 @ 18:11Z; S 1584c849 @ 11:48Z.]**

### 6.3 Runs outside the provenance system

- **What happened.** Prototypes and development runs ran from scratch scripts or into a scratch
  results folder, and an untracked module would not have marked a run dirty. Chapter
  [06](06_debugging_playbook.md) §8 has the details. **[Documented: D-305, D-306, D-406, D-502,
  D-505; REVIEW.md R1-10.]**
- **Rules adopted.**
  - Untracked files under `src/` and `configs/` make a run dirty (D-606, `450938e`).
    **[Documented]**
  - Cited runs are launched from a clean detached worktree with its own `uv sync`, writing to
    the main `results/` through `SUPERPOSITION_RESULTS`. First used for E08's full profile in
    Phase 4, then in every later session. The manifests' `command` fields show the worktree
    paths. **[Documented: S 92884087 @ 17:58Z; S 5ab2965c @ 07:57Z; S 522eb237 @ 18:38Z;
    S 1584c849 @ 2026-10-01T06:39Z; e.g. run `20260930T191445Z-49fd30ed`'s manifest.]**
  - CLAUDE.md records both rules, and the order "commit code → run → narrative → pin → commit
    docs". **[Documented: CLAUDE.md "Conventions".]**
  - Supplementary runs go first, so that the lab, which shows the latest run, shows the cited
    one. **[Documented: D-R05; rectification.md "Order".]**

### 6.4 CI that was never green

- **What happened.** Phase 0 validated the CI config locally with actionlint, a schema check and
  the CI commands (D-007). On GitHub every run failed at job setup, because
  `astral-sh/setup-uv@v10` did not exist. Rectification 1 found it with `gh run list`; it was
  fixed at the owner's request in `b50d062`, and the first run on `b50d062` passed both jobs.
  Phase 6's run 36859923829 passed too, and `c6faea3` records it. **[Documented: D-007;
  phase-3.md open issue 7; S 5ab2965c @ 07:35Z; S 522eb237 @ 19:42–19:54Z; phase-6.md
  "Definition of done".]**
- **Why local checks missed it.** None of D-007's three checks fetches the action from GitHub.
  **[Reconstructed: from D-007's list of checks.]**
- **Rule** **[Recommended]**: after the first push, open the CI result and read it.

## 7. A by-hand version for a one-person project

Everything above was run by an owner directing agent sessions. Here is the same discipline for
one person with no agent. **[Recommended throughout this section.]**

### 7.1 Cadence

1. **One phase at a time.** A phase is a list of exit criteria you can check. Do not start the
   next phase until its report is written and PLAN.md's exit box is ticked. **[Recommended]**
2. **A weekly review.** Once a week, read your open issues and decision entries, and close or
   carry forward each one. **[Recommended]**
3. **Start of each session:** run the state commands of section 3, then `make lint test`.
   **[Recommended]**
4. **End of each session:** commit small, write the lab log entry, and note the next step.
   **[Recommended]**

### 7.2 A lab log instead of transcripts

The transcripts were the only record of commands, errors and dead ends in this project, and
they stay private. Keep your own lab log with
[templates/LAB_LOG.md](templates/LAB_LOG.md): date, commit, commands run, run IDs, what you saw,
decisions to file, questions left. Write it as you go, not afterwards. Anything you run outside
the CLI goes in it, labeled "uncited". **[Recommended]**

### 7.3 Queue cited runs from a clean worktree

Why: a run records whether the tree was dirty when it starts. A separate checkout of a commit
cannot be made dirty by your editor while the queue runs. **[Recommended; the reason is
Documented in CLAUDE.md "Conventions".]**

1. **Commit everything the run needs.** The worktree checks out `HEAD`; uncommitted changes do
   not come along. Check that nothing is pending, including untracked files under `src/` and
   `configs/`. **[Recommended]**

   ```bash
   git status --porcelain --untracked-files=all src configs
   ```

   Expected: no output. If there is output, commit or remove it first. **[Recommended]**
2. **Create the worktree and its environment.** **[Recommended]**

   ```bash
   git worktree add --detach ../superposition-runs HEAD
   cd ../superposition-runs
   uv sync --locked
   ```

   Checked in a scratch clone, with the path filled in ([VERIFICATION.md](VERIFICATION.md)).

   Expected: a detached checkout at your commit, with its own `.venv`. If `uv sync --locked`
   fails, `uv.lock` is out of date: fix it in the main tree and commit. **[Recommended]**
3. **Run, writing into the main `results/`.** Replace the path with your main clone's absolute
   path. Use `--device cpu` for cited runs, as D-104 did. **[Recommended]**

   ```bash
   SUPERPOSITION_RESULTS=/absolute/path/to/superposition/results uv run superposition run intro_pentagon --profile quick --device cpu
   ```

   Checked in a scratch clone, with the path filled in ([VERIFICATION.md](VERIFICATION.md)).

   To queue several, run them one after another in the same shell, supplementary runs first.
   Keep editing in the main tree meanwhile; it does not affect these runs. **[Recommended]**
4. **Check the run is citable.** Open the new run's `manifest.json` and confirm `git.dirty` is
   `false` and `git.commit` is the commit you meant. **[Recommended]**

   ```bash
   python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['git'])" results/intro_pentagon/RUN_ID/manifest.json
   ```

   Replace `RUN_ID` with your run's folder name; checked that way ([VERIFICATION.md](VERIFICATION.md)).
5. **Remove the worktree.** **[Recommended]**

   ```bash
   cd -
   git worktree remove ../superposition-runs
   ```

   Checked in a scratch clone, with the path filled in ([VERIFICATION.md](VERIFICATION.md)).
6. **Write the narrative from the run, then pin it** (chapter [07](07_reporting_and_review.md)
   §1), then commit the docs. **[Recommended]**

### 7.4 One writer at a time

1. Turn off your editor's automatic commit and sync for the repository while you work in a
   terminal (in VS Code, the settings named in section 6.2). **[Recommended]**
2. Before writing a phase report, list the commits since the phase began and confirm each is
   yours and intended. **[Recommended]**

   ```bash
   git log --format="%h %an %ad %s" --date=iso -20
   ```

3. After any test that starts a server, check that nothing is left running. **[Recommended]**

   ```bash
   pgrep -fl "streamlit run"
   ```

   Expected: no output (exit status 1). If a server is listed, find which test started it.
   **[Recommended]**
4. If a commit fails on `.git/index.lock`, check that no other git process is running and that
   the working tree is intact before retrying. Never delete the lock while another git process
   is alive. **[Recommended]**

### 7.5 Cost, by hand

- Read compute from the manifests, never from memory: each has `wall_clock_s`. Label every
  runtime you did not measure as an estimate (D-R06). **[Recommended]**
- Track your own time in the lab log if you want to know it later. This project cannot tell you
  how long the human side took. **[Recommended]**
