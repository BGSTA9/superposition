# 01 · Timeline: how this repository was actually built

This chapter is the retrospective record. It lists every phase and session in the order they
happened, with the commits, the main outputs and the problems. A narrative of what went wrong
and how each problem was handled follows the tables. Other chapters cite this one when they
need a date or a session.

Every row and claim carries one label (see [README.md](README.md)):
**[Documented]** means the cited evidence shows it directly. **[Reconstructed]** means it was
inferred, and the text says from what. **[Recommended]** is advice, not history.

**Evidence used.** The git history (131 commits, `90e2033`..`2603068`) is the authoritative
timeline. The phase reports in [`docs/progress/`](../progress/) are the build's own account. The
run manifests under `results/` (git-ignored, 35 run directories) give run times and outcomes.
The Claude Code session transcripts were read only through a private extraction script, and are
cited as `[S <session prefix> @ <UTC time>]`. The planning conversation that produced SPEC.md is
not visible; what the owner reports about it is cited as [Planning conversation, per owner].

All times are UTC. The owner's machine runs on Europe/Istanbul time (+03:00), which is the
offset git records, so a commit stamped 16:02 +03:00 appears here as 13:02.

## The sessions

Eight Claude Code sessions built the repository. Their transcripts are on the owner's machine
in `.cookbook-evidence/transcripts/` (git-ignored, never committed).

| Prefix | Full session ID | Session title | Start → end (UTC) | What it served |
|---|---|---|---|---|
| 35cab3f0 | 35cab3f0-84b6-4883-ab05-1df3c520919f | Superposition phases 0-1 | 09-29 08:06 → 13:06 | Phases 0 and 1, then Phase 2 on a second prompt at 12:32 |
| 2df87f77 | 2df87f77-e28f-4f1c-8d46-0153558c6a11 | Phase 3 implementation and verification | 09-29 13:23 → 14:35 | the Phase 3 draft, stopped by a usage limit at 14:08 |
| 92884087 | 92884087-b2df-41f8-8d49-ca10216ca5e7 | Phase 4 implementation and verification | 09-29 16:29 → 18:40 | Phase 4, built beside the unfinished Phase 3 |
| 57d571bd | 57d571bd-e4d7-43c8-bdba-02df2d4f2489 | Phase 5 implementation and verification | 09-29 19:58 → 22:04 | Phase 5, Phase 3 still open |
| 5ab2965c | 5ab2965c-bf23-4ff1-87e6-bce20d27601a | Phase 3 rectification session 1 | 09-30 07:33 → 13:44 | closing Phase 3 |
| 522eb237 | 522eb237-167a-4c21-95d7-874cc9f6245d | Second rectification session | 09-30 18:10 → 19:54 | six owner decisions (D-R01–D-R06), then the CI fix |
| 1584c849 | 1584c849-8374-433b-a605-08187fc4987c | Phase 6 implementation and verification | 09-30 20:13 → 10-01 13:11 | Phase 6, the independent review, the push |
| e7e666c7 | e7e666c7-9e1e-49b4-b317-d7bef6447f70 | Visitor tour guide for paper study | 10-01 16:37 → 16:54 | docs/TOUR.md |

[Documented: the transcripts' first and last timestamps and session titles, from the extraction
script.]

No transcript exists for the planning conversation in which SPEC.md was drafted, or for the
owner's own work outside Claude Code. [Documented: absence from `.cookbook-evidence/`.]

## Phase by phase

| Date (UTC) | Session or phase | Commits | Key outputs | Issues |
|---|---|---|---|---|
| before 09-29 | Planning (outside Claude Code) | none | SPEC.md drafted with Claude; the paper read, a JAX verification run of E00, a closed-form linear optimum, E03 thresholds read from p. 17 [Planning conversation, per owner] | No transcript. What was tried and discarded is unknown. |
| 09-29 07:32–07:35 | Owner setup | `90e2033`, `7ac9cef` | the repository and SPEC.md v1.0 [Documented: git] | `7ac9cef` also committed a `.DS_Store`, untracked again in `7673c23` [Documented: git stat] |
| 09-29 08:06–08:23 | Phase 0, session 35cab3f0 | `7673c23`..`f31595b`, report `1815b59` (5) | uv project and lockfile, package skeleton, Makefile, CI and pre-commit, PLAN.md, CLAUDE.md, doc skeletons, D-001–D-007 [Documented: phase-0.md] | The PDF reader needed poppler, installed with Homebrew mid-session [S 35cab3f0 @ 08:06–08:07Z]. CI could not run on GitHub yet; its validity rested on actionlint and a local run (D-007). |
| 09-29 08:25–10:02 | Phase 1, session 35cab3f0 | `529f724`..`8c69d54` (18) | utils, config, data, models, train, metrics, runs, acceptance, plotting, E00 and E01, CLI, report, notebooks 00–02; 87 unit tests; E00 PASS 11/11 and E01 PASS 12/12 on quick and full profiles [Documented: phase-1.md] | A density test failed at one seed (a 3.45σ false positive), handled openly; D-113's premise was wrong and D-114 corrected it; E00's S = 0.9 loss did not match SPEC's reference value [Documented: phase-1.md "Open issues"] |
| 09-29 12:32–13:06 | Phase 2, session 35cab3f0 | `af93f25`..`cc20819` (7) | the lab's Home and Playground pages, the probe panel, `report --pin` and `docs/cited_runs.yaml`; 103 unit tests [Documented: phase-2.md] | `e3d80a3` was committed under the owner's identity from outside the session, sweeping up staged docs and a local `.claude/launch.json` [Documented: phase-2.md "Commit note"; S 35cab3f0 @ 13:05Z] |
| 09-29 13:23–14:35 | Phase 3 draft, session 2df87f77 | `4355d5d`, `acec078` (both from outside the session) | theory.py, E02–E04 modules and two plotting modules, written but not registered, configured, tested, run or documented [Documented: phase-3.md "How this phase ran"] | `4355d5d` (13:44, most of theory.py) was committed from outside the session while it was still working [Documented: git; the session ran no `git commit`]. Usage limit at 14:08, with nothing committed by the session [S 2df87f77 @ 14:08Z]. At 14:33 the owner's commit of the rest was blocked by pre-commit on two ruff errors; the session fixed them, and `acec078` (14:37) followed [S 2df87f77 @ 14:33–14:35Z; git]. No Phase 3 report. |
| 09-29 16:29–18:40 | Phase 4, session 92884087 | `fd60e7c`..`ed3ca74` (16) | E05–E08, `adversarial.py`, three lab pages, notebooks 06–08; 143 unit tests; E05 PASS 2/2, E06 PARTIAL 2/4, E07 PASS 3/3, E08 PARTIAL 2/4 [Documented: phase-4.md] | Started while Phase 3 was open (D-401): the prompt asked the session to read a phase-3.md that did not exist [S 92884087 @ 16:29Z]. `7bde9b1` and `7ac6491` committed report drafts with placeholders from outside the session. E08's computation in superposition did not reproduce. |
| 09-29 19:58–22:04 | Phase 5, session 57d571bd | `a8cb60e`..`9afa841` (12) | `interp.py`, `sae.py`, E09–E11, the SAE lab page, notebooks 09–10; 174 unit tests; E09, E10 PASS 3/3; E11 PASS 8/8 [Documented: phase-5.md] | Phase 3 still unfinished. E11's λ grid was refined after development runs (D-505). E10 graded the SAE chosen with ground truth (D-504). E09's density came from an uncited pilot (D-502). |
| 09-30 07:33–13:44 | Rectification session 1, session 5ab2965c | `617e88a`..`3c05719` (28) | Phase 3 closed: E02–E04 registered, tested, run from a clean worktree and pinned; batch-256 settled by logged CLI runs (D-309); test tiers split (D-310); notebooks 03–05 and three lab pages; 226 unit tests; E02 PASS 6/6, E03 PASS 4/4, E04 PASS 9/9 [Documented: phase-3.md] | Session limit 09:40, resumed 12:58 [S 5ab2965c @ 09:40Z, 12:58Z]. Two more commits from outside the session (`92525fc`, `5e50b46`). Four leaked Streamlit servers found; the test bug behind them fixed (`8a02181`). CI had failed on every run since the first push. |
| 09-30 18:10–19:54 | Rectification session 2, session 522eb237 | `81ad83c`..`67298aa` (12), then `b50d062` | owner decisions D-R01–D-R06 and D-R07; graded PARTIAL criteria and supplementary citations; E05 and E08–E11 re-run from a clean worktree; 233 unit tests [Documented: rectification.md] | E10 → PARTIAL 1/3 and E11 → PARTIAL 7/8 after re-grading. The reference implementations were consulted for E08, for diagnosis only (D-R01). CI fixed by pinning `astral-sh/setup-uv@v10.2.0` at the owner's request [S 522eb237 @ 19:42Z]. |
| 09-30 20:13 – 10-01 13:11 | Phase 6, session 1584c849 | `ad8ad78`..`41aad4b` (23), then `c6faea3` | E12; five fresh-context reviewers, 59 findings (REVIEW.md); E03 → PARTIAL 3/4; E02 held-out seeds; README gallery; fresh-clone test; 260 unit tests; CI green on GitHub [Documented: phase-6.md] | Three session limits [S 1584c849 @ 2026-09-30T20:27Z, 2026-10-01T07:16Z, 07:38Z]; the reviewers kept running in the background. Two `index.lock` collisions with VS Code's git (phase-6.md). Two decisions left to the owner. |
| 10-01 13:06–16:01 | Phase 6 session, a last request | `2564cf1` (from outside the session) | the lab's sidebar logo and lab-name credit | The session made the change and a test [S 1584c849 @ 13:09–13:11Z] but did not commit. It landed in `2564cf1` with a generic message. [Reconstructed: from the commit's files and the transcript's edits.] |
| 10-01 16:37–16:55 | Tour, session e7e666c7 | `5719ef9` (from outside the session) | docs/TOUR.md, plus links in README, CLAUDE.md, STUDY_GUIDE and CHANGELOG | Same pattern: committed with a generic message just after the session ended. [Reconstructed: from the commit's files and the session's edit list.] |
| 10-01 19:43–19:44 | Owner | `635584e`, `2603068` | COOKBOOK_SPEC.md; `.cookbook-evidence/` git-ignored | No transcript. [Documented: git] |

**Commit authorship.** Every commit carries the owner's identity: `90e2033` under a GitHub
no-reply address, consistent with the repository being created on GitHub [Reconstructed: from
the address], and the other 130 under the owner's local git identity [Documented: `git log
--format='%an'`]. 118 of the 131 carry a `Co-Authored-By: Claude` trailer, the ones a session
made. The other 13 are the owner's own or
VS Code's commits: `90e2033`, `7ac9cef`, `e3d80a3`, `4355d5d`, `acec078`, `7bde9b1`,
`7ac6491`, `92525fc`, `5e50b46`, `2564cf1`, `5719ef9`, `635584e` and `2603068`. [Documented:
`git log --format='%h %(trailers:key=Co-Authored-By,valueonly)'`.]

**Unit-test count over time:** 2 (Phase 0) → 87 (Phase 1) → 103 (Phase 2) → 143 (Phase 4) → 174
(Phase 5) → 226 (rectification 1) → 233 (rectification 2) → 260 (Phase 6). [Documented: the
"Test status" table of each report.] At `2603068` the fast tier collects 261, because
`2564cf1` added `test_sidebar_shows_the_lab_credit` after the Phase 6 report. [Documented:
`git show --stat 2564cf1`; `uv run pytest -m "not slow" --collect-only -q`.]

## Statuses over time

| Exp. | First reported | Changes | Now |
|---|---|---|---|
| E00 | PASS 11/11 (Phase 1) | re-run in Phase 6 with an exact-loss row | PASS 11/11 |
| E01 | PASS 12/12 (Phase 1) | none (docs only) | PASS 12/12 |
| E02 | PASS 6/6 (rectification 1) | held-out seeds 1 and 2: PARTIAL 5/6 each (D-605) | PASS 6/6 on the cited seed; status left to the owner |
| E03 | PASS 4/4 (rectification 1) | cluster rule tightened after review R2-01 (D-603) | PARTIAL 3/4 |
| E04 | PASS 9/9 (rectification 1) | none | PASS 9/9 |
| E05 | PASS 2/2 (Phase 4) | stage B a documented variant (D-R02) | PASS 2/2 |
| E06 | PARTIAL 2/4 (Phase 4) | none | PARTIAL 2/4 |
| E07 | PASS 3/3 (Phase 4) | none | PASS 3/3 |
| E08 | PARTIAL 2/4 (Phase 4) | sparse criterion re-derived from p. 37 (D-R01) | PARTIAL 2/4 |
| E09 | PASS 3/3 (Phase 5) | pilot cited (D-R03); criteria tightened (D-604) | PASS 3/3 |
| E10 | PASS 3/3 (Phase 5) | graded on an SAE chosen without ground truth (D-R04) | PARTIAL 1/3 |
| E11 | PASS 8/8 (Phase 5) | pentagon PARTIAL; held-out seed (D-R05) | PARTIAL 7/8 |
| E12 | INFO (Phase 6) | — | INFO (extension) |

[Documented: the acceptance tables of phase-1.md, phase-3.md, phase-4.md, phase-5.md,
rectification.md and phase-6.md; `docs/cited_runs.yaml`.]

## Compute actually used

The 35 run directories under `results/` add up to 18,292 s of wall-clock (about 5.1 hours):
15,479 s on the M2's CPU and 2,813 s on Apple MPS. [Documented: the `wall_clock_s` field of every
`manifest.json`.] This excludes every run that is not on disk: the scratchpad prototypes and
development runs, the dirty-tree runs that session 35cab3f0 moved aside, and the planning
conversation's JAX run. Their compute is unknown.

## What went wrong, and how each problem was handled

The tables make the build look orderly. It was not, and the disorder is the most useful part to
learn from. Each item below says what happened, how it was found and what was done.

### 1. Phase 3 was left unfinished across three sessions

- **What happened.** The Phase 3 session hit a usage limit at 14:08 on 09-29 with `theory.py`
  and the E02–E04 modules written but nothing registered, tested, run or documented.
  [Documented: S 2df87f77 @ 14:08Z; phase-3.md "How this phase ran".] The draft reached git
  through two commits made from outside the session: `4355d5d` at 13:44, while the session was
  still working, and `acec078` at 14:37, after the owner's commit attempt at 14:33 [Documented:
  git; S 2df87f77 @ 14:33Z]. The owner then started Phase 4 with the usual prompt, which asked
  the session to read `docs/progress/phase-3.md`. That file did not exist. [Documented: S
  92884087 @ 16:29Z.]
- **How it was handled at the time.** The Phase 4 session noticed, checked that `make lint test`
  was green (the prompt's only stop condition), and went ahead. It cited SPEC §11's permission
  to build experiments in parallel, left the Phase 3 code untouched and unregistered, and
  recorded this in D-401. Phase 5 did the same. [Documented: D-401; phase-4.md and phase-5.md
  opening notes.]
- **How it was fixed.** On 09-30 the owner ran a rectification session whose only job was to
  close Phase 3. It first checked that nothing else was writing to the repository, showed what
  the external commits had changed, and only then registered, tested, ran and documented E02–E04.
  [Documented: phase-3.md.] The planning conversation had recommended not starting Phase 6
  while Phase 3 was open, and running two rectification sessions first [Planning conversation,
  per owner]; the owner did run both before Phase 6 [Documented: phase-3.md; rectification.md].
- **Lesson** [Recommended]. A stop condition that only checks "tests are green" cannot catch an
  unfinished phase, because unregistered code has no tests to fail. Check the previous phase
  report exists and its exit criteria are ticked before starting the next phase.

### 2. Commits made from outside a session

- **What happened.** VS Code had the repository open with autosave, smart commit and automatic
  sync. Commits under the owner's identity appeared mid-session and swept up half-written work:
  Phase 2 docs (`e3d80a3`), a Phase 4 report draft with unfilled placeholders (`7bde9b1`,
  `7ac6491`), Phase 3 DECISIONS entries and a notebook (`92525fc`), and an intermediate lab page
  (`5e50b46`). [Documented: phase-2.md, phase-4.md and phase-3.md commit notes.] The last two
  sessions' work also landed this way, after they ended (`2564cf1`, `5719ef9`).
  [Reconstructed: from the commits' files and the sessions' edit lists.]
- **Why it matters.** A run records `git.dirty` when it starts. An autosave of a tracked file
  while a queued run starts makes that run uncitable. A pushed draft with placeholders is a
  published half-truth. [Documented: phase-4.md open issue 8; CLAUDE.md.]
- **How it was handled.** History was never rewritten. Each phase report disclosed the foreign
  commits. From rectification 2 on, sessions drafted long docs in the scratchpad, committed
  small pieces promptly, and launched cited runs from a clean detached `git worktree` with its
  own `uv sync`. [Documented: rectification.md "Before changing anything"; D-606.]

### 3. Runs outside the provenance system

- **What happened.** Several decisions rest partly on runs that left no manifest in `results/`.
  The Phase 3 draft explored E02 and E03 with scratchpad scripts on CPU and MPS before any
  criterion was written. Phase 4 prototyped E06 and E08 the same way. Phase 5 ran E09–E11
  development runs into a scratchpad results folder. [Documented: S 2df87f77 @ 13:35–14:02Z;
  S 92884087 @ 16:47–17:04Z; S 57d571bd @ 20:22–21:07Z; D-305, D-306, D-406, D-502, D-505.]
- **How it was handled.** Each such run is disclosed in the DECISIONS entry that relied on it.
  The decisions that mattered were re-based on logged CLI runs: the batch-256 choice on six
  clean CLI runs (D-309), and E09's density pilot as a cited profile (D-R03). Untracked code now
  marks a run dirty (D-606).
- **What is still unknown.** The scratchpad outputs are not part of the repository, so what they
  contained beyond the numbers quoted in DECISIONS is unknown. One E11 development run cited in a
  mechanism claim was no longer on disk when the reviewer looked (REVIEW R4-07).

### 4. Criteria seen, grids refined, and selection by ground truth

- **Criteria written after exploration.** E02's ≥ 80% agreement and E03's thresholds were
  written after the draft had seen exploratory numbers; both entries say so (D-305, D-306).
  E06's choice of settings discloses a prototype (D-406). E10's ground-truth-free selection rule
  was fixed while the previous run's summary was known (D-R04). [Documented.]
- **A grid refined after a failed run.** E11's λ grid was refined around λ ≈ 0.6 after two
  development runs, so that the pentagon's "falls to none" could pass at one λ (D-505). The
  owner later made that criterion PARTIAL and asked for a held-out seed; the held-out seed did
  not confirm the window. [Documented: D-R05; rectification.md; REVIEW R4-06.]
- **Selection by ground truth.** E10 first graded the best SAE chosen by recovery, which a
  practitioner without ground truth could not do (D-504; phase-5.md open issue 2). The owner
  replaced it with a rule that never looks at recovery (D-R04). E10 went from PASS 3/3 to
  PARTIAL 1/3.

### 5. Acceptance criteria that turned out to be wrong

- **E08.** SPEC's "more than m/2 features computed well" came from the spec, not the paper.
  The owner had it re-derived from p. 37 and footnotes 16–17, with the original result kept
  visible as INFO (D-R01). [Documented.]
- **E03.** The cluster criterion counted features by Dᵢ alone, so one band of values counted as
  two "clusters". The review caught it, and E03 became PARTIAL 3/4 (R2-01, D-603). [Documented.]
- **E09.** The motif detector matched a weight pattern, not the mechanism, and counted motifs on
  unexplained neurons. Tightened after review R4-01 (D-604). E09 still passes. [Documented.]
- **A wrong belief, corrected.** D-113 claimed E01 represented fewer features than pp. 11–12
  show. A 300-dpi reading of the figures showed no shortfall, and D-114 corrected it. [Documented:
  phase-1.md open issue 2.]

### 6. Results that depend on the seed

The review found that E02's n = 3, m = 2 agreement (0.819 against a 0.80 bar) rests on one seed.
Seeds 1 and 2 were run under a rule fixed beforehand and gave 0.764 and 0.792 (D-605). The
status was not changed by the session; it is the owner's decision. [Documented: REVIEW R1-01;
phase-6.md open issue 1.]

### 7. Tooling failures

- **CI never ran green until `b50d062`.** Every CI run since the first push failed at job setup,
  because `astral-sh/setup-uv@v10` did not exist as a tag. Found in rectification 1, fixed at the
  owner's request at the end of rectification 2. [Documented: phase-3.md open issue 7;
  rectification.md open issue 7.]
- **Leaked servers.** The app launch test killed the CLI but not its Streamlit child, leaving
  four servers running for 10–19 hours. Fixed with a process group in `8a02181`. [Documented:
  phase-3.md.]
- **Usage limits.** Three sessions were cut by a usage or session limit. Rectification 1 and
  Phase 6 resumed in the same session once the limit reset and the owner asked them to continue.
  The Phase 3 draft was never resumed; a
  new session closed it the next day. [Documented: S 2df87f77 @ 14:08Z; S 5ab2965c @ 09:40Z and
  12:58Z; S 1584c849 @ 2026-09-30T20:27Z, 2026-10-01T06:24Z, 07:16Z and 11:21Z.]

## Gaps in this timeline

- The planning conversation: its length, what was tried and discarded, and the JAX code are
  unknown. Only the owner's summary (COOKBOOK_SPEC §4) is available.
- Scratchpad runs: their configs, seeds and full outputs are unknown; only the numbers quoted in
  DECISIONS are known.
- Human time: the transcripts give session start and end times, but not how long the owner spent
  reading, reviewing or deciding between sessions. This cookbook gives no human-effort estimates.
