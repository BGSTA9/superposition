# Verification of this cookbook

How the cookbook was checked, as COOKBOOK_SPEC §8 requires: its claims against their evidence,
its commands in a fresh clone, the rebuild kit both ways, and the integrity of the repository.
All checks ran on 2026-10-02 on the owner's Apple M2 laptop (macOS, uv 0.12.19), from the
starting commit `2603068`.

## 1. Fidelity of the [Documented] claims (§8.1)

**Who checked.** A verifier agent with fresh context, which had not written any part of the
cookbook. It worked read-only on the repository and on the transcript-extraction output (never
the raw transcripts), and checked "[Planning conversation, per owner]" items only against
COOKBOOK_SPEC §4.

**How claims were sampled.** Every line carrying **[Documented]** was indexed per file, and
claims were taken at fixed positions in each file's list (about 20%, 55% and 85% for chapters
01–10; 35% and 75% for each recipe card and the rebuild kit's README), with a second pass in the
longest chapters. A third pass added every claim that generalizes ("every", "never", "all") and
the README's list of inconsistencies, because those were the likeliest to be wrong.

**Result: 88 claims checked. 83 Confirmed, 3 Wrong, 1 Unsupported, 1 Mislabeled.** At least
three claims came from each of chapters 01–10, at least one (most often two) from each card
E00–E12, and five from the rebuild kit's README. Every claim that failed was fixed or relabeled:

| File | Claim | Verdict | Fix |
|---|---|---|---|
| 01_timeline.md (Phase 3 row and §1); also E02.md, E03.md | The owner committed the Phase 3 draft (`4355d5d`, `acec078`) after the 14:08Z usage limit | Wrong | `4355d5d` is dated 13:44Z, while the session was still working; only `acec078` (14:37Z) came after the limit and the owner's 14:33Z commit attempt. All three files now give the order. |
| 02_the_research_method.md, step 7 | Every phase started with the same prompt, including a `make lint test` baseline | Wrong | Rewritten: Phases 0–1 shared one prompt, Phase 2's added re-reading the state files, and only Phases 3–6 required a green baseline. Each prompt is cited. |
| 09_skills_and_drills.md, skill 9 | `select_best` never picks a NaN | Wrong (minor) | Now: never picks a NaN loss unless a whole group is NaN, as its docstring says. |
| 08_project_management.md §6 | `4355d5d` and `acec078` were committed by the owner | Unsupported | Now: both came from outside the session (no Co-Authored-By trailer; the session ran no `git commit`); whether the owner or VS Code's automatic sync made `4355d5d` is unknown. |
| README.md, inconsistency 1 | D-105's text was appended while the first E01 run was training | Mislabeled | Relabeled **[Reconstructed]**, with its basis (timestamps only). |

Notes the verifier attached to Confirmed claims were also acted on: E12.md now gives the exact
commit times of the retried commits (`ad8ad78` at 06:39:21Z, the other three at 06:39:48–49Z),
and 02_the_research_method.md now cites the phase reports for the worktree launches of Phase 6.

**Other flags.** No raw transcript text, no email address, and no invented number was found.
The derived totals (18,292 s, 5,210 s, 118 of 131 commits) match the manifests and git. The
only reconstruction presented as documented was the one relabeled above. The appendix lists all
88 claims.

**Corrections made while integrating, before the verifier ran.** Reading the chapters against
the evidence also caught these, now fixed:
- 04_scaffolding.md and 08_project_management.md said or implied that CI first passed at the
  Phase 6 head. The first run on `b50d062` already passed both jobs [S 522eb237 @ 19:54Z].
- 04_scaffolding.md said Part B had been checked with "uv 0.9.15"; the only uv on the machine is
  0.12.19, and Part B was re-run with it (§2 below).
- 02_the_research_method.md said every cited run came from a worktree "from Phase 4 on"; the
  cited runs of E06 and E07 ran from the main tree while it was clean (their manifests).
- 02_the_research_method.md attributed the slow-tier split to the rectification-2 decisions;
  it was in the owner's rectification-1 brief [S 5ab2965c @ 2026-09-30T07:33Z].
- 01_timeline.md now notes that the fast tier has 261 tests at `2603068` (260 in phase-6.md).

## 2. Commands (§8.2)

**The fresh clone.** `git clone --branch cookbook` of the cookbook's working branch into an
empty temporary folder: no `results/`, no paper PDF, no `.venv` until `uv sync --locked` made
one. `SUPERPOSITION_RESULTS` and `VIRTUAL_ENV` were unset for every command. Commands ran one
after another from the clone's root, each with its exit code, wall-clock and output logged.

**What ran, and what came out.**

| Set | Commands | Outcome |
|---|---|---|
| 04_scaffolding.md Part A, then every other unmarked command, in file order, deduplicated | 105 | every exit code as the text expects |
| 05_engine.md, the runnable inline commands, and the PDF commands | 38 | every exit code as the text expects |
| 04_scaffolding.md Part B, in a new empty folder outside any repository | 28 | B1–B10 work as written, from `uv init` to the commit; only `git push` and `gh run list` were not run (marked) |
| The worktree flows of 08_project_management.md §7 and 09_skills_and_drills.md drill 15.3, with the placeholder paths and run ID filled in | 2 flows | both made a clean run (`dirty: false`, PASS 11/11) and removed the worktree |

Exit codes that are not 0 are the expected ones, and the text says so each time:
- every rebuild-kit test command and `make rebuild-test`, as shipped (the skeleton raises
  `NotImplementedError`);
- `git diff --no-index` comparing the kit with the answer key (exit 1 means the files differ);
- `pgrep -fl "streamlit run"` (exit 1 means no stray server).

**E00 and E01, quick profiles.** Both reproduced their cited runs exactly:

| Exp. | Fresh-clone run | Status | Wall-clock | Criterion values identical to the cited run |
|---|---|---|---|---|
| E00 | `20261002T124559Z-5eeb804c` | PASS 11/11 | 69 s | 18 of 18 (cited `20261001T071529Z-5eeb804c`) |
| E01 | `20261002T125625Z-2cb0bbc4` | PASS 12/12 | 521 s | 14 of 14 (cited `20260929T093755Z-2cb0bbc4`) |

Both manifests record `dirty: false`. The two worktree-flow runs of E00 gave the same values.
Other measured results: `make test` 261 passed in 55 s; `make test-slow` 12 passed in 422 s;
`make lint` clean; the nine pre-commit hooks passed; the slow replication tests of E00, E02,
E03, E04 and E12 passed.

**The paper PDF.** The commands that read the git-ignored PDF (chapter 03 §1–2, drill 16.1)
ran with the owner's copy placed in the clone's `references/`, as a reader's own setup would
have it. They matched the text: 62 pages, a 576 × 792 pt page, the heading search hitting
pp. 9, 10, 35, 53 and 54, and a 300-dpi render of p. 17.

**The other experiments' commands, checked three ways.** All 72 `superposition run`,
`figures`, `report` and `app` commands in the cookbook were checked against
`superposition list` (the experiment exists), against the subcommand's `--help` (every flag
exists, and the profile is defined in that experiment's config), and against the run manifests
under `results/`. Every one passed the first two checks. Each of the 21 run commands marked
**[Verified against manifest]** has a manifest with the same experiment, profile and seed.

**Commands written inline in prose** were collected too. The runnable ones (`git log --
SPEC.md`, `git show --stat 2564cf1`, `git diff 1fb44c4 HEAD -- configs/uniform_geometry.yaml`,
the trailer count, `make format`, `uv python install 3.12`, three pytest invocations and a
torch import) ran in the clone. The rest name a tool (`pdftotext`) or describe what a session
once did (`gh run view --log-failed`); they are not instructions.

**Not run (59 commands, each marked [Not verified] with its reason in the text).**
- They modify tracked docs: `figures --publish` and `report --pin`.
- They were never run in this repository's history, and their runtimes are only estimates:
  the full profiles of E02–E07 and E09–E12, `make test-paper`, and the paper-tier tests.
- They need the git-ignored `results/` folder of the owner's machine: `figures --run <cited
  ID>`.
- They start a long-running server: `superposition app`. The slow tier's launch test covers
  it, and it passed.
- They need credentials: `git push`, `gh run list`.
- `brew install poppler` installs system software.
- `git checkout` of a kit file discards your work, and the M5 script is yours to write.

**One side effect.** Chapter 04's `uv python install 3.12` downloaded a uv-managed Python
3.12.14 into uv's own store on the machine. It changes no project file.

## 3. The rebuild kit (§8.3)

Run in the fresh clone, on the committed kit:

| Step | Command | Result |
|---|---|---|
| As shipped | `make rebuild-test` | 64 failed (data 14, models 12, train 13, metrics 25), every failure `NotImplementedError` |
| Real implementations substituted in place (`src/superposition/{data,models,train,metrics}.py`, imports pointed at the kit) | `make rebuild-test` | 64 passed |
| Reverted with `git checkout -- rebuild/` | `git status --porcelain` | empty |
| After the revert | `make rebuild-test` | 64 failed, all `NotImplementedError` |

The kit is out of the default checks. `make lint` passes with it present: ruff skips `rebuild/`,
even for files named explicitly as pre-commit passes them. Pyright's include list does not
contain it. The default `pytest` (testpaths `tests/`) and CI do not collect it.

## 4. Integrity (§8.4)

- **Changes outside the new files.** `git diff --stat 2603068`, excluding `docs/cookbook/` and
  `rebuild/`, shows only the edits §2 allows:

  | File | Change |
  |---|---|
  | `README.md` | one sentence linking the cookbook (+3 lines) |
  | `docs/STUDY_GUIDE.md` | one sentence linking the cookbook |
  | `Makefile` | the `rebuild-test` target and its `.PHONY` entry |
  | `pyproject.toml` | ruff `extend-exclude = ["rebuild"]` and `force-exclude = true`, with a comment |

  Nothing in `src/`, `configs/`, `tests/`, `results/`, the notebooks, the app or the existing
  docs changed.
- **`make lint test`** passes on the final tree (`make test`: 261 passed).
- **`.cookbook-evidence/`** is git-ignored (`.gitignore` line 32), and `git ls-files
  .cookbook-evidence` lists nothing.

## Appendix: the 88 sampled claims

Claims are paraphrased, and line numbers refer to the files as they were when checked. "S"
cites the transcript-extraction output.

| # | file:line | claim (short paraphrase) | evidence checked | verdict | correction or note |
|---|---|---|---|---|---|
| 1 | 01_timeline.md:53 | Phase 3 draft: theory.py, E02–E04 modules, two plotting modules written but not registered, configured, tested, run or documented | phase-3.md l.7–9 | Confirmed | |
| 2 | 01_timeline.md:53 (Issues cell) | Usage limit at 14:08; "the owner then committed the draft" (4355d5d, acec078) | `git show 4355d5d acec078`; S 2df87f77 @ 13:44, 14:08, 14:33–14:35Z | Wrong | Order is wrong for 4355d5d. It is dated 16:44:24 +0300 (13:44:24Z), mid-session and 24 min **before** the 14:08Z limit, while the session's 14:08:59Z summary still reported nothing committed. Only acec078 (14:37:18Z) followed the limit and the owner's 14:33Z commit attempt. Same implication at 01:116–117 and E02.md:116–117. |
| 3 | 01_timeline.md:24–38 | Eight sessions; prefixes, titles, start/end times | extracted/INDEX.md | Confirmed | |
| 4 | 01_timeline.md:63–70 | 131 commits; 90e2033 under a GitHub no-reply address, 130 under the local identity; 118 with Co-Authored-By; the 13 others listed | `git log --format='%an|%ae'`; trailer count; list of non-trailer commits | Confirmed | The 13-commit list matches exactly. |
| 5 | 01_timeline.md:100–102 | 35 run dirs, 18,292 s; 15,479 s CPU, 2,813 s MPS | all 35 `manifest.json` | Confirmed | 18,292.2 s; cpu 15,479.2; mps 2,813.0 |
| 6 | 01_timeline.md:116–118 | Phase 4 prompt asked to read phase-3.md, which did not exist | S 92884087 @ 16:29:08Z; 18:40Z report | Confirmed | |
| 7 | 01_timeline.md:137–142 | External commits e3d80a3, 7bde9b1, 7ac6491, 92525fc (Phase 3 DECISIONS + notebook), 5e50b46 (intermediate lab page) | phase-2.md commit note; phase-3.md l.40–52; `git show --stat` | Confirmed | |
| 8 | 01_timeline.md:189–190 | E09 motif detector tightened after R4-01 (D-604); E09 still passes | D-604; REVIEW R4-01; REPRODUCTION status table | Confirmed | |
| 9 | 02_the_research_method.md:136–137 | metrics.py has 17 "Paper:" references | `grep -c "Paper:" src/superposition/metrics.py` → 17 | Confirmed | |
| 10 | 02_the_research_method.md:295–296 | SPEC §11 defines Phases 0–6, a closing ritual and rules for parallel work | SPEC.md §11 | Confirmed | |
| 11 | 02_the_research_method.md:298–300 | Every phase started with the same prompt: read CLAUDE.md, SPEC, PLAN, last report; check `make lint test`; do phase; verify; commit; report; stop | S 2df87f77 @ 13:23Z, S 92884087 @ 16:29Z (cited); also S 35cab3f0 @ 08:06Z and 12:32Z, S 57d571bd @ 19:58Z, S 1584c849 @ 20:13Z | Wrong | Overgeneralized. Phases 3–6 used that template. Phases 0 and 1 shared one prompt ("Read SPEC.md in full, then the PDF… Execute Phase 0 and Phase 1…"): no CLAUDE.md/PLAN/report to read and no baseline check. Phase 2's prompt had no `make lint test` baseline step. The cited evidence covers only Phases 3 and 4. |
| 12 | 02_the_research_method.md:358 | 35 run directories; every manifest records `dirty: false` | all manifests | Confirmed | |
| 13 | 02_the_research_method.md:360–364 | First worktree run was E08 full (Phase 4); from rectification 1 every cited run used a worktree; E01, E06, E07 cited runs ran from the clean main tree | manifests `command`/`git`; S 92884087 @ 17:58Z; phase-6.md l.16 | Confirmed (note) | The three main-tree runs are correct. For Phase 6 runs the manifest `command` is a relative `.venv/bin/superposition`, which does not by itself show a worktree; phase-6.md "Before changing anything" does. |
| 14 | 02_the_research_method.md:414–418 | Overclaims that survived to review: R2-02, R3-01, R3-03 | REVIEW.md rows | Confirmed | |
| 15 | 02_the_research_method.md:456–459 | Closing Phase 3 and the slow-tier split went into rectification 1's brief; the other five became D-R01–D-R05, recorded before their runs | S 5ab2965c @ 07:33Z (brief items incl. "split the test tiers"); DECISIONS "Rectification session 2" preamble | Confirmed | |
| 16 | 03_from_paper_to_spec.md:97–98 | Build session ran a heading-to-page loop over pp. 38–62 | S 35cab3f0 @ 08:19:41Z | Confirmed | |
| 17 | 03_from_paper_to_spec.md:215–222 | SPEC reference table (0.182/0.182/0.183; 0.059/0.122/0.124; 0.0196/0.067/0.068) | SPEC.md §7 E00 "Reference values" | Confirmed | |
| 18 | 03_from_paper_to_spec.md:319–324 | Where under-specified items were resolved: D-101, SPEC §3, 0.9ⁱ, 8 restarts, 0.5 threshold, "orthogonal" (D-105, D-106), 100,000 samples (D-106), D-109, D-113 | D-101, D-105, D-106, D-109, D-113; SPEC §3, §7 | Confirmed (note) | D-105 has no explicit "orthogonal" threshold; it fixes the 0.05 superposition-score bar (which D-505 calls "E01's orthogonality bar"). |
| 19 | 04_scaffolding.md:31–34 | Phase 0 files committed within three seconds, 11:22:25–11:22:27 +03:00 | `git log 7673c23^..f31595b` | Confirmed | |
| 20 | 04_scaffolding.md:34–37 | f31595b added PLAN, CLAUDE.md, references/README, CITATION, CHANGELOG, DECISIONS, skeletons; author placeholder (D-006) still open | `git show --stat f31595b`; D-006; phase-0.md; phase-6.md open issue 8 | Confirmed | |
| 21 | 04_scaffolding.md:128 | Phase 0 had one `slow` marker and two placeholder tests | phase-0.md l.17, 38, 46 | Confirmed | |
| 22 | 04_scaffolding.md:211–218 | `setup-uv@v10` (d6bd81f) broke CI; 14 failed runs; 522eb237 pinned v10.2.0 in b50d062 at owner's request and saw both jobs pass; run 36859923829 green at 41aad4b | `git show d6bd81f b50d062`; phase-3.md open issue 7; S 522eb237 @ 19:42–19:54Z; phase-6.md l.225 | Confirmed | |
| 23 | 05_engine.md:44–66 | Phase 1 code committed 09:01–09:05Z as 14 logical commits (table of hashes and times) | `git log 1815b59..340f825` | Confirmed | Every hash and time matches. |
| 24 | 05_engine.md:150–154 | Optimizer (Adam, 1e-3, 1024, 10,000) fixed in SPEC §6 after the verification run; D-101 | D-101; COOKBOOK_SPEC §4 | Confirmed | |
| 25 | 05_engine.md:316–322 | L* gives 0.183, 0.124, 0.068; derived in planning; session re-checked at 08:29Z | SPEC §7 E00; §4; S 35cab3f0 @ 08:29:11Z | Confirmed | |
| 26 | 05_engine.md:353–354 | Paired seeds: each ReLU instance starts from its linear twin's W, same data | D-103 | Confirmed | |
| 27 | 05_engine.md:498–502 | Only the linear optimum in Phase 1 (4558329); D-301/D-302 written up on 09-30 | `git show --stat 4558329`; DECISIONS "Phase 3" preamble | Confirmed | |
| 28 | 05_engine.md:548–551 | Graded PARTIAL added in 81ad83c for E11's pentagon (D-R05) | `git show --stat 81ad83c`; D-R05; rectification.md item 5; `test_graded_partial_criteria` exists | Confirmed | |
| 29 | 06_debugging_playbook.md:74–79 | E08 prototypes (2/4 restarts, 30,000 steps, lr 3e-3, batch 4,096) gave 8–16; scratchpad script at 17:04Z; D-409 "not cited" | D-409; phase-4.md open issue 1; S 92884087 @ 17:04Z | Confirmed | |
| 30 | 06_debugging_playbook.md:134–138 | SPEC §3: 2D hard to optimize, lowest loss kept (p. 23); 3 of 8 ReLU restarts at S = 0.9 settled for four features | SPEC l.133–135, 426 | Confirmed | |
| 31 | 06_debugging_playbook.md:300–304 | E09 densities from a pilot on the graded quantity; 0.1 would give 0.86 | D-502; REVIEW R4-10; REPRODUCTION l.1177 | Confirmed | |
| 32 | 06_debugging_playbook.md:476–482 | D-603: overlapping windows; expected PARTIAL known; rule committed before grading run; PASS 4/4 → PARTIAL 3/4 | D-603; REVIEW R2-01; phase-3.md l.164; phase-6.md l.50 | Confirmed | |
| 33 | 07_reporting_and_review.md:110–111 | Append-only; D-114 withdrew D-113's premise | DECISIONS preamble; D-114 | Confirmed | |
| 34 | 07_reporting_and_review.md:195–201 | Four reviewers + a fifth (E12); spawned 20:17–20:18Z and 06:50Z; two interrupted by a session limit and resumed | D-607; extraction "Agents spawned"; S 1584c849 @ 20:28Z; phase-6.md l.17 | Confirmed | |
| 35 | 07_reporting_and_review.md:202–203 | 59 findings: 0 critical, 12 major, 29 minor, 18 notes | REVIEW.md reviewer table (sums) | Confirmed | |
| 36 | 07_reporting_and_review.md:240–258 | Failure-mode table: R1-02, R2-03, R5-01, R4-10, R1-06, R2-10, R3-06, R3-07, R4-06, R3-12, R3-14, R3-04, R1-07, R3-15, R4-07, R1-10, R1-09, R5-04; stale "E12 arrives in Phase 6" | REVIEW.md rows; phase-6.md l.113 | Confirmed | |
| 37 | 07_reporting_and_review.md:281 | Co-Authored-By trailers on 118 of 131 commits | `git log` trailer count at 2603068 | Confirmed | Evidence column cites another cookbook chapter (01_timeline); git confirms it. |
| 38 | 07_reporting_and_review.md:287 | Push and CI fixes made at the owner's request | S 522eb237 @ 19:42:36Z; S 1584c849 @ 12:09:20Z | Confirmed | |
| 39 | 08_project_management.md:30–34 | Phase 4 relied on SPEC §11's parallelism clause (D-401) | SPEC §11 "Parallelism"; D-401 rationale | Confirmed | |
| 40 | 08_project_management.md:68–79 | State-tracking table: PLAN legend [x]/[ ]/[~]; CLAUDE.md ≤ ~150 lines; cited_runs.yaml written only by `report --pin` | PLAN.md l.8; SPEC §10 l.649; D-202; cited_runs.yaml header | Confirmed (note) | The "when updated" cadence for CLAUDE.md ("when a new gotcha is learned") is not stated in the cited evidence. |
| 41 | 08_project_management.md:130–140 | Test-status table per phase (e.g. 87 in ≈6 s; 5 in 12.5 min; 226 in 41 s; 233 in 33 s; 260 in 49 s; 12 in 429 s) | phase-1/2/3/5/6.md and rectification.md test tables | Confirmed | |
| 42 | 08_project_management.md:170–196 | Per-experiment runs and seconds; cited quick runs ≈5,210 s; four supplementary 1,742 s | manifests; cited_runs.yaml | Confirmed | 5,211.1 s and 1,742.0 s; every per-experiment total matches. |
| 43 | 08_project_management.md:244–246 | Phase 3 draft reached git "the same way", in 4355d5d and acec078, "committed by the owner from outside the session" | cited S 2df87f77 @ 14:33–14:35Z; `git show 4355d5d` | Unsupported | The cited evidence (owner's Commit & Sync attempt at 14:33Z, pre-commit failure) covers acec078 (14:37Z) only. 4355d5d was committed at 13:44Z during the session; nothing shows who or what made it (owner or VS Code smart commit). |
| 44 | 08_project_management.md:266–269 | Prototypes/dev runs ran from scratch scripts or into a scratch results folder; untracked module wouldn't mark a run dirty | D-305, D-306, D-406, D-502, D-505; REVIEW R1-10 | Confirmed (note) | "Scratch results folder" is shown by transcripts (e.g. S 57d571bd @ 20:22Z, 21:01Z), not by the cited decisions. |
| 45 | 09_skills_and_drills.md:50–57 | Loss order fixed by SPEC §3; tests check it against a loop and that instance 0's gradient is bit-identical | SPEC §3; tests/unit/test_train.py (`torch.equal`) | Confirmed | |
| 46 | 09_skills_and_drills.md:129–135 | test_metrics implements SPEC §6's table (2/5, D* 0.4, 1/2, 2/3, 3/4, 1, 0, (n−1)/n) plus two hypothesis tests | tests/unit/test_metrics.py | Confirmed | |
| 47 | 09_skills_and_drills.md:320 | `train.select_best` picks the lowest-loss instance per group and never a NaN | train.py `select_best` docstring and code | Wrong | Overstated: NaN is never selected **unless a whole group is NaN** (05_engine.md:303 and 06:139–140 state this correctly). |
| 48 | 09_skills_and_drills.md:493–495 | runs.py writes commit, dirty, command, times, wall-clock; D-202 pin refuses dirty runs; D-R07 adds labeled supplementary citations | runs.py; report.py; D-202; D-R07 | Confirmed | |
| 49 | 10_next_project.md:38–39 | SPEC §1 mission: replicate, lab, interpretability tooling, learning instrument | SPEC §1 | Confirmed | |
| 50 | 10_next_project.md:127–131 | SPEC's references are "not acceptance criteria"; 0.0196 below converged 0.0214; selection bias leading hypothesis | SPEC l.415; REPRODUCTION E00 "Hypothesis" | Confirmed | |
| 51 | 10_next_project.md:198–202 | REPRODUCTION has claim/figure/status/evidence/discrepancies/run IDs; tables regenerated, narrative by hand; E10, E11 PARTIAL via owner decisions, E03 via R2-01 | SPEC §10 l.642; D-R04; D-R05; D-603 | Confirmed | |
| 52 | experiments/E00.md:115–120 | Diagnostics became INFO rows (D-107, "instead of an ad hoc script"); E00 re-run after `rm -rf results` at 08:41–08:42Z | D-107; S 35cab3f0 @ 08:42:53Z | Confirmed | |
| 53 | experiments/E00.md:212–221 | 0.0214 vs 0.0196; geometry same; 12 arrangements 0.02142–0.02165; 3.7% noise; 1 of 200 at/below; reference linear losses below optimum | REPRODUCTION E00; cited acceptance.json | Confirmed | |
| 54 | experiments/E01.md:105–109 | Full profile defined at 08:54; D-113 written at 09:02 on the 14/20, 67/80 premise | S 35cab3f0 @ 08:54:45Z, 09:02:25Z; D-113 | Confirmed | |
| 55 | experiments/E01.md:177–178 | Full run `20260929T090856Z-6d287488` (MPS) PASS 12/12 with 12/20 and 68/80 | its manifest and acceptance.json | Confirmed | |
| 56 | experiments/E01.md:181–186 | D-113's premise came from low-res renders; D-114 corrected it in a new entry | D-114; phase-1.md open issue 2 | Confirmed | |
| 57 | experiments/E02.md:140–143 | Quick 487 s vs batch1024 1,597 s, both PASS 6/6; labels identical 144/144 (n = 2), 140/144 (n = 3), 5–4 votes | manifests of 090540Z/075829Z; D-309; phase-3.md l.167–177 | Confirmed | |
| 58 | experiments/E02.md:234–239 | Starting sign: 535/545 vs 266/515 (seed 0); 40 stuck models re-optimized, none changed | REVIEW R1-02; REPRODUCTION l.288–292 | Confirmed | |
| 59 | experiments/E03.md:200–204 | D-603 discloses expected PARTIAL; thresholds identical since 1fb44c4, only comments changed | D-603; `git diff 1fb44c4 HEAD -- configs/uniform_geometry.yaml` | Confirmed | |
| 60 | experiments/E03.md:205–208 | Draft's 50,000-step MPS E03 run passed the four old criteria but had no triangle cluster | D-306; D-309 | Confirmed | |
| 61 | experiments/E04.md:196–203 | 200,000-sample selection committed in 230a51d minutes after a 07:56Z evaluation; config diff adds only `eval_samples` and comments | S 5ab2965c @ 07:56:41Z; 230a51d at 07:57:05Z; `git diff 1fb44c4 HEAD -- configs/nonuniform_geometry.yaml` | Confirmed | |
| 62 | experiments/E05.md:125–128 | Quick run from main tree at 5a825d0: `20260929T172920Z-2fae0fc5`, 516 s, PASS 2/2 | its manifest/acceptance; S 92884087 @ 17:29Z | Confirmed | |
| 63 | experiments/E05.md:236–237 | Owner decided stage B is a documented variant (D-R02); §4 item 3 recommended it | D-R02; COOKBOOK_SPEC §4 item 3 | Confirmed | |
| 64 | experiments/E06.md:204–209 | R3-03: V falls 1.25 → 1.10 while 1/D* rises 1.39 → 1.61; 2.29 vs 1.51 at 1/D* ≈ 1.82 | REVIEW R3-03; REPRODUCTION l.771–776 | Confirmed | |
| 65 | experiments/E07.md:113–115 | Quick run at 5a825d0: `20260929T174443Z-d54df07e`, 419 s, PASS 3/3 | manifest; cited_runs.yaml | Confirmed | |
| 66 | experiments/E07.md:186–190 | At 0.12: two monosemantic neurons, 7 features vs p. 33's one and six-in-four; 5–8% of restarts; full profile never run | REPRODUCTION E07 notes; phase-4.md open issue 4 | Confirmed | |
| 67 | experiments/E08.md:143–145 | Session did not consult the references "so that the choice stays yours" | phase-4.md open issue 1; D-409 "Not done" | Confirmed | |
| 68 | experiments/E08.md:276–281 | R3-01 trend reproduces, two details don't; dense computes only 13; R3-11 | REVIEW R3-01, R3-11; phase-4.md | Confirmed | |
| 69 | experiments/E09.md:125–126 | Role rules on path gains (D-501), sanity-checked on a hand-built motif at 20:12Z | D-501; S 57d571bd @ 20:11:55–20:12:23Z | Confirmed | |
| 70 | experiments/E09.md:246–250 | Motif detector too loose (R4-01); D-604: 134 → 68 models; status stayed PASS | REVIEW R4-01; D-604 | Confirmed | |
| 71 | experiments/E10.md:108–112 | Dev run into scratch results; strict dead rule resampled nothing though 13–28% never fired; changed to 1e-5 | S 57d571bd @ 20:50Z, 20:58Z; D-503 rationale | Confirmed | |
| 72 | experiments/E10.md:212–216 | At 0.001 the λ = 0.3 SAEs have FVU 0.117–0.121; rule picks λ = 0.1; one PASS has FVU 0.0996 | rectification.md open issue 2; cited acceptance.json | Confirmed | |
| 73 | experiments/E11.md:108–112 | First dev run via CLI into scratch results; grid listed; PARTIAL 5/8; stuck restart at 0.4; pair reappearing at 0.2 | S 57d571bd @ 21:01Z; D-505 "History" | Confirmed | |
| 74 | experiments/E11.md:207–209 | Phase 5 run `20260929T212948Z-027b0327` PASS 8/8 | its acceptance.json | Confirmed | |
| 75 | experiments/E12.md:101–102 | Session limit at 20:27Z; resumed 06:24Z next day | S 1584c849 @ 20:27:03Z, 06:24:06Z | Confirmed | |
| 76 | experiments/E12.md:103–108 | Commits ad8ad78, 79ed991, aa8e6b0, 79f1937; first commit attempt failed, retried; two index.lock collisions | `git log`; S 1584c849 @ 06:39Z; phase-6.md l.13–14 | Confirmed (note) | ad8ad78 itself succeeded at 06:39:21Z; the other three were retried at 06:39:47Z. |
| 77 | experiments/E12.md:217–221 | R5-03: restricted families; unrestricted optimum barely changes labels; reviewer's bisection not in a logged run | REVIEW R5-03; REPRODUCTION E12 l.54–62; D-607 caveat | Confirmed | |
| 78 | rebuild/README.md:230 | E00 cited run took 68 s on an Apple M2 CPU | manifest (68.432 s, cpu, arm64 macOS) | Confirmed | "M2" is from the phase reports/configs, not the manifest. |
| 79 | rebuild/README.md:240–249 | E00 table (0.18269 … 0.02117; closed form; reference) and pentagon gaps 69.8–74.0°, norms 1.076–1.111, biases −0.198 to −0.211 | cited acceptance.json | Confirmed | |
| 80 | rebuild/README.md:251–255 | 3.7% relative SD for 20,000 samples; 2 of 8 restarts at S = 0.9 kept four features | acceptance.json (`rel_std_pct` 3.68; `restart_outcomes`) | Confirmed | |
| 81 | rebuild/README.md:256–257 | Pentagon loss 0.0214 on 1,000,000 samples above SPEC's 0.0196 | acceptance.json (0.02139) | Confirmed | |
| 82 | rebuild/README.md:272–276 | E01 run 472 s; counts 5,7,9,11,12,12,14 and 20,27,33,41,52,61,67; ±2 from R1-06 | acceptance.json; manifest; REVIEW R1-06 | Confirmed | |
| 83 | README.md:171–173 (errata 1, labeled [Documented]) | D-105's text was appended at 08:49Z while the 08:46Z E01 run trained | S 35cab3f0 @ 08:46:24Z, 08:49:30Z | Mislabeled | That D-105 was in the 08:49 block is an inference from timestamps. E01.md:101–104 itself labels it [Reconstructed]. Strong inference (it is the only DECISIONS write between 08:20Z and 09:02Z), but it should be [Reconstructed] here too. |
| 84 | README.md:178–179 (errata 4) | Cited E07 acceptance.json still keys the sidestepping row `1-S=0.1` | acceptance.json keys; REVIEW R3-04 | Confirmed | |
| 85 | README.md:180–181 (errata 5) | [0.355, 0.42] is the union; the overlap is [0.38, 0.395] | D-603; arithmetic | Confirmed | |
| 86 | README.md:182–186 (errata 6, 7) | 0.001 is also gone from the E11 grid; E06/E07 cited runs ran from the main tree, against D-606's wording | configs/l1_removes_superposition.yaml; D-505; D-606; manifests | Confirmed | |
| 87 | README.md:187–190 (errata 8, 9) | Bias-init reduction is 28.3% (0.000675 → 0.000484), not 29%; 2564cf1 added one unit test | `20260929T180008Z-db58449f` acceptance.json; `git show 2564cf1` | Confirmed | 261 itself not re-counted (pytest not run); one added test confirmed. |
| 88 | README.md:191–201 (errata 10–14) | Makefile "latest runs"; PLAN quotes superseded E03 run; config runtime comments quote first runs (E00 62 s); L1 margin 0.5 → 0.8 not in phase-1.md; PDF is 576 × 792 pt | Makefile l.46; PLAN l.132–134; configs/intro_pentagon.yaml l.21; S 35cab3f0 @ 08:33:06Z; phase-1.md; `pdfinfo` | Confirmed | |
