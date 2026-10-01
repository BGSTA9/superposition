# Phase 6 report: extensions, review, polish

Date: 2026-10-01 · Commits `ad8ad78`..(this report) · SPEC §11 Phase 6, §7 E12, §10, §12.
Decisions D-601–D-608.

> **Phase 6 is the last phase.** This report also checks SPEC §12's definition of done (at the
> end).

## Before changing anything

- The tree was clean at `b50d062` (origin/main), and `make lint test` was green: lint clean,
  233 tests passed.
- VS Code had the repository open, as in earlier sessions. Two commits collided with its git
  process (`index.lock`); both were retried after checking that the working tree was intact.
- Every one of the 22 commits since `b50d062` is this session's; none came from VS Code.
- Every cited run was launched from a clean detached worktree (D-606).
- The session hit a usage limit twice. The reviewers that were interrupted were resumed with
  their context intact.

## What was built

**E12 activation_functions** (an extension, p. 51; D-601, D-602, D-608).
- **What it runs.** E02's phase diagrams with the output ReLU replaced by gelu, silu,
  softplus, leaky_relu and sigmoid. Everything else is E02's, including the starting weights.
- **Three diagrams per activation:**
  - the vote of the trained models;
  - the exact loss of E02's weight families under that activation (*family theory*);
  - the *unrestricted* optimum over all four parameters (n = 2, m = 1).
- **New theory code.** `theory.activation_square_integral`, `activation_error_1d` and
  `tied_output_expected_losses` compute E02's exact losses for any registered activation. They
  use Gauss–Legendre quadrature split where the activation's argument changes sign: exact for
  the ReLU and leaky ReLU, and accurate to 1e-9 against adaptive quadrature for the smooth ones.
  The family optimizer takes an activation.
- **Profiles.** Quick: n = 2, m = 1 on an 8 × 8 grid, 507 s. Full: both problems on E02's quick
  grid; not run.
- **Lab, notebook, guide.** An E12 tab on the Phase diagrams page, notebook
  `11_extensions.ipynb`, and study-guide stop 11.

**The independent review** (D-607; docs/REVIEW.md).
- **Who.** Five reviewers with fresh context, each a separate agent: four for E00–E11 and one
  for E12 after it was written.
- **How.** Each re-derived its experiments from the cited pages, then audited the code and the
  evidence. They worked read-only, without the reference implementations.
- **Output.** 59 findings: 0 critical, 12 major, 29 minor, 18 notes.
- **What was done.** Every finding is resolved or listed as open in REVIEW.md. The ones that
  changed a number or a status were fixed in code and re-run from a clean commit:

| Finding | Change | Effect |
|---|---|---|
| R2-01 | E03 counts a Dᵢ cluster only when its features sit in that fraction's polytope (D-603) | **E03: PASS 4/4 → PARTIAL 3/4** |
| R4-01, R4-02 | E09 counts only role-consistent motifs; "named" means nothing unexplained; a baseline on untrained weights (D-604) | E09 still PASS 3/3; motifs in 68 models, not 134 |
| R1-01 | E02 held-out seeds 1 and 2, with the reading rule fixed beforehand (D-605) | n = 3 agreement 0.819 / 0.764 / 0.792: fails across seeds; **left to the owner** |
| R1-02, R1-03 | E02 INFO rows `initial_sign` and `exact_loss_vs_theory` (D-605) | the starting sign explains the stuck models; restricted families beaten in 59 of 144 cells |
| R1-11 | E00 reports the graded linear model's exact loss | within 0.01% of the optimum |
| R1-10 | untracked files under `src/` and `configs/` make a run dirty (D-606) | provenance |
| R5-01–R5-06, R5-10 | E12's unrestricted optimum, best models by exact loss, tied rankings, an activation-aware lab (D-608) | the E12 write-up separates landscape from optimization |

The other findings were overclaims or inaccuracies in the docs. They were corrected in
REPRODUCTION.md (E00–E02, E04–E11), the README, notebook 08, and DECISIONS D-404 and D-408.

**Docs.**
- **REPRODUCTION.md:** the E12 section, and sections rewritten for every finding. A hand-written
  note under the generated status table carries E02's seed result.
- **README:** a reproduction status table, a gallery that now includes E08 and E12, and
  corrected captions.
- **Study guide:** stop 11, and a menu of 14 extension projects drawn from the open questions
  (pp. 45–46).
- **Also:** GLOSSARY ("output activation"), PAPER_MAP, DESIGN (held-out seeds, worktree runs,
  provenance), CLAUDE.md, CHANGELOG, PLAN.
- **Status table:** it now shows "report only" for an experiment without graded criteria.

## Runs

All runs used the quick profile, on CPU (Apple M2), from clean detached worktrees.

| Exp. | Run ID | Seed | Commit | Wall-clock | Status | Cited as |
|---|---|---|---|---|---|---|
| E02 | `20261001T065043Z-133b16d8` | 1 | `450938e` | 498 s | PARTIAL 5/6 | `phase_change:seed1` |
| E02 | `20261001T065904Z-afab247b` | 2 | `450938e` | 498 s | PARTIAL 5/6 | `phase_change:seed2` |
| E02 | `20261001T070725Z-4e3325bf` | 0 | `450938e` | 481 s | PASS 6/6 | `phase_change` |
| E00 | `20261001T071529Z-5eeb804c` | 0 | `450938e` | 68 s | PASS 11/11 | `intro_pentagon` |
| E03 | `20261001T071640Z-2cb2dc61` | 0 | `450938e` | 346 s | PARTIAL 3/4 | `uniform_geometry` |
| E09 (pilot) | `20261001T072229Z-1e8bafbd` | 0 | `450938e` | 200 s | PASS 3/3 | `mech_interp_abs:pilot` |
| E09 | `20261001T072551Z-3fdd1d10` | 0 | `450938e` | 255 s | PASS 3/3 | `mech_interp_abs` |
| E12 | `20261001T112822Z-6151c7ff` | 0 | `0ad3679` | 507 s | INFO | `activation_functions` |

**Reproducibility.** Each seed-0 re-run reproduces its previous cited run in every value they
share, and the published figures of E00, E02, E03 and E09 came out byte-identical. Only the
grading and the new rows differ.

**Not cited.** Two earlier E12 runs of the same profile, from `79f1937` (the run the review
read) and `450938e`, give the same values in every row they share with the cited run.

## Test status

| Tier | Command | Result |
|---|---|---|
| Lint | `make lint` | clean; pyright 0 errors |
| Unit + property (CI tier) | `make test` | **260 passed** in 49 s with `-n auto` (233 before, + 27 new); 108 s serially |
| Slow tier | `make test-slow` | **12 passed** in 429 s (under SPEC's 15 minutes): E00, reduced E02–E04 and E12, the lab tests, notebooks 00–03 |
| Paper tier: notebooks 04–11 | `pytest -m paper --nbmake …` with `SUPERPOSITION_RESULTS=results/` | **8 passed** (04, 05, 08, 09, 11 in 30 s; 06, 07, 10 in 15 s), reusing the cited runs |
| Paper tier: replication tests | not re-run as a whole | The quick profiles of E00, E02, E03, E09 and E12 were re-run through the CLI (table above), and the updated tests' assertions hold on those records |
| Lab | in-app browser on the real results | the E12 tab, the regime shares, and a SiLU cell ordered by exact loss; only Streamlit's two health-probe 404s in the console |

**Fresh-clone test (SPEC §11): passed.**
1. `git clone` of the repository at `beca49a`. The clone has no paper PDF, as it should.
2. `uv sync`.
3. `uv run pytest -m "not slow"`: 260 passed, 20 deselected, in 108 s.
4. `uv run superposition run intro_pentagon --profile quick`: **PASS 11/11**.
5. `uv run superposition app`: the lab served the clone's Home page with the status table read
   from the cited runs.

One stale string found there ("E12 arrives in Phase 6") is fixed.

New unit tests:
- E12 theory:
  - the quadrature is exact for piecewise-linear activations, and within 1e-9 of adaptive
    quadrature for the smooth ones;
  - the 1D terms are special cases of the 2D integral;
  - Monte-Carlo checks of the exact loss for five activations;
  - vectorized against single-model losses;
  - E12's wider box leaves E02's ReLU labels unchanged.
- E12 report logic:
  - each activation's own optimal weights reproduce its theory;
  - changes and shares are counted correctly;
  - the compass search finds a quadratic's minimum;
  - the unrestricted optimum never ends above the families;
  - ties are kept;
  - phase-cell models are ordered by exact loss but vote by sampled loss.
- Review fixes:
  - E03: a band between 2/5 and 3/8 is not two clusters;
  - E09: role-consistent motifs, and fully named inhibitors;
  - provenance: untracked code makes the tree dirty.

## Acceptance table

| Exp. | Status | Graded | Change in Phase 6 |
|---|---|---|---|
| E00 | PASS | 11/11 | exact linear loss reported |
| E01 | PASS | 12/12 | docs: restart selection noise (±2 features) |
| E02 | PASS (cited seed) | 6/6 | **n = 3 agreement fails on held-out seeds 1 and 2 (mean 0.792); open for the owner** |
| E03 | **PARTIAL** | 3/4 | the cluster criterion is tightened; only the digons are backed by polytopes |
| E04 | PASS | 9/9 | docs |
| E05 | PASS | 2/2 | docs |
| E06 | PARTIAL | 2/4 | docs: n = 20, m = 5 is a real case of non-tracking |
| E07 | PASS | 3/3 | docs |
| E08 | PARTIAL | 2/4 | docs: the neuron picture and the dead-side explanation narrowed |
| E09 | PASS | 3/3 | criteria tightened; still passes |
| E10 | PARTIAL | 1/3 | docs |
| E11 | PARTIAL | 7/8 | docs; the held-out seed no longer "confirms the window" |
| E12 | INFO | — | new extension |

Overall: seven PASS (E00, E01, E02, E04, E05, E07, E09), five PARTIAL (E03, E06, E08, E10,
E11), no FAIL, and E12 report-only. E02's PASS carries the open seed question.

**E12's results** (n = 2, m = 1, 8 × 8 grid, one seed):

| Activation | Superposition cells: vote / family theory / unrestricted | Same-sign starts reaching superposition |
|---|---|---|
| relu | 39 / 46 / 48 | 117/223 |
| leaky_relu | 39 / 46 / 48 | 117/223 |
| gelu | 22 / 42 / 42 | 3/204 |
| silu | 16 / 28 / 30 | 2/133 |
| softplus | 48 / 46 / 46 | 216/223 |
| sigmoid | 48 / 46 / 48 | 223/223 |

**Phase 6 exit criteria** (SPEC §11):

| Criterion | Evidence | Status |
|---|---|---|
| E12 | module, config, figures, INFO acceptance, REPRODUCTION section, notebook 11, an app tab; cited quick run | done |
| Independent review, logged in REVIEW.md | 5 reviewers, 59 findings, each resolved or open | done |
| Complete every doc and the README gallery | README (gallery E00–E12, status table), REPRODUCTION, REVIEW, STUDY_GUIDE, GLOSSARY, PAPER_MAP, DESIGN, CHANGELOG, PLAN, CLAUDE.md | done |
| Fresh-clone test | clone → `uv sync` → fast tests → quick E00 → app | passed |
| Closing steps | `make lint test` green; quick profiles run; figures published; docs updated; committed | done |

## Decisions made

- **D-601** Exact losses for any output activation by split Gauss–Legendre quadrature; the
  accuracy check is reported per run.
- **D-602** E12's design: what varies, a search box that is the same for every activation,
  the profiles, and what is reported.
- **D-603** E03: a Dᵢ cluster counts only in its polytope. This tightens D-306.
- **D-604** E09: role-consistent motifs; fully named neurons; a baseline on untrained weights.
- **D-605** E02: held-out seeds with a reading rule fixed in advance; the starting sign; exact
  losses.
- **D-606** Untracked code makes a run dirty.
- **D-607** How the review was run.
- **D-608** E12: the unrestricted optimum, best models by exact loss, tied rankings.
- **Corrected in place:** D-404 (a cosine typo) and D-408 (an estimator claim), each with a
  note.

## Open issues

1. **E02 (R1-01; your decision).** The n = 3 agreement passes on the cited seed (0.819) and
   fails on seeds 1 and 2 (0.764, 0.792). The status table shows the cited PASS, with a note.
   Should E02 become PARTIAL?
2. **E03 (R2-01; your decision).** At E03's graded edge threshold (0.05) only the digons
   are separate factors. At 0.2, pentagons and an antiprism separate (notebook 05). Should the
   cluster criterion read factors at 0.2? Choosing it now, after seeing the result, would be
   tuning, so I left it.
3. **Why E02's stuck models are more common than p. 15 suggests (R1-02).** It is the starting
   sign. The paper's initialization or optimizer, both unstated, are candidates, and consulting
   the reference notebook under SPEC §2's rule is an option.
4. **Restart selection on noisy losses:** E01 (R1-06), E03 (R2-10) and E08 (R3-06). Selecting
   on more samples, or on exact losses, would need re-runs.
5. **E08 (R3-07, R3-10, R3-14):** paired diagnostic variants; suppressing the dense
   side-diagnostic rows; test pins.
6. **E12 (R5-02, R5-03, R5-07, R5-11):** longer training for softplus and sigmoid; boundary
   positions below the grid's resolution; more seeds; the full profile with n = 3, m = 2.
7. **Fast tier time.** 49 s with `-n auto` (`make test`, CI), but 108 s serially. SPEC §5's
   "under 60 s" holds only in parallel. The cost is the app tests' shared fixture and a few
   theory tests.
8. **Carried over:**
   - full profiles not run (D-R06);
   - the owner placeholders in LICENSE, CITATION.cff and the notebooks' `<OWNER>`;
   - E10's fragile FVU bar and E11's pentagon window (rectification report);
   - E09's inhibition discrepancy (D-R03).

## Definition of done (SPEC §12)

| Item | Status |
|---|---|
| All phases complete, exit criteria verified | yes (phase reports 0–6; the rectification reports) |
| CI green | yes at `b50d062` (after the setup-uv fix). This phase's commits are not pushed, so CI has not run on them; locally the CI commands pass |
| Fresh-clone test | passed (above) |
| Every experiment has a quick run, published figures and an honest REPRODUCTION entry | yes (E00–E12) |
| The app and all notebooks run | yes. The app ran from the fresh clone and from the main tree. All twelve notebooks ran this phase: 00–03 in the slow tier, 04–11 against the cited runs |
| Docs complete; every REVIEW.md item resolved or listed as open | yes |

## What you should look at and learn from this phase

1. **Read docs/REVIEW.md before anything else.** It is a list of the ways a careful
   replication can still overclaim. Examples:
   - a criterion that counts the same band of values twice (E03);
   - a detector that matches a weight pattern rather than the mechanism (E09);
   - a PASS that holds on one seed and not on two others (E02);
   - an explanation that sounded right until its own numbers were read again (E08's dead
     sides).
2. **Open the lab's Phase diagrams page, E12 tab.**
   - Compare SiLU's three diagrams (vote, family theory, unrestricted).
   - Pick a SiLU cell where the vote says "dedicated" and the theory "superposition".
   - The table lists the models by exact loss: the best one agrees with the theory, and the
     stuck ones started with same-sign weights.
3. **Run notebook 11.** Its section 2 solves one cell exactly for all six activations. The
   ReLU's antipodal pair costs 0.0033 there; the same pair costs 0.0128 with SiLU.

Three questions:

1. In m = 1, why must a model whose two weights start with the same sign pass through a worse
   state to reach the antipodal pair? Why would GELU and SiLU make that crossing harder than the
   ReLU, while softplus and sigmoid make it easier?
2. E02's n = 3 agreement is 0.819, 0.764 and 0.792 on three seeds, against a bar of 0.80. What
   status would you give it, and what would you want to know first?
3. The E03 reviewer showed that one band of Dᵢ near 0.39 counts as both a pentagon cluster and
   an antiprism cluster. What is the general lesson for criteria that bin a continuous quantity
   by nearest reference value, and how would you design one that cannot be fooled this way?
