# Phase 3 report: theory and geometry

Date: 2026-09-30 · Draft: `4355d5d`, `acec078` (2026-09-29) · Closed in a rectification session:
`617e88a`..(this report) · SPEC §11 Phase 3, §6 `theory.py`, §7 E02–E04, §8 pages 3–5, §9
notebooks 03–05

> **How this phase ran.** Phase 3 was drafted on 2026-09-29 and stopped at a usage limit with
> `theory.py`, E02–E04 and two plotting modules written but not registered, configured, tested,
> run or documented. Phases 4 and 5 were then built beside it (D-401). This session closed it.
> It was a rectification session, not Phase 6: no new experiment was started.

## Before changing anything

**Was anything else writing to the repository?**
- *Other Claude sessions:* four session processes had this repository as their working
  directory (Phases 0–1, 3, 4 and 5). All reported `isRunning: false` and had no child
  processes. There were no scheduled tasks and no cron jobs.
- *Leftover scratchpad jobs:* no processes. The Phase 3 draft's scratchpad held finished
  exploratory outputs: E02 grids on CPU (10,000 steps × batch 256, and a variant with a 10×
  cosine learning rate) and on Apple MPS (50,000 × 1,024), and E03 sweeps on CPU (10,000 × 256)
  and MPS (50,000 × 1,024). They are summarized in D-309 as uncited evidence.
- *Four orphaned Streamlit servers* (ports 64368, 51003, 57762, 57952; parent PID 1; 10–19
  hours old). They only read. The cause was a test bug, fixed in this session (below). The four
  servers are still running; see open issue 9.
- *VS Code* had the repository open with `files.autoSave: afterDelay`,
  `git.enableSmartCommit: true` and `git.confirmSync: false`. That explains the earlier
  external commits. It also happened during this session (below).
- Nothing had written to the repository for 9.5 hours; the tree was clean at `9afa841`.

**Commits `7bde9b1` and `7ac6491`** (both made under the owner's identity from outside the
Phase 4 session, both touching only `docs/progress/phase-4.md`):
- `7bde9b1` added a 202-line draft of the Phase 4 report with two unfilled placeholders:
  `SLOW_RESULT` (a row of the slow-test table) and `E08_FULL_NOTE` (in open issue 1).
- `7ac6491` (+25/−15) filled `E08_FULL_NOTE` with the E08 full-profile result, added a commit
  note, updated the two E08 rows of the acceptance table and reworded some prose;
  `SLOW_RESULT` was still unfilled.
- `56d59df` then filled `SLOW_RESULT`, and `ed3ca74` rewrote the commit note. The report at
  `HEAD` has no placeholders. History was not rewritten.

**External commits during this session.** Two commits were made under the owner's identity from
outside this session, most likely by VS Code's Commit & Sync, and each swept up my uncommitted
work in progress:
- `92525fc` (11:19:22, "Update PAPER_MAP and add new notebook for mathematical understanding"):
  the Phase 3 entries of docs/DECISIONS.md (without D-309, added later in `a7dc9db`), the
  Phase 3 rows of docs/PAPER_MAP.md, and notebook 03.
- `5e50b46` (16:10:00, "docs: update changelog and documentation for Phase 3 completion; …"):
  the CHANGELOG entry, CLAUDE.md, PLAN.md, README and D-310's slow-tier list, all as intended,
  plus an intermediate version of the Phase diagrams page and `plotting/interactive.py` that
  still tried click selection; `55b9b0a` replaced it.

History was not rewritten. `origin/main` was also advanced from outside this session (to
`5e50b46` at the time of writing); I did not push.

## What was built

**Library**
- `theory.py` (from the draft, now tested): the linear loss decomposition, the exact 0- and
  1-sparse terms, the exact expected loss of a ReLU output model whose outputs each read at most
  one other feature, weight families W ⊥ v with their exact optima, and theoretical phase
  diagrams (D-301, D-302). All expectations are closed form.
- `metrics.component_geometry` and `metrics.geometry_census` (from the draft, now tested):
  tegum factors named by features, dimensions and uniformity (D-308).
- Named variant profiles: a config file may define profiles beyond `quick` and `full`, and
  `superposition run --profile` accepts them. Used for the batch comparison (D-309).
- `lab`: loaders for the three new pages, and `phase_cell_shared_losses`, which compares a
  phase-diagram cell's trained models with every family, exactly where possible and on shared
  samples otherwise.

**Experiments** (registered, each with quick, full and batch1024 profiles):
- **E02 `phase_change`**, **E03 `uniform_geometry`**, **E04 `nonuniform_geometry`**. Changes to
  the draft before the cited runs: E04's lint suppressions replaced by real fixes, its Spearman
  helper made safe on constant input, and its best-restart selection moved to 200,000
  evaluation samples in every part (D-307). After the runs, plotting only: E02's figures tell
  the two n = 3 "dedicated" families apart and count the voters.

**Lab**
- **Phase diagrams:** a map of either problem size under four views, a cell picked with two
  sliders, and per cell the ten trained models, the vote, representative weights and every
  family's optimum. D-311 explains why sliders replace SPEC §8's click.
- **Geometry:** D* and Dᵢ curves; per sparsity the geometry graph, the Dᵢ histogram and the
  tegum factors.
- **Non-uniform:** the perturbed feature's density slider with the crossing loss curves;
  correlated sets; local bases; the collapse.
- All three were checked in the in-app browser against the cited runs, which led to D-311 and
  three layout fixes.

**Notebooks:** `03_the_math` (exact computations only), `04_phase_change` and `05_geometry`.
They follow the learning-path format: goals, predictions before each run, four exercises with
solutions. Every solution was run against the cited runs. Notebooks 04 and 05 compute the
tables behind the report's narrative: the disagreement analysis and the census at three edge
thresholds.

**Tests**
- `test_theory.py` (27 tests): F against nested adaptive quadrature, and every exact loss
  against Monte Carlo. Families, optimizer and diagrams on known answers.
- `test_phase3_experiments.py` (12): every E02–E04 criterion passes on ideal hand-built
  weights and fails on broken ones. Labeling the theory's optimal weights reproduces the
  theory's labels in every cell.
- Polytope naming (2), `phase_cell_models` and `phase_cell_shared_losses` (2), named profiles
  (1), AppTests for the three pages (8 cases).
- Replication: reduced E02–E04 tests in the slow tier, and full quick-profile tests in the paper
  tier.

**Test tiers** (D-310): `slow` stays under 15 minutes; replication tests that re-run a whole
quick profile move to a new `paper` marker with `make test-paper`. The paper tier is opt-in:
it is deselected unless the `-m` expression names `paper`, so `pytest -m "not slow"` (CI, and
SPEC's fresh-clone test) stays fast.

**A test bug fixed:** `test_superposition_app_launches` terminated only the `superposition app`
CLI, whose Streamlit child kept running with its output piped to a reader that had gone away.
It could not even log its own shutdown, so SIGTERM left it running. The test now starts the
command in its own process group with output discarded and stops the whole group; a before and
after check confirmed the leak and the fix.

## Test status

| Tier | Command | Result |
|---|---|---|
| Lint | `make lint` (ruff check, ruff format --check, pyright basic) | clean; pyright 0 errors |
| Unit + property (CI tier) | `make test` | **226 passed in 41 s** (174 before this session, 52 new) |
| Slow tier | `make test-slow` | **11 passed in 5 min 56 s** (budget: 15 min): E00 (68 s), the reduced E02–E04 tests (22, 18 and 52 s), the three lab tests, notebooks 00–03 (notebook 02: 170 s, with `device="auto"` on Apple MPS); 18 paper-tier items deselected |
| Paper tier: E02–E04 | `pytest -m paper tests/replication/test_{phase_change,uniform_geometry,nonuniform_geometry}.py` | **3 passed** in 494, 373 and 293 s, from an empty results folder: the cited outcomes reproduce |
| Paper tier: notebooks 04–05 | `pytest -m paper --nbmake notebooks/04_*.ipynb notebooks/05_*.ipynb` | from an empty results folder (each trains its quick runs): **2 passed** in 284 and 382 s (`device="auto"`, Apple MPS); with `SUPERPOSITION_RESULTS=results/` (reusing the cited runs): 2 passed in 24 and 3 s |
| Tier selection | `pytest --collect-only` with each `-m` | `-m "not slow"` selects none of the 29 replication and notebook items; `-m slow` 11; `-m paper` 18 |

**A tier bug found and fixed.** In the first combined paper-tier run, notebooks 04 and 05 failed:
the replication tests had just made figure-less quick runs in the session's shared temporary
results folder, and the notebooks reused them. Notebooks 06–10 had the same reuse helper and
would have failed the same way under `make test-paper`; earlier phases ran them only against
`results/`. All seven now reuse only runs with figures (`5d9cd7e`), checked with a figure-less
newer copy of each cited run next to the original.

The rest of the paper tier (E01, E05–E11 and notebooks 06–10) was not re-run in this session;
see open issue 10.

## Acceptance table

Cited runs: quick profile, CPU (Apple M2), seed 0, clean commit `230a51d`.

| Exp. | Criterion | Value | Status |
|---|---|---|---|
| E02 | n = 2, m = 1: three regions in the empirical diagram | 88 superposition, 29 not learned, 27 dedicated cells | PASS |
| E02 | n = 2, m = 1: three regions in the theoretical diagram | 106 / 19 / 19 | PASS |
| E02 | n = 2, m = 1: empirical and theoretical labels agree on ≥ 80% of cells | 0.875 (126/144) | PASS |
| E02 | n = 3, m = 2: three regions, empirical | 79 superposition, 47 dedicated, 18 not learned | PASS |
| E02 | n = 3, m = 2: three regions, theoretical | 53 / 72 / 19 | PASS |
| E02 | n = 3, m = 2: agreement ≥ 80% | 0.819 (118/144) | PASS |
| E02 | disagreements and McGrath's confused regime | best model has the theory's label in every disagreeing cell; confused wins 16 / 13 cells without biases, none with | INFO |
| E03 | D* ≈ 1 up to 1/(1 − S) = 1.2 | 0.998, 0.999 | PASS |
| E03 | plateau at 0.5 ± 0.03 (report the span) | 2.34–3.79 (paper ≈ 2.6–3.7) | PASS |
| E03 | falls after the plateau, ≈ 0.25 at 8 | 0.264 at 7.85 | PASS |
| E03 | Dᵢ cluster at ≥ 3 characteristic fractions | 2/3, 1/2, 2/5, 3/8 (no 3/4) | PASS |
| E03 | Σᵢ Dᵢ against m = 30; tegum-factor census | 29.84–29.99 | INFO |
| E04 | (a) denser → neighbours repel | ρ = 0.991; 76.3° at 10× | PASS |
| E04 | (a) sparser → neighbours close in | ρ = 1.0; 55.7° at 0.126× | PASS |
| E04 | (a) collapse to two digons (report the threshold) | best is two digons up to 0.251× (paper ≈ 0.4×) | PASS |
| E04 | (b) anticorrelated more negative than correlated | mean cos −1.0 vs −0.010 | PASS |
| E04 | (b) correlated orthogonal when there is room | \|cos\| 0.013, 0.007 | PASS |
| E04 | (b) correlated side by side when there is not | 6 represented, pairs adjacent (≈ 65°) | PASS |
| E04 | (c) local almost-orthogonal bases | within 0.007 < across 0.216 | PASS |
| E04 | (d) sparse hexagon, pairs adjacent | yes, within-pair cos 0.41–0.43 | PASS |
| E04 | (d) collapse with density | ρ = 0.979; mean cos 0.42 → 1.0 | PASS |

Overall: **E02 PASS (6/6), E03 PASS (4/4), E04 PASS (9/9).** Quick-profile wall-clock on an
Apple M2 CPU: 487 s, 365 s and 287 s (SPEC §5: ≤ 10 minutes each).

**The batch-256 decision (D-309).** The draft had chosen batch 256 for the quick profiles "so
they fit 10 minutes on a CPU", pending two exploratory GPU runs. Both had finished and are
summarized in D-309, but the decision rests on six CLI runs from the same clean commit on CPU:
each quick profile against a `batch1024` profile that differs only in the batch size (1,597,
1,472 and 952 s). Batch 1024 changes no criterion's outcome:
- E02's labels are identical in 284 of 288 cells; the other 4 are 5–4 votes flipping.
- E03 has the same plateau, tail and clusters.
- E04 has the same collapse threshold and geometries.

The only visible effect is E03's transition from 1 to 1/2, which neither setting has converged.
Quick profiles keep batch 256; full profiles use 1024.

**Phase 3 exit criteria** (SPEC §11: `theory.py`; E02, E03 and E04; their notebooks and app
pages):

| Criterion | Evidence | Status |
|---|---|---|
| `theory.py` | implemented; 27 tests against Monte Carlo and quadrature | done |
| E02–E04 implemented, run, reported | registered with configs; quick runs from a clean tree, pinned in `docs/cited_runs.yaml`; REPRODUCTION.md sections with honest discrepancies | done |
| Notebooks 03–05 | committed without outputs; 03 in the slow tier, 04–05 in the paper tier; all pass under nbmake | done |
| App pages Phase diagrams, Geometry, Non-uniform | AppTests with and without runs; checked in the in-app browser against the cited runs (a lagging click selection was replaced by sliders, D-311; three layout fixes) | done |
| Closing steps (§11) | `make lint test` green; quick profiles run; figures published (≤ 200 KB each); REPRODUCTION, CHANGELOG, PLAN updated; committed | done |

## Decisions made

- **D-301** Exact expectations in closed form (F integrated in y analytically, then exactly by
  Gauss–Legendre on each cubic piece), validated against quadrature and Monte Carlo.
- **D-302** Families W ⊥ v with one scale per tegum factor and one bias per feature; grid then
  golden section; McGrath's confused family and a zero-bias variant reported beside them.
- **D-303** E02 grids: density on the y axis (the figures' reading of p. 14), 12 × 12 with
  importance 1 kept off the grid.
- **D-304** E02 labels and the vote (discard the worst of ten, ties to the lowest loss).
- **D-305** E02 criteria for both problem sizes, with the draft's exploratory agreement
  disclosed.
- **D-306** E03 thresholds read from p. 17, with the disclosure that the draft had seen an
  exploratory D* curve.
- **D-307** E04 setups, criteria and 200,000-sample restart selection.
- **D-308** Naming tegum factors by (features, dimensions, uniformity).
- **D-309** Batch 256 for E02–E04's quick profiles, from logged CLI runs.
- **D-310** Test tiers: slow under 15 minutes; opt-in paper tier.
- **D-311** The Phase diagrams page picks cells with sliders: click selection arrived one click
  late in Streamlit 1.64.0.

## Open issues

1. **E02's n = 3, m = 2 agreement is narrow** (0.819 against 0.80). The disagreements are
   close votes lost to models stuck in worse minima: the best voting model has the theory's
   label in every disagreeing cell (docs/REPRODUCTION.md). More restarts per cell would
   likely raise the agreement; the quick profile keeps the paper's ten.
2. **The paper's families are not the whole solution space.** In 4 n = 2 cells an asymmetric
   pair beats W ⊥ [1, 1] (which forces equal lengths) by 5.4–6.5% in exact loss. In one dense
   n = 3 cell a joint superposition of all three features beats every family by 1.3%, and p. 15
   excludes that family on purpose. Neither changes a label or a graded number. A continuous
   family (free lengths, or W ⊥ v for free v) would be the natural extension.
3. **No exact loss for general n = 3 models.** Trained n = 3 models read two other features per
   output, which needs 3D integrals. Their comparisons with the families use shared samples,
   which are only indicative at densities 0.01–0.015: a 4,000,000-sample comparison showed a
   spurious 16% gap there before the exact family optimum was checked by brute force.
4. **E03 differences from pp. 17–19:** no tetrahedron (3/4) cluster; the 2/3 cluster is not
   backed by triangle factors in the cited run; D* leaves 1 later than the paper's curve
   (≈ 0.1 higher around 1/(1 − S) = 1.5–1.9). Longer training moves it toward the paper.
5. **E04 differences:** the pentagon snaps at ≤ 0.25× the others' density, not ≈ 0.4×; the
   collapse regimes of (d) sit at somewhat higher densities than p. 25 gives.
6. **Full profiles of E02–E04 were not run.** Estimates are in the configs: about 2, 5 and 2
   hours on a laptop CPU.
7. **CI on GitHub is broken.** All 14 CI runs so far (since the first push on 2026-09-29) failed
   at job setup: `astral-sh/setup-uv@v10` does not exist. This is outside this session's list:
   pin an existing version of the action in `.github/workflows/ci.yml` and `slow.yml`.
8. **External commits.** VS Code's autosave and smart commit can commit and push work in
   progress (`92525fc` and `5e50b46` in this session; `e3d80a3`, `7bde9b1` and `7ac6491`
   before). An autosave
   of a tracked file while a run starts makes that run uncitable (CLAUDE.md). Consider turning
   off smart commit and sync for this repository while agent sessions run.
9. **Four orphaned Streamlit servers are still running** (PIDs 1934, 21501, 71273, 74232; ports
   64368, 51003, 57762, 57952). They are harmless leftovers of the old launch test, and I left
   them because nothing asked me to stop them. Their output pipe is broken, so SIGTERM does not
   stop them: `kill -9 1934 21501 71273 74232`.
10. **The paper tier was run only in part.** This session ran the new E02–E04 paper tests and
    notebooks 04–05. The other paper tests (E01, E05–E11, notebooks 06–10) were verified in
    their own phases and were not re-run: `make test-paper` takes hours.
11. **No independent verifier.** SPEC §11 pairs every experiment with a verifier agent; this
    session's instructions were not to start agents unasked. Phase 6's independent review
    should cover E02–E04.
12. **Carried over:** E08's computation in superposition (Phase 4, issue 1); E06's n = 20,
    m = 5 setting; owner placeholders (LICENSE, CITATION.cff, notebooks' `<OWNER>`).

## What the owner should look at and learn from this phase

Open `docs/figures/phase_change/phase_n2_m1.png` next to p. 15. Then open the lab's **Phase
diagrams** page, pick one of the cells marked × with the sliders, and compare the trained models' losses with
the winning family's. The vote lost, but the best model found the theory's solution. The other
models started with both weights on the same side and never got out (notebook 04, exercise 3).
Then read notebook 03 section 4 and notebook 04 section 3 together: the first shows how a phase
diagram comes out of a few exactly solved families, and the second shows where those families
stop being the whole story (asymmetric pairs). Finally, drag the **Geometry** page's slider from
1/(1 − S) = 2 to 6 and watch the digons give way to pentagons and square antiprisms, then to
one big component.

Three questions:

1. In E02, why does a model whose two weights start with the same sign end with the extra
   feature dropped, even when the antipodal pair is much cheaper? What would change if the
   paper had used more than ten models per cell, or dropped more than one?
2. Why can the family W ⊥ [1, 1] not express an asymmetric pair [a, −c] with a ≠ c, and why is
   the asymmetric pair better when the two features differ in importance? (Hint: which output
   benefits from a positive bias?)
3. E03's Dᵢ add up to m at every sparsity, yet beyond 1/(1 − S) ≈ 6 they no longer sit at the
   paper's fractions. What does Σᵢ Dᵢ = m tell you about the geometry, and what does it not?
