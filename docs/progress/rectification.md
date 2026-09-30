# Rectification report (session 2)

Date: 2026-09-30 · Commits `81ad83c`..(this report) · Owner decisions D-R01–D-R06, and the
implementation decision D-R07, in docs/DECISIONS.md

> **Scope.** This was the second rectification session, not Phase 6. It applied six owner
> decisions about E05 and E08–E11. It started no new experiment, and it ran no full profile.

## Before changing anything

- The tree was clean at `3c05719`, and `origin/main` pointed at the same commit.
- VS Code had the repository open, as in session 1 (autosave, smart commit, auto-sync). To keep
  it from committing half-written work:
  - reports and long docs were drafted in the scratchpad;
  - code was committed in small pieces;
  - the cited runs were launched from a clean detached worktree.
- No external commit happened during this session. Every commit after `3c05719` is this
  session's. Nothing was pushed.
- The four orphaned Streamlit servers of Phase 3's open issue 9 are no longer running (no
  `streamlit` process), and no other session or job was writing to the repository.

## What the owner decided, and what came of it

**1. E08: consult the references on dead sides; re-derive the sparse criterion (D-R01).**
- *Consulted:* Anthropic's archived `toy_models.ipynb` and ARENA §1.5.4 (pages 1 and 2). This
  was for diagnosis only, and no code was copied. D-R01 records exactly what was read.
- *What they showed:*
  - Neither has a fix for dead output sides. Anthropic's notebook contains only the intro
    model, and ARENA's |x| model starts the output bias at zero, as we do.
  - The one substantive difference is the weight scale. ARENA's Kaiming init runs over the
    whole 3-D tensor, which makes each weight about 5.3× smaller than SPEC §3's Xavier init.
  - The paper's "1,000 models" (p. 33) concerns E07, not |x|.
- *Diagnostic variant* at that scale (0.187 × Xavier, 4 restarts, never the default):
  - at 1 − S = 0.01 the loss falls 6× (0.00785 → 0.00127), 22 of 109 dead sides revive
    (against 2 of 87), and 25 features are computed well, more than m/2;
  - at 0.001 almost nothing changes (loss 0.00103 → 0.00099).
- *Re-derived criterion,* from p. 37's "more features than it has neurons" while "usefully
  approximat[ing] the computation": more than m = 40 features with a loss below 75% of
  predicting the mean, at 1 − S = 0.001.
- *Result:* 23 features, a FAIL. SPEC's original criterion, reported next to it, gives 3
  (> 20 needed). The uncited full profile gives 29 by the new criterion.
- *Within one sparsity* (footnote 17), the trained models' loss is 1.3× (at 1 − S = 0.3) to
  54× (at 0.001) that of a model with no superposition at all (the 20 most important features
  computed exactly). The failure is one of optimization.
- *Status:* **E08 stays PARTIAL 2/4.**

**2. E05: stage B as a documented variant (D-R02).**
- The octahedron criterion now reports its fraction: 48/48 restarts (100%).
- The trajectory types: six kinds, varying at step C and in whether the collapsed stage lasts
  1,000 steps.
- New INFO `stage_b_placement`: while the sets are collapsed, their centroids sit close to
  orthogonal (median cosine −0.04 to −0.06). Only 6–18% of collapsed snapshots are opposed as
  in the paper's panel B, although at every density some run ends the stage opposed.
- This is reported as the correlated-feature orthogonality of pp. 23–24, not as a discrepancy.
- *Status:* **E05 stays PASS 2/2.**

**3. E09: the density pilot as a cited run; the inhibitor discrepancy (D-R03).**
- *Pilot:* the pilot of D-502 is now profile `pilot`, re-run through the CLI and cited as
  `mech_interp_abs:pilot` (16 restarts, all four densities). It supports the choice of the
  graded densities:
  - no superposition at 1 − S = 0.5 (rebuild 100%, no motif);
  - at 0.3 and 0.2, everything named and the motif in 6–14 of 16 restarts;
  - at 0.1, unexplained neurons (2, 6 and 1) and a failing rebuild.
- *Inhibitor discrepancy:* p. 39 says inhibition converts positive interference into negative
  interference. Ours cancels it: the net is within −2.8% to +2.5% of the interference, median
  +0.6%, and only 6 of 18 best-model motifs are negative. This is reported as a discrepancy on
  that sub-claim.
- *Status:* **E09 stays PASS 3/3.**

**4. E10: grade an SAE selected without ground truth (D-R04).**
- *Rule,* fixed and committed before the run: the sparsest SAE (held-out L0) among those with
  held-out FVU ≤ 0.1.
- *Result:* it recovers 3/5 pentagon features at cos > 0.95 (FAIL), 11/11 at n = 20,
  1 − S = 0.01 (PASS), and 10/13 at 0.001 (FAIL). **E10 goes from PASS 3/3 to PARTIAL 1/3.**
- *Upper bound:* the oracle-best SAE still recovers everything.
- *Pass rates:* 10/27, 16/18 and 6/18 settings.
- *At 0.001:* the recovering λ = 0.3 SAEs explain 88% of the variance, just short of the bar.
  The bar was not moved.
- *TopK:* on the E03 model the rule picks TopK k = 16 (2.2% recovered, FVU 0.057) over k = 8
  (100%, FVU 0.176).

**5. E11: the pentagon is PARTIAL; held-out check; p. 42 (D-R05).**
- *PARTIAL:* "falls to none" is now PARTIAL when the only superposition-free model keeps one
  feature (a graded PARTIAL status). **E11 goes from PASS 8/8 to PARTIAL 7/8.**
- *Held-out check:* the unchanged quick profile on seed 1, cited as
  `l1_removes_superposition:heldout`, gives PARTIAL 5/8:
  - n = 20 passes 4/4 again;
  - the pentagon's window moves to λ = 0.6 (again with one feature), and "declines" and "loss
    rises" fail. Its restarts miss the lower-objective states that seed 0 found at λ = 0.55
    and 0.65.
- *Against p. 42:* L1 kills features from the least important up on both seeds. The
  basis-aligned antipodal pairs survive longest, as "especially if they're not basis aligned"
  implies. The paper's "quite easy to get rid of superposition" holds for the pentagon only
  with one feature left.

**6. Full profiles and runtimes (D-R06).** No full profile was run. Every runtime that was not
measured is now labeled as an estimate: configs, README, CLAUDE.md, the Makefile, notebooks 02
and 07, the Phase 3–5 reports, and three DECISIONS entries annotated in place. The
quick-profile runtimes in the configs and notebooks were updated to this session's
measurements.

## What was built

- **Library.**
  - `acceptance.graded`: a graded criterion with an explicit PASS, PARTIAL or FAIL; a PARTIAL
    does not count as passed.
  - `report`: supplementary citations under keys `<experiment>:<label>`, via `superposition
    report --pin EXP --run ID --label L`, rendered into `AUTO:<code>:<label>` blocks (D-R07).
- **E08.**
  - The re-derived criterion, with the original reported as INFO.
  - The within-sparsity references, and the exact model loss from per-feature losses.
  - The init-scale variant, and both variants reported side by side.
  - The capacity figure and the Neurons page show the features usefully computed.
- **E10.** `select_without_ground_truth` and its graded criterion, INFO `oracle_upper_bound`,
  `pass_rates` and `topk_k_contrast`. The selected SAE appears in the pentagon figure and is
  preselected in the SAE lab.
- **E11.** The three-way "falls to none".
- **E05.** The octahedron fraction and `stage_b_placement`.
- **E09.** The `pilot` profile, and the spread of the net interference.
- **Notebooks.** 08 and 10 were updated for the new criteria and rows; 02, 07, 08, 09 and 10
  for runtimes.

## Runs

All runs used the quick profile unless noted, on CPU (Apple M2), launched one after another from
a clean detached worktree at `fd32b02`, with results written to `results/`.

| Exp. | Run ID | Seed | Wall-clock | Status | Cited as |
|---|---|---|---|---|---|
| E09 (profile `pilot`) | `20260930T183948Z-1e8bafbd` | 0 | 212 s | PASS 3/3 | `mech_interp_abs:pilot` |
| E09 | `20260930T184323Z-3fdd1d10` | 0 | 258 s | PASS 3/3 | `mech_interp_abs` |
| E11 | `20260930T184744Z-e8c803d0` | 1 | 546 s | PARTIAL 5/8 | `l1_removes_superposition:heldout` |
| E11 | `20260930T185653Z-c4feb0cf` | 0 | 548 s | PARTIAL 7/8 | `l1_removes_superposition` |
| E05 | `20260930T190604Z-2fae0fc5` | 0 | 518 s | PASS 2/2 | `learning_dynamics` |
| E08 | `20260930T191445Z-49fd30ed` | 0 | 493 s | PARTIAL 2/4 | `abs_computation` |
| E10 | `20260930T192302Z-13117b27` | 0 | 415 s | PARTIAL 1/3 | `sae_recovery` |

**Reproducibility.** Each seed-0 run reproduces its previous cited run exactly in every row that
existed before: E08's per-feature losses and E10's SAE table are identical, and E05's, E09's and
E11's earlier acceptance values are identical. Only the grading and the new rows differ.

**Order.** The pilot and the held-out seed ran before their experiment's cited run, so that the
lab pages and notebooks, which show the latest run, show the cited one.

## Acceptance summary

| Exp. | Before this session | After | What changed |
|---|---|---|---|
| E05 | PASS 2/2 | PASS 2/2 | fraction reached (48/48) and stage B reported; documented variant |
| E08 | PARTIAL 2/4 | PARTIAL 2/4 | sparse criterion re-derived: 23 usefully computed (> 40 needed); original 3 (> 20) |
| E09 | PASS 3/3 | PASS 3/3 | pilot cited; inhibitor discrepancy on a sub-claim |
| E10 | PASS 3/3 | **PARTIAL 1/3** | graded by a rule without ground truth; the oracle passes 3/3 as an upper bound |
| E11 | PASS 8/8 | **PARTIAL 7/8** | pentagon PARTIAL (one feature left); held-out seed PARTIAL 5/8 |

The overall picture is now eight PASS (E00–E05, E07, E09) and four PARTIAL (E06, E08, E10,
E11), with no FAIL.

## Test status

| Tier | Command | Result |
|---|---|---|
| Lint | `make lint` | clean; pyright 0 errors |
| Unit + property (CI tier) | `make test` | **233 passed** in 33 s (226 before + 7 new) |
| Notebooks 06, 08, 09, 10 (paper tier) | `pytest -m paper --nbmake …` with `SUPERPOSITION_RESULTS=results/` | **4 passed** in 22 s, reusing the cited runs |
| Paper-tier replication tests, E05 and E08–E11 | not re-run | Their updated assertions were applied to the cited runs' acceptance records and all hold; the tests themselves would re-run the same deterministic quick profiles |
| Lab | headless render of Neurons, SAE lab, Dynamics, Home against the cited runs | no exceptions; the SAE lab preselects the rule's SAE |
| Slow tier | not run | — |

New unit tests: graded PARTIAL statuses; supplementary citations (pinning, status rows,
rendering, missing block, invalid label; the CLI refuses `--label` with several pins); E08's
criteria (a case that passes on m but not m/2) and its reference losses; E10's selection rule
(recovery plays no part, FVU bar, ties, fallback); E11's PARTIAL; E05's stage-B placement.

## Decisions made

- **D-R01** (owner) E08: the consultation, the init-scale variant, and the re-derived criterion.
- **D-R02** (owner) E05: stage B as a documented variant.
- **D-R03** (owner) E09: the pilot as a cited run; the inhibitor discrepancy.
- **D-R04** (owner) E10: the selection rule, disclosing what was known when it was fixed.
- **D-R05** (owner) E11: the pentagon PARTIAL; the held-out seed.
- **D-R06** (owner) no full profiles; runtime labels.
- **D-R07** supplementary cited runs.

## Open issues

1. **E08's computation in superposition still does not reproduce.** The smaller init helps at
   1 − S = 0.01 but not at 0.001, where each side sees about 5,000 examples in 10,000 steps.
   Untested next steps:
   - the small init with longer training;
   - the small init combined with a positive output bias;
   - more neurons (footnote 16).

   Each would be a variant, not a change of the replication default.
2. **E10's rule is fragile at 1 − S = 0.001.** The λ = 0.3 SAEs that recover everything sit at
   FVU 0.117–0.121, just outside the pre-fixed 0.1 bar. Any other rule (a different bar, or the
   downstream loss with the SAE spliced in) would have to be fixed before a run, as this one
   was. How recovery depends on TopK k (E03 model) is still untested.
3. **E11's pentagon between λ = 0.5 and 0.7.** 16 restarts do not settle which near-degenerate
   state wins, and the cited seed's clean "declines" and "loss rises" there are not robust. The
   full profile (64 restarts) was not run.
4. **E09's inhibition** cancels the interference rather than converting it; this stays reported.
5. **Full profiles** of E02–E11 were not run (by decision); their runtimes are estimates.
6. **The paper tier was not re-run as a whole.** The last full pass of each paper test is in
   its own phase report, and E02–E04's is in Phase 3's.
7. **Carried over:**
   - CI was broken (`astral-sh/setup-uv@v10` in both workflows). Fixed after this report, at the
     owner's request, by pinning `v10.2.0`;
   - E06's n = 20, m = 5 setting;
   - E02's narrow n = 3 agreement;
   - E07 at 1 − S = 0.12;
   - the owner placeholders (LICENSE, CITATION.cff, the notebooks' `<OWNER>`).

## What the owner should look at

1. Read E08's within-sparsity table in docs/REPRODUCTION.md. At every sparsity the trained
   model does worse than a model with no superposition at all, 54× worse at 1 − S = 0.001.
   That is the clearest statement of why E08 is an optimization problem, and it only became
   visible once losses were compared within a sparsity, as footnote 17 allows.
2. Open the SAE lab's E10 tab on the pentagon. The selected SAE is preselected: its five
   latents sit close to the five features, but two miss the 0.95 bar. Then compare the E03
   rows of the TopK k contrast.
3. Read E11's held-out paragraph next to the cited run's trade-off table. The order in which
   features die holds on both seeds; the exact λ where superposition ends does not.

Three questions:

1. Why does a smaller initial weight scale revive dead sides at 1 − S = 0.01 but not at 0.001?
   Think about how many examples each side sees, and what pushes the output bias up.
2. The practitioner's rule picks the sparsest SAE with FVU ≤ 0.1. At 1 − S = 0.001 the
   SAEs that recover every feature have FVU ≈ 0.12. What would you look at on a real model to
   notice that a slightly worse reconstruction is the better dictionary?
3. On the held-out seed, E11 lands in the one-feature state at λ = 0.6 and in an antipodal pair
   at 0.65, the opposite of seed 0. With training objectives within 0.03% of each other at
   λ = 0.6, what does "the model at λ" even mean, and how would you report it?
