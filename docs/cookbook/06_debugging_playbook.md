# 06 · Debugging playbook

This chapter collects the decision procedures that this replication needed when a result did
not come out as expected. Each section covers one kind of problem and has four parts:

1. **Symptoms**: what you see.
2. **Decision procedure** **[Recommended]**: what to check first, which evidence settles the
   question, and when to stop.
3. **What happened here**: this repository's example, with evidence.
4. **Checklist** **[Recommended]**: a short list to run through by hand.

Labels follow [README.md](README.md): **[Documented]** means the cited evidence shows it,
**[Reconstructed]** means it was inferred (the text says from what), and **[Recommended]** is
advice, not history. Terms such as *antipodal pair*, *SAE* and *feature dimensionality* are
defined in the [glossary](../GLOSSARY.md). The dates and sessions behind each story are in
[01_timeline.md](01_timeline.md).

**The rule above all the others.** SPEC §2 fixes the order of work when a result does not
reproduce: check the implementation against the paper, then add restarts or steps within
reason, then document what you tried, then report PARTIAL or FAIL with evidence and
hypotheses. A criterion is never weakened to make it pass. **[Documented: SPEC.md §2 "Honest
reporting".]** Every procedure below is a refinement of that order.

| Symptom | Section |
|---|---|
| A sparse or hard setting falls far short of the paper | [1](#1-training-failure-versus-model-limitation) |
| Some restarts reach the paper's solution and others do not | [2](#2-restarts-for-hard-optimization) |
| The "best restart" changes when you re-evaluate it | [3](#3-sampling-noise-in-restart-selection) |
| A reference or sanity number disagrees with yours | [4](#4-a-sanity-value-that-disagrees) |
| A pass rests on a grid, regime or seed you chose after looking | [5](#5-forking-paths-and-held-out-confirmation) |
| A result is graded on a choice only the ground truth allows | [6](#6-selection-by-ground-truth) |
| The end state matches the paper but the path does not | [7](#7-trajectory-variance-versus-failure) |
| A number in the docs has no run behind it | [8](#8-provenance-gaps) |
| A criterion looks wrong after you have seen results | [9](#9-when-and-how-to-revise-a-criterion-honestly) |
| A test fails, or passes for the wrong reason | [10](#10-a-test-that-fails-or-a-test-that-should) |

## 1. Training failure versus model limitation

**Symptoms.** A model trained at the paper's setting does much worse than the paper says. More
steps, restarts or a different learning rate do not close the gap. It is tempting to conclude
"the model cannot do this", which would be a claim about the representation, when the cause
may be the optimizer.

**Decision procedure** **[Recommended]**

1. Check the implementation on a small case with a known answer first. If the small case is
   exact, the code that computes losses and gradients is probably right. **[Recommended]**
2. Build a *reference solution* at the same setting: weights you write down by hand that the
   same architecture can express. Compute its loss on the same task. If your trained model is
   worse than a solution the model class can provably reach, the gap is optimization, not
   capacity. **[Recommended]**
3. Compare losses only within one setting. Footnote 17 (p. 56) warns that losses at different
   sparsities measure different tasks. **[Recommended]**
4. Try the cheap optimization knobs once each (restarts, steps, learning rate, batch size) and
   write down every result, including the failures. If none moves the number toward the
   reference, stop turning knobs. **[Recommended]**
5. Write a diagnostic for your mechanism hypothesis and log it in the run, not in a scratch
   script. Then read its numbers against the hypothesis. A diagnostic that shows *some* effect
   is not proof that the effect explains the whole gap. **[Recommended]**
6. Test the hypothesis with a variant that is never the replication default. Pair it with the
   default runs (same seeds, same instance draws) so that differences are not luck.
   **[Recommended]**
7. Only after an honest attempt, consult a reference implementation to diagnose, and record
   what you read (SPEC §2). **[Recommended]**
8. Report PARTIAL or FAIL. Say "optimization failure" only when step 2's reference supports it.
   **[Recommended]**

**What happened here: E08's dead output ReLUs.** (Decisions are in
[DECISIONS.md](../DECISIONS.md).)

- E08(a), the n = 3, m = 6 model, computed |x| exactly: one positive-side and one
  negative-side neuron per feature, every per-feature loss 0. That cleared the implementation.
  **[Documented: REPRODUCTION.md § E08 "(a) passes exactly".]**
- E08(b) at 1 − S = 0.001 computed 3 features well against SPEC's "more than m/2 = 20".
  Prototypes before the quick profile was fixed tried 2 or 4 restarts, 30,000 steps, learning
  rate 3e-3 and batch 4,096; they gave 8–16 features, and longer training did not help at all.
  **[Documented: D-409 "What was tried"; phase-4.md open issue 1.]** These prototypes were
  scratchpad runs outside the provenance system. **[Documented: a scratchpad prototype
  script, S 92884087 @ 17:04Z; D-409 says "not cited".]**
- The diagnosis: at extreme sparsity each *side* of a feature (xᵢ > 0 or xᵢ < 0) is trained
  only by its own examples. With the output bias at zero, a side whose initial gain is negative
  is clipped by the output ReLU and gets no gradient. INFO `b_side_diagnostics` logged it: 0 of
  89 sides dead at initialization were learned at 0.001, against 23 of 111 alive ones.
  **[Documented: D-409; REPRODUCTION.md § E08 "A contributing factor".]**
- The review then read those numbers again: 111 sides started alive, more than either
  criterion needs, yet only 23 were computed. "Dead sides" was demoted from *the cause* to *a
  contributing factor*, with the untested alternatives listed. **[Documented: REVIEW.md R3-02.]**
- The owner allowed the reference implementations to be consulted for diagnosis. Neither had a
  remedy; the one substantive difference was ARENA's weight scale, about 5.3× smaller. A
  variant at that scale (0.187 × Xavier, never the default) cut the loss 6× at 1 − S = 0.01
  (0.00785 → 0.00127) and revived 22 of 109 dead sides, but changed almost nothing at 0.001
  (0.00103 → 0.00099). **[Documented: D-R01; rectification.md item 1.]** The review noted that
  the variant's restarts start from other random draws than the default runs, so the
  comparison is unpaired. **[Documented: REVIEW.md R3-07 (open).]**
- The settling evidence was a within-sparsity reference (step 2). INFO
  `b_within_sparsity_references` compares each best model with a model that has no
  superposition at all: the 20 most important features computed exactly. The trained model's
  loss is 5.5× that reference when dense and 54× at 1 − S = 0.001 (0.00103 against 0.0000192).
  A model that cannot match the no-superposition solution its own architecture can express is
  failing to optimize. **[Documented: REPRODUCTION.md § E08 "Within one sparsity"; D-R01
  "Footnotes 16 and 17".]**
- E08 stayed PARTIAL 2/4, reported as an optimization failure, not as evidence about what
  superposition can compute. **[Documented: rectification.md acceptance summary.]**

**Checklist** **[Recommended]**

- [ ] A known-answer case passes exactly.
- [ ] A hand-built reference at the same setting exists, and its loss is computed.
- [ ] No loss is compared across sparsities (footnote 17).
- [ ] Every knob tried is listed with its result, failures included.
- [ ] The diagnostic's own numbers support the explanation you write.
- [ ] Variants are paired with the default runs, and never become the default.

## 2. Restarts for hard optimization

**Symptoms.** Training the same setting from different initializations gives different
geometries. Some restarts find the paper's solution; others settle for fewer features or a
worse arrangement.

**Decision procedure** **[Recommended]**

1. Record the outcome of every restart, not only the best. A count per outcome is enough.
   **[Recommended]**
2. If the best restart matches the paper and the others are worse local optima, that is the
   expected behavior for small hidden spaces. Use best-of-k, as the paper does (p. 23), and
   select on the training objective. **[Recommended]**
3. If your statistic aggregates restarts (a vote, a mean), stuck restarts change the answer.
   Find what predicts getting stuck, log it as a diagnostic, and report it. **[Recommended]**
4. Do not write "more restarts would fix it" unless you have run more restarts. **[Recommended]**
5. Stop adding restarts when the best-of-k outcome is the same at two values of k.
   **[Recommended]**

**What happened here.**

- SPEC §3 carries the paper's note that 2D hidden spaces are hard for gradient descent and that
  the authors kept the lowest-loss fit (p. 23). SPEC §7 E00's reference notes add that 3 of 8
  ReLU restarts at S = 0.9 settled for four features in the independent JAX verification run.
  **[Documented: SPEC.md §3 "Optimization notes", §7 E00 "Reference values".]**
- The engine's `select_best` returns the lowest-loss instance per condition and never picks a
  NaN unless a whole group is NaN. **[Documented: `src/superposition/train.py`
  `select_best`.]**
- E00's quick profile reached all five features in 6 of 8 restarts at S = 0.9; the full
  profile (32 restarts) did in 21. Both pass, so best-of-8 was enough. **[Documented:
  REPRODUCTION.md § E00 "Robustness"; runs `20261001T071529Z-5eeb804c`,
  `20260929T090615Z-eb7b9e3b`.]**
- E02 aggregates ten models per cell by a vote. Phase 3's report said the narrow n = 3
  agreement came from stuck models and that "more restarts per cell would likely raise" it.
  **[Documented: phase-3.md open issue 1.]** The review called that unsupported and found what
  predicts getting stuck: the starting sign of the two weights. On seed 0, 535 of 545
  opposite-sign starts reached superposition against 266 of 515 same-sign starts. It became
  INFO `initial_sign`. **[Documented: REVIEW.md R1-01, R1-02; D-605.]** Why same-sign starts
  stick more often than p. 15 suggests is still open. **[Documented: REVIEW.md "Open items"
  3.]**

**Checklist** **[Recommended]**

- [ ] Per-restart outcomes are logged.
- [ ] Selection uses the training objective, on enough samples (section 3).
- [ ] If restarts are aggregated, a diagnostic explains the stuck ones.
- [ ] Any claim about "more restarts" is backed by a run.

## 3. Sampling noise in restart selection

**Symptoms.** You pick the best restart by a loss estimated from samples. A re-evaluation on
more samples picks a different one, or your best loss sits *below* a provable optimum.

**Decision procedure** **[Recommended]**

1. Measure the noise of one loss estimate: evaluate the same model many times on independent
   samples and take the spread. **[Recommended]**
2. Compare that noise with the spread between restarts. If they are similar, the choice is
   partly a coin flip. **[Recommended]**
3. Re-evaluate the candidates on a larger common sample, or use an exact loss if you have one.
   **[Recommended]**
4. When you compare two states, score both on the *same* samples (common random numbers). Two
   estimates from different samples differ by their noise as well as by their states.
   **[Recommended]**
5. Remember the winner's curse: the minimum of several noisy estimates is biased low. Report
   the chosen model's loss on fresh samples. **[Recommended]**

**What happened here.**

- E00's best-of-8 linear losses sit 0.13–1.0% below the exact linear optimum, which no model
  can beat in expectation. The exact loss of the same models is within 0.01% of it. The gap is
  noise plus the winner's curse. **[Documented: REPRODUCTION.md § E00; REVIEW.md R1-11.]**
- E04 moved best-restart selection to 200,000 evaluation samples, and E11 scores every instance
  on the same 200,000 samples. **[Documented: phase-3.md "Experiments"; D-307; D-505.]**
- Three experiments still select on noisy losses: E01 (another restart wins in 7 of 14
  conditions on more samples; counts move by about ±2), E03 (close to a coin flip at some
  sparsities) and E08 (noise about as large as the spread between restarts). All three are
  open, because fixing them needs re-runs. **[Documented: REVIEW.md R1-06, R2-10, R3-06;
  "Open items" 4.]**
- E11's held-out comparison first called two seeds' objectives "a tie to 0.03%". Each was
  estimated on its own samples with about 0.4% standard error. On common samples the reviewer
  found seed 0's state lower by about 0.25%. **[Documented: REVIEW.md R4-06; REPRODUCTION.md §
  E11 "Held-out check".]**

**Checklist** **[Recommended]**

- [ ] The relative noise of one loss estimate is known.
- [ ] It is small against the spread between restarts, or selection uses more samples.
- [ ] Comparisons between states use common samples.
- [ ] Reported losses come from fresh samples, not the selection samples.

## 4. A sanity value that disagrees

**Symptoms.** A reference number (from another implementation, a previous run, or a figure)
does not match yours. No acceptance criterion depends on it, but leaving it unexplained would
be dishonest.

**Decision procedure** **[Recommended]**

1. Keep reference values out of the acceptance criteria. They are sanity checks.
   **[Recommended]**
2. Check that the two runs learned the same thing (geometry, norms, biases) before comparing
   losses. **[Recommended]**
3. Check convergence and other local optima: train longer from your solution, and from each
   plausible alternative arrangement. **[Recommended]**
4. Measure the noise of the estimator the reference used (section 3). **[Recommended]**
5. Test the reference against something provable. If it beats a closed-form optimum, it is
   biased. **[Recommended]**
6. Put each diagnostic in the run as an INFO row, so the evidence has a run ID.
   **[Recommended]**
7. Report the explanation as a hypothesis, with how likely the reference's value is under it.
   **[Recommended]**

**What happened here: E00's 0.0196.**

- SPEC's reference loss for the S = 0.9 pentagon was 0.0196; the best pentagon here scored
  about 0.0214, roughly 9% higher. D-106 had already made the reference losses INFO rows, never
  criteria. **[Documented: SPEC.md §7 E00; D-106; D-107.]**
- D-107 added three report-only diagnostics to every E00 run: the best restart on 1,000,000
  samples (0.02139), the spread of 200 independent 20,000-sample estimates (3.68% relative
  noise; 1 of 200 at or below 0.0196), and all 12 cyclic arrangements trained from an ideal
  pentagon (0.02142–0.02165). **[Documented: D-107; REPRODUCTION.md § E00 AUTO block.]**
- The reference's *linear* losses all sat below the exact linear optimum, which points to a
  noisy, selection-biased estimate. **[Documented: REPRODUCTION.md § E00 item 4; phase-1.md
  open issue 1.]**
- The review corrected two overstatements: the 12 arrangements span about 1%, so they cannot be
  ranked, and the explanation had to be retitled as a hypothesis with its probability.
  **[Documented: REVIEW.md R1-04, R1-05.]**

**A second case: a figure read at low resolution.** D-113 claimed that E01 represented fewer
features at 1 − S = 0.001 than pp. 11–12 show. A 300-dpi reading showed no shortfall, and D-114
withdrew the premise in a new entry. **[Documented: D-113, D-114; phase-1.md open issue 2.]**
The lesson: before calling a discrepancy, read the figure at a resolution that can show it.
**[Recommended]**

**Checklist** **[Recommended]**

- [ ] The reference is an INFO row, not a criterion.
- [ ] Geometry agrees before losses are compared.
- [ ] Your solution is converged, and alternatives were tried.
- [ ] Both estimators' noise is known.
- [ ] The reference passes a provable bound.
- [ ] The write-up says "hypothesis".

## 5. Forking paths and held-out confirmation

*Forking paths* are the choices you make after seeing data: which grid, which regime, which
threshold. Each one raises the chance that a pass is luck. A *held-out check* re-runs the
unchanged design on data it has never seen (here, a fresh seed: new initializations, batches
and evaluation samples).

**Symptoms.** A criterion passes at one point of a grid that you refined, or in a regime you
chose from a pilot, or on the only seed you ran.

**Decision procedure** **[Recommended]**

1. List every choice made after seeing outcomes, and disclose each one where the decision is
   recorded. **[Recommended]**
2. Before the confirming run, write and commit the reading rule: what counts as confirmed, and
   what you will do if it is not. **[Recommended]**
3. Run the unchanged profile on a fresh seed. Do not change the grid for it. **[Recommended]**
4. Compare states on common samples (section 3). **[Recommended]**
5. Report the result either way. Do not edit the cited grade. If the status is now a judgment
   call, say so and leave it to whoever owns the decision. **[Recommended]**
6. Stop at the pre-declared number of held-out runs. Running seeds until one agrees is another
   forking path. **[Recommended]**

**What happened here.**

- **E11.** The criteria were fixed before any run. A first development run gave PARTIAL 5/8;
  restarts were doubled and evaluation moved to shared samples. A second development run
  failed the pentagon's "falls to none". Its checkpoints suggested a window near λ ≈ 0.61–0.67
  that the grid skipped, so 0.55, 0.65 and 0.7 were added (and 0.15 and 1 dropped). The cited
  run passed 8/8, with the pentagon free of superposition only at λ = 0.65, where one feature
  was left. **[Documented: D-505 "History (disclosed)"; phase-5.md open issue 1.]** Both
  development runs went into a scratchpad results folder. **[Documented: S 57d571bd @
  20:22–21:07Z; D-505.]**
- The owner made the pentagon PARTIAL and asked for a held-out seed (D-R05). Seed 1 put the
  pentagon's superposition-free point at λ = 0.6 instead, and the whole run was PARTIAL 5/8.
  The review then showed that seed 1's state at 0.6 was a restart that missed a lower state.
  The held-out seed no longer "confirms the window", which rests on the cited seed alone.
  **[Documented: D-R05; run `20260930T184744Z-e8c803d0`; REVIEW.md R4-06; REPRODUCTION.md §
  E11 "Discrepancies".]**
- **E02.** The n = 3 agreement passed at 0.819 against 0.80 on one seed. Before running seeds
  1 and 2, D-605 fixed the rule: if the three-seed mean is below 0.80, say so and leave the
  status to the owner. They gave 0.764 and 0.792 (mean 0.792). The cited grade was not edited,
  and the status question is still open. **[Documented: D-605; runs
  `20261001T065043Z-133b16d8`, `20261001T065904Z-afab247b`; phase-6.md open issue 1.]**
- **E09.** The graded densities were chosen from a pilot on the graded quantity. Grading
  1 − S = 0.1 as well would have failed (0.86). The PASS is now scoped to its regime.
  **[Documented: D-502; REVIEW.md R4-10.]**
- Earlier criteria were written after exploratory numbers had been seen (E02, E03) or after a
  prototype (E06). Each entry discloses it. **[Documented: D-305, D-306, D-406.]**

**Checklist** **[Recommended]**

- [ ] Every post-hoc choice is disclosed in its decision entry.
- [ ] The reading rule for the held-out run is committed before the run.
- [ ] The held-out run uses the unchanged profile and a fresh seed.
- [ ] The result is reported whatever it is, and the cited grade is not edited.

## 6. Selection by ground truth

**Symptoms.** Your pipeline picks one model or dictionary out of a sweep, and the picking step
uses the true answer (the true features, the true number of features). A practitioner on a real
model could not do that.

**Decision procedure** **[Recommended]**

1. For each selection step, ask: could someone without the ground truth make this choice?
   **[Recommended]**
2. If not, report the selected result as an *upper bound* (an oracle), as INFO.
   **[Recommended]**
3. Write a ground-truth-free rule, commit it before the grading run, and disclose what results
   were known when you wrote it. **[Recommended]**
4. Look for hidden leaks: a sweep designed around the true size leaks the truth even if the
   rule does not. **[Recommended]**
5. Check how sensitive the rule is to its threshold, report it, and do not move the threshold
   afterwards. **[Recommended]**

**What happened here: E10.**

- Phase 5 graded the SAE with the best recovery of the true features, and E10 passed 3/3. The
  decision entry and the phase report both flagged that a practitioner could not choose that
  way. **[Documented: D-504 "Grading"; phase-5.md open issue 2.]**
- The owner required a rule that does not use ground truth (D-R04). The rule: the sparsest SAE
  (lowest held-out L0) among those with held-out FVU ≤ 0.1, ties broken by FVU. It was
  committed before the grading run, and the entry discloses that the previous run's summary
  was known. **[Documented: D-R04.]**
- Result: 3/5 pentagon features recovered (FAIL), 11/11 at 1 − S = 0.01 (PASS), 10/13 at
  0.001 (FAIL). E10 went from PASS 3/3 to PARTIAL 1/3; the oracle still passes all three as an
  upper bound. **[Documented: rectification.md item 4; run `20260930T192302Z-13117b27`.]**
- At 1 − S = 0.001 the SAEs that recover everything have FVU 0.117–0.121, just outside the bar.
  The bar was not moved. **[Documented: rectification.md open issue 2.]**
- The review found a leak in step 4's sense: the sweep's dictionary sizes were built from the
  true n. The reviewer's re-run without the n-sized dictionaries changed no outcome. It also
  noted that the grading run reproduced the inspected run bit for bit, and that no bar at 0.05,
  0.1 or 0.12 changes the count. **[Documented: REVIEW.md R4-04, R4-11; REPRODUCTION.md § E10.]**

**Checklist** **[Recommended]**

- [ ] Every selection step passes the "no ground truth" question, or is labeled an oracle.
- [ ] The rule was committed before the run, with a disclosure of what was known.
- [ ] The sweep itself does not encode the answer.
- [ ] Threshold sensitivity is reported; the threshold is unchanged.

## 7. Trajectory variance versus failure

**Symptoms.** The final state matches the paper, but the path to it does not look like the
paper's figure.

**Decision procedure** **[Recommended]**

1. Read what the paper claims: a final state, a typical path, or one example of a path.
   **[Recommended]**
2. Grade the claim that the paper makes. Report the path as a distribution over runs.
   **[Recommended]**
3. Count how often the paper's path appears in your runs. **[Recommended]**
4. If you call your path a variant, find the paper's own reason for it and check that the
   reason applies to your setup. **[Recommended]**
5. Call it a documented variant only if the graded claim holds. Otherwise it is a failure.
   **[Recommended]**

**What happened here: E05 stage B.**

- All 48 runs ended as the octahedron (graded, PASS). During stage B the two feature sets
  collapsed onto nearly orthogonal directions, not onto opposite ends of one axis as in the
  paper's panel. **[Documented: phase-4.md open issue 3; REPRODUCTION.md § E05.]**
- The owner decided to treat it as a documented variant unless the octahedron criterion failed,
  and to report fractions and trajectory types (D-R02). INFO `stage_b_placement` gives a median
  centroid cosine of −0.058, −0.044 and −0.037 at the three densities, with only 6%, 13% and
  18% of collapsed snapshots opposed. **[Documented: D-R02; REPRODUCTION.md § E05 "Documented
  variant".]**
- The first justification cited p. 23's sentence about *correlated* features, but the two sets
  are independent. The review corrected it to the PCA-like solution of p. 28, and stated how
  rare the paper's panel is in these runs and that only the end state is graded.
  **[Documented: REVIEW.md R2-03.]**

**Checklist** **[Recommended]**

- [ ] The graded claim is the one the paper makes.
- [ ] Path frequencies are reported over all runs.
- [ ] The justification's premise holds for your setup.

## 8. Provenance gaps

*Provenance* means every number traces to a logged run with its config, seed and git commit
(SPEC §2). Here a run is logged when the CLI writes `results/<experiment>/<run_id>/` with a
`manifest.json`.

**Symptoms.** A number in the docs or a decision comes from a script, a notebook or a run that
is not in `results/`, or from a run whose manifest says the tree was clean when it was not.

**Decision procedure** **[Recommended]**

1. For every number you are about to write, find its run ID. If there is none, label it
   "uncited exploratory" or re-run it through the logged path. **[Recommended]**
2. If a decision rests on uncited runs, re-base it on logged runs before citing it.
   **[Recommended]**
3. Turn a useful pilot into a named profile, so it can be re-run and cited. **[Recommended]**
4. Make sure the dirty-tree check covers untracked code, and launch cited runs from a clean
   copy of the code (chapter 08). **[Recommended]**
5. Never cite something that lives in a temporary folder. **[Recommended]**

**What happened here.**

- **Scratchpad runs.** The Phase 3 draft explored E02 and E03 with scratchpad scripts; Phase 4
  prototyped E05, E06 and E08; Phase 5 ran E09–E11 development runs into a scratchpad results
  folder. Each decision that used them discloses it. What they produced beyond the numbers
  quoted in DECISIONS is unknown. **[Documented: S 2df87f77 @ 13:35–14:02Z; S 92884087 @
  16:47–17:04Z; S 57d571bd @ 20:22–21:07Z; D-305, D-306, D-309, D-406, D-409, D-502, D-505.]**
- **Re-basing.** The batch-256 choice rested on scratchpad runs until rectification 1 ran six
  CLI runs from one clean commit and based D-309 on them. **[Documented: D-309; phase-3.md
  "The batch-256 decision".]**
- **The E09 pilot.** D-502 chose E09's graded densities from an uncited pilot. D-R03 made it a
  `pilot` profile, re-ran it through the CLI and cited it as `mech_interp_abs:pilot` (D-R07).
  It is not bit-identical to the original, because the module orders densities differently.
  **[Documented: D-502, D-R03, D-R07; run `20261001T072229Z-1e8bafbd`.]**
- **A vanished run.** A mechanism claim in E11 cited a development run that was no longer on
  disk when the reviewer looked. It now cites the cited run's own restarts. **[Documented:
  REVIEW.md R4-07.]**
- **Untracked code.** `runs.git_info` ignored untracked files, so a run that imported an
  uncommitted module was recorded as clean. D-606 fixed it in `450938e`; no cited run was
  affected. **[Documented: REVIEW.md R1-10; D-606.]**
- **Dirty runs moved aside.** In Phase 1 the session moved superseded dirty-tree runs out of
  `results/` into its scratchpad before launching the cited runs. They are not on disk now.
  **[Documented: S 35cab3f0 @ 09:05Z; their absence from `results/`.]**

**Checklist** **[Recommended]**

- [ ] Every number has a run ID, or a label saying it has none.
- [ ] Decisions are re-based on logged runs before citing.
- [ ] Pilots are profiles.
- [ ] `git status` is clean, including untracked files under `src/` and `configs/`.

## 9. When and how to revise a criterion honestly

**Symptoms.** After seeing results you believe a criterion is wrong: it tests something the
paper does not claim, it counts the same thing twice, or it is too lenient.

**Decision procedure** **[Recommended]**

1. Find where the criterion came from: the paper's text, or a proposal in your own spec.
   **[Recommended]**
2. State the reason for the change in terms that do not depend on which way the result goes: a
   flaw (double counting) or a mismatch with the paper's words. **[Recommended]**
3. Write the new criterion, its rationale, the page and the alternatives in a decision entry.
   Disclose what results you knew. Commit it before the run that grades it. **[Recommended]**
4. Keep the old criterion's value visible as an INFO row. **[Recommended]**
5. Prefer changes that tighten. A change that loosens needs the paper's text to require it.
   **[Recommended]**
6. If the only reason to change is the outcome, do not change it. Record the question as open.
   **[Recommended]**
7. Re-run from a clean commit and cite the new run. **[Recommended]**

**What happened here.**

- **E08.** SPEC's "more than m/2 features computed well" came from the spec, not the paper.
  The owner had it re-derived from p. 37 and footnotes 16–17: more than m = 40 features
  usefully computed (below 75% of the mean-prediction loss). It was fixed before the run, with
  the previous run's numbers known, and the original is INFO. The new criterion also failed
  (23 against more than 40); the original gave 3. **[Documented: D-R01; rectification.md item
  1.]** The planning conversation had recommended this re-derivation and keeping the original
  FAIL visible. **[Planning conversation, per owner]**
- **E03.** The review found that the cluster rule counted one band of Dᵢ values as two
  clusters, because the windows around 2/5 and 3/8 overlap. D-603 tightened the rule to count a
  feature only inside the matching polytope. The entry says the expected result (PARTIAL) was
  known, and that the rule was committed before the grading run. E03 went from PASS 4/4 to
  PARTIAL 3/4. **[Documented: REVIEW.md R2-01; D-603.]**
- **A change not made.** At an edge threshold of 0.2 instead of E03's 0.05, pentagons and an
  antiprism separate. The session did not switch, because choosing the threshold after seeing
  the result would be tuning; the owner has not decided. **[Documented: phase-6.md open issue 2;
  REVIEW.md "Open items" 2.]**
- **Other revisions, all tightening:** E11's three-way "falls to none" (D-R05) and E09's
  role-consistent motifs (D-604). **[Documented: D-R05, D-604.]**

**Checklist** **[Recommended]**

- [ ] The criterion's source (paper or spec) is identified.
- [ ] The reason would hold whatever the outcome.
- [ ] The decision entry is committed before the grading run, with a disclosure.
- [ ] The old value is still reported.
- [ ] Nothing was loosened without the paper's text.

## 10. A test that fails, or a test that should

**Symptoms.** A unit test fails on a seed, or a margin looks arbitrary, or a test passes while
the thing it is meant to check is broken.

**Decision procedure** **[Recommended]**

1. Re-run the failing check over many seeds and look at the distribution. A calibrated
   statistic should look like its reference distribution. **[Recommended]**
2. If it is calibrated, the failure was a fixed-seed false positive. Fix the test's design and
   add a test of the stronger property. Write down why. **[Recommended]**
3. If you change a margin, give a principled reason. "Observed value" alone is not one.
   **[Recommended]**
4. Never change implementation code to satisfy a test (SPEC §2). **[Recommended]**
5. Make sure tests fail when they should: feed them a broken case. **[Recommended]**

**What happened here.**

- A per-instance density test failed at seed 2 by 3.45σ. A 3,000-draw study gave mean z −0.03
  and standard deviation 1.001, so the sampler was unbiased. The test moved to seed 12 with a
  comment, and a 64-seed calibration test was added. **[Documented: phase-1.md "Test status";
  `tests/unit/test_data.py` `test_density_z_scores_are_calibrated`.]**
- An L1 test's margin was relaxed from 0.5 to 0.8, with a code comment giving the observed
  ratio (≈ 0.62). **[Documented: `tests/unit/test_train.py` line 99; S 35cab3f0 @ 08:33Z.]** The
  Phase 1 report does not mention it. **[Documented: phase-1.md mentions only the density
  test.]**
- Notebooks in the paper tier reused figure-less runs left by replication tests in the same
  temporary folder, and failed. The fix made them reuse only runs with figures (`5d9cd7e`),
  checked with a figure-less copy next to each cited run. **[Documented: phase-3.md "A tier bug
  found and fixed".]**
- The app launch test stopped the CLI but not its Streamlit child, leaving servers running for
  hours. A before-and-after check confirmed the leak and the fix (`8a02181`). **[Documented:
  phase-3.md "A test bug fixed".]**
- E08's replication test pins "≥ 39 monosemantic", the current failing outcome, and says so.
  The review listed it as open. **[Documented: REVIEW.md R3-14.]**

**Checklist** **[Recommended]**

- [ ] A failing statistical test was checked across many seeds.
- [ ] Every changed margin has a written reason.
- [ ] No implementation code changed to make a test pass.
- [ ] Each criterion's tests include a case that must fail.
