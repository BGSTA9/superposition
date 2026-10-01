# Independent review

Findings of the Phase 6 independent review (SPEC §11) and what was done about each.

**How the review was run (D-607).** Five reviewers with fresh context, each a separate agent
that had seen none of this repository's history:
- each re-derived its experiments from the cited paper pages before reading the code;
- each then audited the implementation;
- each checked every number in docs/REPRODUCTION.md against the cited runs, looking for bugs
  and overclaims.

They worked read-only on the repository and did not consult the reference implementations.
Every finding came back numbered, with evidence.

| Reviewer | Scope | Critical | Major | Minor | Notes |
|---|---|---|---|---|---|
| R1 | engine (data, models, train, metrics, theory, runs, report); E00, E01, E02 | 0 | 2 | 5 | 4 |
| R2 | E03, E04, E05 | 0 | 2 | 5 | 4 |
| R3 | E06, E07, E08 | 0 | 3 | 8 | 4 |
| R4 | E09, E10, E11 | 0 | 1 | 8 | 2 |
| R5 | E12 (after it was written) | 0 | 4 | 3 | 4 |

**How to read the resolutions.**
- **Fixed** means the code changed, the experiment was re-run from a clean commit, and the new
  run is cited.
- **Corrected** means a statement in the docs was wrong or too strong and has been rewritten.
- **Open** means the item is listed here and in the phase report, but not resolved.

Where a resolution quotes a number that only the reviewer's own analysis produced (their
scripts are not part of the repository), it says so. Numbers the docs rely on were
re-computed by code in this repository, in a logged run, before being cited.

## What the review changed

- **E03: PASS 4/4 → PARTIAL 3/4.** The criterion that Dᵢ clusters at three characteristic
  fractions counted a feature by its Dᵢ alone. 2/5 and 3/8 are too close for that, so one band
  of values counted as two "clusters". Counted only in their polytopes (D-603), the clusters
  are backed at 1/2 alone.
- **E02: the n = 3 agreement is not robust across seeds.** The cited seed gives 0.819 against
  the 0.80 bar. Held-out seeds give 0.764 and 0.792, so the three-seed mean is 0.792. Under the
  rule fixed before those runs (D-605), this is reported, and the status is left to the owner
  (R1-01, open).
- **E09: still PASS 3/3, on tightened criteria (D-604).**
  - Motifs now count only when their neurons carry the motif's roles: 68 models instead of 134.
  - "Named" now means nothing unexplained is left in the neuron.
  - A baseline shows that the role rules name about half the neurons of an *untrained* model.
- **E12: the extension's write-up separates the loss landscape from optimization.** It adds an
  unrestricted optimum per cell (D-608). It does not claim that some activations "induce more
  superposition" where the difference comes from training.
- **Overclaims corrected** in E00, E01, E02 and E04–E11, the README, two notebooks and three
  earlier DECISIONS entries.
- **Provenance.** A run now counts as dirty if an untracked file under `src/` or `configs/`
  could have changed what it ran (D-606).

## R1 · engine, E00 intro_pentagon, E01 basic_results, E02 phase_change

R1 found the engine and the theory correct. Its checks included:
- the loss reduction order and the per-instance initialization;
- the closed-form integrals against scipy;
- the family optimizer against Nelder–Mead in all 144 cells;
- every number in E00–E02 against the cited runs.

| ID | Severity | Finding | Resolution |
|---|---|---|---|
| R1-01 | major | E02's n = 3 agreement (0.819 against 0.80) rests on one seed. Resampling each cell's models puts its mean near 0.80. The batch-1024 run shares seed 0's starting weights, so it is not independent. "More restarts would raise it" is unsupported. | **Fixed, and open.** Seeds 1 and 2 were run and cited (`phase_change:seed1`, `:seed2`) under a rule fixed beforehand (D-605). n = 3 gives 0.819, 0.764 and 0.792 (mean 0.792); n = 2 holds on every seed (mean 0.870). The docs now say the n = 3 criterion does not hold across seeds, and the false sentences are gone. **Open:** whether E02's status should change is the owner's call. |
| R1-02 | major | E02's n = 2 empirical diagram does not look like p. 15. Many models in theory-superposition cells get stuck, the starting sign of the two weights decides it, and the docs credited McGrath and the paper's discard rule instead. | **Fixed.** New INFO `initial_sign` (D-605). In those cells, 535/545 opposite-sign starts reach superposition against 266/515 same-sign starts on seed 0 (similar on seeds 1 and 2). The docs record the discrepancy with p. 15, drop the wrong attribution, and give hypotheses. **Open:** the cause, untested, may be the paper's initialization scale or optimizer, neither of which it states. |
| R1-03 | minor | Trained n = 2 models beat the restricted families in many more cells than the 4 the docs named. | **Fixed.** New INFO `exact_loss_vs_theory`: more than 0.1% better in 59 of 144 cells, more than 1% in 24, at most 6.5%. The docs now say "exact loss of each restricted family". E12's unrestricted optimum (D-608) shows the restriction moves 2 of 64 ReLU labels on its grid. |
| R1-04 | minor | E00's arrangement diagnostic cannot rank the 12 arrangements: they span about 1%, twice the evaluation noise, and the quick and full runs order them differently. | **Corrected.** The docs now say the arrangements cannot be told apart. The reviewer's convergence check (the pentagon stays at 0.0214) is quoted as the reviewer's. |
| R1-05 | minor | E00's heading "what it is" presents the 0.0196 explanation as fact. | **Corrected.** Retitled as a hypothesis, with its probability: a draw that low happens a few percent of the time. |
| R1-06 | minor | E01's best restart at sparse settings is chosen by 20,000-sample noise. Re-evaluated on more samples, another restart wins in 7 of 14 conditions, and counts shift by about ±2. All criteria still pass. | **Corrected** (the docs give the ±2). **Open:** selecting restarts on more samples would need an E01 re-run. |
| R1-07 | minor | Several E02 numbers come from notebook 04 or uncited runs, not from a run's record. | **Partly fixed.** The headline comparisons are now INFO rows (`initial_sign`, `exact_loss_vs_theory`). The notebook-04 numbers that remain are labeled as such; the reviewer reproduced them. |
| R1-08 | note | The majority vote is SPEC's choice; p. 14 averages the kept models. | **Corrected.** E02's notes say so, and point out that the first panel of each figure is the averaged view. |
| R1-09 | note | README said the status table comes from "the latest runs". | **Corrected:** it comes from the cited runs. |
| R1-10 | note | `runs.git_info` ignored untracked files, so a run that imported an uncommitted module was recorded as clean. | **Fixed** (D-606, with a unit test). No cited run was affected. |
| R1-11 | note | E00's linear check graded a sampled loss. The same model's exact loss is within 0.01% of the optimum. | **Fixed.** `exact_loss` is now reported next to the graded value. The grading is unchanged. |

## R2 · E03 uniform_geometry, E04 nonuniform_geometry, E05 learning_dynamics

R2 found E04 and E05(a) sound. Its checks included:
- every number of E03–E05 against the cited runs;
- the octahedron detection on all 48 final states;
- the jump detector, re-derived independently.

| ID | Severity | Finding | Resolution |
|---|---|---|---|
| R2-01 | major | E03's cluster criterion counts a feature by Dᵢ alone. One band near 0.39 counts as both a pentagon and a square-antiprism cluster, and the tail of a band below 2/3 counts as triangles. Only 1/2 and 2/5 survive a geometry check. | **Fixed** (D-603, with a negative unit test). The graded count requires the matching polytope, and the old count is INFO. **E03 is now PARTIAL 3/4**: at E03's edge threshold of 0.05 only the digons qualify. **Open:** whether factors should be read at 0.2, where pentagons and an antiprism separate, is the owner's call. Choosing that after seeing the result would be tuning. |
| R2-02 | major | README's E03 caption claimed triangles, pentagons and square antiprisms. | **Corrected.** |
| R2-03 | minor | E05(b)'s stage B was justified with a p. 23 sentence about correlated features, but the two sets are independent. The paper's panel B is rare (6 of 48 runs), the published B panel shows the variant, and only the end state is graded. | **Corrected.** The right justification is that independent directions have room to be orthogonal, the PCA-like solution of p. 28. The rarity and the graded scope are now stated. |
| R2-04 | minor (favorable) | E04(a)'s "snap comes later" compared with the caption's 0.4×; the paper's own loss figure puts the boundary at about 0.27×, between our grid points. | **Corrected:** no discrepancy with the figure. The per-feature loss match is quoted as the reviewer's reading. |
| R2-05 | minor | E05's "differently normalized loss" cannot explain a relative drop. | **Corrected:** the paper's final loss per feature differs (about 0.053 against 0.074), and the hypothesis is now a different density or n. |
| R2-06 | minor | The factor census at 1/(1 − S) = 4.83 has three pieces, not two. | **Corrected.** |
| R2-07 | minor | E05(a) organizes by step 10,000, not 5,000, and every restart overshoots (0.58–0.66), not one. | **Corrected.** |
| R2-08 | note | The plateau is not wider than the paper's when the same rule is applied to the paper's curve. | **Corrected.** |
| R2-09 | note | E04(b)'s anticorrelation test is lenient (the data exceed it), and the two setups' densities differ. | **Corrected:** stated in E04's notes. |
| R2-10 | note | E03's choice between two restarts is close to a coin flip at some sparsities, though no outcome changes. | **Corrected** (stated). **Open:** more evaluation samples, as E04 uses. |
| R2-11 | note | D-404 gave cross-set cosines as −1/3 or −2/3; they are +1/3 and −2/3. | **Corrected** in place. |

## R3 · E06 adversarial, E07 privileged_basis, E08 abs_computation

R3 found the PARTIAL statuses of E06 and E08 honest and E07's PASS sound. Its checks included:
- the attack budget and the reference damage;
- the robustness of E07's polysemanticity counts to thresholds and re-selection;
- E08's per-feature losses and both reference losses.

| ID | Severity | Finding | Resolution |
|---|---|---|---|
| R3-01 | major | "The neuron picture of p. 37 reproduces" overclaims. Monosemantic neurons do not sit on the most important features (p. 38), and the dense model spends many neurons on few features. | **Corrected** in REPRODUCTION.md and notebook 08: the trend reproduces, those two details do not. |
| R3-02 | major | "Why: dead sides at initialization" is contradicted by the diagnostic's own numbers: 111 sides start alive at 1 − S = 0.001, but only 23 end up computed. | **Corrected.** Dead sides are now "a contributing factor", with the untested alternatives listed. Notebook 08's exercise solution is fixed. |
| R3-03 | major | E06's n = 20, m = 5 failure was called "what the claim predicts". Vulnerability actually falls while 1/D* rises, and the README stated the claim without qualification. | **Corrected.** The docs now report it as a genuine case where vulnerability does not track superposition, with the importance explanation marked partial; the README is qualified. |
| R3-04 | minor | E07's sidestepping row labeled "0.1" compares with the 0.12 model, two sparsities. | **Corrected:** labeled by the sparsity actually used, and the cross-sparsity loss comparison is withdrawn. |
| R3-05 | minor | D-408 claimed the precise estimator keeps "computed well" counts from crossing the bar by noise. Re-estimates move them by up to 4. | **Corrected** in D-408 and in E08's text. The graded count at 0.75 is stable. |
| R3-06 | minor | E08 picks its best restart on a 20,000-sample loss whose noise matches the spread between restarts, and saves only that restart's weights. | **Open** (stated in E08's notes). |
| R3-07 | minor | E08's diagnostic variants start from other random draws than the default runs, so the comparison is unpaired. | **Corrected** (stated). **Open:** a paired re-run. |
| R3-08 | minor | "More than m/2 usefully computed" was framed as beyond capacity without superposition. A model without superposition can usefully compute up to m features, one side each. | **Corrected.** |
| R3-09 | minor | "Regimes match p. 33 closely": at 1 − S = 0.03 the best model represents 9 features, not 10. | **Corrected:** listed as a discrepancy. |
| R3-10 | minor | The side diagnostic overcounts when dense, because a side predicted by its mean passes the bar. | **Corrected** (stated). **Open:** suppressing those rows in code. |
| R3-11 | minor | Polysemantic counts at 1 − S = 0.001 include weights that barely moved from initialization. | **Corrected** (stated). |
| R3-12 | note | The Spearman criterion cannot show tracking, since both quantities rise with sparsity. | **Corrected:** stated, with the shape agreement that does hold. |
| R3-13 | note | E06's 8.7× depends on choosing the worst attack per input; one attack per model gives about 3.7×. | **Corrected** (stated, as the reviewer's number). |
| R3-14 | note | E08's replication test pins "≥ 39 monosemantic", the current failing outcome (disclosed in the test), and no test pins the one-sided loss ratio. | **Open.** |
| R3-15 | note | Untraceable bias medians, the README's "latest runs", and an overstated "no side starts dead". | **Corrected.** |

## R4 · E09 mech_interp_abs, E10 sae_recovery, E11 l1_removes_superposition

R4 found the E10 and E11 PARTIAL statuses honest and their PASSing criteria not lenient. Its
checks included:
- the footnote-15 invariance of path gains;
- the rebuild against three ablated variants;
- the SAE recovery metric on all 67 SAEs;
- every number of E09–E11.

| ID | Severity | Finding | Resolution |
|---|---|---|---|
| R4-01 | major | The motif detector checks a weight pattern only. It found "motifs" on unexplained neurons, inflating the counts (134 models instead of 66–68), and README's "most restarts" was wrong. | **Fixed** (D-604, with a unit test). The graded motif and every statistic now use role-consistent motifs: 68 of 240 models, 8–12 of 20 per graded configuration, 1–4 at 0.1. The weight-pattern counts are reported next to them. |
| R4-02 | minor | The inhibitor role names almost any neuron: on random weights about half the neurons get a role. | **Fixed** (D-604). "Named" now means fully named, still 48/48 in the graded models, and INFO `named_at_initialization` gives the baseline: 49–54% named, 30–33% fully named. |
| R4-03 | minor | "Raise exactly the losses the role predicts": in 5 of 96 cases a feature the role does not name rises by 5–8%. | **Corrected.** |
| R4-04 | minor | The ground-truth-free SAE rule runs on a sweep built from the true n. | **Corrected** (stated). The reviewer's re-run without the n-sized dictionaries changes no outcome. |
| R4-05 | minor | "TopK L0 = k by construction" and "2–56 latents resampled" were wrong. | **Corrected.** |
| R4-06 | minor | E11's held-out "tie to 0.03%" compared estimates from different samples. On common samples seed 0's state is lower by about 0.25%, so seed 1's window is a restart miss. | **Corrected.** The held-out seed no longer "confirms the window". |
| R4-07 | minor | A mechanism claim cited an uncited development run that is no longer on disk. | **Corrected:** it now cites the cited run's own restarts, read by the reviewer. |
| R4-08 | minor | The λ = 0.003 step does not test footnote 21: there is no λ between 0 and 0.003, and the step is within the spread between restarts. | **Corrected.** |
| R4-09 | minor | E11's n = 20 PASS is removal by killing features, not by rearranging them. | **Corrected** (stated). |
| R4-10 | note | E09's graded regime was chosen on the graded quantity. Grading 1 − S = 0.1 too would fail. | **Corrected:** the PASS is scoped to the regime, with the 0.86 that 0.1 would give. |
| R4-11 | note | E11's paper test pinned a seed-dependent criterion; D-R04's "fixed before the run" did not say the same data had been seen. | **Fixed** (the test no longer pins it) and **corrected** (stated). |

## R5 · E12 activation_functions

R5 reviewed E12 after its code was written, on its first quick run. It found the pipeline
numerically sound. Its checks included:
- the quadrature against scipy at 38 points;
- the exact model losses against 400,000-sample Monte Carlo;
- that the theory and the model use the same σ;
- that nothing leaks between activations;
- every row against the checkpoints.

Its findings were about what the rows can be read to mean.

| ID | Severity | Finding | Resolution |
|---|---|---|---|
| R5-01 | major | The empirical differences between activations come mostly from optimization, not from the optimum. Softplus and sigmoid's extra superposition cells are the ReLU's stuck models. GELU and SiLU rarely reach the antipodal pair from same-sign starts. | **Fixed.** The cited run reports `initial_sign` per activation, and the write-up attributes every difference to the landscape or to training. It does not claim that an activation "induces more superposition" where the theory does not differ. |
| R5-02 | major | Softplus and sigmoid have not converged at the sparse densities after 10,000 steps (up to 2.3× their optimum), which inflates trained-loss ratios against the ReLU. | **Corrected.** The write-up reports convergence and compares activations only by their theoretical optima. **Open:** a longer-training variant. |
| R5-03 | major | "Identical theoretical diagrams" for ReLU, softplus and sigmoid is an artifact of the grid and of the restricted families. | **Fixed** (D-608). The unrestricted optimum over all four parameters is computed per cell and reported. The write-up names the grid's resolution as a limit. **Open:** boundary positions at sub-cell resolution. |
| R5-04 | major | The first E12 run predates the `initial_sign` rows, and the config's runtime comment was stale. | **Fixed:** E12 was re-run from a clean commit after every change, and the runtime is recorded. |
| R5-05 | minor | A cell's "best model" was the lowest sampled loss, so the app's ordering and a study-guide task could land on the wrong model. | **Fixed** (D-608). Best models are chosen by exact loss, and the lab shows and sorts by it. |
| R5-06 | minor | Rankings of activations hid ties. | **Fixed:** tied groups. |
| R5-07 | minor | GELU and SiLU's cell-by-cell maps depend on each cell's starting signs, so they will not reproduce across seeds; only the shares might. | **Corrected** (stated). **Open:** more seeds. |
| R5-08 | note | Most of the loss ratio between activations measures how well σ can output x, not superposition. Sigmoid beats the ReLU in 2 cells. | **Corrected** (stated). |
| R5-09 | note | The ‖W‖ > 0.5 threshold is not activation-invariant, but the trained norms are bimodal, so no label depends on it. | Noted. No change. |
| R5-10 | note | `lab.phase_cell_shared_losses` assumed the ReLU. | **Fixed:** `PhaseProblem` carries its activation. |
| R5-11 | note | The quadrature check is trivially exact for ReLU and leaky ReLU; the quick profile covers only n = 2, m = 1. | **Corrected** (stated). **Open:** the full profile, with n = 3, was not run. |

## Open items

1. **E02 (R1-01):** the n = 3 agreement fails on two of three seeds. Should E02's status change?
   (The owner's decision.)
2. **E03 (R2-01):** should the cluster criterion read factors at an edge threshold of 0.2
   instead of 0.05? Deciding after seeing the outcome would be tuning. (The owner's decision.)
3. **E02 (R1-02):** why same-sign starts get stuck more often in our runs than the paper's
   figure suggests. The paper's initialization or optimizer are candidates.
4. **Restart selection by noisy losses:** E01 (R1-06), E03 (R2-10) and E08 (R3-06).
5. **E08 (R3-07, R3-10, R3-14):** paired variants, suppressing the dense side-diagnostic rows,
   and the test pins.
6. **E12 (R5-02, R5-03, R5-07, R5-11):** longer training for softplus and sigmoid; boundaries at
   sub-cell resolution; more seeds; the full profile with n = 3.
