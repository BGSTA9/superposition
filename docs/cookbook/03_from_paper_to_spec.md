# 03 · From paper to SPEC: an inch-by-inch recipe

A SPEC is the document you build from: what the paper says (with pages), what you decided where
it is silent, what counts as success, and in what order you will work. In this repository it is
[SPEC.md](../../SPEC.md), version 1.0, 734 lines. It was drafted with Claude in a planning
conversation you cannot see in the repository (COOKBOOK_SPEC §1), committed as `7ac9cef`, and
never edited afterwards (`git log -- SPEC.md`). **[Documented]**

This chapter teaches you to write one yourself. It covers reading, the page map, notation,
under-specification, criteria, verification runs, reference values and closed-form checks, and
ends with one paper section (pp. 9–11) worked end to end into SPEC text.

Labels: **[Documented]** (the evidence shows it), **[Reconstructed]** (inferred; says from
what), **[Recommended]** (advice, not history). Terms: [glossary](../GLOSSARY.md). Decisions:
[DECISIONS.md](../DECISIONS.md). Template: [templates/SPEC_TEMPLATE.md](templates/SPEC_TEMPLATE.md).

## 1. Reading strategy: the text layer plus rendered pages

**Why two layers.** A PDF's text layer (what `pdftotext` extracts) is searchable and quotable,
but it misses everything drawn as an image: figure labels, panel titles, many equations. In
this paper the figures carry claims. When this chapter was written, the text layer of p. 2 did
not contain the word "sparsity" even though the p. 2 figure is labeled 0%, 80% and 90%
sparsity; and on p. 11 the density labels of the seven panels (1 − S = 1.0 … 0.001) are
visible only in the rendered page. **[Documented]** (checked with `pdftotext` and `pdftoppm` on
the local PDF while writing this chapter.)

**What was done here.**
- Planning conversation: full read from the text layer; pp. 2, 13, 14, 17 and 19 rendered and
  read visually. [Planning conversation, per owner]
- Session 35cab3f0 (Phases 0–1): the PDF reader needed poppler, installed at 08:07Z; pp. 41–62
  extracted with `pdftotext`; headings mapped to pages with a per-page loop at 08:19Z; pp. 11
  and 12 rendered at 300 dpi at 09:46–09:47Z to settle D-113/D-114. **[Documented]** [S 35cab3f0]
- Session 2df87f77 (Phase 3 draft): `pdftotext -layout` on pp. 12–16 and the footnote pages
  56–57 (13:25Z); p. 17 and p. 19 rendered at 300 dpi and their charts cropped (13:34–13:35Z),
  before E03's thresholds were written (D-306). **[Documented]** [S 2df87f77]

**By hand.** **[Recommended]** Commands need poppler (`brew install poppler` on macOS [Not verified: installs system software; poppler was already installed on the machine that checked this cookbook]) and the
PDF, which is git-ignored, so they cannot run in a fresh clone. Outputs go to a temporary folder.

1. Get a fixed copy and check it. Action: count pages.
   ```bash
   pdfinfo references/toy_models_of_superposition.pdf
   ```
   Needs the git-ignored paper PDF in `references/`; checked with the owner's copy ([VERIFICATION.md](VERIFICATION.md)).
   Expected: `Pages: 62`. If not 62, your print differs and every page reference will be off;
   re-print as [references/README.md](../../references/README.md) describes.
2. Extract one text file per page, so a search hit tells you the page.
   ```bash
   mkdir -p "${TMPDIR:-/tmp}/paper"
   for p in $(seq 1 62); do pdftotext -layout -f "$p" -l "$p" references/toy_models_of_superposition.pdf "${TMPDIR:-/tmp}/paper/p$(printf %02d "$p").txt"; done
   ```
   Needs the git-ignored paper PDF in `references/`; checked with the owner's copy ([VERIFICATION.md](VERIFICATION.md)).
   Check: 62 files, `p01.txt` … `p62.txt`.
3. Pass 1: skim every section's opening paragraph and every figure caption. Write one line per
   section: its claim and its pages.
4. Pass 2: read the section you are specifying in full, with the rendered page beside the text.
   ```bash
   pdftoppm -r 300 -f 17 -l 17 -png references/toy_models_of_superposition.pdf "${TMPDIR:-/tmp}/paper/p17"
   ```
   Needs the git-ignored paper PDF in `references/`; checked with the owner's copy ([VERIFICATION.md](VERIFICATION.md)).
   Expected: `p17-17.png` (pdftoppm appends the page number). Read numbers off figures only at
   300 dpi; low-resolution reading produced D-113's wrong premise. **[Documented]** (D-114.)
5. Write facts with pages as you read (section 3 below), not afterwards.

**Study aids used before any code** [Planning conversation, per owner]: a seven-stage reading
plan; a hand-built interactive pentagon (five features in two dimensions, with ReLU and a
negative bias, showing interference); "explain it back" questions; and a Johnson–Lindenstrauss
warm-up. The warm-up drew N = 2d random unit vectors and recorded the largest |cos| between
any two:

| d | Largest \|cos\| |
|---|---|
| 2 | 0.98 |
| 10 | 0.79 |
| 100 | 0.41 |
| 1,000 | 0.15 |

Drill **[Recommended]**: reproduce the trend before reading further. Why it matters: in high
dimension, many directions can be almost orthogonal, which is the room superposition uses.

```bash
uv run python -c "import numpy as np; rng = np.random.default_rng(0); [print(d, round(float(np.abs(np.triu((V := (lambda A: A / np.linalg.norm(A, axis=1, keepdims=True))(rng.standard_normal((2 * d, d)))) @ V.T, 1)).max()), 2)) for d in (2, 10, 100, 1000)]"
```

Expected: four lines, largest |cos| falling with d. With seed 0 this printed 0.98, 0.79, 0.41
and 0.15 when this chapter was written; other seeds give nearby values. If it does not fall,
check you normalized the rows and excluded the diagonal.

## 2. Building the page map

**Purpose.** One table from paper section → pages → code → experiment → notebook → app page, so
every later reference can be checked.

**What was done here.** PAPER_MAP.md was written in Phase 0, "fully mapped with page ranges"
(phase-0.md), and extended as experiments landed (e.g. `92525fc`, Phase 3 rows). The planning
conversation built its map by searching each page's text, and checked every page reference in
SPEC.md against per-page text. [Planning conversation, per owner] The build session ran its own
heading-to-page loop over pp. 38–62. **[Documented]** [S 35cab3f0 @ 08:19Z]

**By hand.** **[Recommended]**
1. List the headings from the web article's table of contents.
2. Search the per-page files for each heading.
   ```bash
   grep -l -E "^ *(Experiment Setup|Basic Results)" "${TMPDIR:-/tmp}"/paper/p*.txt
   ```
   Needs the per-page text files from section 1; checked ([VERIFICATION.md](VERIFICATION.md)).
   Expected: the hits include p. 9 and p. 10, plus false positives (when this chapter was
   written, also pp. 35, 53 and 54, where the words recur). Open each hit and keep the one where
   the heading starts a section.
3. Write the table: section · pages · code · experiment · notebook · app page. Columns you cannot
   fill yet get "—".
4. Check: every page number in your SPEC falls inside a range in the map.

**Failure mode.** A map built from a different print. Page numbers here refer to a 62-page
print; references/README.md says how to make one (US Letter, default margins). **[Documented]**
When this chapter was written, `pdfinfo` reported the local copy's page size as 576 × 792 pt,
which is not US Letter's 612 × 792; so check the page count and a few headings, not the paper
size. **[Documented]** (pdfinfo output, observed while writing.)

## 3. Notation and conventions

**Purpose.** One symbol, one meaning, one name in code, one sign convention.

**What was done here.** SPEC §3 has a notation table (symbol · meaning · code name), and fixes
these conventions **[Documented]**:

| Convention | Paper | SPEC choice |
|---|---|---|
| Sparsity | S, the probability a feature is zero (p. 9) | code uses `feature_probability` p = 1 − S |
| Bias sign | +b on p. 10; ReLU(WᵀWx − b) on p. 14 | +b everywhere, note the sign where it matters |
| Loss reduction | L = Σₓ Σᵢ Iᵢ(xᵢ − x′ᵢ)² (p. 10) | sum over features → mean over batch → sum over instances |
| Initialization | not stated | Xavier normal per instance, std √(2/(n + m)); never `xavier_normal_` on a 3-D tensor |
| Feature order | most to least important in figures (p. 10) | the same in every figure |
| 2D comparison | rotate and flip (p. 22) | canonicalize first; the rule is D-109 |

CLAUDE.md repeats these as "§3 pitfalls" for every session. **[Documented]**

**By hand.** **[Recommended]** Build the table with columns: symbol · paper meaning · page ·
code name · shape · convention notes. Look specifically for (a) a quantity the paper names in
two ways (S versus 1 − S), (b) signs that change between sections, (c) sums that might be means.
For (c), ask what the choice changes: summing the loss over instances keeps instances
independent, and for Adam the gradient's scale barely matters (phase-0.md, question 1).

## 4. Finding what is under-specified

**Purpose.** Every value the code needs either has a page or has a DECISIONS entry.

**Method.** **[Recommended]** Walk the pipeline and ask "which page?" for each item: data
distribution → importance → model → initialization → optimizer and learning rate → batch →
steps → schedule → restarts → evaluation samples → restart selection → every threshold → how
you read a figure. Mark each item **stated** (with page), **ambiguous** (two readings) or
**absent**.

**Real examples from this repository** **[Documented]**:

| Entry | What the paper leaves open | Kind |
|---|---|---|
| SPEC §3 | the intro figure's importance (p. 2): "not stated", 0.9ⁱ verified by a run | absent |
| D-101 | optimizer settings (p. 10 gives only the loss) | absent |
| D-109 | how to rotate and flip 2D solutions (p. 22) | absent |
| D-110 | every numeric threshold (graph edges, fraction tolerance, polysemanticity) | absent |
| D-113 | step and restart counts for E00/E01 figures | absent |
| D-303 | p. 14's "sparsity from 1.0 to 0.01" against figure axes labeled density 1 − S | ambiguous |
| D-307 | densities for E04 parts (b)–(d) | absent |
| D-408 | E08(a) density and importance | absent |
| D-502 | E09's density and importance | absent |

**Failure modes.**
- Deciding with an uncited pilot: D-502 chose E09's densities from a scratchpad pilot; D-R03
  later made the pilot a cited run. **[Documented]**
- Resolving an ambiguity silently. D-303 is the good pattern: it quotes both readings and says
  the figures decide. **[Documented]**

## 5. Designing acceptance criteria: paper-grounded versus proposed

A criterion has five parts: the claim (paper words and page), an observable, a statistic, a
threshold, and which models it applies to (here usually the best restart). Tag its **source**:
*paper* when the page states the claim, *proposal* when you invented the operationalization.
SPEC §7 calls its thresholds "initial proposals". **[Documented]**

**The E08 lesson** **[Documented]**:
1. SPEC §7 E08(b) asked that, when sparse, "the number of features computed well exceeds m/2",
   with "computed well" meaning a per-feature loss below 50% of predicting the mean. D-408
   measured exactly that.
2. Phase 4's run computed 3 features well at 1 − S = 0.001 (phase-4.md open issue 1). Restarts,
   longer training, other learning rates and batches gave 8–16 (D-409).
3. The planning conversation then pointed out that m/2 came from the spec, not the paper, and
   recommended re-deriving it from p. 37 and footnotes 16–17 with the original FAIL kept
   visible. [Planning conversation, per owner]
4. D-R01 re-derived it. p. 37 says that, when sparse, the model represents more features than
   it has neurons while usefully approximating the computation. So: more than m = 40 features
   with loss below 75% of the mean baseline. 75% is the midpoint between computing one side of
   |x| exactly (≈ 50%) and not computing the feature (100%).
5. Result: 23 features, FAIL; the original criterion, reported as INFO, gives 3 (rectification.md).
   E08 stays PARTIAL 2/4.
6. The review later found the framing still off: a model without superposition can usefully
   compute up to m features, one side each (R3-08, corrected).

**Checklist per criterion.** **[Recommended]**
1. Source: page and paraphrase, or "proposal" and why.
2. Baseline: what does a model *without* the effect score? (E08: no superposition gives up to m
   one-sided features, so "> m/2" did not test superposition at all.)
3. Ideal and broken weights: unit tests where the criterion passes and fails as it should.
4. Seeds: will one seed decide it? (E02's 0.819 against 0.80 did not hold on seeds 1 and 2,
   D-605.)
5. Commit it before the run, and write down what you had already seen (D-305 and D-306 are
   honest examples of that disclosure).

## 6. Verification runs before committing to numbers: the E00 lesson

**What was done** [Planning conversation, per owner]: before E00's criteria existed, E00 ran in
JAX for both models with n = 5, m = 2, Iᵢ = 0.9ⁱ, S ∈ {0, 0.8, 0.9}, 8 restarts, Adam at 1e-3,
batch 1,024, 10,000 steps, Xavier std √(2/(n + m)); the closed-form linear optimum matched.

**What it left in the SPEC** **[Documented]**: the importance curve for p. 2 (SPEC §3), the
optimizer defaults (SPEC §6; D-101), and a reference table:

| S | ReLU best loss | Linear best loss | Linear exact optimum |
|---|---|---|---|
| 0 | 0.182 | 0.182 | 0.183 |
| 0.8 | 0.059 | 0.122 | 0.124 |
| 0.9 | 0.0196 | 0.067 | 0.068 |

**What happened next** **[Documented]** (REPRODUCTION.md E00; phase-1.md open issue 1; D-107):
- The build's S = 0.9 pentagon lost 0.0214 on 1,000,000 samples, about 9% above 0.0196, with the
  same geometry (norms 1.08–1.11, biases −0.20 to −0.21).
- Logged diagnostics: a 20,000-sample estimate has 3.7% noise, not SPEC's 1–2%; 1 of 200 such
  estimates reaches 0.0196; all 12 cyclic feature arrangements, trained from an ideal pentagon,
  converge to 0.02142–0.02165.
- The reference linear losses lie *below* the provable optimum, the mark of picking the luckiest
  of several noisy estimates.
- Conclusion, labeled a hypothesis (R1-05): 0.0196 is a low draw. No criterion depends on it.

**By hand: a verification-run plan.** **[Recommended]**
1. Implement the setup independently and minimally; same hyperparameters as the SPEC.
2. Evaluate on ≥ 100,000 samples; for the best restart, re-evaluate on fresh samples.
3. Record all restarts, seeds and code; keep it runnable.
4. Copy results into the SPEC as reference values, with the evaluation sample count.

## 7. Reference values versus criteria

- **Reference values** come from *your own* verification run. They are sanity checks: SPEC §7
  says so for E00, D-106 made them INFO rows, and the cited run shows them as "sanity check"
  rows. **[Documented]**
- **Criteria** come from the paper's claim, through an operationalization you commit before
  running. E03's anchors (D* ≈ 1 up to 1.2, a plateau at 0.5, ≈ 0.25 at 8) were read off the
  p. 17 figure at 300 dpi [Planning conversation, per owner]; those are readings of the
  *paper*, so they can be criteria. **[Documented]** (SPEC §7 E03; D-306.)
- Rule **[Recommended]**: a number from your own run never becomes a pass bar. A number read
  from the paper can, with a tolerance you justify.

## 8. Closed-form checks

**Why.** A case where you know the exact answer tests the loss, the data sampler and the
evaluation code at once, independent of training.

**The linear optimum** (SPEC §7 E00; `theory.linear_optimum_loss`). Each feature is 0 with
probability 1 − p, else U[0, 1], so E[xᵢ] = p/2 and E[xᵢ²] = p/3, and its variance is
v = p/3 − p²/4. With the best bias b = (I − WᵀW)μ, the linear model's loss is
v·‖D^{1/2}(I − WᵀW)‖²_F with D = diag(I). WᵀW has rank ≤ m, so the best choice keeps the m
largest importances exactly and predicts the mean for the rest:

  L* = Σ_{i ≥ m} Iᵢ (p/3 − p²/4)

For n = 5, m = 2, Iᵢ = 0.9ⁱ: Σ_{i≥2} Iᵢ = 0.81 + 0.729 + 0.6561 = 2.1951, so L* = 0.183,
0.124 and 0.068 at p = 1, 0.2 and 0.1. **[Documented]** (theory.py docstring; SPEC §7.)

**Where it was used** **[Documented]**:
- It matched the planning conversation's JAX runs. [Planning conversation, per owner]
- The build session re-checked the values [S 35cab3f0 @ 08:29Z] and made them a unit test of
  the loss and evaluation code (`4558329`).
- It is graded in E00 (best linear loss within 3%; 0.13%, 0.38% and 1.01%), and the model's
  exact loss sits within 0.01% (R1-11), so the sampled gap is noise plus the winner's curse.
- D-301 validated the Phase 3 closed forms against nested adaptive quadrature (to 1e-11) and
  Monte Carlo (within 4 standard errors); D-302's tests recover the optimum I·(p/3 − p²/4)
  with b = p/2 for a dropped feature.

**Check it yourself.**

```bash
uv run python -c "from superposition.data import geometric_importance; from superposition.theory import linear_optimum_loss; I = geometric_importance(5, 0.9); print([round(linear_optimum_loss(I, 2, p), 5) for p in (1.0, 0.2, 0.1)])"
uv run pytest tests/unit/test_train.py::test_evaluation_reproduces_linear_optimum -q
uv run pytest tests/unit/test_theory.py -q -k "monte_carlo or quadrature"
```

Expected: `[0.18292, 0.12439, 0.06768]`; `3 passed`; `17 passed, 23 deselected` (counts when
this chapter was written). If the first differs, check the variance formula and that the sum
starts at i = m with zero-based indices.

## 9. Worked example: pp. 9–11 → E00 and E01

This re-derives one section into SPEC text. Steps marked **[Recommended]** are the re-derivation;
**[Documented]** marks what SPEC.md and the repository actually contain.

**Step A. Read.** Extract pp. 9–11 per page (section 1) and render p. 2 and p. 11, whose
figures hold the claims. **[Recommended]**

**Step B. Facts with pages.** **[Recommended]**, each checked against SPEC §3 **[Documented]**:

| Fact (paraphrased) | Page | In SPEC |
|---|---|---|
| xᵢ = 0 with probability Sᵢ, else U[0, 1]; usually Sᵢ = S | p. 9 | §3 Data |
| each feature has an importance Iᵢ weighting its error | p. 9; fn 1 | §3 Data |
| linear model x′ = WᵀWx + b; ReLU output model x′ = ReLU(WᵀWx + b) | p. 10 | §3 Models |
| the bias lets the model set unrepresented features to their expected value | p. 10 (and p. 11 figure) | E00 bias criterion |
| loss Σₓ Σᵢ Iᵢ(xᵢ − x′ᵢ)² | p. 10 | §3 Loss |
| show WᵀW and b, features ordered by importance | p. 10 | §3 Visualization |
| ‖Wᵢ‖ says whether feature i is represented | p. 11 | §3 Metrics |
| superposition score Σ_{j≠i}(Ŵᵢ·Wⱼ)²: 0 if orthogonal, ≥ 1 if others can activate i as strongly | p. 11 | §3 Metrics |
| n = 20, m = 5, Iᵢ = 0.7ⁱ, at seven densities 1 − S = 1 … 0.001 | p. 11 (figure) | E01 setup |
| linear learns the top m; ReLU matches it when dense and superposes as sparsity rises | p. 12 | E01 criteria |
| n = 80, m = 20 with Iᵢ = 0.9ⁱ | p. 12 | E01 setup |
| intro: n = 5 in m = 2; orthogonal pair, antipodal pairs, pentagon at 0%, 80%, 90% sparsity | p. 2 (figure only) | E00 |

**Step C. Notation.** Adopt the paper's symbols; add p = 1 − S for code; fix +b; fix the loss
reduction order (section 3). **[Documented]** (SPEC §3.)

**Step D. Under-specification list.** **[Recommended]**, with where each was resolved
**[Documented]**: optimizer, learning rate, batch, steps (SPEC §6 from the verification run;
D-101); initialization (SPEC §3); the intro figure's importance (SPEC §3, 0.9ⁱ); its
sparsities (SPEC §7 E00 uses S ∈ {0, 0.8, 0.9}, the figure's 0%, 80% and 90%); restarts (8,
with p. 23's advice to keep the lowest loss); "represented" threshold 0.5 (SPEC §3); "orthogonal" thresholds (D-105,
D-106); evaluation samples (D-106: 100,000); canonicalization (D-109); E01 steps and restarts
(D-113).

**Step E. Draft criteria, tagged.** **[Recommended]** tagging of the criteria SPEC §7 contains
**[Documented]**:

| Criterion | Source |
|---|---|
| S = 0: exactly {0, 1} represented, \|cos\| < 0.1 | paper (p. 2 figure); 0.1 proposal |
| S = 0.8: {0, 1, 2, 3} as two antipodal pairs, cos < −0.95 | paper (p. 2); −0.95 proposal |
| S = 0.9: all 5, gaps 72° ± 10°, norms > 1, biases < 0 | paper (p. 2 pentagon; negative bias p. 10); tolerances proposal (the verification run's notes in SPEC §7 give norms 1.08–1.11, biases −0.20 to −0.21) |
| linear: exactly {0, 1}, orthogonal, at every S | paper (p. 12: top m) |
| features with ‖Wᵢ‖ < 0.05 have bias (1 − S)/2 ± 0.02 | paper (p. 10); the value derived (mean of xᵢ); tolerance proposal |
| best linear loss within 3% of the closed form | derived, not in the paper; 3% proposal |
| E01: ReLU count never decreases (tolerance 1) and exceeds m when sparsest | paper (pp. 11–12); tolerance proposal (D-105) |
| E01: superposition appears first among less important features | paper (p. 11) |

**Step F. Verification run plan.** As in section 6, with the settings the planning conversation
used: what was run is from [Planning conversation, per owner]; the plan itself is
**[Recommended]**.

**Step G. Closed-form check.** L* = Σ_{i≥2} 0.9ⁱ(p/3 − p²/4) = 0.183, 0.124, 0.068 (section 8).
**[Documented]**

**Step H. The resulting SPEC text.** SPEC §7 "E00 intro_pentagon (p. 2)" holds the setup, three
"Accept" blocks (ReLU, linear, both models' bias), the exact-loss criterion with its numbers and
the instruction to add a unit test, and the reference table marked "for sanity checks, not as
acceptance criteria". SPEC §7 E01 holds the setup, the figure layout and the criteria above.
**[Documented]** Read them now, next to your own draft.

**Step I. What the build did with it.** E00 passed 11/11 and E01 12/12 on the quick and full
profiles; D-105 and D-106 turned the words into tests; the 0.0196 gap became logged diagnostics;
D-113's premise about E01 was withdrawn by D-114 after a 300-dpi reading. **[Documented]**
(phase-1.md.) The recipe cards [experiments/E00.md](experiments/E00.md) and
[experiments/E01.md](experiments/E01.md) take it from here.

**Drill.** **[Recommended]** Repeat steps A–H for p. 17 (E03) without opening SPEC §7 E03. Then
compare your thresholds with SPEC's and with D-306, and say which of yours are paper readings and
which are proposals.
