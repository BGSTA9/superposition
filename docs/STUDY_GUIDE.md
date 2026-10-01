# Study guide

An ordered path through the paper and this repository for a reader who knows Python, PyTorch
and linear algebra and is new to mechanistic interpretability. Each stop is:
**read** (paper pages) → **notebook** → **app page** → **explain it back** (questions to answer
aloud or in writing, without looking) → **exercises** (at the end of the notebook).

Pages refer to the 62-page PDF print (see `references/README.md`). Stops 0–11 follow the
notebooks; the last section is a menu of extension projects drawn from the paper's open
questions (pp. 45–46). docs/REVIEW.md lists what an independent review found in each
experiment: read an experiment's review entries after its stop, as a lesson in what can go
wrong in a replication.

## Stop 0 · Orientation

- **Read:** `README.md`, `CLAUDE.md` ("§3 pitfalls"), `docs/DESIGN.md` ("Layers").
- **Notebook:** `notebooks/00_orientation.ipynb`.
- **App:** `uv run superposition app` → **Home**: the status table and the paper map. Filter the
  map for a page number you are reading.
- **Explain it back:**
  1. What does the leading `inst` dimension of every tensor mean, and why can a whole sparsity
     sweep train as one model?
  2. What does a run directory contain, and which fields make a run reproducible?
  3. Why are losses at different sparsities not comparable (footnote 17)?

## Stop 1 · Features, directions and superposition (pp. 1–8)

- **Read:** pp. 1–8, slowly. Keep `docs/GLOSSARY.md` open.
- **Notebook:** `notebooks/01_features_directions_superposition.ipynb`.
- **App:** **Playground**, preset "p. 2 · pentagon". In the probe panel, activate feature 0
  alone and note which features receive a positive readout: those are its neighbours in the 2D
  view (features 2 and 3 in the default model; a trained pentagon need not be in index order).
  Then activate those two neighbours together and watch their interference compound into a
  spurious output for feature 0.
- **Explain it back:**
  1. What is a linear representation? Give the formula for the hidden vector of an input with
     several active features.
  2. Why is the hidden space of the ReLU output model not a privileged basis, while the ReLU
     hidden model's is? What experiment in the notebook shows the difference?
  3. State the formal definition of superposition (p. 8). Why does it only make sense for
     represented features?
  4. Why does superposition need sparsity *and* a nonlinearity to pay off?

## Stop 2 · Demonstrating superposition (pp. 8–12)

- **Read:** pp. 8–12, then the introduction figure on p. 2 again.
- **Notebook:** `notebooks/02_demonstrating_superposition.ipynb` (runs E00; trains E01's
  n = 20 model).
- **Results:** `docs/REPRODUCTION.md`, sections E00 and E01; figures in `docs/figures/`.
- **App:** **Playground**. Step through the four p. 2 presets, then "p. 11 · n = 20, m = 5".
  Before each Train, predict the number of represented features; afterwards read the Dᵢ table.
  Switch the model to Linear at the same setting and compare the loss with the closed-form
  optimum shown in the metrics row.
- **Explain it back:**
  1. Describe the data, the two models and the loss of pp. 9–10 from memory.
  2. What does the linear model learn at every sparsity, and why is its optimal loss
     Σ_{i≥m} Iᵢ(p/3 − p²/4)?
  3. Walk through the p. 2 figure: what changes between S = 0, 0.8 and 0.9, and why does the
     least important feature disappear first?
  4. In a trained pentagon, why are the biases negative and the norms above 1? What
     interference remains?
  5. Read one column of the p. 11 figure: what do the WᵀW heatmap, the bias column and the bar
     colors each tell you?

## Stop 3 · The math (pp. 12–14)

- **Read:** pp. 12–14 ("Mathematical Understanding"), and footnotes 10–12.
- **Notebook:** `notebooks/03_the_math.ipynb` (exact computations only; under a minute).
- **Code:** `src/superposition/theory.py`: the linear decomposition, the 0- and 1-sparse terms,
  the exact loss of a model whose outputs read at most one other feature (docs/DECISIONS.md
  D-301).
- **App:** **Playground** probe panel. Activate one feature of a pentagon and read the other
  outputs before and after the bias: that is the 1-sparse interference term.
- **Explain it back:**
  1. Write the linear model's loss as feature benefit plus interference. Why can an antipodal
     pair never beat dropping the less important feature in the linear model?
  2. Why do L₀ and L₁ dominate the loss as S → 1, and what does each one penalize?
  3. Why is negative interference free on 1-sparse inputs, and what does a negative bias cost?
  4. What is the "toy model of the toy model", and why does a crossing of two families' loss
     curves make a first-order phase change?

## Stop 4 · Superposition as a phase change (pp. 14–16, 51–53)

- **Read:** pp. 14–16, then McGrath's comment on pp. 51–53.
- **Notebook:** `notebooks/04_phase_change.ipynb` (runs E02, or reuses your latest quick run).
- **Results:** `docs/REPRODUCTION.md`, section E02.
- **App:** **Phase diagrams**. Switch between the empirical and theoretical views, pick cells
  marked × with the two sliders, and compare the losses of the trained models with the winning
  family's optimum. Then show the theory with every bias fixed at 0.
- **Explain it back:**
  1. How is each empirical cell labeled, and why does the paper discard the highest-loss model?
  2. What are the three (and, for n = 3, four) families of the theoretical diagram, and which
     scalars are optimized in each?
  3. Where do empirical and theoretical diagrams disagree, and how do you tell a boundary cell
     from a stuck model?
  4. Why does McGrath's confused solution disappear once biases are free?

## Stop 5 · The geometry of superposition (pp. 16–25)

- **Read:** pp. 16–21 (uniform superposition, Dᵢ, polytopes, tegum products), then pp. 21–25
  (non-uniform superposition).
- **Notebook:** `notebooks/05_geometry.ipynb` (runs E03 and E04, or reuses your latest quick
  runs).
- **Results:** `docs/REPRODUCTION.md`, sections E03 and E04.
- **App:** **Geometry**: drag the sparsity slider across the 1/2 plateau and watch the geometry
  graph break into digons, then into larger factors. **Non-uniform**: move feature 0's density
  and find where the best solution snaps from a pentagon to two digons; then step through the
  collapse of correlated pairs.
- **Explain it back:**
  1. What does D* measure, and why is it "sticky" at 1 and 1/2?
  2. Why does a uniform polytope give every vertex Dᵢ = (dimensions)/(features)? Why does a
     triangular bipyramid show up at 2/3 and 1/2 rather than 3/5?
  3. When one feature gets sparser, why do its neighbours close in, and why does the pentagon
     eventually snap?
  4. Why do correlated features prefer to be orthogonal, anticorrelated ones antipodal, and
     when do correlated pairs collapse onto their principal component?

## Stop 6 · Learning dynamics (pp. 26–28)

- **Read:** pp. 26–28; revisit p. 17 (the D* plateau) and pp. 18–19 (Dᵢ) first.
- **Notebook:** `notebooks/06_learning_dynamics.ipynb` (runs E05, or reuses your latest quick
  run).
- **Results:** `docs/REPRODUCTION.md`, section E05.
- **App:** **Dynamics**. On the first tab, pick a restart and drag the step slider across the
  orange jump: watch Σᵢ Dᵢ and the loss change together. On the second tab, choose 1 − S = 0.01
  and scrub a run from step 0 to the end; name each stage before the label tells you.
- **Explain it back:**
  1. What is an "energy level" of a feature, and why do features in the uniform setup end at
     Dᵢ ≈ 1/2 or 0 at 1/(1 − S) = 3?
  2. Why can a model sit on a loss plateau for tens of thousands of steps with one dimension
     badly used, and what happens to the loss when it fixes it?
  3. Why do antipodal pairs in the final octahedron always join features from *different*
     correlated sets?
  4. What is the "collapsed" stage, and why is it the solution a linear model would learn?

## Stop 7 · Superposition and adversarial examples (pp. 28–30)

- **Read:** pp. 28–30.
- **Notebook:** `notebooks/07_adversarial.ipynb` (computes an attack by hand, then runs E06).
- **Results:** `docs/REPRODUCTION.md`, section E06.
- **App:** **Adversarial**. Compare the two graded settings' curves; then open the extended
  n = 20 setting and find where vulnerability stops tracking 1/D*.
- **Explain it back:**
  1. Write the paper's analytic attack for feature i. Why is it the best L2 perturbation of
     that size for raising output i?
  2. Why is the ratio (L′ − L)/(L′₀ − L₀), rather than L′ − L alone, the right measure across
     sparsities?
  3. Why does vulnerability track features per dimension, 1/D*?
  4. Why does a fixed-fraction budget stop measuring superposition at extreme sparsity?

## Stop 8 · Neurons and computation in superposition (pp. 31–39)

- **Read:** pp. 31–39 and footnote 15 (p. 56).
- **Notebook:** `notebooks/08_neurons_and_computation.ipynb` (runs E07 and E08).
- **Results:** `docs/REPRODUCTION.md`, sections E07 and E08.
- **App:** **Neurons**. On the E07 tab slide from dense to sparse and count monosemantic
  neurons; read the sidestepping table. On the E08 tab compare the |W₁| stacks at 1 − S = 1 and
  0.001, then read the capacity chart against the m/2 line. The **Playground** preset "p. 32 ·
  ReLU hidden layer" trains an E07-sized model live.
- **Explain it back:**
  1. Why does a hidden ReLU make the neuron basis privileged? What does a stack plot show that
     WᵀW cannot?
  2. Describe the monosemantic → polysemantic transition of p. 33 in your own words.
  3. How does the ReLU hidden layer model "sidestep" its ReLU, and why does it want to?
  4. Why does |x| need two neurons per feature without superposition, and what does footnote
     15's rescaling leave unchanged?
  5. What is a "dead side" in the absolute-value model, and why does it matter only when
     features are sparse?

## Stop 9 · Mechanistic interpretability of a toy model (pp. 38–39)

- **Read:** pp. 38–39 (the motif figures) and footnote 15 again.
- **Notebook:** `notebooks/09_mech_interp.ipynb` (hand-built models first, then E09's trained
  ones).
- **Results:** `docs/REPRODUCTION.md`, section E09.
- **App:** **Neurons** → E09 tab. Pick the n = 6, m = 10 model, find the two shaded motif
  neurons, and predict from the weights which feature's loss rises most when you zero each
  inhibitor; then check the ablation matrix. Switch to a context model at 1 − S = 0.1 and
  count the unexplained neurons.
- **Explain it back:**
  1. What is a path gain, and why is it the right quantity to read a neuron by when features
     are sparse?
  2. Why do reciprocal output weights make asymmetric superposition cheap for one feature and
     expensive for the other?
  3. How does an inhibitor turn positive interference into (nearly) nothing, and why does it
     have to read the heavy feature on the same side as the asymmetric neuron?
  4. What does an ablation test that reading the weights cannot? What does the rebuild test
     that ablations cannot?

## Stop 10 · Solving superposition: SAEs and L1 (pp. 40–44)

- **Read:** pp. 40–44 ("The Strategic Picture of Superposition") and footnotes 20–21.
- **Notebook:** `notebooks/10_sparse_autoencoders.ipynb` (an SAE on the pentagon by hand, then
  E10 and E11).
- **Results:** `docs/REPRODUCTION.md`, sections E10 and E11.
- **App:** **SAE lab**. On the live tab, train a 5-latent and then a 20-latent SAE on the
  pentagon and compare the 2D views; switch to TopK k = 1. On the E10 tab, find an SAE that
  reconstructs well but misses the features. On the E11 tab, drag λ and watch the pentagon lose
  features.
- **Explain it back:**
  1. Write the SAE's encoder, decoder and loss. What is each constraint (unit-norm decoder,
     L1 or TopK, resampling) for?
  2. Why can an SAE reconstruct h well without finding the features? Give a 2D example.
  3. Why does "solving superposition" in the paper's sense amount to being able to enumerate
     features (p. 41)?
  4. Why does an L1 penalty on h remove superposition mostly by killing features, and why does
     it not penalize an antipodal pair along a basis axis?

## Stop 11 · Extensions: activation functions (p. 51) and the open questions (pp. 45–46)

- **Read:** p. 51 (Redwood's comment), pp. 45–46 ("Open Questions"), and p. 13 again (why the
  ReLU makes negative interference free).
- **Notebook:** `notebooks/11_extensions.ipynb` (the six activations, one cell solved exactly
  for each, E12's diagrams, and landscape versus optimization).
- **Results:** `docs/REPRODUCTION.md`, section E12 (an extension: no pass/fail).
- **App:** **Phase diagrams** → E12 tab. Compare the regime shares across activations, then pick
  SiLU and find a cell where the vote says "dedicated" but the theory says "superposition";
  read the models table to see whether the lowest-loss model agrees with the theory.
- **Explain it back:**
  1. Which property of the ReLU makes an antipodal pair cheap, and which of the five other
     activations keep it?
  2. Why can losses be compared across activations within one cell, but not across cells of
     different density (footnote 17)?
  3. How do you tell a difference in the loss landscape from a difference in what training
     reaches? Which E12 rows answer that?
  4. Why might a model whose two weights start with the same sign never reach the antipodal
     pair in m = 1?

## Extension projects (the paper's open questions, pp. 45–46)

Each project names the question as the paper asks it (paraphrased), why it matters, and a first
experiment you can run with this library in an afternoon. Keep the replication defaults
untouched: put a variant in its own config profile or experiment, and log its choices in
docs/DECISIONS.md (SPEC §2).

1. **Is there a statistical test for catching superposition?** Superposition is defined on
   weights we can read only in toy models.
   - *First experiment:* train E01's n = 20, m = 5 models at several densities. For each, compute
     a statistic of the hidden activations alone, such as the kurtosis of h along random
     directions, or how many SAE latents survive (E10).
   - *Check:* does the statistic separate the dense, superposition-free models from the sparse
     ones without looking at W?
2. **Can we change the phase diagram so features avoid superposition?** The paper names L1 on
   activations, adversarial training and the activation function as promising.
   - *First experiment:* E02's n = 2, m = 1 grid with `TrainConfig.l1_hidden` (E11's
     mechanism) or with E06's adversarial attack added to training (p. 30).
   - *Check:* as with E12, compare against the theory per cell. Does the superposition region
     shrink, or do features just die?
3. **Are there closed-form solutions beyond n = 2, m = 1?** `theory.py` solves restricted
   families exactly.
   - *First experiment:* add a two-scale antipodal family (unequal norms, review finding R1-03).
     Then try n = 3, m = 2 families whose outputs read two other features, which needs 3D
     integrals (extend `activation_square_integral`).
4. **How realistic are these toy models?**
   - *First experiment:* train a small MLP (one hidden layer) on a real dataset, then measure
     its neurons' polysemanticity (`metrics.neuron_polysemanticity`). Compare with E07's
     sparsity sweep, using input sparsity as the knob.
5. **Can we estimate the feature importance and sparsity curves of real models?**
   - *First experiment:* invert the toy model. Train E01-style models with known importance
     curves and fit the curve back from the loss and neuron sparsity, as the paper suggests.
6. **Does superposition go away at scale?**
   - *First experiment:* E03's uniform setup at fixed n/m with growing m (30, 60, 120), on the
     same density grid.
   - *Check:* does the 1/2 plateau of D* move or stay put?
7. **What is the most principled definition of superposition?**
   - *First experiment:* compare the repo's three measures on E01's models across sparsity:
     the superposition score (p. 11), Dᵢ (p. 18) and the rank test (p. 8). Where do they
     disagree, and why?
8. **How important are polysemantic neurons?**
   - *First experiment:* in E08's n = 100, m = 40 models, ablate monosemantic and polysemantic
     neurons separately (`interp.ablation_matrix`).
   - *Check:* what share of the loss reduction does each group carry?
9. **How many features fit in superposition?**
   - *First experiment:* sweep m at a fixed sparsity in E03's setup, and plot the represented
     features against m. Compare with a compressed-sensing bound of order m·log(n/m).
10. **Is the phase change related to phase changes in compressed sensing?**
    - *First experiment:* overlay E02's theoretical boundary with the sparsity at which ℓ₁
      recovery of x from h = Wx succeeds for W at that cell's optimum.
11. **How does superposition relate to non-robust features?**
    - *First experiment:* E06 measures the vulnerability of a model in superposition. Train
      adversarially (p. 30) at growing budgets, and track D* and the loss on clean inputs.
12. **What computation can be done in superposition?**
    - *First experiment:* replace E08's |x| target with max(x, 0), x², or a product of two
      features, keeping the absolute-value model.
    - *Check:* does the asymmetric motif of E09 reappear?
13. **How does superposition change when features are not independent?**
    - *First experiment:* E04(b) and (d) with more sets, and anticorrelated sets in m > 2.
    - *Check:* can anticorrelated features pack more than m/D* features?
14. **Do models use nonlinear representations?**
    - *First experiment:* give the ReLU output model a deeper encoder and decoder (two layers
      each) with a 2D bottleneck on uncorrelated sparse data.
    - *Check:* does h stay a linear function of x?
