# Study guide

An ordered path through the paper and this repository for a reader who knows Python, PyTorch
and linear algebra and is new to mechanistic interpretability. Each stop is:
**read** (paper pages) → **notebook** → **app page** → **explain it back** (questions to answer
aloud or in writing, without looking) → **exercises** (at the end of the notebook).

Pages refer to the 62-page PDF print (see `references/README.md`). Stops are added as their
notebooks and app pages land (PLAN.md); the menu of extension projects drawn from the paper's
open questions (pp. 45–46) is written in Phase 6.

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

> Stops 3–5 (the math, phase changes, geometry; notebooks 03–05) arrive with Phase 3.

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
