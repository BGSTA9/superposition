# Paper map

Every section of *Toy Models of Superposition* mapped to the code, experiments, notebooks and
app pages that implement or explain it. Pages refer to the 62-page PDF print (see
`references/README.md`). Cells marked "—" have no implementation by design. The lab's Home
page shows this table (it is parsed from this file, so edit it here). docs/REVIEW.md lists what
the Phase 6 review found in each experiment.

| Paper section | Pages | Code | Experiment | Notebook | App page |
|---|---|---|---|---|---|
| Introduction; Key results | 1–3 | — | E00 intro_pentagon | 02 | Playground (p. 2 presets) |
| Definitions and Motivation: Features, Directions, and Superposition | 3–8 | `metrics.is_superposition` | — | 01 | Home |
| ↳ Empirical Phenomena; What are Features? | 4–5 | — | — | 01 | — |
| ↳ Features as Directions | 5–6 | — | — | 01 | — |
| ↳ Privileged vs Non-privileged Bases | 6–7 | — | — | 01 | — |
| ↳ The Superposition Hypothesis | 7–8 | — | — | 01 | — |
| ↳ Summary: A Hierarchy of Feature Properties | 8 | `metrics.is_superposition` | — | 01 | — |
| Demonstrating Superposition: Experiment Setup | 9–10 | `data`, `models`, `train`, `lab` | E00, E01 | 02 | Playground |
| Demonstrating Superposition: Basic Results | 10–12 | `metrics`, `plotting` | E01 basic_results | 02 | Playground (p. 11 preset) |
| Mathematical Understanding | 12–14 | `theory` (`linear_loss_decomposition`, `one_sparse_decomposition`, `k_sparse_weights`, `relu_output_expected_loss`, `linear_optimum_loss`); `metrics.loss_by_num_active` | — | 03 | Playground probe panel (interference) |
| Superposition as a Phase Change | 14–16 | `theory` (families, `optimize_family`, `theoretical_phase_diagram`), `plotting.phase` | E02 phase_change | 03, 04 | Phase diagrams |
| The Geometry of Superposition: Uniform Superposition | 16–21 | `metrics` (D*, Dᵢ, graph), `plotting.geometry` | E03 uniform_geometry | 05 | Geometry |
| ↳ Feature Dimensionality; Why These Geometric Structures? | 18–20 | `metrics.feature_dimensionality`, `metrics.geometry_graph`, `metrics.component_geometry`, `metrics.geometry_census` | E03 | 05 | Geometry |
| ↳ Aside: Polytopes and Low-Rank Matrices | 20–21 | `theory.Family` (W ⊥ v by projection); `metrics` (simplex test) | E02 (families) | 03, 05 | Phase diagrams |
| The Geometry of Superposition: Non-Uniform Superposition | 21–26 | `data.CorrelatedSets`, `data.AnticorrelatedSets`, `metrics.canonicalize_2d`, `experiments.nonuniform_geometry` | E04 nonuniform_geometry | 05 | Non-uniform |
| Superposition and Learning Dynamics | 26–28 | `train` (snapshots), `experiments.learning_dynamics` (jump detector, geometric stages), `plotting.dynamics` | E05 learning_dynamics | 06 | Dynamics |
| Relationship to Adversarial Robustness | 28–30 | `adversarial` (analytic attacks, vulnerability), `plotting.adversarial` | E06 adversarial | 07 | Adversarial |
| Superposition in a Privileged Basis | 31–34 | `models.TiedModel(hidden_act="relu")`, `metrics.neuron_polysemanticity`, `plotting.neurons` (stack plots) | E07 privileged_basis | 08 | Playground (ReLU hidden, W view); Neurons |
| Computation in Superposition | 34–39 | `models.AbsModel`, `metrics.normalize_hidden_units`, `metrics.per_feature_loss` | E08 abs_computation | 08 | Neurons |
| ↳ The Asymmetric Superposition Motif | 38–39 | `interp` (path gains, neuron roles, motif detector, ablations, rebuild), `plotting.interp` | E09 mech_interp_abs | 09 | Neurons (E09 tab) |
| The Strategic Picture of Superposition | 40–44 | — | E10, E11 | 10 | SAE lab |
| ↳ Approach 1: models without superposition (L1) | 42 | `train(l1_hidden=…)`, `plotting.l1` | E11 l1_removes_superposition | 10 | SAE lab (E11 tab) |
| ↳ Approach 2: finding an overcomplete basis | 42–44 | `sae` (SAEs, TopK, resampling, ground-truth recovery), `plotting.sae` | E10 sae_recovery | 10 | SAE lab |
| Discussion | 45 | — | — | 11 | — |
| Open Questions | 45–46 | — | extension menu in STUDY_GUIDE | 11 | — |
| Related Work | 47–50 | — | — | — | — |
| Comments & Replications (Redwood, McGrath, OpenAI) | 51–53 | `models` activation registry; `theory` confused family; `theory.activation_square_integral` (exact losses for any output activation); `plotting.activations` | E12 activation_functions; E02 | 04, 11 | Phase diagrams (E12 tab) |
| Citation information | 55 | `references/README.md`, `CITATION.cff` | — | — | — |
| Footnotes (1: importance; 15: abs rescaling; 17: losses not comparable; 20: sparse coding; 21: a little L1 is free) | 56–57 | `train` loss; `metrics.normalize_hidden_units` (15); `interp.path_gains` (15-invariant); `sae` (20) | E08, E09 (15); E10 (20); E11 (21) | 08–10 | Neurons, SAE lab |
| Appendix: Nonlinear Compression | 60 | — | — | — | — |
| Appendix: Compressed sensing lower bounds | 61–62 | — | — | — | — |
