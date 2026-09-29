# Paper map

Every section of *Toy Models of Superposition* mapped to the code, experiments, notebooks and
app pages that implement or explain it. Pages refer to the 62-page PDF print (see
`references/README.md`). Cells marked "—" have no implementation by design; "planned (Phase N)"
cells are filled in as the phases land (PLAN.md).

| Paper section | Pages | Code | Experiment | Notebook | App page |
|---|---|---|---|---|---|
| Introduction; Key results | 1–3 | — | E00 intro_pentagon | 02 | planned (Phase 2) |
| Definitions and Motivation: Features, Directions, and Superposition | 3–8 | `metrics.is_superposition` | — | 01 | Home |
| ↳ Empirical Phenomena; What are Features? | 4–5 | — | — | 01 | — |
| ↳ Features as Directions | 5–6 | — | — | 01 | — |
| ↳ Privileged vs Non-privileged Bases | 6–7 | — | — | 01 | — |
| ↳ The Superposition Hypothesis | 7–8 | — | — | 01 | — |
| ↳ Summary: A Hierarchy of Feature Properties | 8 | `metrics.is_superposition` | — | 01 | — |
| Demonstrating Superposition: Experiment Setup | 9–10 | `data`, `models`, `train` | E00, E01 | 02 | Playground |
| Demonstrating Superposition: Basic Results | 10–12 | `metrics`, `plotting` | E01 basic_results | 02 | Playground |
| Mathematical Understanding | 12–14 | planned (Phase 3): `theory` | — | 03 | — |
| Superposition as a Phase Change | 14–16 | planned (Phase 3): `theory` | E02 phase_change | 04 | Phase diagrams |
| The Geometry of Superposition: Uniform Superposition | 16–21 | `metrics` (D*, Dᵢ, graph) | E03 uniform_geometry | 05 | Geometry |
| ↳ Feature Dimensionality; Why These Geometric Structures? | 18–20 | `metrics.feature_dimensionality`, `metrics.geometry_graph` | E03 | 05 | Geometry |
| ↳ Aside: Polytopes and Low-Rank Matrices | 20–21 | `metrics` (simplex test) | — | 05 | — |
| The Geometry of Superposition: Non-Uniform Superposition | 21–26 | `data.CorrelatedSets`, `data.AnticorrelatedSets` | E04 nonuniform_geometry | 05 | Non-uniform |
| Superposition and Learning Dynamics | 26–28 | `train` (snapshots) | E05 learning_dynamics | 06 | Dynamics |
| Relationship to Adversarial Robustness | 28–30 | planned (Phase 4): `adversarial` | E06 adversarial | 07 | Adversarial |
| Superposition in a Privileged Basis | 31–34 | `models.TiedModel(hidden_act="relu")` | E07 privileged_basis | 08 | Neurons |
| Computation in Superposition | 34–39 | `models.AbsModel` | E08 abs_computation | 08 | Neurons |
| ↳ The Asymmetric Superposition Motif | 38–39 | planned (Phase 5): `interp` | E09 mech_interp_abs | 09 | Neurons |
| The Strategic Picture of Superposition | 40–44 | — | — | 10 | — |
| ↳ Approach 1: models without superposition (L1) | 42 | `train(l1_hidden=…)` | E11 l1_removes_superposition | 10 | SAE lab |
| ↳ Approach 2: finding an overcomplete basis | 42–43 | planned (Phase 5): `sae` | E10 sae_recovery | 10 | SAE lab |
| Discussion | 45 | — | — | 11 | — |
| Open Questions | 45–46 | — | extension menu in STUDY_GUIDE | 11 | — |
| Related Work | 47–50 | — | — | — | — |
| Comments & Replications (Redwood, McGrath, OpenAI) | 51–53 | `models` activation registry; `theory` confused family | E12 activation_functions; E02 | 04, 11 | Phase diagrams |
| Citation information | 55 | `references/README.md`, `CITATION.cff` | — | — | — |
| Footnotes (1: importance; 15: abs rescaling; 17: losses not comparable) | 56–57 | `train` loss; `plotting` (Phase 4) | — | — | — |
| Appendix: Nonlinear Compression | 60 | — | — | — | — |
| Appendix: Compressed sensing lower bounds | 61–62 | — | — | — | — |
