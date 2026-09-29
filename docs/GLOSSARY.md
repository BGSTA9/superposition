# Glossary

Short definitions in the paper's sense, each with the pages where the paper introduces or uses
the term. Pages refer to the 62-page PDF print (see `references/README.md`).

**Feature** (pp. 4–5, 9). A property of the input that the network represents. The paper
weighs three working definitions (arbitrary functions, human-interpretable properties, and
"properties a sufficiently large model would dedicate a neuron to") and writes with the last in
mind. In the toy models each input coordinate xᵢ *is* a feature, which is what makes them a
ground truth.

**Linear representation** (pp. 5–6, 8). Features correspond to directions: features f₁, f₂, …
with activations x_{f₁}, x_{f₂}, … are represented by x_{f₁}W_{f₁} + x_{f₂}W_{f₂} + …. In the
toy model, Wᵢ (column i of W) is feature i's direction.

**Privileged basis** (pp. 6–7, 31). A representation where something in the architecture
(typically an elementwise nonlinearity) makes the basis directions special, so features have
an incentive to align with them. Basis directions of a privileged basis are called *neurons*.
The linear and ReLU output toy models have no privileged hidden basis (any rotation O gives the
same model, p. 31); the ReLU hidden layer model does.

**Superposition** (pp. 7–8). Representing more features than there are dimensions, as
almost-orthogonal directions. Formally (p. 8), a linear representation exhibits superposition
if WᵀW is not invertible. This repository applies the test to represented features only.

**Polysemanticity** (pp. 4, 7, 31–33). A neuron responding to several unrelated features. It is
what superposition looks like from the point of view of a privileged basis.

**Sparsity S** (p. 9). The probability that a feature is zero: xᵢ = 0 with probability Sᵢ,
otherwise xᵢ ~ U[0, 1]. The code uses the feature probability (density) p = 1 − S; plots use
1 − S or 1/(1 − S) as the paper does.

**Importance Iᵢ** (p. 9, footnote 1). A scalar multiplier on feature i's squared error in the
loss, modelling that some features matter more for the task than others.

**Interference** (pp. 7, 13). The error caused when non-orthogonal features activate each
other: with feature j active, the model reads Wᵢ·Wⱼ xⱼ into feature i. The linear loss trades
*feature benefit* Σ Iᵢ(1 − ‖Wᵢ‖²)² against interference Σ_{i≠j} Iⱼ(Wⱼ·Wᵢ)² (p. 13). A ReLU
makes negative interference free for 1-sparse inputs.

**Antipodal pair (digon)** (pp. 12, 14, 18–19). Two features stored in one dimension with
opposite directions, Wⱼ = −Wᵢ. Each has feature dimensionality 1/2.

**Dimensions per feature D\*** (p. 17). D* = m/‖W‖²_F; since ‖Wᵢ‖² ≈ 1 for represented features,
‖W‖²_F counts learned features. It is "sticky" at 1 and 1/2.

**Feature dimensionality Dᵢ** (p. 18). Dᵢ = ‖Wᵢ‖² / Σⱼ(Ŵᵢ·Wⱼ)², the fraction of a dimension
feature i gets. It clusters at characteristic fractions: 1 (dedicated), 3/4 (tetrahedron), 2/3
(triangle), 1/2 (antipodal pair), 2/5 (pentagon), 3/8 (square antiprism), 0 (not learned)
(p. 19).

**Uniform polytope** (pp. 19–20). A polytope whose vertices are all equivalent (e.g. the
tetrahedron, triangle, pentagon). Features embedded as its vertices all get the same Dᵢ. Many
toy-model solutions are Thomson-problem solutions (p. 13, 19).

**Tegum product** (p. 20). Combining polytopes by embedding them in orthogonal subspaces
(e.g. a triangular bipyramid = triangle ⊕ digon). Features in different tegum factors do not
interfere; in the feature geometry graph, tegum factors are the connected components.

**Phase change** (pp. 14–16). A discontinuous change in the optimal solution as sparsity or
relative importance varies (here in the generalized sense of footnote 13): a feature switches
between not learned, stored in superposition, and given a dedicated dimension.

**Sparse autoencoder (SAE)** (pp. 42–43, footnote 20). Not named in the paper; it is the modern
realization of the paper's "Approach 2: finding an overcomplete basis": learn an overcomplete,
sparse code for hidden activations so that each dictionary direction corresponds to a feature.
The toy models provide the ground-truth feature directions to score it against (E10).

**Dictionary learning / sparse coding** (pp. 42–43, footnote 20). Factorizing activations
H ≈ A·B with A sparse, where the rows of B form an overcomplete dictionary of feature
directions.
