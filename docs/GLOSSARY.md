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

**Weight family, W ⊥ v** (pp. 14–15, 21). A set of solutions named by the direction v that W
does not represent: project the standard basis onto the complement of v and normalize. W ⊥ [1, 1]
is an antipodal pair; W ⊥ [1, 1, 1] a triangle. The theoretical phase diagrams optimize each
family's scale and biases exactly and keep the cheapest (E02; D-302).

**Output activation σ** (pp. 10, 13, 51). The nonlinearity in x′ = σ(WᵀWx + b); the paper
uses the ReLU. What makes the ReLU special for superposition is that it outputs exactly 0 for
any negative pre-activation, so negative interference is free (p. 13). Redwood found that the
phase diagrams change with σ (p. 51); E12 compares six (`models.ACTIVATIONS`).

**Confused feature** (pp. 51–53). McGrath's solution W₁ ≈ W₂: two features share one direction
with the same sign, so either one lights up both outputs. It is optimal in his exact analysis,
which ignores biases; E02 reports where it wins with and without biases.

**Correlated / anticorrelated feature set** (p. 23). Features that are zero or nonzero together
(correlated), or of which at most one is active at a time (anticorrelated). Correlated features
prefer to be orthogonal, then side by side, then collapse onto their principal component;
anticorrelated features prefer to be antipodal (pp. 23–25; E04).

**Monosemantic / polysemantic neuron** (pp. 31–33). With a privileged basis, a neuron is
monosemantic when it carries one feature and polysemantic when it carries several. In this repo
a neuron carries feature i when |W[h, i]| is at least 25% of its largest weight (E07, E08).

**Neuron stack plot** (p. 32). One column per neuron; each block is one of its weights, with
height |weight| and a color per feature (here: the feature's superposition score); negative
weights hang below the axis. Polysemantic neurons are the columns with more than one block.

**Energy-level jump** (p. 26). During training a feature's dimensionality Dᵢ jumps from one
level (e.g. 0) to another (e.g. 1/2) in a short window, and the loss drops at the same time.

**Adversarial vulnerability** (p. 29). How much an L2 perturbation of 0.1 × the average input
norm, built from a row of WᵀW, raises the loss, relative to a model without superposition:
(L′ − L)/(L′₀ − L₀). It grows with features per dimension, 1/D*.

**Sparse autoencoder (SAE)** (pp. 42–43, footnote 20). Not named in the paper; it is the modern
realization of the paper's "Approach 2: finding an overcomplete basis": learn an overcomplete,
sparse code for hidden activations so that each dictionary direction corresponds to a feature.
The toy models provide the ground-truth feature directions to score it against (E10).

**Dictionary learning / sparse coding** (pp. 42–43, footnote 20). Factorizing activations
H ≈ A·B with A sparse, where the rows of B form an overcomplete dictionary of feature
directions.

**Side neuron** (p. 36). In the absolute-value model, a neuron computing ReLU(xᵢ) (the positive
side of feature i) or ReLU(−xᵢ) (the negative side); |xᵢ| is the sum of the two, so computing
|x| without superposition needs two neurons per feature.

**Path gain** (pp. 38–39). For the absolute-value model, G_h[i, j] = |W₁[h, i]|·W₂[j, h]: how
much output j moves per unit of feature i through neuron h when only feature i is active (on
the side h reads). It does not change under the footnote-15 rescaling; E09 names neurons from it.

**Asymmetric superposition** (p. 39). One neuron stores two features with unequal input weights
and reciprocal output weights (e.g. W₁ = [2, −½], W₂ = [½, 2]): each is read out with gain 1,
but the *heavy* feature (large input weight) interferes strongly with the *victim* and not the
reverse.

**Inhibition** (p. 39). A second neuron reads a little of the heavy feature (on the same side)
and writes strongly negative to the victim, cancelling the positive interference, which the
output ReLU would otherwise pass. With asymmetric superposition it forms the **asymmetric
superposition motif** ("motif" in the circuits sense: a recurring pattern of connections, p. 38).

**Ablation** (zero / mean). Replacing a neuron's activation by 0, or by its average over the
data, and measuring how each feature's loss changes. If a neuron's role is right, the losses
that rise are the ones the role predicts (E09).

**Rebuild test** (SPEC §6). Building a model by hand from the inferred roles and comparing its
loss with the trained model's: the share of the trained model's loss reduction (over predicting
the mean) that the rebuild recovers measures how much of the model the explanation covers.

**Dead latent, resampling** (pp. 42–44 context). An SAE latent that (almost) never fires gets no
gradient and never recovers. Resampling re-initializes it toward inputs the SAE reconstructs
badly.

**TopK SAE.** A sparse autoencoder that keeps only the k largest latent activations per input
instead of paying an L1 penalty (E10 compares both).
