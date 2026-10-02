"""Superposition and geometry metrics: pure functions of weight tensors.

Paper: Summary: A Hierarchy of Feature Properties, p. 8 (formal superposition); Basic
Results, p. 11 (‖Wᵢ‖ and the superposition score); Mathematical Understanding, p. 13
(k-sparse loss terms); Uniform Superposition, p. 17 (D*); Feature Dimensionality, pp. 18–20
(Dᵢ, characteristic fractions, feature geometry graphs, tegum factors); Non-Uniform
Superposition, p. 22 (canonicalizing 2D solutions); Superposition in a Privileged Basis and
Computation in Superposition, pp. 31–38 and footnote 15 (neurons, per-feature loss).

Every function takes ``W`` shaped ``(inst, hidden, feat)`` (column ``W[k, :, i]`` is feature
i's direction Wᵢ in instance k) and returns per-instance or per-feature tensors. Nothing here
mutates its inputs or depends on global state, except :func:`loss_by_num_active` and
:func:`per_feature_loss`, which run a model on sampled inputs from an explicit generator.
"""

# Rebuild kit, milestone M4 (rebuild/README.md). Signatures and docstrings are
# copied from src/superposition/metrics.py; every body raises NotImplementedError. Implement
# them until `uv run pytest -c rebuild/pytest.ini rebuild/tests/test_metrics.py` passes.

from __future__ import annotations

import math
from collections.abc import Callable

import einops
import networkx as nx
import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import Tensor

from superposition_rebuild.data import IndependentSparse

# Norms below this count as exactly zero when normalizing (Ŵᵢ = Wᵢ/‖Wᵢ‖).
_ZERO_NORM = 1e-12

# Characteristic feature dimensionalities (p. 19): name and value, from most to least.
CHARACTERISTIC_FRACTIONS: tuple[tuple[str, float], ...] = (
    ("dedicated dimension", 1.0),
    ("tetrahedron", 3 / 4),
    ("triangle", 2 / 3),
    ("antipodal pair (digon)", 1 / 2),
    ("pentagon", 2 / 5),
    ("square antiprism", 3 / 8),
    ("not learned", 0.0),
)


def gram(W: Float[Tensor, "inst hidden feat"]) -> Float[Tensor, "inst feat feat"]:
    """WᵀW: entry (i, j) is Wᵢ·Wⱼ. Paper: Basic Results, p. 10."""
    raise NotImplementedError


def feature_norms(W: Float[Tensor, "inst hidden feat"]) -> Float[Tensor, "inst feat"]:
    """‖Wᵢ‖: about 1 if feature i is represented, about 0 if not. Paper: p. 11."""
    raise NotImplementedError


def _unit_columns(W: Float[Tensor, "inst hidden feat"]) -> Float[Tensor, "inst hidden feat"]:
    """Ŵᵢ = Wᵢ/‖Wᵢ‖, with zero columns left at zero."""
    raise NotImplementedError


def cosine_similarity_matrix(
    W: Float[Tensor, "inst hidden feat"],
) -> Float[Tensor, "inst feat feat"]:
    """cos(Wᵢ, Wⱼ) = Ŵᵢ·Ŵⱼ; rows/columns of zero features are zero."""
    raise NotImplementedError


def _projection_squares(W: Float[Tensor, "inst hidden feat"]) -> Float[Tensor, "inst feat feat"]:
    """Entry (i, j) is (Ŵᵢ·Wⱼ)², the squared projection of Wⱼ onto feature i's direction."""
    raise NotImplementedError


def superposition_score(W: Float[Tensor, "inst hidden feat"]) -> Float[Tensor, "inst feat"]:
    """Σ_{j≠i} (Ŵᵢ·Wⱼ)²: how strongly other features activate feature i's direction.

    Paper: Basic Results, p. 11. It is 0 when feature i is orthogonal to every other feature,
    and ≥ 1 when other features can activate Wᵢ as strongly as feature i itself. Zero features
    get 0.
    """
    raise NotImplementedError


def feature_dimensionality(W: Float[Tensor, "inst hidden feat"]) -> Float[Tensor, "inst feat"]:
    """Dᵢ = ‖Wᵢ‖² / Σⱼ (Ŵᵢ·Wⱼ)², the fraction of a dimension feature i gets.

    Paper: Feature Dimensionality, p. 18. The sum includes j = i, whose term is ‖Wᵢ‖², so
    Dᵢ ≤ 1. Dᵢ = 1/2 for an antipodal pair, 0 for a feature that is not learned. We define
    Dᵢ = 0 when Wᵢ = 0 (SPEC §3); the formula tends to 0 continuously as ‖Wᵢ‖ → 0 whenever any
    other feature is represented.
    """
    raise NotImplementedError


def dims_per_feature(W: Float[Tensor, "inst hidden feat"]) -> Float[Tensor, " inst"]:
    """D* = m / ‖W‖²_F, the "dimensions per feature". Paper: Uniform Superposition, p. 17.

    ‖Wᵢ‖² ≈ 1 for learned features and ≈ 0 otherwise, so ‖W‖²_F counts learned features.
    """
    raise NotImplementedError


def represented_mask(
    W: Float[Tensor, "inst hidden feat"], threshold: float = 0.5
) -> Bool[Tensor, "inst feat"]:
    """Features with ‖Wᵢ‖ > ``threshold`` count as represented (SPEC §3 default 0.5)."""
    raise NotImplementedError


def is_superposition(
    W: Float[Tensor, "inst hidden feat"], threshold: float = 0.5, rtol: float = 1e-2
) -> Bool[Tensor, " inst"]:
    """Formal superposition test: WᵀW is not invertible on the represented features.

    Paper: Summary: A Hierarchy of Feature Properties, p. 8. Restricted to the represented
    set R, W_Rᵀ W_R is invertible iff rank(W_R) = |R|. Numerically, a singular value counts
    toward the rank when it exceeds ``rtol`` times the largest singular value. With |R| > m the
    answer is always True; with |R| ≤ m it is True only for (near-)linearly dependent features.
    """
    raise NotImplementedError


def angles_2d(W: Float[Tensor, "inst 2 feat"]) -> Float[Tensor, "inst feat"]:
    """Angle of each feature direction in a 2D hidden space, in radians in (−π, π]."""
    raise NotImplementedError


def canonicalize_2d(
    W: Float[Tensor, "inst 2 feat"], threshold: float = 0.5
) -> Float[Tensor, "inst 2 feat"]:
    """Rotate and reflect 2D solutions into a canonical orientation.

    Paper: Perturbing a Single Feature, p. 22 ("we canonicalize them, rotating them to align
    with each other in a consistent manner"). The hidden space of the linear and ReLU output
    models has no privileged basis (p. 31), so solutions are only defined up to an orthogonal
    transform; this picks one representative.

    Convention:

    1. Rotation: the *anchor*, the most important represented feature (lowest index with
       ‖Wᵢ‖ > threshold; feature 0 if none is represented), points straight up (+y).
    2. Reflection: the first feature off the vertical axis lies in the left half-plane
       (x < 0), reflecting x → −x if needed. "First" means represented features by index,
       then unrepresented ones by index; features within ``1e-6·max‖Wᵢ‖`` of the axis are
       skipped because they do not distinguish the two reflections.
    """
    raise NotImplementedError


def _rotation_matrices(theta: Float[Tensor, " inst"]) -> Float[Tensor, "inst 2 2"]:
    raise NotImplementedError


def geometry_graph(
    W: Float[Tensor, "hidden feat"],
    edge_threshold: float = 0.05,
    represented_threshold: float = 0.5,
) -> tuple[nx.Graph, list[set[int]]]:
    """The feature geometry graph of one instance, and its tegum-factor components.

    Paper: Feature Dimensionality, p. 18 ("each feature is a node and edge weights are based on
    the absolute value of the dot product feature embedding vectors"); Why These Geometric
    Structures?, p. 20 (subgraphs are disconnected iff they are in different tegum factors).

    Nodes are all features, with attributes ``norm``, ``dimensionality`` and ``represented``.
    Edges join represented features whose |Wᵢ·Wⱼ| exceeds ``edge_threshold`` (weight = |Wᵢ·Wⱼ|).

    Returns:
        ``(graph, components)``: components are the connected components among represented
        features, largest first (ties broken by smallest feature index).
    """
    raise NotImplementedError


def classify_fraction(D: Float[Tensor, ...], tol: float = 0.02) -> Int[Tensor, ...]:
    """Index of the nearest characteristic fraction within ``tol``, else −1.

    Paper: Feature Dimensionality, p. 19. Indices refer to :data:`CHARACTERISTIC_FRACTIONS`
    (1, 3/4, 2/3, 1/2, 2/5, 3/8, 0).
    """
    raise NotImplementedError


# Uniform polytopes the paper finds (p. 19), keyed by (features, dimensions spanned).
POLYTOPES: dict[tuple[int, int], str] = {
    (1, 1): "dedicated dimension",
    (2, 1): "digon (antipodal pair)",
    (3, 2): "triangle",
    (4, 3): "tetrahedron",
    (5, 2): "pentagon",
    (8, 3): "square antiprism",
}


def component_geometry(
    W: Float[Tensor, "hidden feat"],
    component: list[int] | set[int],
    rank_rtol: float = 0.1,
    dimensionality_tol: float = 0.05,
) -> tuple[int, int, str]:
    """Name the geometry of one tegum factor (a connected component of the geometry graph).

    Paper: Feature Dimensionality and Why These Geometric Structures?, pp. 18–20. A factor of
    k features spanning r dimensions is a uniform polytope when every feature has Dᵢ = r/k
    (its dimensions are shared equally); the paper's characteristic ones are listed in
    :data:`POLYTOPES`. Dᵢ alone cannot tell them apart (2/5 and 3/8 differ by only 0.025),
    but (k, r) can. Dᵢ is computed within the factor, where it equals the model's Dᵢ because
    factors are orthogonal.

    Args:
        W: one instance, ``(hidden, feat)``.
        component: feature indices of the factor.
        rank_rtol: singular values above this fraction of the largest count toward the rank r
            (trained solutions are only approximately low rank).
        dimensionality_tol: every Dᵢ must be within this of r/k to count as uniform.

    Returns:
        ``(k, r, name)``: name is a polytope from :data:`POLYTOPES` if (k, r) is one and the
        factor is uniform, else ``"k in r (uniform)"`` or ``"k in r (non-uniform)"``.
    """
    raise NotImplementedError


def feature_factors(
    W: Float[Tensor, "hidden feat"],
    edge_threshold: float = 0.05,
    represented_threshold: float = 0.5,
    rank_rtol: float = 0.1,
    dimensionality_tol: float = 0.05,
) -> list[str | None]:
    """The name of the tegum factor each feature belongs to, for one instance (None if the
    feature is not represented).

    Paper: pp. 18–20. Components come from :func:`geometry_graph`; each is named by
    :func:`component_geometry`.
    """
    raise NotImplementedError


def geometry_census(
    W: Float[Tensor, "hidden feat"],
    edge_threshold: float = 0.05,
    represented_threshold: float = 0.5,
    rank_rtol: float = 0.1,
    dimensionality_tol: float = 0.05,
) -> dict[str, int]:
    """How many represented features sit in each kind of tegum factor, for one instance.

    Paper: pp. 18–20. Components come from :func:`geometry_graph`; each is named by
    :func:`component_geometry`. Returns feature counts per name (not factor counts), so the
    values sum to the number of represented features.
    """
    raise NotImplementedError


def polytope_of_fraction(value: float) -> str | None:
    """The uniform polytope of :data:`POLYTOPES` whose features all have Dᵢ = ``value``
    (r/k for k features in r dimensions; p. 19), or None."""
    raise NotImplementedError


def neuron_polysemanticity(
    W: Float[Tensor, "inst hidden feat"], rel_threshold: float = 0.25, abs_threshold: float = 0.05
) -> Int[Tensor, "inst hidden"]:
    """Number of features each neuron carries, above a relative-weight threshold.

    Paper: Superposition in a Privileged Basis, pp. 31–33 (a neuron storing more than one
    feature is polysemantic). For neuron h, feature i counts when
    |W[h, i]| ≥ ``rel_threshold`` · maxⱼ |W[h, j]|. A neuron whose largest weight is below
    ``abs_threshold`` is dead and counts 0 features. Only meaningful for models with a
    privileged hidden basis (ReLU hidden layer, absolute value).
    """
    raise NotImplementedError


def normalize_hidden_units(
    W1: Float[Tensor, "inst hidden feat"], W2: Float[Tensor, "inst feat hidden"]
) -> tuple[Float[Tensor, "inst hidden feat"], Float[Tensor, "inst feat hidden"]]:
    """Rescale every hidden unit so its largest-magnitude W₁ weight has magnitude 1.

    Paper: footnote 15 (p. 56). Scaling neuron h's row of W₁ by α > 0 and its column of W₂ by
    α⁻¹ leaves ReLU(W₂ ReLU(W₁x) + b) unchanged (ReLU is positively homogeneous), so this only
    fixes a degree of freedom for visualization and comparison. Dead units (all-zero rows) are
    left at zero.
    """
    raise NotImplementedError


def loss_by_num_active(
    model: torch.nn.Module,
    k: int,
    importance: Float[Tensor, "*inst feat"],
    n_samples: int,
    generator: torch.Generator,
    low: float = 0.0,
    high: float = 1.0,
) -> Float[Tensor, " inst"]:
    """Monte-Carlo estimate of the k-sparse loss term Lₖ (Paper: p. 13).

    The paper groups the expected loss by the number of active features:
    L = Σₖ pᵏ Sⁿ⁻ᵏ Lₖ (for a shared feature probability p = 1 − S), where Lₖ sums the expected
    loss over every set A of exactly k active features: Lₖ = Σ_{|A|=k} E[loss | A]. For example
    L₀ = Σᵢ Iᵢ ReLU(bᵢ)² for the ReLU output model, and L₁ is the 1-sparse term. We estimate
    Lₖ = C(n, k) · E_A[E[loss | A]] with A uniform over k-subsets and active values
    U[``low``, ``high``].

    The model must expose ``n_features``, ``n_inst`` and ``target(x)``. Runs without gradients
    on the generator's device.
    """
    raise NotImplementedError


@torch.no_grad()
def per_feature_loss(
    model: torch.nn.Module,
    feature_probability: Float[Tensor, " inst"],
    n_samples: int,
    generator: torch.Generator,
    low: float = 0.0,
    high: float = 1.0,
    features_per_chunk: int = 8,
    shared_samples: bool = False,
) -> Float[Tensor, "inst feat"]:
    """Expected squared error of each output feature, E[(targetᵢ − x′ᵢ)²] (no importance).

    Paper: Computation in Superposition, pp. 36–37 (which features the model computes well).
    Plain Monte Carlo is too noisy for very sparse features (at 1 − S = 0.001 a feature is
    active once per thousand samples), so we condition on feature i's own activity, which is
    exact: E[ℓᵢ] = pᵢ E[ℓᵢ | xᵢ ≠ 0] + (1 − pᵢ) E[ℓᵢ | xᵢ = 0]. Both conditional terms are
    estimated on ``n_samples`` draws of the other features, with xᵢ forced to a fresh U[low,
    high] value or to 0.

    The model must expose ``n_features``, ``n_inst`` and ``target(x)``; features are
    independent with probability ``feature_probability`` per instance.

    With ``shared_samples`` every instance sees the *same* inputs (common random numbers), so
    differences between instances, e.g. variants of one model with a neuron ablated
    (:mod:`superposition.interp`), carry no sampling noise from the inputs themselves. It
    requires one feature probability shared by all instances.
    """
    raise NotImplementedError
