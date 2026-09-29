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

from superposition.data import IndependentSparse

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
    return einops.einsum(W, W, "inst hidden f1, inst hidden f2 -> inst f1 f2")


def feature_norms(W: Float[Tensor, "inst hidden feat"]) -> Float[Tensor, "inst feat"]:
    """‖Wᵢ‖: about 1 if feature i is represented, about 0 if not. Paper: p. 11."""
    return torch.linalg.vector_norm(W, dim=1)


def _unit_columns(W: Float[Tensor, "inst hidden feat"]) -> Float[Tensor, "inst hidden feat"]:
    """Ŵᵢ = Wᵢ/‖Wᵢ‖, with zero columns left at zero."""
    return W / feature_norms(W).clamp_min(_ZERO_NORM)[:, None, :]


def cosine_similarity_matrix(
    W: Float[Tensor, "inst hidden feat"],
) -> Float[Tensor, "inst feat feat"]:
    """cos(Wᵢ, Wⱼ) = Ŵᵢ·Ŵⱼ; rows/columns of zero features are zero."""
    unit = _unit_columns(W)
    return einops.einsum(unit, unit, "inst hidden f1, inst hidden f2 -> inst f1 f2")


def _projection_squares(W: Float[Tensor, "inst hidden feat"]) -> Float[Tensor, "inst feat feat"]:
    """Entry (i, j) is (Ŵᵢ·Wⱼ)², the squared projection of Wⱼ onto feature i's direction."""
    unit = _unit_columns(W)
    return einops.einsum(unit, W, "inst hidden fi, inst hidden fj -> inst fi fj") ** 2


def superposition_score(W: Float[Tensor, "inst hidden feat"]) -> Float[Tensor, "inst feat"]:
    """Σ_{j≠i} (Ŵᵢ·Wⱼ)²: how strongly other features activate feature i's direction.

    Paper: Basic Results, p. 11. It is 0 when feature i is orthogonal to every other feature,
    and ≥ 1 when other features can activate Wᵢ as strongly as feature i itself. Zero features
    get 0.
    """
    squares = _projection_squares(W)
    return squares.sum(dim=-1) - torch.diagonal(squares, dim1=-2, dim2=-1)


def feature_dimensionality(W: Float[Tensor, "inst hidden feat"]) -> Float[Tensor, "inst feat"]:
    """Dᵢ = ‖Wᵢ‖² / Σⱼ (Ŵᵢ·Wⱼ)², the fraction of a dimension feature i gets.

    Paper: Feature Dimensionality, p. 18. The sum includes j = i, whose term is ‖Wᵢ‖², so
    Dᵢ ≤ 1. Dᵢ = 1/2 for an antipodal pair, 0 for a feature that is not learned. We define
    Dᵢ = 0 when Wᵢ = 0 (SPEC §3); the formula tends to 0 continuously as ‖Wᵢ‖ → 0 whenever any
    other feature is represented.
    """
    squares = _projection_squares(W)
    numerator = feature_norms(W) ** 2
    denominator = squares.sum(dim=-1)
    return torch.where(
        denominator > 0, numerator / denominator.clamp_min(_ZERO_NORM), torch.zeros_like(numerator)
    )


def dims_per_feature(W: Float[Tensor, "inst hidden feat"]) -> Float[Tensor, " inst"]:
    """D* = m / ‖W‖²_F, the "dimensions per feature". Paper: Uniform Superposition, p. 17.

    ‖Wᵢ‖² ≈ 1 for learned features and ≈ 0 otherwise, so ‖W‖²_F counts learned features.
    """
    n_hidden = W.shape[1]
    frob_sq = (W**2).sum(dim=(1, 2))
    return n_hidden / frob_sq.clamp_min(_ZERO_NORM)


def represented_mask(
    W: Float[Tensor, "inst hidden feat"], threshold: float = 0.5
) -> Bool[Tensor, "inst feat"]:
    """Features with ‖Wᵢ‖ > ``threshold`` count as represented (SPEC §3 default 0.5)."""
    return feature_norms(W) > threshold


def is_superposition(
    W: Float[Tensor, "inst hidden feat"], threshold: float = 0.5, rtol: float = 1e-2
) -> Bool[Tensor, " inst"]:
    """Formal superposition test: WᵀW is not invertible on the represented features.

    Paper: Summary: A Hierarchy of Feature Properties, p. 8. Restricted to the represented
    set R, W_Rᵀ W_R is invertible iff rank(W_R) = |R|. Numerically, a singular value counts
    toward the rank when it exceeds ``rtol`` times the largest singular value. With |R| > m the
    answer is always True; with |R| ≤ m it is True only for (near-)linearly dependent features.
    """
    mask = represented_mask(W, threshold)
    masked = W * mask[:, None, :]
    rank = torch.linalg.matrix_rank(masked, rtol=rtol)
    return rank < mask.sum(dim=-1)


def angles_2d(W: Float[Tensor, "inst 2 feat"]) -> Float[Tensor, "inst feat"]:
    """Angle of each feature direction in a 2D hidden space, in radians in (−π, π]."""
    if W.shape[1] != 2:
        raise ValueError(f"angles_2d needs a 2D hidden space, got hidden={W.shape[1]}")
    return torch.atan2(W[:, 1, :], W[:, 0, :])


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
    if W.shape[1] != 2:
        raise ValueError(f"canonicalize_2d needs a 2D hidden space, got hidden={W.shape[1]}")
    n_features = W.shape[-1]
    mask = represented_mask(W, threshold)
    angles = angles_2d(W)
    # First represented feature (argmax returns the first True; 0 if none is True).
    anchor = mask.int().argmax(dim=-1)
    theta = torch.gather(angles, 1, anchor[:, None])[:, 0]
    rotation = _rotation_matrices(math.pi / 2 - theta)
    rotated = einops.einsum(rotation, W, "inst i j, inst j feat -> inst i feat")
    # Reflection: priority = index for represented features, index + n for the others,
    # +inf for features on the vertical axis (including the anchor itself).
    x = rotated[:, 0, :]
    tol = 1e-6 * feature_norms(W).amax(dim=-1, keepdim=True).clamp_min(_ZERO_NORM)
    index = torch.arange(n_features, device=W.device).expand_as(x).to(W.dtype)
    priority = torch.where(mask, index, index + n_features)
    priority = torch.where(x.abs() > tol, priority, torch.full_like(priority, math.inf))
    deciding = priority.argmin(dim=-1)
    deciding_x = torch.gather(x, 1, deciding[:, None])[:, 0]
    flip = torch.isfinite(priority.amin(dim=-1)) & (deciding_x > 0)
    sign = torch.where(flip, -1.0, 1.0).to(W.dtype)
    return torch.stack([x * sign[:, None], rotated[:, 1, :]], dim=1)


def _rotation_matrices(theta: Float[Tensor, " inst"]) -> Float[Tensor, "inst 2 2"]:
    cos, sin = torch.cos(theta), torch.sin(theta)
    return torch.stack([torch.stack([cos, -sin], -1), torch.stack([sin, cos], -1)], -2)


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
    if W.ndim != 2:
        raise ValueError(f"geometry_graph takes one instance (hidden, feat), got {tuple(W.shape)}")
    batched = W[None]
    norms = feature_norms(batched)[0].detach().cpu().numpy()
    dims = feature_dimensionality(batched)[0].detach().cpu().numpy()
    represented = represented_mask(batched, represented_threshold)[0].cpu().numpy()
    dots = np.abs(gram(batched)[0].detach().cpu().numpy())
    graph = nx.Graph()
    for i in range(W.shape[1]):
        graph.add_node(
            i, norm=float(norms[i]), dimensionality=float(dims[i]), represented=bool(represented[i])
        )
    rep = np.flatnonzero(represented)
    for a, i in enumerate(rep):
        for j in rep[a + 1 :]:
            if dots[i, j] > edge_threshold:
                graph.add_edge(int(i), int(j), weight=float(dots[i, j]))
    components = [set(c) for c in nx.connected_components(graph.subgraph(rep.tolist()))]
    components.sort(key=lambda c: (-len(c), min(c)))
    return graph, components


def classify_fraction(D: Float[Tensor, ...], tol: float = 0.02) -> Int[Tensor, ...]:
    """Index of the nearest characteristic fraction within ``tol``, else −1.

    Paper: Feature Dimensionality, p. 19. Indices refer to :data:`CHARACTERISTIC_FRACTIONS`
    (1, 3/4, 2/3, 1/2, 2/5, 3/8, 0).
    """
    values = torch.tensor([v for _, v in CHARACTERISTIC_FRACTIONS], dtype=D.dtype, device=D.device)
    distance = (D[..., None] - values).abs()
    nearest = distance.argmin(dim=-1)
    within = distance.min(dim=-1).values <= tol
    return torch.where(within, nearest, torch.full_like(nearest, -1))


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
    index = sorted(component)
    block = W[:, index]
    singular = torch.linalg.svdvals(block.double())
    rank = int((singular > rank_rtol * singular.max()).sum()) if singular.numel() else 0
    k = len(index)
    dims = feature_dimensionality(block[None])[0]
    uniform = bool(((dims - rank / k).abs() <= dimensionality_tol).all()) if rank else False
    if uniform and (k, rank) in POLYTOPES:
        return k, rank, POLYTOPES[(k, rank)]
    return k, rank, f"{k} in {rank} ({'uniform' if uniform else 'non-uniform'})"


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
    _, components = geometry_graph(W, edge_threshold, represented_threshold)
    census: dict[str, int] = {}
    for component in components:
        k, _, name = component_geometry(W, component, rank_rtol, dimensionality_tol)
        census[name] = census.get(name, 0) + k
    return census


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
    magnitude = W.abs()
    largest = magnitude.amax(dim=-1, keepdim=True)
    counts = (magnitude >= rel_threshold * largest).sum(dim=-1)
    return torch.where(largest[..., 0] >= abs_threshold, counts, torch.zeros_like(counts))


def normalize_hidden_units(
    W1: Float[Tensor, "inst hidden feat"], W2: Float[Tensor, "inst feat hidden"]
) -> tuple[Float[Tensor, "inst hidden feat"], Float[Tensor, "inst feat hidden"]]:
    """Rescale every hidden unit so its largest-magnitude W₁ weight has magnitude 1.

    Paper: footnote 15 (p. 56). Scaling neuron h's row of W₁ by α > 0 and its column of W₂ by
    α⁻¹ leaves ReLU(W₂ ReLU(W₁x) + b) unchanged (ReLU is positively homogeneous), so this only
    fixes a degree of freedom for visualization and comparison. Dead units (all-zero rows) are
    left at zero.
    """
    scale = W1.abs().amax(dim=-1).clamp_min(_ZERO_NORM)  # (inst, hidden)
    return W1 / scale[..., None], W2 * scale[:, None, :]


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
    n_features = int(model.n_features)  # type: ignore[arg-type]
    n_inst = int(model.n_inst)  # type: ignore[arg-type]
    if not 0 <= k <= n_features:
        raise ValueError(f"k must be in [0, {n_features}], got {k}")
    target_fn: Callable[[Tensor], Tensor] = model.target  # type: ignore[assignment]
    device = generator.device
    shape = (n_samples, n_inst, n_features)
    # A uniform random k-subset: the indices of the k largest of n i.i.d. uniforms.
    keys = torch.rand(shape, generator=generator, device=device)
    active = torch.zeros(shape, dtype=torch.bool, device=device)
    if k > 0:
        active.scatter_(-1, keys.topk(k, dim=-1).indices, True)
    values = low + (high - low) * torch.rand(shape, generator=generator, device=device)
    x = torch.where(active, values, torch.zeros((), device=device))
    with torch.no_grad():
        out = model(x)
        per_sample = (importance.to(device) * (target_fn(x) - out) ** 2).sum(dim=-1)
    return math.comb(n_features, k) * per_sample.mean(dim=0)


@torch.no_grad()
def per_feature_loss(
    model: torch.nn.Module,
    feature_probability: Float[Tensor, " inst"],
    n_samples: int,
    generator: torch.Generator,
    low: float = 0.0,
    high: float = 1.0,
    features_per_chunk: int = 8,
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
    """
    n_features = int(model.n_features)  # type: ignore[arg-type]
    n_inst = int(model.n_inst)  # type: ignore[arg-type]
    device = generator.device
    target_fn: Callable[[Tensor], Tensor] = model.target  # type: ignore[assignment]
    dist = IndependentSparse(feature_probability, n_inst, n_features, low, high)
    p = dist.feature_probability.to(device)
    active = torch.zeros(n_inst, n_features, device=device)
    inactive = torch.zeros(n_inst, n_features, device=device)
    for start in range(0, n_features, features_per_chunk):
        chunk = torch.arange(start, min(start + features_per_chunk, n_features), device=device)
        k = len(chunk)
        x = dist.sample(n_samples, generator)[:, None].expand(-1, k, -1, -1).clone()
        forced = low + (high - low) * torch.rand(
            (n_samples, k, n_inst), generator=generator, device=device
        )
        onehot = F.one_hot(chunk, n_features).bool()[None, :, None, :]  # (1, k, 1, feat)
        for value, out_sum in ((forced[..., None], active), (torch.zeros(()), inactive)):
            xs = torch.where(onehot, value.to(device), x)
            flat = xs.reshape(n_samples * k, n_inst, n_features)
            err = (target_fn(flat) - model(flat)) ** 2
            err = err.reshape(n_samples, k, n_inst, n_features)
            own = err[:, torch.arange(k), :, chunk]  # (k, batch, inst)
            out_sum[:, chunk] = own.mean(dim=1).T
    return p * active + (1 - p) * inactive
