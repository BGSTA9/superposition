"""Closed-form and analytic results about the toy models.

Paper: Mathematical Understanding, pp. 12–14 (linear model loss; sparsity decomposition; the
1-sparse loss term); Superposition as a Phase Change, pp. 14–16 (theoretical phase diagrams);
Comments & Replications, pp. 51–53 (McGrath's exact n = 2, m = 1 analysis and the "confused
feature" solution).

Everything here is exact up to floating-point rounding; nothing is sampled. The Monte-Carlo
estimates in ``train.evaluate_loss`` and ``metrics.loss_by_num_active`` validate it in the
unit tests.

**The integral behind every ReLU-output term.** With independent features that are 0 with
probability 1 − p and U[0, 1] otherwise, every loss term of an output that depends on at most
one other feature is a mixture, over which of the two features are active, of

    F(γ, α, β, c) = ∫₀¹∫₀¹ (γx − ReLU(αx + βy + c))² dy dx,

where x is the output's own feature (γ = 1 if it is active, else 0) and y the other one.
:func:`relu_square_integral` computes F in closed form: it integrates y analytically, which
leaves a piecewise cubic in x whose breakpoints are known, and integrates each piece with a
3-point Gauss–Legendre rule, which is exact for polynomials of degree ≤ 5. See
docs/DECISIONS.md (D-301).

**Other output activations (E12, an extension).** Redwood found that the phase diagrams "look
quite different depending on the activation function" (p. 51). For an output x′ = σ(WᵀWx + b)
with any registered σ, :func:`activation_square_integral` computes F_σ, the same integral with σ
in place of the ReLU, by piecewise Gauss–Legendre quadrature split where the argument of σ
crosses 0 (docs/DECISIONS.md D-601). It is exact for the ReLU and the leaky ReLU, whose pieces
are polynomials. For the smooth activations, 24 nodes per piece are accurate to better than
1e-9 across the box E12 searches (tested against adaptive quadrature); E12 searches with 12,
whose error reaches about 1e-5 only at the box's far corners, and re-evaluates every optimum
it finds with 24.

**Sign convention.** The phase-change section writes ReLU(WᵀWx − b) (p. 14). We use +b
everywhere (SPEC §3), so the paper's b is our −b; no result depends on it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import einops
import networkx as nx
import numpy as np
import torch
from jaxtyping import Float
from numpy.typing import ArrayLike, NDArray
from torch import Tensor

FloatArray = NDArray[np.float64]

# 3-point Gauss–Legendre rule on [−1, 1]: exact for polynomials of degree ≤ 5.
_GL_NODES = np.array([-math.sqrt(3 / 5), 0.0, math.sqrt(3 / 5)])
_GL_WEIGHTS = np.array([5 / 9, 8 / 9, 5 / 9])

# Entries of a family template below this magnitude count as exactly zero.
_TEMPLATE_TOL = 1e-12

# Gauss–Legendre nodes per piece for non-ReLU activations (D-601; accuracy tested in
# tests/unit/test_theory.py).
DEFAULT_QUADRATURE_NODES = 16


def uniform_feature_variance(feature_probability: float | Tensor) -> float | Tensor:
    """Variance of one feature: p/3 − p²/4.

    With xᵢ = 0 w.p. 1 − p and U[0, 1] otherwise: E[xᵢ] = p/2, E[xᵢ²] = p/3.
    """
    p = feature_probability
    return p / 3 - p**2 / 4


def linear_optimum_loss(
    importance: Float[Tensor, " feat"], n_hidden: int, feature_probability: float
) -> float:
    """Exact minimum expected loss of the linear model x′ = WᵀWx + b.

    Paper: Mathematical Understanding, p. 13 (the linear model never benefits from
    superposition; footnote 12). With independent features of equal variance v = p/3 − p²/4 the
    optimum represents the m most important features perfectly (WᵀW = projection onto them)
    and predicts the mean p/2 through b for the rest, so the loss is

        L* = Σ_{i ≥ m} Iᵢ (p/3 − p²/4),

    where features are sorted by decreasing importance. Proof sketch: after the optimal bias
    b = (I − WᵀW)μ the loss is v · ‖D^{1/2}(I − WᵀW)‖²_F with D = diag(I); D^{1/2}WᵀW has rank
    ≤ m, and the best rank-m approximation of the diagonal D^{1/2} keeps its m largest entries.

    Returns the per-sample importance-weighted squared error summed over features (the value
    :func:`superposition.train.evaluate_loss` estimates).
    """
    sorted_importance, _ = torch.sort(importance.double(), descending=True)
    variance = float(uniform_feature_variance(feature_probability))
    return float(sorted_importance[n_hidden:].sum()) * variance


# --- the linear model: feature benefit vs interference (p. 13) -------------------------------


@dataclass
class LinearLossTerms:
    """The exact expected loss of x′ = WᵀWx + b, split as on p. 13. All tensors are per instance.

    Attributes:
        feature_benefit: Σᵢ Iᵢ vᵢ (1 − ‖Wᵢ‖²)², the loss from not representing features fully.
        interference: Σᵢ Σ_{j≠i} Iᵢ vⱼ (Wᵢ·Wⱼ)², the loss from non-orthogonal features.
        bias: Σᵢ Iᵢ (μᵢ − Σⱼ(WᵀW)ᵢⱼμⱼ − bᵢ)², zero when b predicts the mean residual.
        interference_matrix: ``(inst, feat, feat)``, entry (i, j) = Iᵢ vⱼ (Wᵢ·Wⱼ)², i ≠ j
            (the interference feature j causes in output i).
    """

    feature_benefit: Float[Tensor, " inst"]
    interference: Float[Tensor, " inst"]
    bias: Float[Tensor, " inst"]
    interference_matrix: Float[Tensor, "inst feat feat"]

    @property
    def total(self) -> Float[Tensor, " inst"]:
        return self.feature_benefit + self.interference + self.bias


def _per_feature_probability(
    feature_probability: float | Tensor, n_inst: int, n_features: int
) -> Float[Tensor, "inst feat"]:
    p = torch.as_tensor(feature_probability, dtype=torch.float64)
    if p.ndim == 1 and p.shape[0] == n_inst and n_inst != n_features:
        p = p[:, None]
    return torch.broadcast_to(p, (n_inst, n_features))


def linear_loss_decomposition(
    W: Float[Tensor, "inst hidden feat"],
    b: Float[Tensor, "inst feat"],
    importance: Float[Tensor, "*inst feat"],
    feature_probability: float | Tensor,
) -> LinearLossTerms:
    """Exact expected loss of the linear model, as feature benefit + interference + bias.

    Paper: Mathematical Understanding, p. 13:

        L ~ Σᵢ Iᵢ (1 − ‖Wᵢ‖²)²  +  Σ_{i≠j} Iⱼ (Wⱼ·Wᵢ)².

    The paper states it for Gaussian inputs of unit variance (footnote 10). For our data the
    error of output i is eᵢ = Σⱼ (δᵢⱼ − (WᵀW)ᵢⱼ) xⱼ − bᵢ, and because the xⱼ are independent
    with variance vⱼ = pⱼ/3 − pⱼ²/4 and mean μⱼ = pⱼ/2,

        E[eᵢ²] = vᵢ (1 − ‖Wᵢ‖²)² + Σ_{j≠i} vⱼ (Wᵢ·Wⱼ)² + (μᵢ − Σⱼ (WᵀW)ᵢⱼ μⱼ − bᵢ)².

    So the paper's two forces appear exactly, each weighted by a feature variance, plus a bias
    term that the optimal b sets to zero. The interference sum is the paper's after swapping
    the names i and j.

    Args:
        W: ``(inst, hidden, feat)``; b: ``(inst, feat)``.
        importance: Iᵢ, ``(feat,)`` or ``(inst, feat)``.
        feature_probability: p = 1 − S, a scalar, ``(inst,)``, ``(feat,)`` or ``(inst, feat)``.
    """
    W = W.double()
    n_inst, _, n_feat = W.shape
    importance = torch.broadcast_to(importance.double(), (n_inst, n_feat))
    p = _per_feature_probability(feature_probability, n_inst, n_feat)
    variance = p / 3 - p**2 / 4
    mean = p / 2
    gram = einops.einsum(W, W, "inst hidden f1, inst hidden f2 -> inst f1 f2")
    norms_sq = torch.diagonal(gram, dim1=-2, dim2=-1)
    benefit = (importance * variance * (1 - norms_sq) ** 2).sum(-1)
    off_diagonal = gram * (1 - torch.eye(n_feat, dtype=gram.dtype))
    matrix = importance[:, :, None] * variance[:, None, :] * off_diagonal**2
    residual_mean = mean - einops.einsum(gram, mean, "inst i j, inst j -> inst i") - b.double()
    bias = (importance * residual_mean**2).sum(-1)
    return LinearLossTerms(benefit, matrix.sum(dim=(-2, -1)), bias, matrix)


# --- the exact integral F and the 1-sparse decomposition (p. 13) -----------------------------


def _positive_interval(slope: FloatArray, intercept: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Endpoints of {t ∈ [0, 1] : slope·t + intercept > 0} (an interval, possibly empty)."""
    safe = np.where(slope == 0, 1.0, slope)
    root = np.clip(-intercept / safe, 0.0, 1.0)
    positive = (intercept > 0).astype(np.float64)
    lo = np.where(slope > 0, root, 0.0)
    hi = np.where(slope > 0, 1.0, np.where(slope < 0, root, positive))
    return lo, hi


def relu_square_integral(own: ArrayLike, a: ArrayLike, beta: ArrayLike, c: ArrayLike) -> FloatArray:
    """F(γ, α, β, c) = ∫₀¹∫₀¹ (γx − ReLU(αx + βy + c))² dy dx, exactly, elementwise.

    Paper: Mathematical Understanding, p. 13. Special cases give every term the paper writes:
    the 1-sparse feature benefit ∫(x − ReLU(‖Wᵢ‖²x + bᵢ))² is F(1, ‖Wᵢ‖², 0, bᵢ); the 1-sparse
    interference ∫ReLU(Wⱼ·Wᵢ x + bⱼ)² is F(0, 0, Wⱼ·Wᵢ, bⱼ); the zero-input term ReLU(b)² is
    F(0, 0, 0, b).

    Method: F = γ²/3 − ∫∫_{u>0} (2γxu − u²) with u = αx + βy + c. For fixed x the set
    {y ∈ [0, 1] : u > 0} is an interval with endpoints clip(−(αx + c)/β, 0, 1), so the inner
    integral is a polynomial in the endpoints: a cubic in x between the breakpoints where
    αx + c = 0 or αx + c = −β. Each of the (at most three) pieces is integrated with a 3-point
    Gauss–Legendre rule, which is exact for that degree.

    All arguments broadcast; the result is a float64 array of their broadcast shape.
    """
    own, a, beta, c = np.broadcast_arrays(
        *(np.asarray(v, dtype=np.float64) for v in (own, a, beta, c))
    )
    # Breakpoints of the piecewise-cubic inner integral, clipped into [0, 1].
    safe_a = np.where(a == 0, 1.0, a)
    x1 = np.where(a == 0, 0.0, -c / safe_a)
    x2 = np.where(a == 0, 0.0, -(c + beta) / safe_a)
    edges = np.sort(
        np.stack([np.zeros_like(a), np.clip(x1, 0, 1), np.clip(x2, 0, 1), np.ones_like(a)]),
        axis=0,
    )
    lo, hi = edges[:-1], edges[1:]  # (3 pieces, *shape)
    half = (hi - lo) / 2
    x = (lo + hi)[..., None] / 2 + half[..., None] * _GL_NODES  # (3, *shape, 3 nodes)
    own_x, a_x, beta_x, c_x = (v[None, ..., None] for v in (own, a, beta, c))
    t = a_x * x + c_x  # u = t + βy
    y_lo, y_hi = _positive_interval(np.broadcast_to(beta_x, t.shape), t)

    def antiderivative(y: FloatArray) -> FloatArray:
        # ∫ (2γx·u − u²) dy with u = t + βy, expanded so β = 0 needs no special case.
        return 2 * own_x * x * (t * y + beta_x * y**2 / 2) - (
            t**2 * y + t * beta_x * y**2 + beta_x**2 * y**3 / 3
        )

    inner = antiderivative(y_hi) - antiderivative(y_lo)
    positive_part = (half * (inner * _GL_WEIGHTS).sum(-1)).sum(0)
    return own**2 / 3 - positive_part


def relu_square_1d(beta: ArrayLike, c: ArrayLike) -> FloatArray:
    """∫₀¹ ReLU(βy + c)² dy in closed form: F(0, 0, β, c), the 1-sparse interference term."""
    beta, c = np.broadcast_arrays(np.asarray(beta, np.float64), np.asarray(c, np.float64))
    lo, hi = _positive_interval(beta, c)

    def antiderivative(y: FloatArray) -> FloatArray:  # ∫ (c + βy)² dy, expanded
        return c**2 * y + c * beta * y**2 + beta**2 * y**3 / 3

    return antiderivative(hi) - antiderivative(lo)


def own_error_1d(a: ArrayLike, c: ArrayLike) -> FloatArray:
    """∫₀¹ (x − ReLU(ax + c))² dx in closed form: F(1, a, 0, c), the 1-sparse benefit term."""
    a, c = np.broadcast_arrays(np.asarray(a, np.float64), np.asarray(c, np.float64))
    lo, hi = _positive_interval(a, c)

    def antiderivative(x: FloatArray) -> FloatArray:  # ∫ (2x·u − u²) dx with u = ax + c
        return (2 * a - a**2) * x**3 / 3 + (c - a * c) * x**2 - c**2 * x

    return 1 / 3 - (antiderivative(hi) - antiderivative(lo))


def pair_output_loss(
    p_own: ArrayLike,
    p_other: ArrayLike,
    a: ArrayLike,
    beta: ArrayLike,
    c: ArrayLike,
    activation: str = "relu",
    nodes: int = DEFAULT_QUADRATURE_NODES,
) -> FloatArray:
    """E[(xᵢ − σ(a xᵢ + β xⱼ + c))²] for independent sparse xᵢ, xⱼ, exactly (elementwise).

    The expected squared error of one output of the ReLU output model that reads its own
    feature with coefficient a = ‖Wᵢ‖² and one other feature with β = Wᵢ·Wⱼ (Paper: p. 13,
    "the integral over x decomposes into a term for each sparsity pattern"). The four sparsity
    patterns of (xᵢ, xⱼ) are weighted by their probabilities: neither active (ReLU(c)²), only
    the own feature, only the other one, and both (the 2D integral F(1, a, β, c)).

    ``activation`` names the output nonlinearity σ (``superposition.models.ACTIVATIONS``). The
    ReLU uses the closed forms above; any other σ uses :func:`activation_square_integral` and
    its 1D counterparts with ``nodes`` Gauss–Legendre nodes per piece (E12, p. 51).
    """
    p_own, p_other = np.asarray(p_own, dtype=np.float64), np.asarray(p_other, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    if activation != "relu":
        return _activation_pair_output_loss(p_own, p_other, a, beta, c, activation, nodes)
    return (
        (1 - p_own) * (1 - p_other) * np.maximum(c, 0.0) ** 2
        + p_own * (1 - p_other) * own_error_1d(a, c)
        + (1 - p_own) * p_other * relu_square_1d(beta, c)
        + p_own * p_other * relu_square_integral(1.0, a, beta, c)
    )


# --- other output activations (E12; Paper: Comments & Replications, p. 51) -------------------


def _gauss_legendre_unit(nodes: int) -> tuple[Tensor, Tensor]:
    """Gauss–Legendre nodes and weights on [0, 1], float64."""
    x, w = np.polynomial.legendre.leggauss(nodes)
    return torch.from_numpy((x + 1) / 2), torch.from_numpy(w / 2)


def _pieces(lo: Tensor, hi: Tensor, nodes: int) -> tuple[Tensor, Tensor]:
    """Quadrature points and weights ``(*lo.shape, nodes)`` on the intervals [lo, hi]."""
    t, w = _gauss_legendre_unit(nodes)
    width = (hi - lo)[..., None]
    return lo[..., None] + width * t, width * w


def _root(slope: Tensor, intercept: Tensor) -> Tensor:
    """Where slope·t + intercept = 0, clipped into [0, 1] (0 when slope = 0: no split)."""
    safe = torch.where(slope == 0, torch.ones_like(slope), slope)
    return torch.where(slope == 0, torch.zeros_like(slope), (-intercept / safe).clamp(0.0, 1.0))


def _as_flat_tensors(*values: ArrayLike) -> tuple[tuple[int, ...], list[Tensor]]:
    arrays = np.broadcast_arrays(*(np.asarray(v, dtype=np.float64) for v in values))
    return arrays[0].shape, [torch.from_numpy(np.ascontiguousarray(a).reshape(-1)) for a in arrays]


def activation_square_integral(
    activation: str,
    own: ArrayLike,
    a: ArrayLike,
    beta: ArrayLike,
    c: ArrayLike,
    nodes: int = DEFAULT_QUADRATURE_NODES,
    chunk: int = 8192,
) -> FloatArray:
    """F_σ(γ, α, β, c) = ∫₀¹∫₀¹ (γx − σ(αx + βy + c))² dy dx, elementwise.

    Paper: Comments & Replications, p. 51 (Redwood: the phase diagrams depend on the
    activation). The generalization of :func:`relu_square_integral` to any registered
    activation σ (``superposition.models.ACTIVATIONS``, evaluated in float64 with the same
    torch function the model trains with).

    Method (docs/DECISIONS.md D-601): split x ∈ [0, 1] where αx + c = 0 or αx + c + β = 0,
    and, for each x, split y ∈ [0, 1] where αx + βy + c = 0; integrate every piece with an
    ``nodes``-point Gauss–Legendre rule. On each piece σ's argument keeps one sign, so the
    integrand is a polynomial for the ReLU and leaky ReLU (exact) and smooth for the others.

    All arguments broadcast; the result is a float64 array of their broadcast shape.
    """
    from superposition.models import get_activation

    sigma = get_activation(activation)
    shape, (own_t, a_t, beta_t, c_t) = _as_flat_tensors(own, a, beta, c)
    out = torch.empty_like(a_t)
    for start in range(0, a_t.numel(), chunk):
        part = slice(start, start + chunk)
        g, al, be, cc = own_t[part], a_t[part], beta_t[part], c_t[part]
        edges = torch.sort(
            torch.stack(
                [
                    torch.zeros_like(al),
                    _root(al, cc),
                    _root(al, cc + be),
                    torch.ones_like(al),
                ]
            ),
            dim=0,
        ).values
        x, wx = _pieces(edges[:-1], edges[1:], nodes)  # (3, k, q)
        t = al[:, None] * x + cc[:, None]
        split = _root(be[:, None].expand_as(t), t)
        y, wy = _pieces(
            torch.stack([torch.zeros_like(split), split]),
            torch.stack([split, torch.ones_like(split)]),
            nodes,
        )  # (2, 3, k, q, q)
        u = t[..., None] + be[:, None, None] * y
        error = (g[:, None, None] * x[..., None] - sigma(u)) ** 2
        inner = (error * wy).sum(-1).sum(0)  # (3, k, q)
        out[part] = (inner * wx).sum(-1).sum(0)
    return out.reshape(shape).numpy()


def activation_error_1d(
    activation: str,
    own: ArrayLike,
    a: ArrayLike,
    c: ArrayLike,
    nodes: int = DEFAULT_QUADRATURE_NODES,
) -> FloatArray:
    """∫₀¹ (γx − σ(ax + c))² dx, split where ax + c = 0 (elementwise).

    With γ = 1 it is the 1-sparse feature benefit term; with γ = 0 (and a = β) the 1-sparse
    interference term ∫σ(βy + c)² (p. 13, with σ in place of the ReLU).
    """
    from superposition.models import get_activation

    sigma = get_activation(activation)
    shape, (own_t, a_t, c_t) = _as_flat_tensors(own, a, c)
    split = _root(a_t, c_t)
    x, w = _pieces(
        torch.stack([torch.zeros_like(split), split]),
        torch.stack([split, torch.ones_like(split)]),
        nodes,
    )  # (2, k, q)
    error = (own_t[:, None] * x - sigma(a_t[:, None] * x + c_t[:, None])) ** 2
    return (error * w).sum(-1).sum(0).reshape(shape).numpy()


def _activation_pair_output_loss(
    p_own: FloatArray,
    p_other: FloatArray,
    a: ArrayLike,
    beta: ArrayLike,
    c: FloatArray,
    activation: str,
    nodes: int,
) -> FloatArray:
    """:func:`pair_output_loss` for a non-ReLU σ; skips the terms whose weight is 0."""
    from superposition.models import get_activation

    sigma = get_activation(activation)
    zero = sigma(torch.from_numpy(np.array(c, dtype=np.float64))).numpy() ** 2
    total = (1 - p_own) * (1 - p_other) * zero
    total = total + p_own * (1 - p_other) * activation_error_1d(activation, 1.0, a, c, nodes)
    if np.any(p_other != 0):
        total = total + (1 - p_own) * p_other * activation_error_1d(activation, 0.0, beta, c, nodes)
        total = total + p_own * p_other * activation_square_integral(
            activation, 1.0, a, beta, c, nodes
        )
    return np.asarray(total, dtype=np.float64)


@dataclass
class OneSparseTerms:
    """L₀ and L₁ of the ReLU output model (p. 13), per instance.

    Attributes:
        zero: L₀ = Σᵢ Iᵢ ReLU(bᵢ)², the loss on the all-zero input.
        benefit: ``(inst, feat)``, Iᵢ ∫₀¹ (x − ReLU(‖Wᵢ‖²x + bᵢ))² dx.
        interference: ``(inst, feat, feat)``, entry (j, i) = Iⱼ ∫₀¹ ReLU(Wⱼ·Wᵢ x + bⱼ)² dx for
            j ≠ i: the loss output j suffers when only feature i is active.
    """

    zero: Float[Tensor, " inst"]
    benefit: Float[Tensor, "inst feat"]
    interference: Float[Tensor, "inst feat feat"]

    @property
    def one(self) -> Float[Tensor, " inst"]:
        """L₁ = Σᵢ benefitᵢ + Σ_{i≠j} interference."""
        return self.benefit.sum(-1) + self.interference.sum(dim=(-2, -1))


def one_sparse_decomposition(
    W: Float[Tensor, "inst hidden feat"],
    b: Float[Tensor, "inst feat"],
    importance: Float[Tensor, "*inst feat"],
) -> OneSparseTerms:
    """The exact 0- and 1-sparse loss terms of the ReLU output model x′ = ReLU(WᵀWx + b).

    Paper: Mathematical Understanding, p. 13. The expected loss splits by the number k of
    active features, L = Σₖ pᵏ(1 − p)ⁿ⁻ᵏ Lₖ (for a shared p), with

        L₀ = Σᵢ Iᵢ ReLU(bᵢ)²,
        L₁ = Σᵢ ∫ Iᵢ (xᵢ − ReLU(‖Wᵢ‖² xᵢ + bᵢ))² dxᵢ + Σ_{i≠j} ∫ Iⱼ ReLU(Wⱼ·Wᵢ xᵢ + bⱼ)² dxᵢ.

    Lₖ here is the sum over active sets of the conditional expected loss, the convention of
    :func:`superposition.metrics.loss_by_num_active`. The first sum is the feature benefit
    (the ReLU never makes a feature's own readout worse); the second is interference, which
    is free whenever Wⱼ·Wᵢ x + bⱼ ≤ 0: that is why negative interference and negative biases
    help (p. 13).
    """
    W64 = W.detach().double()
    n_inst, _, n_feat = W64.shape
    importance = torch.broadcast_to(importance.detach().double(), (n_inst, n_feat))
    gram = einops.einsum(W64, W64, "inst hidden f1, inst hidden f2 -> inst f1 f2").numpy()
    bias = b.detach().double().numpy()
    imp = importance.numpy()
    diag = np.diagonal(gram, axis1=-2, axis2=-1)
    zero = (imp * np.maximum(bias, 0.0) ** 2).sum(-1)
    benefit = imp * own_error_1d(diag, bias)
    # Output j (row) when only feature i (column) is active: ∫ ReLU(gram[j, i] x + b_j)².
    interference = imp[:, :, None] * relu_square_1d(gram, bias[:, :, None])
    interference *= 1 - np.eye(n_feat)
    return OneSparseTerms(
        torch.from_numpy(zero), torch.from_numpy(benefit), torch.from_numpy(interference)
    )


def k_sparse_weights(n_features: int, feature_probability: float) -> list[float]:
    """pᵏ(1 − p)ⁿ⁻ᵏ for k = 0 … n: the weight of Lₖ in L = Σₖ pᵏ(1 − p)ⁿ⁻ᵏ Lₖ (p. 13).

    (The paper writes (1 − S)ᵏ Sⁿ⁻ᵏ.) As S → 1 the k = 0 and k = 1 weights dominate.
    """
    p = feature_probability
    return [p**k * (1 - p) ** (n_features - k) for k in range(n_features + 1)]


def relu_output_expected_loss(
    W: Float[Tensor, "hidden feat"] | FloatArray,
    b: Float[Tensor, " feat"] | FloatArray,
    importance: ArrayLike,
    feature_probability: ArrayLike,
    activation: str = "relu",
) -> float:
    """Exact expected loss of one ReLU output model x′ = ReLU(WᵀWx + b).

    Paper: p. 13 (sum over sparsity patterns). Exact when every output reads at most one other
    feature (each row of WᵀW has at most one nonzero off-diagonal entry), which covers every
    n = 2 model and every phase-diagram family of pp. 14–15. Other models raise
    ``NotImplementedError`` (they need integrals over three or more dimensions).

    Args:
        W: ``(hidden, feat)`` for one model.
        b: ``(feat,)``.
        importance: ``(feat,)``.
        feature_probability: scalar or ``(feat,)``.
        activation: the output nonlinearity; another name than ``"relu"`` gives the loss of
            x′ = σ(WᵀWx + b) by quadrature (E12, p. 51; :func:`pair_output_loss`).
    """
    W_np = np.asarray(W.detach().cpu() if isinstance(W, Tensor) else W, dtype=np.float64)
    b_np = np.asarray(b.detach().cpu() if isinstance(b, Tensor) else b, dtype=np.float64)
    gram = W_np.T @ W_np
    n = gram.shape[0]
    imp = np.broadcast_to(np.asarray(importance, dtype=np.float64), (n,))
    p = np.broadcast_to(np.asarray(feature_probability, dtype=np.float64), (n,))
    total = 0.0
    for i in range(n):
        others = [j for j in range(n) if j != i and gram[i, j] != 0.0]
        if len(others) > 1:
            raise NotImplementedError(
                f"output {i} reads {len(others)} other features; exact loss supports at most one"
            )
        j = others[0] if others else None
        beta = gram[i, j] if j is not None else 0.0
        p_other = p[j] if j is not None else 0.0
        total += imp[i] * float(
            pair_output_loss(p[i], p_other, gram[i, i], beta, b_np[i], activation)
        )
    return total


def tied_output_expected_losses(
    W: FloatArray,
    b: FloatArray,
    importance: ArrayLike,
    feature_probability: ArrayLike,
    activation: str = "relu",
    nodes: int = DEFAULT_QUADRATURE_NODES,
) -> FloatArray:
    """Exact expected loss of many models x′ = σ(WᵀWx + b) at once, one per row.

    The vectorized form of :func:`relu_output_expected_loss` for the family optima of a phase
    diagram (every output reads at most one other feature; entries of WᵀW below 1e-12 count
    as zero). E12 uses it to re-evaluate each family's optimum with more quadrature nodes.

    Args:
        W: ``(models, hidden, feat)``; b: ``(models, feat)``.
        importance, feature_probability: ``(models, feat)``, ``(models,)`` or scalars.
    """
    W = np.asarray(W, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n_models, _, n = W.shape
    gram = np.einsum("khi,khj->kij", W, W)
    imp = np.broadcast_to(np.asarray(importance, dtype=np.float64), (n_models, n))
    p = np.asarray(feature_probability, dtype=np.float64)
    p = np.broadcast_to(p[:, None] if p.ndim == 1 else p, (n_models, n))
    total = np.zeros(n_models)
    for i in range(n):
        off = gram[:, i, :].copy()
        off[:, i] = 0.0
        off[np.abs(off) < 1e-12] = 0.0
        if np.any((off != 0).sum(-1) > 1):
            raise NotImplementedError("an output reads two or more other features")
        j = np.abs(off).argmax(-1)
        beta = off[np.arange(n_models), j]
        p_other = np.where(beta != 0, p[np.arange(n_models), j], 0.0)
        total += imp[:, i] * pair_output_loss(
            p[:, i], p_other, gram[:, i, i], beta, b[:, i], activation, nodes
        )
    return total


# --- theoretical phase diagrams (pp. 14–16) --------------------------------------------------

NOT_LEARNED = "not_learned"
SUPERPOSITION = "superposition"
DEDICATED = "dedicated"
CONFUSED = "confused"
PHASE_LABELS: tuple[str, ...] = (NOT_LEARNED, SUPERPOSITION, DEDICATED, CONFUSED)


@dataclass(frozen=True)
class Family:
    """A family of weight configurations: "W ⊥ v", the direction W does not represent.

    Paper: pp. 14–15. The template is built by projecting the standard basis onto the
    orthogonal complement of ``ignored`` (the construction of p. 21) and normalizing every
    nonzero column to unit length. ``label`` is what the family means for the extra feature
    (the last one), in the terms of the phase diagram.
    """

    name: str
    ignored: tuple[float, ...]
    label: str
    description: str = ""

    def template(self) -> tuple[FloatArray, FloatArray]:
        """``(W0, G0)``: unit-column template W0 ``(n − 1, n)`` and its Gram matrix W0ᵀW0.

        Zero columns are the features the family does not represent.
        """
        v = np.asarray(self.ignored, dtype=np.float64)
        n = v.shape[0]
        v_hat = v / np.linalg.norm(v)
        projection = np.eye(n) - np.outer(v_hat, v_hat)
        norms = np.linalg.norm(projection, axis=0)
        directions = np.where(norms > _TEMPLATE_TOL, projection / np.maximum(norms, 1e-300), 0.0)
        # Coordinates in an orthonormal basis of the complement (the m = n − 1 hidden space).
        basis = np.linalg.svd(projection)[0][:, : n - 1]
        W0 = basis.T @ directions
        W0[np.abs(W0) < _TEMPLATE_TOL] = 0.0
        G0 = directions.T @ directions
        G0[np.abs(G0) < _TEMPLATE_TOL] = 0.0
        return W0, G0

    def factors(self) -> list[list[int]]:
        """Tegum factors of the template: connected components of represented features.

        Paper: p. 20 ("two subgraphs are disconnected if and only if they are in different
        tegum factors"). Features in different factors never interfere, so the family's loss
        is a sum over factors, each with its own weight scale.
        """
        _, G0 = self.template()
        represented = [i for i in range(G0.shape[0]) if G0[i, i] > 0.5]
        graph = nx.Graph()
        graph.add_nodes_from(represented)
        graph.add_edges_from(
            (i, j) for i in represented for j in represented if i < j and G0[i, j] != 0.0
        )
        return sorted((sorted(c) for c in nx.connected_components(graph)), key=lambda c: c[0])

    def unrepresented(self) -> list[int]:
        _, G0 = self.template()
        return [i for i in range(G0.shape[0]) if G0[i, i] <= 0.5]


# n = 2, m = 1 (p. 14). The extra feature is feature 1 (0-based); feature 0 has importance 1.
N2M1_FAMILIES: tuple[Family, ...] = (
    Family("W ⊥ [0, 1]", (0.0, 1.0), NOT_LEARNED, "W = [1, 0]: the extra feature is dropped"),
    Family("W ⊥ [1, 0]", (1.0, 0.0), DEDICATED, "W = [0, 1]: the extra feature gets the dimension"),
    Family("W ⊥ [1, 1]", (1.0, 1.0), SUPERPOSITION, "W = [1, −1]: an antipodal pair"),
)
# McGrath's "confused feature" solution W₁ ≈ W₂ (pp. 51–53): optional fourth family.
N2M1_CONFUSED = Family(
    "W ⊥ [1, −1]", (1.0, -1.0), CONFUSED, "W = [1, 1]: both features share one direction"
)

# n = 3, m = 2 (p. 15). The extra feature is feature 2; features 0 and 1 have importance 1.
N3M2_FAMILIES: tuple[Family, ...] = (
    Family("W ⊥ [0, 0, 1]", (0.0, 0.0, 1.0), NOT_LEARNED, "the extra feature is dropped"),
    Family(
        "W ⊥ [1, 0, 0]", (1.0, 0.0, 0.0), DEDICATED, "extra gets a dimension; another is dropped"
    ),
    Family(
        "W ⊥ [0, 1, 1]", (0.0, 1.0, 1.0), SUPERPOSITION, "extra forms an antipodal pair with one"
    ),
    Family(
        "W ⊥ [1, 1, 0]", (1.0, 1.0, 0.0), DEDICATED, "extra gets a dimension; the others are a pair"
    ),
)
# The n = 3 analogue of McGrath's solution: the extra feature shares a direction with one other.
N3M2_CONFUSED = Family(
    "W ⊥ [0, 1, −1]", (0.0, 1.0, -1.0), CONFUSED, "extra shares one direction with another"
)


@dataclass(frozen=True)
class OptimizerSettings:
    """Search settings for a family's free scalars (weight scale per factor, bias per feature).

    The search is a coarse grid followed by golden-section refinement, nested: for a fixed
    scale each output's bias is optimized on its own (outputs are separable given the scale),
    then the scale is optimized. ``scale_max`` and ``bias_range`` bound the coarse grids; the
    refinement brackets the best grid point. See docs/DECISIONS.md (D-302).

    ``fit_bias = False`` fixes every bias at 0: the setting of McGrath's exact analysis
    ("ignoring bias terms", p. 51), used to compare with the confused-feature regime.

    ``activation`` is the output nonlinearity (E12 varies it, p. 51); a non-ReLU activation's
    losses use quadrature with ``quadrature_nodes`` nodes per piece (D-601).
    """

    fit_bias: bool = True
    scale_max: float = 2.0
    scale_grid: int = 41
    bias_range: tuple[float, float] = (-1.5, 1.5)
    bias_grid: int = 121
    golden_iterations: int = 40
    activation: str = "relu"
    quadrature_nodes: int = DEFAULT_QUADRATURE_NODES


def _last_argmin(values: FloatArray) -> NDArray[np.int64]:
    """Argmin along the last axis, ties resolved to the *largest* index.

    A ReLU output is flat in its bias wherever it is always off (ReLU(b) = 0 for b ≤ 0), so
    the grid has exact ties there; the optimum, if not in the flat part, lies just to the
    right of the flat part's last grid point (e.g. b = p/2 for an unrepresented feature).
    """
    return values.shape[-1] - 1 - np.argmin(values[..., ::-1], axis=-1)


def _golden_minimize(
    fn: Callable[[FloatArray], FloatArray], lo: FloatArray, hi: FloatArray, iterations: int
) -> tuple[FloatArray, FloatArray]:
    """Vectorized golden-section search of ``fn`` on [lo, hi] (elementwise); (argmin, min)."""
    ratio = (math.sqrt(5) - 1) / 2
    a, b = lo.astype(np.float64), hi.astype(np.float64)
    c, d = b - ratio * (b - a), a + ratio * (b - a)
    fc, fd = fn(c), fn(d)
    for _ in range(iterations):
        left = fc < fd  # the minimum lies in [a, d]
        a, b = np.where(left, a, c), np.where(left, d, b)
        new_c = np.where(left, b - ratio * (b - a), d)
        new_d = np.where(left, c, a + ratio * (b - a))
        f_new = fn(np.where(left, new_c, new_d))
        fc, fd = np.where(left, f_new, fd), np.where(left, fc, f_new)
        c, d = new_c, new_d
    x = (a + b) / 2
    return x, fn(x)


@dataclass
class FactorSolution:
    """The optimum of one tegum factor, per cell: loss, weight scale and biases."""

    loss: FloatArray  # (cells,)
    scale: FloatArray  # (cells,)
    bias: FloatArray  # (cells, k)


def optimize_factor(
    G0: FloatArray,
    importance: FloatArray,
    feature_probability: FloatArray,
    settings: OptimizerSettings | None = None,
) -> FactorSolution:
    """Minimize a tegum factor's exact expected loss over its scale s and biases, per cell.

    The factor's Gram matrix is s²·G0 (G0 has a unit diagonal); output i then reads its own
    feature with coefficient s² and the (at most one) other feature j with s²·G0ᵢⱼ. Given s,
    each output's bias is optimized separately. The search evaluates a coarse (scale × bias)
    grid in one vectorized call, then refines the scale by golden section around the best
    grid point, re-optimizing the biases (grid + golden section) at every trial scale. Cells
    with identical inputs are solved once.

    Args:
        G0: ``(k, k)`` template Gram block of the factor (k = 1 or 2); all zeros for an
            unrepresented feature, whose output is ReLU(b).
        importance: ``(cells, k)``; feature_probability: ``(cells, k)``.
    """
    settings = settings or OptimizerSettings()
    k = G0.shape[0]
    if k > 2:
        raise NotImplementedError("factors of more than two features need 3D integrals")
    importance = np.asarray(importance, dtype=np.float64)
    p = np.asarray(feature_probability, dtype=np.float64)
    keys, inverse = np.unique(np.concatenate([importance, p], axis=1), axis=0, return_inverse=True)
    solution = _optimize_unique_factor(G0, keys[:, :k], keys[:, k:], settings)
    inverse = inverse.reshape(-1)
    return FactorSolution(
        loss=solution.loss[inverse], scale=solution.scale[inverse], bias=solution.bias[inverse]
    )


def _optimize_unique_factor(
    G0: FloatArray, importance: FloatArray, p: FloatArray, settings: OptimizerSettings
) -> FactorSolution:
    k = G0.shape[0]
    n_cells = importance.shape[0]
    partner = [next((j for j in range(k) if j != i and G0[i, j] != 0.0), None) for i in range(k)]

    def output_loss(i: int, scale: FloatArray, bias: FloatArray) -> FloatArray:
        """Output i's expected error; ``scale`` and ``bias`` broadcast against ``(cells, …)``."""
        j = partner[i]
        s2 = scale**2
        pad = (slice(None),) + (None,) * (np.broadcast(s2, bias).ndim - 1)
        beta = s2 * G0[i, j] if j is not None else np.zeros_like(s2)
        p_other = p[:, j][pad] if j is not None else 0.0
        return pair_output_loss(
            p[:, i][pad],
            p_other,
            s2 * G0[i, i],
            beta,
            bias,
            settings.activation,
            settings.quadrature_nodes,
        )

    bias_grid = np.linspace(*settings.bias_range, settings.bias_grid)
    bias_step = bias_grid[1] - bias_grid[0]

    def best_biases(scale: FloatArray) -> tuple[FloatArray, FloatArray]:
        """For per-cell scales ``(cells,)``: optimal biases ``(cells, k)`` and the loss."""
        biases, total = [], np.zeros(n_cells)
        for i in range(k):
            if not settings.fit_bias:
                biases.append(np.zeros(n_cells))
                total = total + importance[:, i] * output_loss(i, scale, np.zeros(n_cells))
                continue
            values = output_loss(i, scale[:, None], bias_grid[None, :])
            start = bias_grid[_last_argmin(values)]
            b_i, loss_i = _golden_minimize(
                lambda c, i=i: output_loss(i, scale, c),
                start - bias_step,
                start + bias_step,
                settings.golden_iterations,
            )
            biases.append(b_i)
            total = total + importance[:, i] * loss_i
        return np.stack(biases, axis=1), total

    if not np.any(G0):  # unrepresented features: the output is ReLU(b), no scale to search
        bias, loss = best_biases(np.zeros(n_cells))
        return FactorSolution(loss=loss, scale=np.zeros(n_cells), bias=bias)
    scale_grid = np.linspace(0.0, settings.scale_max, settings.scale_grid)
    trial_biases = bias_grid if settings.fit_bias else np.zeros(1)
    coarse = sum(
        importance[:, i, None]
        * output_loss(i, scale_grid[None, :, None], trial_biases[None, None, :]).min(axis=-1)
        for i in range(k)
    )  # (cells, scales): the grid optimum over biases at each grid scale
    best = scale_grid[_last_argmin(np.asarray(coarse))]
    step = scale_grid[1] - scale_grid[0]
    scale, _ = _golden_minimize(
        lambda s: best_biases(s)[1],
        np.maximum(best - step, 0.0),
        best + step,
        settings.golden_iterations,
    )
    bias, loss = best_biases(scale)
    return FactorSolution(loss=loss, scale=scale, bias=bias)


@dataclass
class FamilySolution:
    """A family's optimal loss per cell, with its weights (for figures and the lab)."""

    family: Family
    loss: FloatArray  # (cells,)
    W: FloatArray  # (cells, n − 1, n)
    b: FloatArray  # (cells, n)


def optimize_family(
    family: Family,
    importance: ArrayLike,
    feature_probability: ArrayLike,
    settings: OptimizerSettings | None = None,
) -> FamilySolution:
    """Minimize the exact expected loss of the ReLU output model over one family, per cell.

    Paper: pp. 14–15 ("closed form solutions for the loss of different weight configurations
    as a function of importance and sparsity"). The free scalars are one weight scale per
    tegum factor of the template and one bias per feature (docs/DECISIONS.md D-302). Tegum
    factors never interfere, so the loss is the sum of the factors' optima; an unrepresented
    feature's output is ReLU(bᵢ), optimized the same way (its optimum is the mean p/2).

    Args:
        family: the family.
        importance: ``(cells, n)`` or ``(n,)``.
        feature_probability: ``(cells, n)``, ``(cells,)`` (shared by all features) or a scalar.
    """
    W0, G0 = family.template()
    n = G0.shape[0]
    importance = np.atleast_2d(np.asarray(importance, dtype=np.float64))
    p = np.asarray(feature_probability, dtype=np.float64)
    n_cells = max(importance.shape[0], p.shape[0] if p.ndim else 1)
    importance = np.broadcast_to(importance, (n_cells, n))
    p = np.broadcast_to(p[:, None] if p.ndim == 1 else p, (n_cells, n))
    loss = np.zeros(n_cells)
    scales = np.zeros((n_cells, n))
    bias = np.zeros((n_cells, n))
    for factor in family.factors():
        solution = optimize_factor(
            G0[np.ix_(factor, factor)], importance[:, factor], p[:, factor], settings
        )
        loss += solution.loss
        scales[:, factor] = solution.scale[:, None]
        bias[:, factor] = solution.bias
    for i in family.unrepresented():
        solution = optimize_factor(np.zeros((1, 1)), importance[:, [i]], p[:, [i]], settings)
        loss += solution.loss
        bias[:, i] = solution.bias[:, 0]
    W = W0[None] * scales[:, None, :]
    return FamilySolution(family=family, loss=loss, W=W, b=bias)


@dataclass
class PhaseDiagram:
    """A theoretical phase diagram over (relative importance × feature density).

    Attributes:
        importance: ``(n_imp,)`` relative importance of the extra feature (x axis).
        density: ``(n_dens,)`` feature density 1 − S of every feature (y axis).
        families: the candidate families.
        losses: ``(n_dens, n_imp, n_families)`` optimal loss of each family.
        solutions: one :class:`FamilySolution` per family (cells flattened density-major).
    """

    importance: FloatArray
    density: FloatArray
    families: tuple[Family, ...]
    losses: FloatArray
    solutions: list[FamilySolution] = field(default_factory=list)

    @property
    def winner(self) -> NDArray[np.int64]:
        """``(n_dens, n_imp)`` index of the lowest-loss family."""
        return self.losses.argmin(axis=-1)

    @property
    def labels(self) -> NDArray[np.str_]:
        """``(n_dens, n_imp)`` extra-feature label of the winning family."""
        names = np.array([f.label for f in self.families])
        return names[self.winner]

    @property
    def loss_gap(self) -> FloatArray:
        """Second-best minus best family loss, per cell (how decisive the win is)."""
        ordered = np.sort(self.losses, axis=-1)
        return ordered[..., 1] - ordered[..., 0]

    @property
    def relative_gap(self) -> FloatArray:
        """``loss_gap`` divided by the best loss (NaN where the best loss is 0)."""
        best = np.sort(self.losses, axis=-1)[..., 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(best > 0, self.loss_gap / best, np.nan)


def theoretical_phase_diagram(
    families: Sequence[Family],
    importance: ArrayLike,
    density: ArrayLike,
    base_importance: float = 1.0,
    settings: OptimizerSettings | None = None,
) -> PhaseDiagram:
    """Optimal family per (extra-feature importance, feature density) cell (pp. 14–15).

    Every feature but the last has importance ``base_importance`` (1.0 in the paper); the last,
    "extra" feature has importance ``importance[k]``; all features share density 1 − S.
    """
    importance = np.asarray(importance, dtype=np.float64)
    density = np.asarray(density, dtype=np.float64)
    n = len(families[0].ignored)
    dens, imp = np.meshgrid(density, importance, indexing="ij")  # (n_dens, n_imp)
    cell_importance = np.full((dens.size, n), base_importance)
    cell_importance[:, -1] = imp.ravel()
    solutions = [optimize_family(f, cell_importance, dens.ravel(), settings) for f in families]
    losses = np.stack([s.loss.reshape(dens.shape) for s in solutions], axis=-1)
    return PhaseDiagram(importance, density, tuple(families), losses, solutions)
