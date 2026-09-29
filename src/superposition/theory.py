"""Closed-form and analytic results about the toy models.

Paper: Mathematical Understanding, pp. 12–14 (linear model loss; sparsity decomposition);
Superposition as a Phase Change, pp. 14–15 (analytic phase diagrams).

Phase 1 implements the exact optimum of the linear model, which E00 uses as a ground truth
for the loss and evaluation code. The loss decompositions and the theoretical phase diagrams
are implemented in Phase 3 (PLAN.md).
"""

from __future__ import annotations

import torch
from jaxtyping import Float
from torch import Tensor


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
