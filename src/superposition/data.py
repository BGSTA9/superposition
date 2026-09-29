"""Synthetic feature distributions for the toy models.

Paper: Demonstrating Superposition, Experiment Setup, p. 9 (independent sparse features);
Correlated and Anticorrelated Features, p. 23 (feature sets); Computation in Superposition,
Experiment Setup, p. 35 (values on [−1, 1]).

Every distribution returns a batch shaped ``(batch, inst, feat)``: the leading instance
dimension lets one call train many independent models (restarts, sparsity levels) at once.
Each instance can have its own feature probability, so a whole sparsity sweep is one model.

Conventions (SPEC §3):

* A feature is zero with probability S (its *sparsity*) and otherwise uniform on
  ``[low, high]`` (default [0, 1]).
* Code uses the **feature probability** p = 1 − S. Plots use 1 − S or 1/(1 − S).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import torch
from jaxtyping import Float
from torch import Tensor


@runtime_checkable
class FeatureDistribution(Protocol):
    """Anything that samples feature vectors ``x`` for ``n_inst`` instances (SPEC §6)."""

    n_features: int
    n_inst: int

    def sample(
        self, batch_size: int, generator: torch.Generator
    ) -> Float[Tensor, "batch inst feat"]:
        """Draw ``batch_size`` samples per instance on ``generator``'s device."""
        ...


def geometric_importance(n_features: int, decay: float) -> Float[Tensor, " feat"]:
    """Importance Iᵢ = decayⁱ for i = 0 … n−1, most important first.

    Paper: pp. 11–12, 32, 36 use 0.7ⁱ, 0.9ⁱ, 0.75ⁱ and 0.8ⁱ; p. 17 uses Iᵢ = 1 (decay 1).
    Footnote 1: importance is a scalar multiplier on each feature's squared error.
    """
    return decay ** torch.arange(n_features, dtype=torch.float32)


def _as_probability_table(
    feature_probability: float | Sequence[float] | Tensor, n_inst: int, n_cols: int, what: str
) -> Float[Tensor, "inst col"]:
    """Broadcast a probability spec to ``(n_inst, n_cols)``.

    Accepted shapes: scalar (shared), ``(n_inst,)`` (one per instance), ``(1, n_cols)``
    (one per column, shared across instances) or ``(n_inst, n_cols)``.
    """
    p = torch.as_tensor(feature_probability, dtype=torch.float32)
    if p.ndim == 1:
        if p.shape[0] != n_inst:
            raise ValueError(
                f"a 1-D {what} must have one entry per instance ({n_inst}), got {tuple(p.shape)}"
            )
        p = p[:, None]
    if p.ndim > 2:
        raise ValueError(f"{what} must be a scalar, (inst,), (1, {n_cols}) or (inst, {n_cols})")
    table = torch.broadcast_to(p, (n_inst, n_cols)).clone()
    if ((table < 0) | (table > 1)).any():
        raise ValueError(f"{what} values must lie in [0, 1]")
    return table


class IndependentSparse:
    """Independent sparse features: xᵢ = 0 with probability 1 − pᵢ, else U[low, high].

    Paper: Experiment Setup, p. 9 ("We let xᵢ = 0 with probability Sᵢ, but is otherwise
    uniformly distributed between [0, 1]"); p. 35 for the [−1, 1] absolute-value task.

    **Sampling trick (one uniform per entry).** We draw u ~ U[0, 1) once per entry, set the
    feature active when u < p, and use u/p as its value. Conditioned on u < p, u is uniform on
    [0, p), so u/p is exactly U[0, 1): same distribution as two independent draws, at half the
    random-number cost, which dominates CPU training time. See docs/DECISIONS.md (D-102).

    Args:
        feature_probability: p = 1 − S. Scalar, ``(n_inst,)``, ``(1, n_features)`` or
            ``(n_inst, n_features)``.
        n_inst: number of independent instances.
        n_features: number of features n.
        low, high: range of nonzero values ([0, 1] by default; [−1, 1] for |x|).
    """

    def __init__(
        self,
        feature_probability: float | Sequence[float] | Tensor,
        n_inst: int,
        n_features: int,
        low: float = 0.0,
        high: float = 1.0,
    ) -> None:
        if not high > low:
            raise ValueError(f"need high > low, got low={low}, high={high}")
        self.n_inst = n_inst
        self.n_features = n_features
        self.low = low
        self.high = high
        self.feature_probability = _as_probability_table(
            feature_probability, n_inst, n_features, "feature_probability"
        )

    def sample(
        self, batch_size: int, generator: torch.Generator
    ) -> Float[Tensor, "batch inst feat"]:
        device = generator.device
        p = self.feature_probability.to(device)
        u = torch.rand(
            (batch_size, self.n_inst, self.n_features), generator=generator, device=device
        )
        active = u < p
        value = self.low + (self.high - self.low) * (u / p.clamp_min(1e-12))
        return torch.where(active, value, torch.zeros((), device=device))


class CorrelatedSets:
    """Sets of features that are zero or nonzero together.

    Paper: Correlated Feature Sets, p. 23: "we represent this by linking the choice of whether
    all the features in a correlated feature set are zero or not together … We simply have the
    same sample determine whether they're zero." Nonzero values are drawn independently.

    Features are laid out set-major: set 0 is features ``0 … set_size−1``, and so on.

    Args:
        n_sets: number of correlated sets.
        set_size: features per set.
        feature_probability: probability that a set is active. Scalar, ``(n_inst,)``,
            ``(1, n_sets)`` or ``(n_inst, n_sets)``.
        n_inst: number of instances.
        low, high: range of nonzero values.
    """

    def __init__(
        self,
        n_sets: int,
        set_size: int,
        feature_probability: float | Sequence[float] | Tensor,
        n_inst: int,
        low: float = 0.0,
        high: float = 1.0,
    ) -> None:
        self.n_sets = n_sets
        self.set_size = set_size
        self.n_inst = n_inst
        self.n_features = n_sets * set_size
        self.low = low
        self.high = high
        self.set_probability = _as_probability_table(
            feature_probability, n_inst, n_sets, "feature_probability"
        )

    def sample(
        self, batch_size: int, generator: torch.Generator
    ) -> Float[Tensor, "batch inst feat"]:
        device = generator.device
        p = self.set_probability.to(device)
        set_active = (
            torch.rand((batch_size, self.n_inst, self.n_sets), generator=generator, device=device)
            < p
        )
        values = self.low + (self.high - self.low) * torch.rand(
            (batch_size, self.n_inst, self.n_sets, self.set_size),
            generator=generator,
            device=device,
        )
        x = values * set_active[..., None]
        return x.reshape(batch_size, self.n_inst, self.n_features)


class AnticorrelatedSets:
    """Sets in which at most one feature is active at a time.

    Paper: Anticorrelated Feature Sets, p. 23: "we'll have the feature set be entirely zero
    with probability S, but then only have one randomly selected feature in the set be
    uniformly sampled from [0,1] if it's active, with the others being zero."

    Args:
        n_sets: number of anticorrelated sets.
        set_size: features per set.
        feature_probability: probability that a set is active (1 − S). Scalar, ``(n_inst,)``,
            ``(1, n_sets)`` or ``(n_inst, n_sets)``.
        n_inst: number of instances.
        low, high: range of the active member's value.
    """

    def __init__(
        self,
        n_sets: int,
        set_size: int,
        feature_probability: float | Sequence[float] | Tensor,
        n_inst: int,
        low: float = 0.0,
        high: float = 1.0,
    ) -> None:
        self.n_sets = n_sets
        self.set_size = set_size
        self.n_inst = n_inst
        self.n_features = n_sets * set_size
        self.low = low
        self.high = high
        self.set_probability = _as_probability_table(
            feature_probability, n_inst, n_sets, "feature_probability"
        )

    def sample(
        self, batch_size: int, generator: torch.Generator
    ) -> Float[Tensor, "batch inst feat"]:
        device = generator.device
        p = self.set_probability.to(device)
        shape = (batch_size, self.n_inst, self.n_sets)
        set_active = torch.rand(shape, generator=generator, device=device) < p
        member = torch.randint(self.set_size, shape, generator=generator, device=device)
        value = self.low + (self.high - self.low) * torch.rand(
            shape, generator=generator, device=device
        )
        one_hot = torch.nn.functional.one_hot(member, self.set_size).to(value.dtype)
        x = one_hot * (value * set_active)[..., None]
        return x.reshape(batch_size, self.n_inst, self.n_features)


class Composite:
    """Concatenate several distributions along the feature axis.

    Used to mix independent, correlated and anticorrelated groups in one input
    (Paper: pp. 23–25). Parts are sampled in order from the same generator.
    """

    def __init__(self, parts: Sequence[FeatureDistribution]) -> None:
        if not parts:
            raise ValueError("Composite needs at least one part")
        n_inst = {part.n_inst for part in parts}
        if len(n_inst) != 1:
            raise ValueError(f"all parts must have the same n_inst, got {sorted(n_inst)}")
        self.parts = list(parts)
        self.n_inst = n_inst.pop()
        self.n_features = sum(part.n_features for part in parts)

    def sample(
        self, batch_size: int, generator: torch.Generator
    ) -> Float[Tensor, "batch inst feat"]:
        return torch.cat([part.sample(batch_size, generator) for part in self.parts], dim=-1)
