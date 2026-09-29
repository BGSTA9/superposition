"""Helpers shared by experiments that train tied models over a grid of sparsities.

Paper: Demonstrating Superposition, pp. 9–12. A *grid* is a set of conditions (feature
probabilities p = 1 − S) × restarts, trained as one multi-instance model: instance
``cond · n_restarts + r`` is restart r of condition ``cond``. Restarts matter because small
models, especially with a 2D hidden space, often land in worse local optima; the paper keeps
the lowest-loss restart (p. 23), which :func:`superposition.train.select_best` implements.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
from jaxtyping import Float, Int
from torch import Tensor

from superposition import metrics
from superposition.config import TrainConfig
from superposition.data import IndependentSparse
from superposition.models import TiedKind, TiedModel
from superposition.train import select_best, train_from_config


@dataclass
class GridResult:
    """A trained grid of tied models (all tensors on CPU).

    Attributes:
        kind: ``linear``, ``relu_output`` or ``relu_hidden``.
        W: ``(inst, hidden, feat)`` trained weights.
        b: ``(inst, feat)`` trained biases.
        final_loss: ``(inst,)`` loss on fresh evaluation samples.
        feature_probability: ``(inst,)`` p = 1 − S of each instance.
        condition: ``(inst,)`` condition index of each instance.
        restart: ``(inst,)`` restart index of each instance.
        best: ``(cond,)`` lowest-loss instance per condition.
        importance: ``(feat,)`` the importance vector used.
        loss_history: ``(log, inst)`` training loss; ``log_steps`` its step numbers.
        wall_time_s: training time.
    """

    kind: str
    W: Float[Tensor, "inst hidden feat"]
    b: Float[Tensor, "inst feat"]
    final_loss: Float[Tensor, " inst"]
    feature_probability: Float[Tensor, " inst"]
    condition: Int[Tensor, " inst"]
    restart: Int[Tensor, " inst"]
    best: Int[Tensor, " cond"]
    importance: Float[Tensor, " feat"]
    loss_history: Float[Tensor, "log inst"]
    log_steps: list[int]
    wall_time_s: float

    @property
    def n_conditions(self) -> int:
        return int(self.best.shape[0])

    def condition_probability(self, cond: int) -> float:
        return float(self.feature_probability[self.best[cond]])

    def best_weights(self, cond: int) -> tuple[Tensor, Tensor, float]:
        """``(W, b, loss)`` of the lowest-loss restart of condition ``cond``."""
        k = int(self.best[cond])
        return self.W[k], self.b[k], float(self.final_loss[k])

    def to_checkpoint(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "W": self.W,
            "b": self.b,
            "final_loss": self.final_loss,
            "feature_probability": self.feature_probability,
            "condition": self.condition,
            "restart": self.restart,
            "best": self.best,
            "importance": self.importance,
            "loss_history": self.loss_history,
            "log_steps": self.log_steps,
            "wall_time_s": self.wall_time_s,
        }

    @classmethod
    def from_checkpoint(cls, state: dict[str, Any]) -> GridResult:
        return cls(**state)


def train_sparsity_grid(
    kind: TiedKind,
    n_features: int,
    n_hidden: int,
    importance: Float[Tensor, " feat"],
    feature_probabilities: Sequence[float],
    n_restarts: int,
    train_config: TrainConfig,
    seed: int,
    device: torch.device,
    progress: bool = False,
) -> GridResult:
    """Train ``len(feature_probabilities) × n_restarts`` instances of one tied model kind.

    Every instance sees data with its own feature probability (SPEC §6 data.py), so the whole
    sweep is a single vectorized model. Initialization depends only on (seed, instance).
    """
    n_cond = len(feature_probabilities)
    n_inst = n_cond * n_restarts
    condition = torch.arange(n_cond).repeat_interleave(n_restarts)
    restart = torch.arange(n_restarts).repeat(n_cond)
    p = torch.tensor(feature_probabilities, dtype=torch.float32)[condition]
    model = TiedModel.from_kind(kind, n_features, n_hidden, n_inst, seed=seed).to(device)
    dist = IndependentSparse(p, n_inst, n_features)
    result = train_from_config(model, dist, importance, train_config, seed=seed, progress=progress)
    return GridResult(
        kind=kind,
        W=model.W.detach().cpu(),
        b=model.b.detach().cpu(),
        final_loss=result.final_loss,
        feature_probability=p,
        condition=condition,
        restart=restart,
        best=select_best(result.final_loss, condition),
        importance=importance.cpu(),
        loss_history=result.loss_history,
        log_steps=result.log_steps,
        wall_time_s=result.wall_time_s,
    )


def instance_table(
    grid: GridResult, represented_threshold: float, superposition_threshold: float
) -> pd.DataFrame:
    """One row per instance: loss, counts of represented / superposed features, D*."""
    W = grid.W
    represented = metrics.represented_mask(W, represented_threshold)
    score = metrics.superposition_score(W)
    superposed = represented & (score > superposition_threshold)
    is_best = torch.zeros(W.shape[0], dtype=torch.bool)
    is_best[grid.best] = True
    return pd.DataFrame(
        {
            "model": grid.kind,
            "condition": grid.condition.numpy(),
            "feature_probability": grid.feature_probability.numpy(),
            "restart": grid.restart.numpy(),
            "instance": range(W.shape[0]),
            "final_loss": grid.final_loss.numpy(),
            "is_best": is_best.numpy(),
            "n_represented": represented.sum(-1).numpy(),
            "n_superposed": superposed.sum(-1).numpy(),
            "dims_per_feature": metrics.dims_per_feature(W).numpy(),
            "is_superposition": metrics.is_superposition(W, represented_threshold).numpy(),
        }
    )


def feature_table(grid: GridResult) -> pd.DataFrame:
    """One row per (instance, feature): norm, bias, superposition score, Dᵢ."""
    W = grid.W
    n_inst, _, n_feat = W.shape
    frame = pd.DataFrame(
        {
            "model": grid.kind,
            "instance": torch.arange(n_inst).repeat_interleave(n_feat).numpy(),
            "feature": torch.arange(n_feat).repeat(n_inst).numpy(),
            "importance": grid.importance.repeat(n_inst).numpy(),
            "norm": metrics.feature_norms(W).flatten().numpy(),
            "bias": grid.b.flatten().numpy(),
            "superposition_score": metrics.superposition_score(W).flatten().numpy(),
            "dimensionality": metrics.feature_dimensionality(W).flatten().numpy(),
        }
    )
    frame["condition"] = grid.condition.repeat_interleave(n_feat).numpy()
    frame["feature_probability"] = grid.feature_probability.repeat_interleave(n_feat).numpy()
    frame["restart"] = grid.restart.repeat_interleave(n_feat).numpy()
    return frame


def represented_features(W: Float[Tensor, "hidden feat"], threshold: float) -> list[int]:
    """Indices of features with ‖Wᵢ‖ > threshold, for one instance."""
    return torch.nonzero(metrics.represented_mask(W[None], threshold)[0]).flatten().tolist()


def antipodal_pairs(
    W: Float[Tensor, "hidden feat"], threshold: float, cos_max: float
) -> list[tuple[int, int]]:
    """Pairs (i < j) of represented features with cos(Wᵢ, Wⱼ) < ``cos_max`` (one instance)."""
    rep = represented_features(W, threshold)
    cos = metrics.cosine_similarity_matrix(W[None])[0]
    return [(i, j) for a, i in enumerate(rep) for j in rep[a + 1 :] if float(cos[i, j]) < cos_max]


def format_probability(p: float) -> str:
    """Compact label for a feature probability, e.g. 1 − S = 0.003."""
    return f"{p:.3g}"
