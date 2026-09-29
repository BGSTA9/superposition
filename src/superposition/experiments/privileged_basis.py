"""E07 privileged_basis: superposition in a model whose hidden layer has neurons.

Paper: Superposition in a Privileged Basis, pp. 31–34.

The ReLU hidden layer model h = ReLU(Wx), x′ = ReLU(Wᵀh + b) (no hidden bias, p. 34) has a
privileged basis: the ReLU acts on each hidden dimension separately, so rotating the hidden
space changes the model and W itself (features → neurons) can be read directly (p. 31).

Claims (n = 10, m = 5, Iᵢ = 0.75ⁱ, sparsity swept; the lowest-loss of many models per
setting, pp. 32–33):

* dense data → every neuron is monosemantic ("every feature gets a dedicated neuron");
* as sparsity rises, neurons shift from monosemantic to polysemantic, in a neuron-level phase
  change (p. 34);
* the model avoids its ReLU when it can: "given a hidden layer bias, the model will set all the
  biases to be positive, shifting the neurons into a positive regime where they behave
  linearly" (p. 34). We document this with a hidden-bias variant (never the replication
  default) and by measuring how often each model's ReLU actually clips.

Operationalization of the criteria: docs/DECISIONS.md D-407.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import einops
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from jaxtyping import Float
from torch import Tensor, nn

from superposition import metrics
from superposition.acceptance import Criterion, check, info
from superposition.config import ExperimentConfig, TrainConfig, from_dict
from superposition.data import IndependentSparse, geometric_importance
from superposition.experiments.base import Experiment
from superposition.experiments.common import (
    GridResult,
    format_probability,
    instance_table,
    train_sparsity_grid,
)
from superposition.models import TiedModel
from superposition.plotting.neurons import (
    NeuronPanel,
    neuron_count_figure,
    neuron_order_by_top_feature,
    neuron_rows,
)
from superposition.plotting.style import save_figure
from superposition.runs import RunDir
from superposition.train import select_best, train_from_config
from superposition.utils import EVAL_STREAM, derive_seed, make_generator

PAGE = "pp. 32–33"


def _default_train() -> TrainConfig:
    return TrainConfig(steps=10_000, batch_size=256, log_every=500)


@dataclass
class HiddenBiasVariant:
    """A variant *with* a hidden bias, h = ReLU(Wx + c), to document p. 34 (never graded)."""

    enabled: bool = True
    feature_probabilities: list[float] = field(default_factory=lambda: [1.0, 0.1, 0.03])
    n_restarts: int = 20


@dataclass
class PrivilegedBasisThresholds:
    """Polysemanticity and acceptance thresholds (D-407)."""

    rel_threshold: float = 0.25  # feature counts for a neuron at ≥ 25% of its largest |weight|
    abs_threshold: float = 0.05  # a neuron whose largest |weight| is below this is dead
    count_tolerance: int = 1
    represented_norm: float = 0.5
    superposition_score: float = 0.05
    activity_samples: int = 20_000  # inputs used to measure how often the ReLU clips


@dataclass
class PrivilegedBasisConfig(ExperimentConfig):
    """E07 settings. ``feature_probabilities`` are 1 − S, densest first (as on pp. 32–33)."""

    n_features: int = 10
    n_hidden: int = 5
    importance_decay: float = 0.75
    feature_probabilities: list[float] = field(
        default_factory=lambda: [1.0, 0.5, 0.3, 0.2, 0.15, 0.12, 0.1, 0.07, 0.05, 0.03, 0.01]
    )
    n_restarts: int = 100
    # Rows of the p. 33-style figure (nearest grid points): one per regime the paper names.
    figure_probabilities: list[float] = field(
        default_factory=lambda: [1.0, 0.3, 0.15, 0.12, 0.07, 0.03]
    )
    hidden_bias: HiddenBiasVariant = field(default_factory=HiddenBiasVariant)
    thresholds: PrivilegedBasisThresholds = field(default_factory=PrivilegedBasisThresholds)
    train: TrainConfig = field(default_factory=_default_train)


class ReluHiddenWithBias(TiedModel):
    """Variant of the ReLU hidden layer model with a hidden bias: h = ReLU(Wx + c).

    Paper: p. 34 explains why the replication model has no hidden bias: the model would set
    c > 0 and keep every neuron in its linear regime. This class exists only to measure that.
    """

    def __init__(self, n_features: int, n_hidden: int, n_inst: int, seed: int = 0) -> None:
        super().__init__(n_features, n_hidden, n_inst, "relu", "relu", bias=True, seed=seed)
        self.c = nn.Parameter(torch.zeros(n_inst, n_hidden))

    def pre_activation(
        self, x: Float[Tensor, "batch inst feat"]
    ) -> Float[Tensor, "batch inst hidden"]:
        pre = einops.einsum(self.W, x, "inst hidden feat, batch inst feat -> batch inst hidden")
        return pre + self.c

    def hidden(self, x: Float[Tensor, "batch inst feat"]) -> Float[Tensor, "batch inst hidden"]:
        return F.relu(self.pre_activation(x))


# --- analysis -----------------------------------------------------------------------------------


def neuron_counts(W: Float[Tensor, "inst hidden feat"], thr: PrivilegedBasisThresholds) -> Tensor:
    """Features per neuron (0 dead, 1 monosemantic, ≥ 2 polysemantic) for each instance."""
    return metrics.neuron_polysemanticity(W, thr.rel_threshold, thr.abs_threshold)


@torch.no_grad()
def clipped_fraction(
    W: Float[Tensor, "inst hidden feat"],
    c: Float[Tensor, "inst hidden"] | None,
    feature_probability: Float[Tensor, " inst"],
    n_samples: int,
    generator: torch.Generator,
) -> Float[Tensor, " inst"]:
    """How often the hidden ReLU clips: the fraction of (input, neuron) pairs with a negative
    pre-activation, over inputs with at least one active feature (p. 34).

    Near 0 means the neurons sit in their linear regime and the ReLU does nothing: the model has
    sidestepped it.
    """
    n_inst, _, n_feat = W.shape
    x = IndependentSparse(feature_probability, n_inst, n_feat).sample(n_samples, generator)
    pre = einops.einsum(W, x, "inst hidden feat, batch inst feat -> batch inst hidden")
    if c is not None:
        pre = pre + c
    nonzero = (x != 0).any(dim=-1, keepdim=True).float()  # (batch, inst, 1)
    clipped = (pre < 0).float() * nonzero
    return clipped.sum(dim=(0, 2)) / (nonzero.sum(dim=(0, 2)) * W.shape[1]).clamp_min(1.0)


def summarize(grid: GridResult, thr: PrivilegedBasisThresholds) -> pd.DataFrame:
    """One row per sparsity (best restart): neuron counts, represented features, loss."""
    rows = []
    counts_all = neuron_counts(grid.W, thr)
    poly_all = (counts_all >= 2).sum(dim=-1)
    for cond in range(grid.n_conditions):
        k = int(grid.best[cond])
        W = grid.W[k]
        counts = counts_all[k]
        rep = metrics.represented_mask(W[None], thr.represented_norm)[0]
        score = metrics.superposition_score(W[None])[0]
        same = (grid.condition == cond) & (poly_all == poly_all[k])
        rows.append(
            {
                "condition": cond,
                "feature_probability": grid.condition_probability(cond),
                "best_instance": k,
                "loss": float(grid.final_loss[k]),
                "monosemantic": int((counts == 1).sum()),
                "polysemantic": int((counts >= 2).sum()),
                "dead": int((counts == 0).sum()),
                "features_per_neuron": counts.tolist(),
                "represented": int(rep.sum()),
                "superposed": int((rep & (score > thr.superposition_score)).sum()),
                "restarts_same_poly_count": float(same.sum())
                / float((grid.condition == cond).sum()),
            }
        )
    return pd.DataFrame(rows)


def evaluate(
    config: PrivilegedBasisConfig, summary: pd.DataFrame, variant: dict | None
) -> list[Criterion]:
    """E07 criteria (SPEC §7; D-407)."""
    thr = config.thresholds
    labels = [format_probability(p) for p in summary["feature_probability"]]
    poly = summary["polysemantic"].tolist()
    dense = summary.to_dict("records")[0]
    criteria = [
        check(
            "dense_all_monosemantic",
            f"Dense (1 − S = {labels[0]}): every neuron is monosemantic (exactly one feature at "
            f"≥ {thr.rel_threshold:.0%} of its largest |weight|)",
            PAGE,
            int(dense["monosemantic"]) == config.n_hidden,
            {
                "features_per_neuron": dense["features_per_neuron"],
                "monosemantic": int(dense["monosemantic"]),
            },
            f"{config.n_hidden}/{config.n_hidden} neurons carry exactly 1 feature",
        )
    ]
    running_max, drops = 0, []
    for lab, count in zip(labels, poly, strict=True):
        if count < running_max - thr.count_tolerance:
            drops.append(lab)
        running_max = max(running_max, count)
    criteria.append(
        check(
            "polysemantic_count_non_decreasing",
            f"The polysemantic-neuron count never falls more than {thr.count_tolerance} below "
            "any denser setting's count",
            PAGE,
            not drops,
            {"counts": dict(zip(labels, poly, strict=True)), "violations_at": drops},
            {"tolerance": thr.count_tolerance},
        )
    )
    criteria.append(
        check(
            "polysemantic_at_sparsest",
            f"Polysemantic neurons exist at the sparsest setting (1 − S = {labels[-1]})",
            PAGE,
            poly[-1] > 0,
            poly[-1],
            "> 0",
        )
    )
    criteria.append(
        info(
            "per_sparsity_summary",
            "Best model per sparsity: neuron counts, features per neuron, represented and "
            "superposed features, loss, and the share of restarts with the same polysemantic "
            "count",
            PAGE,
            {
                f"1-S={lab}": {
                    "mono": int(row["monosemantic"]),
                    "poly": int(row["polysemantic"]),
                    "dead": int(row["dead"]),
                    "features_per_neuron": row["features_per_neuron"],
                    "represented": int(row["represented"]),
                    "superposed": int(row["superposed"]),
                    "loss": round(float(row["loss"]), 5),
                    "same_poly_share": round(float(row["restarts_same_poly_count"]), 2),
                }
                for lab, row in zip(labels, summary.to_dict("records"), strict=True)
            },
        )
    )
    if variant is not None:
        criteria.append(
            info(
                "relu_sidestepping",
                "p. 34: with a hidden bias the model sets it positive and keeps neurons linear. "
                "Per sparsity: share of hidden biases > 0 and how often the ReLU clips "
                "(negative pre-activations on inputs with an active feature), with vs without "
                "a hidden bias; loss of each best model",
                "p. 34",
                variant,
            )
        )
    return criteria


# --- run ------------------------------------------------------------------------------------------


def run_hidden_bias_variant(
    config: PrivilegedBasisConfig,
    grid: GridResult,
    device: torch.device,
    logger: logging.Logger,
) -> tuple[dict, dict]:
    """Train the hidden-bias variant and compare how often the ReLU clips, with and without."""
    variant = config.hidden_bias
    thr = config.thresholds
    probs = variant.feature_probabilities
    r = variant.n_restarts
    n_inst = len(probs) * r
    condition = torch.arange(len(probs)).repeat_interleave(r)
    p = torch.tensor(probs, dtype=torch.float32)[condition]
    importance = geometric_importance(config.n_features, config.importance_decay)
    model = ReluHiddenWithBias(config.n_features, config.n_hidden, n_inst, seed=config.seed)
    model = model.to(device)
    dist = IndependentSparse(p, n_inst, config.n_features)
    logger.info(f"hidden-bias variant: {len(probs)} sparsities × {r} restarts")
    result = train_from_config(model, dist, importance, config.train, config.seed)
    best = select_best(result.final_loss, condition)
    W = model.W.detach().cpu()[best]
    c = model.c.detach().cpu()[best]
    generator = make_generator(derive_seed(config.seed, EVAL_STREAM, 3))
    clipped_with = clipped_fraction(W, c, p[best], thr.activity_samples, generator)
    # The replication model (no hidden bias) at the nearest grid sparsity, for comparison.
    grid_probs = np.array([grid.condition_probability(k) for k in range(grid.n_conditions)])
    nearest = [int(np.abs(grid_probs - q).argmin()) for q in probs]
    W0 = grid.W[grid.best[nearest]]
    clipped_without = clipped_fraction(
        W0, None, torch.tensor(grid_probs[nearest], dtype=torch.float32), thr.activity_samples,
        generator,
    )  # fmt: skip
    report = {
        f"1-S={format_probability(q)}": {
            "hidden_biases_positive": f"{int((c[i] > 0).sum())}/{config.n_hidden}",
            "mean_hidden_bias": round(float(c[i].mean()), 3),
            "clipped_with_bias": round(float(clipped_with[i]), 4),
            "clipped_without_bias": round(float(clipped_without[i]), 4),
            "loss_with_bias": round(float(result.final_loss[best[i]]), 5),
            "loss_without_bias": round(float(grid.final_loss[grid.best[nearest[i]]]), 5),
        }
        for i, q in enumerate(probs)
    }
    state = {
        "W": W,
        "c": c,
        "b": model.b.detach().cpu()[best],
        "feature_probability": torch.tensor(probs),
        "final_loss": result.final_loss[best],
    }
    return report, state


def run(
    config: PrivilegedBasisConfig, run_dir: RunDir, device: torch.device, logger: logging.Logger
) -> list[Criterion]:
    thr = config.thresholds
    importance = geometric_importance(config.n_features, config.importance_decay)
    logger.info(
        f"ReLU hidden layer model: n={config.n_features}, m={config.n_hidden}, "
        f"{len(config.feature_probabilities)} sparsities × {config.n_restarts} restarts"
    )
    grid = train_sparsity_grid(
        "relu_hidden",
        config.n_features,
        config.n_hidden,
        importance,
        config.feature_probabilities,
        config.n_restarts,
        config.train,
        seed=config.seed,
        device=device,
        progress=True,
    )
    logger.info(f"  trained in {grid.wall_time_s:.1f} s")
    run_dir.save_checkpoint("relu_hidden", grid.to_checkpoint())
    table = instance_table(grid, thr.represented_norm, thr.superposition_score)
    counts = neuron_counts(grid.W, thr)
    table["monosemantic"] = (counts == 1).sum(dim=-1).numpy()
    table["polysemantic"] = (counts >= 2).sum(dim=-1).numpy()
    run_dir.write_table(table, "metrics")
    summary = summarize(grid, thr)
    run_dir.write_table(summary, "summary")

    variant = None
    if config.hidden_bias.enabled:
        variant, state = run_hidden_bias_variant(config, grid, device, logger)
        run_dir.save_checkpoint("hidden_bias_variant", state)
    return evaluate(config, summary, variant)


# --- figures ------------------------------------------------------------------------------------


def figures(run_dir: RunDir) -> list[Path]:
    """The p. 33 rows (W and stack plot per regime) and the neuron-count summary."""
    grid = GridResult.from_checkpoint(run_dir.load_checkpoint("relu_hidden"))
    summary = run_dir.read_table("summary")
    config = from_dict(PrivilegedBasisConfig, yaml.safe_load(run_dir.config_path.read_text()))
    probs = np.array([grid.condition_probability(k) for k in range(grid.n_conditions)])
    chosen = sorted({int(np.abs(probs - q).argmin()) for q in config.figure_probabilities})
    panels = []
    for cond in chosen:
        W, _, _ = grid.best_weights(cond)
        counts = neuron_counts(W[None], config.thresholds)[0].numpy()
        Wn = W.numpy()
        panels.append(
            NeuronPanel(
                title=f"1−S = {format_probability(probs[cond])}",
                W=Wn,
                superposition=metrics.superposition_score(W[None])[0].numpy(),
                polysemantic=counts,
                order=neuron_order_by_top_feature(Wn),
            )
        )
    paths = save_figure(
        neuron_rows(
            panels,
            suptitle=f"E07 · ReLU hidden layer, n = {grid.W.shape[2]}, m = {grid.W.shape[1]}: "
            "W (features × neurons) and neuron stack plots, best of "
            f"{int((grid.condition == 0).sum())} restarts",
        ),
        run_dir.figures_dir / "neurons",
    )
    paths += save_figure(
        neuron_count_figure(
            summary["feature_probability"].to_numpy(),
            summary["monosemantic"].to_numpy(),
            summary["polysemantic"].to_numpy(),
            title="E07 · monosemantic → polysemantic as sparsity rises",
        ),
        run_dir.figures_dir / "neuron_counts",
    )
    return paths


EXPERIMENT = Experiment(
    name="privileged_basis",
    code="E07",
    title="ReLU hidden layer: monosemantic → polysemantic neurons as sparsity rises (n = 10, m = 5)",
    paper_pages="pp. 31–34",
    config_cls=PrivilegedBasisConfig,
    run=run,
    figures=figures,
)
