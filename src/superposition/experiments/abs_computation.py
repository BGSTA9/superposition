"""E08 abs_computation: computing |x| with ReLU neurons, with and without superposition.

Paper: Computation in Superposition, pp. 34–39.

Setup (p. 35): features are 0 with probability S, otherwise uniform on [−1, 1]; the target is
y = |x|; the model is h = ReLU(W₁x), y′ = ReLU(W₂h + b) with untied W₁, W₂. It *needs* its
hidden ReLU, since |x| = ReLU(x) + ReLU(−x), so unlike the ReLU hidden layer model (E07) it
cannot sidestep the nonlinearity.

**(a) n = 3, m = 6, no superposition (p. 36).** "For each input feature xᵢ, it constructs a
'positive side' neuron ReLU(xᵢ) and a 'negative side' neuron ReLU(−xᵢ). It then adds these
together to compute absolute value." Visualized after footnote-15 normalization (each neuron's
largest |W₁| weight scaled to 1, W₂ inversely).

**(b) n = 100, m = 40, Iᵢ = 0.8ⁱ, sparsity swept (pp. 36–38).** Dense → each neuron represents
a single feature; sparse → "this model, too, uses superposition to represent more features than
it has neurons", and gradient descent "manages to find solutions that usefully approximate the
computation even when each neuron encodes a mix of multiple features" (p. 37). Graded (D-R01):
at the sparsest setting, more than m features are *usefully computed* (per-feature loss below
75% of predicting the mean, i.e. closer to one side of |x| computed exactly than to nothing);
SPEC's original criterion (more than m/2 features *computed well*, below 50%) is reported
next to it. Many neurons have one "primary" feature with a large weight plus "secondary"
features with smaller ones (p. 38).

Operationalization of the criteria: docs/DECISIONS.md D-408, D-R01. Diagnostic variants (never
the replication default): a positive initial output bias (D-409) and a smaller initial weight
scale (D-R01).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import einops
import numpy as np
import pandas as pd
import torch
import yaml
from jaxtyping import Float
from torch import Tensor

from superposition import metrics
from superposition.acceptance import Criterion, check, info
from superposition.config import ExperimentConfig, TrainConfig, from_dict
from superposition.data import IndependentSparse, geometric_importance
from superposition.experiments.base import Experiment
from superposition.experiments.common import format_probability
from superposition.models import AbsModel
from superposition.plotting.neurons import (
    NeuronPanel,
    abs_weights_figure,
    capacity_figure,
    neuron_order_by_top_feature,
    neuron_rows,
)
from superposition.plotting.style import save_figure
from superposition.runs import RunDir
from superposition.theory import uniform_feature_variance
from superposition.train import select_best, train_from_config
from superposition.utils import EVAL_STREAM, derive_seed, make_generator

PAGE_A = "p. 36"
PAGE_B = "pp. 36–38"


@dataclass
class NoSuperpositionSetup:
    """Part (a): n = 3 features, m = 6 neurons (p. 36). Density and importance: D-408."""

    n_features: int = 3
    n_hidden: int = 6
    importance_decay: float = 1.0
    feature_probability: float = 1.0
    n_restarts: int = 8


@dataclass
class SparsitySweepSetup:
    """Part (b): n = 100, m = 40, Iᵢ = 0.8ⁱ, the p. 37 sparsities (1 − S, densest first)."""

    n_features: int = 100
    n_hidden: int = 40
    importance_decay: float = 0.8
    feature_probabilities: list[float] = field(
        default_factory=lambda: [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
    )
    n_restarts: int = 4


@dataclass
class AbsThresholds:
    """Acceptance and role thresholds (SPEC §7 E08; D-408)."""

    per_feature_loss_max: float = 0.05  # (a) vs the loss of predicting the mean
    # (b) per-feature loss below this fraction of the mean baseline: SPEC's original criterion
    # ("computed well", both sides needed), reported next to the graded one (D-R01)
    computed_well: float = 0.5
    # (b) graded (D-R01): "usefully computed", closer to one side of |x| computed exactly (≈ 0.5)
    # than to not computing the feature at all (1)
    usefully_computed: float = 0.75
    rel_threshold: float = 0.25  # a feature counts for a neuron at ≥ 25% of its largest |W₁|
    dead_output: float = 0.05  # a neuron whose normalized W₂ column is below this is dead
    primary_ratio: float = 2.0  # primary feature ≥ 2× the next-largest |W₁| weight (p. 38)
    side_computed_max: float = 0.25  # a side is computed if its 1-sparse relative error < this
    loss_samples: int = 20_000  # samples per conditional term of the per-feature loss


@dataclass
class BiasInitVariant:
    """A diagnostic variant (never graded, never the default): the output bias starts at
    ``bias_init`` > 0 instead of 0, so no output ReLU starts dead (D-409)."""

    enabled: bool = True
    bias_init: float = 0.1
    feature_probabilities: list[float] = field(default_factory=lambda: [0.01, 0.001])
    n_restarts: int = 2


@dataclass
class InitScaleVariant:
    """A diagnostic variant (never graded, never the default): both weight matrices start at
    ``init_scale`` × SPEC §3's Xavier init (D-R01).

    The default 0.187 = √((n + m)/(n·m)) for n = 100, m = 40 gives the standard deviation
    √(2/(n·m)) of a Kaiming-uniform init whose fan-in is the whole m × n matrix: the scale of
    the ARENA reference implementation's |x| model, consulted for diagnosis only (D-R01).
    """

    enabled: bool = True
    init_scale: float = 0.187
    feature_probabilities: list[float] = field(default_factory=lambda: [0.01, 0.001])
    n_restarts: int = 4


@dataclass
class AbsComputationConfig(ExperimentConfig):
    """E08 settings; ``train`` is shared by both parts."""

    no_superposition: NoSuperpositionSetup = field(default_factory=NoSuperpositionSetup)
    sweep: SparsitySweepSetup = field(default_factory=SparsitySweepSetup)
    bias_variant: BiasInitVariant = field(default_factory=BiasInitVariant)
    init_variant: InitScaleVariant = field(default_factory=InitScaleVariant)
    thresholds: AbsThresholds = field(default_factory=AbsThresholds)


# --- analysis -----------------------------------------------------------------------------------


def neuron_feature_counts(
    W1n: Float[Tensor, "inst hidden feat"],
    W2n: Float[Tensor, "inst feat hidden"],
    thr: AbsThresholds,
) -> Tensor:
    """Features per neuron from normalized W₁ (0 = dead, 1 = mono, ≥ 2 = polysemantic).

    After normalization every live neuron's largest |W₁| weight is 1, so the relative
    threshold is absolute here. A neuron whose outgoing weights (normalized W₂ column) are all
    below ``thr.dead_output``, or whose W₁ row is zero, affects no output and counts as dead.
    """
    counts = metrics.neuron_polysemanticity(W1n, thr.rel_threshold, abs_threshold=1e-6)
    silent = W2n.abs().amax(dim=1) < thr.dead_output  # (inst, hidden)
    return torch.where(silent, torch.zeros_like(counts), counts)


def side_roles(W1n: np.ndarray, W2n: np.ndarray, thr: AbsThresholds) -> list[str]:
    """Label each neuron of one model as positive-side(i), negative-side(i) or other.

    Paper: p. 36. A side neuron reads one feature (every other |W₁| weight below
    ``rel_threshold``), and writes positively to that feature's output (every other output
    weight below ``rel_threshold`` times it). The sign of its W₁ weight is the side.
    """
    roles = []
    for h in range(W1n.shape[0]):
        row = W1n[h]
        i = int(np.abs(row).argmax())
        others_in = np.delete(np.abs(row), i)
        out = W2n[:, h]
        others_out = np.delete(np.abs(out), i)
        reads_one = bool((others_in < thr.rel_threshold * abs(row[i])).all())
        writes_own = out[i] > 0 and bool((others_out < thr.rel_threshold * out[i]).all())
        if reads_one and writes_own:
            roles.append(f"{'positive' if row[i] > 0 else 'negative'}-side({i})")
        else:
            roles.append("other")
    return roles


def primary_secondary(W1n: np.ndarray, counts: np.ndarray, ratio: float) -> int:
    """Polysemantic neurons whose largest |W₁| weight is ≥ ``ratio`` × the next (p. 38)."""
    magnitude = np.sort(np.abs(W1n), axis=1)[:, ::-1]
    has_primary = magnitude[:, 0] >= ratio * magnitude[:, 1]
    return int(((counts >= 2) & has_primary).sum())


def side_gain(W1: Tensor, W2: Tensor) -> Tensor:
    """Gain of each feature side through the network, ``(inst, feat, 2)`` for (+, −).

    For an input with only feature i active, xᵢ = s·v (v > 0, s = ±1), the output's
    pre-activation is v · Σₕ W₂[i, h] ReLU(s·W₁[h, i]) + bᵢ. With bᵢ = 0 at initialization,
    a side whose gain is ≤ 0 gets no gradient through the output ReLU from its own examples.
    """
    gains = [
        einops.einsum(W2, torch.relu(sign * W1), "inst feat hidden, inst hidden feat -> inst feat")
        for sign in (1.0, -1.0)
    ]
    return torch.stack(gains, dim=-1)


@torch.no_grad()
def side_errors(model: AbsModel, n_values: int = 64) -> Tensor:
    """1-sparse relative error of each feature side, ``(inst, feat, 2)`` for (+, −).

    Only feature i active with value ±v on a grid of v ∈ (0, 1]: mean (v − y′ᵢ)² divided by
    mean v², so 0 = side computed exactly and 1 = side not computed at all. At very sparse
    settings almost every active input is 1-sparse, so this is where per-feature loss comes from.
    """
    n_inst, n_feat = model.n_inst, model.n_features
    v = torch.linspace(1.0 / n_values, 1.0, n_values)
    errors = []
    for sign in (1.0, -1.0):
        eye = torch.eye(n_feat)  # (active feature, feat)
        x = sign * v[:, None, None] * eye[None]  # (value, active, feat)
        x = x.reshape(-1, 1, n_feat).expand(-1, n_inst, -1)
        out = model(x).reshape(n_values, n_feat, n_inst, n_feat)
        own = out.diagonal(dim1=1, dim2=3)  # (value, inst, feat)
        errors.append(((v[:, None, None] - own) ** 2).mean(dim=0) / (v**2).mean())
    return torch.stack(errors, dim=-1)


def side_diagnostics(trained: TrainedAbs, computed_max: float) -> list[dict]:
    """Per sparsity: are the feature sides that end up computed the ones alive at init?

    A side counts as computed when its 1-sparse relative error is below ``computed_max``, and
    as alive at init when its initial gain (:func:`side_gain`) is positive. Docs: D-409.
    """
    alive = side_gain(trained.W1_init, trained.W2_init) > 0
    computed = side_errors(trained.model()) < computed_max
    rows = []
    for c, p in enumerate(trained.feature_probability.tolist()):
        a, d = alive[c], computed[c]
        rows.append(
            {
                "feature_probability": p,
                "sides_alive_at_init": int(a.sum()),
                "computed_if_alive": f"{int((a & d).sum())}/{int(a.sum())}",
                "computed_if_dead": f"{int((~a & d).sum())}/{int((~a).sum())}",
                "features_both_sides": int(d.all(dim=-1).sum()),
                "features_one_side": int((d.sum(dim=-1) == 1).sum()),
                "features_no_side": int((~d).all(dim=-1).sum()),
            }
        )
    return rows


@dataclass
class TrainedAbs:
    """Best restarts of one set of absolute-value models (CPU tensors)."""

    W1: Tensor  # (cond, hidden, feat)
    W2: Tensor  # (cond, feat, hidden)
    b: Tensor  # (cond, feat)
    feature_probability: Tensor  # (cond,)
    final_loss: Tensor  # (cond,)
    all_final_loss: Tensor  # (inst,)
    condition: Tensor  # (inst,)
    importance: Tensor  # (feat,)
    wall_time_s: float
    W1_init: Tensor  # (cond, hidden, feat): the best restarts' initial weights
    W2_init: Tensor  # (cond, feat, hidden)

    def model(self) -> AbsModel:
        n_cond, n_hidden, n_feat = self.W1.shape
        model = AbsModel(n_feat, n_hidden, n_cond)
        with torch.no_grad():
            model.W1.copy_(self.W1)
            model.W2.copy_(self.W2)
            model.b.copy_(self.b)
        return model


def train_abs(
    n_features: int,
    n_hidden: int,
    importance_decay: float,
    probabilities: list[float],
    n_restarts: int,
    train: TrainConfig,
    seed: int,
    device: torch.device,
    bias_init: float = 0.0,
    init_scale: float = 1.0,
) -> TrainedAbs:
    """Train ``len(probabilities) × n_restarts`` absolute-value models; keep the best of each.

    ``bias_init`` ≠ 0 and ``init_scale`` ≠ 1 are only used by the diagnostic variants; SPEC §3
    initializes biases to 0 and weights with the Xavier init.
    """
    condition = torch.arange(len(probabilities)).repeat_interleave(n_restarts)
    p = torch.tensor(probabilities, dtype=torch.float32)[condition]
    n_inst = len(condition)
    model = AbsModel(n_features, n_hidden, n_inst, seed=seed)
    with torch.no_grad():
        model.b.fill_(bias_init)
        model.W1.mul_(init_scale)
        model.W2.mul_(init_scale)
    model = model.to(device)
    W1_init, W2_init = model.W1.detach().cpu().clone(), model.W2.detach().cpu().clone()
    dist = IndependentSparse(p, n_inst, n_features, low=-1.0, high=1.0)
    importance = geometric_importance(n_features, importance_decay)
    result = train_from_config(model, dist, importance, train, seed, progress=True)
    best = select_best(result.final_loss, condition)
    return TrainedAbs(
        W1=model.W1.detach().cpu()[best],
        W2=model.W2.detach().cpu()[best],
        b=model.b.detach().cpu()[best],
        feature_probability=torch.tensor(probabilities, dtype=torch.float32),
        final_loss=result.final_loss[best],
        all_final_loss=result.final_loss,
        condition=condition,
        importance=importance,
        wall_time_s=result.wall_time_s,
        W1_init=W1_init[best],
        W2_init=W2_init[best],
    )


def loss_ratios(trained: TrainedAbs, n_samples: int, seed: int) -> Tensor:
    """Per-feature loss divided by the loss of predicting the mean, ``(cond, feat)``.

    The mean-prediction baseline is Var(|xᵢ|) = p/3 − p²/4 (|xᵢ| is U[0, 1] when active);
    importance cancels in the ratio.
    """
    generator = make_generator(derive_seed(seed, EVAL_STREAM, 4))
    per_feature = metrics.per_feature_loss(
        trained.model(), trained.feature_probability, n_samples, generator, low=-1.0, high=1.0
    )
    baseline = uniform_feature_variance(trained.feature_probability)
    return per_feature / torch.as_tensor(baseline)[:, None]


def model_losses(ratios: Tensor, importance: Tensor, probabilities: Tensor) -> Tensor:
    """Each model's importance-weighted loss Σᵢ Iᵢ ℓᵢ, ``(cond,)``, from its exact per-feature
    losses ℓᵢ = ratioᵢ · V(p) (more precise than a plain Monte Carlo loss when sparse)."""
    baseline = torch.as_tensor(uniform_feature_variance(probabilities))
    return (ratios * baseline[:, None] * importance).sum(dim=-1)


def reference_losses(
    importance: Tensor, probabilities: Tensor, n_hidden: int
) -> tuple[Tensor, Tensor]:
    """Two reference losses per sparsity, ``(cond,)`` each.

    Footnote 17: losses are comparable only within one sparsity, so each model is compared with
    references at its own sparsity. *Mean*: every feature predicted by its mean, Σᵢ Iᵢ V(p).
    *No superposition*: both sides of the m/2 most important features computed exactly (two
    neurons each, p. 36) and every other feature at its mean, Σ_{i ≥ m/2} Iᵢ V(p). Here
    V(p) = p/3 − p²/4 and ``importance`` is sorted most important first.
    """
    baseline = torch.as_tensor(uniform_feature_variance(probabilities))
    return baseline * importance.sum(), baseline * importance[n_hidden // 2 :].sum()


# --- run ----------------------------------------------------------------------------------------


def run(
    config: AbsComputationConfig, run_dir: RunDir, device: torch.device, logger: logging.Logger
) -> list[Criterion]:
    thr = config.thresholds
    a = config.no_superposition
    logger.info(f"(a) n={a.n_features}, m={a.n_hidden}, 1−S={a.feature_probability:g}")
    small = train_abs(
        a.n_features,
        a.n_hidden,
        a.importance_decay,
        [a.feature_probability],
        a.n_restarts,
        config.train,
        config.seed,
        device,
    )
    run_dir.save_checkpoint("n3_m6", vars(small))
    criteria = evaluate_small(config, small, loss_ratios(small, thr.loss_samples, config.seed))

    s = config.sweep
    logger.info(
        f"(b) n={s.n_features}, m={s.n_hidden}: {len(s.feature_probabilities)} sparsities × "
        f"{s.n_restarts} restarts"
    )
    sweep = train_abs(
        s.n_features,
        s.n_hidden,
        s.importance_decay,
        s.feature_probabilities,
        s.n_restarts,
        config.train,
        config.seed,
        device,
    )
    logger.info(f"    trained in {sweep.wall_time_s:.1f} s")
    run_dir.save_checkpoint("n100_m40", vars(sweep))
    ratios = loss_ratios(sweep, thr.loss_samples, config.seed)
    summary = summarize_sweep(config, sweep, ratios)
    run_dir.write_table(summary, "metrics")
    run_dir.write_table(
        pd.DataFrame(
            ratios.numpy(),
            index=[format_probability(float(p)) for p in sweep.feature_probability],
        ).reset_index(names="feature_probability"),
        "per_feature_loss_ratio",
    )
    criteria += evaluate_sweep(config, summary)
    sides = side_diagnostics(sweep, thr.side_computed_max)
    criteria.append(
        info(
            "b_side_diagnostics",
            "(b) Which feature sides get computed (1-sparse relative error < 25%), split by "
            "whether the side's gain Σₕ W₂[i,h]·ReLU(±W₁[h,i]) was positive at initialization. "
            "Meaningful where inputs are mostly 1-sparse (1 − S ≤ 0.01)",
            PAGE_B,
            {f"1-S={format_probability(row['feature_probability'])}": row for row in sides},
            group="b_n100_m40",
        )
    )
    v = config.bias_variant
    if v.enabled:
        logger.info(
            f"(b) bias-init variant: b₀ = {v.bias_init:g} at 1−S = {v.feature_probabilities}"
        )
        criteria.append(
            run_variant(
                config,
                sweep,
                ratios,
                sides,
                device,
                "b_bias_init_variant",
                f"(b) Diagnostic variant, not the replication (D-409): output bias initialized to "
                f"{v.bias_init:g} instead of 0 ({v.n_restarts} restarts), against the default runs",
                v.feature_probabilities,
                v.n_restarts,
                bias_init=v.bias_init,
            )
        )
    w = config.init_variant
    if w.enabled:
        logger.info(
            f"(b) init-scale variant: {w.init_scale:g}× Xavier at 1−S = {w.feature_probabilities}"
        )
        criteria.append(
            run_variant(
                config,
                sweep,
                ratios,
                sides,
                device,
                "b_init_scale_variant",
                f"(b) Diagnostic variant, not the replication (D-R01): both weight matrices "
                f"initialized at {w.init_scale:g}× the Xavier init, the scale of the ARENA "
                f"reference |x| model ({w.n_restarts} restarts), against the default runs",
                w.feature_probabilities,
                w.n_restarts,
                init_scale=w.init_scale,
            )
        )
    return criteria


def run_variant(
    config: AbsComputationConfig,
    sweep: TrainedAbs,
    ratios: Tensor,
    sides: list[dict],
    device: torch.device,
    id: str,
    description: str,
    probabilities: list[float],
    n_restarts: int,
    bias_init: float = 0.0,
    init_scale: float = 1.0,
) -> Criterion:
    """Retrain a few sparse settings with a different initialization (diagnostic only) and
    compare them with the default runs at the same sparsities (D-409, D-R01)."""
    s, thr = config.sweep, config.thresholds
    variant = train_abs(
        s.n_features,
        s.n_hidden,
        s.importance_decay,
        probabilities,
        n_restarts,
        config.train,
        config.seed,
        device,
        bias_init=bias_init,
        init_scale=init_scale,
    )
    variant_ratios = loss_ratios(variant, thr.loss_samples, config.seed)
    variant_sides = side_diagnostics(variant, thr.side_computed_max)
    default_loss = model_losses(ratios, sweep.importance, sweep.feature_probability)
    variant_loss = model_losses(variant_ratios, variant.importance, variant.feature_probability)
    probs = sweep.feature_probability.tolist()
    report = {}
    for c, p in enumerate(probabilities):
        base = int(np.argmin(np.abs(np.array(probs) - p)))
        report[f"1-S={format_probability(p)}"] = {
            "loss_default": round(float(default_loss[base]), 6),
            "loss_variant": round(float(variant_loss[c]), 6),
            "computed_well_default": int((ratios[base] < thr.computed_well).sum()),
            "computed_well_variant": int((variant_ratios[c] < thr.computed_well).sum()),
            "usefully_computed_default": int((ratios[base] < thr.usefully_computed).sum()),
            "usefully_computed_variant": int((variant_ratios[c] < thr.usefully_computed).sum()),
            "dead_sides_computed_default": sides[base]["computed_if_dead"],
            "dead_sides_computed_variant": variant_sides[c]["computed_if_dead"],
            "alive_sides_computed_variant": variant_sides[c]["computed_if_alive"],
            "both_sides_variant": variant_sides[c]["features_both_sides"],
            "one_side_variant": variant_sides[c]["features_one_side"],
        }
    return info(id, description, PAGE_B, report, group="b_n100_m40")


def evaluate_small(
    config: AbsComputationConfig, small: TrainedAbs, ratios: Tensor
) -> list[Criterion]:
    """E08(a) criteria."""
    thr = config.thresholds
    W1n, W2n = metrics.normalize_hidden_units(small.W1, small.W2)
    roles = side_roles(W1n[0].numpy(), W2n[0].numpy(), thr)
    n = config.no_superposition.n_features
    per_feature = {
        i: {
            "positive": sum(r == f"positive-side({i})" for r in roles),
            "negative": sum(r == f"negative-side({i})" for r in roles),
        }
        for i in range(n)
    }
    ok_roles = all(v["positive"] == 1 and v["negative"] == 1 for v in per_feature.values())
    ratio = ratios[0]
    return [
        check(
            "a_positive_and_negative_side_neurons",
            f"(a) n = {n}, m = {config.no_superposition.n_hidden}: each feature gets one "
            "positive-side and one negative-side neuron (after footnote-15 normalization)",
            PAGE_A,
            ok_roles,
            {"roles": roles, "per_feature": per_feature},
            "exactly one of each per feature",
            group="a_n3_m6",
        ),
        check(
            "a_per_feature_loss",
            f"(a) Every per-feature loss is below {thr.per_feature_loss_max:.0%} of the loss of "
            "predicting the mean",
            PAGE_A,
            bool((ratio < thr.per_feature_loss_max).all()),
            [round(float(v), 6) for v in ratio],
            f"< {thr.per_feature_loss_max}",
            group="a_n3_m6",
        ),
        info(
            "a_weights",
            "(a) Normalized W₁ rows (per neuron), W₂ columns and b of the best model",
            PAGE_A,
            {
                "W1": W1n[0].double().numpy().round(3),
                "W2": W2n[0].T.double().numpy().round(3),
                "b": small.b[0].double().numpy().round(4),
                "loss": round(float(small.final_loss[0]), 7),
            },
            group="a_n3_m6",
        ),
    ]


def summarize_sweep(
    config: AbsComputationConfig, sweep: TrainedAbs, ratios: Tensor
) -> pd.DataFrame:
    """One row per sparsity (best restart)."""
    thr = config.thresholds
    W1n, W2n = metrics.normalize_hidden_units(sweep.W1, sweep.W2)
    counts = neuron_feature_counts(W1n, W2n, thr)
    loss = model_losses(ratios, sweep.importance, sweep.feature_probability)
    mean_ref, no_superposition_ref = reference_losses(
        sweep.importance, sweep.feature_probability, config.sweep.n_hidden
    )
    rows = []
    for c in range(len(sweep.feature_probability)):
        well = ratios[c] < thr.computed_well
        useful = ratios[c] < thr.usefully_computed
        rows.append(
            {
                "feature_probability": float(sweep.feature_probability[c]),
                "loss": float(sweep.final_loss[c]),
                "loss_exact": float(loss[c]),
                "loss_mean_reference": float(mean_ref[c]),
                "loss_no_superposition_reference": float(no_superposition_ref[c]),
                "computed_well": int(well.sum()),
                "computed_well_features": torch.nonzero(well).flatten().tolist(),
                "usefully_computed": int(useful.sum()),
                "usefully_computed_features": torch.nonzero(useful).flatten().tolist(),
                "monosemantic": int((counts[c] == 1).sum()),
                "polysemantic": int((counts[c] >= 2).sum()),
                "dead": int((counts[c] == 0).sum()),
                "primary_plus_secondary": primary_secondary(
                    W1n[c].numpy(), counts[c].numpy(), thr.primary_ratio
                ),
                "max_features_per_neuron": int(counts[c].max()),
            }
        )
    return pd.DataFrame(rows)


def evaluate_sweep(config: AbsComputationConfig, summary: pd.DataFrame) -> list[Criterion]:
    """E08(b) criteria: dense monosemantic neurons, and (D-R01) more features usefully computed
    than neurons at the sparsest setting, with SPEC's original criterion reported beside it."""
    thr = config.thresholds
    m = config.sweep.n_hidden
    labels = [format_probability(p) for p in summary["feature_probability"]]
    records = summary.to_dict("records")
    dense, sparse = records[0], records[-1]
    return [
        check(
            "b_dense_monosemantic",
            f"(b) Dense (1 − S = {labels[0]}): every neuron is monosemantic",
            PAGE_B,
            int(dense["monosemantic"]) == m,
            {
                "monosemantic": int(dense["monosemantic"]),
                "polysemantic": int(dense["polysemantic"]),
                "dead": int(dense["dead"]),
            },
            f"{m}/{m} neurons carry exactly 1 feature",
            group="b_n100_m40",
        ),
        check(
            "b_sparse_more_features_than_neurons",
            f"(b) Sparse (1 − S = {labels[-1]}): the model represents more features than it has "
            f"neurons while usefully approximating |x|: more than m = {m} features usefully "
            f"computed (per-feature loss < {thr.usefully_computed:.0%} of predicting the mean; "
            "p. 37, D-R01)",
            PAGE_B,
            int(sparse["usefully_computed"]) > m,
            {
                "usefully_computed": int(sparse["usefully_computed"]),
                "polysemantic_neurons": int(sparse["polysemantic"]),
                "features": list(sparse["usefully_computed_features"]),
            },
            f"> {m}",
            group="b_n100_m40",
        ),
        info(
            "b_sparse_original_criterion",
            f"(b) SPEC's original sparse criterion, no longer graded (D-R01): more than m/2 = "
            f"{m // 2} features computed well (per-feature loss < {thr.computed_well:.0%} of "
            f"predicting the mean) at 1 − S = {labels[-1]}",
            PAGE_B,
            {
                "computed_well": int(sparse["computed_well"]),
                "threshold": f"> {m / 2:g}",
                "would_pass": int(sparse["computed_well"]) > m / 2,
            },
            group="b_n100_m40",
        ),
        info(
            "b_per_sparsity_summary",
            "(b) Best model per sparsity: features usefully computed and computed well, neuron "
            f"counts, polysemantic neurons with a primary feature (≥ {thr.primary_ratio:g}× the "
            "next weight, p. 38), the most features on one neuron, loss",
            PAGE_B,
            {
                f"1-S={lab}": {
                    "usefully_computed": int(row["usefully_computed"]),
                    "computed_well": int(row["computed_well"]),
                    "mono": int(row["monosemantic"]),
                    "poly": int(row["polysemantic"]),
                    "dead": int(row["dead"]),
                    "primary_plus_secondary": int(row["primary_plus_secondary"]),
                    "max_features_per_neuron": int(row["max_features_per_neuron"]),
                    "loss": round(float(row["loss"]), 5),
                }
                for lab, row in zip(labels, records, strict=True)
            },
            group="b_n100_m40",
        ),
        info(
            "b_within_sparsity_references",
            "(b) Each best model's loss (exact per-feature estimator) against two references at "
            "the same sparsity (footnote 17): every feature at its mean, and no superposition "
            f"(both sides of the m/2 = {m // 2} most important features exact, the rest at "
            "their mean)",
            PAGE_B,
            {
                f"1-S={lab}": {
                    "loss": float(f"{row['loss_exact']:.4g}"),
                    "mean_reference": float(f"{row['loss_mean_reference']:.4g}"),
                    "no_superposition_reference": float(
                        f"{row['loss_no_superposition_reference']:.4g}"
                    ),
                    "loss_over_no_superposition": round(
                        float(row["loss_exact"] / row["loss_no_superposition_reference"]), 3
                    ),
                }
                for lab, row in zip(labels, records, strict=True)
            },
            group="b_n100_m40",
        ),
    ]


# --- figures ------------------------------------------------------------------------------------


def _load(run_dir: RunDir, name: str) -> TrainedAbs:
    return TrainedAbs(**run_dir.load_checkpoint(name))


def figures(run_dir: RunDir) -> list[Path]:
    """(a) the p. 36 raw weights; (b) the p. 37 stack plots and a capacity summary."""
    config = from_dict(AbsComputationConfig, yaml.safe_load(run_dir.config_path.read_text()))
    thr = config.thresholds
    paths: list[Path] = []

    small = _load(run_dir, "n3_m6")
    W1n, W2n = metrics.normalize_hidden_units(small.W1, small.W2)
    roles = side_roles(W1n[0].numpy(), W2n[0].numpy(), thr)
    order = np.lexsort(([r.startswith("negative") for r in roles], neuron_top(W1n[0].numpy())))
    paths += save_figure(
        abs_weights_figure(
            W1n[0].numpy(),
            W2n[0].numpy(),
            order,
            roles,
            title=f"E08(a) · |x| with n = {W1n.shape[2]}, m = {W1n.shape[1]} "
            "(footnote-15 normalized)",
        ),
        run_dir.figures_dir / "abs_n3_m6",
    )

    sweep = _load(run_dir, "n100_m40")
    W1n, W2n = metrics.normalize_hidden_units(sweep.W1, sweep.W2)
    counts = neuron_feature_counts(W1n, W2n, thr)
    score = metrics.superposition_score(W1n)
    panels = [
        NeuronPanel(
            title=f"1−S = {format_probability(float(p))}",
            W=W1n[c].numpy(),
            superposition=score[c].numpy(),
            polysemantic=counts[c].numpy(),
            order=neuron_order_by_top_feature(W1n[c].numpy()),
        )
        for c, p in enumerate(sweep.feature_probability)
    ]
    paths += save_figure(
        neuron_rows(
            panels,
            suptitle=f"E08(b) · |W₁| per neuron, n = {W1n.shape[2]}, m = {W1n.shape[1]}, "
            "neurons sorted by their largest feature's importance",
            absolute=True,
            show_matrix=False,
        ),
        run_dir.figures_dir / "abs_stack_plots",
    )
    summary = run_dir.read_table("metrics")
    fig = capacity_figure(
        summary["feature_probability"].to_numpy(),
        summary["computed_well"].to_numpy(),
        summary["monosemantic"].to_numpy(),
        summary["polysemantic"].to_numpy(),
        int(W1n.shape[1]),
        title="E08(b) · computing |x| in superposition",
        usefully_computed=summary["usefully_computed"].to_numpy()
        if "usefully_computed" in summary
        else None,
    )
    paths += save_figure(fig, run_dir.figures_dir / "abs_capacity")
    return paths


def neuron_top(W1n: np.ndarray) -> np.ndarray:
    """Index of each neuron's largest-|W₁| feature."""
    return np.abs(W1n).argmax(axis=1)


EXPERIMENT = Experiment(
    name="abs_computation",
    code="E08",
    title="Computing |x| with ReLU neurons: side neurons (n = 3, m = 6) and superposition "
    "(n = 100, m = 40)",
    paper_pages="pp. 34–39",
    config_cls=AbsComputationConfig,
    run=run,
    figures=figures,
)
