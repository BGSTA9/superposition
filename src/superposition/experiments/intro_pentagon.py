"""E00 intro_pentagon: five features in two dimensions, as sparsity increases.

Paper: introduction figure, p. 2 ("As Sparsity Increases, Models Use 'Superposition' To
Represent More Features Than Dimensions"); bias behaviour, p. 10; linear baseline, pp. 10–13.

Claim (p. 2), for the ReLU output model with n = 5 features in m = 2 dimensions:

* 0% sparsity: the two most important features get dedicated orthogonal dimensions; the
  other three are not embedded.
* 80% sparsity: the four most important features are stored as antipodal pairs; the least
  important is not embedded.
* 90% sparsity: all five features are embedded as a pentagon, with positive interference.

The linear model, by contrast, represents only the top m = 2 features at every sparsity, and
its optimal loss is known in closed form (theory.linear_optimum_loss).

Importance is not stated for this figure; Iᵢ = 0.9ⁱ reproduces it (SPEC §3). Acceptance
criteria and thresholds follow SPEC §7 E00; they live in the config, not in the code.
"""

from __future__ import annotations

import itertools
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from superposition import metrics
from superposition.acceptance import Criterion, check, info
from superposition.config import ExperimentConfig
from superposition.data import IndependentSparse, geometric_importance
from superposition.experiments.base import Experiment
from superposition.experiments.common import (
    GridResult,
    antipodal_pairs,
    feature_table,
    instance_table,
    represented_features,
    train_sparsity_grid,
)
from superposition.models import TiedModel
from superposition.plotting.arrows import arrows_grid
from superposition.plotting.style import (
    INK_SECONDARY,
    MODEL_COLORS,
    MODEL_LABELS,
    paper_style,
    save_figure,
)
from superposition.plotting.weights import WeightsPanel, weights_grid
from superposition.runs import RunDir
from superposition.theory import linear_loss_decomposition, linear_optimum_loss
from superposition.train import evaluate_loss, train
from superposition.utils import PROBE_STREAM, derive_seed, make_generator

PAGE = "p. 2"


@dataclass
class IntroPentagonThresholds:
    """Acceptance thresholds (SPEC §7 E00)."""

    represented_norm: float = 0.5
    orthogonal_abs_cos: float = 0.1
    antipodal_cos: float = -0.95
    pentagon_gap_deg: float = 72.0
    pentagon_gap_tol_deg: float = 10.0
    pentagon_min_norm: float = 1.0
    unrepresented_norm: float = 0.05
    bias_tol: float = 0.02
    linear_optimum_rel_tol: float = 0.03
    superposition_score: float = 0.05


@dataclass
class PentagonDiagnostics:
    """Report-only diagnostics of the S = 0.9 pentagon's loss (INFO rows, no thresholds).

    They exist because the best pentagon's loss sits above the SPEC's reference value; see
    docs/REPRODUCTION.md (E00). ``precise_eval_samples`` re-evaluates the best restart;
    ``noise_repeats`` × ``noise_eval_samples`` measures how noisy a small-sample estimate is;
    every cyclic arrangement of the five features is trained from an ideal pentagon start for
    ``arrangement_steps`` steps to find the best arrangement's loss.
    """

    enabled: bool = True
    sparsity: float = 0.9
    precise_eval_samples: int = 1_000_000
    noise_eval_samples: int = 20_000
    noise_repeats: int = 200
    arrangement_steps: int = 10_000
    arrangement_init_norm: float = 1.1
    arrangement_init_bias: float = -0.2


@dataclass
class IntroPentagonConfig(ExperimentConfig):
    """E00 settings. ``sparsities`` are S values; the code trains on p = 1 − S."""

    n_features: int = 5
    n_hidden: int = 2
    importance_decay: float = 0.9
    sparsities: list[float] = field(default_factory=lambda: [0.0, 0.8, 0.9])
    n_restarts: int = 8
    models: list[str] = field(default_factory=lambda: ["relu_output", "linear"])
    # Reference losses from an independent verification run (SPEC §7 E00); sanity checks only.
    reference_relu_loss: list[float] = field(default_factory=lambda: [0.182, 0.059, 0.0196])
    reference_linear_loss: list[float] = field(default_factory=lambda: [0.182, 0.122, 0.067])
    thresholds: IntroPentagonThresholds = field(default_factory=IntroPentagonThresholds)
    diagnostics: PentagonDiagnostics = field(default_factory=PentagonDiagnostics)


# --- geometry helpers (one instance) ---------------------------------------------------------


def cyclic_order(W: torch.Tensor) -> tuple[int, ...]:
    """Features in angular order around the circle, starting at feature 0.

    Of the two directions of travel, the one whose second entry is smaller is returned, so
    reflections of the same arrangement compare equal.
    """
    angles = metrics.angles_2d(W[None])[0].numpy()
    order = [int(i) for i in np.argsort(angles)]
    start = order.index(0)
    ccw = order[start:] + order[:start]
    cw = [ccw[0], *reversed(ccw[1:])]
    return tuple(min(ccw, cw, key=lambda seq: seq[1]))


def all_cyclic_orders(n: int) -> list[tuple[int, ...]]:
    """Every arrangement of n points on a cycle, up to rotation and reflection."""
    orders = []
    for perm in itertools.permutations(range(1, n)):
        if perm[0] < perm[-1]:  # keep one of each reflected pair
            orders.append((0, *perm))
    return orders


def _cos(W: torch.Tensor) -> torch.Tensor:
    return metrics.cosine_similarity_matrix(W[None])[0]


def sorted_angular_gaps_deg(W: torch.Tensor) -> list[float]:
    """Gaps between consecutive feature directions around the circle, in degrees (sum 360)."""
    angles = np.sort(np.degrees(metrics.angles_2d(W[None])[0].numpy()) % 360.0)
    gaps = np.diff(np.concatenate([angles, [angles[0] + 360.0]]))
    return sorted(float(g) for g in gaps)


# --- run -------------------------------------------------------------------------------------


def run(
    config: IntroPentagonConfig, run_dir: RunDir, device: torch.device, logger: logging.Logger
) -> list[Criterion]:
    importance = geometric_importance(config.n_features, config.importance_decay)
    probabilities = [1.0 - s for s in config.sparsities]
    grids: dict[str, GridResult] = {}
    for kind in config.models:
        logger.info(
            f"training {kind}: {len(probabilities)} sparsities × {config.n_restarts} restarts"
        )
        grid = train_sparsity_grid(
            kind,  # type: ignore[arg-type]
            config.n_features,
            config.n_hidden,
            importance,
            probabilities,
            config.n_restarts,
            config.train,
            seed=config.seed,
            device=device,
            progress=True,
        )
        logger.info(f"  done in {grid.wall_time_s:.1f} s")
        run_dir.save_checkpoint(kind, grid.to_checkpoint())
        grids[kind] = grid
    thr = config.thresholds
    run_dir.write_table(
        pd.concat(
            [
                instance_table(g, thr.represented_norm, thr.superposition_score)
                for g in grids.values()
            ]
        ),
        "metrics",
    )
    run_dir.write_table(pd.concat([feature_table(g) for g in grids.values()]), "features")
    criteria = evaluate(config, grids)
    if config.diagnostics.enabled and "relu_output" in grids:
        criteria += pentagon_diagnostics(config, grids["relu_output"], device, logger)
    return criteria


def pentagon_diagnostics(
    config: IntroPentagonConfig, relu: GridResult, device: torch.device, logger: logging.Logger
) -> list[Criterion]:
    """INFO rows probing the S = 0.9 loss: precise loss, small-sample noise, arrangements."""
    diag = config.diagnostics
    if diag.sparsity not in config.sparsities:
        return []
    cond = config.sparsities.index(diag.sparsity)
    p = 1.0 - diag.sparsity
    importance = relu.importance.to(device)
    W, b, loss = relu.best_weights(cond)
    best = TiedModel.from_kind("relu_output", config.n_features, config.n_hidden, 1).to(device)
    with torch.no_grad():
        best.W.copy_(W[None])
        best.b.copy_(b[None])
    dist = IndependentSparse(p, 1, config.n_features)
    seed = derive_seed(config.seed, PROBE_STREAM)
    precise = evaluate_loss(
        best, dist, importance, diag.precise_eval_samples, make_generator(seed, device)
    ).item()
    logger.info(
        f"diagnostics: best pentagon loss on {diag.precise_eval_samples:,} samples = {precise:.5f}"
    )
    estimates = torch.tensor(
        [
            evaluate_loss(
                best,
                dist,
                importance,
                diag.noise_eval_samples,
                make_generator(seed + 1 + r, device),
            ).item()
            for r in range(diag.noise_repeats)
        ]
    )
    reference = config.reference_relu_loss[cond]

    orders = all_cyclic_orders(config.n_features)
    arranged = TiedModel.from_kind("relu_output", config.n_features, config.n_hidden, len(orders))
    with torch.no_grad():
        arranged.b.fill_(diag.arrangement_init_bias)
        for k, order in enumerate(orders):
            for slot, feature in enumerate(order):
                angle = 2 * math.pi * slot / config.n_features
                arranged.W[k, :, feature] = diag.arrangement_init_norm * torch.tensor(
                    [math.cos(angle), math.sin(angle)]
                )
    arranged = arranged.to(device)
    logger.info(
        f"diagnostics: training all {len(orders)} cyclic arrangements from an ideal pentagon"
    )
    arrangement_result = train(
        arranged,
        IndependentSparse(p, len(orders), config.n_features),
        importance,
        steps=diag.arrangement_steps,
        batch_size=config.train.batch_size,
        lr=config.train.lr,
        eval_samples=diag.precise_eval_samples,
        seed=seed,
    )
    arranged_W = arranged.W.detach().cpu()
    still_pentagon = metrics.represented_mask(arranged_W, config.thresholds.represented_norm).all(
        -1
    )
    losses = {
        "".join(map(str, order)): round(float(arrangement_result.final_loss[k]), 5)
        for k, order in enumerate(orders)
        if bool(still_pentagon[k])
    }
    best_order = cyclic_order(W)
    return [
        info(
            "pentagon_loss_precise",
            f"S = {diag.sparsity:g} best restart re-evaluated on {diag.precise_eval_samples:,} samples",
            PAGE,
            {
                "loss_100k_selection_eval": round(loss, 5),
                "loss_precise": round(precise, 5),
                "reference": reference,
            },
            group="relu_output",
        ),
        info(
            "small_sample_eval_noise",
            f"Spread of {diag.noise_repeats} independent {diag.noise_eval_samples:,}-sample loss estimates of that model",
            PAGE,
            {
                "mean": round(float(estimates.mean()), 5),
                "rel_std_pct": round(float(100 * estimates.std() / estimates.mean()), 2),
                "q2.5": round(float(estimates.quantile(0.025)), 5),
                "q97.5": round(float(estimates.quantile(0.975)), 5),
                "min": round(float(estimates.min()), 5),
                "fraction_at_or_below_reference": round(
                    float((estimates <= reference).float().mean()), 4
                ),
            },
            group="relu_output",
        ),
        info(
            "pentagon_arrangements",
            f"Loss ({diag.precise_eval_samples:,} samples) of each cyclic feature arrangement, trained "
            f"{diag.arrangement_steps:,} steps from an ideal pentagon; best restart's arrangement shown",
            "pp. 2, 22–23",
            {
                "best_restart_arrangement": "".join(map(str, best_order)),
                "min": min(losses.values()) if losses else None,
                "max": max(losses.values()) if losses else None,
                "n_arrangements_still_pentagons": len(losses),
                "by_arrangement": dict(sorted(losses.items(), key=lambda kv: kv[1])),
            },
            group="relu_output",
        ),
    ]


def evaluate(config: IntroPentagonConfig, grids: dict[str, GridResult]) -> list[Criterion]:
    """Turn trained grids into E00's acceptance criteria (SPEC §7)."""
    thr = config.thresholds
    criteria: list[Criterion] = []
    all_features = list(range(config.n_features))
    top_m = list(range(config.n_hidden))

    relu = grids.get("relu_output")
    if relu is not None:
        for cond, s in enumerate(config.sparsities):
            W, b, _ = relu.best_weights(cond)
            rep = represented_features(W, thr.represented_norm)
            cos = _cos(W)
            if s == 0.0:
                abs_cos = abs(float(cos[0, 1]))
                criteria.append(
                    check(
                        "S0_dedicated_orthogonal_pair",
                        "S = 0: exactly features {0, 1} represented, |cos| < 0.1",
                        PAGE,
                        rep == [0, 1] and abs_cos < thr.orthogonal_abs_cos,
                        {"represented": rep, "abs_cos_01": round(abs_cos, 4)},
                        {"represented": [0, 1], "abs_cos_max": thr.orthogonal_abs_cos},
                        group="relu_output",
                    )
                )
            elif s == 0.8:
                pairs = antipodal_pairs(W, thr.represented_norm, thr.antipodal_cos)
                partnered = sorted({i for pair in pairs for i in pair})
                partner_cos = {
                    i: round(float(cos[i, [j for j in rep if j != i]].min()), 4) for i in rep
                }
                criteria.append(
                    check(
                        "S0.8_two_antipodal_pairs",
                        "S = 0.8: exactly {0, 1, 2, 3} represented, each with a partner at cos < −0.95",
                        PAGE,
                        rep == [0, 1, 2, 3] and partnered == rep,
                        {
                            "represented": rep,
                            "antipodal_pairs": pairs,
                            "min_partner_cos": partner_cos,
                        },
                        {"represented": [0, 1, 2, 3], "partner_cos_max": thr.antipodal_cos},
                        group="relu_output",
                    )
                )
            elif s == 0.9:
                gaps = sorted_angular_gaps_deg(W)
                norms = metrics.feature_norms(W[None])[0]
                gaps_ok = all(
                    abs(g - thr.pentagon_gap_deg) <= thr.pentagon_gap_tol_deg for g in gaps
                )
                criteria.append(
                    check(
                        "S0.9_pentagon",
                        "S = 0.9: all 5 represented; angular gaps 72° ± 10°; all norms > 1; all biases < 0",
                        PAGE,
                        rep == all_features
                        and gaps_ok
                        and bool((norms > thr.pentagon_min_norm).all())
                        and bool((b < 0).all()),
                        {
                            "represented": rep,
                            "gaps_deg": [round(g, 1) for g in gaps],
                            "norms": [round(float(v), 3) for v in norms],
                            "biases": [round(float(v), 3) for v in b],
                        },
                        {
                            "gap_deg": f"{thr.pentagon_gap_deg} ± {thr.pentagon_gap_tol_deg}",
                            "norm_min": thr.pentagon_min_norm,
                            "bias_max": 0.0,
                        },
                        group="relu_output",
                    )
                )
        criteria.append(_restart_outcomes(config, relu))

    linear = grids.get("linear")
    if linear is not None:
        for cond, s in enumerate(config.sparsities):
            W, _, _ = linear.best_weights(cond)
            rep = represented_features(W, thr.represented_norm)
            abs_cos = abs(float(_cos(W)[0, 1]))
            criteria.append(
                check(
                    f"S{s:g}_top2_orthogonal",
                    f"S = {s:g}: exactly features {{0, 1}} represented, orthogonal (|cos| < 0.1)",
                    "pp. 10–11",
                    rep == top_m and abs_cos < thr.orthogonal_abs_cos,
                    {"represented": rep, "abs_cos_01": round(abs_cos, 4)},
                    {"represented": top_m, "abs_cos_max": thr.orthogonal_abs_cos},
                    group="linear",
                )
            )
        for cond, s in enumerate(config.sparsities):
            p = 1.0 - s
            W, b, loss = linear.best_weights(cond)
            optimum = linear_optimum_loss(linear.importance, config.n_hidden, p)
            rel = abs(loss - optimum) / optimum
            # The same model's exact expected loss (p. 13): the sampled estimate's error is
            # mostly sampling noise and the winner's curse of picking the lowest of 8 (R1-11).
            exact = float(linear_loss_decomposition(W[None], b[None], linear.importance, p).total)
            criteria.append(
                check(
                    f"S{s:g}_loss_vs_closed_form",
                    f"S = {s:g}: best linear loss within 3% of Σ_{{i≥m}} Iᵢ(p/3 − p²/4)",
                    "p. 13",
                    rel <= thr.linear_optimum_rel_tol,
                    {
                        "loss": round(loss, 5),
                        "optimum": round(optimum, 5),
                        "rel_diff": round(rel, 4),
                        "exact_loss": round(exact, 6),
                        "exact_rel_diff": round(abs(exact - optimum) / optimum, 6),
                    },
                    {"rel_diff_max": thr.linear_optimum_rel_tol},
                    group="linear",
                    notes=f"graded on {config.train.eval_samples:,} fresh samples; exact_loss is "
                    "the same model's exact expected loss (theory.linear_loss_decomposition)",
                )
            )

    for kind, grid in grids.items():
        criteria.append(_bias_check(config, grid, kind))
    for kind, grid in grids.items():
        references = (
            config.reference_relu_loss if kind == "relu_output" else config.reference_linear_loss
        )
        criteria.append(
            info(
                "best_loss_vs_reference",
                "Best-restart loss per S next to the independent reference run (sanity check)",
                PAGE,
                {
                    f"S={s:g}": {"ours": round(grid.best_weights(c)[2], 5), "reference": ref}
                    for c, (s, ref) in enumerate(zip(config.sparsities, references, strict=True))
                },
                group=kind,
            )
        )
    if relu is not None and 0.9 in config.sparsities:
        cond = config.sparsities.index(0.9)
        W, _, _ = relu.best_weights(cond)
        criteria.append(
            info(
                "pentagon_dimensionality",
                "S = 0.9 best model: feature dimensionalities Dᵢ (pentagon: 2/5) and D*",
                "pp. 17–19",
                {
                    "D_i": [round(float(v), 3) for v in metrics.feature_dimensionality(W[None])[0]],
                    "D_star": round(float(metrics.dims_per_feature(W[None])[0]), 3),
                },
                group="relu_output",
            )
        )
    return criteria


def _bias_check(config: IntroPentagonConfig, grid: GridResult, kind: str) -> Criterion:
    """Features with ‖Wᵢ‖ < 0.05 should have bᵢ ≈ E[xᵢ] = (1 − S)/2 (p. 10)."""
    thr = config.thresholds
    rows, ok = [], True
    for cond, s in enumerate(config.sparsities):
        W, b, _ = grid.best_weights(cond)
        norms = metrics.feature_norms(W[None])[0]
        expected = (1.0 - s) / 2
        for i in torch.nonzero(norms < thr.unrepresented_norm).flatten().tolist():
            within = abs(float(b[i]) - expected) <= thr.bias_tol
            ok &= within
            rows.append({"S": s, "feature": i, "bias": round(float(b[i]), 4), "expected": expected})
    return check(
        "unrepresented_bias_is_mean",
        "Every feature with ‖Wᵢ‖ < 0.05 has bias within 0.02 of (1 − S)/2",
        "p. 10",
        ok,
        rows,
        {"norm_max": thr.unrepresented_norm, "abs_diff_max": thr.bias_tol},
        group=kind,
    )


def _restart_outcomes(config: IntroPentagonConfig, grid: GridResult) -> Criterion:
    """How many restarts (not just the best) reach each paper geometry: evidence for p. 23."""
    thr = config.thresholds
    outcomes = {}
    for cond, s in enumerate(config.sparsities):
        counts: dict[str, int] = {}
        for k in torch.nonzero(grid.condition == cond).flatten().tolist():
            n_rep = len(represented_features(grid.W[k], thr.represented_norm))
            key = f"{n_rep} features"
            counts[key] = counts.get(key, 0) + 1
        outcomes[f"S={s:g}"] = dict(sorted(counts.items()))
    return info(
        "restart_outcomes",
        f"Represented-feature counts across all {config.n_restarts} restarts per S",
        "p. 23",
        outcomes,
        group="relu_output",
    )


# --- figures ---------------------------------------------------------------------------------


def _load_grids(run_dir: RunDir) -> dict[str, GridResult]:
    grids = {}
    for path in sorted(run_dir.checkpoints_dir.glob("*.pt")):
        grids[path.stem] = GridResult.from_checkpoint(run_dir.load_checkpoint(path.stem))
    return grids


def figures(run_dir: RunDir) -> list[Path]:
    """Arrows (p. 2), WᵀW + bars (pp. 10–11) and per-restart losses for a saved E00 run."""
    grids = _load_grids(run_dir)
    order = [k for k in ("relu_output", "linear") if k in grids]
    paths: list[Path] = []

    rows = []
    for kind in order:
        grid = grids[kind]
        panels = []
        for cond in range(grid.n_conditions):
            W, _, loss = grid.best_weights(cond)
            s = 1.0 - grid.condition_probability(cond)
            canon = metrics.canonicalize_2d(W[None])[0].numpy()
            n_rep = int(metrics.represented_mask(W[None])[0].sum())
            panels.append((f"S = {s:.1f}", canon, f"{n_rep} features · loss {loss:.4f}"))
        rows.append((MODEL_LABELS[kind], panels))
    fig = arrows_grid(
        rows, suptitle="E00 · five features in two dimensions (best of restarts, canonicalized)"
    )
    paths += save_figure(fig, run_dir.figures_dir / "pentagon_arrows")

    panels_w, labels = [], []
    for kind in order:
        grid = grids[kind]
        labels.append((len(panels_w), f"{MODEL_LABELS[kind]} model"))
        for cond in range(grid.n_conditions):
            W, b, _ = grid.best_weights(cond)
            s = 1.0 - grid.condition_probability(cond)
            panels_w.append(
                WeightsPanel(
                    title=f"S = {s:.1f}",
                    gram=metrics.gram(W[None])[0].numpy(),
                    bias=b.numpy(),
                    norms=metrics.feature_norms(W[None])[0].numpy(),
                    superposition=metrics.superposition_score(W[None])[0].numpy(),
                )
            )
    fig = weights_grid(panels_w, suptitle="E00 · WᵀW, bias and feature norms", group_labels=labels)
    paths += save_figure(fig, run_dir.figures_dir / "weights_best")

    paths += save_figure(_restart_loss_figure(grids, order), run_dir.figures_dir / "restart_losses")
    return paths


def _restart_loss_figure(grids: dict[str, GridResult], order: list[str]):
    """Final loss of every restart, one panel per S (losses across S are not comparable,
    footnote 17), with the closed-form linear optimum as a reference line."""
    first = grids[order[0]]
    n_cond = first.n_conditions
    with paper_style():
        fig, axes = plt.subplots(1, n_cond, figsize=(3.0 * n_cond, 2.8), squeeze=False)
        for cond in range(n_cond):
            ax = axes[0, cond]
            p = first.condition_probability(cond)
            for pos, kind in enumerate(order):
                grid = grids[kind]
                losses = grid.final_loss[grid.condition == cond].numpy()
                jitter = np.linspace(-0.12, 0.12, len(losses))
                ax.scatter(
                    pos + jitter,
                    losses,
                    s=26,
                    color=MODEL_COLORS[kind],
                    alpha=0.85,
                    edgecolors="white",
                    linewidths=0.8,
                    label=MODEL_LABELS[kind],
                    zorder=3,
                )
                ax.scatter(
                    [pos + 0.28], [losses.min()], marker="<", s=34, color=INK_SECONDARY, zorder=3
                )
            optimum = linear_optimum_loss(first.importance, first.W.shape[1], p)
            ax.axhline(
                optimum,
                color=INK_SECONDARY,
                linewidth=1.0,
                linestyle=(0, (4, 3)),
                zorder=1,
                label="closed-form linear optimum",
            )
            ax.set_xticks(range(len(order)), [MODEL_LABELS[k] for k in order])
            ax.set_xlim(-0.5, len(order) - 0.4)
            ax.set_title(f"S = {1 - p:.1f}", fontsize=9.5, loc="left")
            ax.grid(axis="y")
            if cond == 0:
                ax.set_ylabel("final loss (fresh samples)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        unique = dict(zip(labels, handles, strict=True))
        fig.legend(list(unique.values()), list(unique.keys()), loc="upper right", ncol=len(unique))
        fig.suptitle(
            "E00 · every restart's loss (◀ marks the best)", x=0.02, ha="left", fontsize=10.5
        )
        fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


EXPERIMENT = Experiment(
    name="intro_pentagon",
    code="E00",
    title="Five features in two dimensions: dedicated dims → antipodal pairs → pentagon",
    paper_pages="p. 2",
    config_cls=IntroPentagonConfig,
    run=run,
    figures=figures,
)
