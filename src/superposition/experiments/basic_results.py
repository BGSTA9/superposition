"""E01 basic_results: linear vs ReLU output models across sparsity.

Paper: Demonstrating Superposition, Basic Results, pp. 10–12.

Claims (n = 20, m = 5, Iᵢ = 0.7ⁱ, p. 11; and n = 80, m = 20, Iᵢ = 0.9ⁱ, p. 12):

* The linear model always learns the top-m features, orthogonally, whatever the sparsity
  ("if optimized well enough, the linear model solution does not depend on sparsity level").
* The ReLU output model matches it on dense data (1 − S = 1). As sparsity increases it
  represents more features by letting them be non-orthogonal: superposition starts with the
  less important features, first as antipodal pairs, and eventually every feature is in
  superposition.

Acceptance criteria follow SPEC §7 E01; how each is operationalized is recorded in
docs/DECISIONS.md (D-105).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

from superposition import metrics
from superposition.acceptance import Criterion, check, info
from superposition.config import ExperimentConfig
from superposition.data import geometric_importance
from superposition.experiments.base import Experiment
from superposition.experiments.common import (
    GridResult,
    antipodal_pairs,
    feature_table,
    format_probability,
    instance_table,
    represented_features,
    train_sparsity_grid,
)
from superposition.plotting.style import (
    INK_SECONDARY,
    MODEL_COLORS,
    MODEL_LABELS,
    paper_style,
    save_figure,
)
from superposition.plotting.weights import WeightsPanel, weights_grid
from superposition.runs import RunDir


@dataclass
class SizeSetting:
    """One model size from the paper."""

    label: str = "n20_m5"
    n_features: int = 20
    n_hidden: int = 5
    importance_decay: float = 0.7
    paper_page: str = "p. 11"


@dataclass
class BasicResultsThresholds:
    """Acceptance thresholds (SPEC §7 E01; operationalization in DECISIONS.md D-105)."""

    represented_norm: float = 0.5
    superposition_score: float = 0.05
    antipodal_cos: float = -0.95
    count_tolerance: int = 1
    relu_linear_loss_rel_tol: float = 0.05


def _default_settings() -> list[SizeSetting]:
    return [
        SizeSetting(),
        SizeSetting(
            label="n80_m20", n_features=80, n_hidden=20, importance_decay=0.9, paper_page="p. 12"
        ),
    ]


@dataclass
class BasicResultsConfig(ExperimentConfig):
    """E01 settings. ``feature_probabilities`` are 1 − S, densest first (as on p. 11)."""

    settings: list[SizeSetting] = field(default_factory=_default_settings)
    feature_probabilities: list[float] = field(
        default_factory=lambda: [1.0, 0.3, 0.1, 0.03, 0.01, 0.003, 0.001]
    )
    n_restarts: int = 4
    models: list[str] = field(default_factory=lambda: ["linear", "relu_output"])
    thresholds: BasicResultsThresholds = field(default_factory=BasicResultsThresholds)


def run(
    config: BasicResultsConfig, run_dir: RunDir, device: torch.device, logger: logging.Logger
) -> list[Criterion]:
    thr = config.thresholds
    instances, features, criteria = [], [], []
    for setting in config.settings:
        importance = geometric_importance(setting.n_features, setting.importance_decay)
        grids: dict[str, GridResult] = {}
        for kind in config.models:
            logger.info(
                f"{setting.label} · training {kind}: {len(config.feature_probabilities)} "
                f"sparsities × {config.n_restarts} restarts"
            )
            grid = train_sparsity_grid(
                kind,  # type: ignore[arg-type]
                setting.n_features,
                setting.n_hidden,
                importance,
                config.feature_probabilities,
                config.n_restarts,
                config.train,
                seed=config.seed,
                device=device,
                progress=True,
            )
            logger.info(f"  done in {grid.wall_time_s:.1f} s")
            run_dir.save_checkpoint(f"{setting.label}__{kind}", grid.to_checkpoint())
            grids[kind] = grid
            table = instance_table(grid, thr.represented_norm, thr.superposition_score)
            table.insert(0, "setting", setting.label)
            instances.append(table)
            ftable = feature_table(grid)
            ftable.insert(0, "setting", setting.label)
            features.append(ftable)
        criteria += evaluate_setting(config, setting, grids)
    run_dir.write_table(pd.concat(instances), "metrics")
    run_dir.write_table(pd.concat(features), "features")
    return criteria


def _summary(grid: GridResult, cond: int, thr: BasicResultsThresholds) -> dict:
    """Represented set, superposed set, max superposition score and loss of the best restart."""
    W, _, loss = grid.best_weights(cond)
    rep = represented_features(W, thr.represented_norm)
    score = metrics.superposition_score(W[None])[0]
    superposed = [i for i in rep if float(score[i]) > thr.superposition_score]
    max_score = max((float(score[i]) for i in rep), default=0.0)
    return {
        "represented": rep,
        "superposed": superposed,
        "max_score": max_score,
        "loss": loss,
        "pairs": antipodal_pairs(W, thr.represented_norm, thr.antipodal_cos),
        "D_star": float(metrics.dims_per_feature(W[None])[0]),
    }


def evaluate_setting(
    config: BasicResultsConfig, setting: SizeSetting, grids: dict[str, GridResult]
) -> list[Criterion]:
    """E01 acceptance criteria for one model size (SPEC §7)."""
    thr = config.thresholds
    page = setting.paper_page
    group = setting.label
    m = setting.n_hidden
    top_m = list(range(m))
    probs = config.feature_probabilities
    labels = [format_probability(p) for p in probs]
    criteria: list[Criterion] = []

    linear = grids.get("linear")
    lin: list[dict] = []
    if linear is not None:
        lin = [_summary(linear, c, thr) for c in range(len(probs))]
        per_p = {
            f"1-S={lab}": {
                "n_represented": len(s["represented"]),
                "is_top_m": s["represented"] == top_m,
                "max_score": round(s["max_score"], 4),
            }
            for lab, s in zip(labels, lin, strict=True)
        }
        failing = [
            lab
            for lab, s in zip(labels, lin, strict=True)
            if not (s["represented"] == top_m and s["max_score"] < thr.superposition_score)
        ]
        criteria.append(
            check(
                "linear_top_m_orthogonal",
                f"Linear: exactly the top {m} features represented, superposition score < "
                f"{thr.superposition_score}, at every sparsity",
                page,
                not failing,
                {"per_sparsity": per_p, "failing": failing},
                {"represented": f"features 0–{m - 1}", "max_score_below": thr.superposition_score},
                group=group,
            )
        )

    relu = grids.get("relu_output")
    if relu is None:
        return criteria
    rel = [_summary(relu, c, thr) for c in range(len(probs))]
    counts = [len(s["represented"]) for s in rel]

    dense = rel[0]
    dense_ok = dense["represented"] == top_m and dense["max_score"] < thr.superposition_score
    loss_rel = abs(dense["loss"] - lin[0]["loss"]) / lin[0]["loss"] if lin else float("nan")
    criteria.append(
        check(
            "relu_matches_linear_when_dense",
            f"ReLU at 1 − S = {labels[0]}: same top-{m} orthogonal solution as the linear model "
            f"(loss within {thr.relu_linear_loss_rel_tol:.0%})",
            page,
            dense_ok and bool(lin) and loss_rel <= thr.relu_linear_loss_rel_tol,
            {
                "represented": dense["represented"],
                "max_score": round(dense["max_score"], 4),
                "loss_relu": round(dense["loss"], 5),
                "loss_linear": round(lin[0]["loss"], 5) if lin else None,
                "rel_loss_diff": round(loss_rel, 4),
            },
            {
                "represented": f"features 0–{m - 1}",
                "max_score_below": thr.superposition_score,
                "rel_loss_diff_max": thr.relu_linear_loss_rel_tol,
            },
            group=group,
        )
    )

    running_max, drops = 0, []
    for lab, count in zip(labels, counts, strict=True):
        if count < running_max - thr.count_tolerance:
            drops.append(lab)
        running_max = max(running_max, count)
    criteria.append(
        check(
            "relu_count_non_decreasing",
            f"ReLU: represented-feature count never falls more than {thr.count_tolerance} below "
            "any denser setting's count",
            page,
            not drops,
            {"counts": dict(zip(labels, counts, strict=True)), "violations_at": drops},
            {"tolerance": thr.count_tolerance},
            group=group,
        )
    )
    criteria.append(
        check(
            "relu_exceeds_m_at_sparsest",
            f"ReLU: more than m = {m} features represented at 1 − S = {labels[-1]}",
            page,
            counts[-1] > m,
            counts[-1],
            f"> {m}",
            group=group,
        )
    )

    first = next((c for c, s in enumerate(rel) if s["superposed"]), None)
    if first is None:
        criteria.append(
            check(
                "relu_superposition_less_important_first",
                "ReLU: superposition first appears among the less important features",
                "p. 11",
                False,
                "no superposition at any sparsity",
                "superposed features all less important than every orthogonal represented one",
                group=group,
            )
        )
    else:
        s = rel[first]
        orthogonal = [i for i in s["represented"] if i not in s["superposed"]]
        ok = bool(orthogonal) and 0 in orthogonal and min(s["superposed"]) > max(orthogonal)
        criteria.append(
            check(
                "relu_superposition_less_important_first",
                "ReLU: at the first sparsity with superposition, the most important feature is "
                "untouched and every superposed feature is less important than every orthogonal one",
                "p. 11",
                ok,
                {
                    "first_sparsity": labels[first],
                    "orthogonal_represented": orthogonal,
                    "superposed": s["superposed"],
                },
                "min(superposed) > max(orthogonal), feature 0 orthogonal",
                group=group,
            )
        )

    pairs_by_p = {lab: s["pairs"] for lab, s in zip(labels, rel, strict=True)}
    intermediate = [lab for lab in labels[1:-1] if pairs_by_p[lab]]
    criteria.append(
        check(
            "relu_antipodal_pairs_at_intermediate_sparsity",
            "ReLU: antipodal pairs (cos < −0.95) appear at an intermediate sparsity; report where",
            "p. 12",
            bool(intermediate),
            {"with_pairs": intermediate, "pairs_by_sparsity": pairs_by_p},
            f"≥ 1 pair at some 1 − S strictly between {labels[0]} and {labels[-1]}",
            group=group,
        )
    )
    criteria.append(
        info(
            "per_sparsity_summary",
            "Best-restart summary per sparsity (ReLU output; linear loss for comparison)",
            page,
            {
                f"1-S={lab}": {
                    "relu_represented": len(s["represented"]),
                    "relu_superposed": len(s["superposed"]),
                    "relu_D_star": round(s["D_star"], 3),
                    "relu_loss": round(s["loss"], 5),
                    "linear_loss": round(lin[c]["loss"], 5) if lin else None,
                }
                for c, (lab, s) in enumerate(zip(labels, rel, strict=True))
            },
            group=group,
        )
    )
    return criteria


# --- figures ---------------------------------------------------------------------------------


def _load(run_dir: RunDir) -> dict[str, dict[str, GridResult]]:
    grids: dict[str, dict[str, GridResult]] = {}
    for path in sorted(run_dir.checkpoints_dir.glob("*.pt")):
        label, kind = path.stem.split("__")
        grids.setdefault(label, {})[kind] = GridResult.from_checkpoint(
            run_dir.load_checkpoint(path.stem)
        )
    return grids


def _panel(grid: GridResult, cond: int, title: str) -> WeightsPanel:
    W, b, _ = grid.best_weights(cond)
    return WeightsPanel(
        title=title,
        gram=metrics.gram(W[None])[0].numpy(),
        bias=b.numpy(),
        norms=metrics.feature_norms(W[None])[0].numpy(),
        superposition=metrics.superposition_score(W[None])[0].numpy(),
    )


def figures(run_dir: RunDir) -> list[Path]:
    """The p. 11 / p. 12 grids per model size, plus a represented-count summary."""
    grids = _load(run_dir)
    paths: list[Path] = []
    for label, by_kind in grids.items():
        panels, group_labels = [], []
        if "linear" in by_kind:
            lin = by_kind["linear"]
            sparsest = lin.n_conditions - 1
            p = lin.condition_probability(sparsest)
            group_labels.append((0, "Linear"))
            panels.append(_panel(lin, sparsest, f"1−S = {format_probability(p)}"))
        if "relu_output" in by_kind:
            relu = by_kind["relu_output"]
            group_labels.append((len(panels), "ReLU output model"))
            for cond in range(relu.n_conditions):
                p = relu.condition_probability(cond)
                panels.append(_panel(relu, cond, f"1−S = {format_probability(p)}"))
        any_grid = next(iter(by_kind.values()))
        n, m = any_grid.W.shape[2], any_grid.W.shape[1]
        fig = weights_grid(
            panels,
            suptitle=f"E01 · n = {n}, m = {m}: WᵀW, bias and ‖Wᵢ‖ colored by superposition",
            group_labels=group_labels,
        )
        paths += save_figure(fig, run_dir.figures_dir / f"weights_{label}")
    paths += save_figure(_count_figure(grids), run_dir.figures_dir / "represented_counts")
    return paths


def _count_figure(grids: dict[str, dict[str, GridResult]]):
    """Represented features (best restart) vs feature density, one panel per model size."""
    with paper_style():
        fig, axes = plt.subplots(1, len(grids), figsize=(4.2 * len(grids), 3.2), squeeze=False)
        for ax, by_kind in zip(axes[0], grids.values(), strict=True):
            any_grid = next(iter(by_kind.values()))
            n, m = any_grid.W.shape[2], any_grid.W.shape[1]
            for kind, grid in by_kind.items():
                probs = [grid.condition_probability(c) for c in range(grid.n_conditions)]
                counts = [
                    int(metrics.represented_mask(grid.best_weights(c)[0][None])[0].sum())
                    for c in range(grid.n_conditions)
                ]
                ax.plot(
                    probs, counts, marker="o", color=MODEL_COLORS[kind], label=MODEL_LABELS[kind]
                )
            ax.axhline(
                m,
                color=INK_SECONDARY,
                linewidth=1.0,
                linestyle=(0, (4, 3)),
                label="m (hidden dimensions)",
            )
            ax.set_xscale("log")
            ax.invert_xaxis()
            ax.set_xlabel("feature density 1 − S  (dense → sparse)")
            ax.set_ylabel("represented features (‖Wᵢ‖ > 0.5)")
            ax.set_title(f"n = {n}, m = {m}", loc="left")
            ax.grid(axis="y")
            ax.set_ylim(0, None)
        handles, names = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, names, loc="upper right", ncol=3)
        fig.suptitle(
            "E01 · features represented by the best restart", x=0.02, ha="left", fontsize=10.5
        )
        fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


EXPERIMENT = Experiment(
    name="basic_results",
    code="E01",
    title="Linear vs ReLU output models across sparsity (n = 20, m = 5 and n = 80, m = 20)",
    paper_pages="pp. 10–12",
    config_cls=BasicResultsConfig,
    run=run,
    figures=figures,
)
