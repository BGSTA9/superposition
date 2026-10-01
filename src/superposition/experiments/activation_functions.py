"""E12 activation_functions (extension): E02's phase diagrams with other output activations.

Paper: Comments & Replications, p. 51. Redwood Research replicated the phase diagrams and
"found the phase diagrams look quite different depending on the activation function,
suggesting that in this toy model some activation functions induce more polysemanticity than
others." The paper shows no such diagram, so E12 has no pass/fail (SPEC §7 E12): it reports
the differences.

Setup: E02 exactly (pp. 14–16; :mod:`.phase_change`), with the output ReLU of x′ =
ReLU(WᵀWx + b) replaced by each of relu, gelu, silu, softplus, leaky_relu (negative slope 0.01,
PyTorch's default) and sigmoid. Every activation trains the same instances from the same
initialization (same seed), and every model is labeled by E02's rule (D-304): the extra
feature is not represented, superposed (antipodal), given a dedicated dimension, or confused
(W₁ ≈ W₂, pp. 51–53). Ten models per cell vote after the highest-loss one is dropped (p. 14).

The theoretical diagrams minimize each candidate family's exact expected loss with σ in place
of the ReLU (:func:`superposition.theory.activation_square_integral`, D-601) over a wider box
than E02's, because sigmoid and softplus optima need larger weights and more negative biases
(D-602). Each family optimum is re-evaluated with twice as many quadrature nodes, and the
difference is reported.

What is reported (all INFO), per problem and activation:

* the regions: cells per label, empirical and theoretical;
* agreement of each activation's empirical diagram with its own theory;
* how its diagrams differ from the ReLU's, cell by cell;
* losses, which *are* comparable across activations within one cell (same task, footnote 17):
  the best trained model against the theory, and both against the ReLU;
* the share of cells in superposition, per activation (Redwood's "more polysemanticity").
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from superposition import theory
from superposition.acceptance import Criterion, info
from superposition.config import ExperimentConfig, TrainConfig
from superposition.experiments.base import Experiment
from superposition.experiments.phase_change import (
    MAIN_REGIMES,
    PhaseChangeThresholds,
    PhaseGrid,
    PhaseGridResult,
    ProblemSetting,
    TheoryResult,
    cell_frame,
    extra_feature_regime,
    families_for,
    initial_sign_outcomes,
    train_phase_grid,
)
from superposition.models import get_activation
from superposition.plotting.activations import activation_shares_figure
from superposition.plotting.phase import phase_grid_figure
from superposition.plotting.style import save_figure
from superposition.runs import RunDir

DEFAULT_ACTIVATIONS = ("relu", "gelu", "silu", "softplus", "leaky_relu", "sigmoid")
REGIMES = (*MAIN_REGIMES, theory.CONFUSED)


@dataclass
class ActivationTheorySettings:
    """The family optimizer's box and quadrature (docs/DECISIONS.md D-601, D-602).

    One box for every activation, so the ReLU's diagram is computed exactly as the others'.
    ``check_nodes`` re-evaluates every family optimum with more nodes to report the
    quadrature error.
    """

    scale_max: float = 4.0
    scale_grid: int = 41
    bias_min: float = -8.0
    bias_max: float = 4.0
    bias_grid: int = 121
    golden_iterations: int = 30
    quadrature_nodes: int = 12
    check_nodes: int = 24

    def optimizer(self, activation: str) -> theory.OptimizerSettings:
        return theory.OptimizerSettings(
            fit_bias=True,
            scale_max=self.scale_max,
            scale_grid=self.scale_grid,
            bias_range=(self.bias_min, self.bias_max),
            bias_grid=self.bias_grid,
            golden_iterations=self.golden_iterations,
            activation=activation,
            quadrature_nodes=self.quadrature_nodes,
        )


def _default_settings() -> list[ProblemSetting]:
    return [ProblemSetting()]


def _default_grid() -> PhaseGrid:
    return PhaseGrid(n_importance=8, n_density=8)


def _default_train() -> TrainConfig:
    return TrainConfig(steps=10_000, batch_size=256)


@dataclass
class ActivationFunctionsConfig(ExperimentConfig):
    """E12 settings (SPEC §7 E12; D-602). Everything but the activations is E02's."""

    activations: list[str] = field(default_factory=lambda: list(DEFAULT_ACTIVATIONS))
    reference: str = "relu"
    settings: list[ProblemSetting] = field(default_factory=_default_settings)
    grid: PhaseGrid = field(default_factory=_default_grid)
    models_per_cell: int = 10
    discard_worst: int = 1
    base_importance: float = 1.0
    theory: ActivationTheorySettings = field(default_factory=ActivationTheorySettings)
    thresholds: PhaseChangeThresholds = field(default_factory=PhaseChangeThresholds)
    train: TrainConfig = field(default_factory=_default_train)


# --- theory ----------------------------------------------------------------------------------


def compute_activation_theory(
    setting: ProblemSetting, config: ActivationFunctionsConfig, activation: str
) -> tuple[TheoryResult, dict[str, Any]]:
    """The theoretical diagram of one problem for one activation, and its quadrature check.

    Returns ``(result, check)``: ``result.losses_no_bias`` is NaN (E12 does not compute the
    bias-free variant); ``check`` holds the largest absolute and relative change of any family
    optimum's loss when re-evaluated with ``check_nodes`` nodes, and how many cells change label.
    """
    paper, confused = families_for(setting)
    families = (*paper, confused)
    grid = config.grid
    diagram = theory.theoretical_phase_diagram(
        families,
        grid.importance(),
        grid.density(),
        config.base_importance,
        config.theory.optimizer(activation),
    )
    shape = diagram.losses.shape[:2]
    dens, imp = np.meshgrid(grid.density(), grid.importance(), indexing="ij")
    cell_importance = np.full((dens.size, setting.n_features), config.base_importance)
    cell_importance[:, -1] = imp.ravel()
    checked = np.stack(
        [
            theory.tied_output_expected_losses(
                s.W,
                s.b,
                cell_importance,
                dens.ravel(),
                activation,
                config.theory.check_nodes,
            ).reshape(shape)
            for s in diagram.solutions
        ],
        axis=-1,
    )
    diff = np.abs(checked - diagram.losses)
    n_paper = len(paper)
    check = {
        "nodes": config.theory.quadrature_nodes,
        "check_nodes": config.theory.check_nodes,
        "max_abs_change": float(diff.max()),
        "max_rel_change": float((diff / np.maximum(checked, 1e-300)).max()),
        "cells_changing_label": int(
            (checked[..., :n_paper].argmin(-1) != diagram.losses[..., :n_paper].argmin(-1)).sum()
        ),
    }
    result = TheoryResult(
        family_names=[f.name for f in families],
        family_labels=[f.label for f in families],
        losses=diagram.losses,
        W=np.stack([s.W.reshape(*shape, *s.W.shape[1:]) for s in diagram.solutions]),
        b=np.stack([s.b.reshape(*shape, -1) for s in diagram.solutions]),
        losses_no_bias=np.full_like(diagram.losses, np.nan),
    )
    return result, check


# --- per-cell table --------------------------------------------------------------------------


def exact_model_losses(
    empirical: PhaseGridResult, base_importance: float, activation: str
) -> np.ndarray | None:
    """Exact expected loss of every trained model, or None when some output reads two or more
    other features (n = 3: no exact form; the run's sampled losses are used instead)."""
    n = empirical.W.shape[-1]
    importance = np.full((empirical.W.shape[0], n), base_importance)
    importance[:, -1] = empirical.extra_importance.numpy()
    try:
        return theory.tied_output_expected_losses(
            empirical.W.double().numpy(),
            empirical.b.double().numpy(),
            importance,
            empirical.density.double().numpy(),
            activation,
        )
    except NotImplementedError:
        return None


def activation_frame(
    config: ActivationFunctionsConfig,
    activation: str,
    empirical: PhaseGridResult,
    predicted: TheoryResult,
) -> pd.DataFrame:
    """E02's per-cell table for one activation, plus the best model's loss and the theory's."""
    frame = cell_frame(config, empirical, predicted).drop(columns=["theory_no_bias"])
    frame.insert(1, "activation", activation)
    exact = exact_model_losses(empirical, config.base_importance, activation)
    losses = empirical.final_loss.double().numpy() if exact is None else exact
    frame["best_loss"] = empirical.by_cell(losses).reshape(len(frame), -1).min(1)
    frame["best_loss_source"] = "sampled (20,000)" if exact is None else "exact"
    n_paper = len(predicted.family_labels) - 1
    frame["theory_loss"] = predicted.losses[..., :n_paper].min(-1).ravel()
    frame["theory_loss_with_confused"] = predicted.losses.min(-1).ravel()
    return frame


# --- criteria (all INFO) ---------------------------------------------------------------------


def _counts(values: Any) -> dict[str, int]:
    counts = Counter(str(v) for v in values)
    return {r: counts[r] for r in REGIMES if counts[r]}


def _changes(before: np.ndarray, after: np.ndarray) -> dict[str, int]:
    """Counts of label changes ``"a → b"`` between two diagrams of the same grid."""
    moved = Counter(f"{a} → {b}" for a, b in zip(before, after, strict=True) if a != b)
    return dict(moved.most_common())


def _ratio_summary(ratio: np.ndarray) -> dict[str, float]:
    return {
        "median": round(float(np.median(ratio)), 4),
        "min": round(float(ratio.min()), 4),
        "max": round(float(ratio.max()), 4),
    }


def evaluate_activations(
    config: ActivationFunctionsConfig,
    setting: ProblemSetting,
    frames: dict[str, pd.DataFrame],
    checks: dict[str, dict[str, Any]],
    signs: dict[str, dict[str, Any] | None] | None = None,
) -> list[Criterion]:
    """E12's report rows for one problem (SPEC §7 E12: no pass/fail; D-602)."""
    criteria: list[Criterion] = []
    reference = frames.get(config.reference)
    for activation, frame in frames.items():
        group = f"{setting.label} · {activation}"
        empirical, own = np.asarray(frame["empirical"]), np.asarray(frame["theory"])
        with_confused = np.asarray(frame["theory_with_confused"])
        criteria.append(
            info(
                "regions",
                "Cells per regime: empirical majority, theory (the paper's families), and "
                "theory with McGrath's confused family",
                "pp. 14–16, 51–53",
                {
                    "empirical": _counts(empirical),
                    "theory": _counts(own),
                    "theory_with_confused": _counts(with_confused),
                    "cells": len(frame),
                },
                group=group,
            )
        )
        criteria.append(
            info(
                "agreement_with_own_theory",
                "Share of cells where the empirical majority equals this activation's theory",
                "pp. 14–15",
                {
                    "paper_families": round(float((empirical == own).mean()), 4),
                    "with_confused_family": round(float((empirical == with_confused).mean()), 4),
                    "best_model_vs_paper_families": round(
                        float((np.asarray(frame["best_model"]) == own).mean()), 4
                    ),
                },
                group=group,
            )
        )
        gap = np.asarray(frame["best_loss"]) / np.asarray(frame["theory_loss_with_confused"])
        losses: dict[str, Any] = {
            "best_model_over_theory": _ratio_summary(gap),
            "best_model_loss_source": str(frame["best_loss_source"].iloc[0]),
        }
        if reference is not None and activation != config.reference:
            ref_empirical = np.asarray(reference["empirical"])
            ref_theory = np.asarray(reference["theory"])
            criteria.append(
                info(
                    "vs_reference",
                    f"Cells whose label differs from the {config.reference} diagram's, and how "
                    "they changed (reference → this activation)",
                    "p. 51",
                    {
                        "empirical_same": round(float((empirical == ref_empirical).mean()), 4),
                        "theory_same": round(float((own == ref_theory).mean()), 4),
                        "empirical_changes": _changes(ref_empirical, empirical),
                        "theory_changes": _changes(ref_theory, own),
                    },
                    group=group,
                )
            )
            losses["theory_over_reference_theory"] = _ratio_summary(
                np.asarray(frame["theory_loss_with_confused"])
                / np.asarray(reference["theory_loss_with_confused"])
            )
            losses["best_model_over_reference_best_model"] = _ratio_summary(
                np.asarray(frame["best_loss"]) / np.asarray(reference["best_loss"])
            )
        criteria.append(
            info(
                "losses",
                "Loss ratios over the cells (comparable within a cell, footnote 17): the best "
                "trained model against the theory's optimum (families incl. confused), and "
                f"both against {config.reference}",
                "p. 15, footnote 17",
                losses,
                group=group,
            )
        )
        sign = (signs or {}).get(activation)
        if sign is not None:
            criteria.append(
                info(
                    "initial_sign",
                    "Models reaching superposition in the cells where this activation's theory "
                    "has it, by the starting sign of their two weights (every model; the same "
                    "starting weights for every activation)",
                    "p. 15",
                    sign,
                    group=group,
                )
            )
        criteria.append(
            info(
                "quadrature_check",
                "Change of every family optimum's loss when re-evaluated with more quadrature "
                "nodes (D-601)",
                "p. 51",
                checks[activation],
                group=group,
            )
        )
    shares = {
        activation: {
            "empirical": round(float((frame["empirical"] == theory.SUPERPOSITION).mean()), 4),
            "theory": round(float((frame["theory"] == theory.SUPERPOSITION).mean()), 4),
            "theory_with_confused": round(
                float((frame["theory_with_confused"] == theory.SUPERPOSITION).mean()), 4
            ),
            "empirical_confused": round(float((frame["empirical"] == theory.CONFUSED).mean()), 4),
        }
        for activation, frame in frames.items()
    }
    criteria.append(
        info(
            "superposition_share",
            "Share of cells where the extra feature is in superposition, per activation "
            "(Redwood: some activations 'induce more polysemanticity')",
            "p. 51",
            {
                "shares": shares,
                "ranking_empirical": sorted(shares, key=lambda a: -shares[a]["empirical"]),
                "ranking_theory": sorted(shares, key=lambda a: -shares[a]["theory"]),
            },
            group=setting.label,
        )
    )
    return criteria


# --- run -------------------------------------------------------------------------------------


def run(
    config: ActivationFunctionsConfig,
    run_dir: RunDir,
    device: torch.device,
    logger: logging.Logger,
) -> list[Criterion]:
    for activation in config.activations:
        get_activation(activation)  # fail fast on an unknown name
    criteria: list[Criterion] = []
    tables = []
    for setting in config.settings:
        frames: dict[str, pd.DataFrame] = {}
        checks: dict[str, dict[str, Any]] = {}
        signs: dict[str, dict[str, Any] | None] = {}
        grid = config.grid
        for activation in config.activations:
            logger.info(
                f"{setting.label} · {activation}: training {grid.n_density} × "
                f"{grid.n_importance} cells × {config.models_per_cell} models"
            )
            empirical = train_phase_grid(setting, config, device, progress=True, out_act=activation)
            logger.info(f"  done in {empirical.wall_time_s:.1f} s")
            run_dir.save_checkpoint(
                f"{setting.label}__{activation}__empirical", empirical.to_checkpoint()
            )
            start = time.perf_counter()
            predicted, checks[activation] = compute_activation_theory(setting, config, activation)
            logger.info(f"  theoretical diagram in {time.perf_counter() - start:.1f} s (D-601)")
            run_dir.save_checkpoint(
                f"{setting.label}__{activation}__theory", predicted.to_checkpoint()
            )
            frames[activation] = activation_frame(config, activation, empirical, predicted)
            labels, _, _ = extra_feature_regime(
                empirical.W,
                setting.n_features - 1,
                config.thresholds.represented_norm,
                config.thresholds.superposition_score,
            )
            signs[activation] = initial_sign_outcomes(empirical, labels, predicted.labels())
        criteria += evaluate_activations(config, setting, frames, checks, signs)
        tables += list(frames.values())
    run_dir.write_table(pd.concat(tables, ignore_index=True), "cells")
    return criteria


# --- figures ---------------------------------------------------------------------------------


def figures(run_dir: RunDir) -> list[Path]:
    """Per problem: the diagrams of every activation (empirical, theory, change from the
    reference), and the regime shares next to the activation curves."""
    import yaml

    from superposition.config import from_dict

    config = from_dict(ActivationFunctionsConfig, yaml.safe_load(run_dir.config_path.read_text()))
    cells = run_dir.read_table("cells")
    grid = config.grid
    importance, density = grid.importance(), grid.density()
    shape = (len(density), len(importance))
    voters = config.models_per_cell - config.discard_worst
    paths: list[Path] = []
    for setting in config.settings:
        frame = cells[cells["setting"] == setting.label]
        by_act = {a: pd.DataFrame(frame[frame["activation"] == a]) for a in config.activations}

        def labels(
            activation: str, column: str, by_act: dict[str, pd.DataFrame] = by_act
        ) -> np.ndarray:
            return np.asarray(by_act[activation][column]).reshape(shape)

        reference = config.reference if config.reference in by_act else None
        rows: list[tuple[str, list[tuple[str, str, dict]]]] = [
            (
                f"empirical (majority of {voters})",
                [("regimes", a, {"labels": labels(a, "empirical")}) for a in config.activations],
            ),
            (
                "theory (families + confused)",
                [
                    ("regimes", a, {"labels": labels(a, "theory_with_confused")})
                    for a in config.activations
                ],
            ),
        ]
        if reference is not None:
            rows.append(
                (
                    f"empirical: changed vs {reference}",
                    [
                        (
                            "agreement",
                            a,
                            {
                                "empirical": labels(a, "empirical"),
                                "theory": labels(reference, "empirical"),
                            },
                        )
                        for a in config.activations
                    ],
                )
            )
        fig = phase_grid_figure(
            rows,
            importance,
            density,
            suptitle=(
                f"E12 · n = {setting.n_features}, m = {setting.n_hidden}: the extra feature's "
                "regime with each output activation"
            ),
        )
        paths += save_figure(fig, run_dir.figures_dir / f"diagrams_{setting.label}")
        u = np.linspace(-3.0, 3.0, 301)
        curves = {a: get_activation(a)(torch.from_numpy(u)).numpy() for a in config.activations}
        shares = {
            a: {
                "empirical": _counts(by_act[a]["empirical"]),
                "theory": _counts(by_act[a]["theory_with_confused"]),
            }
            for a in config.activations
        }
        fig = activation_shares_figure(
            u,
            curves,
            shares,
            suptitle=(
                f"E12 · n = {setting.n_features}, m = {setting.n_hidden}: activations and the "
                "share of cells per regime"
            ),
        )
        paths += save_figure(fig, run_dir.figures_dir / f"shares_{setting.label}")
    return paths


EXPERIMENT = Experiment(
    name="activation_functions",
    code="E12",
    title="Extension: E02's phase diagrams with six output activations",
    paper_pages="p. 51",
    config_cls=ActivationFunctionsConfig,
    run=run,
    figures=figures,
)
