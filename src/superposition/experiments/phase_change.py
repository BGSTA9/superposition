"""E02 phase_change: when does an extra feature get dropped, superposed, or its own dimension?

Paper: Superposition as a Phase Change, pp. 14–16; Comments & Replications (McGrath),
pp. 51–53.

Setup (p. 14): n = 2 features in m = 1 dimension, ReLU output model. Feature 0 has importance
1; the "extra" feature's relative importance runs over [0.1, 10] and the density 1 − S of all
features over [0.01, 1] (the paper says "sparsity from 1.0 to 0.01" and labels the axis
"Feature Density (1−S)", which we follow; docs/DECISIONS.md D-303). Ten models per cell; the
highest-loss one is discarded and the rest decide the cell. The same experiment is run for
n = 3 features in m = 2 dimensions (p. 16), where the third feature is the extra one.

Each trained model's extra feature is labeled (docs/DECISIONS.md D-304):

* ``not_learned``: ‖W_e‖ ≤ 0.5;
* ``dedicated``: represented and orthogonal to every other represented feature
  (Σ_{j≠e} (Ŵ_e·Wⱼ)² over represented j is ≤ 0.05);
* ``superposition``: represented and sharing its direction, with opposite sign (antipodal);
* ``confused``: represented and sharing its direction with the same sign (McGrath's
  W₁ ≈ W₂ solution, pp. 51–53).

The theoretical diagrams come from :mod:`superposition.theory`: each family's exact expected
loss is minimized over its free scalars, and the cheapest family wins.

Sign convention: p. 14 writes ReLU(WᵀWx − b); we use +b (SPEC §3). Labels do not depend on it.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import pandas as pd
import torch
from jaxtyping import Float, Int
from torch import Tensor

from superposition import metrics, theory
from superposition.acceptance import Criterion, check, info
from superposition.config import ExperimentConfig, TrainConfig
from superposition.data import IndependentSparse
from superposition.experiments.base import Experiment
from superposition.models import TiedModel
from superposition.plotting.phase import family_loss_figure, phase_figure
from superposition.plotting.style import save_figure
from superposition.runs import RunDir
from superposition.train import train_from_config

MAIN_REGIMES = (theory.NOT_LEARNED, theory.SUPERPOSITION, theory.DEDICATED)


@dataclass
class ProblemSetting:
    """One phase-diagram problem: n features in m dimensions; the last feature is the extra."""

    label: str = "n2_m1"
    n_features: int = 2
    n_hidden: int = 1
    paper_page: str = "p. 15"


def _default_settings() -> list[ProblemSetting]:
    return [ProblemSetting(), ProblemSetting("n3_m2", 3, 2, "p. 16")]


@dataclass
class PhaseGrid:
    """Log-spaced grids (docs/DECISIONS.md D-303). Even counts keep importance 1 off the grid,
    where "drop the extra feature" and "drop the other one" tie exactly."""

    importance_min: float = 0.1
    importance_max: float = 10.0
    n_importance: int = 12
    density_min: float = 0.01
    density_max: float = 1.0
    n_density: int = 12

    def importance(self) -> np.ndarray:
        return np.geomspace(self.importance_min, self.importance_max, self.n_importance)

    def density(self) -> np.ndarray:
        return np.geomspace(self.density_min, self.density_max, self.n_density)


@dataclass
class TheorySettings:
    """Settings of the theoretical diagrams (theory.OptimizerSettings; D-302)."""

    scale_max: float = 2.0
    scale_grid: int = 41
    bias_min: float = -1.5
    bias_max: float = 1.5
    bias_grid: int = 121
    golden_iterations: int = 40

    def optimizer(self, fit_bias: bool = True) -> theory.OptimizerSettings:
        return theory.OptimizerSettings(
            fit_bias=fit_bias,
            scale_max=self.scale_max,
            scale_grid=self.scale_grid,
            bias_range=(self.bias_min, self.bias_max),
            bias_grid=self.bias_grid,
            golden_iterations=self.golden_iterations,
        )


@dataclass
class PhaseChangeThresholds:
    """Acceptance thresholds and labeling thresholds (SPEC §7 E02; D-304, D-305)."""

    represented_norm: float = 0.5
    superposition_score: float = 0.05
    agreement_min: float = 0.8
    near_boundary_relative_gap: float = 0.05


def _default_train() -> TrainConfig:
    return TrainConfig(steps=10_000, batch_size=256)


@dataclass
class PhaseChangeConfig(ExperimentConfig):
    """E02 settings (SPEC §7)."""

    settings: list[ProblemSetting] = field(default_factory=_default_settings)
    grid: PhaseGrid = field(default_factory=PhaseGrid)
    models_per_cell: int = 10
    discard_worst: int = 1
    base_importance: float = 1.0
    theory: TheorySettings = field(default_factory=TheorySettings)
    thresholds: PhaseChangeThresholds = field(default_factory=PhaseChangeThresholds)
    train: TrainConfig = field(default_factory=_default_train)


class PhaseGridConfig(Protocol):
    """What training and voting on a phase grid need; E02's and E12's configs both provide it."""

    seed: int
    grid: PhaseGrid
    models_per_cell: int
    discard_worst: int
    base_importance: float
    thresholds: PhaseChangeThresholds
    train: TrainConfig


# --- labeling (pure functions) ---------------------------------------------------------------


def extra_feature_regime(
    W: Float[Tensor, "inst hidden feat"],
    feature: int,
    represented_threshold: float = 0.5,
    superposition_threshold: float = 0.05,
) -> tuple[np.ndarray, Float[Tensor, " inst"], Float[Tensor, " inst"]]:
    """Label one feature's regime in every instance; also return its norm and superposition.

    Paper: pp. 14–16 (the three outcomes) and pp. 51–53 (the confused-feature outcome). The
    superposition score counts represented other features only, so a weakly embedded feature
    below the representation threshold does not make a dedicated feature look superposed.

    Returns:
        ``(labels, norm, score)``: labels is a string array ``(inst,)`` of
        ``theory.PHASE_LABELS``; norm is ‖W_e‖; score is Σ_{j≠e, j represented} (Ŵ_e·Wⱼ)².
    """
    norms = metrics.feature_norms(W)  # (inst, feat)
    represented = norms > represented_threshold
    others = represented.clone()
    others[:, feature] = False
    unit = W[:, :, feature] / norms[:, feature, None].clamp_min(1e-12)
    projection = torch.einsum("ih,ihf->if", unit, W)  # Ŵ_e·Wⱼ
    score = (projection**2 * others).sum(-1)
    cosine = projection / norms.clamp_min(1e-12)
    partner = torch.where(others, cosine.abs(), torch.full_like(cosine, -1.0)).argmax(-1)
    partner_cos = cosine.gather(1, partner[:, None])[:, 0]
    labels = np.where(
        ~represented[:, feature].numpy(),
        theory.NOT_LEARNED,
        np.where(
            (score <= superposition_threshold).numpy(),
            theory.DEDICATED,
            np.where((partner_cos > 0).numpy(), theory.CONFUSED, theory.SUPERPOSITION),
        ),
    )
    return labels, norms[:, feature], score


def cell_majority(
    labels: np.ndarray, losses: np.ndarray, discard_worst: int = 1
) -> tuple[np.ndarray, list[dict[str, int]]]:
    """Majority label per cell after discarding the highest-loss models (p. 14).

    ``labels`` and ``losses`` are ``(cells, models)``. Ties between labels are broken by the
    lowest loss among the tied labels' models (docs/DECISIONS.md D-304).

    Returns:
        ``(majority, counts)``: majority label per cell and the kept models' label counts.
    """
    majority, all_counts = [], []
    for row_labels, row_losses in zip(labels, losses, strict=True):
        order = np.argsort(row_losses, kind="stable")
        kept = order[: len(order) - discard_worst]
        counts = Counter(str(row_labels[k]) for k in kept)
        top = max(counts.values())
        tied = {label for label, c in counts.items() if c == top}
        winner = next(str(row_labels[k]) for k in kept if row_labels[k] in tied)
        majority.append(winner)
        all_counts.append(dict(counts))
    return np.array(majority), all_counts


def near_boundary(labels: np.ndarray) -> np.ndarray:
    """Cells whose 8-neighbourhood contains a different label (a regime boundary)."""
    rows, cols = labels.shape
    out = np.zeros_like(labels, dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == dc == 0:
                continue
            shifted = np.full_like(labels, "", dtype=object)
            r0, r1 = max(dr, 0), rows + min(dr, 0)
            c0, c1 = max(dc, 0), cols + min(dc, 0)
            shifted[r0 - dr : r1 - dr, c0 - dc : c1 - dc] = labels[r0:r1, c0:c1]
            out |= (shifted != "") & (shifted != labels)
    return out


# --- training --------------------------------------------------------------------------------


@dataclass
class PhaseGridResult:
    """A trained empirical phase diagram (CPU tensors). Instance = cell · models + model.

    Cells are density-major: cell = d · n_importance + i.
    """

    setting: str
    W: Float[Tensor, "inst hidden feat"]
    b: Float[Tensor, "inst feat"]
    final_loss: Float[Tensor, " inst"]
    extra_importance: Float[Tensor, " inst"]
    density: Float[Tensor, " inst"]
    cell: Int[Tensor, " inst"]
    importance_grid: np.ndarray
    density_grid: np.ndarray
    models_per_cell: int
    wall_time_s: float
    W_init: Float[Tensor, "inst hidden feat"] | None = None  # before training (R1-02)

    def to_checkpoint(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_checkpoint(cls, state: dict[str, Any]) -> PhaseGridResult:
        return cls(**state)

    def by_cell(self, values: np.ndarray | Tensor) -> np.ndarray:
        """Reshape a per-instance array to ``(n_density, n_importance, models)``."""
        array = values.numpy() if isinstance(values, Tensor) else np.asarray(values)
        return array.reshape(len(self.density_grid), len(self.importance_grid), -1)


def train_phase_grid(
    setting: ProblemSetting,
    config: PhaseGridConfig,
    device: torch.device,
    progress: bool = False,
    out_act: str = "relu",
) -> PhaseGridResult:
    """Train ``models_per_cell`` ReLU output models at every (density, importance) cell.

    ``out_act`` swaps the output ReLU for another registered activation (E12, p. 51); E02
    always uses the ReLU.
    """
    importance_grid, density_grid = config.grid.importance(), config.grid.density()
    n_cells = len(importance_grid) * len(density_grid)
    n_inst = n_cells * config.models_per_cell
    cell = torch.arange(n_cells).repeat_interleave(config.models_per_cell)
    density = torch.tensor(np.repeat(density_grid, len(importance_grid)), dtype=torch.float32)
    extra = torch.tensor(np.tile(importance_grid, len(density_grid)), dtype=torch.float32)
    importance = torch.full((n_inst, setting.n_features), config.base_importance)
    importance[:, -1] = extra[cell]
    model = TiedModel(
        setting.n_features, setting.n_hidden, n_inst, out_act=out_act, seed=config.seed
    ).to(device)
    W_init = model.W.detach().cpu().clone()
    dist = IndependentSparse(density[cell], n_inst, setting.n_features)
    result = train_from_config(
        model, dist, importance, config.train, seed=config.seed, progress=progress
    )
    return PhaseGridResult(
        setting=setting.label,
        W=model.W.detach().cpu(),
        b=model.b.detach().cpu(),
        final_loss=result.final_loss,
        extra_importance=extra[cell],
        density=density[cell],
        cell=cell,
        importance_grid=importance_grid,
        density_grid=density_grid,
        models_per_cell=config.models_per_cell,
        wall_time_s=result.wall_time_s,
        W_init=W_init,
    )


def initial_sign_outcomes(
    empirical: PhaseGridResult, labels: np.ndarray, theory_labels: np.ndarray
) -> dict[str, Any] | None:
    """n = 2, m = 1: how often a model ends in superposition where the theory says it should,
    split by whether its two weights started with opposite signs (R1-02).

    In one dimension the two weights must pass through zero to become antipodal, so the
    starting sign may decide the outcome. Every model counts (not only the nine voters).
    """
    if empirical.W_init is None or empirical.W.shape[1:] != (1, 2):
        return None
    per_model_theory = np.repeat(np.asarray(theory_labels).ravel(), empirical.models_per_cell)
    target = per_model_theory == theory.SUPERPOSITION
    opposite = (empirical.W_init[:, 0, 0] * empirical.W_init[:, 0, 1] < 0).numpy()
    reached = np.asarray(labels) == theory.SUPERPOSITION
    out: dict[str, Any] = {}
    for name, start in (("opposite_sign", opposite), ("same_sign", ~opposite)):
        models = target & start
        out[name] = f"{int((reached & models).sum())}/{int(models.sum())}"
    return out


def exact_vs_theory(
    empirical: PhaseGridResult, base_importance: float, theory_loss: np.ndarray, activation: str
) -> dict[str, Any] | None:
    """The best trained model's exact loss against the cheapest paper family's, per cell.

    Only for models whose outputs each read at most one other feature (every n = 2 model); the
    families are restricted (one weight scale per tegum factor, D-302), so a trained model can
    beat them, e.g. an antipodal pair with unequal lengths (R1-03).
    """
    n = empirical.W.shape[-1]
    importance = np.full((empirical.W.shape[0], n), base_importance)
    importance[:, -1] = empirical.extra_importance.numpy()
    try:
        exact = theory.tied_output_expected_losses(
            empirical.W.double().numpy(),
            empirical.b.double().numpy(),
            importance,
            empirical.density.double().numpy(),
            activation,
        )
    except NotImplementedError:
        return None
    best = empirical.by_cell(exact).reshape(-1, empirical.models_per_cell).min(1)
    family = np.asarray(theory_loss).ravel()
    below = (family - best) / family  # > 0: the trained model beats the family optimum
    return {
        "cells": len(best),
        "beats_theory_by_more_than": {
            f"{t:g}": int((below > t).sum()) for t in (0.001, 0.005, 0.01)
        },
        "largest_margin": round(float(below.max()), 4),
        "worst_gap": round(float(-below.min()), 4),
    }


def families_for(setting: ProblemSetting) -> tuple[tuple[theory.Family, ...], theory.Family]:
    """The paper's families (pp. 14–15) and McGrath's confused family for a problem size."""
    if (setting.n_features, setting.n_hidden) == (2, 1):
        return theory.N2M1_FAMILIES, theory.N2M1_CONFUSED
    if (setting.n_features, setting.n_hidden) == (3, 2):
        return theory.N3M2_FAMILIES, theory.N3M2_CONFUSED
    raise ValueError(f"no theoretical families for n={setting.n_features}, m={setting.n_hidden}")


@dataclass
class TheoryResult:
    """Theoretical diagrams for one problem: paper families + confused, with and without bias."""

    family_names: list[str]
    family_labels: list[str]
    losses: np.ndarray  # (dens, imp, families); the last family is the confused one
    W: np.ndarray  # (families, dens, imp, m, n)
    b: np.ndarray  # (families, dens, imp, n)
    losses_no_bias: np.ndarray  # (dens, imp, families), all biases fixed at 0 (McGrath)

    def to_checkpoint(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_checkpoint(cls, state: dict[str, Any]) -> TheoryResult:
        return cls(**state)

    def labels(self, include_confused: bool = False, no_bias: bool = False) -> np.ndarray:
        """Winning family's label per cell; the paper's families only unless asked."""
        losses = self.losses_no_bias if no_bias else self.losses
        n = len(self.family_labels) if include_confused else len(self.family_labels) - 1
        names = np.array(self.family_labels[:n])
        return names[losses[..., :n].argmin(-1)]

    def relative_gap(self, include_confused: bool = False) -> np.ndarray:
        n = len(self.family_labels) if include_confused else len(self.family_labels) - 1
        ordered = np.sort(self.losses[..., :n], axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(
                ordered[..., 0] > 0, (ordered[..., 1] - ordered[..., 0]) / ordered[..., 0], np.inf
            )


def compute_theory(setting: ProblemSetting, config: PhaseChangeConfig) -> TheoryResult:
    """Both theoretical diagrams of a problem on the experiment's grid."""
    paper, confused = families_for(setting)
    families = (*paper, confused)
    grid = config.grid
    diagram = theory.theoretical_phase_diagram(
        families,
        grid.importance(),
        grid.density(),
        config.base_importance,
        config.theory.optimizer(),
    )
    no_bias = theory.theoretical_phase_diagram(
        families,
        grid.importance(),
        grid.density(),
        config.base_importance,
        config.theory.optimizer(fit_bias=False),
    )
    shape = diagram.losses.shape[:2]
    return TheoryResult(
        family_names=[f.name for f in families],
        family_labels=[f.label for f in families],
        losses=diagram.losses,
        W=np.stack([s.W.reshape(*shape, *s.W.shape[1:]) for s in diagram.solutions]),
        b=np.stack([s.b.reshape(*shape, -1) for s in diagram.solutions]),
        losses_no_bias=no_bias.losses,
    )


# --- run -------------------------------------------------------------------------------------


def run(
    config: PhaseChangeConfig, run_dir: RunDir, device: torch.device, logger: logging.Logger
) -> list[Criterion]:
    criteria: list[Criterion] = []
    tables = []
    for setting in config.settings:
        grid = config.grid
        logger.info(
            f"{setting.label}: training {grid.n_density} × {grid.n_importance} cells × "
            f"{config.models_per_cell} models"
        )
        empirical = train_phase_grid(setting, config, device, progress=True)
        logger.info(f"  done in {empirical.wall_time_s:.1f} s")
        run_dir.save_checkpoint(f"{setting.label}__empirical", empirical.to_checkpoint())
        logger.info(f"{setting.label}: theoretical diagrams (exact losses, D-301/D-302)")
        predicted = compute_theory(setting, config)
        run_dir.save_checkpoint(f"{setting.label}__theory", predicted.to_checkpoint())
        table, setting_criteria = evaluate_setting(config, setting, empirical, predicted)
        tables.append(table)
        criteria += setting_criteria
    run_dir.write_table(pd.concat(tables), "cells")
    return criteria


def cell_frame(
    config: PhaseGridConfig, empirical: PhaseGridResult, predicted: TheoryResult
) -> pd.DataFrame:
    """One row per cell: empirical majority, label counts, mean norm/superposition, theory."""
    thr = config.thresholds
    extra = empirical.W.shape[-1] - 1
    labels, norm, score = extra_feature_regime(
        empirical.W, extra, thr.represented_norm, thr.superposition_score
    )
    labels_c = empirical.by_cell(labels).reshape(-1, empirical.models_per_cell)
    losses_c = empirical.by_cell(empirical.final_loss).reshape(-1, empirical.models_per_cell)
    majority, counts = cell_majority(labels_c, losses_c, config.discard_worst)
    keep = np.argsort(losses_c, axis=1, kind="stable")[
        :, : empirical.models_per_cell - config.discard_worst
    ]
    norm_c = np.take_along_axis(empirical.by_cell(norm).reshape(losses_c.shape), keep, 1)
    score_c = np.take_along_axis(empirical.by_cell(score).reshape(losses_c.shape), keep, 1)
    best = labels_c[np.arange(len(labels_c)), losses_c.argmin(1)]
    n_dens, n_imp = len(empirical.density_grid), len(empirical.importance_grid)
    boundary = near_boundary(predicted.labels())
    frame = pd.DataFrame(
        {
            "setting": empirical.setting,
            "density": np.repeat(empirical.density_grid, n_imp),
            "importance": np.tile(empirical.importance_grid, n_dens),
            "empirical": majority,
            "best_model": best,
            "counts": [str(dict(sorted(c.items()))) for c in counts],
            "mean_norm": norm_c.mean(1),
            "mean_superposition": score_c.mean(1),
            "theory": predicted.labels().ravel(),
            "theory_with_confused": predicted.labels(include_confused=True).ravel(),
            "theory_no_bias": predicted.labels(include_confused=True, no_bias=True).ravel(),
            "theory_relative_gap": predicted.relative_gap().ravel(),
            "near_theory_boundary": boundary.ravel(),
        }
    )
    for f, name in enumerate(predicted.family_names):
        frame[f"loss {name}"] = predicted.losses[..., f].ravel()
    frame["agree"] = frame["empirical"] == frame["theory"]
    return frame


def evaluate_setting(
    config: PhaseChangeConfig,
    setting: ProblemSetting,
    empirical: PhaseGridResult,
    predicted: TheoryResult,
) -> tuple[pd.DataFrame, list[Criterion]]:
    """E02 acceptance criteria for one problem size (SPEC §7; D-305)."""
    thr = config.thresholds
    page, group = setting.paper_page, setting.label
    frame = cell_frame(config, empirical, predicted)
    col = {name: np.asarray(frame[name]) for name in frame.columns}
    criteria: list[Criterion] = []
    for source, column in (("empirical", "empirical"), ("theoretical", "theory")):
        counts = Counter(str(v) for v in col[column])
        criteria.append(
            check(
                f"three_regions_{source}",
                f"{source.capitalize()} diagram shows all three regions (not represented, "
                "superposition, dedicated dimension)",
                page,
                set(MAIN_REGIMES) <= set(counts),
                {"regions_present": sorted(counts), "cells_per_region": dict(counts)},
                {"required": list(MAIN_REGIMES)},
                group=group,
            )
        )
    agree = col["agree"].astype(bool)
    agreement = float(agree.mean())
    criteria.append(
        check(
            "empirical_theory_agreement",
            "Empirical majority label equals the theoretical label (paper's families)",
            page,
            agreement >= thr.agreement_min,
            {"agreement": round(agreement, 4), "cells": len(agree), "agreeing": int(agree.sum())},
            {"min": thr.agreement_min},
            group=group,
        )
    )
    best_agreement = float((col["best_model"] == col["theory"]).mean())
    criteria.append(
        info(
            "best_model_agreement",
            "Agreement when each cell is labeled by its lowest-loss model instead of the majority",
            page,
            {"agreement": round(best_agreement, 4)},
            group=group,
        )
    )
    near = col["near_theory_boundary"].astype(bool) | (
        col["theory_relative_gap"] < thr.near_boundary_relative_gap
    )
    rows = np.flatnonzero(~agree)
    criteria.append(
        info(
            "disagreements",
            "Disagreeing cells: where they are relative to the theoretical boundaries",
            "pp. 15, 52",
            {
                "n_disagree": len(rows),
                "on_or_next_to_a_theory_boundary": int(near[rows].sum()),
                "empirical_label_counts": dict(Counter(str(v) for v in col["empirical"][rows])),
                "cells": [
                    {
                        "density": round(float(col["density"][r]), 4),
                        "importance": round(float(col["importance"][r]), 3),
                        "empirical": str(col["empirical"][r]),
                        "theory": str(col["theory"][r]),
                        "best_model": str(col["best_model"][r]),
                        "counts": str(col["counts"][r]),
                        "relative_gap": round(float(col["theory_relative_gap"][r]), 4),
                    }
                    for r in rows
                ],
            },
            group=group,
        )
    )
    labels, _, _ = extra_feature_regime(
        empirical.W, empirical.W.shape[-1] - 1, thr.represented_norm, thr.superposition_score
    )
    signs = initial_sign_outcomes(empirical, labels, predicted.labels())
    if signs is not None:
        criteria.append(
            info(
                "initial_sign",
                "Models reaching superposition in the cells where the theory has it, by the "
                "starting sign of their two weights (every model, not only the voters)",
                "p. 15",
                signs,
                group=group,
            )
        )
    n_paper = len(predicted.family_labels) - 1
    exact = exact_vs_theory(
        empirical, config.base_importance, predicted.losses[..., :n_paper].min(-1), "relu"
    )
    if exact is not None:
        criteria.append(
            info(
                "exact_loss_vs_theory",
                "Best trained model's exact loss against the cheapest paper family's optimum, per "
                "cell: how many cells it beats the family by more than 0.1%, 0.5% and 1% "
                "(families are restricted, D-302), the largest margin and the largest gap",
                "p. 15",
                exact,
                group=group,
            )
        )
    confused_models = sum(str(c).count(f"'{theory.CONFUSED}'") for c in col["counts"])
    criteria.append(
        info(
            "confused_regime",
            "McGrath's confused-feature regime: empirical cells, and where the confused family "
            "wins in theory with and without biases",
            "pp. 51–53",
            {
                "empirical_confused_cells": int((col["empirical"] == theory.CONFUSED).sum()),
                "cells_with_a_confused_model": confused_models,
                "theory_confused_wins_with_bias": int(
                    (col["theory_with_confused"] == theory.CONFUSED).sum()
                ),
                "theory_confused_wins_without_bias": int(
                    (col["theory_no_bias"] == theory.CONFUSED).sum()
                ),
                "agreement_with_confused_family": round(
                    float((col["empirical"] == col["theory_with_confused"]).mean()), 4
                ),
            },
            group=group,
        )
    )
    return frame, criteria


# --- figures ---------------------------------------------------------------------------------


def _load(run_dir: RunDir) -> dict[str, tuple[PhaseGridResult, TheoryResult]]:
    out = {}
    for path in sorted(run_dir.checkpoints_dir.glob("*__empirical.pt")):
        label = path.stem.split("__")[0]
        out[label] = (
            PhaseGridResult.from_checkpoint(run_dir.load_checkpoint(f"{label}__empirical")),
            TheoryResult.from_checkpoint(run_dir.load_checkpoint(f"{label}__theory")),
        )
    return out


def figures(run_dir: RunDir) -> list[Path]:
    """Empirical vs theoretical diagrams with the agreement map, and family loss curves."""
    import yaml

    from superposition.config import from_dict

    config = from_dict(PhaseChangeConfig, yaml.safe_load(run_dir.config_path.read_text()))
    cells = run_dir.read_table("cells")
    paths: list[Path] = []
    for label, (empirical, predicted) in _load(run_dir).items():
        voters = empirical.models_per_cell - config.discard_worst
        frame = cells[cells["setting"] == label]
        shape = (len(empirical.density_grid), len(empirical.importance_grid))
        imp, dens = empirical.importance_grid, empirical.density_grid
        n, m = empirical.W.shape[2], empirical.W.shape[1]
        panels = [
            (
                "norm_superposition",
                "Empirical (norm × superposition)",
                {
                    "norm": np.asarray(frame["mean_norm"]).reshape(shape),
                    "superposition": np.asarray(frame["mean_superposition"]).reshape(shape),
                },
            ),
            (
                "regimes",
                f"Empirical (majority of {voters})",
                {"labels": np.asarray(frame["empirical"]).reshape(shape)},
            ),
            ("regimes", "Theoretical (paper families)", {"labels": predicted.labels()}),
            (
                "agreement",
                f"Disagreements ({int((~np.asarray(frame['agree'])).sum())} of {len(frame)} cells)",
                {
                    "empirical": np.asarray(frame["empirical"]).reshape(shape),
                    "theory": predicted.labels(),
                    "near_boundary": near_boundary(predicted.labels()),
                },
            ),
        ]
        fig = phase_figure(
            panels, imp, dens, suptitle=f"E02 · n = {n}, m = {m}: what happens to the extra feature"
        )
        paths += save_figure(fig, run_dir.figures_dir / f"phase_{label}")
        extras = [
            (
                "regimes",
                "Theory + confused family",
                {"labels": predicted.labels(include_confused=True)},
            ),
            (
                "regimes",
                "Theory, biases fixed at 0 (McGrath)",
                {"labels": predicted.labels(include_confused=True, no_bias=True)},
            ),
        ]
        fig = phase_figure(
            extras, imp, dens, suptitle=f"E02 · n = {n}, m = {m}: the confused family (pp. 51–53)"
        )
        paths += save_figure(fig, run_dir.figures_dir / f"theory_variants_{label}")
        picks = [0, len(imp) // 2, len(imp) - 1]
        fig = family_loss_figure(
            dens,
            np.stack([predicted.losses[:, k, :] for k in picks]),
            predicted.family_names,
            predicted.family_labels,
            [float(imp[k]) for k in picks],
            suptitle=f"E02 · n = {n}, m = {m}: each family's optimal loss (exact)",
        )
        paths += save_figure(fig, run_dir.figures_dir / f"family_losses_{label}")
    return paths


EXPERIMENT = Experiment(
    name="phase_change",
    code="E02",
    title="Phase diagrams of the extra feature: not learned / superposition / dedicated",
    paper_pages="pp. 14–16",
    config_cls=PhaseChangeConfig,
    run=run,
    figures=figures,
)
