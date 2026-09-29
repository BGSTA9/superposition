"""E06 adversarial: vulnerability to adversarial examples rises with superposition.

Paper: Relationship to Adversarial Robustness, pp. 29–30.

Claim (p. 29): with L2 attacks of at most 0.1 of the average input norm, built analytically
from the rows of WᵀW, "vulnerability to adversarial examples sharply increases as
superposition forms (increasing by >3x), and … the level of vulnerability closely tracks the
number of features per dimension (the reciprocal of feature dimensionality)".

Setup. The p. 29 figure plots 1/(1 − S) from 1 to 10 with features per dimension 1/D* rising
to about 5 there, which matches the uniform setup of p. 17 (n = 400, m = 30, Iᵢ = 1; D* ≈ 0.2
at 1/(1 − S) = 10) and not the n = 20, m = 5 model. So the first, graded setting is the
uniform setup over the figure's range; SPEC §7's example n = 20, m = 5 (Iᵢ = 0.7ⁱ, p. 11) is
graded over the same range as a robustness check, and also swept to the p. 11 range
1 − S ≥ 0.001 and reported without grading (docs/DECISIONS.md D-405, D-406).

The attack and the vulnerability measure live in :mod:`superposition.adversarial`.
"""

from __future__ import annotations

import dataclasses
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from superposition import metrics
from superposition.acceptance import Criterion, check, info
from superposition.adversarial import (
    analytic_attack,
    attack_budget,
    no_superposition_model,
    vulnerability,
)
from superposition.config import ExperimentConfig
from superposition.data import IndependentSparse, geometric_importance
from superposition.experiments.base import Experiment
from superposition.experiments.common import GridResult, instance_table, train_sparsity_grid
from superposition.models import TiedModel
from superposition.plotting.adversarial import (
    VulnerabilityCurve,
    vulnerability_figure,
    vulnerability_scatter,
)
from superposition.plotting.style import save_figure
from superposition.runs import RunDir
from superposition.utils import EVAL_STREAM, derive_seed, make_generator

PAGE = "p. 29"


@dataclass
class AdversarialSetting:
    """One sparsity sweep of ReLU output models."""

    label: str = "uniform_n400_m30"
    n_features: int = 400
    n_hidden: int = 30
    importance_decay: float = 1.0
    inverse_density_min: float = 1.0
    inverse_density_max: float = 10.0
    n_points: int = 16
    n_restarts: int = 2
    batch_size: int | None = 256  # training batch; None uses config.train.batch_size
    attack_samples: int = 4_000
    graded: bool = True
    paper_page: str = "p. 29"

    def inverse_density(self) -> np.ndarray:
        return np.geomspace(self.inverse_density_min, self.inverse_density_max, self.n_points)


def _default_settings() -> list[AdversarialSetting]:
    return [
        AdversarialSetting(),
        AdversarialSetting(
            label="n20_m5",
            n_features=20,
            n_hidden=5,
            importance_decay=0.7,
            n_restarts=4,
            batch_size=None,
            attack_samples=20_000,
            paper_page="pp. 11, 29",
        ),
        AdversarialSetting(
            label="n20_m5_extended",
            n_features=20,
            n_hidden=5,
            importance_decay=0.7,
            inverse_density_min=10.0,
            inverse_density_max=1000.0,
            n_points=9,
            n_restarts=4,
            batch_size=None,
            attack_samples=100_000,
            graded=False,
            paper_page="pp. 11, 29",
        ),
    ]


@dataclass
class AdversarialThresholds:
    """Budget and acceptance thresholds (SPEC §6 adversarial.py, §7 E06)."""

    budget_fraction: float = 0.1  # ε = 0.1 · E‖x‖₂ (p. 29)
    rise_min: float = 2.0  # V(sparsest)/V(densest) > 2 (paper: > 3)
    spearman_min: float = 0.8  # ρ(V, 1/D*) across the sweep
    represented_norm: float = 0.5
    superposition_score: float = 0.05


@dataclass
class AdversarialConfig(ExperimentConfig):
    """E06 settings."""

    settings: list[AdversarialSetting] = field(default_factory=_default_settings)
    thresholds: AdversarialThresholds = field(default_factory=AdversarialThresholds)


def measure_vulnerability(
    grid: GridResult,
    setting: AdversarialSetting,
    budget_fraction: float,
    seed: int,
    device: torch.device,
) -> pd.DataFrame:
    """Attack the best restart of every sparsity; one row per sparsity.

    Each best model and the no-superposition reference see the same inputs (drawn at that
    sparsity) and the same budget ε, so their damages are directly comparable.
    """
    n, m = setting.n_features, setting.n_hidden
    best = grid.best
    k = len(best)
    probs = grid.feature_probability[best]
    model = TiedModel.from_kind("relu_output", n, m, k).to(device)
    with torch.no_grad():
        model.W.copy_(grid.W[best])
        model.b.copy_(grid.b[best])
    reference = no_superposition_model(n, m, k).to(device)
    generator = make_generator(derive_seed(seed, EVAL_STREAM, 2), device)
    x = IndependentSparse(probs, k, n).sample(setting.attack_samples, generator)
    epsilon = attack_budget(x, budget_fraction)
    importance = grid.importance
    attacked = analytic_attack(model, x, importance, epsilon)
    ref = analytic_attack(reference, x, importance, epsilon)
    W = grid.W[best]
    dstar = metrics.dims_per_feature(W)
    chosen = attacked.chosen.float()
    return pd.DataFrame(
        {
            "setting": setting.label,
            "inverse_density": (1.0 / probs).numpy(),
            "feature_probability": probs.numpy(),
            "best_instance": best.numpy(),
            "final_loss": grid.final_loss[best].numpy(),
            "dims_per_feature": dstar.numpy(),
            "features_per_dimension": (1.0 / dstar).numpy(),
            "epsilon": attacked.epsilon.numpy(),
            "clean_loss": attacked.clean_loss.numpy(),
            "attacked_loss": attacked.attacked_loss.numpy(),
            "damage": attacked.damage.numpy(),
            "reference_damage": ref.damage.numpy(),
            "reference_analytic": (importance[:m].max() * attacked.epsilon**2).numpy(),
            "vulnerability": vulnerability(attacked, ref).numpy(),
            "most_attacked_feature": chosen.argmax(dim=-1).numpy(),
            "most_attacked_share": (chosen.amax(dim=-1) / chosen.sum(dim=-1)).numpy(),
        }
    )


def run(
    config: AdversarialConfig, run_dir: RunDir, device: torch.device, logger: logging.Logger
) -> list[Criterion]:
    thr = config.thresholds
    tables, instances, criteria = [], [], []
    for setting in config.settings:
        importance = geometric_importance(setting.n_features, setting.importance_decay)
        probs = (1.0 / setting.inverse_density()).tolist()
        train = config.train
        if setting.batch_size is not None:
            train = dataclasses.replace(train, batch_size=setting.batch_size)
        logger.info(
            f"{setting.label}: n={setting.n_features}, m={setting.n_hidden}, "
            f"{len(probs)} sparsities × {setting.n_restarts} restarts, batch {train.batch_size}"
        )
        grid = train_sparsity_grid(
            "relu_output",
            setting.n_features,
            setting.n_hidden,
            importance,
            probs,
            setting.n_restarts,
            train,
            seed=config.seed,
            device=device,
            progress=True,
        )
        logger.info(f"  trained in {grid.wall_time_s:.1f} s; attacking the best restarts")
        run_dir.save_checkpoint(setting.label, grid.to_checkpoint())
        table = measure_vulnerability(grid, setting, thr.budget_fraction, config.seed, device)
        tables.append(table)
        inst = instance_table(grid, thr.represented_norm, thr.superposition_score)
        inst.insert(0, "setting", setting.label)
        instances.append(inst)
        criteria += evaluate_setting(setting, table, thr)
    run_dir.write_table(pd.concat(tables), "vulnerability")
    run_dir.write_table(pd.concat(instances), "metrics")
    return criteria


def evaluate_setting(
    setting: AdversarialSetting, table: pd.DataFrame, thr: AdversarialThresholds
) -> list[Criterion]:
    """E06 criteria for one sweep (SPEC §7; D-405). Ungraded settings report INFO only."""
    table = table.sort_values("inverse_density")
    v = table["vulnerability"].to_numpy()
    fpd = table["features_per_dimension"].to_numpy()
    x = table["inverse_density"].to_numpy()
    rise = float(v[-1] / v[0])
    rho = float(spearmanr(v, fpd).statistic)  # type: ignore[attr-defined]
    group = setting.label
    span = f"1/(1 − S) ∈ [{x[0]:g}, {x[-1]:g}]"
    rise_value = {
        "rise": round(rise, 3),
        "V_densest": round(float(v[0]), 3),
        "V_sparsest": round(float(v[-1]), 3),
        "V_max": round(float(v.max()), 3),
        "V_max_at": round(float(x[v.argmax()]), 3),
    }
    rows = {
        f"{xi:.3g}": {
            "V": round(float(vi), 3),
            "1/D*": round(float(fi), 3),
            "eps": round(float(e), 4),
            "most_attacked": int(a),
        }
        for xi, vi, fi, e, a in zip(
            x, v, fpd, table["epsilon"], table["most_attacked_feature"], strict=True
        )
    }
    ref_error = float((table["reference_damage"] / table["reference_analytic"] - 1.0).abs().max())
    if not setting.graded:
        return [
            info(
                "vulnerability_extended",
                f"Vulnerability and 1/D* over {span} (not graded): rise V(sparsest)/V(densest) "
                "and Spearman ρ(V, 1/D*)",
                setting.paper_page,
                {**rise_value, "spearman": round(rho, 3), "per_sparsity": rows},
                group=group,
            )
        ]
    return [
        check(
            "vulnerability_rises",
            f"Vulnerability rises as superposition forms: V(sparsest)/V(densest) > "
            f"{thr.rise_min:g} over {span} (paper: > 3×)",
            PAGE,
            rise > thr.rise_min,
            rise_value,
            f"> {thr.rise_min:g}",
            group=group,
        ),
        check(
            "vulnerability_tracks_features_per_dimension",
            f"Spearman ρ between vulnerability and features per dimension 1/D* ≥ "
            f"{thr.spearman_min:g} across the sweep",
            "pp. 29–30",
            rho >= thr.spearman_min,
            round(rho, 4),
            f"≥ {thr.spearman_min:g}",
            group=group,
        ),
        info(
            "vulnerability_per_sparsity",
            "Per sparsity: vulnerability V, features per dimension 1/D*, budget ε and the "
            "feature most often targeted by the worst-case attack",
            PAGE,
            rows,
            group=group,
            notes=f"reference damage matches max Iᵢ·ε² to within {ref_error:.1e} (relative)",
        ),
    ]


# --- figures ------------------------------------------------------------------------------------


def _curves(run_dir: RunDir) -> list[VulnerabilityCurve]:
    table = run_dir.read_table("vulnerability")
    curves = []
    for label, group in table.groupby("setting", sort=False):
        group = group.sort_values("inverse_density")
        curves.append(
            VulnerabilityCurve(
                label=str(label),
                inverse_density=group["inverse_density"].to_numpy(),
                vulnerability=group["vulnerability"].to_numpy(),
                features_per_dimension=group["features_per_dimension"].to_numpy(),
            )
        )
    return curves


def figures(run_dir: RunDir) -> list[Path]:
    """The p. 29 two-panel figure per setting and the p. 30 scatter."""
    curves = _curves(run_dir)
    paths = save_figure(
        vulnerability_figure(
            curves, title="E06 · adversarial vulnerability and features per dimension (p. 29)"
        ),
        run_dir.figures_dir / "vulnerability",
    )
    graded = [c for c in curves if not c.label.endswith("_extended")]
    paths += save_figure(
        vulnerability_scatter(graded, title="E06 · vulnerability vs 1/D* (p. 30)"),
        run_dir.figures_dir / "vulnerability_vs_features_per_dimension",
    )
    return paths


EXPERIMENT = Experiment(
    name="adversarial",
    code="E06",
    title="Adversarial vulnerability rises with superposition and tracks 1/D*",
    paper_pages="pp. 29–30",
    config_cls=AdversarialConfig,
    run=run,
    figures=figures,
)
