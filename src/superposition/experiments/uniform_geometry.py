"""E03 uniform_geometry: dimensions per feature, feature dimensionality, and polytopes.

Paper: Uniform Superposition, p. 17; Feature Dimensionality, pp. 18–19; Why These Geometric
Structures?, pp. 19–20; Aside: Polytopes and Low-Rank Matrices, pp. 20–21.

Setup (p. 17): n = 400 features of equal importance (Iᵢ = 1) and equal sparsity in m = 30
dimensions, ReLU output model, with 1/(1 − S) log-spaced over [1, 10].

Claims:

* D* = m/‖W‖²_F is "sticky" at 1 (dense) and at 1/2 (antipodal pairs) (p. 17), then falls.
* Each feature's dimensionality Dᵢ clusters at a few fractions (3/4, 2/3, 1/2, 2/5, 3/8) that
  correspond to uniform polytopes, often combined as tegum products (pp. 18–20).
* When features are "packed efficiently", Σᵢ Dᵢ ≈ m (p. 18).

The acceptance criteria and how they are read off the p. 17 figure are in
docs/DECISIONS.md (D-306).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from superposition import metrics
from superposition.acceptance import Criterion, check, info
from superposition.config import ExperimentConfig, TrainConfig
from superposition.experiments.base import Experiment
from superposition.experiments.common import GridResult, instance_table, train_sparsity_grid
from superposition.plotting.geometry import (
    census_figure,
    dimensionality_figure,
    dstar_figure,
    geometry_graph_figure,
)
from superposition.plotting.style import save_figure
from superposition.runs import RunDir

PAGE = "p. 17"


@dataclass
class UniformGeometryThresholds:
    """Acceptance thresholds (SPEC §7 E03), read off the p. 17 figure (D-306)."""

    represented_norm: float = 0.5
    superposition_score: float = 0.05
    dense_until: float = 1.2
    dense_min_dstar: float = 0.95
    plateau_value: float = 0.5
    plateau_tol: float = 0.03
    plateau_min_span: float = 1.25
    tail_at: float = 8.0
    tail_value: float = 0.25
    tail_tol: float = 0.05
    fraction_tol: float = 0.02
    cluster_min_share: float = 0.2
    cluster_min_count: int = 8
    min_clusters: int = 3
    edge_threshold: float = 0.05
    rank_rtol: float = 0.1
    uniform_tol: float = 0.05


def _default_train() -> TrainConfig:
    return TrainConfig(steps=10_000, batch_size=256)


@dataclass
class UniformGeometryConfig(ExperimentConfig):
    """E03 settings. The grid is 1/(1 − S), log-spaced (SPEC §7: quick 20 points, full ≥ 60)."""

    n_features: int = 400
    n_hidden: int = 30
    inverse_density_min: float = 1.0
    inverse_density_max: float = 10.0
    n_points: int = 20
    n_restarts: int = 2
    # 1/(1 − S) values whose geometry graphs are drawn (nearest grid points), as on p. 19.
    graph_at: list[float] = field(default_factory=lambda: [1.4, 1.7, 2.9, 4.3, 5.0, 9.0])
    thresholds: UniformGeometryThresholds = field(default_factory=UniformGeometryThresholds)
    train: TrainConfig = field(default_factory=_default_train)

    def inverse_density(self) -> np.ndarray:
        return np.geomspace(self.inverse_density_min, self.inverse_density_max, self.n_points)


# --- analysis (pure functions of the trained grid) --------------------------------------------


@dataclass
class Summary:
    """Best-restart statistics per grid point."""

    inverse_density: np.ndarray  # (points,)
    dstar: np.ndarray  # (points,)
    dimensionality: list[np.ndarray]  # per point, Dᵢ of every feature
    represented: list[np.ndarray]  # per point, bool mask
    census: list[dict[str, int]]  # per point, features per factor type
    sum_dimensionality: np.ndarray  # (points,), Σᵢ Dᵢ
    # per point, the name of each feature's tegum factor (None if not represented; D-308)
    factors: list[list[str | None]] | None = None


def summarize(grid: GridResult, thr: UniformGeometryThresholds) -> Summary:
    """D*, Dᵢ, represented masks, tegum factors and their census of each best restart."""
    inverse, dstar, dims, reps, census, sums, factors = [], [], [], [], [], [], []
    for cond in range(grid.n_conditions):
        W, _, _ = grid.best_weights(cond)
        inverse.append(1.0 / grid.condition_probability(cond))
        dstar.append(float(metrics.dims_per_feature(W[None])[0]))
        d = metrics.feature_dimensionality(W[None])[0]
        dims.append(d.numpy())
        reps.append(metrics.represented_mask(W[None], thr.represented_norm)[0].numpy())
        names = metrics.feature_factors(
            W, thr.edge_threshold, thr.represented_norm, thr.rank_rtol, thr.uniform_tol
        )
        factors.append(names)
        point: dict[str, int] = {}
        for name in names:
            if name is not None:
                point[name] = point.get(name, 0) + 1
        census.append(point)
        sums.append(float(d.sum()))
    return Summary(np.array(inverse), np.array(dstar), dims, reps, census, np.array(sums), factors)


def plateau_span(
    inverse_density: np.ndarray, dstar: np.ndarray, value: float, tol: float
) -> tuple[int, int] | None:
    """Indices (first, last) of the longest contiguous run with |D* − value| ≤ tol.

    Runs are compared by the ratio of their end points in 1/(1 − S) (the log-axis width).
    """
    inside = np.abs(dstar - value) <= tol
    best: tuple[int, int] | None = None
    start = None
    for k, flag in enumerate([*inside, False]):
        if flag and start is None:
            start = k
        if not flag and start is not None:
            run = (start, k - 1)
            width = inverse_density[run[1]] / inverse_density[run[0]]
            if best is None or width > inverse_density[best[1]] / inverse_density[best[0]]:
                best = run
            start = None
    return best


def fraction_clusters(
    dimensionality: list[np.ndarray],
    represented: list[np.ndarray],
    tol: float,
    min_share: float,
    min_count: int,
    factors: Sequence[Sequence[str | None]] | None = None,
) -> dict[str, dict]:
    """For each non-trivial characteristic fraction: its largest cluster in any one model.

    A fraction counts as a cluster when, in at least one best model, at least ``min_share`` of
    the represented features (and at least ``min_count``) have their nearest characteristic
    fraction (within ``tol``) equal to it (docs/DECISIONS.md D-306). With ``factors`` (each
    feature's tegum factor, per model), a feature counts only if its factor is the polytope of
    that fraction, e.g. a feature at 2/5 only if it sits in a pentagon (D-603): p. 19's "each
    fraction corresponds to a specific weight geometry", and 2/5 and 3/8 are too close for
    Dᵢ alone to tell apart.
    """
    fractions = [
        (k, name, value)
        for k, (name, value) in enumerate(metrics.CHARACTERISTIC_FRACTIONS)
        if 0.0 < value < 1.0
    ]
    out: dict[str, dict] = {}
    for k, name, value in fractions:
        polytope = metrics.polytope_of_fraction(value)
        best_share, best_count, best_point = 0.0, 0, -1
        for point, (dims, rep) in enumerate(zip(dimensionality, represented, strict=True)):
            if not rep.any():
                continue
            nearest = metrics.classify_fraction(torch.as_tensor(dims[rep]), tol).numpy()
            hit = nearest == k
            if factors is not None:
                names = np.array(
                    [factors[point][int(i)] for i in np.flatnonzero(rep)], dtype=object
                )
                hit &= names == polytope
            count = int(hit.sum())
            share = count / int(rep.sum())
            if share > best_share:
                best_share, best_count, best_point = share, count, point
        out[f"{name} ({value:.3g})"] = {
            "cluster": best_share >= min_share and best_count >= min_count,
            "max_share": round(best_share, 3),
            "count": best_count,
            "at_point": best_point,
        }
    return out


def evaluate(config: UniformGeometryConfig, summary: Summary) -> list[Criterion]:
    """E03 acceptance criteria (SPEC §7; D-306)."""
    thr = config.thresholds
    x, dstar = summary.inverse_density, summary.dstar
    criteria: list[Criterion] = []

    dense = x <= thr.dense_until * (1 + 1e-9)
    criteria.append(
        check(
            "dense_dstar_is_1",
            f"D* ≈ 1 (≥ {thr.dense_min_dstar}) at every 1/(1 − S) ≤ {thr.dense_until}",
            PAGE,
            bool(dense.any()) and bool((dstar[dense] >= thr.dense_min_dstar).all()),
            {f"{v:.3f}": round(float(d), 4) for v, d in zip(x[dense], dstar[dense], strict=True)},
            {"min_dstar": thr.dense_min_dstar, "up_to": thr.dense_until},
        )
    )

    span = plateau_span(x, dstar, thr.plateau_value, thr.plateau_tol)
    width = float(x[span[1]] / x[span[0]]) if span else 1.0
    criteria.append(
        check(
            "half_plateau",
            f"A plateau at {thr.plateau_value} ± {thr.plateau_tol} over a contiguous range "
            f"(end points at least {thr.plateau_min_span}× apart in 1/(1 − S)); report the span",
            PAGE,
            span is not None and width >= thr.plateau_min_span,
            {
                "span": [round(float(x[span[0]]), 3), round(float(x[span[1]]), 3)]
                if span
                else None,
                "width_ratio": round(width, 3),
                "dstar_on_plateau": [round(float(d), 4) for d in dstar[span[0] : span[1] + 1]]
                if span
                else [],
                "paper_span": [2.6, 3.7],
            },
            {"value": thr.plateau_value, "tol": thr.plateau_tol, "min_span": thr.plateau_min_span},
        )
    )

    tail_index = int(np.argmin(np.abs(np.log(x) - np.log(thr.tail_at))))
    after = slice(span[1] + 1, None) if span else slice(len(x), None)
    below = bool((dstar[after] < thr.plateau_value - thr.plateau_tol).all())
    tail_ok = abs(float(dstar[tail_index]) - thr.tail_value) <= thr.tail_tol
    criteria.append(
        check(
            "falls_after_plateau",
            f"D* falls below the plateau after it and is {thr.tail_value} ± {thr.tail_tol} at "
            f"1/(1 − S) ≈ {thr.tail_at:g}",
            PAGE,
            span is not None and below and tail_ok and span[1] < tail_index,
            {
                "at": round(float(x[tail_index]), 3),
                "dstar": round(float(dstar[tail_index]), 4),
                "all_after_plateau_below": below,
                "dstar_after_plateau": [round(float(d), 4) for d in dstar[after]],
            },
            {"value": thr.tail_value, "tol": thr.tail_tol},
        )
    )

    rule = {
        "min_clusters": thr.min_clusters,
        "cluster": f"≥ {thr.cluster_min_share:.0%} and ≥ {thr.cluster_min_count} of one "
        "model's represented features",
    }
    if summary.factors is None:
        raise ValueError("E03's cluster criterion needs each feature's tegum factor (D-603)")
    for criterion_id, factors in (("fraction_clusters", summary.factors), ("by_value", None)):
        clusters = fraction_clusters(
            summary.dimensionality,
            summary.represented,
            thr.fraction_tol,
            thr.cluster_min_share,
            thr.cluster_min_count,
            factors,
        )
        found = [name for name, c in clusters.items() if c["cluster"]]
        for c in clusters.values():
            c["at_inverse_density"] = round(float(x[c.pop("at_point")]), 3)
        if factors is not None:
            criteria.append(
                check(
                    criterion_id,
                    f"Nonzero Dᵢ cluster within ±{thr.fraction_tol} of at least "
                    f"{thr.min_clusters} characteristic fractions, each feature counted only "
                    "in the polytope of its fraction (D-603)",
                    "p. 19",
                    len(found) >= thr.min_clusters,
                    {"clusters": found, "by_fraction": clusters},
                    {**rule, "membership": "Dᵢ near the fraction and in its polytope (D-308)"},
                )
            )
        else:
            criteria.append(
                info(
                    "fraction_clusters_by_value",
                    "The same count by Dᵢ alone (the rule of D-306, before D-603): a band "
                    "between 2/5 and 3/8 counts for both",
                    "p. 19",
                    {"clusters": found, "by_fraction": clusters, **rule},
                )
            )

    criteria.append(
        info(
            "sum_dimensionality",
            f"Σᵢ Dᵢ of the best model at each sparsity, against m = {config.n_hidden}",
            "p. 18",
            {
                f"{v:.3f}": {"sum_D": round(float(s), 3), "D_star": round(float(d), 4)}
                for v, s, d in zip(x, summary.sum_dimensionality, dstar, strict=True)
            },
        )
    )
    criteria.append(
        info(
            "tegum_factor_census",
            "Represented features per tegum-factor type (geometry-graph components named by "
            "size, rank and uniformity)",
            "pp. 19–20",
            {f"{v:.3f}": c for v, c in zip(x, summary.census, strict=True)},
        )
    )
    return criteria


# --- run -------------------------------------------------------------------------------------


def run(
    config: UniformGeometryConfig, run_dir: RunDir, device: torch.device, logger: logging.Logger
) -> list[Criterion]:
    x = config.inverse_density()
    logger.info(
        f"training n = {config.n_features}, m = {config.n_hidden}: {len(x)} sparsities × "
        f"{config.n_restarts} restarts"
    )
    grid = train_sparsity_grid(
        "relu_output",
        config.n_features,
        config.n_hidden,
        torch.ones(config.n_features),
        list(1.0 / x),
        config.n_restarts,
        config.train,
        seed=config.seed,
        device=device,
        progress=True,
    )
    logger.info(f"  done in {grid.wall_time_s:.1f} s")
    run_dir.save_checkpoint("relu_output", grid.to_checkpoint())
    thr = config.thresholds
    table = instance_table(grid, thr.represented_norm, thr.superposition_score)
    table.insert(3, "inverse_density", 1.0 / table["feature_probability"])
    run_dir.write_table(table, "metrics")
    summary = summarize(grid, thr)
    features = pd.DataFrame(
        {
            "inverse_density": np.repeat(summary.inverse_density, config.n_features),
            "feature": np.tile(np.arange(config.n_features), len(x)),
            "dimensionality": np.concatenate(summary.dimensionality),
            "represented": np.concatenate(summary.represented),
        }
    )
    run_dir.write_table(features, "best_features")
    return evaluate(config, summary)


# --- figures ---------------------------------------------------------------------------------


def structure_by_node(W: torch.Tensor, thr: UniformGeometryThresholds) -> dict[int, str]:
    """The name of the tegum factor each represented feature belongs to (for coloring)."""
    _, components = metrics.geometry_graph(W, thr.edge_threshold, thr.represented_norm)
    names: dict[int, str] = {}
    for component in components:
        _, _, name = metrics.component_geometry(W, component, thr.rank_rtol, thr.uniform_tol)
        names.update(dict.fromkeys(component, name))
    return names


def figures(run_dir: RunDir) -> list[Path]:
    """D* curve (p. 17), Dᵢ scatter (p. 19), geometry graphs (p. 19), factor census."""
    import yaml

    from superposition.config import from_dict

    config = from_dict(UniformGeometryConfig, yaml.safe_load(run_dir.config_path.read_text()))
    thr = config.thresholds
    grid = GridResult.from_checkpoint(run_dir.load_checkpoint("relu_output"))
    summary = summarize(grid, thr)
    x = summary.inverse_density
    span = plateau_span(x, summary.dstar, thr.plateau_value, thr.plateau_tol)
    plateau = (float(x[span[0]]), float(x[span[1]])) if span else None
    n, m = config.n_features, config.n_hidden
    paths = save_figure(
        dstar_figure(x, summary.dstar, plateau, title=f"E03 · n = {n}, m = {m}: D* = m / ‖W‖²_F"),
        run_dir.figures_dir / "dstar",
    )
    paths += save_figure(
        dimensionality_figure(
            x, summary.dimensionality, summary.dstar, title=f"E03 · n = {n}, m = {m}: every Dᵢ"
        ),
        run_dir.figures_dir / "dimensionality",
    )
    panels = []
    for target in config.graph_at:
        point = int(np.argmin(np.abs(np.log(x) - np.log(target))))
        W, _, _ = grid.best_weights(point)
        graph, _ = metrics.geometry_graph(W, thr.edge_threshold, thr.represented_norm)
        panels.append((f"1/(1−S) = {x[point]:.2f}", graph, structure_by_node(W, thr)))
    paths += save_figure(
        geometry_graph_figure(panels, title="E03 · feature geometry graphs (represented features)"),
        run_dir.figures_dir / "geometry_graphs",
    )
    paths += save_figure(
        census_figure(x, summary.census, n, title="E03 · features in each kind of tegum factor"),
        run_dir.figures_dir / "factor_census",
    )
    return paths


EXPERIMENT = Experiment(
    name="uniform_geometry",
    code="E03",
    title="Uniform superposition: D*, feature dimensionality and polytopes (n = 400, m = 30)",
    paper_pages="pp. 16–21",
    config_cls=UniformGeometryConfig,
    run=run,
    figures=figures,
)
