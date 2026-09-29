"""E04 nonuniform_geometry: deforming polytopes, correlated and anticorrelated features.

Paper: Non-Uniform Superposition, pp. 21–26.

Four parts, all with the ReLU output model and equal importance Iᵢ = 1:

(a) Perturbing a single feature (p. 22): n = 5, m = 2, four features at density 1 − S = 0.05
    and the fifth ("perturbed", feature 0 here) swept from 0.1× to 10× that density. Denser →
    its neighbours are pushed away; sparser → they close in; below a threshold (≈ 2.5× less
    dense in the paper) the pentagon snaps to two digons with the perturbed feature at ≈ 0,
    where the loss curves of the two geometries cross.
(b) Correlated and anticorrelated sets in m = 2 (p. 24): 2 correlated pairs (n = 4) lie
    orthogonal within a pair; 2 anticorrelated pairs lie antipodal; 3 correlated pairs (n = 6)
    lie side by side.
(c) Local almost-orthogonal bases (pp. 24–25): 2 correlated sets of 10 features, m = 10;
    within a set the features are (nearly) orthogonal.
(d) Collapse of correlated features (p. 25): 3 correlated pairs, m = 2, sweeping density;
    sparse → a hexagon with pair members adjacent; denser → each pair collapses onto its
    principal component (a + b)/√2.

Densities the paper does not state for (b)–(d), and how every claim is graded, are recorded
in docs/DECISIONS.md (D-307). Feature 0 is the perturbed feature in (a) so that the
canonicalization (metrics.canonicalize_2d) puts it at the top, as on p. 22.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
from scipy.stats import rankdata

from superposition import metrics
from superposition.acceptance import Criterion, check, info
from superposition.config import ExperimentConfig, TrainConfig
from superposition.data import AnticorrelatedSets, CorrelatedSets, IndependentSparse
from superposition.experiments.base import Experiment
from superposition.models import TiedModel
from superposition.plotting.arrows import plot_feature_arrows
from superposition.plotting.style import (
    INK,
    INK_SECONDARY,
    PANEL,
    SERIES,
    paper_style,
    save_figure,
)
from superposition.plotting.weights import plot_gram
from superposition.runs import RunDir
from superposition.train import train_from_config

# Set colors as on p. 24 (red, blue, green).
SET_COLORS = ("#c8262c", "#2446b8", "#1a9e5c")
GEOMETRY_COLORS = {
    "pentagon": "#f08c1a",
    "digons (perturbed dropped)": "#d8c20a",
    "digons (other dropped)": "#a8950a",
    "triangle": "#3fbf7f",
    "other": "#8a8984",
}


@dataclass
class PerturbedSetting:
    """Part (a), p. 22."""

    n_features: int = 5
    n_hidden: int = 2
    base_density: float = 0.05
    ratio_min: float = 0.1
    ratio_max: float = 10.0
    n_ratios: int = 21
    n_restarts: int = 32
    eval_samples: int = 200_000


@dataclass
class CorrelationSetting:
    """Parts (b) and (c): set sizes, density and restarts (D-307)."""

    density: float = 0.02
    n_restarts: int = 16
    local_basis_sets: int = 2
    local_basis_set_size: int = 10
    local_basis_hidden: int = 10
    local_basis_restarts: int = 8


@dataclass
class CollapseSetting:
    """Part (d), p. 25: three correlated pairs in m = 2, density swept."""

    n_sets: int = 3
    set_size: int = 2
    density_min: float = 0.01
    density_max: float = 0.5
    n_densities: int = 12
    n_restarts: int = 16


@dataclass
class NonuniformThresholds:
    """Acceptance thresholds (SPEC §7 E04; D-307)."""

    represented_norm: float = 0.5
    antipodal_cos: float = -0.95
    orthogonal_abs_cos: float = 0.1
    collapsed_cos: float = 0.95
    zero_norm: float = 0.1
    trend_min_spearman: float = 0.8
    angle_margin_deg: float = 1.0


def _default_train() -> TrainConfig:
    return TrainConfig(steps=10_000, batch_size=256)


@dataclass
class NonuniformGeometryConfig(ExperimentConfig):
    """E04 settings (SPEC §7)."""

    perturbed: PerturbedSetting = field(default_factory=PerturbedSetting)
    correlation: CorrelationSetting = field(default_factory=CorrelationSetting)
    collapse: CollapseSetting = field(default_factory=CollapseSetting)
    thresholds: NonuniformThresholds = field(default_factory=NonuniformThresholds)
    train: TrainConfig = field(default_factory=_default_train)


# --- geometry helpers (one instance, m = 2) ---------------------------------------------------


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation: the Pearson correlation of the ranks (ties get mean ranks)."""
    return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])


def cyclic_neighbours(W: torch.Tensor, feature: int, features: list[int]) -> tuple[int, int]:
    """The two angular neighbours of ``feature`` among ``features`` (2D, one instance)."""
    angles = metrics.angles_2d(W[None])[0].numpy()
    order = sorted(features, key=lambda i: angles[i])
    k = order.index(feature)
    return order[k - 1], order[(k + 1) % len(order)]


def angle_between_deg(W: torch.Tensor, i: int, j: int) -> float:
    cos = float(metrics.cosine_similarity_matrix(W[None])[0, i, j])
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def neighbour_angle_deg(W: torch.Tensor, feature: int) -> float:
    """Mean angle between ``feature`` and its two angular neighbours (72° in a pentagon)."""
    rep = list(range(W.shape[1]))
    left, right = cyclic_neighbours(W, feature, rep)
    return (angle_between_deg(W, feature, left) + angle_between_deg(W, feature, right)) / 2


def classify_perturbed(W: torch.Tensor, perturbed: int, thr: NonuniformThresholds) -> str:
    """Geometry of one n = 5, m = 2 solution, as in the p. 22 loss-curve legend."""
    rep = metrics.represented_mask(W[None], thr.represented_norm)[0]
    k = int(rep.sum())
    if k == 5:
        return "pentagon"
    if k == 3:
        return "triangle"
    if k == 4:
        cos = metrics.cosine_similarity_matrix(W[None])[0]
        idx = torch.nonzero(rep).flatten().tolist()
        partnered = all(
            any(float(cos[i, j]) < thr.antipodal_cos for j in idx if j != i) for i in idx
        )
        if partnered:
            return (
                "digons (perturbed dropped)"
                if not bool(rep[perturbed])
                else "digons (other dropped)"
            )
    return "other"


def adjacent_in_cycle(W: torch.Tensor, pairs: list[tuple[int, int]]) -> bool:
    """Whether each pair's members are angular neighbours among all features (one instance)."""
    features = list(range(W.shape[1]))
    return all(b in cyclic_neighbours(W, a, features) for a, b in pairs)


def set_pairs(n_sets: int, set_size: int) -> list[list[int]]:
    """Feature indices of each set (set-major layout of data.CorrelatedSets)."""
    return [list(range(s * set_size, (s + 1) * set_size)) for s in range(n_sets)]


def mean_within_set_cos(W: torch.Tensor, sets: list[list[int]]) -> float:
    cos = metrics.cosine_similarity_matrix(W[None])[0]
    values = [
        float(cos[i, j]) for group in sets for a, i in enumerate(group) for j in group[a + 1 :]
    ]
    return float(np.mean(values))


def mean_abs_cos(W: torch.Tensor, sets: list[list[int]], within: bool) -> float:
    """Mean |cos| over pairs of features in the same set (``within``) or in different sets."""
    cos = metrics.cosine_similarity_matrix(W[None])[0].abs()
    set_of = {i: s for s, group in enumerate(sets) for i in group}
    n = W.shape[1]
    values = [
        float(cos[i, j])
        for i in range(n)
        for j in range(i + 1, n)
        if (set_of[i] == set_of[j]) == within
    ]
    return float(np.mean(values))


# --- training --------------------------------------------------------------------------------


def _train(dist: Any, n_features: int, n_hidden: int, config: NonuniformGeometryConfig,
           device: torch.device, eval_samples: int | None = None) -> dict[str, Any]:  # fmt: skip
    model = TiedModel.from_kind("relu_output", n_features, n_hidden, dist.n_inst, seed=config.seed)
    model = model.to(device)
    train_config = config.train
    if eval_samples is not None:
        train_config = TrainConfig(**{**train_config.__dict__, "eval_samples": eval_samples})
    result = train_from_config(
        model, dist, torch.ones(n_features), train_config, seed=config.seed, progress=True
    )
    return {
        "W": model.W.detach().cpu(),
        "b": model.b.detach().cpu(),
        "final_loss": result.final_loss,
        "wall_time_s": result.wall_time_s,
    }


def train_perturbed(config: NonuniformGeometryConfig, device: torch.device) -> dict[str, Any]:
    s = config.perturbed
    ratios = np.geomspace(s.ratio_min, s.ratio_max, s.n_ratios)
    n_inst = s.n_ratios * s.n_restarts
    probability = torch.full((n_inst, s.n_features), s.base_density)
    probability[:, 0] = torch.tensor(np.repeat(ratios * s.base_density, s.n_restarts))
    state = _train(
        IndependentSparse(probability, n_inst, s.n_features),
        s.n_features,
        s.n_hidden,
        config,
        device,
        s.eval_samples,
    )
    state.update(ratios=ratios, condition=torch.arange(s.n_ratios).repeat_interleave(s.n_restarts))
    return state


def train_correlation(config: NonuniformGeometryConfig, device: torch.device) -> dict[str, Any]:
    c = config.correlation
    setups = {
        "correlated_2x2": CorrelatedSets(2, 2, c.density, c.n_restarts),
        "anticorrelated_2x2": AnticorrelatedSets(2, 2, c.density, c.n_restarts),
        "correlated_3x2": CorrelatedSets(3, 2, c.density, c.n_restarts),
    }
    out = {}
    for name, dist in setups.items():
        out[name] = _train(dist, dist.n_features, 2, config, device)
        out[name].update(n_sets=dist.n_sets, set_size=dist.set_size)
    dist = CorrelatedSets(
        c.local_basis_sets, c.local_basis_set_size, c.density, c.local_basis_restarts
    )
    out["local_bases"] = _train(dist, dist.n_features, c.local_basis_hidden, config, device)
    out["local_bases"].update(n_sets=dist.n_sets, set_size=dist.set_size)
    return out


def train_collapse(config: NonuniformGeometryConfig, device: torch.device) -> dict[str, Any]:
    s = config.collapse
    densities = np.geomspace(s.density_min, s.density_max, s.n_densities)
    n_inst = s.n_densities * s.n_restarts
    p = torch.tensor(np.repeat(densities, s.n_restarts), dtype=torch.float32)
    dist = CorrelatedSets(s.n_sets, s.set_size, p, n_inst)
    state = _train(dist, dist.n_features, 2, config, device)
    state.update(
        densities=densities,
        condition=torch.arange(s.n_densities).repeat_interleave(s.n_restarts),
        n_sets=s.n_sets,
        set_size=s.set_size,
    )
    return state


def best_per_condition(state: dict[str, Any]) -> list[int]:
    """Lowest-loss instance of each condition."""
    loss, cond = state["final_loss"], state["condition"]
    return [
        int(torch.nonzero(cond == c).flatten()[loss[cond == c].argmin()])
        for c in range(int(cond.max()) + 1)
    ]


# --- evaluation ------------------------------------------------------------------------------


def evaluate_perturbed(
    config: NonuniformGeometryConfig, state: dict[str, Any]
) -> tuple[list[Criterion], pd.DataFrame]:
    thr = config.thresholds
    ratios, cond, W_all, loss = state["ratios"], state["condition"], state["W"], state["final_loss"]
    kinds = [classify_perturbed(W_all[k], 0, thr) for k in range(len(W_all))]
    frame = pd.DataFrame(
        {
            "ratio": ratios[cond.numpy()],
            "restart": np.tile(np.arange(config.perturbed.n_restarts), len(ratios)),
            "geometry": kinds,
            "loss": loss.numpy(),
            "perturbed_norm": metrics.feature_norms(W_all)[:, 0].numpy(),
        }
    )
    best_overall = best_per_condition(state)
    best_pentagon, share = [], []
    for c in range(len(ratios)):
        members = [k for k in torch.nonzero(cond == c).flatten().tolist() if kinds[k] == "pentagon"]
        if members:
            k = min(members, key=lambda i: float(loss[i]))
            best_pentagon.append(k)
            share.append(neighbour_angle_deg(W_all[k], 0))
        else:
            best_pentagon.append(-1)
            share.append(float("nan"))
    share_arr = np.array(share)
    has = ~np.isnan(share_arr)
    denser, sparser = has & (ratios >= 1 - 1e-9), has & (ratios <= 1 + 1e-9)

    def trend(mask: np.ndarray) -> float:
        return spearman(ratios[mask], share_arr[mask]) if mask.sum() >= 3 else float("nan")

    rho_dense, rho_sparse = trend(denser), trend(sparser)
    densest = share_arr[denser][-1] if denser.any() else float("nan")
    sparsest = share_arr[sparser][0] if sparser.any() else float("nan")
    uniform = 72.0
    angles = {
        f"{r:.3g}": round(float(a), 2)
        for r, a in zip(ratios, share_arr, strict=True)
        if not np.isnan(a)
    }
    criteria = [
        check(
            "denser_neighbours_repel",
            "Denser perturbed feature → its neighbours move away: the neighbour angle grows with "
            f"density (Spearman ≥ {thr.trend_min_spearman} over ratios ≥ 1; densest > 72° + {thr.angle_margin_deg}°)",
            "p. 22",
            rho_dense >= thr.trend_min_spearman and densest > uniform + thr.angle_margin_deg,
            {
                "spearman": round(rho_dense, 3),
                "densest_angle_deg": round(float(densest), 2),
                "angle_by_ratio": angles,
            },
            {
                "min_spearman": thr.trend_min_spearman,
                "densest_above_deg": uniform + thr.angle_margin_deg,
            },
            group="a_perturbed",
        ),
        check(
            "sparser_neighbours_close_in",
            "Sparser perturbed feature → its neighbours close in: the neighbour angle shrinks with "
            f"density (Spearman ≥ {thr.trend_min_spearman} over pentagon solutions at ratios ≤ 1; "
            f"sparsest pentagon < 72° − {thr.angle_margin_deg}°)",
            "p. 22",
            rho_sparse >= thr.trend_min_spearman and sparsest < uniform - thr.angle_margin_deg,
            {
                "spearman": round(rho_sparse, 3),
                "sparsest_pentagon_angle_deg": round(float(sparsest), 2),
                "sparsest_pentagon_ratio": round(float(ratios[sparser][0]), 4)
                if sparser.any()
                else None,
            },
            {
                "min_spearman": thr.trend_min_spearman,
                "sparsest_below_deg": uniform - thr.angle_margin_deg,
            },
            group="a_perturbed",
        ),
    ]
    best_kinds = [kinds[k] for k in best_overall]
    collapsed = [
        r
        for r, kind in zip(ratios, best_kinds, strict=True)
        if kind == "digons (perturbed dropped)"
    ]
    W0 = W_all[best_overall[0]]
    norm0 = float(metrics.feature_norms(W0[None])[0, 0])
    threshold = max(collapsed) if collapsed else None
    criteria.append(
        check(
            "collapse_to_digons",
            f"Sparsest setting: the best solution is two digons with the perturbed feature at ≈ 0 "
            f"(‖W‖ < {thr.zero_norm}); report the threshold",
            "p. 22",
            best_kinds[0] == "digons (perturbed dropped)" and norm0 < thr.zero_norm,
            {
                "sparsest_ratio": round(float(ratios[0]), 4),
                "best_geometry": best_kinds[0],
                "perturbed_norm": round(norm0, 4),
                "largest_ratio_with_collapse": round(float(threshold), 4) if threshold else None,
                "best_geometry_by_ratio": {
                    f"{r:.3g}": g for r, g in zip(ratios, best_kinds, strict=True)
                },
                "paper_threshold": "≈ 1/2.5 = 0.4",
            },
            {"geometry": "digons (perturbed dropped)", "norm_below": thr.zero_norm},
            group="a_perturbed",
        )
    )
    kind_arr, loss_arr, cond_arr = np.array(kinds), loss.numpy(), cond.numpy()
    crossing = {
        f"{r:.3g}": {
            g: round(float(loss_arr[(cond_arr == c) & (kind_arr == g)].min()), 6)
            for g in sorted(set(kind_arr[cond_arr == c]))
        }
        for c, r in enumerate(ratios)
    }
    criteria.append(
        info(
            "loss_by_geometry",
            "Lowest loss of each geometry at each density ratio (the crossing loss curves)",
            "p. 22",
            crossing,
            group="a_perturbed",
        )
    )
    return criteria, frame


def evaluate_correlation(
    config: NonuniformGeometryConfig, states: dict[str, Any]
) -> list[Criterion]:
    thr = config.thresholds
    best = {name: int(s["final_loss"].argmin()) for name, s in states.items()}

    def W_of(name: str) -> torch.Tensor:
        return states[name]["W"][best[name]]

    def sets_of(name: str) -> list[list[int]]:
        return set_pairs(states[name]["n_sets"], states[name]["set_size"])

    corr = mean_within_set_cos(W_of("correlated_2x2"), sets_of("correlated_2x2"))
    anti = mean_within_set_cos(W_of("anticorrelated_2x2"), sets_of("anticorrelated_2x2"))
    criteria = [
        check(
            "anticorrelated_more_negative",
            "Anticorrelated features interfere more negatively (lower mean within-set cosine) "
            "than correlated features (2 sets of 2, m = 2)",
            "pp. 23–24",
            anti < corr,
            {"anticorrelated_mean_cos": round(anti, 4), "correlated_mean_cos": round(corr, 4)},
            "anticorrelated < correlated",
            group="b_correlation",
        )
    ]
    cos = metrics.cosine_similarity_matrix(W_of("correlated_2x2")[None])[0]
    within = [abs(float(cos[a, b])) for a, b in sets_of("correlated_2x2")]
    criteria.append(
        check(
            "correlated_orthogonal_when_room",
            f"2 correlated pairs in m = 2: each pair orthogonal (|cos| < {thr.orthogonal_abs_cos})",
            "p. 24",
            max(within) < thr.orthogonal_abs_cos,
            {"within_pair_abs_cos": [round(v, 4) for v in within]},
            {"abs_cos_max": thr.orthogonal_abs_cos},
            group="b_correlation",
        )
    )
    W3 = W_of("correlated_3x2")
    pairs3 = [tuple(g) for g in sets_of("correlated_3x2")]
    represented = int(metrics.represented_mask(W3[None], thr.represented_norm)[0].sum())
    adjacent = adjacent_in_cycle(W3, pairs3)  # type: ignore[arg-type]
    criteria.append(
        check(
            "correlated_adjacent_when_no_room",
            "3 correlated pairs in m = 2: all 6 represented and each pair's members side by side "
            "(angular neighbours)",
            "p. 24",
            represented == 6 and adjacent,
            {
                "represented": represented,
                "pairs_adjacent": adjacent,
                "within_pair_angle_deg": [round(angle_between_deg(W3, a, b), 1) for a, b in pairs3],
            },
            {"represented": 6, "pairs_adjacent": True},
            group="b_correlation",
        )
    )
    per_restart = {
        name: [round(mean_within_set_cos(W, sets_of(name)), 3) for W in states[name]["W"]]
        for name in ("correlated_2x2", "anticorrelated_2x2", "correlated_3x2")
    }
    criteria.append(
        info(
            "within_set_cos_all_restarts",
            "Mean within-set cosine of every restart (not just the best)",
            "pp. 23–24",
            per_restart,
            group="b_correlation",
        )
    )
    WL = W_of("local_bases")
    sets_l = sets_of("local_bases")
    within_l, across_l = mean_abs_cos(WL, sets_l, True), mean_abs_cos(WL, sets_l, False)
    all_restarts = [
        (round(mean_abs_cos(W, sets_l, True), 4), round(mean_abs_cos(W, sets_l, False), 4))
        for W in states["local_bases"]["W"]
    ]
    criteria.append(
        check(
            "local_almost_orthogonal_bases",
            "2 correlated sets of 10 in m = 10: mean |cos| within a set < mean |cos| across sets",
            "pp. 24–25",
            within_l < across_l,
            {
                "within": round(within_l, 4),
                "across": round(across_l, 4),
                "all_restarts_within_across": all_restarts,
            },
            "within < across",
            group="c_local_bases",
        )
    )
    return criteria


def evaluate_collapse(config: NonuniformGeometryConfig, state: dict[str, Any]) -> list[Criterion]:
    thr = config.thresholds
    densities = state["densities"]
    sets = set_pairs(state["n_sets"], state["set_size"])
    pairs = [tuple(g) for g in sets]
    best = best_per_condition(state)
    W_best = [state["W"][k] for k in best]
    mean_cos = np.array([mean_within_set_cos(W, sets) for W in W_best])
    collapsed = [
        sum(
            float(metrics.cosine_similarity_matrix(W[None])[0, a, b]) > thr.collapsed_cos
            for a, b in pairs
        )
        for W in W_best
    ]
    W0 = W_best[0]
    represented = int(metrics.represented_mask(W0[None], thr.represented_norm)[0].sum())
    adjacent = adjacent_in_cycle(W0, pairs)  # type: ignore[arg-type]
    criteria = [
        check(
            "sparse_hexagon_pairs_adjacent",
            f"Sparsest density: all 6 represented, pair members adjacent, no pair collapsed "
            f"(cos < {thr.collapsed_cos})",
            "p. 25",
            represented == 6 and adjacent and collapsed[0] == 0,
            {
                "density": round(float(densities[0]), 4),
                "represented": represented,
                "pairs_adjacent": adjacent,
                "within_pair_cos": [
                    round(float(metrics.cosine_similarity_matrix(W0[None])[0, a, b]), 3)
                    for a, b in pairs
                ],
            },
            {"represented": 6, "pairs_adjacent": True, "collapsed": 0},
            group="d_collapse",
        )
    ]
    rho = spearman(densities, mean_cos)
    criteria.append(
        check(
            "collapse_with_density",
            f"Within-pair cosine rises toward 1 as density rises (Spearman ≥ {thr.trend_min_spearman}; "
            f"densest setting mean cos > {thr.collapsed_cos})",
            "p. 25",
            rho >= thr.trend_min_spearman and mean_cos[-1] > thr.collapsed_cos,
            {
                "spearman": round(rho, 3),
                "mean_within_pair_cos": {
                    f"{d:.3g}": round(float(c), 3) for d, c in zip(densities, mean_cos, strict=True)
                },
            },
            {"min_spearman": thr.trend_min_spearman, "densest_above": thr.collapsed_cos},
            group="d_collapse",
        )
    )
    norms = [metrics.feature_norms(W[None])[0].numpy() for W in W_best]
    criteria.append(
        info(
            "collapsed_sets",
            "Collapsed pairs (cos > 0.95) per density, and the norms of collapsed features "
            "(the paper draws them at r = 1/√2 ≈ 0.707)",
            "p. 25",
            {
                f"{d:.3g}": {
                    "collapsed_pairs": n_col,
                    "collapsed_norms": [
                        round(float(norm[i]), 3)
                        for a, b in pairs
                        if float(metrics.cosine_similarity_matrix(W[None])[0, a, b])
                        > thr.collapsed_cos
                        for i in (a, b)
                    ],
                }
                for d, n_col, W, norm in zip(densities, collapsed, W_best, norms, strict=True)
            },
            group="d_collapse",
        )
    )
    return criteria


# --- run -------------------------------------------------------------------------------------


def run(
    config: NonuniformGeometryConfig, run_dir: RunDir, device: torch.device, logger: logging.Logger
) -> list[Criterion]:
    s = config.perturbed
    logger.info(f"(a) perturbed feature: {s.n_ratios} density ratios × {s.n_restarts} restarts")
    perturbed = train_perturbed(config, device)
    run_dir.save_checkpoint("a_perturbed", perturbed)
    logger.info(f"  done in {perturbed['wall_time_s']:.1f} s")
    logger.info("(b, c) correlated and anticorrelated sets")
    correlation = train_correlation(config, device)
    run_dir.save_checkpoint("bc_correlation", correlation)
    c = config.collapse
    logger.info(f"(d) collapse: {c.n_densities} densities × {c.n_restarts} restarts")
    collapse = train_collapse(config, device)
    run_dir.save_checkpoint("d_collapse", collapse)
    logger.info(f"  done in {collapse['wall_time_s']:.1f} s")
    criteria, frame = evaluate_perturbed(config, perturbed)
    run_dir.write_table(frame, "perturbed_restarts")
    criteria += evaluate_correlation(config, correlation)
    criteria += evaluate_collapse(config, collapse)
    return criteria


# --- figures ---------------------------------------------------------------------------------


def _canonical(W: torch.Tensor) -> np.ndarray:
    return metrics.canonicalize_2d(W[None])[0].numpy()


def _set_colors(n_sets: int, set_size: int) -> list[str]:
    return [SET_COLORS[s % len(SET_COLORS)] for s in range(n_sets) for _ in range(set_size)]


def perturbed_figures(
    config: NonuniformGeometryConfig, state: dict[str, Any], run_dir: RunDir
) -> list[Path]:
    thr = config.thresholds
    ratios, cond, W_all, loss = state["ratios"], state["condition"], state["W"], state["final_loss"]
    kinds = [classify_perturbed(W_all[k], 0, thr) for k in range(len(W_all))]
    cmap = plt.get_cmap("viridis")
    color_of = lambda r: cmap((np.log10(r) + 1) / 2)  # noqa: E731 (0.1× → 0, 10× → 1)
    paths: list[Path] = []
    with paper_style():
        fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.9))
        for ax, wanted, title in (
            (axes[0], "digons (perturbed dropped)", "Digon (square) solutions"),
            (axes[1], "pentagon", "Pentagon solutions"),
        ):
            ax.set_facecolor(PANEL)
            circle = Circle((0, 0), 1.0, fill=False, color=INK_SECONDARY, linewidth=0.5, alpha=0.4)
            ax.add_patch(circle)
            for c, r in enumerate(ratios):
                members = [
                    k for k in torch.nonzero(cond == c).flatten().tolist() if kinds[k] == wanted
                ]
                if not members:
                    continue
                W = _canonical(W_all[min(members, key=lambda i: float(loss[i]))])
                for i in range(W.shape[1]):
                    ax.plot([0, W[0, i]], [0, W[1, i]], color=color_of(r), linewidth=0.8, alpha=0.6)
                ax.scatter(W[0], W[1], s=14, color=color_of(r), zorder=3, linewidths=0)
            ax.set_xlim(-1.45, 1.45)
            ax.set_ylim(-1.45, 1.45)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(title + " (best per density)", fontsize=9.5, loc="left")
        mappable = ScalarMappable(cmap=cmap, norm=Normalize(-1, 1))
        cbar = fig.colorbar(mappable, ax=axes, ticks=[-1, 0, 1], fraction=0.03, pad=0.02)
        cbar.ax.set_yticklabels(["0.1×", "1×", "10×"])
        cbar.set_label("perturbed feature's density (relative)", fontsize=8)
        fig.suptitle(
            "E04 (a) · n = 5, m = 2: perturbing one feature's density (feature 0, top)",
            x=0.02,
            ha="left",
            fontsize=10.5,
        )
    paths += save_figure(fig, run_dir.figures_dir / "perturbed_solutions")
    with paper_style():
        fig, ax = plt.subplots(figsize=(7.2, 3.6))
        for kind, color in GEOMETRY_COLORS.items():
            mask = np.array([k == kind for k in kinds])
            if mask.any():
                ax.scatter(
                    ratios[cond.numpy()][mask],
                    loss.numpy()[mask],
                    s=9,
                    color=color,
                    label=kind,
                    alpha=0.8,
                    linewidths=0,
                )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("perturbed feature's density relative to the others (0.1× … 10×)")
        ax.set_ylabel("final loss (every restart)")
        ax.axvline(1.0, color=INK_SECONDARY, linewidth=0.6, linestyle=(0, (3, 3)))
        ax.legend(loc="upper left", fontsize=7.5)
        ax.set_title(
            "E04 (a) · each geometry has its own loss curve; the best one switches where they cross",
            loc="left",
            fontsize=9.5,
        )
        fig.tight_layout()
    paths += save_figure(fig, run_dir.figures_dir / "perturbed_losses")
    return paths


def correlation_figures(states: dict[str, Any], run_dir: RunDir, per_row: int = 5) -> list[Path]:
    rows = [
        ("correlated_2x2", "2 correlated pairs: orthogonal"),
        ("anticorrelated_2x2", "2 anticorrelated pairs: antipodal"),
        ("correlated_3x2", "3 correlated pairs: side by side"),
    ]
    paths: list[Path] = []
    with paper_style():
        fig, axes = plt.subplots(
            len(rows), per_row, figsize=(2.1 * per_row, 2.3 * len(rows)), squeeze=False
        )
        for r, (name, label) in enumerate(rows):
            s = states[name]
            order = torch.argsort(s["final_loss"])[:per_row].tolist()
            colors = _set_colors(s["n_sets"], s["set_size"])
            for c, k in enumerate(order):
                plot_feature_arrows(
                    axes[r, c], _canonical(s["W"][k]), annotate=False, colors=colors, limit=1.4
                )
                axes[r, c].set_title(
                    f"loss {float(s['final_loss'][k]):.2e}",
                    fontsize=7,
                    loc="left",
                    color=INK_SECONDARY,
                )
            axes[r, 0].text(
                -0.1,
                0.5,
                label,
                transform=axes[r, 0].transAxes,
                rotation=90,
                ha="center",
                va="center",
                fontsize=8.5,
                color=INK,
            )
        fig.suptitle(
            "E04 (b) · m = 2: five lowest-loss restarts each; color = set",
            x=0.02,
            ha="left",
            fontsize=10.5,
        )
        fig.tight_layout(rect=(0.02, 0, 1, 0.95))
    paths += save_figure(fig, run_dir.figures_dir / "correlated_sets")
    s = states["local_bases"]
    k = int(s["final_loss"].argmin())
    W = s["W"][k]
    with paper_style():
        fig, ax = plt.subplots(figsize=(4.4, 4.2))
        plot_gram(ax, metrics.gram(W[None])[0].numpy())
        half = s["set_size"] - 0.5
        ax.axhline(half, color=INK, linewidth=0.8, linestyle=(0, (3, 2)))
        ax.axvline(half, color=INK, linewidth=0.8, linestyle=(0, (3, 2)))
        ax.set_title("E04 (c) · WᵀW, 2 correlated sets of 10 in m = 10", fontsize=9.5, loc="left")
        fig.tight_layout()
    paths += save_figure(fig, run_dir.figures_dir / "local_bases_gram")
    return paths


def collapse_figures(
    state: dict[str, Any], run_dir: RunDir, thr: NonuniformThresholds
) -> list[Path]:
    densities = state["densities"]
    best = best_per_condition(state)
    colors = _set_colors(state["n_sets"], state["set_size"])
    sets = set_pairs(state["n_sets"], state["set_size"])
    picks = np.unique(np.linspace(0, len(densities) - 1, 6).round().astype(int))
    paths: list[Path] = []
    with paper_style():
        fig, axes = plt.subplots(1, len(picks) + 1, figsize=(2.3 * (len(picks) + 1) + 0.6, 2.8),
                                 gridspec_kw={"width_ratios": [1] * len(picks) + [1.6]})  # fmt: skip
        for ax, c in zip(axes[:-1], picks[::-1], strict=True):
            plot_feature_arrows(
                ax, _canonical(state["W"][best[c]]), annotate=False, colors=colors, limit=1.4
            )
            circle = Circle(
                (0, 0),
                1 / np.sqrt(2),
                fill=False,
                color=INK_SECONDARY,
                linewidth=0.5,
                linestyle=(0, (2, 2)),
            )
            ax.add_patch(circle)
            ax.set_title(f"1 − S = {densities[c]:.3g}", fontsize=8.5, loc="left")
        mean_cos = [mean_within_set_cos(state["W"][k], sets) for k in best]
        ax = axes[-1]
        ax.plot(densities, mean_cos, marker="o", color=SERIES[0])
        ax.axhline(thr.collapsed_cos, color=INK_SECONDARY, linewidth=0.8, linestyle=(0, (3, 3)))
        ax.set_xscale("log")
        ax.set_ylim(-1.05, 1.05)
        ax.set_xlabel("density 1 − S")
        ax.set_ylabel("mean within-pair cos")
        ax.grid(axis="y")
        fig.suptitle("E04 (d) · three correlated pairs in m = 2: more PCA-like as density rises (dashed circle r = 1/√2)",
                     x=0.02, ha="left", fontsize=10.5)  # fmt: skip
        fig.tight_layout(rect=(0, 0, 1, 0.9))
    paths += save_figure(fig, run_dir.figures_dir / "collapse")
    return paths


def figures(run_dir: RunDir) -> list[Path]:
    """Perturbed pentagon (p. 22), correlated sets (p. 24), local bases (p. 25), collapse (p. 25)."""
    import yaml

    from superposition.config import from_dict

    config = from_dict(NonuniformGeometryConfig, yaml.safe_load(run_dir.config_path.read_text()))
    paths = perturbed_figures(config, run_dir.load_checkpoint("a_perturbed"), run_dir)
    paths += correlation_figures(run_dir.load_checkpoint("bc_correlation"), run_dir)
    paths += collapse_figures(run_dir.load_checkpoint("d_collapse"), run_dir, config.thresholds)
    return paths


EXPERIMENT = Experiment(
    name="nonuniform_geometry",
    code="E04",
    title="Non-uniform superposition: perturbed pentagon, correlated sets, collapse to PCA",
    paper_pages="pp. 21–26",
    config_cls=NonuniformGeometryConfig,
    run=run,
    figures=figures,
)
