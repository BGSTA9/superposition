"""E05 learning_dynamics: energy-level jumps and learning as geometric transformations.

Paper: Superposition and Learning Dynamics, pp. 26–28.

**(a) Discrete "energy level" jumps (p. 26).** In the uniform setup of p. 17 (many features of
equal importance and sparsity), features jump between dimensionalities during training, and
each jump comes with a sudden drop in the loss. We train n = 400, m = 30 models at
1/(1 − S) = 3, inside the D* = 1/2 plateau (the paper's plateau spans roughly [2.6, 3.7] on
p. 17), snapshot every feature's Dᵢ every k steps, and evaluate the loss of every snapshot on
one fixed batch so the curve has no sampling noise between snapshots.

**(b) Learning as geometric transformations (p. 27).** n = 6 features in m = 3 dimensions,
as two correlated sets of three. The paper's trajectory: (A) weights start close to zero,
(B) the two sets push apart, (C) each set expands into a triangle, (D) the triangles rotate
into an antiprism, an octahedron whose antipodal pairs join features of different sets. Other
trajectories exist ("sometimes the model gets pulled directly into the antiprism
configuration from the start or organize features into antipodal pairs"). We classify every
snapshot into a geometric stage and report the stage sequences (trajectory types).

How the SPEC §7 criteria are operationalized (what counts as a jump, a loss drop, a stage) is
recorded in docs/DECISIONS.md (D-401 – D-404).
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from jaxtyping import Float
from torch import Tensor

from superposition import metrics
from superposition.acceptance import Criterion, check, info
from superposition.config import ExperimentConfig, TrainConfig
from superposition.data import CorrelatedSets, IndependentSparse, geometric_importance
from superposition.experiments.base import Experiment
from superposition.models import TiedModel
from superposition.plotting.dynamics import (
    dimensionality_over_training,
    stage_raster,
    trajectories_3d,
    trajectory_panels,
)
from superposition.plotting.style import save_figure
from superposition.runs import RunDir
from superposition.train import importance_weighted_loss, train_from_config
from superposition.utils import EVAL_STREAM, derive_seed, make_generator

PAGE_A = "p. 26"
PAGE_B = "p. 27"

# Geometric stages of part (b), in the order the paper's trajectory visits them (p. 27).
STAGES: tuple[str, ...] = ("small", "collapsed", "transition", "prism", "octahedron")
SMALL, COLLAPSED, TRANSITION, PRISM, OCTAHEDRON = range(len(STAGES))


# --- configuration ------------------------------------------------------------------------------


def _energy_train() -> TrainConfig:
    return TrainConfig(
        steps=80_000, batch_size=256, log_every=100, snapshot_every=100, eval_samples=20_000
    )


def _transformation_train() -> TrainConfig:
    return TrainConfig(steps=30_000, log_every=50, snapshot_every=50, eval_samples=20_000)


@dataclass
class EnergyLevelSetup:
    """Part (a): the uniform setup of p. 17 at one sparsity inside the 1/2 plateau."""

    n_features: int = 400
    n_hidden: int = 30
    importance_decay: float = 1.0
    inverse_density: float = 3.0  # 1/(1 − S)
    n_restarts: int = 4
    loss_eval_samples: int = 8192  # one fixed batch, reused for every snapshot's loss


@dataclass
class JumpThresholds:
    """What counts as a jump and as a coinciding loss drop (D-402)."""

    jump: float = 0.15  # |ΔDᵢ| (SPEC §7)
    jump_window: int = 2_000  # steps: the "short window" of the jump
    stable_window: int = 5_000  # steps the feature sits at one level before it jumps
    stable_tol: float = 0.05  # max |Dᵢ − Dᵢ(start)| during the stable window
    drop_factor: float = 3.0  # loss falls ≥ this × faster than during the stable window
    level_tol: float = 0.05  # for the final census of features at 1/2 and at 0


@dataclass
class TransformationSetup:
    """Part (b): two correlated sets of three features in m = 3 (p. 27)."""

    n_sets: int = 2
    set_size: int = 3
    n_hidden: int = 3
    importance_decay: float = 1.0
    set_probabilities: list[float] = field(default_factory=lambda: [0.01, 0.03, 0.1])
    # Multipliers on the Xavier init std. The first one is graded: p. 27 says weights start
    # "randomly close to zero" (D-403). Others are reported as variants.
    init_scales: list[float] = field(default_factory=lambda: [0.1])
    n_restarts: int = 16
    loss_eval_samples: int = 20_000
    train: TrainConfig = field(default_factory=_transformation_train)


@dataclass
class StageThresholds:
    """Geometric stage of a snapshot in part (b) (D-404)."""

    small_norm: float = 0.3  # mean ‖Wᵢ‖ below this: weights still near the origin (A)
    collapsed_cos: float = 0.8  # each set bunched along one direction (B)
    opposed_cos: float = -0.9  # the two set centroids point in opposite directions
    antipodal_cos: float = -0.95  # a cross-set partner this close to −1 is antipodal
    orthogonal_cos: float = 0.1  # within-set |cos| below this: orthogonal
    represented_norm: float = 0.5
    min_stage_steps: int = 1_000  # shorter stages are left out of a trajectory type


@dataclass
class LearningDynamicsConfig(ExperimentConfig):
    """E05 settings. ``train`` holds part (a)'s optimizer settings; part (b) has its own."""

    energy: EnergyLevelSetup = field(default_factory=EnergyLevelSetup)
    jumps: JumpThresholds = field(default_factory=JumpThresholds)
    transformations: TransformationSetup = field(default_factory=TransformationSetup)
    stages: StageThresholds = field(default_factory=StageThresholds)
    train: TrainConfig = field(default_factory=_energy_train)


# --- shared helpers ---------------------------------------------------------------------------


@torch.no_grad()
def snapshot_losses(
    W: Float[Tensor, "time inst hidden feat"],
    b: Float[Tensor, "time inst feat"],
    x: Float[Tensor, "batch inst feat"],
    importance: Float[Tensor, "*inst feat"],
) -> Float[Tensor, "time inst"]:
    """Loss of every snapshot of a ReLU output model on one fixed batch ``x``.

    Using the same inputs for every snapshot means differences between snapshots come from
    the weights alone, so small loss drops are visible (p. 26 plots a loss curve).
    """
    n_time, n_inst, n_hidden, n_feat = W.shape
    model = TiedModel(n_feat, n_hidden, n_inst, out_act="relu")
    losses = torch.empty(n_time, n_inst)
    for t in range(n_time):
        model.W.copy_(W[t])
        model.b.copy_(b[t])
        losses[t] = importance_weighted_loss(model(x), x, importance)
    return losses


def _stack_snapshots(snapshots: list[dict[str, Tensor]]) -> tuple[Tensor, Tensor]:
    return torch.stack([s["W"] for s in snapshots]), torch.stack([s["b"] for s in snapshots])


# --- part (a): energy-level jumps -------------------------------------------------------------


@dataclass
class Jump:
    """One detected jump of one feature, with the loss around it."""

    restart: int
    feature: int
    start_step: int
    end_step: int
    d_start: float
    d_end: float
    loss_drop: float  # loss decrease over the jump window plus one more window
    background_drop: float  # loss decrease over an equally long stretch of the stable window
    plateau_range: float  # max − min loss during the stable window (fluctuation level)
    coincides_with_drop: bool


def find_jumps(
    steps: np.ndarray,
    dimensionality: np.ndarray,
    loss: np.ndarray,
    thr: JumpThresholds,
    restart: int = 0,
) -> list[Jump]:
    """Jumps of Dᵢ between levels in one training run, and whether the loss drops with them.

    Paper: p. 26. For each feature, the first snapshot t such that

    * |Dᵢ(t + w) − Dᵢ(t)| > ``thr.jump``, with w = ``thr.jump_window`` steps (a short window);
    * Dᵢ stayed within ``thr.stable_tol`` of Dᵢ(t) for the ``thr.stable_window`` steps before t
      (a jump *from a level*, not the initial organization of the weights).

    The loss drop is L(t) − L(t + 2w): the drop lags the start of a jump, so we measure over the
    jump window and one more. It coincides with the jump when it is positive, at least
    ``thr.drop_factor`` times the background drop over an equally long stretch of the stable
    window (scaled from the whole stable window), and larger than the loss's fluctuation range
    during the stable window.

    Args:
        steps: ``(T,)`` evenly spaced snapshot steps.
        dimensionality: ``(T, feat)`` Dᵢ per snapshot.
        loss: ``(T,)`` loss per snapshot on a fixed batch.
    """
    if len(steps) < 2:
        return []
    spacing = int(steps[1] - steps[0])
    w = max(1, thr.jump_window // spacing)
    s = max(1, thr.stable_window // spacing)
    n_time = len(steps)
    jumps: list[Jump] = []
    if n_time <= s + w:
        return jumps
    change = np.abs(dimensionality[w:] - dimensionality[:-w])  # (T − w, feat), change from t
    for f in range(dimensionality.shape[1]):
        candidates = np.flatnonzero(change[:, f] > thr.jump)
        for t in candidates:
            if t < s:
                continue
            window = dimensionality[t - s : t + 1, f]
            if np.abs(window - dimensionality[t, f]).max() > thr.stable_tol:
                continue
            end = min(t + 2 * w, n_time - 1)
            drop = float(loss[t] - loss[end])
            stable = loss[t - s : t + 1]
            background = float(loss[t - s] - loss[t]) * (end - t) / s
            plateau_range = float(stable.max() - stable.min())
            coincides = (
                drop > 0 and drop >= thr.drop_factor * max(background, 0.0) and drop > plateau_range
            )
            jumps.append(
                Jump(
                    restart=restart,
                    feature=f,
                    start_step=int(steps[t]),
                    end_step=int(steps[t + w]),
                    d_start=float(dimensionality[t, f]),
                    d_end=float(dimensionality[t + w, f]),
                    loss_drop=drop,
                    background_drop=background,
                    plateau_range=plateau_range,
                    coincides_with_drop=coincides,
                )
            )
            break
    return jumps


def level_census(dimensionality: np.ndarray, tol: float) -> dict[str, int]:
    """How many features end at Dᵢ ≈ 1/2 (digons), ≈ 0 (not learned), or elsewhere."""
    half = np.abs(dimensionality - 0.5) <= tol
    zero = dimensionality <= tol
    return {
        "at_1/2": int(half.sum()),
        "at_0": int(zero.sum()),
        "other": int((~half & ~zero).sum()),
    }


def run_energy_levels(
    config: LearningDynamicsConfig, device: torch.device, logger: logging.Logger
) -> dict[str, Any]:
    """Train part (a) and return its trajectories (all tensors on CPU)."""
    setup = config.energy
    n, m, k = setup.n_features, setup.n_hidden, setup.n_restarts
    p = 1.0 / setup.inverse_density
    importance = geometric_importance(n, setup.importance_decay)
    model = TiedModel.from_kind("relu_output", n, m, k, seed=config.seed).to(device)
    dist = IndependentSparse(p, k, n)
    logger.info(
        f"(a) energy levels: n={n}, m={m}, 1/(1−S)={setup.inverse_density:g}, "
        f"{k} restarts × {config.train.steps} steps"
    )
    result = train_from_config(model, dist, importance, config.train, config.seed, progress=True)
    logger.info(f"    trained in {result.wall_time_s:.1f} s")
    W, b = _stack_snapshots(result.snapshots)
    n_time = W.shape[0]
    dims = metrics.feature_dimensionality(W.reshape(n_time * k, m, n)).reshape(n_time, k, n)
    eval_gen = make_generator(derive_seed(config.seed, EVAL_STREAM, 1))
    x = dist.sample(setup.loss_eval_samples, eval_gen)
    loss = snapshot_losses(W, b, x, importance)
    return {
        "steps": np.asarray(result.snapshot_steps),
        "dimensionality": dims.numpy(),
        "loss": loss.numpy(),
        "W_final": W[-1],
        "b_final": b[-1],
        "final_loss": result.final_loss,
        "loss_history": result.loss_history,
        "log_steps": result.log_steps,
        "inverse_density": setup.inverse_density,
        "wall_time_s": result.wall_time_s,
    }


def evaluate_energy_levels(
    config: LearningDynamicsConfig, energy: dict[str, Any]
) -> tuple[list[Criterion], list[Jump]]:
    """E05(a) criteria (SPEC §7; D-402)."""
    thr = config.jumps
    steps, dims, loss = energy["steps"], energy["dimensionality"], energy["loss"]
    n_restarts = dims.shape[1]
    jumps = [
        j
        for r in range(n_restarts)
        for j in find_jumps(steps, dims[:, r], loss[:, r], thr, restart=r)
    ]
    coinciding = [j for j in jumps if j.coincides_with_drop]
    best = max(coinciding, key=lambda j: j.loss_drop, default=None)
    per_restart = {
        f"restart {r}": {
            "jumps": sum(j.restart == r for j in jumps),
            "with_loss_drop": sum(j.restart == r for j in coinciding),
        }
        for r in range(n_restarts)
    }
    criteria = [
        check(
            "energy_jump_with_loss_drop",
            f"(a) At least one jump of a feature's Dᵢ by > {thr.jump} within {thr.jump_window:,} "
            f"steps, from a level held for {thr.stable_window:,} steps, coinciding with a drop "
            "in the loss curve",
            PAGE_A,
            bool(coinciding),
            {
                "per_restart": per_restart,
                "largest": None
                if best is None
                else {
                    "restart": best.restart,
                    "feature": best.feature,
                    "steps": f"{best.start_step}–{best.end_step}",
                    "D": f"{best.d_start:.3f} → {best.d_end:.3f}",
                    "loss_drop": round(best.loss_drop, 5),
                    "background_drop": round(best.background_drop, 5),
                    "plateau_range": round(best.plateau_range, 5),
                },
            },
            {
                "jump": f"|ΔDᵢ| > {thr.jump} within {thr.jump_window} steps",
                "loss_drop": f"> {thr.drop_factor}× background and > plateau range",
            },
            group="a_energy_levels",
        )
    ]
    W = energy["W_final"]
    final_dims = dims[-1]
    criteria.append(
        info(
            "energy_final_state",
            "(a) Final state per restart: D* (plateau value 1/2), Σᵢ Dᵢ (vs m) and how many "
            "features end at Dᵢ ≈ 1/2 or ≈ 0",
            "pp. 17, 26",
            {
                f"restart {r}": {
                    "D_star": round(float(metrics.dims_per_feature(W[r : r + 1])[0]), 4),
                    "sum_D": round(float(final_dims[r].sum()), 3),
                    **level_census(final_dims[r], thr.level_tol),
                    "loss": round(float(energy["final_loss"][r]), 4),
                }
                for r in range(n_restarts)
            },
            group="a_energy_levels",
        )
    )
    criteria.append(
        info(
            "energy_jumps",
            "(a) Every detected jump (feature, steps, Dᵢ before → after, loss drop over the jump "
            "and one more window, background drop, plateau range)",
            PAGE_A,
            [
                {
                    "restart": j.restart,
                    "feature": j.feature,
                    "steps": f"{j.start_step}–{j.end_step}",
                    "D": f"{j.d_start:.2f} → {j.d_end:.2f}",
                    "loss_drop": round(j.loss_drop, 4),
                    "background": round(j.background_drop, 4),
                    "plateau_range": round(j.plateau_range, 4),
                    "coincides": j.coincides_with_drop,
                }
                for j in jumps
            ],
            group="a_energy_levels",
        )
    )
    return criteria, jumps


# --- part (b): geometric transformations -------------------------------------------------------


def set_index(n_sets: int, set_size: int) -> np.ndarray:
    """Correlated-set index of each feature (set-major layout, as in ``CorrelatedSets``)."""
    return np.repeat(np.arange(n_sets), set_size)


def stage_codes(
    W: Float[Tensor, "batch hidden feat"], sets: np.ndarray, thr: StageThresholds
) -> np.ndarray:
    """Geometric stage (an index into :data:`STAGES`) of each weight matrix.

    Paper: p. 27. Rotation-invariant tests, applied in this order:

    * ``small``: mean ‖Wᵢ‖ < ``small_norm``, weights still near the origin (stage A);
    * ``octahedron``: every feature represented, each with a partner in the *other* set at
      cos < ``antipodal_cos``, and all within-set |cos| < ``orthogonal_cos`` (stage D: three
      orthogonal antipodal pairs joining the two sets);
    * ``collapsed``: in every set the mean within-set cos ≥ ``collapsed_cos``: each set moves
      as one direction, the linear (PCA-like) solution of stage B;
    * ``prism``: the sets have expanded and their centroids are opposed (cos ≤
      ``opposed_cos``) but their members are not yet antipodal: two triangles on opposite
      sides that have not rotated into the antiprism (stage C);
    * ``transition``: anything else (sets expanding, centroids not yet opposed).

    Only two sets are supported (the paper's setup).
    """
    labels = np.unique(sets)
    if len(labels) != 2:
        raise ValueError(f"stage_codes supports exactly two sets, got {len(labels)}")
    norms = metrics.feature_norms(W)
    cos = metrics.cosine_similarity_matrix(W)
    set_t = torch.as_tensor(sets, device=W.device)
    same = set_t[:, None] == set_t[None, :]
    off_diag = ~torch.eye(len(sets), dtype=torch.bool, device=W.device)
    within_pairs = same & off_diag
    within_max_abs = cos.abs().masked_fill(~within_pairs, 0.0).amax(dim=(-2, -1))
    partner = cos.masked_fill(same, float("inf")).amin(dim=-1)  # (batch, feat)
    collapsed = torch.ones(W.shape[0], dtype=torch.bool, device=W.device)
    centroids = []
    for label in labels:
        members = torch.as_tensor(np.flatnonzero(sets == label), device=W.device)
        block = cos[:, members][:, :, members]
        k = len(members)
        within_mean = (block.sum(dim=(-2, -1)) - block.diagonal(dim1=-2, dim2=-1).sum(-1)) / (
            k * (k - 1)
        )
        collapsed &= within_mean >= thr.collapsed_cos
        centroids.append(W[:, :, members].mean(dim=-1))
    centroid_cos = torch.nn.functional.cosine_similarity(centroids[0], centroids[1], dim=-1)
    small = norms.mean(dim=-1) < thr.small_norm
    octahedron = (
        (norms > thr.represented_norm).all(dim=-1)
        & (partner < thr.antipodal_cos).all(dim=-1)
        & (within_max_abs < thr.orthogonal_cos)
    )
    prism = centroid_cos <= thr.opposed_cos
    codes = torch.full((W.shape[0],), TRANSITION, dtype=torch.long)
    codes[prism.cpu()] = PRISM
    codes[collapsed.cpu()] = COLLAPSED
    codes[octahedron.cpu()] = OCTAHEDRON
    codes[small.cpu()] = SMALL
    return codes.numpy()


def trajectory_type(codes: np.ndarray, spacing: int, min_steps: int) -> list[int]:
    """The stages one run passes through, leaving out stages shorter than ``min_steps``.

    Consecutive snapshots with the same stage form a segment; segments lasting at least
    ``min_steps`` steps are kept (the final segment is always kept), and repeated stages are
    merged. Returns stage codes in order.
    """
    segments: list[tuple[int, int]] = []  # (code, length in snapshots)
    for code in codes.tolist():
        if segments and segments[-1][0] == code:
            segments[-1] = (code, segments[-1][1] + 1)
        else:
            segments.append((code, 1))
    kept = [
        code
        for i, (code, length) in enumerate(segments)
        if length * spacing >= min_steps or i == len(segments) - 1
    ]
    merged: list[int] = []
    for code in kept:
        if not merged or merged[-1] != code:
            merged.append(code)
    return merged


def type_name(codes: list[int]) -> str:
    return " → ".join(STAGES[c] for c in codes)


def canonical_frame_3d(W_final: np.ndarray, sets: np.ndarray) -> np.ndarray:
    """A rotation that puts the final geometry in a readable orientation (display only).

    z points from set B's centroid to set A's (the axis along which the sets split, p. 27), and
    x is the first feature's direction projected off z. The hidden space has no privileged
    basis (p. 31), so any fixed rotation is equally valid; applying one rotation to every
    snapshot keeps the trajectories comparable.
    """
    a = W_final[:, sets == sets.min()].mean(axis=1)
    b = W_final[:, sets != sets.min()].mean(axis=1)
    z = a - b
    if np.linalg.norm(z) < 1e-6:
        z = np.linalg.svd(W_final)[0][:, 0]
    z = z / np.linalg.norm(z)
    x = W_final[:, 0] - (W_final[:, 0] @ z) * z
    if np.linalg.norm(x) < 1e-6:
        x = np.eye(3)[np.argmin(np.abs(z))]
        x = x - (x @ z) * z
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    return np.stack([x, y, z])  # rows: new axes; apply as R @ W


def run_transformations(
    config: LearningDynamicsConfig, device: torch.device, logger: logging.Logger
) -> dict[str, Any]:
    """Train part (b): every (init scale, set probability, restart) as one instance."""
    setup = config.transformations
    n = setup.n_sets * setup.set_size
    scales = setup.init_scales
    probs = setup.set_probabilities
    r = setup.n_restarts
    n_inst = len(scales) * len(probs) * r
    scale_idx = torch.arange(len(scales)).repeat_interleave(len(probs) * r)
    prob_idx = torch.arange(len(probs)).repeat_interleave(r).repeat(len(scales))
    restart = torch.arange(r).repeat(len(scales) * len(probs))
    p = torch.tensor(probs, dtype=torch.float32)[prob_idx]
    model = TiedModel.from_kind("relu_output", n, setup.n_hidden, n_inst, seed=config.seed)
    with torch.no_grad():
        model.W.mul_(torch.tensor(scales, dtype=torch.float32)[scale_idx][:, None, None])
    model = model.to(device)
    dist = CorrelatedSets(setup.n_sets, setup.set_size, p, n_inst)
    importance = geometric_importance(n, setup.importance_decay)
    logger.info(
        f"(b) transformations: {setup.n_sets} sets × {setup.set_size} in m={setup.n_hidden}; "
        f"init scales {scales} × set probabilities {probs} × {r} restarts, "
        f"{setup.train.steps} steps"
    )
    result = train_from_config(model, dist, importance, setup.train, config.seed, progress=True)
    logger.info(f"    trained in {result.wall_time_s:.1f} s")
    W, b = _stack_snapshots(result.snapshots)
    n_time = W.shape[0]
    sets = set_index(setup.n_sets, setup.set_size)
    codes = stage_codes(W.reshape(n_time * n_inst, setup.n_hidden, n), sets, config.stages)
    eval_gen = make_generator(derive_seed(config.seed, EVAL_STREAM, 1))
    x = dist.sample(setup.loss_eval_samples, eval_gen)
    loss = snapshot_losses(W, b, x, importance)
    return {
        "steps": np.asarray(result.snapshot_steps),
        "W": W,
        "b": b,
        "stages": codes.reshape(n_time, n_inst).T,  # (inst, time)
        "loss": loss.numpy().T,  # (inst, time)
        "final_loss": result.final_loss,
        "init_scale": torch.tensor(scales)[scale_idx],
        "set_probability": p,
        "scale_index": scale_idx,
        "prob_index": prob_idx,
        "restart": restart,
        "sets": sets,
        "wall_time_s": result.wall_time_s,
    }


def evaluate_transformations(
    config: LearningDynamicsConfig, trans: dict[str, Any]
) -> tuple[list[Criterion], pd.DataFrame]:
    """E05(b) criteria (SPEC §7; D-404) and a per-run table."""
    setup, thr = config.transformations, config.stages
    stages, steps = trans["stages"], trans["steps"]
    spacing = int(steps[1] - steps[0])
    rows = []
    for k in range(stages.shape[0]):
        codes = stages[k]
        kind = trajectory_type(codes, spacing, thr.min_stage_steps)
        reached = np.flatnonzero(codes == OCTAHEDRON)
        rows.append(
            {
                "instance": k,
                # exact config values (the tensors hold float32 copies)
                "init_scale": float(setup.init_scales[int(trans["scale_index"][k])]),
                "set_probability": float(setup.set_probabilities[int(trans["prob_index"][k])]),
                "restart": int(trans["restart"][k]),
                "final_stage": STAGES[int(codes[-1])],
                "octahedron": bool(codes[-1] == OCTAHEDRON),
                "first_octahedron_step": int(steps[reached[0]]) if len(reached) else -1,
                "prism_steps": int((codes == PRISM).sum() * spacing),
                "trajectory_type": type_name(kind),
                "final_loss": float(trans["final_loss"][k]),
            }
        )
    groups: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["init_scale"], row["set_probability"]), []).append(row)
    primary_scale = float(setup.init_scales[0])
    per_density = {
        f"1-S={p:g}": f"{sum(r['octahedron'] for r in g)}/{len(g)}"
        for (scale, p), g in groups.items()
        if scale == primary_scale
    }
    any_octahedron = any(r["octahedron"] for r in rows if r["init_scale"] == primary_scale)
    criteria = [
        check(
            "octahedron_final_state",
            "(b) At least one restart ends in the octahedron: three orthogonal antipodal pairs, "
            f"each joining features of different correlated sets (init scale "
            f"{primary_scale:g}× Xavier)",
            PAGE_B,
            any_octahedron,
            {"octahedra_per_set_density": per_density},
            "≥ 1 restart",
            group="b_transformations",
        )
    ]
    types = {
        f"init {scale:g}× · 1-S={p:g}": dict(Counter(r["trajectory_type"] for r in g).most_common())
        for (scale, p), g in groups.items()
    }
    criteria.append(
        info(
            "trajectory_types",
            "(b) Trajectory types: the stages each restart passes through (stages shorter than "
            f"{thr.min_stage_steps:,} steps left out), counted per init scale and set density",
            PAGE_B,
            types,
            group="b_transformations",
        )
    )
    timing = {}
    for (scale, p), g in groups.items():
        reached = [r["first_octahedron_step"] for r in g if r["first_octahedron_step"] >= 0]
        timing[f"init {scale:g}× · 1-S={p:g}"] = {
            "median_first_octahedron_step": int(np.median(reached)) if reached else None,
            "restarts_with_prism_stage": sum(r["prism_steps"] >= thr.min_stage_steps for r in g),
            "median_prism_steps": int(np.median([r["prism_steps"] for r in g])),
        }
    table = pd.DataFrame(rows)
    criteria.append(
        info(
            "trajectory_timing",
            "(b) When the octahedron is first reached, and how long runs sit in the aligned "
            "'prism' stage (C) before rotating into the antiprism (D)",
            PAGE_B,
            timing,
            group="b_transformations",
        )
    )
    return criteria, table


# --- run --------------------------------------------------------------------------------------


def run(
    config: LearningDynamicsConfig, run_dir: RunDir, device: torch.device, logger: logging.Logger
) -> list[Criterion]:
    energy = run_energy_levels(config, device, logger)
    criteria, jumps = evaluate_energy_levels(config, energy)
    run_dir.save_checkpoint("energy_levels", energy)
    run_dir.write_table(pd.DataFrame([asdict(j) for j in jumps]), "jumps")

    trans = run_transformations(config, device, logger)
    trans_criteria, table = evaluate_transformations(config, trans)
    criteria += trans_criteria
    run_dir.save_checkpoint("transformations", trans)
    run_dir.write_table(table, "metrics")
    return criteria


# --- figures ------------------------------------------------------------------------------------


def representative_restart(table: pd.DataFrame, init_scale: float) -> int:
    """The run drawn in the p. 27-style figure (selection rule, not a cherry-pick of numbers).

    The first restart, at the lowest set density, whose trajectory passes through the prism
    stage and ends as an octahedron (the paper's A → B → C → D); failing that, the first run
    ending as an octahedron; failing that, instance 0.
    """
    runs = sorted(
        (r for r in table.to_dict("records") if r["init_scale"] == init_scale),
        key=lambda r: (r["set_probability"], r["restart"]),
    )
    paper_like = [r for r in runs if r["octahedron"] and "prism" in r["trajectory_type"]]
    ending = [r for r in runs if r["octahedron"]]
    for candidates in (paper_like, ending):
        if candidates:
            return int(candidates[0]["instance"])
    return 0


def _stage_moments(codes: np.ndarray) -> list[int]:
    """Snapshot indices to draw, like the paper's A–D: the start, the middle of the collapsed
    stage, the middle of the prism stage (or of the transition if there is none), the end."""
    moments = [0]
    middle_stage = PRISM if (codes == PRISM).any() else TRANSITION
    for stage in (COLLAPSED, middle_stage):
        where = np.flatnonzero(codes == stage)
        if len(where):
            moments.append(int(where[len(where) // 2]))
    moments.append(len(codes) - 1)
    return sorted(set(moments))


def figures(run_dir: RunDir) -> list[Path]:
    """p. 26 energy-level figures, p. 27 trajectory figure, stage raster, 3D HTML."""
    paths: list[Path] = []
    energy = run_dir.load_checkpoint("energy_levels")
    jumps = run_dir.read_table("jumps").to_dict("records")
    steps, dims, loss = energy["steps"], energy["dimensionality"], energy["loss"]
    coinciding = [j for j in jumps if j["coincides_with_drop"]]
    best = int(max(coinciding, key=lambda j: j["loss_drop"])["restart"]) if coinciding else 0
    for r in range(dims.shape[1]):
        marks = [
            (int(j["feature"]), int(j["start_step"]), int(j["end_step"]))
            for j in coinciding
            if j["restart"] == r
        ]
        fig = dimensionality_over_training(
            steps,
            dims[:, r],
            loss[:, r],
            marks,
            title=f"E05(a) · n = {dims.shape[2]}, m = {energy['W_final'].shape[1]}, "
            f"1/(1−S) = {energy['inverse_density']:g} · restart {r}",
        )
        name = "energy_levels" if r == best else f"energy_levels_restart{r}"
        paths += save_figure(fig, run_dir.figures_dir / name)

    trans = run_dir.load_checkpoint("transformations")
    table = run_dir.read_table("metrics")
    records = table.to_dict("records")
    stages, tsteps = trans["stages"], trans["steps"]
    scales = sorted({float(r["init_scale"]) for r in records})
    for scale in scales:
        rows = sorted(
            (r for r in records if r["init_scale"] == scale),
            key=lambda r: (r["set_probability"], r["restart"]),
        )
        index = np.array([int(r["instance"]) for r in rows])
        labels = [f"1−S={r['set_probability']:g} · r{int(r['restart'])}" for r in rows]
        fig = stage_raster(
            tsteps,
            stages[index],
            labels,
            STAGES,
            title=f"E05(b) · geometric stage over training (init {scale:g}× Xavier)",
        )
        suffix = "" if scale == scales[0] or len(scales) == 1 else f"_init{scale:g}"
        paths += save_figure(fig, run_dir.figures_dir / f"stage_raster{suffix}")
    primary = float(records[0]["init_scale"])
    k = representative_restart(table, primary)
    sets = np.asarray(trans["sets"])
    rotation = canonical_frame_3d(trans["W"][-1, k].numpy(), sets)
    snaps = np.einsum("ij,tjf->tif", rotation, trans["W"][:, k].numpy())
    codes = stages[k]
    row = next(r for r in records if r["instance"] == k)
    fig = trajectory_panels(
        snaps,
        tsteps,
        trans["loss"][k],
        codes,
        STAGES,
        sets,
        _stage_moments(codes),
        title=f"E05(b) · 1−S = {row['set_probability']:g}, restart {int(row['restart'])}: "
        f"{row['trajectory_type']}",
    )
    paths += save_figure(fig, run_dir.figures_dir / "trajectories")
    html = run_dir.figures_dir / "trajectories_3d.html"
    trajectories_3d(snaps, tsteps, sets, title=f"E05(b) · {row['trajectory_type']}").write_html(
        html, include_plotlyjs="cdn"
    )
    paths.append(html)
    return paths


EXPERIMENT = Experiment(
    name="learning_dynamics",
    code="E05",
    title="Energy-level jumps (n = 400, m = 30) and 3D geometric transformations (n = 6, m = 3)",
    paper_pages="pp. 26–28",
    config_cls=LearningDynamicsConfig,
    run=run,
    figures=figures,
)
