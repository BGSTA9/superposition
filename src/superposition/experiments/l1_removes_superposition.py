"""E11 l1_removes_superposition: an L1 penalty on h trades loss for less superposition.

Paper: Approach 1: Creating Models Without Superposition, p. 42: "It's actually quite easy to
get rid of superposition in the toy models described in this paper, albeit at the cost of a
higher loss. Simply apply an L1 regularization term to the hidden layer activations (i.e. add
λ‖h‖₁ to the loss). This actually has a nice interpretation in terms of killing features below
a certain importance threshold, especially if they're not basis aligned."

We add λ E‖h‖₁ to the training loss of the ReLU output model (``train(..., l1_hidden=λ)``,
one λ per instance) and sweep λ on two setups in superposition at λ = 0:

* the E00 pentagon (n = 5, m = 2, Iᵢ = 0.9ⁱ, 1 − S = 0.1; p. 2);
* the E01 model at a sparse setting (n = 20, m = 5, Iᵢ = 0.7ⁱ, 1 − S = 0.01; p. 11).

For each λ we keep the restart with the lowest *training objective* (reconstruction loss +
λ E‖h‖₁, since that is what training minimizes) and measure its reconstruction loss (without
the penalty; losses at one sparsity are comparable, footnote 17) and its superposition:
represented features (‖Wᵢ‖ > 0.5), superposed ones (superposition score > 0.05), the formal
test (WᵀW not invertible on the represented features, p. 8) and D*. Criteria: D-505; a
superposition-free model with a single feature left passes "falls to none" only as PARTIAL,
since one feature cannot be in superposition with anything (D-R05).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

import pandas as pd
import torch
import yaml
from torch import Tensor

from superposition import metrics
from superposition.acceptance import FAIL, PARTIAL, PASS, Criterion, check, graded, info
from superposition.config import ExperimentConfig, from_dict
from superposition.data import IndependentSparse, geometric_importance
from superposition.experiments.base import Experiment
from superposition.models import TiedModel
from superposition.plotting.arrows import arrows_grid
from superposition.plotting.l1 import tradeoff_curve_figure, tradeoff_figure
from superposition.plotting.style import save_figure
from superposition.plotting.weights import WeightsPanel, weights_grid
from superposition.runs import RunDir
from superposition.train import select_best, train
from superposition.utils import EVAL_STREAM, derive_seed, make_generator

PAGE = "p. 42"
L1_EVAL_KEY = 11  # derive_seed key for E11's penalty evaluation


@dataclass
class L1Setup:
    """One toy model swept over λ."""

    label: str = "pentagon"
    source: str = "E00, p. 2"
    n_features: int = 5
    n_hidden: int = 2
    importance_decay: float = 0.9
    feature_probability: float = 0.1
    n_restarts: int = 8
    figure_lambdas: list[float] = field(default_factory=lambda: [0.0, 0.1, 0.3, 0.55, 0.65])


@dataclass
class L1Thresholds:
    """Acceptance thresholds (SPEC §7 E11; D-505)."""

    represented_norm: float = 0.5
    superposition_score: float = 0.05  # a represented feature with a larger score is superposed
    count_tolerance: int = 1  # superposed count may rise by this much between λ values
    loss_tolerance: float = 0.02  # a relative loss decrease up to this counts as flat (noise)
    eval_samples: int = 200_000  # shared samples for the loss and E‖h‖₁ of every instance
    # "falls to none" passes only if a superposition-free model keeps at least this many
    # features; with fewer it is PARTIAL (a lone feature cannot be in superposition; D-R05)
    nontrivial_represented: int = 2


@dataclass
class L1Config(ExperimentConfig):
    """E11 settings: λ grid (0 first, increasing) and the setups."""

    lambdas: list[float] = field(
        default_factory=lambda: [0.0, 0.003, 0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5,
                                 0.55, 0.6, 0.65, 0.7, 0.8]
    )  # fmt: skip
    setups: list[L1Setup] = field(default_factory=lambda: [L1Setup()])
    thresholds: L1Thresholds = field(default_factory=L1Thresholds)


@dataclass
class L1Sweep:
    """One setup's trained sweep (CPU tensors). Instance ``c · restarts + r`` is restart r of
    λ index c."""

    label: str
    lambdas: list[float]
    W: Tensor  # (inst, hidden, feat)
    b: Tensor  # (inst, feat)
    final_loss: Tensor  # (inst,) reconstruction loss, no penalty (shared samples)
    penalty: Tensor  # (inst,) E‖h‖₁
    objective: Tensor  # (inst,) loss + λ penalty
    condition: Tensor  # (inst,)
    best: Tensor  # (cond,) lowest objective per λ
    feature_probability: float
    wall_time_s: float


@torch.no_grad()
def shared_evaluation(
    W: Tensor,
    b: Tensor,
    importance: Tensor,
    feature_probability: float,
    n_samples: int,
    generator: torch.Generator,
    chunk: int = 8192,
) -> tuple[Tensor, Tensor]:
    """Reconstruction loss and E‖h‖₁ of every instance on the *same* inputs.

    Common random numbers: all λ values and restarts are scored on one set of samples, so
    their differences carry no sampling noise from the inputs (D-505). Returns
    ``(loss, penalty)``, each ``(inst,)``, for the ReLU output model x′ = ReLU(WᵀWx + b).
    """
    dist = IndependentSparse(feature_probability, 1, W.shape[-1])
    loss = torch.zeros(W.shape[0])
    penalty = torch.zeros(W.shape[0])
    done = 0
    while done < n_samples:
        size = min(chunk, n_samples - done)
        x = dist.sample(size, generator)[:, 0]
        h = torch.einsum("ihf,bf->bih", W, x)
        out = torch.relu(torch.einsum("ihf,bih->bif", W, h) + b)
        loss += (importance * (x[:, None, :] - out) ** 2).sum(dim=-1).sum(dim=0)
        penalty += h.abs().sum(dim=-1).sum(dim=0)
        done += size
    return loss / n_samples, penalty / n_samples


def train_l1_sweep(setup: L1Setup, config: L1Config, device: torch.device) -> L1Sweep:
    """Train every λ × restart of one setup as one multi-instance ReLU output model."""
    lambdas = config.lambdas
    condition = torch.arange(len(lambdas)).repeat_interleave(setup.n_restarts)
    lam = torch.tensor(lambdas, dtype=torch.float32)[condition]
    n_inst = len(condition)
    model = TiedModel.from_kind(
        "relu_output", setup.n_features, setup.n_hidden, n_inst, seed=config.seed
    ).to(device)
    dist = IndependentSparse(setup.feature_probability, n_inst, setup.n_features)
    importance = geometric_importance(setup.n_features, setup.importance_decay)
    train_config = config.train
    # train() takes one λ per instance; TrainConfig.l1_hidden (a scalar) is not used here.
    result = train(
        model,
        dist,
        importance,
        steps=train_config.steps,
        batch_size=train_config.batch_size,
        lr=train_config.lr,
        schedule=train_config.schedule,
        l1_hidden=lam.to(device),
        log_every=train_config.log_every,
        eval_samples=train_config.eval_samples,
        warmup_frac=train_config.warmup_frac,
        seed=config.seed,
        progress=True,
    )
    W, b = model.W.detach().cpu(), model.b.detach().cpu()
    generator = make_generator(derive_seed(config.seed, EVAL_STREAM, L1_EVAL_KEY))
    loss, penalty = shared_evaluation(
        W, b, importance, setup.feature_probability, config.thresholds.eval_samples, generator
    )
    objective = loss + lam * penalty
    return L1Sweep(
        label=setup.label,
        lambdas=list(lambdas),
        W=W,
        b=b,
        final_loss=loss,
        penalty=penalty,
        objective=objective,
        condition=condition,
        best=select_best(objective, condition),
        feature_probability=setup.feature_probability,
        wall_time_s=result.wall_time_s,
    )


def sweep_table(sweep: L1Sweep, thr: L1Thresholds) -> pd.DataFrame:
    """One row per λ (best restart): losses and superposition measures."""
    rows = []
    for c, lam in enumerate(sweep.lambdas):
        k = int(sweep.best[c])
        W = sweep.W[k][None]
        represented = metrics.represented_mask(W, thr.represented_norm)[0]
        score = metrics.superposition_score(W)[0]
        superposed = represented & (score > thr.superposition_score)
        rows.append(
            {
                "setup": sweep.label,
                "lambda": lam,
                "instance": k,
                "loss": float(sweep.final_loss[k]),
                "penalty": float(sweep.penalty[k]),
                "objective": float(sweep.objective[k]),
                "represented": int(represented.sum()),
                "superposed": int(superposed.sum()),
                "formal_superposition": bool(metrics.is_superposition(W, thr.represented_norm)[0]),
                "max_score": float(score[represented].max()) if bool(represented.any()) else 0.0,
                # D* = m/‖W‖²_F is undefined (∞) once no feature is left.
                "dims_per_feature": float(metrics.dims_per_feature(W)[0])
                if bool(represented.any())
                else float("nan"),
                "represented_features": torch.nonzero(represented).flatten().tolist(),
            }
        )
    return pd.DataFrame(rows)


def superposition_free(row: dict) -> bool:
    """At least one feature represented, none superposed, and WᵀW invertible on them."""
    return (
        int(row["represented"]) >= 1
        and int(row["superposed"]) == 0
        and not bool(row["formal_superposition"])
    )


def evaluate_setup(setup: L1Setup, table: pd.DataFrame, thr: L1Thresholds) -> list[Criterion]:
    """The four E11 criteria for one setup (D-505)."""
    rows = [dict(r) for r in table.to_dict("records")]
    zero = rows[0]
    m = setup.n_hidden
    superposed = [int(r["superposed"]) for r in rows]
    rises = [b - a for a, b in pairwise(superposed)]
    free = [r for r in rows if float(r["lambda"]) > 0 and superposition_free(r)]
    first_free = free[0] if free else None
    several = [r for r in free if int(r["represented"]) >= thr.nontrivial_represented]
    falls_status = PASS if several else PARTIAL if free else FAIL
    losses = [float(r["loss"]) for r in rows]
    drops = [
        (float(rows[i]["lambda"]), (losses[i - 1] - losses[i]) / losses[i - 1])
        for i in range(1, len(rows))
        if losses[i] < losses[i - 1]
    ]
    worst_drop = max((d for _, d in drops), default=0.0)
    up_to = [
        r for r in rows if first_free is None or float(r["lambda"]) <= float(first_free["lambda"])
    ]
    loss_ok = (
        worst_drop <= thr.loss_tolerance
        and first_free is not None
        and float(first_free["loss"]) > float(zero["loss"])
    )
    g = setup.label
    desc = f"{setup.label} ({setup.source}, n = {setup.n_features}, m = {m}, 1 − S = {setup.feature_probability:g})"
    return [
        check(
            f"{g}_superposition_at_zero",
            f"{desc}: at λ = 0 the model is in superposition (more than m features represented; "
            "WᵀW not invertible on them)",
            PAGE,
            int(zero["represented"]) > m and bool(zero["formal_superposition"]),
            {"represented": int(zero["represented"]), "superposed": int(zero["superposed"])},
            f"> {m} represented, formal test true",
            group=g,
        ),
        check(
            f"{g}_superposition_declines",
            f"{desc}: the number of superposed features never increases with λ by more than "
            f"{thr.count_tolerance}",
            PAGE,
            max(rises, default=0) <= thr.count_tolerance,
            {f"λ={r['lambda']:g}": int(r["superposed"]) for r in rows},
            f"rises ≤ {thr.count_tolerance}",
            group=g,
        ),
        graded(
            f"{g}_superposition_falls_to_none",
            f"{desc}: some λ gives a model with no superposition (no superposed feature, WᵀW "
            f"invertible on the represented ones) that keeps at least "
            f"{thr.nontrivial_represented} features; PARTIAL if it keeps only one (D-R05)",
            PAGE,
            falls_status,
            None
            if first_free is None
            else {
                "first_lambda": float(first_free["lambda"]),
                "represented": int(first_free["represented"]),
                "features": list(first_free["represented_features"]),
                "free_lambdas": [float(r["lambda"]) for r in free],
                "free_lambdas_with_several_features": [float(r["lambda"]) for r in several],
            },
            f"exists with ≥ {thr.nontrivial_represented} features (PASS); only with fewer "
            "(PARTIAL)",
            group=g,
        ),
        check(
            f"{g}_loss_rises",
            f"{desc}: the reconstruction loss rises with λ (no drop larger than "
            f"{thr.loss_tolerance:.0%} between consecutive λ), and is higher once superposition "
            "is gone than at λ = 0",
            PAGE,
            loss_ok,
            {
                "loss_at_zero": round(float(zero["loss"]), 6),
                "loss_when_free": None
                if first_free is None
                else round(float(first_free["loss"]), 6),
                "largest_relative_drop": round(worst_drop, 4),
                "drops": [(lam, round(d, 4)) for lam, d in drops],
                "loss_up_to_first_free": [round(float(r["loss"]), 6) for r in up_to],
            },
            f"drops ≤ {thr.loss_tolerance}; loss(free) > loss(0)",
            group=g,
        ),
    ]


def run(
    config: L1Config, run_dir: RunDir, device: torch.device, logger: logging.Logger
) -> list[Criterion]:
    tables, criteria = [], []
    for setup in config.setups:
        logger.info(
            f"{setup.label}: {len(config.lambdas)} values of λ × {setup.n_restarts} restarts"
        )
        sweep = train_l1_sweep(setup, config, device)
        logger.info(f"    trained in {sweep.wall_time_s:.1f} s")
        run_dir.save_checkpoint(setup.label, vars(sweep))
        table = sweep_table(sweep, config.thresholds)
        tables.append(table)
        for r in table.to_dict("records"):
            logger.info(
                f"    λ={r['lambda']:<6g} loss {r['loss']:.5f}  represented {r['represented']:>2}  "
                f"superposed {r['superposed']:>2}  D* {r['dims_per_feature']:.3f}"
            )
        criteria += evaluate_setup(setup, table, config.thresholds)
    full = pd.concat(tables, ignore_index=True)
    run_dir.write_table(full, "metrics")
    criteria.append(
        info(
            "tradeoff",
            "The trade-off per setup: λ → reconstruction loss (no penalty), features represented "
            "and superposed, D* (best restart by training objective)",
            PAGE,
            {
                s.label: [
                    {
                        "lambda": float(r["lambda"]),
                        "loss": round(float(r["loss"]), 6),
                        "represented": int(r["represented"]),
                        "superposed": int(r["superposed"]),
                        "dims_per_feature": round(float(r["dims_per_feature"]), 3),
                    }
                    for r in full.to_dict("records")
                    if r["setup"] == s.label
                ]
                for s in config.setups
            },
        )
    )
    return criteria


# --- figures -------------------------------------------------------------------------------------


def figures(run_dir: RunDir) -> list[Path]:
    """Loss and superposition against λ, the trade-off curve, and weights at chosen λ."""
    config = from_dict(L1Config, yaml.safe_load(run_dir.config_path.read_text()))
    table = run_dir.read_table("metrics")
    paths: list[Path] = []
    paths += save_figure(
        tradeoff_figure(table, {s.label: s.n_hidden for s in config.setups},
                        title="E11 · adding λ‖h‖₁ to the loss (best restart per λ)"),
        run_dir.figures_dir / "l1_sweep",
    )  # fmt: skip
    paths += save_figure(
        tradeoff_curve_figure(table, title="E11 · the trade-off: loss against superposition"),
        run_dir.figures_dir / "l1_tradeoff",
    )
    for setup in config.setups:
        sweep = L1Sweep(**run_dir.load_checkpoint(setup.label))
        chosen = [lam for lam in setup.figure_lambdas if lam in sweep.lambdas]
        rows_t = [dict(r) for r in table.to_dict("records") if r["setup"] == setup.label]
        by_lam = {float(r["lambda"]): r for r in rows_t}
        if setup.n_hidden == 2:
            panels = []
            for lam in chosen:
                k = int(sweep.best[sweep.lambdas.index(lam)])
                r = by_lam[lam]
                panels.append(
                    (
                        f"λ = {lam:g}",
                        sweep.W[k].numpy(),
                        f"{r['represented']} represented, {r['superposed']} superposed",
                    )
                )
            fig = arrows_grid(
                [("", panels)],
                suptitle=f"E11 · {setup.label}: W as λ grows (raw hidden basis: the L1 penalty "
                "makes it privileged)",
            )
            paths += save_figure(fig, run_dir.figures_dir / f"{setup.label}_arrows")
        else:
            panels_w = []
            for lam in chosen:
                k = int(sweep.best[sweep.lambdas.index(lam)])
                W = sweep.W[k][None]
                panels_w.append(
                    WeightsPanel(
                        title=f"λ = {lam:g}",
                        gram=metrics.gram(W)[0].numpy(),
                        bias=sweep.b[k].numpy(),
                        norms=metrics.feature_norms(W)[0].numpy(),
                        superposition=metrics.superposition_score(W)[0].numpy(),
                    )
                )
            fig = weights_grid(
                panels_w, suptitle=f"E11 · {setup.label}: WᵀW, bias and ‖Wᵢ‖ as λ grows"
            )
            paths += save_figure(fig, run_dir.figures_dir / f"{setup.label}_weights")
    return paths


EXPERIMENT = Experiment(
    name="l1_removes_superposition",
    code="E11",
    title="An L1 penalty on the hidden layer removes superposition at the cost of loss",
    paper_pages=PAGE,
    config_cls=L1Config,
    run=run,
    figures=figures,
)
