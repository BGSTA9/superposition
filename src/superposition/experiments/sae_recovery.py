"""E10 sae_recovery: do sparse autoencoders recover a toy model's true features?

Paper: Approach 2: Finding an Overcomplete Basis, pp. 42–44 (footnote 20); p. 44: toy models
can be studied because "we do know what the features are!".

We train toy models in superposition with the settings of earlier experiments, freeze them,
and train sparse autoencoders (:mod:`superposition.sae`) on their hidden activations h = Wx,
drawn from the toy model's own input distribution:

* the E00 pentagon (n = 5, m = 2, Iᵢ = 0.9ⁱ, 1 − S = 0.1; p. 2);
* the sparse E01 models (n = 20, m = 5, Iᵢ = 0.7ⁱ, 1 − S ∈ {0.01, 0.001}; p. 11);
* one E03 model (n = 400, m = 30, Iᵢ = 1, 1 − S = 0.1; p. 17), reported without grading.

For each toy model a sweep of SAEs (dictionary size × L1 coefficient × resampling on/off, plus
TopK variants) trains as one multi-instance model. Every SAE's decoder is compared with the
true feature directions Wᵢ of the *represented* features (‖Wᵢ‖ > 0.5).

Grading uses the best SAE of the sweep, chosen by its ground-truth recovery: the question is
whether an SAE in a standard sweep recovers the features at all. Because that selection uses
the ground truth, the run also reports how many SAE settings pass, and what reconstruction
error says (it can be low while recovery fails). Settings and criteria: D-503, D-504.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import Tensor

from superposition import metrics
from superposition.acceptance import Criterion, check, info
from superposition.config import ExperimentConfig, TrainConfig, from_dict
from superposition.data import IndependentSparse, geometric_importance
from superposition.experiments.base import Experiment
from superposition.experiments.common import train_sparsity_grid
from superposition.plotting.sae import (
    dictionary_2d_figure,
    reconstruction_vs_recovery_figure,
    recovery_sweep_figure,
)
from superposition.plotting.style import save_figure
from superposition.runs import RunDir
from superposition.sae import SAETrainConfig, SparseAutoencoder, recovery, sae_stats, train_sae
from superposition.utils import EVAL_STREAM, derive_seed, make_generator

PAGE = "pp. 42–44"
SAE_EVAL_KEY = 10  # derive_seed key for E10's held-out activations


@dataclass
class SweepSpec:
    """The SAEs trained on one toy model: sizes × (L1 values × resampling) + TopK variants."""

    dictionary_sizes: list[int] = field(default_factory=lambda: [5, 10, 20])
    l1_values: list[float] = field(default_factory=lambda: [0.03, 0.1, 0.3, 1.0])
    resample: list[bool] = field(default_factory=lambda: [False, True])
    topk_values: list[int] = field(default_factory=lambda: [1])  # λ = 0, resampling on


@dataclass
class ToyModelSpec:
    """One frozen toy model (ReLU output model) whose activations the SAEs decompose, and the
    SAE sweep trained on it."""

    label: str = "pentagon"
    source: str = "E00, p. 2"
    n_features: int = 5
    n_hidden: int = 2
    importance_decay: float = 0.9
    feature_probability: float = 0.1
    n_restarts: int = 8
    batch_size: int = 1024
    graded: bool = True
    cos_threshold: float = 0.95  # a feature is recovered when max cos > this
    min_fraction: float = 1.0  # graded: share of represented features that must be recovered
    min_represented: int = 0  # graded precondition (the pentagon must have all 5 features)
    sweep: SweepSpec = field(default_factory=SweepSpec)


@dataclass
class SAERecoveryConfig(ExperimentConfig):
    """E10 settings: the toy models (each with its SAE sweep) and the SAE training settings."""

    toy_models: list[ToyModelSpec] = field(default_factory=lambda: [ToyModelSpec()])
    sae: SAETrainConfig = field(default_factory=SAETrainConfig)
    represented_threshold: float = 0.5


def sweep_settings(sweep: SweepSpec) -> list[dict]:
    """One dict per SAE instance: size, l1, topk, resample."""
    settings = []
    for size in sweep.dictionary_sizes:
        for l1 in sweep.l1_values:
            for resample in sweep.resample:
                settings.append({"size": size, "l1": l1, "topk": 0, "resample": resample})
        for k in sweep.topk_values:
            settings.append({"size": size, "l1": 0.0, "topk": k, "resample": True})
    return settings


def setting_label(row: dict) -> str:
    kind = f"TopK k={int(row['topk'])}" if row["topk"] else f"L1 λ={row['l1']:g}"
    return f"{int(row['size'])} latents, {kind}, resample {'on' if row['resample'] else 'off'}"


# --- toy models ----------------------------------------------------------------------------------


@dataclass
class ToyModel:
    """A trained toy model (best restart): weights and what the SAEs are compared against."""

    label: str
    W: Tensor  # (hidden, feat)
    b: Tensor  # (feat,)
    loss: float
    feature_probability: float
    represented: Tensor  # (feat,) bool

    def sampler(self):
        dist = IndependentSparse(self.feature_probability, 1, self.W.shape[1])
        W = self.W

        def sample(batch: int, generator: torch.Generator) -> Tensor:
            return dist.sample(batch, generator)[:, 0] @ W.T.to(generator.device)

        return sample


def train_toy(
    spec: ToyModelSpec, train: TrainConfig, seed: int, threshold: float, device: torch.device
) -> ToyModel:
    """Train the toy model with its experiment's settings (``train`` with the spec's batch
    size); keep the lowest-loss restart."""
    grid = train_sparsity_grid(
        "relu_output",
        spec.n_features,
        spec.n_hidden,
        geometric_importance(spec.n_features, spec.importance_decay),
        [spec.feature_probability],
        spec.n_restarts,
        replace(train, batch_size=spec.batch_size),
        seed,
        device,
    )
    W, b, loss = grid.best_weights(0)
    return ToyModel(
        label=spec.label,
        W=W,
        b=b,
        loss=loss,
        feature_probability=spec.feature_probability,
        represented=metrics.represented_mask(W[None], threshold)[0],
    )


# --- SAEs ----------------------------------------------------------------------------------------


def train_sweep(
    toy: ToyModel, sweep: SweepSpec, config: SAERecoveryConfig, device: torch.device, index: int
) -> tuple[SparseAutoencoder, pd.DataFrame, Tensor, Tensor]:
    """Train and evaluate every SAE of one toy model's sweep. Returns the SAEs, one table row
    per SAE, each SAE's max cos per feature ``(inst, feat)`` and its live latents."""
    settings = sweep_settings(sweep)
    n_latents = max(s["size"] for s in settings)
    mask = torch.zeros(len(settings), n_latents, dtype=torch.bool)
    for i, s in enumerate(settings):
        mask[i, : s["size"]] = True
    sae = SparseAutoencoder(
        toy.W.shape[0],
        n_latents,
        len(settings),
        l1=[s["l1"] for s in settings],
        topk=[s["topk"] for s in settings],
        resample=[s["resample"] for s in settings],
        latent_mask=mask,
        seed=derive_seed(config.seed, index),
    ).to(device)
    result = train_sae(sae, toy.sampler(), config.sae, seed=config.seed + index, progress=True)
    generator = make_generator(derive_seed(config.seed, EVAL_STREAM, SAE_EVAL_KEY, index), device)
    h = toy.sampler()(config.sae.eval_samples, generator)
    stats = sae_stats(sae, h)
    rec = recovery(sae.unit_decoder().cpu(), stats.alive, toy.W, toy.represented)
    frame = pd.DataFrame(settings)
    frame.insert(0, "toy_model", toy.label)
    frame["instance"] = range(len(settings))
    frame["mean_max_cos"] = rec.mean_max_cos.numpy()
    for t, values in rec.fraction.items():
        frame[f"recovered_{t:g}"] = values.numpy()
    frame["l0"] = stats.l0.numpy()
    frame["mse"] = stats.mse.numpy()
    frame["fvu"] = stats.fvu.numpy()
    frame["dead_fraction"] = stats.dead_fraction.numpy()
    frame["resampled"] = result.resampled.numpy()
    frame["true_l0"] = float(toy.represented.sum()) * toy.feature_probability
    frame["wall_time_s"] = result.wall_time_s
    return sae, frame, rec.max_cos, stats.alive


def best_row(frame: pd.DataFrame, threshold: float) -> dict:
    """The sweep's best SAE by ground truth: most features above ``threshold``, then highest
    mean max cos."""
    column = f"recovered_{threshold:g}"
    ranked = frame.sort_values([column, "mean_max_cos"], ascending=[False, False])
    return dict(ranked.iloc[0].to_dict())


# --- run -----------------------------------------------------------------------------------------


def run(
    config: SAERecoveryConfig, run_dir: RunDir, device: torch.device, logger: logging.Logger
) -> list[Criterion]:
    frames: list[pd.DataFrame] = []
    criteria: list[Criterion] = []
    info_rows: dict[str, dict] = {}
    for index, spec in enumerate(config.toy_models):
        toy = train_toy(spec, config.train, config.seed, config.represented_threshold, device)
        n_rep = int(toy.represented.sum())
        logger.info(
            f"{spec.label} ({spec.source}): loss {toy.loss:.5f}, {n_rep} of {spec.n_features} "
            "features represented"
        )
        sae, frame, max_cos, alive = train_sweep(toy, spec.sweep, config, device, index)
        frame["graded"] = spec.graded
        frames.append(frame)
        run_dir.save_checkpoint(
            spec.label,
            {
                "toy_W": toy.W,
                "toy_b": toy.b,
                "toy_loss": toy.loss,
                "feature_probability": toy.feature_probability,
                "represented": toy.represented,
                "sae_state": {k: v.cpu() for k, v in sae.state_dict().items()},
                "max_cos": max_cos,
                "alive": alive,
                "settings": sweep_settings(spec.sweep),
            },
        )
        best = best_row(frame, spec.cos_threshold)
        column = f"recovered_{spec.cos_threshold:g}"
        passing = int((frame[column] >= spec.min_fraction - 1e-9).sum())
        logger.info(
            f"    {len(frame)} SAEs in {float(frame['wall_time_s'].iloc[0]):.1f} s; best: "
            f"{setting_label(best)} recovers {best[column]:.0%} at cos > {spec.cos_threshold:g} "
            f"({passing}/{len(frame)} settings reach {spec.min_fraction:.0%})"
        )
        summary = {
            "represented": n_rep,
            "best_setting": setting_label(best),
            "best_recovered": round(float(best[column]), 4),
            "best_mean_max_cos": round(float(best["mean_max_cos"]), 4),
            "best_fvu": round(float(best["fvu"]), 4),
            "settings_reaching_threshold": f"{passing}/{len(frame)}",
        }
        if spec.graded:
            ok = n_rep >= spec.min_represented and float(best[column]) >= spec.min_fraction - 1e-9
            criteria.append(
                check(
                    f"{spec.label}_recovered",
                    f"{spec.label} ({spec.source}, n = {spec.n_features}, m = {spec.n_hidden}, "
                    f"1 − S = {spec.feature_probability:g}): ≥ {spec.min_fraction:.0%} of the "
                    f"represented features recovered at cos > {spec.cos_threshold:g} by the best "
                    "SAE of the sweep (selected by ground truth)"
                    + (
                        f"; needs ≥ {spec.min_represented} represented"
                        if spec.min_represented
                        else ""
                    ),
                    PAGE,
                    ok,
                    summary,
                    f"≥ {spec.min_fraction}",
                    group=spec.label,
                )
            )
        else:
            info_rows[spec.label] = summary
    table = pd.concat(frames, ignore_index=True)
    run_dir.write_table(table, "metrics")
    criteria += report(config, table, info_rows)
    return criteria


def report(
    config: SAERecoveryConfig, table: pd.DataFrame, ungraded: dict[str, dict]
) -> list[Criterion]:
    """INFO rows: ungraded toy models, resampling, TopK vs L1, reconstruction vs recovery."""
    out: list[Criterion] = []
    if ungraded:
        out.append(
            info(
                "ungraded_models",
                "Toy models reported without grading (E03 setup): best SAE of the sweep",
                PAGE,
                ungraded,
            )
        )
    resampling, variants, recon = {}, {}, {}
    for spec in config.toy_models:
        rows = [dict(r) for r in table.to_dict("records") if r["toy_model"] == spec.label]
        col = f"recovered_{spec.cos_threshold:g}"
        l1_rows = [r for r in rows if not r["topk"]]
        for flag in (False, True):
            chosen = [r for r in l1_rows if bool(r["resample"]) == flag]
            if chosen:
                resampling.setdefault(spec.label, {})[f"resample_{'on' if flag else 'off'}"] = {
                    "best_recovered": round(max(float(r[col]) for r in chosen), 4),
                    "mean_recovered": round(float(np.mean([r[col] for r in chosen])), 4),
                    "mean_dead_fraction": round(
                        float(np.mean([r["dead_fraction"] for r in chosen])), 4
                    ),
                    "latents_resampled": int(sum(int(r["resampled"]) for r in chosen)),
                }
        topk_rows = [r for r in rows if r["topk"]]
        variants[spec.label] = {
            "l1_best": round(max((float(r[col]) for r in l1_rows), default=float("nan")), 4),
            **{
                f"topk_{int(r['topk'])}_{int(r['size'])}": round(float(r[col]), 4)
                for r in topk_rows
            },
        }
        low_error = [r for r in rows if float(r["fvu"]) < 0.1]
        recon[spec.label] = {
            "settings_with_fvu_below_0.1": len(low_error),
            "of_which_recover": sum(float(r[col]) >= spec.min_fraction - 1e-9 for r in low_error),
            "true_l0": round(float(rows[0]["true_l0"]), 3) if rows else None,
        }
    out += [
        info(
            "resampling",
            "With and without dead-latent resampling (ReLU + L1 SAEs): best and mean recovery, "
            "mean dead-latent fraction on held-out data, latents resampled",
            PAGE,
            resampling,
        ),
        info(
            "topk_vs_l1",
            "TopK SAEs (λ = 0, resampling on) against the best ReLU + L1 SAE: share recovered",
            PAGE,
            variants,
        ),
        info(
            "reconstruction_vs_recovery",
            "Low reconstruction error does not imply recovery: SAEs with FVU < 0.1 and how many "
            "of them recover the features; true L0 = represented features × p",
            PAGE,
            recon,
        ),
    ]
    return out


# --- figures -------------------------------------------------------------------------------------


def figures(run_dir: RunDir) -> list[Path]:
    """The pentagon's dictionary in 2D, the recovery sweeps, and reconstruction vs recovery."""
    config = from_dict(SAERecoveryConfig, yaml.safe_load(run_dir.config_path.read_text()))
    table = run_dir.read_table("metrics")
    paths: list[Path] = []
    for spec in config.toy_models:
        if spec.n_hidden != 2:
            continue
        state = run_dir.load_checkpoint(spec.label)
        rows = [dict(r) for r in table.to_dict("records") if r["toy_model"] == spec.label]
        col = f"recovered_{spec.cos_threshold:g}"
        best = max(rows, key=lambda r: (float(r[col]), float(r["mean_max_cos"])))
        smallest = [r for r in rows if int(r["size"]) == min(int(x["size"]) for x in rows)]
        worst_small = min(smallest, key=lambda r: (float(r[col]), float(r["mean_max_cos"])))
        decoder = state["sae_state"]["W_dec"].numpy()
        mask = state["sae_state"]["latent_mask"].numpy()
        alive = state["alive"].numpy()
        panels = [
            (title, decoder[k], mask[k], alive[k], float(row[col]))
            for title, row in (
                (f"best: {setting_label(best)}", best),
                (setting_label(worst_small), worst_small),
            )
            for k in [int(row["instance"])]
        ]
        paths += save_figure(
            dictionary_2d_figure(
                state["toy_W"].numpy(),
                panels,
                spec.cos_threshold,
                title=f"E10 · {spec.label}: true features (arrows) and SAE decoder directions",
            ),
            run_dir.figures_dir / f"{spec.label}_dictionary",
        )
    paths += save_figure(
        recovery_sweep_figure(
            table,
            {s.label: s.cos_threshold for s in config.toy_models},
            title="E10 · share of represented features recovered, per SAE setting",
        ),
        run_dir.figures_dir / "recovery_sweep",
    )
    paths += save_figure(
        reconstruction_vs_recovery_figure(
            table,
            {s.label: s.cos_threshold for s in config.toy_models},
            title="E10 · reconstruction error vs recovery (every SAE)",
        ),
        run_dir.figures_dir / "reconstruction_vs_recovery",
    )
    return paths


EXPERIMENT = Experiment(
    name="sae_recovery",
    code="E10",
    title="Sparse autoencoders on toy-model activations, scored against the true features",
    paper_pages=PAGE,
    config_cls=SAERecoveryConfig,
    run=run,
    figures=figures,
)
