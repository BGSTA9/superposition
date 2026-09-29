"""E09 mech_interp_abs: reverse-engineering small absolute-value models.

Paper: The Asymmetric Superposition Motif, pp. 38–39 (and Computation in Superposition,
pp. 34–38, for the model).

The p. 39 model is "only doing a limited amount of superposition, and many of the weights can
be understood as simply implementing absolute value in the expected way. However, there are a
few neurons doing something else…": two instances of *asymmetric superposition with
inhibition*. Its figure shows 6 features and 10 neurons, two fewer than the 12 that computing
|x| without superposition needs (two per feature). The paper gives neither density nor
importance.

We train models with m = 2n − 2 neurons in the density range where they do limited
superposition, then run :mod:`superposition.interp` on the best restart of each setting:

* a role for every neuron (positive-side, negative-side, (asymmetric) superposition, inhibitor,
  dead, unexplained);
* zero- and mean-ablation of every neuron, checked against what its role predicts;
* the motif detector, on every restart;
* the rebuild test: an idealized model built from the roles, and the share of the trained
  model's loss reduction (relative to predicting the mean) it recovers.

Denser and sparser settings are run too and reported without grading, to show where the
regime ends. Settings and criteria: docs/DECISIONS.md D-501, D-502.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch import Tensor

from superposition import interp, metrics
from superposition.acceptance import Criterion, check, info
from superposition.config import ExperimentConfig, from_dict
from superposition.data import IndependentSparse, geometric_importance
from superposition.experiments.base import Experiment
from superposition.experiments.common import format_probability
from superposition.models import AbsModel
from superposition.plotting.interp import (
    ablation_figure,
    motif_order,
    rebuild_figure,
    role_census_figure,
    role_weights_figure,
)
from superposition.plotting.style import save_figure
from superposition.runs import RunDir
from superposition.train import select_best, train_from_config
from superposition.utils import EVAL_STREAM, derive_seed, make_generator

PAGE = "pp. 38–39"
ANALYSIS_KEY = 9  # derive_seed key for E09's analysis streams


@dataclass
class ModelShape:
    """One model size: n features, m neurons."""

    n_features: int = 6
    n_hidden: int = 10


@dataclass
class MechInterpThresholds:
    """Acceptance thresholds (SPEC §7 E09; D-502)."""

    named_fraction_min: float = 0.9  # ≥ 90% of live neurons get a named role
    agreement_fraction_min: float = 0.9  # ≥ 90% of named neurons' ablations fit their role
    analysis_samples: int = 20_000  # samples per conditional term of the per-feature loss


@dataclass
class MechInterpConfig(ExperimentConfig):
    """E09 settings. Graded settings are the ``feature_probabilities`` (1 − S) of every shape;
    ``context_probabilities`` are trained and analyzed the same way but only reported."""

    shapes: list[ModelShape] = field(
        default_factory=lambda: [ModelShape(4, 6), ModelShape(5, 8), ModelShape(6, 10)]
    )
    feature_probabilities: list[float] = field(default_factory=lambda: [0.3, 0.2])
    context_probabilities: list[float] = field(default_factory=lambda: [0.5, 0.1])
    importance_decay: float = 1.0
    n_restarts: int = 20
    showcase: ModelShape = field(default_factory=ModelShape)
    roles: interp.RoleThresholds = field(default_factory=interp.RoleThresholds)
    thresholds: MechInterpThresholds = field(default_factory=MechInterpThresholds)


# --- training ------------------------------------------------------------------------------------


@dataclass
class TrainedShape:
    """Every restart of one model size (CPU tensors). Instance ``c · restarts + r`` is restart
    r of condition c; conditions are the graded densities, then the context ones."""

    n_features: int
    n_hidden: int
    probabilities: list[float]
    graded: list[bool]
    W1: Tensor  # (inst, hidden, feat)
    W2: Tensor  # (inst, feat, hidden)
    b: Tensor  # (inst, feat)
    final_loss: Tensor  # (inst,)
    condition: Tensor  # (inst,)
    best: Tensor  # (cond,)
    importance: Tensor  # (feat,)
    wall_time_s: float

    @property
    def label(self) -> str:
        return f"n{self.n_features}_m{self.n_hidden}"

    def model_weights(self, k: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.W1[k], self.W2[k], self.b[k]


def train_shape(shape: ModelShape, config: MechInterpConfig, device: torch.device) -> TrainedShape:
    """Train every density × restart of one model size as a single multi-instance model."""
    probabilities = [*config.feature_probabilities, *config.context_probabilities]
    graded = [True] * len(config.feature_probabilities) + [False] * len(
        config.context_probabilities
    )
    condition = torch.arange(len(probabilities)).repeat_interleave(config.n_restarts)
    p = torch.tensor(probabilities, dtype=torch.float32)[condition]
    n_inst = len(condition)
    model = AbsModel(shape.n_features, shape.n_hidden, n_inst, seed=config.seed).to(device)
    dist = IndependentSparse(p, n_inst, shape.n_features, interp.ABS_LOW, interp.ABS_HIGH)
    importance = geometric_importance(shape.n_features, config.importance_decay)
    result = train_from_config(model, dist, importance, config.train, config.seed, progress=True)
    return TrainedShape(
        n_features=shape.n_features,
        n_hidden=shape.n_hidden,
        probabilities=probabilities,
        graded=graded,
        W1=model.W1.detach().cpu(),
        W2=model.W2.detach().cpu(),
        b=model.b.detach().cpu(),
        final_loss=result.final_loss,
        condition=condition,
        best=select_best(result.final_loss, condition),
        importance=importance,
        wall_time_s=result.wall_time_s,
    )


# --- analysis ------------------------------------------------------------------------------------


@dataclass
class ModelAnalysis:
    """Roles, motifs, ablations and rebuild of one model."""

    shape: str
    feature_probability: float
    graded: bool
    instance: int
    loss: float
    roles: list[interp.NeuronRole]
    motifs: list[interp.Motif]
    ablation: interp.Ablation
    rebuild: interp.Rebuild
    agrees: list[bool | None]  # per neuron: zero and mean ablations both fit (None: no prediction)

    @property
    def live(self) -> list[interp.NeuronRole]:
        return [r for r in self.roles if r.kind != interp.DEAD]

    def counts(self) -> dict[str, int]:
        live = self.live
        named = [r for r in live if r.named]
        return {
            "neurons": len(self.roles),
            "live": len(live),
            "dead": len(self.roles) - len(live),
            "named": len(named),
            "spec_named": sum(r.kind in interp.SPEC_NAMED_KINDS for r in live),
            "agree": sum(bool(self.agrees[r.neuron]) for r in named),
            "motifs": len(self.motifs),
        }


def analyze_model(
    trained: TrainedShape, k: int, thr: interp.RoleThresholds, n_samples: int, seed: int
) -> ModelAnalysis:
    """Run the whole interp pipeline on instance ``k``."""
    W1, W2, b = trained.model_weights(k)
    c = int(trained.condition[k])
    p = trained.probabilities[c]
    generator = make_generator(derive_seed(seed, EVAL_STREAM, ANALYSIS_KEY, k))
    roles = interp.neuron_roles(W1, W2, thr)
    ablation = interp.ablation_matrix(W1, W2, b, p, n_samples, generator)
    agrees = []
    for role in roles:
        zero = interp.ablation_agrees(
            role, ablation.relative("zero")[role.neuron], thr.ablation_rise
        )
        mean = interp.ablation_agrees(
            role, ablation.relative("mean")[role.neuron], thr.ablation_rise
        )
        agrees.append(None if zero is None or mean is None else zero and mean)
    rebuild = interp.rebuild_test(roles, W1, W2, b, trained.importance, p, n_samples, generator)
    return ModelAnalysis(
        shape=trained.label,
        feature_probability=p,
        graded=trained.graded[c],
        instance=k,
        loss=float(trained.final_loss[k]),
        roles=roles,
        motifs=interp.find_motifs(W1, W2, b, thr),
        ablation=ablation,
        rebuild=rebuild,
        agrees=agrees,
    )


def motif_census(trained: TrainedShape, thr: interp.RoleThresholds) -> pd.DataFrame:
    """Every motif in every restart of one model size (one row per motif)."""
    rows = []
    best = set(trained.best.tolist())
    for k in range(trained.W1.shape[0]):
        W1, W2, b = trained.model_weights(k)
        c = int(trained.condition[k])
        for motif in interp.find_motifs(W1, W2, b, thr):
            rows.append(
                {
                    "shape": trained.label,
                    "feature_probability": trained.probabilities[c],
                    "graded": trained.graded[c],
                    "instance": k,
                    "is_best": k in best,
                    "loss": float(trained.final_loss[k]),
                    **vars(motif),
                }
            )
    return pd.DataFrame(rows)


def restarts_with_motif(trained: TrainedShape, census: pd.DataFrame) -> dict[float, str]:
    """Per density, how many restarts contain at least one motif ("k/restarts")."""
    out = {}
    for c, p in enumerate(trained.probabilities):
        n = int((trained.condition == c).sum())
        hits = (
            census.loc[census["feature_probability"] == p, "instance"].nunique()
            if len(census)
            else 0
        )
        out[p] = f"{hits}/{n}"
    return out


def role_rows(analysis: ModelAnalysis, W1n: Tensor, W2n: Tensor) -> pd.DataFrame:
    """The role table of one analyzed model, with its identifying columns."""
    table = interp.role_table(analysis.roles, W1n, W2n, analysis.ablation)
    table.insert(0, "shape", analysis.shape)
    table.insert(1, "feature_probability", analysis.feature_probability)
    table.insert(2, "graded", analysis.graded)
    table["agrees"] = analysis.agrees
    table["in_motif"] = [
        any(r.neuron in (m.asymmetric_neuron, m.inhibitor_neuron) for m in analysis.motifs)
        for r in analysis.roles
    ]
    return table


def ablation_rows(analysis: ModelAnalysis) -> pd.DataFrame:
    """Long table: one row per (neuron, feature, ablation kind)."""
    frames = []
    for kind in ("zero", "mean"):
        delta = getattr(analysis.ablation, kind)
        relative = analysis.ablation.relative(kind)
        n_hidden, n_feat = delta.shape
        frames.append(
            pd.DataFrame(
                {
                    "shape": analysis.shape,
                    "feature_probability": analysis.feature_probability,
                    "kind": kind,
                    "neuron": np.repeat(np.arange(n_hidden), n_feat),
                    "feature": np.tile(np.arange(n_feat), n_hidden),
                    "delta_loss": delta.flatten().numpy(),
                    "relative": relative.flatten().numpy(),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def rebuild_rows(analysis: ModelAnalysis) -> pd.DataFrame:
    """One row per feature: trained, rebuild and mean-prediction losses."""
    rb = analysis.rebuild
    n_feat = rb.trained.shape[0]
    return pd.DataFrame(
        {
            "shape": analysis.shape,
            "feature_probability": analysis.feature_probability,
            "graded": analysis.graded,
            "feature": np.arange(n_feat),
            "trained": rb.trained.numpy(),
            "rebuild": rb.rebuild.numpy(),
            "rebuild_trained_bias": rb.rebuild_trained_bias.numpy(),
            "mean_prediction": rb.mean_prediction.numpy(),
            "fraction_recovered": rb.fraction_by_feature.numpy(),
        }
    )


def summary_row(analysis: ModelAnalysis, motif_restarts: str) -> dict:
    counts = analysis.counts()
    return {
        "shape": analysis.shape,
        "feature_probability": analysis.feature_probability,
        "graded": analysis.graded,
        "instance": analysis.instance,
        "loss": analysis.loss,
        **counts,
        "roles": [r.label for r in analysis.roles],
        "restarts_with_motif": motif_restarts,
        "rebuild_recovered": analysis.rebuild.fraction_recovered,
        "rebuild_recovered_trained_bias": analysis.rebuild.fraction_recovered_trained_bias,
    }


# --- run -----------------------------------------------------------------------------------------


def run(
    config: MechInterpConfig, run_dir: RunDir, device: torch.device, logger: logging.Logger
) -> list[Criterion]:
    thr = config.roles
    analyses: list[ModelAnalysis] = []
    censuses: list[pd.DataFrame] = []
    summary: list[dict] = []
    roles_tables, ablation_tables, rebuild_tables = [], [], []
    for shape in config.shapes:
        logger.info(
            f"n={shape.n_features}, m={shape.n_hidden}: "
            f"{len(config.feature_probabilities) + len(config.context_probabilities)} densities × "
            f"{config.n_restarts} restarts"
        )
        trained = train_shape(shape, config, device)
        logger.info(f"    trained in {trained.wall_time_s:.1f} s")
        run_dir.save_checkpoint(trained.label, vars(trained))
        census = motif_census(trained, thr)
        censuses.append(census)
        per_density = restarts_with_motif(trained, census)
        for k in trained.best.tolist():
            analysis = analyze_model(
                trained, k, thr, config.thresholds.analysis_samples, config.seed
            )
            analyses.append(analysis)
            W1n, W2n = metrics.normalize_hidden_units(trained.W1[k][None], trained.W2[k][None])
            roles_tables.append(role_rows(analysis, W1n[0], W2n[0]))
            ablation_tables.append(ablation_rows(analysis))
            rebuild_tables.append(rebuild_rows(analysis))
            summary.append(summary_row(analysis, per_density[analysis.feature_probability]))
            counts = analysis.counts()
            logger.info(
                f"    1−S={format_probability(analysis.feature_probability)}: "
                f"named {counts['named']}/{counts['live']}, agree {counts['agree']}, "
                f"motifs {counts['motifs']} (restarts with a motif: "
                f"{per_density[analysis.feature_probability]}), rebuild "
                f"{analysis.rebuild.fraction_recovered:.3f}"
            )
    motifs = pd.concat(censuses, ignore_index=True) if censuses else pd.DataFrame()
    summary_frame = pd.DataFrame(summary)
    run_dir.write_table(summary_frame, "metrics")
    run_dir.write_table(pd.concat(roles_tables, ignore_index=True), "roles")
    run_dir.write_table(pd.concat(ablation_tables, ignore_index=True), "ablations")
    run_dir.write_table(pd.concat(rebuild_tables, ignore_index=True), "rebuild")
    run_dir.write_table(motifs, "motifs")
    return evaluate(config, summary_frame, motifs)


def _model_key(row: dict) -> str:
    return f"{row['shape']} 1-S={format_probability(row['feature_probability'])}"


def _select(frame: pd.DataFrame, mask: object) -> pd.DataFrame:
    """Rows of ``frame`` where the boolean ``mask`` holds."""
    return frame.loc[np.asarray(mask, dtype=bool)]


def _total(rows: list[dict], key: str) -> int:
    return int(sum(int(r[key]) for r in rows))


def evaluate(
    config: MechInterpConfig, summary: pd.DataFrame, motifs: pd.DataFrame
) -> list[Criterion]:
    """E09 criteria from the per-model summary and the motif census."""
    thr = config.thresholds
    all_rows: list[dict] = [dict(r) for r in summary.to_dict("records")]
    records = [r for r in all_rows if r["graded"]]
    context = [r for r in all_rows if not r["graded"]]
    live, named = _total(records, "live"), _total(records, "named")
    agree = _total(records, "agree")
    named_fraction = named / live if live else 0.0
    agree_fraction = agree / named if named else 0.0
    densities = ", ".join(format_probability(p) for p in config.feature_probabilities)
    shapes = ", ".join(f"n={s.n_features}/m={s.n_hidden}" for s in config.shapes)
    motif_configs = {_model_key(r): r["restarts_with_motif"] for r in all_rows}
    motif_rows: list[dict] = [dict(r) for r in motifs.to_dict("records")]
    n_models_with_motif = len({(r["shape"], r["instance"]) for r in motif_rows})
    criteria = [
        check(
            "named_roles",
            f"≥ {thr.named_fraction_min:.0%} of live neurons receive a named role (best model of "
            f"each of {shapes} at 1 − S ∈ {{{densities}}})",
            PAGE,
            named_fraction >= thr.named_fraction_min,
            {
                "fraction": round(named_fraction, 4),
                "named": named,
                "live": live,
                "dead": _total(records, "dead"),
                "fraction_spec_roles_only": round(_total(records, "spec_named") / live, 4)
                if live
                else 0.0,
                "per_model": {_model_key(r): f"{r['named']}/{r['live']}" for r in records},
            },
            f"≥ {thr.named_fraction_min}",
        ),
        check(
            "ablations_agree",
            f"Ablation effects agree with the assigned roles: for ≥ {thr.agreement_fraction_min:.0%} "
            "of named neurons, zeroing and mean-ablating each raise the loss of every feature the "
            f"role computes or inhibits (by ≥ {config.roles.ablation_rise:.0%} of predicting the "
            "mean), most on one of them",
            PAGE,
            agree_fraction >= thr.agreement_fraction_min,
            {
                "fraction": round(agree_fraction, 4),
                "agree": agree,
                "named": named,
                "per_model": {_model_key(r): f"{r['agree']}/{r['named']}" for r in records},
            },
            f"≥ {thr.agreement_fraction_min}",
        ),
        check(
            "motif_found",
            "The asymmetric superposition + inhibition motif is found in at least one model "
            "(searched in every restart); restarts with a motif per configuration",
            PAGE,
            n_models_with_motif >= 1,
            {"models_with_motif": n_models_with_motif, "per_configuration": motif_configs},
            "≥ 1 model",
        ),
        info(
            "rebuild",
            "Rebuild test: share of the trained model's loss reduction (vs predicting the mean) "
            "recovered by the model rebuilt by hand from the roles (idealized biases / trained "
            "biases), best model per setting",
            PAGE,
            {
                _model_key(r): {
                    "recovered": round(float(r["rebuild_recovered"]), 4),
                    "recovered_trained_bias": round(float(r["rebuild_recovered_trained_bias"]), 4),
                }
                for r in all_rows
            },
        ),
    ]
    if motif_rows:
        best = [m for m in motif_rows if m["is_best"]]
        criteria.append(
            info(
                "motif_details",
                "Motifs in the best models: input and output size ratios, the positive "
                "interference, the inhibition, and the net 1-sparse interference on the victim "
                "(< 0: converted into negative interference, p. 39)",
                PAGE,
                {
                    "motifs_in_best_models": len(best),
                    "net_interference_negative": f"{sum(m['net_interference'] < 0 for m in best)}/{len(best)}",
                    "all_restarts_net_negative": f"{sum(m['net_interference'] < 0 for m in motif_rows)}/{len(motif_rows)}",
                    "examples": [
                        {
                            "model": f"{m['shape']} 1-S={format_probability(m['feature_probability'])}",
                            "neurons": [int(m["asymmetric_neuron"]), int(m["inhibitor_neuron"])],
                            "heavy": int(m["heavy"]),
                            "victim": int(m["victim"]),
                            "input_ratio": round(float(m["input_ratio"]), 2),
                            "output_ratio": round(float(m["output_ratio"]), 2),
                            "interference": round(float(m["interference"]), 3),
                            "inhibition": round(float(m["inhibition"]), 3),
                            "net": round(float(m["net_interference"]), 3),
                        }
                        for m in best
                    ],
                },
            )
        )
    if len(context):
        criteria.append(
            info(
                "context_densities",
                "Outside the graded regime (reported only): named roles, ablation agreement, "
                "motifs and rebuild of the best model at the context densities",
                PAGE,
                {
                    _model_key(r): {
                        "named": f"{r['named']}/{r['live']}",
                        "agree": f"{r['agree']}/{r['named']}",
                        "motifs": int(r["motifs"]),
                        "restarts_with_motif": r["restarts_with_motif"],
                        "rebuild_recovered": round(float(r["rebuild_recovered"]), 4),
                        "roles": r["roles"],
                    }
                    for r in context
                },
            )
        )
    criteria.append(
        info(
            "roles_graded_models",
            "Role of every neuron in the graded models (best restart)",
            PAGE,
            {_model_key(r): r["roles"] for r in records},
        )
    )
    return criteria


# --- figures -------------------------------------------------------------------------------------


def load_shape(run_dir: RunDir, label: str) -> TrainedShape:
    return TrainedShape(**run_dir.load_checkpoint(label))


def showcase_model(config: MechInterpConfig, summary: pd.DataFrame) -> dict | None:
    """The model drawn like p. 39: the configured shape's best graded model with the most
    motifs (densest first on ties), else any graded best model with a motif."""
    label = f"n{config.showcase.n_features}_m{config.showcase.n_hidden}"
    graded = _select(summary, summary["graded"].astype(bool) & (summary["motifs"] > 0))
    preferred = _select(graded, graded["shape"] == label)
    pool = preferred if len(preferred) else graded
    if not len(pool):
        return None
    pool = pool.sort_values(["motifs", "feature_probability"], ascending=[False, False])
    return pool.iloc[0].to_dict()


def figures(run_dir: RunDir) -> list[Path]:
    """The p. 39-style showcase (weights with roles, motif highlighted), its ablation matrix,
    the rebuild comparison and a role census across settings."""
    config = from_dict(MechInterpConfig, yaml.safe_load(run_dir.config_path.read_text()))
    summary = run_dir.read_table("metrics")
    roles = run_dir.read_table("roles")
    ablations = run_dir.read_table("ablations")
    rebuild = run_dir.read_table("rebuild")
    paths: list[Path] = []
    chosen = showcase_model(config, summary)
    if chosen is not None:
        trained = load_shape(run_dir, chosen["shape"])
        k = int(chosen["instance"])
        W1n, W2n = metrics.normalize_hidden_units(trained.W1[k][None], trained.W2[k][None])
        p = float(chosen["feature_probability"])
        table = _select(
            roles, (roles["shape"] == chosen["shape"]) & (roles["feature_probability"] == p)
        )
        labels = table["role"].tolist()
        order = motif_order(table["kind"].tolist(), table["role"].tolist(), W1n[0].numpy())
        highlight = set(table.loc[table["in_motif"], "neuron"].astype(int))
        title = (
            f"E09 · n = {trained.n_features}, m = {trained.n_hidden}, "
            f"1 − S = {format_probability(p)}: roles (footnote-15 normalized)"
        )
        paths += save_figure(
            role_weights_figure(W1n[0].numpy(), W2n[0].numpy(), labels, order, highlight, title),
            run_dir.figures_dir / "motif_model",
        )
        abl = _select(
            ablations,
            (ablations["shape"] == chosen["shape"]) & (ablations["feature_probability"] == p),
        )
        n_hidden, n_feat = trained.n_hidden, trained.n_features
        grids = {
            kind: _select(abl, abl["kind"] == kind)
            .sort_values(["neuron", "feature"])["relative"]
            .to_numpy()
            .reshape(n_hidden, n_feat)
            for kind in ("zero", "mean")
        }
        paths += save_figure(
            ablation_figure(
                grids["zero"],
                grids["mean"],
                labels,
                order,
                title="E09 · ablations of the model above: Δ loss / loss of predicting the mean",
            ),
            run_dir.figures_dir / "ablations",
        )
    paths += save_figure(
        rebuild_figure(summary, rebuild, title="E09 · rebuild test: loss reduction recovered"),
        run_dir.figures_dir / "rebuild",
    )
    paths += save_figure(
        role_census_figure(summary, roles, title="E09 · neuron roles per setting (best model)"),
        run_dir.figures_dir / "role_census",
    )
    return paths


EXPERIMENT = Experiment(
    name="mech_interp_abs",
    code="E09",
    title="Reverse-engineering |x| models: neuron roles, ablations, the asymmetric "
    "superposition motif, rebuild",
    paper_pages=PAGE,
    config_cls=MechInterpConfig,
    run=run,
    figures=figures,
)
