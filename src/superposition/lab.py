"""Computation behind the interactive lab: live Playground training, feature tables, probes,
and loaders for the results pages (Phase diagrams, Geometry, Non-uniform, Dynamics, Neurons,
Adversarial, SAE lab).

SPEC §8 (The interactive lab). Paper: Demonstrating Superposition, pp. 9–12 (the models and
the WᵀW / ‖Wᵢ‖ views), Feature Dimensionality, pp. 18–19 (the Dᵢ table), Superposition in a
Privileged Basis, p. 31 (the ReLU hidden model); Superposition as a Phase Change, pp. 14–16;
The Geometry of Superposition, pp. 16–25; Learning Dynamics, pp. 26–27; Adversarial
Robustness, p. 29; Privileged Basis and Computation in Superposition, pp. 31–39.

The Streamlit pages in ``app/`` only lay out widgets and figures; everything they compute
comes from this module, so it is unit-tested without a browser. Results are plain NumPy
(picklable, so ``st.cache_data`` can cache them by config).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import einops
import numpy as np
import pandas as pd
import torch
import yaml
from jaxtyping import Float

from superposition import metrics
from superposition.config import TrainConfig, from_dict
from superposition.data import geometric_importance
from superposition.experiments.common import train_sparsity_grid
from superposition.models import TiedKind, TiedModel
from superposition.plotting.neurons import NeuronPanel, neuron_order_by_top_feature
from superposition.runs import RunDir, latest_run, project_root
from superposition.utils import resolve_device

# Control ranges of the Playground (SPEC §8).
N_FEATURES_RANGE = (2, 40)
N_HIDDEN_RANGE = (1, 10)
IMPORTANCE_DECAY_RANGE = (0.5, 1.0)
FEATURE_PROBABILITY_RANGE = (0.001, 1.0)
MODEL_KINDS: tuple[TiedKind, ...] = ("linear", "relu_output", "relu_hidden")
STEPS_RANGE = (100, 50_000)
RESTARTS_RANGE = (1, 32)

# Thresholds used by the Dᵢ table (same defaults as the experiments, SPEC §3 / D-110).
REPRESENTED_THRESHOLD = 0.5
SUPERPOSITION_THRESHOLD = 0.05
FRACTION_TOLERANCE = 0.02


@dataclass(frozen=True)
class PlaygroundConfig:
    """One Playground training job. Defaults reproduce the p. 2 pentagon in a few seconds.

    ``feature_probability`` is 1 − S. Defaults: 5,000 steps × 4 restarts train in ≈ 3 s on an
    Apple M2 CPU (the SPEC §8 target is ≤ 10 s for n = 5, m = 2); see docs/DECISIONS.md D-201.
    """

    n_features: int = 5
    n_hidden: int = 2
    importance_decay: float = 0.9
    feature_probability: float = 0.1
    model_kind: TiedKind = "relu_output"
    steps: int = 5_000
    restarts: int = 4
    seed: int = 0
    batch_size: int = 1024
    lr: float = 1e-3
    eval_samples: int = 20_000

    def validate(self) -> None:
        """Raise ValueError if any control is outside the Playground's ranges."""
        checks = [
            ("n_features", self.n_features, N_FEATURES_RANGE),
            ("n_hidden", self.n_hidden, N_HIDDEN_RANGE),
            ("importance_decay", self.importance_decay, IMPORTANCE_DECAY_RANGE),
            ("feature_probability", self.feature_probability, FEATURE_PROBABILITY_RANGE),
            ("steps", self.steps, STEPS_RANGE),
            ("restarts", self.restarts, RESTARTS_RANGE),
        ]
        for name, value, (low, high) in checks:
            if not low <= value <= high:
                raise ValueError(f"{name}={value} is outside [{low}, {high}]")
        if self.model_kind not in MODEL_KINDS:
            raise ValueError(f"model_kind must be one of {MODEL_KINDS}, got {self.model_kind!r}")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")


@dataclass
class PlaygroundResult:
    """A trained Playground job (NumPy arrays; restart r is row r).

    Attributes:
        W: ``(restarts, hidden, feat)`` weights; ``b``: ``(restarts, feat)`` biases.
        final_loss: ``(restarts,)`` loss on ``eval_samples`` fresh samples.
        best: index of the lowest-loss restart.
        loss_history: ``(log, restarts)`` training-batch loss at ``log_steps``.
        importance: ``(feat,)``; ``train_seconds``: wall-clock training time.
    """

    config: PlaygroundConfig
    W: np.ndarray
    b: np.ndarray
    final_loss: np.ndarray
    best: int
    loss_history: np.ndarray
    log_steps: list[int]
    importance: np.ndarray
    train_seconds: float
    device: str = "cpu"

    @property
    def best_W(self) -> np.ndarray:
        return self.W[self.best]

    @property
    def best_b(self) -> np.ndarray:
        return self.b[self.best]

    @property
    def best_loss(self) -> float:
        return float(self.final_loss[self.best])

    def best_model(self) -> TiedModel:
        """The lowest-loss restart as a one-instance ``TiedModel`` (for the probe panel)."""
        c = self.config
        model = TiedModel.from_kind(c.model_kind, c.n_features, c.n_hidden, n_inst=1, seed=c.seed)
        with torch.no_grad():
            model.W.copy_(torch.as_tensor(self.best_W)[None])
            model.b.copy_(torch.as_tensor(self.best_b)[None])
        return model


def train_playground(config: PlaygroundConfig, device: str = "cpu") -> PlaygroundResult:
    """Train ``config.restarts`` instances of one model and keep them all (SPEC §8).

    Uses the same grid trainer as the experiments (one condition × restarts), so the
    Playground's models are exactly what ``superposition run`` would train with these settings.
    """
    config.validate()
    torch_device = resolve_device(device)
    importance = geometric_importance(config.n_features, config.importance_decay)
    train_config = TrainConfig(
        steps=config.steps,
        batch_size=config.batch_size,
        lr=config.lr,
        log_every=max(1, config.steps // 100),
        eval_samples=config.eval_samples,
    )
    start = time.perf_counter()
    grid = train_sparsity_grid(
        config.model_kind,
        config.n_features,
        config.n_hidden,
        importance,
        [config.feature_probability],
        config.restarts,
        train_config,
        seed=config.seed,
        device=torch_device,
    )
    return PlaygroundResult(
        config=config,
        W=grid.W.numpy(),
        b=grid.b.numpy(),
        final_loss=grid.final_loss.numpy(),
        best=int(grid.best[0]),
        loss_history=grid.loss_history.numpy(),
        log_steps=list(grid.log_steps),
        importance=importance.numpy(),
        train_seconds=time.perf_counter() - start,
        device=str(torch_device),
    )


def config_from_dict(values: dict) -> PlaygroundConfig:
    """Build and validate a PlaygroundConfig from plain values (e.g. widget state)."""
    config = PlaygroundConfig(**values)
    config.validate()
    return config


def config_to_dict(config: PlaygroundConfig) -> dict:
    return asdict(config)


def feature_table(
    W: Float[np.ndarray, "hidden feat"], b: Float[np.ndarray, " feat"], importance: np.ndarray
) -> pd.DataFrame:
    """One row per feature: importance, ‖Wᵢ‖, bias, superposition score, Dᵢ, geometry.

    Paper: p. 11 (norm and superposition score), p. 18 (Dᵢ), p. 19 (characteristic fractions).
    """
    Wt = torch.as_tensor(W, dtype=torch.float32)[None]
    norms = metrics.feature_norms(Wt)[0].numpy()
    score = metrics.superposition_score(Wt)[0].numpy()
    dims = metrics.feature_dimensionality(Wt)[0]
    fraction = metrics.classify_fraction(dims, tol=FRACTION_TOLERANCE).numpy()
    names = [name for name, _ in metrics.CHARACTERISTIC_FRACTIONS]
    represented = norms > REPRESENTED_THRESHOLD
    return pd.DataFrame(
        {
            "feature": np.arange(W.shape[1]),
            "importance": importance,
            "norm ‖Wᵢ‖": norms,
            "bias bᵢ": b,
            "superposition Σⱼ≠ᵢ(Ŵᵢ·Wⱼ)²": score,
            "Dᵢ": dims.numpy(),
            "nearest fraction": [names[k] if k >= 0 else "—" for k in fraction],
            "represented": represented,
            "in superposition": represented & (score > SUPERPOSITION_THRESHOLD),
        }
    )


def summary(result: PlaygroundResult) -> dict[str, float | int | bool]:
    """Headline numbers for the best restart."""
    Wt = torch.as_tensor(result.best_W, dtype=torch.float32)[None]
    represented = metrics.represented_mask(Wt, REPRESENTED_THRESHOLD)[0]
    return {
        "best loss": result.best_loss,
        "represented features": int(represented.sum()),
        "dimensions per feature D*": float(metrics.dims_per_feature(Wt)[0]),
        "superposition (WᵀW singular on represented)": bool(
            metrics.is_superposition(Wt, REPRESENTED_THRESHOLD)[0]
        ),
    }


@dataclass
class ProbeResult:
    """Every intermediate quantity for one input, one feature per entry (SPEC §8 probe panel).

    ``hidden_pre`` is Wx and ``hidden`` is h (they differ only for the ReLU hidden model).
    ``readout`` is Wᵀh, ``readout_plus_bias`` is Wᵀh + b, ``output`` is x′ after the output
    activation (identical to ``readout_plus_bias`` for the linear model), ``target`` is x.
    """

    x: np.ndarray
    hidden_pre: np.ndarray
    hidden: np.ndarray
    readout: np.ndarray
    readout_plus_bias: np.ndarray
    output: np.ndarray
    target: np.ndarray

    def feature_frame(self, importance: np.ndarray) -> pd.DataFrame:
        error = self.output - self.target
        return pd.DataFrame(
            {
                "feature": np.arange(len(self.x)),
                "input x": self.x,
                "readout Wᵀh": self.readout,
                "+ bias": self.readout_plus_bias,
                "output x′": self.output,
                "error x′ − x": error,
                "weighted sq. error": importance * error**2,
            }
        )


def probe(model: TiedModel, x: Float[np.ndarray, " feat"]) -> ProbeResult:
    """Run one input through a one-instance model and return every intermediate stage."""
    if model.n_inst != 1:
        raise ValueError("probe expects a one-instance model (use PlaygroundResult.best_model)")
    xt = torch.as_tensor(np.asarray(x, dtype=np.float32))[None, None]  # (batch=1, inst=1, feat)
    with torch.no_grad():
        hidden_pre = einops.einsum(
            model.W, xt, "inst hidden feat, batch inst feat -> batch inst hidden"
        )
        hidden = model.hidden(xt)
        readout_plus_bias = model.readout(hidden)
        readout = readout_plus_bias - model.b
        output = model(xt)
        target = model.target(xt)
    flat = [
        t[0, 0].numpy().astype(np.float64)
        for t in (xt, hidden_pre, hidden, readout, readout_plus_bias, output, target)
    ]
    return ProbeResult(*flat)


# --- Home page data --------------------------------------------------------------------------


def paper_map(path: Path | None = None) -> pd.DataFrame:
    """The table in docs/PAPER_MAP.md as a DataFrame (single source of truth for the map).

    Markdown code backticks are removed; the columns are the table's header cells.
    """
    path = path or project_root() / "docs" / "PAPER_MAP.md"
    rows = [
        [cell.strip().replace("`", "") for cell in line.strip().strip("|").split("|")]
        for line in path.read_text().splitlines()
        if line.strip().startswith("|")
    ]
    if len(rows) < 2:
        return pd.DataFrame()
    header, body = rows[0], [r for r in rows[2:] if len(r) == len(rows[0])]
    return pd.DataFrame(body, columns=header)


def status_frame() -> pd.DataFrame:
    """One row per experiment: cited status (docs) and latest local run (live).

    Cited statuses come from docs/cited_runs.yaml, so they show even on a fresh clone; the
    latest local run is read from results/ through runs.py.
    """
    from superposition.experiments import REGISTRY
    from superposition.report import load_citations

    citations = load_citations()
    rows = []
    for name, exp in REGISTRY.items():
        cited = citations.get(name)
        run = latest_run(name)
        latest = (
            run.read_acceptance() if run is not None and run.acceptance_path.is_file() else None
        )
        manifest = run.read_manifest() if run is not None else {}
        rows.append(
            {
                "code": exp.code,
                "experiment": name,
                "paper": exp.paper_pages,
                "cited status": f"{cited.overall} ({cited.n_passed}/{cited.n_graded})"
                if cited
                else "—",
                "cited run": cited.run_id if cited else "—",
                "latest local run": run.run_id if run is not None else "none",
                "latest status": f"{latest['overall']} ({latest['n_passed']}/{latest['n_graded']})"
                if latest
                else "—",
                "latest profile · device": f"{latest['profile']} · {manifest.get('device')}"
                if latest
                else "—",
                "clean tree": (not manifest.get("git", {}).get("dirty", True)) if latest else None,
                "command": f"uv run superposition run {name}",
            }
        )
    return pd.DataFrame(rows)


# --- results pages: Dynamics, Neurons, Adversarial (SPEC §8 pages 6–8) -----------------------
# These pages never train paper-scale models; they read a saved run through runs.py.


def run_command(name: str) -> str:
    """The CLI command that creates the run a results page needs."""
    return f"uv run superposition run {name} --profile quick"


def criteria_frame(run: RunDir) -> pd.DataFrame:
    """A run's acceptance criteria as a table: group, status, criterion, value (abridged)."""
    doc = run.read_acceptance()
    return pd.DataFrame(
        [
            {
                "group": c["group"],
                "status": c["status"],
                "criterion": c["description"],
                "value": json.dumps(c["value"], ensure_ascii=False)[:400],
            }
            for c in doc["criteria"]
        ]
    )


@dataclass
class DynamicsRun:
    """What the Dynamics page shows, loaded from a learning_dynamics run (E05).

    Part (a): ``steps`` ``(T,)``, ``dimensionality`` ``(T, restarts, feat)``, ``loss``
    ``(T, restarts)`` on a fixed batch, detected ``jumps``. Part (b): ``trans_steps`` ``(T′,)``,
    ``W`` ``(T′, inst, 3, feat)``, ``stages`` ``(inst, T′)``, ``trans_loss`` ``(inst, T′)``,
    ``sets`` ``(feat,)`` and the per-run ``table``.
    """

    run_id: str
    n_hidden: int
    steps: np.ndarray
    dimensionality: np.ndarray
    loss: np.ndarray
    jumps: list[dict]
    inverse_density: float
    trans_steps: np.ndarray
    W: np.ndarray
    stages: np.ndarray
    trans_loss: np.ndarray
    sets: np.ndarray
    table: pd.DataFrame


def load_dynamics(run: RunDir) -> DynamicsRun:
    """Load an E05 run's trajectories (Paper: pp. 26–27)."""
    energy = run.load_checkpoint("energy_levels")
    trans = run.load_checkpoint("transformations")
    return DynamicsRun(
        run_id=run.run_id,
        n_hidden=int(energy["W_final"].shape[1]),
        steps=np.asarray(energy["steps"]),
        dimensionality=np.asarray(energy["dimensionality"]),
        loss=np.asarray(energy["loss"]),
        jumps=run.read_table("jumps").to_dict("records"),
        inverse_density=float(energy["inverse_density"]),
        trans_steps=np.asarray(trans["steps"]),
        W=trans["W"].numpy(),
        stages=np.asarray(trans["stages"]),
        trans_loss=np.asarray(trans["loss"]),
        sets=np.asarray(trans["sets"]),
        table=run.read_table("metrics"),
    )


def trajectory_in_frame(dynamics: DynamicsRun, instance: int) -> np.ndarray:
    """One part-(b) run's snapshots ``(T′, 3, feat)``, rotated into the display frame."""
    from superposition.experiments.learning_dynamics import canonical_frame_3d

    rotation = canonical_frame_3d(dynamics.W[-1, instance], dynamics.sets)
    return np.einsum("ij,tjf->tif", rotation, dynamics.W[:, instance])


@dataclass
class NeuronModel:
    """One best model for the Neurons page: its stack-plot panel plus a summary row."""

    label: str
    panel: NeuronPanel
    summary: dict


def load_privileged_basis(run: RunDir) -> tuple[list[NeuronModel], pd.DataFrame, dict | None]:
    """E07: one :class:`NeuronModel` per sparsity (best restart), the summary table and the
    hidden-bias (ReLU sidestepping) report if the run has one (Paper: pp. 31–34)."""
    from superposition.experiments.common import GridResult, format_probability
    from superposition.experiments.privileged_basis import PrivilegedBasisConfig, neuron_counts

    config = from_dict(PrivilegedBasisConfig, yaml.safe_load(run.config_path.read_text()))
    grid = GridResult.from_checkpoint(run.load_checkpoint("relu_hidden"))
    summary = run.read_table("summary")
    models = []
    for cond, row in enumerate(summary.to_dict("records")):
        W, _, _ = grid.best_weights(cond)
        Wn = W.numpy()
        label = f"1−S = {format_probability(grid.condition_probability(cond))}"
        models.append(
            NeuronModel(
                label=label,
                panel=NeuronPanel(
                    title=label,
                    W=Wn,
                    superposition=metrics.superposition_score(W[None])[0].numpy(),
                    polysemantic=neuron_counts(W[None], config.thresholds)[0].numpy(),
                    order=neuron_order_by_top_feature(Wn),
                ),
                summary=row,
            )
        )
    variant = next(
        (c["value"] for c in run.read_acceptance()["criteria"] if c["id"] == "relu_sidestepping"),
        None,
    )
    return models, summary, variant


@dataclass
class AbsRun:
    """E08 for the Neurons page: part (a) weights and roles, part (b) panels and summary."""

    small_W1: np.ndarray  # (hidden, feat), normalized
    small_W2: np.ndarray  # (feat, hidden), normalized
    small_roles: list[str]
    sweep: list[NeuronModel]
    summary: pd.DataFrame
    n_hidden: int


def load_abs_computation(run: RunDir) -> AbsRun:
    """Load an E08 run (Paper: pp. 34–39), footnote-15 normalized."""
    from superposition.experiments.abs_computation import (
        AbsComputationConfig,
        TrainedAbs,
        neuron_feature_counts,
        side_roles,
    )
    from superposition.experiments.common import format_probability

    config = from_dict(AbsComputationConfig, yaml.safe_load(run.config_path.read_text()))
    thr = config.thresholds
    small = TrainedAbs(**run.load_checkpoint("n3_m6"))
    W1n, W2n = metrics.normalize_hidden_units(small.W1, small.W2)
    roles = side_roles(W1n[0].numpy(), W2n[0].numpy(), thr)
    sweep = TrainedAbs(**run.load_checkpoint("n100_m40"))
    S1, S2 = metrics.normalize_hidden_units(sweep.W1, sweep.W2)
    counts = neuron_feature_counts(S1, S2, thr)
    score = metrics.superposition_score(S1)
    summary = run.read_table("metrics")
    models = []
    for c, row in enumerate(summary.to_dict("records")):
        label = f"1−S = {format_probability(float(sweep.feature_probability[c]))}"
        W = S1[c].numpy()
        models.append(
            NeuronModel(
                label=label,
                panel=NeuronPanel(
                    title=label,
                    W=W,
                    superposition=score[c].numpy(),
                    polysemantic=counts[c].numpy(),
                    order=neuron_order_by_top_feature(W),
                ),
                summary=row,
            )
        )
    return AbsRun(
        small_W1=W1n[0].numpy(),
        small_W2=W2n[0].numpy(),
        small_roles=roles,
        sweep=models,
        summary=summary,
        n_hidden=int(sweep.W1.shape[1]),
    )


# --- Neurons page, E09 tab: roles and ablations (SPEC §8 page 7) ------------------------------


@dataclass
class InterpModel:
    """One analyzed E09 model: normalized weights, roles, motif neurons and ablations."""

    key: str
    shape: str
    feature_probability: float
    graded: bool
    W1n: np.ndarray  # (hidden, feat)
    W2n: np.ndarray  # (feat, hidden)
    labels: list[str]
    kinds: list[str]
    order: np.ndarray
    highlight: set[int]
    zero: np.ndarray  # (hidden, feat) Δ loss / mean-prediction loss, zero-ablated
    mean: np.ndarray  # (hidden, feat) the same, mean-ablated
    table: pd.DataFrame
    summary: dict


@dataclass
class MechInterpRun:
    """What the Neurons page's E09 tab shows, from a mech_interp_abs run (pp. 38–39)."""

    run_id: str
    summary: pd.DataFrame
    motifs: pd.DataFrame
    rebuild: pd.DataFrame
    models: list[InterpModel]


def load_mech_interp(run: RunDir) -> MechInterpRun:
    """Load an E09 run: every analyzed model with its role table and ablation matrices."""
    from superposition.experiments.common import format_probability
    from superposition.experiments.mech_interp_abs import TrainedShape
    from superposition.plotting.interp import motif_order

    summary = run.read_table("metrics")
    roles = run.read_table("roles")
    ablations = run.read_table("ablations")
    shapes: dict[str, TrainedShape] = {}
    models = []
    for row in summary.to_dict("records"):
        shape, p = str(row["shape"]), float(row["feature_probability"])
        if shape not in shapes:
            shapes[shape] = TrainedShape(**run.load_checkpoint(shape))
        trained = shapes[shape]
        k = int(row["instance"])
        W1n, W2n = metrics.normalize_hidden_units(trained.W1[k][None], trained.W2[k][None])
        table = roles.loc[
            ((roles["shape"] == shape) & (roles["feature_probability"] == p)).to_numpy()
        ]
        abl = ablations.loc[
            ((ablations["shape"] == shape) & (ablations["feature_probability"] == p)).to_numpy()
        ]
        grids = {
            kind: abl.loc[(abl["kind"] == kind).to_numpy()]
            .sort_values(["neuron", "feature"])["relative"]
            .to_numpy()
            .reshape(trained.n_hidden, trained.n_features)
            for kind in ("zero", "mean")
        }
        labels = [str(v) for v in table["role"]]
        kinds = [str(v) for v in table["kind"]]
        n, m = shape.split("_")
        models.append(
            InterpModel(
                key=f"n = {n[1:]}, m = {m[1:]}, 1 − S = {format_probability(p)}"
                + ("" if row["graded"] else " (context)"),
                shape=shape,
                feature_probability=p,
                graded=bool(row["graded"]),
                W1n=W1n[0].numpy(),
                W2n=W2n[0].numpy(),
                labels=labels,
                kinds=kinds,
                order=motif_order(kinds, labels, W1n[0].numpy()),
                highlight={int(h) for h in table.loc[table["in_motif"].to_numpy(bool), "neuron"]},
                zero=grids["zero"],
                mean=grids["mean"],
                table=table.drop(columns=["shape", "feature_probability", "graded"]),
                summary=row,
            )
        )
    return MechInterpRun(
        run_id=run.run_id,
        summary=summary,
        motifs=run.read_table("motifs"),
        rebuild=run.read_table("rebuild"),
        models=models,
    )


# --- SAE lab (SPEC §8 page 9): live SAEs, E10 and E11 --------------------------------------

# Toy models to decompose live: the pentagon and E01's sparse settings (pp. 2, 11).
SAE_TOY_PRESETS: dict[str, PlaygroundConfig] = {
    "pentagon · n = 5, m = 2, 1 − S = 0.1 (E00, p. 2)": PlaygroundConfig(),
    "n = 20, m = 5, 1 − S = 0.01 (E01, p. 11)": PlaygroundConfig(
        n_features=20, n_hidden=5, importance_decay=0.7, feature_probability=0.01
    ),
    "n = 20, m = 5, 1 − S = 0.001 (E01, p. 11)": PlaygroundConfig(
        n_features=20, n_hidden=5, importance_decay=0.7, feature_probability=0.001
    ),
}
SAE_LATENTS_RANGE = (2, 64)
SAE_STEPS_RANGE = (200, 20_000)
SAE_TOPK_RANGE = (0, 8)


@dataclass(frozen=True)
class LiveSAEConfig:
    """One live SAE job (SPEC §8 page 9). ``topk`` = 0 is a ReLU + L1 SAE."""

    n_latents: int = 10
    l1: float = 0.3
    topk: int = 0
    resample: bool = True
    steps: int = 3_000
    batch_size: int = 1024
    seed: int = 0

    def validate(self) -> None:
        for name, value, (low, high) in (
            ("n_latents", self.n_latents, SAE_LATENTS_RANGE),
            ("steps", self.steps, SAE_STEPS_RANGE),
            ("topk", self.topk, SAE_TOPK_RANGE),
        ):
            if not low <= value <= high:
                raise ValueError(f"{name}={value} is outside [{low}, {high}]")
        if self.l1 < 0:
            raise ValueError("l1 must be non-negative")


@dataclass
class LiveSAEResult:
    """A live SAE on one toy model's activations, scored against its true features."""

    decoder: np.ndarray  # (hidden, latent), unit columns
    alive: np.ndarray  # (latent,)
    max_cos: np.ndarray  # (feat,), NaN for unrepresented features
    best_latent: np.ndarray  # (feat,)
    represented: np.ndarray  # (feat,)
    recovered_09: float
    recovered_095: float
    mean_max_cos: float
    l0: float
    fvu: float
    dead_fraction: float
    resampled: int
    mse_history: np.ndarray
    log_steps: list[int]
    train_seconds: float


def train_live_sae(
    toy: PlaygroundResult, config: LiveSAEConfig, device: str = "cpu"
) -> LiveSAEResult:
    """Train one SAE on the best restart's hidden activations h = Wx (Paper: pp. 42–44)."""
    from superposition.data import IndependentSparse
    from superposition.sae import (
        SAETrainConfig,
        SparseAutoencoder,
        recovery,
        sae_stats,
        train_sae,
    )
    from superposition.utils import make_generator

    config.validate()
    torch_device = resolve_device(device)
    W = torch.as_tensor(toy.best_W, dtype=torch.float32)
    p = toy.config.feature_probability
    dist = IndependentSparse(p, 1, W.shape[1])

    def sample(batch: int, generator: torch.Generator) -> torch.Tensor:
        return dist.sample(batch, generator)[:, 0] @ W.T.to(generator.device)

    sae = SparseAutoencoder(
        W.shape[0], config.n_latents, 1, config.l1, config.topk, config.resample, seed=config.seed
    ).to(torch_device)
    train_config = SAETrainConfig(
        steps=config.steps,
        batch_size=config.batch_size,
        resample_every=max(1, config.steps // 5),
        log_every=max(1, config.steps // 100),
    )
    result = train_sae(sae, sample, train_config, seed=config.seed)
    h = sample(50_000, make_generator(config.seed + 1, torch_device))
    stats = sae_stats(sae, h)
    represented = metrics.represented_mask(W[None], REPRESENTED_THRESHOLD)[0]
    rec = recovery(sae.unit_decoder().cpu(), stats.alive, W, represented)
    return LiveSAEResult(
        decoder=sae.unit_decoder()[0].cpu().numpy(),
        alive=stats.alive[0].numpy(),
        max_cos=rec.max_cos[0].numpy(),
        best_latent=rec.best_latent[0].numpy(),
        represented=represented.numpy(),
        recovered_09=float(rec.fraction[0.9][0]),
        recovered_095=float(rec.fraction[0.95][0]),
        mean_max_cos=float(rec.mean_max_cos[0]),
        l0=float(stats.l0[0]),
        fvu=float(stats.fvu[0]),
        dead_fraction=float(stats.dead_fraction[0]),
        resampled=int(result.resampled[0]),
        mse_history=result.mse_history[:, 0].numpy(),
        log_steps=result.log_steps,
        train_seconds=result.wall_time_s,
    )


@dataclass
class SAERecoveryRun:
    """An E10 run for the SAE lab: the sweep table and, per toy model, its weights and SAEs."""

    run_id: str
    table: pd.DataFrame
    thresholds: dict[str, float]
    toys: dict[str, dict]


def load_sae_recovery(run: RunDir) -> SAERecoveryRun:
    """Load an E10 run (Paper: pp. 42–44)."""
    from superposition.experiments.sae_recovery import SAERecoveryConfig

    config = from_dict(SAERecoveryConfig, yaml.safe_load(run.config_path.read_text()))
    toys = {}
    for spec in config.toy_models:
        state = run.load_checkpoint(spec.label)
        toys[spec.label] = {
            "W": state["toy_W"].numpy(),
            "represented": state["represented"].numpy(),
            "decoder": state["sae_state"]["W_dec"].numpy(),
            "mask": state["sae_state"]["latent_mask"].numpy(),
            "alive": state["alive"].numpy(),
            "n_hidden": spec.n_hidden,
        }
    return SAERecoveryRun(
        run_id=run.run_id,
        table=run.read_table("metrics"),
        thresholds={s.label: s.cos_threshold for s in config.toy_models},
        toys=toys,
    )


@dataclass
class L1Run:
    """An E11 run for the SAE lab: the λ table and the best weights per λ and setup."""

    run_id: str
    table: pd.DataFrame
    n_hidden: dict[str, int]
    weights: dict[str, dict[float, tuple[np.ndarray, np.ndarray]]]


def load_l1(run: RunDir) -> L1Run:
    """Load an E11 run (Paper: p. 42)."""
    from superposition.experiments.l1_removes_superposition import L1Config, L1Sweep

    config = from_dict(L1Config, yaml.safe_load(run.config_path.read_text()))
    weights: dict[str, dict[float, tuple[np.ndarray, np.ndarray]]] = {}
    for setup in config.setups:
        sweep = L1Sweep(**run.load_checkpoint(setup.label))
        weights[setup.label] = {
            float(lam): (sweep.W[int(k)].numpy(), sweep.b[int(k)].numpy())
            for lam, k in zip(sweep.lambdas, sweep.best.tolist(), strict=True)
        }
    return L1Run(
        run_id=run.run_id,
        table=run.read_table("metrics"),
        n_hidden={s.label: s.n_hidden for s in config.setups},
        weights=weights,
    )


# --- results pages: Phase diagrams, Geometry, Non-uniform (SPEC §8 pages 3–5) ----------------


@dataclass
class PhaseProblem:
    """One problem size of an E02 run (n = 2, m = 1 or n = 3, m = 2), for the Phase diagrams
    page. Models are cell-major (cell = density index · n_importance + importance index), with
    ``models_per_cell`` consecutive models per cell. ``theory`` is the run's TheoryResult."""

    label: str
    n_features: int
    n_hidden: int
    importance: np.ndarray
    density: np.ndarray
    regime: np.ndarray
    W: np.ndarray
    b: np.ndarray
    loss: np.ndarray
    models_per_cell: int
    cells: pd.DataFrame
    theory: Any

    def grid(self, column: str) -> np.ndarray:
        """A per-cell column of the run's table as a ``(n_density, n_importance)`` array."""
        return np.asarray(self.cells[column]).reshape(len(self.density), len(self.importance))


@dataclass
class PhaseChangeRun:
    """An E02 run: one :class:`PhaseProblem` per problem size (Paper: pp. 14–16, 51–53)."""

    run_id: str
    problems: dict[str, PhaseProblem]
    discard_worst: int


def load_phase_change(run: RunDir) -> PhaseChangeRun:
    """Load an E02 run's trained grids, theoretical diagrams and per-cell table."""
    from superposition.experiments.phase_change import (
        PhaseChangeConfig,
        PhaseGridResult,
        TheoryResult,
        extra_feature_regime,
    )

    config = from_dict(PhaseChangeConfig, yaml.safe_load(run.config_path.read_text()))
    cells = run.read_table("cells")
    thr = config.thresholds
    problems = {}
    for setting in config.settings:
        grid = PhaseGridResult.from_checkpoint(run.load_checkpoint(f"{setting.label}__empirical"))
        theory = TheoryResult.from_checkpoint(run.load_checkpoint(f"{setting.label}__theory"))
        regime, _, _ = extra_feature_regime(
            grid.W, setting.n_features - 1, thr.represented_norm, thr.superposition_score
        )
        problems[setting.label] = PhaseProblem(
            label=setting.label,
            n_features=setting.n_features,
            n_hidden=setting.n_hidden,
            importance=np.asarray(grid.importance_grid),
            density=np.asarray(grid.density_grid),
            regime=regime,
            W=grid.W.numpy(),
            b=grid.b.numpy(),
            loss=grid.final_loss.numpy(),
            models_per_cell=grid.models_per_cell,
            cells=pd.DataFrame(cells[cells["setting"] == setting.label]).reset_index(drop=True),
            theory=theory,
        )
    return PhaseChangeRun(run.run_id, problems, config.discard_worst)


def phase_cell_models(problem: PhaseProblem, d: int, i: int, discard_worst: int) -> pd.DataFrame:
    """The models trained at cell (density ``d``, importance ``i``), lowest loss first, with
    their extra-feature regime and whether they vote (the highest-loss ones do not, p. 14)."""
    per_cell = problem.models_per_cell
    first = (d * len(problem.importance) + i) * per_cell
    order = first + np.argsort(problem.loss[first : first + per_cell], kind="stable")
    rows = []
    for rank, k in enumerate(order):
        rows.append(
            {
                "model": int(k - first),
                "loss": float(problem.loss[k]),
                "regime": str(problem.regime[k]),
                "votes": rank < per_cell - discard_worst,
                "‖W_extra‖": float(np.linalg.norm(problem.W[k][:, -1])),
                "W": np.round(problem.W[k], 3).tolist(),
                "b": np.round(problem.b[k], 3).tolist(),
            }
        )
    return pd.DataFrame(rows)


def phase_cell_theory(problem: PhaseProblem, d: int, i: int) -> pd.DataFrame:
    """Every family's optimal exact loss at one cell, lowest first, with its optimal W and b
    (with fitted biases; and the loss when every bias is fixed at 0, McGrath's setting)."""
    theory = problem.theory
    rows = [
        {
            "family": name,
            "regime": label,
            "paper family": f < len(theory.family_names) - 1,
            "optimal loss": float(theory.losses[d, i, f]),
            "loss with b = 0": float(theory.losses_no_bias[d, i, f]),
            "W": np.round(theory.W[f, d, i], 3).tolist(),
            "b": np.round(theory.b[f, d, i], 3).tolist(),
        }
        for f, (name, label) in enumerate(
            zip(theory.family_names, theory.family_labels, strict=True)
        )
    ]
    return pd.DataFrame(rows).sort_values("optimal loss", kind="stable").reset_index(drop=True)


def phase_cell_shared_losses(
    problem: PhaseProblem,
    d: int,
    i: int,
    discard_worst: int,
    n_samples: int = 1_000_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Every voting model of a cell and every paper family's optimum, scored exactly where
    possible and on the *same* ``n_samples`` inputs (common random numbers).

    A run's final losses come from independent 20,000-sample estimates, too noisy to compare
    models that differ by a few percent, and the lowest of several noisy estimates is biased
    low. The exact loss (theory.py) is used whenever every output of the model reads at most one
    other feature: always for the families, and for every trained n = 2, m = 1 model. Otherwise
    only the shared-sample estimate is available. It removes the noise of *differences* between
    models whose losses come from the same inputs, but not between models whose losses come from
    different rare events (at low density, pairing feature 0 or feature 1 with the extra feature
    is paid for on different double activations), so use enough samples there.

    Rows are sorted by exact loss where every row has one, else by the shared-sample loss.
    """
    from superposition.data import IndependentSparse
    from superposition.theory import relu_output_expected_loss as theory_exact_loss
    from superposition.utils import make_generator

    models = phase_cell_models(problem, d, i, discard_worst)
    voting = pd.DataFrame(models[models["votes"]])
    all_families = phase_cell_theory(problem, d, i)
    families = pd.DataFrame(all_families[all_families["paper family"]])
    W = np.array([*voting["W"], *families["W"]], dtype=np.float32)
    b = np.array([*voting["b"], *families["b"]], dtype=np.float32)
    n_inst = len(W)
    model = TiedModel.from_kind("relu_output", problem.n_features, problem.n_hidden, n_inst)
    with torch.no_grad():
        model.W.copy_(torch.from_numpy(W))
        model.b.copy_(torch.from_numpy(b))
    importance = torch.ones(problem.n_features)
    importance[-1] = float(problem.importance[i])
    dist = IndependentSparse(float(problem.density[d]), 1, problem.n_features)
    generator = make_generator(seed)
    chunk = min(n_samples, 100_000)
    total = torch.zeros(n_inst, dtype=torch.float64)
    done = 0
    with torch.no_grad():
        while done < n_samples:
            size = min(chunk, n_samples - done)
            x = dist.sample(size, generator).expand(-1, n_inst, -1)  # one input for all
            total += (importance * (x - model(x)) ** 2).sum(-1).double().sum(0)
            done += size
    shared = (total / n_samples).numpy()

    def exact(W: list, b: list) -> float:
        try:
            return theory_exact_loss(np.array(W), np.array(b), importance.numpy(), density)
        except NotImplementedError:  # an output reads two or more other features
            return float("nan")

    density = float(problem.density[d])
    rows = [
        {
            "source": f"trained model {int(r['model'])}",
            "regime": r["regime"],
            "exact loss": exact(r["W"], r["b"]),
            "shared-sample loss": float(shared[k]),
            "W": r["W"],
            "b": r["b"],
        }
        for k, r in enumerate(voting.to_dict("records"))
    ]
    rows += [
        {
            "source": f"theory {r['family']}",
            "regime": r["regime"],
            "exact loss": float(r["optimal loss"]),
            "shared-sample loss": float(shared[len(voting) + k]),
            "W": r["W"],
            "b": r["b"],
        }
        for k, r in enumerate(families.to_dict("records"))
    ]
    frame = pd.DataFrame(rows)
    key = "exact loss" if bool(frame["exact loss"].notna().all()) else "shared-sample loss"
    return frame.sort_values(by=key, kind="stable").reset_index(drop=True)


@dataclass
class UniformGeometryRun:
    """An E03 run for the Geometry page: per sparsity point, the best model's weights and its
    D*, Dᵢ, represented mask and tegum-factor census (Paper: pp. 17–20)."""

    run_id: str
    n_features: int
    n_hidden: int
    inverse_density: np.ndarray
    dstar: np.ndarray
    dimensionality: list[np.ndarray]
    represented: list[np.ndarray]
    census: list[dict[str, int]]
    sum_dimensionality: np.ndarray
    W: list[np.ndarray]
    plateau: tuple[float, float] | None
    thresholds: Any


def load_uniform_geometry(run: RunDir) -> UniformGeometryRun:
    """Load an E03 run and summarize its best restart at every sparsity."""
    from superposition.experiments.common import GridResult
    from superposition.experiments.uniform_geometry import (
        UniformGeometryConfig,
        plateau_span,
        summarize,
    )

    config = from_dict(UniformGeometryConfig, yaml.safe_load(run.config_path.read_text()))
    thr = config.thresholds
    grid = GridResult.from_checkpoint(run.load_checkpoint("relu_output"))
    summary = summarize(grid, thr)
    span = plateau_span(summary.inverse_density, summary.dstar, thr.plateau_value, thr.plateau_tol)
    x = summary.inverse_density
    return UniformGeometryRun(
        run_id=run.run_id,
        n_features=config.n_features,
        n_hidden=config.n_hidden,
        inverse_density=x,
        dstar=summary.dstar,
        dimensionality=summary.dimensionality,
        represented=summary.represented,
        census=summary.census,
        sum_dimensionality=summary.sum_dimensionality,
        W=[grid.best_weights(c)[0].numpy() for c in range(grid.n_conditions)],
        plateau=(float(x[span[0]]), float(x[span[1]])) if span else None,
        thresholds=thr,
    )


def geometry_graph_panel(
    geometry: UniformGeometryRun, point: int
) -> tuple[str, Any, dict[int, str]]:
    """``(title, graph, structure_by_node)`` of one sparsity point, for the p. 19 graph view."""
    from superposition.experiments.uniform_geometry import structure_by_node

    thr = geometry.thresholds
    W = torch.from_numpy(geometry.W[point])
    graph, _ = metrics.geometry_graph(W, thr.edge_threshold, thr.represented_norm)
    title = f"1/(1−S) = {geometry.inverse_density[point]:.2f}"
    return title, graph, structure_by_node(W, thr)


@dataclass
class NonuniformRun:
    """An E04 run for the Non-uniform page: the saved training states of parts (a)–(d) and the
    per-restart table of part (a) (Paper: pp. 22–25)."""

    run_id: str
    thresholds: Any
    perturbed: dict[str, Any]
    perturbed_table: pd.DataFrame
    correlation: dict[str, Any]
    collapse: dict[str, Any]


def load_nonuniform(run: RunDir) -> NonuniformRun:
    """Load an E04 run's checkpoints."""
    from superposition.experiments.nonuniform_geometry import NonuniformGeometryConfig

    config = from_dict(NonuniformGeometryConfig, yaml.safe_load(run.config_path.read_text()))
    return NonuniformRun(
        run_id=run.run_id,
        thresholds=config.thresholds,
        perturbed=run.load_checkpoint("a_perturbed"),
        perturbed_table=run.read_table("perturbed_restarts"),
        correlation=run.load_checkpoint("bc_correlation"),
        collapse=run.load_checkpoint("d_collapse"),
    )


def best_in_condition(state: dict[str, Any], condition: int) -> int:
    """Index of the lowest-loss instance trained under ``condition`` (a ratio or a density)."""
    members = torch.nonzero(state["condition"] == condition).flatten()
    return int(members[state["final_loss"][members].argmin()])


def canonical_2d(W: torch.Tensor) -> np.ndarray:
    """One 2D solution rotated and reflected into the canonical frame (p. 22)."""
    return metrics.canonicalize_2d(W[None])[0].numpy()
