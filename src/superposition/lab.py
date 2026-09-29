"""Computation behind the interactive lab: live Playground training, feature tables, probes,
and loaders for the results pages (Dynamics, Neurons, Adversarial).

SPEC §8 (The interactive lab). Paper: Demonstrating Superposition, pp. 9–12 (the models and
the WᵀW / ‖Wᵢ‖ views), Feature Dimensionality, pp. 18–19 (the Dᵢ table), Superposition in a
Privileged Basis, p. 31 (the ReLU hidden model); Learning Dynamics, pp. 26–27; Adversarial
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
