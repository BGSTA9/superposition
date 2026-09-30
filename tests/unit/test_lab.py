"""Unit tests for superposition.lab (the computation behind the Streamlit lab, SPEC §8)."""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from superposition import lab
from superposition.models import TiedModel


def small(**overrides: int) -> lab.PlaygroundConfig:
    """A quick-to-train config: 200 steps, 2 restarts, 2,000 evaluation samples."""
    return lab.config_from_dict({"steps": 200, "restarts": 2, "eval_samples": 2_000, **overrides})


def test_config_validation() -> None:
    lab.PlaygroundConfig().validate()
    for bad in (
        {"n_features": 41},
        {"n_hidden": 0},
        {"importance_decay": 0.4},
        {"feature_probability": 0.0},
        {"model_kind": "tanh"},
        {"steps": 10},
        {"restarts": 0},
        {"seed": -1},
    ):
        with pytest.raises(ValueError):
            lab.config_from_dict({**lab.config_to_dict(lab.PlaygroundConfig()), **bad})


def test_train_playground_shapes_best_and_determinism() -> None:
    config = small(n_features=6, n_hidden=3)
    result = lab.train_playground(config)
    assert result.W.shape == (2, 3, 6)
    assert result.b.shape == (2, 6)
    assert result.final_loss.shape == (2,)
    assert result.best == int(np.argmin(result.final_loss))
    assert result.loss_history.shape[1] == 2
    assert result.importance[0] == 1.0
    assert result.device == "cpu"
    again = lab.train_playground(config)
    np.testing.assert_array_equal(result.W, again.W)
    model = result.best_model()
    assert model.n_inst == 1
    np.testing.assert_allclose(model.W.detach().numpy()[0], result.best_W)


def pentagon() -> np.ndarray:
    angles = 2 * np.pi * np.arange(5) / 5
    return np.stack([np.cos(angles), np.sin(angles)])


def test_feature_table_on_a_pentagon() -> None:
    table = lab.feature_table(pentagon(), np.full(5, -0.2), np.ones(5))
    assert list(table["feature"]) == [0, 1, 2, 3, 4]
    np.testing.assert_allclose(table["Dᵢ"], 0.4, atol=1e-5)
    np.testing.assert_allclose(table["norm ‖Wᵢ‖"], 1.0, atol=1e-6)
    assert set(table["nearest fraction"]) == {"pentagon"}
    assert bool(table["represented"].all()) and bool(table["in superposition"].all())


def test_summary() -> None:
    config = small()
    result = lab.train_playground(config)
    result.W[result.best] = pentagon()
    numbers = lab.summary(result)
    assert numbers["represented features"] == 5
    assert numbers["dimensions per feature D*"] == pytest.approx(0.4, abs=1e-5)
    assert numbers["superposition (WᵀW singular on represented)"] is True


def one_instance(kind: str, W: np.ndarray, b: np.ndarray) -> TiedModel:
    model = TiedModel.from_kind(kind, W.shape[1], W.shape[0], n_inst=1)  # type: ignore[arg-type]
    with torch.no_grad():
        model.W.copy_(torch.as_tensor(W, dtype=torch.float32)[None])
        model.b.copy_(torch.as_tensor(b, dtype=torch.float32)[None])
    return model


@pytest.mark.parametrize("kind", ["linear", "relu_output", "relu_hidden"])
def test_probe_stages_match_the_model_equations(kind: str) -> None:
    W = pentagon() * 1.1
    b = np.full(5, -0.2)
    x = np.array([0.0, 1.0, 0.0, 0.0, 0.7])
    result = lab.probe(one_instance(kind, W, b), x)
    pre = W @ x
    h = np.maximum(pre, 0) if kind == "relu_hidden" else pre
    readout = W.T @ h
    output = readout + b if kind == "linear" else np.maximum(readout + b, 0)
    np.testing.assert_allclose(result.hidden_pre, pre, atol=1e-5)
    np.testing.assert_allclose(result.hidden, h, atol=1e-5)
    np.testing.assert_allclose(result.readout, readout, atol=1e-5)
    np.testing.assert_allclose(result.readout_plus_bias, readout + b, atol=1e-5)
    np.testing.assert_allclose(result.output, output, atol=1e-5)
    np.testing.assert_allclose(result.target, x, atol=1e-7)
    frame = result.feature_frame(np.ones(5))
    assert frame["weighted sq. error"].sum() == pytest.approx(
        float(((output - x) ** 2).sum()), rel=1e-5
    )


def test_probe_rejects_multi_instance_models() -> None:
    with pytest.raises(ValueError):
        lab.probe(TiedModel(5, 2, n_inst=2), np.zeros(5))


def test_paper_map_parses_the_repository_table(tmp_path: Path) -> None:
    table = lab.paper_map()
    assert list(table.columns) == [
        "Paper section",
        "Pages",
        "Code",
        "Experiment",
        "Notebook",
        "App page",
    ]
    assert len(table) >= 20
    assert "`" not in "".join(table.to_numpy().ravel())
    md = tmp_path / "map.md"
    md.write_text("# Map\n\n| a | b |\n|---|---|\n| `x` | 1 |\n| y | 2 |\n\ntext\n")
    parsed = lab.paper_map(md)
    assert parsed.to_dict("list") == {"a": ["x", "y"], "b": ["1", "2"]}


def test_status_frame_without_runs() -> None:
    # conftest points SUPERPOSITION_RESULTS at an empty temporary directory.
    frame = lab.status_frame()
    assert list(frame["experiment"])[:2] == ["intro_pentagon", "basic_results"]
    assert set(frame["latest local run"]) == {"none"}
    assert all(cmd.startswith("uv run superposition run ") for cmd in frame["command"])
    assert not math.isnan(len(frame))


def test_phase_cell_models_orders_by_loss_and_marks_the_non_voter() -> None:
    """E02's vote (p. 14): per cell, the highest-loss model is discarded."""
    per_cell, n_dens, n_imp = 3, 2, 2
    n = per_cell * n_dens * n_imp
    loss = np.arange(n, dtype=float)[::-1].copy()  # later models have lower losses
    problem = lab.PhaseProblem(
        label="n2_m1",
        n_features=2,
        n_hidden=1,
        importance=np.array([0.5, 2.0]),
        density=np.array([0.1, 1.0]),
        regime=np.array(["superposition", "dedicated", "not_learned"] * (n // 3)),
        W=np.zeros((n, 1, 2)),
        b=np.zeros((n, 2)),
        loss=loss,
        models_per_cell=per_cell,
        cells=pd.DataFrame(),
        theory=None,
    )
    models = lab.phase_cell_models(problem, d=1, i=0, discard_worst=1)  # cell 2: models 6–8
    assert list(models["model"]) == [2, 1, 0]
    assert list(models["loss"]) == sorted(models["loss"])
    assert list(models["votes"]) == [True, True, False]
    assert list(models["regime"]) == ["not_learned", "dedicated", "superposition"]
