"""Headless tests of the Streamlit lab pages (streamlit.testing AppTest; SPEC §8).

The pages run exactly as under ``streamlit run``: widgets, caching and all. Training is kept
tiny by pre-setting the Playground's widget state, so this stays in the fast tier.
"""

import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[2] / "app"


@pytest.fixture(autouse=True)
def app_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    # `streamlit run app/Home.py` puts app/ on sys.path; AppTest does not for pages/.
    monkeypatch.setattr(sys, "path", [str(APP), *sys.path])


def test_home_renders_status_and_paper_map() -> None:
    at = AppTest.from_file(str(APP / "Home.py"), default_timeout=60).run()
    assert not at.exception
    headers = [h.value for h in at.header]
    assert headers == ["Reproduction status", "Latest figures", "Lab pages", "Paper map"]
    assert len(at.dataframe) == 2
    # With no local runs (tests use an empty results dir) the page shows the command to run.
    assert any("uv run superposition run intro_pentagon" in c.value for c in at.code)


def playground(steps: int = 500, restarts: int = 2) -> AppTest:
    at = AppTest.from_file(str(APP / "pages" / "1_Playground.py"), default_timeout=120)
    at.session_state["pg_steps"] = steps
    at.session_state["pg_restarts"] = restarts
    return at.run()


def test_playground_trains_and_probes() -> None:
    at = playground()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert labels[:4] == [
        "best loss",
        "represented (‖Wᵢ‖ > 0.5)",
        "D* (dims per feature)",
        "superposition (p. 8)",
    ]
    assert len(at.dataframe) >= 2  # per-feature table and probe table (restarts table in expander)
    at.multiselect[0].set_value([1, 4]).run()
    assert not at.exception
    assert [s.label for s in at.slider if s.label.startswith("x")] == ["x1", "x4"]


def test_playground_preset_and_train_button() -> None:
    at = playground(steps=500)
    at.selectbox(key="pg_preset").set_value(
        "p. 32 · ReLU hidden layer, n = 10, m = 5, 1 − S = 0.1"
    ).run()
    assert at.session_state["pg_kind"] == "relu_hidden"
    assert at.session_state["pg_n"] == 10
    submit = next(b for b in at.button if b.label == "Train")
    submit.click().run()
    assert not at.exception
    assert any("ReLU hidden" in c.value for c in at.caption)


def test_playground_linear_shows_closed_form_optimum() -> None:
    at = AppTest.from_file(str(APP / "pages" / "1_Playground.py"), default_timeout=120)
    for key, value in {"pg_steps": 500, "pg_restarts": 1, "pg_kind": "linear"}.items():
        at.session_state[key] = value
    at.run()
    assert not at.exception
    assert "closed-form linear optimum" in [m.label for m in at.metric]


# --- results pages (Phase 4): Dynamics, Neurons, Adversarial ----------------------------------

RESULT_PAGES = {
    "5_Dynamics.py": ["learning_dynamics"],
    "6_Neurons.py": ["privileged_basis", "abs_computation"],
    "7_Adversarial.py": ["adversarial"],
}
TINY_TRAIN = {"steps": 200, "log_every": 50, "eval_samples": 1000}
# Tiny versions of each experiment: the pages only need a run with the right structure.
TINY_RUNS: dict[str, dict] = {
    "learning_dynamics": {
        "energy": {"n_features": 40, "n_hidden": 6, "n_restarts": 1, "loss_eval_samples": 256},
        "train": {**TINY_TRAIN, "snapshot_every": 50},
        "jumps": {"jump_window": 50, "stable_window": 100},
        "transformations": {
            "set_probabilities": [0.1],
            "n_restarts": 2,
            "loss_eval_samples": 500,
            "train": {**TINY_TRAIN, "snapshot_every": 50},
        },
        "stages": {"min_stage_steps": 50},
    },
    "adversarial": {
        "settings": [
            {"label": "tiny", "n_features": 12, "n_hidden": 3, "n_points": 3, "n_restarts": 1,
             "batch_size": None, "attack_samples": 200},
        ],
        "train": TINY_TRAIN,
    },
    "privileged_basis": {
        "feature_probabilities": [1.0, 0.1],
        "figure_probabilities": [1.0, 0.1],
        "n_restarts": 2,
        "hidden_bias": {"feature_probabilities": [1.0], "n_restarts": 1},
        "thresholds": {"activity_samples": 200},
        "train": TINY_TRAIN,
    },
    "abs_computation": {
        "no_superposition": {"n_restarts": 1},
        "sweep": {"n_features": 12, "n_hidden": 6, "feature_probabilities": [1.0, 0.01],
                  "n_restarts": 1},
        "bias_variant": {"feature_probabilities": [0.01], "n_restarts": 1},
        "thresholds": {"loss_samples": 200},
        "train": TINY_TRAIN,
    },
}  # fmt: skip


@pytest.fixture(scope="module")
def tiny_results(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A results folder holding one tiny complete run of each Phase 4 experiment."""
    from superposition.experiments.runner import run_experiment
    from superposition.runs import RESULTS_ENV

    root = tmp_path_factory.mktemp("results")
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv(RESULTS_ENV, str(root))
        for name, overrides in TINY_RUNS.items():
            run_experiment(name, "quick", device="cpu", overrides=overrides, echo=False)
    return root


@pytest.mark.parametrize("page", sorted(RESULT_PAGES))
def test_result_page_without_runs_shows_the_command(page: str) -> None:
    at = AppTest.from_file(str(APP / "pages" / page), default_timeout=60).run()
    assert not at.exception
    commands = [c.value for c in at.code]
    for name in RESULT_PAGES[page]:
        assert f"uv run superposition run {name} --profile quick" in commands


@pytest.mark.parametrize("page", sorted(RESULT_PAGES))
def test_result_page_renders_a_run(
    page: str, tiny_results: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from superposition.runs import RESULTS_ENV

    monkeypatch.setenv(RESULTS_ENV, str(tiny_results))
    at = AppTest.from_file(str(APP / "pages" / page), default_timeout=120).run()
    assert not at.exception
    assert not at.code  # nothing missing
    assert len(at.dataframe) >= 1


def test_dynamics_scrubber_moves(tiny_results: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from superposition.runs import RESULTS_ENV

    monkeypatch.setenv(RESULTS_ENV, str(tiny_results))
    at = AppTest.from_file(str(APP / "pages" / "5_Dynamics.py"), default_timeout=120).run()
    at.select_slider(key="dyn_step_b").set_value(0).run()
    assert not at.exception
    assert any("stage at step 0" in str(chart.proto) for chart in at.get("plotly_chart"))
