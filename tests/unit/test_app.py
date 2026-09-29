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
