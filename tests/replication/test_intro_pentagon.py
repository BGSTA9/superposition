"""Replication test for E00 intro_pentagon (Paper: p. 2). Slow: trains 48 models.

Runs the quick profile end to end (on CPU, the SPEC's reference hardware) and requires every
graded acceptance criterion of SPEC §7 E00 to pass.
"""

import pytest

from superposition.experiments.runner import run_experiment


@pytest.mark.slow
def test_intro_pentagon_quick_profile_passes() -> None:
    run = run_experiment("intro_pentagon", "quick", device="cpu", echo=False, figures=False)
    doc = run.read_acceptance()
    failing = [(c["group"], c["id"], c["value"]) for c in doc["criteria"] if c["status"] == "FAIL"]
    assert doc["overall"] == "PASS", failing
    assert doc["n_graded"] == 11
    assert run.read_manifest()["status"] == "complete"
