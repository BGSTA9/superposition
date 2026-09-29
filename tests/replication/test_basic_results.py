"""Replication test for E01 basic_results (Paper: pp. 10–12). Slow: ≈ 8 minutes on a CPU.

Runs the quick profile end to end on CPU and requires every graded criterion of SPEC §7 E01,
as operationalized in docs/DECISIONS.md (D-105), to pass for both model sizes.
"""

import pytest

from superposition.experiments.runner import run_experiment


@pytest.mark.slow
def test_basic_results_quick_profile_passes() -> None:
    run = run_experiment("basic_results", "quick", device="cpu", echo=False, figures=False)
    doc = run.read_acceptance()
    failing = [(c["group"], c["id"], c["value"]) for c in doc["criteria"] if c["status"] == "FAIL"]
    assert doc["overall"] == "PASS", failing
    assert doc["n_graded"] == 12  # six criteria for each of the two model sizes
