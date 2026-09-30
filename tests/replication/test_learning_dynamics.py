"""Replication test for E05 learning_dynamics (Paper: pp. 26–28). Paper tier: about 9 minutes on a
laptop CPU (518 s measured for the quick profile: 4 × 80,000 steps of n = 400, m = 30, plus 48
small models).

Runs the quick profile end to end on CPU and requires every graded criterion of SPEC §7 E05
to pass: an energy-level jump with a coinciding loss drop, and an octahedron final state.
"""

import pytest

from superposition.experiments.runner import run_experiment


@pytest.mark.paper
def test_learning_dynamics_quick_profile_passes() -> None:
    run = run_experiment("learning_dynamics", "quick", device="cpu", echo=False, figures=False)
    doc = run.read_acceptance()
    failing = [(c["group"], c["id"], c["value"]) for c in doc["criteria"] if c["status"] == "FAIL"]
    assert doc["overall"] == "PASS", failing
    assert doc["n_graded"] == 2
    by_id = {c["id"]: c for c in doc["criteria"]}
    assert by_id["octahedron_final_state"]["value"]["fraction"] > 0
    assert by_id["stage_b_placement"]["status"] == "INFO"  # the documented variant (D-R02)
