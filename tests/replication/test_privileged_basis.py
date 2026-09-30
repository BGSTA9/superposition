"""Replication test for E07 privileged_basis (Paper: pp. 31–34). Paper tier: about 7 minutes on a
laptop CPU (800 ReLU hidden layer models plus the hidden-bias variant).

Runs the quick profile end to end on CPU and requires every graded criterion of SPEC §7 E07
to pass, and checks the documented ReLU sidestepping (p. 34): with a hidden bias every bias
ends positive and the ReLU almost never clips.
"""

import pytest

from superposition.experiments.runner import run_experiment


@pytest.mark.paper
def test_privileged_basis_quick_profile_passes() -> None:
    run = run_experiment("privileged_basis", "quick", device="cpu", echo=False, figures=False)
    doc = run.read_acceptance()
    failing = [(c["group"], c["id"], c["value"]) for c in doc["criteria"] if c["status"] == "FAIL"]
    assert doc["overall"] == "PASS", failing
    assert doc["n_graded"] == 3
    sidestep = next(c for c in doc["criteria"] if c["id"] == "relu_sidestepping")["value"]
    for row in sidestep.values():
        assert row["hidden_biases_positive"] == "5/5"
        assert row["clipped_with_bias"] < 0.01
