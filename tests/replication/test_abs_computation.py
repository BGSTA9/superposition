"""Replication test for E08 abs_computation (Paper: pp. 34–39). Slow: about 6 minutes on a
laptop CPU.

E08 is a PARTIAL reproduction (docs/REPRODUCTION.md): part (a), side neurons computing |x|
exactly, reproduces; part (b) reproduces the neuron picture of p. 37 but not computation of
more than m/2 features at the sparsest setting (dead output sides at initialization, D-409).
This test requires part (a) to pass and part (b)'s neurons to be monosemantic when dense up to
at most one neuron, and leaves the sparse capacity criterion free to change status: a fix
should show up in the acceptance record and the docs, not break this test.
"""

import pytest

from superposition.experiments.runner import run_experiment


@pytest.mark.slow
def test_abs_computation_quick_profile() -> None:
    run = run_experiment("abs_computation", "quick", device="cpu", echo=False, figures=False)
    doc = run.read_acceptance()
    by_id = {c["id"]: c for c in doc["criteria"]}
    assert by_id["a_positive_and_negative_side_neurons"]["status"] == "PASS"
    assert by_id["a_per_feature_loss"]["status"] == "PASS"
    assert by_id["b_dense_monosemantic"]["value"]["monosemantic"] >= 39
    assert doc["n_graded"] == 4
    assert doc["overall"] in {"PASS", "PARTIAL"}, doc["overall"]
