"""Replication test for E08 abs_computation (Paper: pp. 34–39). Paper tier: about 8 minutes on a
laptop CPU (493 s measured for the quick profile).

E08 is a PARTIAL reproduction (docs/REPRODUCTION.md): part (a), side neurons computing |x|
exactly, reproduces; part (b) reproduces the neuron picture of p. 37 but not computation in
superposition, under the criterion re-derived from p. 37 (more than m features usefully
computed, D-R01) or SPEC's original one (more than m/2 computed well, reported). This test
requires part (a) to pass, part (b)'s neurons to be monosemantic when dense up to at most one
neuron, and both sparse numbers to be reported; it leaves the sparse criterion free to change
status, so that a fix shows up in the acceptance record and the docs, not as a broken test.
"""

import pytest

from superposition.experiments.runner import run_experiment


@pytest.mark.paper
def test_abs_computation_quick_profile() -> None:
    run = run_experiment("abs_computation", "quick", device="cpu", echo=False, figures=False)
    doc = run.read_acceptance()
    by_id = {c["id"]: c for c in doc["criteria"]}
    assert by_id["a_positive_and_negative_side_neurons"]["status"] == "PASS"
    assert by_id["a_per_feature_loss"]["status"] == "PASS"
    assert by_id["b_dense_monosemantic"]["value"]["monosemantic"] >= 39
    sparse = by_id["b_sparse_more_features_than_neurons"]
    original = by_id["b_sparse_original_criterion"]
    # Computed well (< 50%) implies usefully computed (< 75%).
    assert sparse["value"]["usefully_computed"] >= original["value"]["computed_well"]
    for variant in ("b_bias_init_variant", "b_init_scale_variant"):
        assert by_id[variant]["status"] == "INFO"
    assert doc["n_graded"] == 4
    assert doc["overall"] in {"PASS", "PARTIAL"}, doc["overall"]
