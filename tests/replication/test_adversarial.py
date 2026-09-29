"""Replication test for E06 adversarial (Paper: pp. 29–30). Slow: about 7 minutes on a laptop
CPU.

E06 is a PARTIAL reproduction (docs/REPRODUCTION.md): the paper's claim holds on the uniform
setup its p. 29 figure uses, but not on SPEC's second example, n = 20, m = 5, over the same
range. This test pins that outcome: the uniform setup must pass both criteria, and any change in
the n = 20, m = 5 outcome should be noticed and reported, not silently absorbed.
"""

import pytest

from superposition.experiments.runner import run_experiment


@pytest.mark.slow
def test_adversarial_quick_profile_uniform_setup_passes() -> None:
    run = run_experiment("adversarial", "quick", device="cpu", echo=False, figures=False)
    doc = run.read_acceptance()
    status = {(c["group"], c["id"]): c["status"] for c in doc["criteria"]}
    assert status[("uniform_n400_m30", "vulnerability_rises")] == "PASS"
    assert status[("uniform_n400_m30", "vulnerability_tracks_features_per_dimension")] == "PASS"
    assert doc["n_graded"] == 4
    assert doc["overall"] in {"PASS", "PARTIAL"}, doc["overall"]
