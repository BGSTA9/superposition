"""Replication test for E10 sae_recovery (Paper: pp. 42–44). Paper tier: about 7 minutes on a laptop
CPU.

Re-runs the quick profile from scratch and requires every graded criterion: the pentagon's
five features recovered at cos > 0.95, and ≥ 90% of the represented features of both sparse
n = 20, m = 5 models at cos > 0.9, by the best SAE of each sweep. It also checks that the
recovery is not a lone lucky setting on the models where it is robust.
"""

import pytest

from superposition.experiments.runner import run_experiment


@pytest.mark.paper
def test_sae_recovery_quick_profile() -> None:
    run = run_experiment("sae_recovery", "quick", device="cpu", echo=False, figures=False)
    doc = run.read_acceptance()
    by_id = {c["id"]: c for c in doc["criteria"]}
    for criterion in ("pentagon_recovered", "n20_m5_p0.01_recovered", "n20_m5_p0.001_recovered"):
        assert by_id[criterion]["status"] == "PASS", by_id[criterion]
    assert by_id["pentagon_recovered"]["value"]["represented"] == 5
    passing, total = map(
        int, by_id["n20_m5_p0.01_recovered"]["value"]["settings_reaching_threshold"].split("/")
    )
    assert passing >= total // 2
    assert doc["overall"] == "PASS"
