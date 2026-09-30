"""Replication test for E11 l1_removes_superposition (Paper: p. 42). Paper tier: about 9 minutes on a
laptop CPU (548 s measured for the quick profile).

Re-runs the quick profile from scratch. The n = 20, m = 5 setup shows the paper's claim as
stated (superposition gone with several orthogonal features left), so all its criteria are
required. The pentagon is free of superposition only with one feature left, in a narrow λ
window, which scores "falls to none" as PARTIAL (D-505, D-R05); on a held-out seed its λ
ordering is not stable either (docs/REPRODUCTION.md). This test requires the pentagon's other
criteria on the cited seed and leaves "falls to none" free to change status, so that a change
there shows up in the acceptance record and the docs.
"""

import pytest

from superposition.experiments.runner import run_experiment


@pytest.mark.paper
def test_l1_removes_superposition_quick_profile() -> None:
    run = run_experiment(
        "l1_removes_superposition", "quick", device="cpu", echo=False, figures=False
    )
    doc = run.read_acceptance()
    by_id = {c["id"]: c for c in doc["criteria"]}
    for criterion in (
        "n20_m5_superposition_at_zero",
        "n20_m5_superposition_declines",
        "n20_m5_superposition_falls_to_none",
        "n20_m5_loss_rises",
        "pentagon_superposition_at_zero",
        "pentagon_superposition_declines",
    ):
        assert by_id[criterion]["status"] == "PASS", by_id[criterion]
    assert by_id["n20_m5_superposition_falls_to_none"]["value"]["represented"] >= 2
    assert doc["overall"] in {"PASS", "PARTIAL"}, doc["overall"]
