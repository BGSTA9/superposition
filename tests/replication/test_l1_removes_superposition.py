"""Replication test for E11 l1_removes_superposition (Paper: p. 42). Slow: about 9 minutes on a
laptop CPU.

Re-runs the quick profile from scratch. The n = 20, m = 5 setup shows the paper's claim as
stated (superposition gone with several orthogonal features left), so all its criteria are
required. The pentagon's "falls to none" passes only in a narrow λ window with one feature left
(docs/REPRODUCTION.md, D-505); this test requires its other criteria and leaves that one free to
change status, so that a change there shows up in the acceptance record and the docs.
"""

import pytest

from superposition.experiments.runner import run_experiment


@pytest.mark.slow
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
