"""Replication test for E09 mech_interp_abs (Paper: pp. 38–39). Slow: about 5 minutes on a
laptop CPU.

Re-runs the quick profile from scratch and requires every graded criterion: at least 90% of
live neurons named, ablations agreeing with the roles, and the asymmetric superposition +
inhibition motif found in at least one model.
"""

import pytest

from superposition.experiments.runner import run_experiment


@pytest.mark.slow
def test_mech_interp_abs_quick_profile() -> None:
    run = run_experiment("mech_interp_abs", "quick", device="cpu", echo=False, figures=False)
    doc = run.read_acceptance()
    by_id = {c["id"]: c for c in doc["criteria"]}
    for criterion in ("named_roles", "ablations_agree", "motif_found"):
        assert by_id[criterion]["status"] == "PASS", by_id[criterion]
    assert by_id["motif_found"]["value"]["models_with_motif"] >= 1
    assert doc["overall"] == "PASS"
