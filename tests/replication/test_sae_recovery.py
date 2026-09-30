"""Replication test for E10 sae_recovery (Paper: pp. 42–44). Paper tier: about 7 minutes on a laptop
CPU (415 s measured for the quick profile).

Re-runs the quick profile from scratch. E10 grades the SAE a practitioner would pick without
ground truth (the sparsest with held-out FVU ≤ 0.1, D-R04); which models that SAE recovers is
left free to change status. The test requires what is robust: every graded sweep contains an
SAE that recovers the features (the oracle upper bound: the pentagon's five at cos > 0.95, and
≥ 90% of both sparse n = 20 models' at cos > 0.9), recovery at 1 − S = 0.01 is not a lone
lucky setting, and the selected SAE obeys the rule.
"""

import pytest

from superposition.experiments.runner import run_experiment

GRADED = ("pentagon", "n20_m5_p0.01", "n20_m5_p0.001")


@pytest.mark.paper
def test_sae_recovery_quick_profile() -> None:
    run = run_experiment("sae_recovery", "quick", device="cpu", echo=False, figures=False)
    doc = run.read_acceptance()
    by_id = {c["id"]: c for c in doc["criteria"]}
    oracle = by_id["oracle_upper_bound"]["value"]
    assert oracle["pentagon"]["recovered"] == 1.0
    for label in GRADED[1:]:
        assert oracle[label]["recovered"] >= 0.9, oracle[label]
    for label in GRADED:
        selected = by_id[f"{label}_selected_sae_recovers"]
        assert selected["status"] in {"PASS", "FAIL"}
        qualified, _ = map(int, selected["value"]["sae_meeting_fvu_bar"].split("/"))
        if qualified:
            assert selected["value"]["fvu"] <= 0.1, selected
    assert by_id["pentagon_selected_sae_recovers"]["value"]["represented"] == 5
    passing, total = map(int, by_id["pass_rates"]["value"]["n20_m5_p0.01"]["all"].split("/"))
    assert passing >= total // 2
    assert doc["n_graded"] == 3
    assert doc["overall"] in {"PASS", "PARTIAL", "FAIL"}
