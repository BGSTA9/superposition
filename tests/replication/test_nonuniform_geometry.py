"""Replication tests for E04 nonuniform_geometry (Paper: pp. 21–25).

The slow test trains a reduced version (about a minute on a laptop CPU): part (a) at the
sparsest and the uniform density ratio only, and part (b)'s correlated and anticorrelated
pairs. It checks the claims that do not depend on a fine sweep: the pentagon at the uniform
ratio, the collapse to two digons when the perturbed feature is 10× sparser (p. 22), and
anticorrelated pairs interfering more negatively than correlated ones (p. 24). The paper-tier
test re-runs the whole quick profile and pins the cited outcome (docs/REPRODUCTION.md).
"""

import pytest

from superposition.experiments.runner import run_experiment

REDUCED = {
    "perturbed": {"ratio_min": 0.1, "ratio_max": 1.0, "n_ratios": 2, "n_restarts": 16},
    "correlation": {"local_basis_restarts": 2},
    "collapse": {"n_densities": 2},
}


@pytest.mark.slow
def test_nonuniform_geometry_collapse_and_anticorrelation() -> None:
    run = run_experiment(
        "nonuniform_geometry", "quick", device="cpu", overrides=REDUCED, echo=False, figures=False
    )
    by_id = {c["id"]: c for c in run.read_acceptance()["criteria"]}
    best = by_id["collapse_to_digons"]["value"]["best_geometry_by_ratio"]
    assert best == {"0.1": "digons (perturbed dropped)", "1": "pentagon"}
    assert by_id["collapse_to_digons"]["status"] == "PASS"
    assert by_id["anticorrelated_more_negative"]["status"] == "PASS"
