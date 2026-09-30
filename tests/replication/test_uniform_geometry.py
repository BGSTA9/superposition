"""Replication tests for E03 uniform_geometry (Paper: pp. 16–21).

The slow test trains the p. 17 model (n = 400, m = 30) at two sparsities only, one restart
each (about a minute on a laptop CPU): D* must be 1 when features are dense and on the 1/2
plateau at 1/(1 − S) = 3, with Σᵢ Dᵢ close to m. The paper-tier test re-runs the whole quick
profile and pins the cited outcome (docs/REPRODUCTION.md).
"""

import pytest

from superposition.experiments.runner import run_experiment

REDUCED = {"inverse_density_min": 1.0, "inverse_density_max": 3.0, "n_points": 2, "n_restarts": 1}


@pytest.mark.slow
def test_uniform_geometry_dense_and_plateau_points() -> None:
    run = run_experiment(
        "uniform_geometry", "quick", device="cpu", overrides=REDUCED, echo=False, figures=False
    )
    sums = next(c for c in run.read_acceptance()["criteria"] if c["id"] == "sum_dimensionality")
    (dense_x, dense), (plateau_x, plateau) = sorted(
        sums["value"].items(), key=lambda kv: float(kv[0])
    )
    assert float(dense_x) == pytest.approx(1.0) and float(plateau_x) == pytest.approx(3.0)
    assert dense["D_star"] >= 0.95
    assert abs(plateau["D_star"] - 0.5) <= 0.03
    assert abs(plateau["sum_D"] - 30) <= 1.0


@pytest.mark.paper
def test_uniform_geometry_quick_profile_passes() -> None:
    run = run_experiment("uniform_geometry", "quick", device="cpu", echo=False, figures=False)
    doc = run.read_acceptance()
    failing = [(c["id"], c["value"]) for c in doc["criteria"] if c["status"] == "FAIL"]
    assert doc["overall"] == "PASS", failing
    assert doc["n_graded"] == 4
