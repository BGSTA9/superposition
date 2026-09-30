"""Replication tests for E02 phase_change (Paper: pp. 14–16, 51–53).

The slow test trains a reduced n = 2, m = 1 grid (4 × 4 cells, ten models each, about a minute
on a laptop CPU) and checks the claim that does not depend on where exactly the boundaries lie:
at its corners the vote finds the paper's three regions, where the theory does. The paper-tier
test re-runs the whole quick profile and pins the cited outcome (docs/REPRODUCTION.md).
"""

import pandas as pd
import pytest

from superposition import theory
from superposition.experiments.runner import run_experiment

REDUCED = {
    "settings": [{"label": "n2_m1", "n_features": 2, "n_hidden": 1, "paper_page": "p. 15"}],
    "grid": {"n_importance": 4, "n_density": 4},
}


@pytest.mark.slow
def test_phase_change_reduced_grid_finds_the_three_regions() -> None:
    run = run_experiment(
        "phase_change", "quick", device="cpu", overrides=REDUCED, echo=False, figures=False
    )
    cells = run.read_table("cells")
    dense = pd.DataFrame(cells[cells["density"] == cells["density"].max()])
    sparse = pd.DataFrame(cells[cells["density"] == cells["density"].min()])
    dense, sparse = dense.sort_values(by="importance"), sparse.sort_values(by="importance")
    # Dense (p. 15, top row): unimportant extra feature dropped, important one dedicated.
    assert dense.iloc[0]["empirical"] == dense.iloc[0]["theory"] == theory.NOT_LEARNED
    assert dense.iloc[-1]["empirical"] == dense.iloc[-1]["theory"] == theory.DEDICATED
    # Sparse (p. 15, bottom row): stored in superposition at every importance.
    assert set(sparse["theory"]) == {theory.SUPERPOSITION}
    assert (sparse["empirical"] == theory.SUPERPOSITION).mean() >= 0.75
