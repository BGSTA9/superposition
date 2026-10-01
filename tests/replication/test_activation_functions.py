"""Replication tests for E12 activation_functions (extension; Paper: p. 51, with pp. 14–16).

E12 has no pass/fail (SPEC §7 E12). The slow test trains a reduced grid for two activations
and checks what must hold whatever the activations do: the ReLU column behaves like E02 at the
grid's corners, every activation gets every report row, and the theory's quadrature agrees
with its re-evaluation. The paper-tier test re-runs the whole quick profile.
"""

import pandas as pd
import pytest

from superposition import theory
from superposition.experiments.runner import run_experiment

REDUCED = {
    "activations": ["relu", "gelu"],
    "grid": {"n_importance": 4, "n_density": 4},
    "theory": {"scale_grid": 21, "bias_grid": 61, "golden_iterations": 20},
}


def _rows(doc: dict) -> dict:
    return {(c["group"], c["id"]): c for c in doc["criteria"]}


@pytest.mark.slow
def test_activation_functions_reduced_grid() -> None:
    run = run_experiment(
        "activation_functions", "quick", device="cpu", overrides=REDUCED, echo=False
    )
    doc = run.read_acceptance()
    assert doc["overall"] == "INFO" and doc["n_graded"] == 0
    rows = _rows(doc)
    for activation in REDUCED["activations"]:
        group = f"n2_m1 · {activation}"
        for key in ("regions", "agreement_with_own_theory", "losses", "quadrature_check"):
            assert (group, key) in rows
        assert rows[(group, "quadrature_check")]["value"]["cells_changing_label"] == 0
    assert ("n2_m1 · gelu", "vs_reference") in rows
    cells = run.read_table("cells")
    relu = pd.DataFrame(cells[cells["activation"] == "relu"])
    dense = pd.DataFrame(relu[relu["density"] == relu["density"].max()])
    dense = dense.sort_values(by="importance")
    sparse = relu[relu["density"] == relu["density"].min()]
    # E02 (p. 15) at the corners: dropped / dedicated when dense, superposed when sparse.
    assert dense.iloc[0]["theory"] == theory.NOT_LEARNED
    assert dense.iloc[-1]["theory"] == theory.DEDICATED
    assert set(sparse["theory"]) == {theory.SUPERPOSITION}
    assert sorted(p.name for p in run.figures_dir.glob("*.png")) == [
        "diagrams_n2_m1.png",
        "shares_n2_m1.png",
    ]


@pytest.mark.paper
def test_activation_functions_quick_profile() -> None:
    """The whole quick profile: six activations, every row reported, the quadrature check
    within 1e-6 (relative) of every family optimum (D-601)."""
    run = run_experiment("activation_functions", "quick", device="cpu", echo=False)
    doc = run.read_acceptance()
    assert doc["overall"] == "INFO"
    rows = _rows(doc)
    shares = rows[("n2_m1", "superposition_share")]["value"]["shares"]
    assert sorted(shares) == sorted(["relu", "gelu", "silu", "softplus", "leaky_relu", "sigmoid"])
    for activation in shares:
        check = rows[(f"n2_m1 · {activation}", "quadrature_check")]["value"]
        assert check["max_rel_change"] < 1e-6 and check["cells_changing_label"] == 0
