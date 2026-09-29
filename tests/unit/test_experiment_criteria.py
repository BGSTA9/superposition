"""The acceptance logic of E00 and E01, checked on hand-built geometries (no training).

These tests pin down what each criterion *means*: an ideal paper solution must pass, and a
solution that breaks the claim must fail.
"""

import math

import torch

from superposition.acceptance import FAIL, PASS
from superposition.data import geometric_importance
from superposition.experiments.basic_results import (
    BasicResultsConfig,
    SizeSetting,
    evaluate_setting,
)
from superposition.experiments.common import GridResult
from superposition.experiments.intro_pentagon import (
    IntroPentagonConfig,
    all_cyclic_orders,
    cyclic_order,
    evaluate,
    sorted_angular_gaps_deg,
)
from superposition.theory import linear_optimum_loss


def grid_from(
    kind: str,
    weights: list[torch.Tensor],
    biases: list[torch.Tensor],
    losses: list[float],
    probs: list[float],
    importance: torch.Tensor,
) -> GridResult:
    """One instance per condition (a single 'restart')."""
    n_cond = len(weights)
    return GridResult(
        kind=kind,
        W=torch.stack(weights),
        b=torch.stack(biases),
        final_loss=torch.tensor(losses),
        feature_probability=torch.tensor(probs),
        condition=torch.arange(n_cond),
        restart=torch.zeros(n_cond, dtype=torch.long),
        best=torch.arange(n_cond),
        importance=importance,
        loss_history=torch.zeros(1, n_cond),
        log_steps=[0],
        wall_time_s=0.0,
    )


def ideal_e00_grids(config: IntroPentagonConfig) -> dict[str, GridResult]:
    n = config.n_features
    importance = geometric_importance(n, config.importance_decay)
    probs = [1 - s for s in config.sparsities]
    zero = torch.zeros(2, n)
    dedicated = zero.clone()
    dedicated[0, 0], dedicated[1, 1] = 1.0, 1.0
    pairs = zero.clone()
    pairs[:, 0], pairs[:, 2] = torch.tensor([0.0, 1.0]), torch.tensor([0.0, -1.0])
    pairs[:, 1], pairs[:, 3] = torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])
    angles = 2 * math.pi * torch.arange(n) / n
    pentagon = 1.1 * torch.stack([angles.cos(), angles.sin()])
    mean_bias = [torch.tensor([0.0, 0.0] + [p / 2] * (n - 2)) for p in probs]
    relu_biases = [
        mean_bias[0],
        torch.tensor([0.0, 0.0, 0.0, 0.0, probs[1] / 2]),
        torch.full((n,), -0.2),
    ]
    optimum = [linear_optimum_loss(importance, 2, p) for p in probs]
    return {
        "relu_output": grid_from(
            "relu_output",
            [dedicated, pairs, pentagon],
            relu_biases,
            [0.18, 0.06, 0.02],
            probs,
            importance,
        ),
        "linear": grid_from("linear", [dedicated] * 3, mean_bias, optimum, probs, importance),
    }


def by_id(criteria: list) -> dict:
    return {(c.group, c.id): c for c in criteria}


def test_e00_ideal_solutions_pass() -> None:
    config = IntroPentagonConfig()
    results = by_id(evaluate(config, ideal_e00_grids(config)))
    graded = {key: c.status for key, c in results.items() if c.status != "INFO"}
    assert len(graded) == 11
    assert set(graded.values()) == {PASS}, graded


def test_e00_broken_solutions_fail() -> None:
    config = IntroPentagonConfig()
    grids = ideal_e00_grids(config)
    relu = grids["relu_output"]
    relu.W[2, :, 4] = relu.W[2, :, 3] * 1.0 + 0.3  # squash the pentagon: gaps leave 72° ± 10°
    relu.b[2, 0] = 0.05  # one positive bias
    grids["linear"].final_loss[1] *= 1.10  # 10% above the closed-form optimum
    results = by_id(evaluate(config, grids))
    assert results[("relu_output", "S0.9_pentagon")].status == FAIL
    assert results[("linear", "S0.8_loss_vs_closed_form")].status == FAIL
    assert results[("relu_output", "S0_dedicated_orthogonal_pair")].status == PASS


def test_pentagon_helpers() -> None:
    angles = 2 * math.pi * torch.arange(5) / 5
    W = torch.stack([angles.cos(), angles.sin()])
    assert all(abs(g - 72) < 1e-4 for g in sorted_angular_gaps_deg(W))
    assert cyclic_order(W) == (0, 1, 2, 3, 4)
    assert cyclic_order(W[:, [0, 2, 1, 3, 4]]) in all_cyclic_orders(5)
    assert len(all_cyclic_orders(5)) == 12  # (5 − 1)!/2 arrangements


def test_e01_superposition_order_criterion() -> None:
    """Superposition must start with the least important represented features (p. 11)."""
    n, m = 6, 3
    setting = SizeSetting(label="toy", n_features=n, n_hidden=m, importance_decay=0.7)
    config = BasicResultsConfig(
        settings=[setting], feature_probabilities=[1.0, 0.1, 0.01], models=["relu_output"]
    )
    importance = geometric_importance(n, 0.7)
    dense = torch.zeros(m, n)
    dense[0, 0] = dense[1, 1] = dense[2, 2] = 1.0
    good = torch.zeros(m, n)  # features 0, 1 dedicated; 2 and 3 an antipodal pair
    good[0, 0] = good[1, 1] = 1.0
    good[2, 2], good[2, 3] = 1.0, -1.0
    bad = torch.zeros(m, n)  # feature 0 in an antipodal pair, 2 dedicated
    bad[0, 0], bad[0, 1] = 1.0, -1.0
    bad[1, 2] = bad[2, 3] = 1.0
    sparse = torch.zeros(m, n)  # six features as three antipodal pairs
    for d in range(3):
        sparse[d, 2 * d], sparse[d, 2 * d + 1] = 1.0, -1.0
    biases = [torch.zeros(n)] * 3
    for middle, expected in ((good, PASS), (bad, FAIL)):
        grid = grid_from(
            "relu_output",
            [dense, middle, sparse],
            biases,
            [0.1, 0.05, 0.01],
            [1.0, 0.1, 0.01],
            importance,
        )
        results = by_id(evaluate_setting(config, setting, {"relu_output": grid}))
        assert results[("toy", "relu_superposition_less_important_first")].status == expected
        assert results[("toy", "relu_antipodal_pairs_at_intermediate_sparsity")].status == PASS
        assert results[("toy", "relu_exceeds_m_at_sparsest")].status == PASS
