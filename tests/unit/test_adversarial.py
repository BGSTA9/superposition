"""The analytic L2 attack and the vulnerability measure (Paper: p. 29), on hand-built models."""

import math

import numpy as np
import pandas as pd
import pytest
import torch

from superposition.acceptance import FAIL, INFO, PASS
from superposition.adversarial import (
    analytic_attack,
    attack_budget,
    attack_directions,
    no_superposition_model,
    vulnerability,
)
from superposition.experiments.adversarial import (
    AdversarialSetting,
    AdversarialThresholds,
    evaluate_setting,
)
from superposition.models import TiedModel


def tied(W: torch.Tensor, b: torch.Tensor, kind: str = "relu_output") -> TiedModel:
    model = TiedModel.from_kind(kind, W.shape[-1], W.shape[-2], W.shape[0])  # type: ignore[arg-type]
    with torch.no_grad():
        model.W.copy_(W)
        model.b.copy_(b)
    return model


def test_budget_is_a_fraction_of_the_mean_input_norm() -> None:
    x = torch.tensor([[[3.0, 4.0]], [[0.0, 0.0]]])  # norms 5 and 0 → mean 2.5
    assert torch.allclose(attack_budget(x, 0.1), torch.tensor([0.25]))


def test_attack_directions_are_unit_rows_of_the_gram_matrix() -> None:
    W = torch.tensor([[[1.0, -1.0, 0.0]]])  # antipodal pair + one unrepresented feature
    directions = attack_directions(W)
    s = 1 / math.sqrt(2)
    expected = torch.tensor([[[s, -s, 0.0], [-s, s, 0.0], [0.0, 0.0, 0.0]]])
    assert torch.allclose(directions, expected)


def test_reference_damage_is_top_importance_times_budget_squared() -> None:
    n, m = 6, 3
    importance = torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2, 0.1])
    x = torch.rand(500, 2, n, generator=torch.Generator().manual_seed(0))
    epsilon = torch.tensor([0.3, 0.05])
    result = analytic_attack(no_superposition_model(n, m, 2), x, importance, epsilon)
    assert torch.allclose(result.damage, importance[0] * epsilon**2, rtol=1e-5)


def test_antipodal_pair_on_zero_inputs_is_twice_as_vulnerable() -> None:
    """W = [1, −1] in one dimension: the attack on feature 0 moves the input along (1, −1)/√2,
    which raises output 0 by ε√2 (a false positive of squared size 2ε²), versus ε² for the
    reference model: vulnerability exactly 2 on the all-zero input."""
    W = torch.tensor([[[1.0, -1.0]]])
    model = tied(W, torch.zeros(1, 2))
    x = torch.zeros(10, 1, 2)
    importance = torch.ones(2)
    epsilon = torch.tensor([0.1])
    attacked = analytic_attack(model, x, importance, epsilon)
    reference = analytic_attack(no_superposition_model(2, 1, 1), x, importance, epsilon)
    assert torch.allclose(attacked.clean_loss, torch.zeros(1))
    assert torch.allclose(attacked.attacked_loss, torch.tensor([2 * 0.1**2]))
    assert torch.allclose(vulnerability(attacked, reference), torch.tensor([2.0]))
    assert attacked.chosen.sum() == 10


def test_attacked_loss_never_below_clean_loss() -> None:
    g = torch.Generator().manual_seed(1)
    W = torch.randn(3, 4, 12, generator=g)
    model = tied(W, -0.2 * torch.ones(3, 12))
    x = torch.rand(300, 3, 12, generator=g) * (torch.rand(300, 3, 12, generator=g) < 0.2)
    result = analytic_attack(model, x, torch.ones(12), attack_budget(x))
    assert (result.attacked_loss >= result.clean_loss - 1e-7).all()


def _table(v: list[float], fpd: list[float]) -> pd.DataFrame:
    x = np.geomspace(1, 10, len(v))
    return pd.DataFrame(
        {
            "inverse_density": x,
            "vulnerability": v,
            "features_per_dimension": fpd,
            "epsilon": 0.1,
            "most_attacked_feature": 0,
            "reference_damage": 1.0,
            "reference_analytic": 1.0,
        }
    )


def test_e06_criteria_pass_when_vulnerability_rises_with_features_per_dimension() -> None:
    criteria = evaluate_setting(
        AdversarialSetting(), _table([1.0, 1.5, 3.0, 6.0], [1.0, 1.4, 2.5, 4.5]),
        AdversarialThresholds(),
    )  # fmt: skip
    status = {c.id: c.status for c in criteria}
    assert status["vulnerability_rises"] == PASS
    assert status["vulnerability_tracks_features_per_dimension"] == PASS


@pytest.mark.parametrize(
    ("v", "fpd", "failing"),
    [
        ([1.0, 1.2, 1.5, 1.9], [1.0, 1.4, 2.5, 4.5], "vulnerability_rises"),
        ([1.0, 4.0, 3.0, 2.5], [1.0, 1.4, 2.5, 4.5], "vulnerability_tracks_features_per_dimension"),
    ],
)
def test_e06_criteria_fail_without_the_claimed_trend(
    v: list[float], fpd: list[float], failing: str
) -> None:
    criteria = evaluate_setting(AdversarialSetting(), _table(v, fpd), AdversarialThresholds())
    assert {c.id: c.status for c in criteria}[failing] == FAIL


def test_e06_ungraded_setting_only_reports() -> None:
    setting = AdversarialSetting(label="extended", graded=False)
    criteria = evaluate_setting(setting, _table([1, 2, 3], [1, 2, 3]), AdversarialThresholds())
    assert [c.status for c in criteria] == [INFO]
