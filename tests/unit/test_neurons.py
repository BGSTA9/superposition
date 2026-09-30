"""E07/E08 neuron analysis on hand-built models (Paper: pp. 31–39; footnote 15)."""

import pandas as pd
import pytest
import torch

from superposition import metrics
from superposition.acceptance import FAIL, INFO, PASS
from superposition.experiments.abs_computation import (
    AbsComputationConfig,
    AbsThresholds,
    evaluate_sweep,
    model_losses,
    neuron_feature_counts,
    primary_secondary,
    reference_losses,
    side_roles,
)
from superposition.experiments.privileged_basis import (
    PrivilegedBasisConfig,
    ReluHiddenWithBias,
    clipped_fraction,
    evaluate,
)
from superposition.models import AbsModel
from superposition.theory import uniform_feature_variance


def ideal_abs_model(n: int) -> AbsModel:
    """|x| exactly: neuron 2i = ReLU(xᵢ), neuron 2i + 1 = ReLU(−xᵢ), both written to output i
    (p. 36), with each neuron's weights rescaled differently to exercise footnote 15."""
    model = AbsModel(n, 2 * n, n_inst=1)
    W1 = torch.zeros(2 * n, n)
    W2 = torch.zeros(n, 2 * n)
    for i in range(n):
        scale_pos, scale_neg = 0.5 + i, 2.0 + i
        W1[2 * i, i], W2[i, 2 * i] = scale_pos, 1 / scale_pos
        W1[2 * i + 1, i], W2[i, 2 * i + 1] = -scale_neg, 1 / scale_neg
    with torch.no_grad():
        model.W1.copy_(W1[None])
        model.W2.copy_(W2[None])
        model.b.zero_()
    return model


def test_normalizing_hidden_units_keeps_the_function() -> None:
    g = torch.Generator().manual_seed(0)
    model = AbsModel(5, 7, n_inst=2, seed=3)
    x = 2 * torch.rand(64, 2, 5, generator=g) - 1
    before = model(x)
    W1n, W2n = metrics.normalize_hidden_units(model.W1.detach(), model.W2.detach())
    with torch.no_grad():
        model.W1.copy_(W1n)
        model.W2.copy_(W2n)
    assert torch.allclose(model(x), before, atol=1e-6)
    assert torch.allclose(W1n.abs().amax(dim=-1), torch.ones(2, 7))


def test_per_feature_loss_is_zero_for_the_exact_abs_model() -> None:
    model = ideal_abs_model(3)
    loss = metrics.per_feature_loss(
        model, torch.tensor([0.3]), 2_000, torch.Generator().manual_seed(0), -1.0, 1.0
    )
    assert torch.allclose(loss, torch.zeros(1, 3), atol=1e-10)


def test_per_feature_loss_of_the_zero_model_is_the_second_moment() -> None:
    """Output 0 everywhere: E[|xᵢ|²] = p/3 exactly, whatever the other features do."""
    model = AbsModel(4, 3, n_inst=2)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    probs = torch.tensor([0.001, 0.5])
    loss = metrics.per_feature_loss(model, probs, 4_000, torch.Generator().manual_seed(1), -1, 1)
    # the active term is estimated from 4,000 fresh U[−1, 1] draws: E[x²] = 1/3 ± ~1.5%
    assert torch.allclose(loss, (probs / 3)[:, None].expand(2, 4), rtol=0.05)


def test_per_feature_loss_agrees_with_plain_monte_carlo() -> None:
    g = torch.Generator().manual_seed(2)
    model = AbsModel(6, 4, n_inst=1, seed=5)
    p = torch.tensor([0.4])
    exact = metrics.per_feature_loss(model, p, 20_000, g, -1, 1)
    x = torch.rand(200_000, 1, 6, generator=g) * 2 - 1
    x = x * (torch.rand(200_000, 1, 6, generator=g) < 0.4)
    with torch.no_grad():
        mc = ((model.target(x) - model(x)) ** 2).mean(dim=0)
    assert torch.allclose(exact, mc, rtol=0.05, atol=1e-4)


def test_side_roles_of_the_ideal_model() -> None:
    model = ideal_abs_model(3)
    W1n, W2n = metrics.normalize_hidden_units(model.W1.detach(), model.W2.detach())
    roles = side_roles(W1n[0].numpy(), W2n[0].numpy(), AbsThresholds())
    assert roles == [
        "positive-side(0)",
        "negative-side(0)",
        "positive-side(1)",
        "negative-side(1)",
        "positive-side(2)",
        "negative-side(2)",
    ]


def test_side_roles_reject_mixed_neurons() -> None:
    W1n = torch.tensor([[[1.0, 0.6], [-1.0, 0.0]]])
    W2n = torch.tensor([[[1.0, 1.0], [0.0, 0.9]]])  # neuron 1 writes to the wrong output
    assert side_roles(W1n[0].numpy(), W2n[0].numpy(), AbsThresholds()) == ["other", "other"]


def test_neuron_counts_mark_silent_neurons_dead() -> None:
    W1n = torch.tensor([[[1.0, 0.1, 0.0], [1.0, 0.5, 0.3], [0.0, 1.0, 0.0]]])
    W2n = torch.tensor([[[1.0, 1.0, 0.01], [0.0, 0.2, 0.0], [0.0, 0.0, 0.0]]])
    counts = neuron_feature_counts(W1n, W2n, AbsThresholds())
    assert counts.tolist() == [[1, 3, 0]]


def test_primary_plus_secondary_pattern() -> None:
    W1n = torch.tensor([[1.0, 0.3, 0.0], [1.0, 0.9, 0.0], [1.0, 0.0, 0.0]]).numpy()
    counts = torch.tensor([2, 2, 1]).numpy()
    assert primary_secondary(W1n, counts, ratio=2.0) == 1  # only neuron 0


def _abs_summary(dense_mono: int, sparse_well: int, sparse_useful: int) -> pd.DataFrame:
    rows = [
        {"feature_probability": 1.0, "monosemantic": dense_mono, "polysemantic": 40 - dense_mono},
        {"feature_probability": 0.001, "monosemantic": 0, "polysemantic": 40},
    ]
    for row, well, useful in zip(rows, (20, sparse_well), (40, sparse_useful), strict=True):
        row.update(
            dead=0,
            computed_well=well,
            computed_well_features=list(range(well)),
            usefully_computed=useful,
            usefully_computed_features=list(range(useful)),
            primary_plus_secondary=0,
            max_features_per_neuron=1,
            loss=0.0,
            loss_exact=0.002,
            loss_mean_reference=0.004,
            loss_no_superposition_reference=0.001,
        )
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    ("dense_mono", "sparse_well", "sparse_useful", "expected", "original_would_pass"),
    [
        (40, 25, 45, [PASS, PASS], True),
        (38, 25, 45, [FAIL, PASS], True),
        (40, 20, 45, [PASS, PASS], False),  # graded on m, not on SPEC's original m/2 (D-R01)
        (40, 25, 40, [PASS, FAIL], True),  # exactly m usefully computed is not more than m
    ],
)
def test_e08_sweep_criteria(
    dense_mono: int,
    sparse_well: int,
    sparse_useful: int,
    expected: list[str],
    original_would_pass: bool,
) -> None:
    summary = _abs_summary(dense_mono, sparse_well, sparse_useful)
    criteria = evaluate_sweep(AbsComputationConfig(), summary)
    assert [c.status for c in criteria] == [*expected, INFO, INFO, INFO]
    by_id = {c.id: c for c in criteria}
    assert by_id["b_sparse_original_criterion"].value["would_pass"] is original_would_pass
    references = by_id["b_within_sparsity_references"].value["1-S=0.001"]
    assert references["loss_over_no_superposition"] == pytest.approx(2.0)


def test_e08_reference_losses_and_exact_model_loss() -> None:
    importance = 0.8 ** torch.arange(100.0)
    p = torch.tensor([1.0, 0.001])
    mean_ref, no_sp_ref = reference_losses(importance, p, n_hidden=40)
    variance = p / 3 - p**2 / 4
    assert torch.allclose(mean_ref, variance * importance.sum())
    assert torch.allclose(no_sp_ref, variance * importance[20:].sum())
    # Every feature at its mean (ratio 1) is the mean reference; the top 20 exact (ratio 0) and
    # the rest at their mean is the no-superposition reference.
    ones = torch.ones(2, 100)
    assert torch.allclose(model_losses(ones, importance, p), mean_ref)
    top = ones.clone()
    top[:, :20] = 0.0
    assert torch.allclose(model_losses(top, importance, p), no_sp_ref)


def test_e07_criteria_on_a_synthetic_sweep() -> None:
    def summary(poly: list[int]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "feature_probability": [1.0, 0.3, 0.1, 0.03][: len(poly)],
                "monosemantic": [5 - k for k in poly],
                "polysemantic": poly,
                "dead": 0,
                "features_per_neuron": [[1] * 5] * len(poly),
                "represented": 5,
                "superposed": 0,
                "loss": 0.1,
                "restarts_same_poly_count": 1.0,
            }
        )

    config = PrivilegedBasisConfig()
    good = {c.id: c.status for c in evaluate(config, summary([0, 2, 3, 5]), None)}
    assert good["dense_all_monosemantic"] == PASS
    assert good["polysemantic_count_non_decreasing"] == PASS
    assert good["polysemantic_at_sparsest"] == PASS
    bad = {c.id: c.status for c in evaluate(config, summary([1, 4, 1, 0]), None)}
    assert bad["dense_all_monosemantic"] == FAIL
    assert bad["polysemantic_count_non_decreasing"] == FAIL
    assert bad["polysemantic_at_sparsest"] == FAIL


def test_hidden_bias_variant_and_clipping() -> None:
    model = ReluHiddenWithBias(4, 2, n_inst=1)
    with torch.no_grad():
        model.W.copy_(torch.tensor([[[1.0, -1.0, 0.5, 0.0], [0.0, 1.0, -1.0, 1.0]]]))
        model.c.fill_(10.0)
    x = torch.rand(8, 1, 4)
    h = model.hidden(x)
    assert torch.allclose(h, model.pre_activation(x))  # c ≫ 0: the ReLU never clips
    g = torch.Generator().manual_seed(0)
    p = torch.tensor([0.5])
    W = model.W.detach()
    assert clipped_fraction(W, model.c.detach(), p, 2_000, g).item() == 0.0
    assert clipped_fraction(W, None, p, 2_000, g).item() > 0.1


def test_mean_prediction_baseline_matches_abs_target_variance() -> None:
    """Var(|x|) for x = 0 w.p. 1 − p else U[−1, 1] is p/3 − p²/4, the E08 baseline."""
    g = torch.Generator().manual_seed(4)
    p = 0.3
    x = (torch.rand(400_000, generator=g) * 2 - 1) * (torch.rand(400_000, generator=g) < p)
    assert abs(float(x.abs().var()) - uniform_feature_variance(p)) < 1e-3
