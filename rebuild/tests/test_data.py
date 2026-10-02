# Rebuild kit copy of tests/unit/test_data.py (M1). Only the imports differ:
# they point at rebuild/superposition_rebuild/ instead of src/superposition/.
"""Unit tests for the feature distributions (SPEC §6 data.py).

Statistical checks use fixed seeds and a 3-standard-error band, pooled so that each assertion
is one comparison.
"""

import math

import pytest
import torch

from superposition_rebuild.data import (
    AnticorrelatedSets,
    Composite,
    CorrelatedSets,
    FeatureDistribution,
    IndependentSparse,
    geometric_importance,
)
from superposition.utils import make_generator

N = 40_000


def within_3_se(observed: float, expected: float, se: float) -> bool:
    return abs(observed - expected) <= 3 * se


@pytest.mark.parametrize("p", [1.0, 0.3, 0.05, 0.003])
def test_independent_density_matches_p(p: float) -> None:
    dist = IndependentSparse(p, n_inst=2, n_features=10)
    x = dist.sample(N, make_generator(1))
    trials = x.numel()
    density = (x != 0).float().mean().item()
    assert within_3_se(density, p, math.sqrt(p * (1 - p) / trials) + 1e-12)


def test_density_z_scores_are_calibrated() -> None:
    """Across many seeds the density z-scores look standard normal (unbiased sampler).

    This is the stronger check behind the fixed-seed 3-SE tests: seed 2 once gave a 3.45σ
    deviation at p = 0.1, and a 3,000-draw study (mean z −0.03, std 1.001) showed that was a
    fixed-seed false positive, not a bias. See docs/progress/phase-1.md.
    """
    p, n_samples, cells = 0.1, 10_000, 3 * 8
    dist = IndependentSparse(p, n_inst=3, n_features=8)
    z = torch.tensor(
        [
            ((dist.sample(n_samples, make_generator(seed)) != 0).float().mean().item() - p)
            / math.sqrt(p * (1 - p) / (n_samples * cells))
            for seed in range(64)
        ]
    )
    assert abs(z.mean().item()) < 3 / math.sqrt(64)
    assert 0.7 < z.std().item() < 1.3


def test_independent_per_instance_and_per_feature_probabilities() -> None:
    per_instance = torch.tensor([1.0, 0.5, 0.1])
    dist = IndependentSparse(per_instance, n_inst=3, n_features=8)
    # Seed 2 gave a 3.45σ false positive here (see test_density_z_scores_are_calibrated).
    x = dist.sample(N, make_generator(12))
    density = (x != 0).float().mean(dim=(0, 2))
    trials = N * 8
    for observed, p in zip(density.tolist(), per_instance.tolist(), strict=True):
        assert within_3_se(observed, p, math.sqrt(p * (1 - p) / trials) + 1e-12)

    per_feature = torch.tensor([[0.9, 0.2]])
    dist = IndependentSparse(per_feature, n_inst=4, n_features=2)
    x = dist.sample(N, make_generator(3))
    density = (x != 0).float().mean(dim=(0, 1))
    trials = N * 4
    for observed, p in zip(density.tolist(), per_feature[0].tolist(), strict=True):
        assert within_3_se(observed, p, math.sqrt(p * (1 - p) / trials))


@pytest.mark.parametrize(("low", "high"), [(0.0, 1.0), (-1.0, 1.0)])
def test_independent_nonzero_range_and_mean(low: float, high: float) -> None:
    dist = IndependentSparse(0.2, n_inst=2, n_features=10, low=low, high=high)
    x = dist.sample(N, make_generator(4))
    values = x[x != 0]
    assert values.min() >= low
    assert values.max() <= high
    std = (high - low) / math.sqrt(12)
    assert within_3_se(values.mean().item(), (low + high) / 2, std / math.sqrt(values.numel()))
    # Uniformity: the variance of U[low, high] is (high − low)²/12.
    assert values.var().item() == pytest.approx(std**2, rel=0.02)


def test_independent_rejects_bad_probabilities() -> None:
    with pytest.raises(ValueError):
        IndependentSparse(1.5, n_inst=1, n_features=3)
    with pytest.raises(ValueError):
        IndependentSparse(torch.ones(4), n_inst=3, n_features=3)
    with pytest.raises(ValueError):
        IndependentSparse(0.5, n_inst=1, n_features=3, low=1.0, high=0.0)


def test_correlated_sets_switch_together() -> None:
    dist = CorrelatedSets(n_sets=3, set_size=4, feature_probability=0.3, n_inst=2)
    assert dist.n_features == 12
    x = dist.sample(N, make_generator(5))
    active = (x != 0).reshape(N, 2, 3, 4)
    # Within a set, every member is active iff the first one is.
    assert torch.equal(active, active[..., :1].expand_as(active))
    set_density = active[..., 0].float().mean().item()
    trials = N * 2 * 3
    assert within_3_se(set_density, 0.3, math.sqrt(0.3 * 0.7 / trials))
    values = x[x != 0]
    assert values.min() >= 0 and values.max() <= 1
    # Nonzero values are drawn independently: members of an active set differ.
    members = x.reshape(N, 2, 3, 4)[active[..., 0]]
    assert (members[:, 0] != members[:, 1]).float().mean() > 0.99


def test_anticorrelated_sets_never_two_active() -> None:
    dist = AnticorrelatedSets(n_sets=2, set_size=3, feature_probability=0.4, n_inst=3)
    x = dist.sample(N, make_generator(6))
    active = (x != 0).reshape(N, 3, 2, 3)
    per_set = active.sum(dim=-1)
    assert per_set.max() <= 1
    set_density = per_set.float().mean().item()
    trials = N * 3 * 2
    assert within_3_se(set_density, 0.4, math.sqrt(0.4 * 0.6 / trials))
    # The active member is uniform over the set.
    member_share = active.float().sum(dim=(0, 1, 2)) / active.float().sum()
    n_active = active.float().sum().item()
    for share in member_share.tolist():
        assert within_3_se(share, 1 / 3, math.sqrt((1 / 3) * (2 / 3) / n_active))


def test_composite_concatenates_parts() -> None:
    parts = [
        IndependentSparse(0.5, n_inst=2, n_features=3),
        CorrelatedSets(1, 2, 0.5, n_inst=2),
        AnticorrelatedSets(1, 2, 0.5, n_inst=2),
    ]
    dist = Composite(parts)
    assert isinstance(dist, FeatureDistribution)
    assert (dist.n_inst, dist.n_features) == (2, 7)
    x = dist.sample(1000, make_generator(7))
    assert x.shape == (1000, 2, 7)
    corr = (x[..., 3:5] != 0).float()
    assert torch.equal(corr[..., 0], corr[..., 1])
    assert (x[..., 5:7] != 0).sum(dim=-1).max() <= 1
    with pytest.raises(ValueError):
        Composite([IndependentSparse(0.5, 2, 3), IndependentSparse(0.5, 3, 3)])


def test_sampling_is_deterministic_given_generator() -> None:
    dist = IndependentSparse(0.3, n_inst=2, n_features=5)
    a = dist.sample(100, make_generator(8))
    b = dist.sample(100, make_generator(8))
    assert torch.equal(a, b)


def test_geometric_importance() -> None:
    importance = geometric_importance(4, 0.9)
    assert torch.allclose(importance, torch.tensor([1.0, 0.9, 0.81, 0.729]))
    assert torch.equal(geometric_importance(3, 1.0), torch.ones(3))
