# Rebuild kit copy of tests/unit/test_train.py (M3). Only the imports differ:
# they point at rebuild/superposition_rebuild/ instead of src/superposition/.
"""Unit tests for train.py (SPEC §3 loss, §6 train.py) and the closed-form linear optimum."""

import math

import pytest
import torch
from superposition_rebuild.data import IndependentSparse, geometric_importance
from superposition_rebuild.models import AbsModel, TiedModel
from superposition_rebuild.train import (
    TrainResult,
    evaluate_loss,
    importance_weighted_loss,
    lr_multiplier,
    select_best,
    train,
)

from superposition.theory import linear_optimum_loss
from superposition.utils import make_generator


def test_importance_weighted_loss_reduction_order() -> None:
    g = torch.Generator().manual_seed(0)
    out, target = torch.rand(4, 3, 5, generator=g), torch.rand(4, 3, 5, generator=g)
    importance = torch.rand(3, 5, generator=g)
    loss = importance_weighted_loss(out, target, importance)
    expected = torch.zeros(3)
    for k in range(3):
        per_sample = [(importance[k] * (target[b, k] - out[b, k]) ** 2).sum() for b in range(4)]
        expected[k] = torch.stack(per_sample).mean()  # sum over features, mean over batch
    assert loss.shape == (3,)
    assert torch.allclose(loss, expected)


def test_instances_are_independent() -> None:
    """Instance 0's gradient does not depend on instance 1's parameters (SPEC §6)."""
    importance = geometric_importance(5, 0.9)
    x = IndependentSparse(0.3, n_inst=2, n_features=5).sample(256, make_generator(0))

    def grad_of_instance_0(model: TiedModel) -> tuple[torch.Tensor, torch.Tensor]:
        model.zero_grad()
        out = model(x)
        importance_weighted_loss(out, model.target(x), importance).sum().backward()
        assert model.W.grad is not None and model.b.grad is not None
        return model.W.grad[0].clone(), model.b.grad[0].clone()

    a = TiedModel.from_kind("relu_output", 5, 2, n_inst=2, seed=0)
    b = TiedModel.from_kind("relu_output", 5, 2, n_inst=2, seed=0)
    with torch.no_grad():
        b.W[1].mul_(-3.0).add_(1.0)
        b.b[1].fill_(0.7)
    for ga, gb in zip(grad_of_instance_0(a), grad_of_instance_0(b), strict=True):
        assert torch.equal(ga, gb)


def test_training_reduces_loss_and_logs() -> None:
    n, m, n_inst = 5, 2, 3
    model = TiedModel.from_kind("relu_output", n, m, n_inst, seed=1)
    dist = IndependentSparse(0.2, n_inst, n)
    importance = geometric_importance(n, 0.9)
    before = evaluate_loss(model, dist, importance, 20_000, make_generator(5))
    result = train(model, dist, importance, steps=400, log_every=100, eval_samples=20_000)
    assert isinstance(result, TrainResult)
    assert result.log_steps == [0, 100, 200, 300, 399]
    assert result.loss_history.shape == (5, n_inst)
    assert result.final_loss.shape == (n_inst,)
    assert (result.final_loss < before).all()
    assert result.snapshots == []


def test_training_is_deterministic_given_seed() -> None:
    def run() -> torch.Tensor:
        model = TiedModel.from_kind("linear", 4, 2, 2, seed=3)
        dist = IndependentSparse(0.5, 2, 4)
        train(model, dist, torch.ones(4), steps=50, eval_samples=1000, seed=7)
        return model.W.detach().clone()

    assert torch.equal(run(), run())


def test_snapshots() -> None:
    model = TiedModel.from_kind("relu_output", 4, 2, 2)
    dist = IndependentSparse(0.3, 2, 4)
    result = train(model, dist, torch.ones(4), steps=25, snapshot_every=10, eval_samples=1000)
    assert result.snapshot_steps == [0, 10, 20, 25]
    assert set(result.snapshots[0]) == {"W", "b"}
    assert torch.equal(result.snapshots[-1]["W"], model.W.detach().cpu())
    assert not torch.equal(result.snapshots[0]["W"], result.snapshots[-1]["W"])


def test_l1_hidden_shrinks_hidden_activations() -> None:
    def hidden_l1(l1: float) -> float:
        model = TiedModel.from_kind("relu_output", 6, 3, 2, seed=2)
        dist = IndependentSparse(0.3, 2, 6)
        train(model, dist, torch.ones(6), steps=600, l1_hidden=l1, eval_samples=1000)
        x = dist.sample(2000, make_generator(1))
        return model.hidden(x).abs().sum(-1).mean().item()

    assert hidden_l1(0.5) < 0.8 * hidden_l1(0.0)  # observed ratio ≈ 0.62 after 600 steps


def test_abs_model_trains() -> None:
    model = AbsModel(3, 6, n_inst=2, seed=0)
    dist = IndependentSparse(0.5, 2, 3, low=-1.0, high=1.0)
    before = evaluate_loss(model, dist, torch.ones(3), 10_000, make_generator(1))
    result = train(model, dist, torch.ones(3), steps=500, eval_samples=10_000)
    assert (result.final_loss < 0.5 * before).all()


def test_lr_schedules() -> None:
    assert lr_multiplier("constant", 100, 0.05)(73) == 1.0
    cosine = lr_multiplier("cosine", 100, 0.05)
    assert cosine(0) == pytest.approx(1.0)
    assert cosine(50) == pytest.approx(0.5)
    assert cosine(100) == pytest.approx(0.0, abs=1e-12)
    warm = lr_multiplier("warmup_cosine", 100, 0.1)
    assert warm(0) == pytest.approx(0.1)
    assert warm(9) == pytest.approx(1.0)
    assert warm(10) == pytest.approx(1.0)
    assert warm(55) == pytest.approx(0.5)
    with pytest.raises(ValueError):
        lr_multiplier("linear", 100, 0.1)  # type: ignore[arg-type]


def test_select_best() -> None:
    loss = torch.tensor([0.5, 0.2, 0.9, 0.1, 0.3, float("nan"), 0.4])
    groups = [2, 2, 0, 0, 1, 1, 1]
    assert select_best(loss, groups).tolist() == [3, 4, 1]
    result = TrainResult([0], torch.zeros(1, 7), loss, steps=1, eval_samples=1, wall_time_s=0.0)
    assert select_best(result, torch.tensor(groups)).tolist() == [3, 4, 1]


@pytest.mark.parametrize(("p", "reference"), [(1.0, 0.183), (0.2, 0.124), (0.1, 0.068)])
def test_evaluation_reproduces_linear_optimum(p: float, reference: float) -> None:
    """The hand-built optimal linear model scores the closed-form optimum (SPEC §7 E00).

    Optimum for n = 5, m = 2, Iᵢ = 0.9ⁱ: represent features 0 and 1 exactly (W = [e₀ e₁ 0 0 0])
    and predict the mean p/2 for the rest, so L* = Σ_{i≥2} Iᵢ (p/3 − p²/4).
    """
    n, m = 5, 2
    importance = geometric_importance(n, 0.9)
    optimum = linear_optimum_loss(importance, m, p)
    assert optimum == pytest.approx(reference, abs=5e-4)
    model = TiedModel.from_kind("linear", n, m, n_inst=1)
    with torch.no_grad():
        model.W.zero_()
        model.W[0, 0, 0] = 1.0
        model.W[0, 1, 1] = 1.0
        model.b.copy_(torch.tensor([[0.0, 0.0, p / 2, p / 2, p / 2]]))
    dist = IndependentSparse(p, 1, n)
    estimate = evaluate_loss(model, dist, importance, 200_000, make_generator(0)).item()
    # Per-sample loss has std ≤ 0.5·ΣIᵢ; 200k samples give a relative SE below 1% here.
    assert estimate == pytest.approx(optimum, rel=0.02)
    # A worse model (feature 1 dropped) scores strictly higher.
    with torch.no_grad():
        model.W[0, 1, 1] = 0.0
        model.b[0, 1] = p / 2
    worse = evaluate_loss(model, dist, importance, 200_000, make_generator(0)).item()
    assert worse > optimum + 0.5 * importance[1].item() * (p / 3 - p**2 / 4)


def test_evaluate_loss_handles_partial_chunks() -> None:
    model = TiedModel.from_kind("linear", 3, 1, 2)
    dist = IndependentSparse(0.5, 2, 3)
    value = evaluate_loss(model, dist, torch.ones(3), 1234, make_generator(0))
    assert value.shape == (2,)
    assert math.isfinite(value.sum().item())
