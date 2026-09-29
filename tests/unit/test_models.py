"""Unit tests for the toy models (SPEC §3 models and initialization, §6 models.py)."""

import math

import pytest
import torch

from superposition.models import (
    ACTIVATIONS,
    AbsModel,
    TiedModel,
    get_activation,
    register_activation,
    xavier_normal_per_instance,
)


def random_input(batch: int, n_inst: int, n_features: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return torch.rand(batch, n_inst, n_features, generator=g)


def manual_output(model: TiedModel, x: torch.Tensor) -> torch.Tensor:
    """Loop-based reference: x′_k = out_act(W_kᵀ hidden_act(W_k x_k) + b_k) per instance."""
    outs = []
    for k in range(model.n_inst):
        W, b = model.W[k], model.b[k]
        h = model.hidden_act(x[:, k] @ W.T)
        outs.append(model.out_act(h @ W + b))
    return torch.stack(outs, dim=1)


@pytest.mark.parametrize("kind", ["linear", "relu_output", "relu_hidden"])
def test_tied_model_matches_reference(kind: str) -> None:
    model = TiedModel.from_kind(kind, n_features=6, n_hidden=3, n_inst=4, seed=1)  # type: ignore[arg-type]
    with torch.no_grad():
        model.b.copy_(torch.randn(4, 6) * 0.3)
    x = random_input(16, 4, 6)
    assert torch.allclose(model(x), manual_output(model, x), atol=1e-6)


def test_linear_model_is_wtw_x_plus_b() -> None:
    model = TiedModel.from_kind("linear", 5, 2, 3, seed=2)
    x = random_input(8, 3, 5)
    expected = torch.einsum("ifg,big->bif", model.gram(), x) + model.b
    assert torch.allclose(model(x), expected, atol=1e-6)


def test_gram_is_wt_w() -> None:
    model = TiedModel(5, 2, 3, seed=3)
    for k in range(3):
        assert torch.allclose(model.gram()[k], model.W[k].T @ model.W[k])


def test_relu_hidden_has_no_hidden_bias() -> None:
    model = TiedModel.from_kind("relu_hidden", 10, 5, 2)
    names = {name for name, _ in model.named_parameters()}
    assert names == {"W", "b"}  # b is the output bias, shape (inst, feat)
    assert model.b.shape == (2, 10)


def test_bias_false_is_a_fixed_zero_buffer() -> None:
    model = TiedModel(4, 2, 2, bias=False)
    assert "b" not in dict(model.named_parameters())
    assert torch.equal(model.b, torch.zeros(2, 4))


def test_xavier_scale_and_independence_from_instance_count() -> None:
    n, m = 40, 10
    W = xavier_normal_per_instance(200, m, n, seed=0)
    expected_std = math.sqrt(2 / (n + m))
    assert W.std().item() == pytest.approx(expected_std, rel=0.02)
    assert W.mean().abs().item() < 0.01
    # Instance k depends only on (seed, k): adding instances never changes existing ones.
    small = TiedModel(n, m, n_inst=3, seed=5).W
    large = TiedModel(n, m, n_inst=9, seed=5).W
    assert torch.equal(small, large[:3])
    assert not torch.equal(large[0], large[1])
    assert not torch.equal(TiedModel(n, m, 3, seed=6).W, small)


def test_biases_initialize_to_zero() -> None:
    assert torch.equal(TiedModel(5, 2, 3).b, torch.zeros(3, 5))
    assert torch.equal(AbsModel(5, 2, 3).b, torch.zeros(3, 5))


def test_abs_model_shapes_target_and_forward() -> None:
    model = AbsModel(n_features=7, n_hidden=4, n_inst=3, seed=1)
    assert model.W1.shape == (3, 4, 7)
    assert model.W2.shape == (3, 7, 4)
    assert not torch.equal(model.W1[0].flatten()[:4], model.W2[0].flatten()[:4])
    x = random_input(5, 3, 7) * 2 - 1
    assert torch.equal(model.target(x), x.abs())
    expected = torch.relu(
        torch.einsum(
            "ifh,bih->bif", model.W2, torch.relu(torch.einsum("ihf,bif->bih", model.W1, x))
        )
        + model.b
    )
    assert torch.allclose(model(x), expected, atol=1e-6)
    assert model(x).min() >= 0


def test_activation_registry() -> None:
    expected = {"relu", "gelu", "silu", "softplus", "leaky_relu", "sigmoid", "identity"}
    assert expected <= set(ACTIVATIONS)
    x = torch.linspace(-2, 2, 9)
    assert torch.equal(get_activation(None)(x), x)
    assert torch.equal(get_activation("relu")(x), x.clamp_min(0))
    with pytest.raises(KeyError):
        get_activation("nope")
    register_activation("test_square", lambda t: t**2)
    try:
        assert torch.equal(get_activation("test_square")(x), x**2)
        with pytest.raises(KeyError):
            register_activation("test_square", lambda t: t)
    finally:
        ACTIVATIONS.pop("test_square")


def test_from_kind_rejects_unknown_kind() -> None:
    with pytest.raises(ValueError):
        TiedModel.from_kind("tanh_output", 3, 2, 1)  # type: ignore[arg-type]
