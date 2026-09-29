"""Tests for the sparse autoencoder and its ground-truth metrics (sae.py; Paper: pp. 42–44)."""

import pytest
import torch

from superposition import sae as S
from superposition.utils import make_generator


def signed_basis_sae(n_inst: int = 1) -> S.SparseAutoencoder:
    """An SAE with latents ±e₁, ±e₂ and zero biases: it reconstructs any 2D vector exactly."""
    model = S.SparseAutoencoder(2, 4, n_inst, resample=False)
    D = torch.tensor([[1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0]])
    with torch.no_grad():
        model.W_dec.copy_(D.expand(n_inst, -1, -1))
        model.W_enc.copy_(D.T.expand(n_inst, -1, -1))
    return model


def test_decoder_columns_are_unit_norm_after_init_and_steps() -> None:
    model = S.SparseAutoencoder(3, 7, 2, l1=0.1)
    assert torch.allclose(model.W_dec.norm(dim=1), torch.ones(2, 7))
    with torch.no_grad():
        model.W_dec.mul_(3.0)
    model.normalize_decoder()
    assert torch.allclose(model.W_dec.norm(dim=1), torch.ones(2, 7))


def test_init_depends_only_on_seed_and_instance() -> None:
    one = S.SparseAutoencoder(3, 5, 1, seed=4)
    three = S.SparseAutoencoder(3, 5, 3, seed=4)
    assert torch.equal(one.W_dec[0], three.W_dec[0])
    assert torch.equal(one.W_enc[0], one.W_dec[0].T)


def test_signed_basis_sae_reconstructs_exactly() -> None:
    model = signed_basis_sae()
    h = torch.tensor([[0.5, -0.25], [0.0, 0.0], [-1.0, 0.0]])
    stats = S.sae_stats(model, h)
    assert float(stats.mse[0]) == pytest.approx(0.0, abs=1e-12)
    assert float(stats.fvu[0]) == pytest.approx(0.0, abs=1e-12)
    assert float(stats.l0[0]) == pytest.approx((2 + 0 + 1) / 3)
    assert stats.alive[0].tolist() == [True, True, False, True]  # +e₁, −e₁, −e₂ fire
    assert float(stats.dead_fraction[0]) == pytest.approx(0.25)


def test_topk_keeps_the_k_largest_per_instance() -> None:
    model = signed_basis_sae(3)
    with torch.no_grad():
        model.top_k.copy_(torch.tensor([0, 1, 2]))
    h = torch.tensor([[[0.5, -0.25]]]).expand(1, 3, 2)
    f = model.encode(h)[0]
    assert (f > 0).sum(dim=-1).tolist() == [2, 1, 2]
    assert f[1].tolist() == [0.5, 0.0, 0.0, 0.0]


def test_masked_latents_never_fire() -> None:
    mask = torch.tensor([[True, True, False, False]])
    model = S.SparseAutoencoder(2, 4, 1, latent_mask=mask)
    f = model.encode(torch.randn(64, 1, 2))
    assert torch.equal(f[..., 2:], torch.zeros(64, 1, 2))


def test_recovery_uses_the_signed_cosine_and_live_latents_only() -> None:
    true = torch.tensor([[1.0, 0.0, 0.6], [0.0, 2.0, 0.8]])  # three features in 2D
    represented = torch.tensor([True, True, False])
    decoder = torch.stack(
        [
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),  # both recovered exactly
            torch.tensor([[-1.0, 0.0], [0.0, -1.0]]),  # antiparallel: not recovered
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),  # exact, but latent 1 is dead
        ]
    )
    alive = torch.tensor([[True, True], [True, True], [True, False]])
    rec = S.recovery(decoder, alive, true, represented)
    assert rec.fraction[0.9].tolist() == pytest.approx([1.0, 0.0, 0.5])
    assert rec.mean_max_cos[0] == pytest.approx(1.0)
    assert torch.isnan(rec.max_cos[:, 2]).all()  # unrepresented features are not scored
    assert rec.best_latent[0, :2].tolist() == [0, 1]


def test_resampling_targets_badly_reconstructed_inputs() -> None:
    model = S.SparseAutoencoder(2, 3, 2, resample=[True, False])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    h = torch.tensor([[[3.0, 4.0]] * 2] * 8)  # every input is (3, 4)
    model.loss_terms(h)[0].sum().backward()
    optimizer.step()
    before = model.W_dec.detach().clone()
    dead = torch.tensor([[False, True, False], [False, True, False]])
    counts = S.resample_dead_latents(model, optimizer, h, dead, 0.2, make_generator(0))
    assert counts.tolist() == [1, 0]
    direction = torch.tensor([3.0, 4.0]) - model.b_dec[0].detach()
    direction = direction / direction.norm()
    assert torch.allclose(model.W_dec[0, :, 1], direction, atol=1e-6)
    assert float(model.b_enc[0, 1].detach()) == 0.0
    assert torch.equal(optimizer.state[model.W_dec]["exp_avg"][0, :, 1], torch.zeros(2))
    assert torch.equal(model.W_dec[1].detach(), before[1])  # resampling is off for instance 1
    assert torch.equal(model.W_dec[0, :, [0, 2]].detach(), before[0, :, [0, 2]])


def test_training_learns_two_orthogonal_features_and_instances_are_independent() -> None:
    W = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    def sample(batch: int, generator: torch.Generator) -> torch.Tensor:
        active = torch.rand(batch, 2, generator=generator) < 0.2
        values = torch.rand(batch, 2, generator=generator)
        return (active * values) @ W.T

    config = S.SAETrainConfig(steps=1500, batch_size=256, resample_every=0)
    pair = S.SparseAutoencoder(2, 4, 2, l1=[0.1, 0.3], resample=False, seed=1)
    S.train_sae(pair, sample, config, seed=2)
    alone = S.SparseAutoencoder(2, 4, 1, l1=0.1, resample=False, seed=1)
    S.train_sae(alone, sample, config, seed=2)
    assert torch.allclose(pair.W_dec[0], alone.W_dec[0], atol=1e-5)
    h = sample(20_000, make_generator(5))
    stats = S.sae_stats(pair, h)
    rec = S.recovery(pair.unit_decoder(), stats.alive, W, torch.tensor([True, True]))
    assert float(rec.fraction[0.95][0]) == 1.0  # the λ = 0.1 SAE found both features
