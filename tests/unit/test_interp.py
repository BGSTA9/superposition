"""Known-answer tests for neuron roles, motifs, ablations and the rebuild test (interp.py).

Paper: pp. 36 (side neurons) and 38–39 (asymmetric superposition with inhibition).
"""

import pytest
import torch

from superposition import interp, metrics
from superposition.utils import make_generator


def exact_abs_model(n: int = 3) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """|x| = ReLU(x) + ReLU(−x): a positive- and a negative-side neuron per feature (p. 36)."""
    W1 = torch.zeros(2 * n, n)
    W2 = torch.zeros(n, 2 * n)
    for i in range(n):
        W1[2 * i, i], W1[2 * i + 1, i] = 1.0, -1.0
        W2[i, 2 * i] = W2[i, 2 * i + 1] = 1.0
    return W1, W2, torch.zeros(n)


def motif_model() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """A hand-built model with one asymmetric superposition + inhibition motif (p. 39).

    Feature 0 is heavy, 1 the victim, 2 the inhibitor's main feature. Neuron 1 stores 0 with
    weight −1 and 1 with 0.25, outputs 1 and 4 (reciprocal); neuron 3 is the positive side of
    feature 2 plus a small read of −0.2 from feature 0 and an output of −25 onto feature 1.
    """
    W1 = torch.zeros(6, 3)
    W2 = torch.zeros(3, 6)
    W1[0, 0], W2[0, 0] = 1.0, 1.0  # positive-side(0)
    W1[1, 0], W1[1, 1], W2[0, 1], W2[1, 1] = -1.0, 0.25, 1.0, 4.0  # asymmetric(0, 1)
    W1[2, 1], W2[1, 2] = -1.0, 1.0  # negative-side(1)
    W1[3, 2], W1[3, 0], W2[2, 3], W2[1, 3] = 1.0, -0.2, 1.0, -25.0  # inhibitor(1) + side(2)
    W1[4, 2], W2[2, 4] = -1.0, 1.0  # negative-side(2)
    return W1, W2, torch.zeros(3)  # neuron 5 is dead


def test_side_neurons_of_the_exact_model() -> None:
    W1, W2, _ = exact_abs_model()
    labels = [r.label for r in interp.neuron_roles(W1, W2)]
    assert labels == [
        "positive-side(0)", "negative-side(0)", "positive-side(1)", "negative-side(1)",
        "positive-side(2)", "negative-side(2)",
    ]  # fmt: skip


def test_roles_of_the_motif_model() -> None:
    W1, W2, _ = motif_model()
    roles = interp.neuron_roles(W1, W2)
    assert [r.label for r in roles] == [
        "positive-side(0)",
        "asymmetric-superposition(0, 1)",
        "negative-side(1)",
        "inhibitor(1) + positive-side(2)",
        "negative-side(2)",
        "dead",
    ]
    assert roles[3].triggers == (0,)
    assert roles[3].predicted_rise == (1, 2)
    assert roles[1].predicted_rise == (0, 1)
    assert [r.named for r in roles] == [True] * 5 + [False]


def test_roles_do_not_change_under_footnote_15_rescaling() -> None:
    W1, W2, _ = motif_model()
    alpha = torch.tensor([3.0, 0.5, 2.0, 7.0, 0.1, 1.0])
    before = [r.label for r in interp.neuron_roles(W1, W2)]
    after = [r.label for r in interp.neuron_roles(W1 * alpha[:, None], W2 / alpha[None, :])]
    assert before == after
    W1n, W2n = metrics.normalize_hidden_units(W1[None], W2[None])
    assert [r.label for r in interp.neuron_roles(W1n[0], W2n[0])] == before


def test_normal_superposition_is_not_asymmetric() -> None:
    """W₁ = [1, −1], W₂ = [1, 1] is the paper's normal superposition (p. 39)."""
    W1 = torch.tensor([[1.0, -1.0, 0.0]])
    W2 = torch.tensor([[1.0], [1.0], [0.0]])
    (role,) = interp.neuron_roles(W1, W2)
    assert role.kind == interp.SUPERPOSITION and role.features == (0, 1)
    # Sizes flipped on only one side is not asymmetric superposition either.
    (role,) = interp.neuron_roles(torch.tensor([[1.0, -0.25, 0.0]]), W2)
    assert role.kind == interp.SUPERPOSITION


def test_three_feature_neuron_is_unexplained() -> None:
    W1 = torch.tensor([[1.0, 1.0, 1.0]])
    W2 = torch.tensor([[1.0], [1.0], [1.0]])
    (role,) = interp.neuron_roles(W1, W2)
    assert role.kind == interp.UNEXPLAINED and not role.named


def test_motif_detector() -> None:
    W1, W2, b = motif_model()
    (motif,) = interp.find_motifs(W1, W2, b)
    assert (motif.asymmetric_neuron, motif.inhibitor_neuron) == (1, 3)
    assert (motif.heavy, motif.victim) == (0, 1)
    assert motif.input_ratio == pytest.approx(4.0)
    assert motif.output_ratio == pytest.approx(4.0)
    assert motif.interference == pytest.approx(4.0)  # |−1| · 4
    assert motif.inhibition == pytest.approx(-5.0)  # |−0.2| · (−25)
    # Only neurons reading feature 0 on its negative side count: 4 − 5 + b₁ = −1.
    assert motif.net_interference == pytest.approx(-1.0)


def test_no_motif_without_the_inhibitor_or_the_flip() -> None:
    W1, W2, b = motif_model()
    weak = W2.clone()
    weak[1, 3] = -5.0  # inhibition 0.2 · 5 = 1 < 0.5 · 4
    assert interp.find_motifs(W1, weak, b) == []
    wrong_side = W1.clone()
    wrong_side[3, 0] = 0.2  # reads feature 0's positive side: never fires with neuron 1
    assert interp.find_motifs(wrong_side, W2, b) == []
    equal = W2.clone()
    equal[1, 1] = 1.0  # outputs not reciprocal
    assert interp.find_motifs(W1, equal, b) == []


def test_ablations_of_the_exact_model_match_closed_forms() -> None:
    """Zeroing a side neuron costs E[x² ; that side] = p/6. Mean-ablating it replaces h by
    E[h] = p/4 =: c, costing p/6 − pc/2 + c² (x ~ U[0, 1] on the side it reads)."""
    p = 0.3
    W1, W2, b = exact_abs_model()
    ab = interp.ablation_matrix(W1, W2, b, p, 40_000, make_generator(0))
    assert torch.allclose(ab.baseline, torch.zeros(3), atol=1e-7)
    assert torch.allclose(ab.hidden_mean, torch.full((6,), p / 4), atol=3e-3)
    for h in range(6):
        i = h // 2
        assert float(ab.zero[h, i]) == pytest.approx(p / 6, rel=0.03)
        c = p / 4
        assert float(ab.mean[h, i]) == pytest.approx(p / 6 - p * c / 2 + c**2, rel=0.05)
        others = [j for j in range(3) if j != i]
        assert torch.allclose(ab.zero[h, others], torch.zeros(2), atol=1e-7)
    assert torch.allclose(ab.mean_prediction, torch.full((3,), p / 3 - p**2 / 4))


def test_ablations_agree_with_the_motif_roles() -> None:
    W1, W2, b = motif_model()
    roles = interp.neuron_roles(W1, W2)
    ab = interp.ablation_matrix(W1, W2, b, 0.05, 20_000, make_generator(1))
    for role in roles:
        for kind in ("zero", "mean"):
            assert interp.ablation_agrees(role, ab.relative(kind)[role.neuron], 0.05), role.label
    # Zeroing the inhibitor exposes feature 0's interference on feature 1.
    assert float(ab.relative("zero")[3, 1]) > 1.0


def test_ablation_agreement_rules() -> None:
    side = interp.NeuronRole(0, interp.POSITIVE_SIDE, (1,), computes=(1,))
    assert interp.ablation_agrees(side, torch.tensor([0.0, 0.5, 0.01]), 0.05)
    assert not interp.ablation_agrees(side, torch.tensor([0.0, 0.03, 0.0]), 0.05)  # too small
    assert not interp.ablation_agrees(side, torch.tensor([0.9, 0.5, 0.0]), 0.05)  # wrong feature
    dead = interp.NeuronRole(1, interp.DEAD)
    assert interp.ablation_agrees(dead, torch.tensor([0.0, 0.01, -0.01]), 0.05)
    assert not interp.ablation_agrees(dead, torch.tensor([0.0, 0.2, 0.0]), 0.05)
    assert (
        interp.ablation_agrees(interp.NeuronRole(2, interp.UNEXPLAINED), torch.ones(3), 0.05)
        is None
    )


def test_rebuild_reproduces_idealized_models() -> None:
    """A model that already is its idealization is rebuilt exactly (recovered share 1)."""
    for W1, W2, b in (exact_abs_model(), motif_model()):
        roles = interp.neuron_roles(W1, W2)
        W1r, W2r = interp.rebuild_weights(roles, W1, W2)
        assert torch.allclose(W1r, W1) and torch.allclose(W2r, W2)
        rb = interp.rebuild_test(roles, W1, W2, b, torch.ones(3), 0.05, 5_000, make_generator(2))
        assert rb.fraction_recovered == pytest.approx(1.0)
        assert torch.allclose(rb.rebuild, rb.trained)


def test_rebuild_drops_unexplained_neurons_and_idealizes_the_rest() -> None:
    W1, W2, b = exact_abs_model(2)
    # An extra neuron that reads feature 0 but only interferes with output 1: no role.
    W1 = torch.cat([W1 * 1.1, torch.tensor([[0.7, 0.0]])])
    W2 = torch.cat([W2 / 1.1, torch.tensor([[0.02], [0.3]])], dim=1)
    roles = interp.neuron_roles(W1, W2)
    assert roles[-1].kind == interp.UNEXPLAINED
    W1n, W2n = metrics.normalize_hidden_units(W1[None], W2[None])
    W1r, W2r = interp.rebuild_weights(roles, W1n[0], W2n[0])
    assert torch.equal(W1r[-1], torch.zeros(2)) and torch.equal(W2r[:, -1], torch.zeros(2))
    assert torch.allclose(W1r[:4], exact_abs_model(2)[0])
    rb = interp.rebuild_test(roles, W1, W2, b, torch.ones(2), 0.1, 5_000, make_generator(3))
    # The dropped neuron only hurt, so the rebuild beats the trained model (share > 1).
    assert rb.fraction_recovered > 1.0
    assert float((rb.rebuild - rb.trained).max()) <= 1e-6
    assert rb.fraction_recovered_trained_bias == pytest.approx(rb.fraction_recovered, abs=1e-6)


def test_idealized_bias_predicts_the_mean_of_unused_features() -> None:
    W2 = torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, -2.0]])
    b = interp.idealized_bias(W2, 0.2)
    assert torch.allclose(b, torch.tensor([0.0, 0.1, 0.1]))  # E|x| = p/2


def test_role_table_columns() -> None:
    W1, W2, b = motif_model()
    roles = interp.neuron_roles(W1, W2)
    ab = interp.ablation_matrix(W1, W2, b, 0.05, 2_000, make_generator(4))
    table = interp.role_table(roles, W1, W2, ab)
    assert list(table["role"]) == [r.label for r in roles]
    assert {"zero: agrees", "mean: agrees", "predicted rise"} <= set(table.columns)
