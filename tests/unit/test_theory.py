"""theory.py against Monte-Carlo estimates and independent quadrature (SPEC §6 ``theory.py``).

Paper: Mathematical Understanding, p. 13 (linear loss; the sparsity decomposition and its L₀,
L₁ terms); Superposition as a Phase Change, pp. 14–16 (families and theoretical diagrams);
Aside: Polytopes and Low-Rank Matrices, p. 21 (the projection construction).

Every exact value is compared with a sampled estimate of the same quantity, with a tolerance
of a few standard errors, or with scipy's adaptive quadrature at tolerance 1e-12.
"""

import math

import numpy as np
import pytest
import torch
from scipy.integrate import quad

from superposition import metrics, theory
from superposition.data import IndependentSparse
from superposition.models import TiedModel, get_activation
from superposition.utils import make_generator

N_SIGMA = 4.0


def tied(kind: str, W: torch.Tensor, b: torch.Tensor) -> TiedModel:
    """A model with the given weights ``(inst, hidden, feat)`` and biases ``(inst, feat)``."""
    model = TiedModel.from_kind(kind, W.shape[2], W.shape[1], W.shape[0])  # type: ignore[arg-type]
    with torch.no_grad():
        model.W.copy_(W)
        model.b.copy_(b)
    return model


def monte_carlo_loss(
    model: TiedModel,
    feature_probability: float | torch.Tensor,
    importance: torch.Tensor,
    n_samples: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-instance mean and standard error of the importance-weighted loss on fresh samples."""
    dist = IndependentSparse(feature_probability, model.n_inst, model.n_features)
    x = dist.sample(n_samples, make_generator(seed))
    with torch.no_grad():
        per_sample = (importance * (model.target(x) - model(x)) ** 2).sum(-1).double()
    return per_sample.mean(0), per_sample.std(0) / math.sqrt(n_samples)


def reference_integral(own: float, a: float, beta: float, c: float) -> float:
    """F(γ, α, β, c) = ∫₀¹∫₀¹ (γx − ReLU(αx + βy + c))² dy dx by nested adaptive quadrature,
    with every kink passed to the integrator as a breakpoint."""

    def inner(x: float) -> float:
        t = a * x + c
        kinks = [-t / beta] if beta != 0 and 0 < -t / beta < 1 else None
        value, _ = quad(
            lambda y: (own * x - max(t + beta * y, 0.0)) ** 2,
            0,
            1,
            points=kinks,
            epsabs=1e-13,
            epsrel=1e-13,
            limit=200,
        )
        return value

    kinks = [x for x in ([-c / a, -(c + beta) / a] if a != 0 else []) if 0 < x < 1] or None
    value, _ = quad(inner, 0, 1, points=kinks, epsabs=1e-13, epsrel=1e-13, limit=200)
    return value


# --- the exact integral F --------------------------------------------------------------------


def test_relu_square_integral_matches_adaptive_quadrature() -> None:
    rng = np.random.default_rng(1)
    for _ in range(40):
        own = float(rng.integers(0, 2))
        a, beta, c = (float(v) for v in rng.normal(0.0, 1.5, 3))
        if rng.random() < 0.2:
            beta = 0.0
        if rng.random() < 0.1:
            a = 0.0
        exact = float(theory.relu_square_integral(own, a, beta, c))
        assert exact == pytest.approx(reference_integral(own, a, beta, c), abs=1e-11)


def test_one_dimensional_closed_forms_are_special_cases_of_F() -> None:
    rng = np.random.default_rng(2)
    a, beta, c = rng.normal(0.0, 1.5, (3, 50))
    assert np.allclose(theory.own_error_1d(a, c), theory.relu_square_integral(1.0, a, 0.0, c))
    assert np.allclose(
        theory.relu_square_1d(beta, c), theory.relu_square_integral(0.0, 0.0, beta, c)
    )
    # Known values: a perfect readout has no error; a constant output c ≥ 0 costs c².
    assert float(theory.own_error_1d(1.0, 0.0)) == pytest.approx(0.0, abs=1e-15)
    assert float(theory.relu_square_integral(0.0, 0.0, 0.0, 0.7)) == pytest.approx(0.49)
    assert float(theory.relu_square_integral(0.0, 0.0, 0.0, -0.7)) == 0.0
    # Predicting nothing for an active feature costs E[x²] = 1/3.
    assert float(theory.own_error_1d(0.0, -1.0)) == pytest.approx(1 / 3)


# --- the linear model (p. 13) ----------------------------------------------------------------


def test_linear_decomposition_matches_monte_carlo() -> None:
    n_inst, m, n = 4, 3, 6
    gen = torch.Generator().manual_seed(0)
    W = torch.randn(n_inst, m, n, generator=gen) * 0.6
    b = torch.randn(n_inst, n, generator=gen) * 0.2
    importance = 0.8 ** torch.arange(n, dtype=torch.float32)
    p = torch.tensor([0.1, 0.3, 0.7, 1.0])[:, None]  # one density per instance
    terms = theory.linear_loss_decomposition(W, b, importance, p[:, 0])
    mean, se = monte_carlo_loss(tied("linear", W, b), p, importance, 400_000, seed=1)
    assert torch.all((terms.total - mean).abs() < N_SIGMA * se)
    assert torch.all(terms.feature_benefit >= 0) and torch.all(terms.interference >= 0)


def test_linear_decomposition_terms_on_known_geometries() -> None:
    p, importance = 0.2, torch.tensor([1.0, 0.5, 0.25])
    variance, mean = p / 3 - p**2 / 4, p / 2
    # W = [I₂ | 0] with b predicting the mean of the dropped feature: the closed-form optimum.
    W = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    b = torch.tensor([[0.0, 0.0, mean]])
    terms = theory.linear_loss_decomposition(W, b, importance, p)
    assert float(terms.interference[0]) == pytest.approx(0.0, abs=1e-12)
    assert float(terms.bias[0]) == pytest.approx(0.0, abs=1e-12)
    assert float(terms.total[0]) == pytest.approx(
        theory.linear_optimum_loss(importance, 2, p), rel=1e-9
    )
    # An antipodal pair in one dimension: benefit 0, interference Iᵢ·v per feature.
    W = torch.tensor([[[1.0, -1.0]]])
    terms = theory.linear_loss_decomposition(W, torch.zeros(1, 2), torch.ones(2), p)
    assert float(terms.feature_benefit[0]) == pytest.approx(0.0, abs=1e-12)
    assert float(terms.interference[0]) == pytest.approx(2 * variance)
    assert terms.interference_matrix[0, 0, 1] == pytest.approx(variance)


# --- the ReLU output model: sparsity decomposition (p. 13) -----------------------------------


def test_one_sparse_terms_match_monte_carlo() -> None:
    n_inst, m, n = 3, 2, 5
    gen = torch.Generator().manual_seed(3)
    W = torch.randn(n_inst, m, n, generator=gen) * 0.7
    b = torch.randn(n_inst, n, generator=gen) * 0.3
    importance = 0.9 ** torch.arange(n, dtype=torch.float32)
    model = tied("relu_output", W, b)
    terms = theory.one_sparse_decomposition(W, b, importance)
    # L₀ is deterministic: the loss on the zero input.
    L0 = metrics.loss_by_num_active(model, 0, importance, 10, make_generator(0))
    assert torch.allclose(terms.zero.float(), L0, atol=1e-6)
    # L₁ = Σᵢ E[loss | only feature i active]; two independent estimates bound its noise.
    estimates = torch.stack(
        [
            metrics.loss_by_num_active(model, 1, importance, 400_000, make_generator(s))
            for s in range(2)
        ]
    )
    spread = (estimates[0] - estimates[1]).abs()
    tolerance = torch.maximum(3 * spread, 0.005 * terms.one.float())
    assert torch.all((terms.one.float() - estimates.mean(0)).abs() < tolerance)


def test_negative_interference_is_free_on_one_sparse_inputs() -> None:
    """p. 13: ReLU makes negative interference free, and a negative bias converts small
    positive interference into none."""
    W = torch.tensor([[[1.0, -1.0]]])  # antipodal pair
    terms = theory.one_sparse_decomposition(W, torch.zeros(1, 2), torch.ones(2))
    assert torch.all(terms.interference == 0)
    W = torch.tensor([[[1.0, 0.3], [0.0, 0.954]]])  # positive interference 0.3 between the two
    with_bias = theory.one_sparse_decomposition(W, torch.full((1, 2), -0.35), torch.ones(2))
    without = theory.one_sparse_decomposition(W, torch.zeros(1, 2), torch.ones(2))
    assert float(without.interference.sum()) > 0
    assert float(with_bias.interference.sum()) == 0.0


def test_k_sparse_weights_cover_every_sparsity_pattern() -> None:
    """Lₖ sums over all C(n, k) active sets, so the binomial expansion gives Σ C(n, k)·wₖ = 1."""
    weights = theory.k_sparse_weights(5, 0.3)
    assert sum(math.comb(5, k) * w for k, w in enumerate(weights)) == pytest.approx(1.0)
    assert weights[0] == pytest.approx(0.7**5)


@pytest.mark.parametrize(
    ("W", "b", "p"),
    [
        ([[1.0, -1.0]], [-0.1, -0.1], 0.1),  # antipodal pair
        ([[0.9, 0.9]], [0.05, 0.05], 0.5),  # McGrath's confused solution
        ([[1.1, -0.8]], [-0.2, 0.1], 0.03),  # asymmetric pair
        ([[1.0, 0.0]], [0.0, 0.25], 1.0),  # the extra feature dropped
    ],
)
def test_exact_relu_output_loss_matches_monte_carlo(W: list, b: list, p: float) -> None:
    importance = torch.tensor([1.0, 0.7])
    exact = theory.relu_output_expected_loss(np.array(W), np.array(b), importance.numpy(), p)
    model = tied("relu_output", torch.tensor([W]), torch.tensor([b]))
    mean, se = monte_carlo_loss(model, p, importance, 1_000_000, seed=7)
    assert abs(exact - float(mean[0])) < N_SIGMA * float(se[0]) + 1e-9


def test_exact_loss_recombines_from_k_sparse_terms() -> None:
    """For n = 2: L = S²L₀ + pS·L₁ + p²L₂, with L₂ estimated by Monte Carlo."""
    W, b, p = torch.tensor([[[1.05, -0.95]]]), torch.tensor([[-0.08, 0.02]]), 0.3
    importance = torch.tensor([1.0, 0.6])
    model = tied("relu_output", W, b)
    exact = theory.relu_output_expected_loss(W[0], b[0], importance.numpy(), p)
    terms = theory.one_sparse_decomposition(W, b, importance)
    w0, w1, w2 = theory.k_sparse_weights(2, p)
    L2 = (exact - w0 * float(terms.zero[0]) - w1 * float(terms.one[0])) / w2
    estimates = [
        float(metrics.loss_by_num_active(model, 2, importance, 400_000, make_generator(s))[0])
        for s in range(2)
    ]
    tolerance = 3 * abs(estimates[0] - estimates[1]) + 1e-4
    assert abs(L2 - float(np.mean(estimates))) < tolerance


def test_exact_loss_refuses_outputs_reading_two_features() -> None:
    W = np.array([[1.0, 0.5, 0.5], [0.0, 0.8, -0.6]])  # output 0 reads features 1 and 2
    with pytest.raises(NotImplementedError):
        theory.relu_output_expected_loss(W, np.zeros(3), np.ones(3), 0.1)


# --- families and the theoretical phase diagrams (pp. 14–16) ---------------------------------


def test_family_templates_follow_the_projection_construction() -> None:
    antipodal = theory.N2M1_FAMILIES[2]
    W0, G0 = antipodal.template()
    assert W0.shape == (1, 2)
    assert np.allclose(G0, [[1.0, -1.0], [-1.0, 1.0]])
    assert antipodal.factors() == [[0, 1]] and antipodal.unrepresented() == []
    dropped = theory.N2M1_FAMILIES[0]
    assert dropped.factors() == [[0]] and dropped.unrepresented() == [1]
    confused_G0 = theory.N2M1_CONFUSED.template()[1]
    assert np.allclose(confused_G0, np.ones((2, 2)))
    # n = 3, m = 2: W ⊥ [1, 1, 0] puts features 0 and 1 in an antipodal pair and gives the extra
    # feature its own dimension; the two are separate tegum factors.
    pair_plus_dedicated = theory.N3M2_FAMILIES[3]
    W0, G0 = pair_plus_dedicated.template()
    assert W0.shape == (2, 3)
    assert pair_plus_dedicated.factors() == [[0, 1], [2]]
    assert np.allclose(G0, [[1, -1, 0], [-1, 1, 0], [0, 0, 1]])
    # p. 21: W ⊥ (1, 1, 1) is a triangle, every Dᵢ = 2/3.
    triangle = theory.Family("W ⊥ [1, 1, 1]", (1.0, 1.0, 1.0), theory.SUPERPOSITION)
    W0, _ = triangle.template()
    dims = metrics.feature_dimensionality(torch.tensor(W0)[None])[0]
    assert torch.allclose(dims, torch.full((3,), 2 / 3, dtype=dims.dtype))


def test_optimize_family_recovers_known_optima() -> None:
    p, extra = 0.4, 0.3
    solution = theory.optimize_family(theory.N2M1_FAMILIES[0], [[1.0, extra]], [p])
    # Feature 0 is stored perfectly (loss 0); the dropped extra feature predicts its mean p/2.
    assert solution.loss[0] == pytest.approx(extra * (p / 3 - p**2 / 4), rel=1e-7)
    assert solution.b[0, 1] == pytest.approx(p / 2, abs=1e-6)
    assert abs(solution.W[0, 0, 0]) == pytest.approx(1.0, abs=1e-4)


@pytest.mark.parametrize("family_index", [0, 1, 2])
@pytest.mark.parametrize(("importance", "p"), [(0.5, 0.05), (2.0, 0.3), (1.2, 1.0)])
def test_optimize_family_is_self_consistent_and_not_beaten_by_local_search(
    family_index: int, importance: float, p: float
) -> None:
    family = theory.N2M1_FAMILIES[family_index]
    solution = theory.optimize_family(family, [[1.0, importance]], [p])
    W, b = solution.W[0], solution.b[0]
    exact = theory.relu_output_expected_loss(W, b, [1.0, importance], p)
    assert exact == pytest.approx(solution.loss[0], rel=1e-10, abs=1e-14)
    # Perturb the free scalars (scale, both biases): no nearby point is better.
    W0, _ = family.template()
    scale = float(np.linalg.norm(W[:, family.factors()[0][0]]))
    rng = np.random.default_rng(family_index)
    for _ in range(200):
        s = max(scale + rng.normal(0, 0.02), 0.0)
        trial_b = b + rng.normal(0, 0.02, 2)
        trial = theory.relu_output_expected_loss(s * W0, trial_b, [1.0, importance], p)
        assert trial >= solution.loss[0] - 1e-10


def test_optimal_family_loss_matches_monte_carlo() -> None:
    p, importance = 0.05, [1.0, 1.5]
    solution = theory.optimize_family(theory.N2M1_FAMILIES[2], [importance], [p])
    model = tied(
        "relu_output",
        torch.tensor(solution.W, dtype=torch.float32),
        torch.tensor(solution.b, dtype=torch.float32),
    )
    mean, se = monte_carlo_loss(model, p, torch.tensor(importance), 1_000_000, seed=11)
    assert abs(solution.loss[0] - float(mean[0])) < N_SIGMA * float(se[0])


def test_theoretical_diagrams_have_the_papers_regions() -> None:
    """The corners of pp. 15–16: dense and unimportant → not learned; dense and important →
    dedicated dimension; sparse → superposition (n = 2) or the pair-plus-dedicated solution for
    an important extra feature (n = 3)."""
    importance, density = np.array([0.1, 0.5, 2.0, 10.0]), np.array([0.01, 0.1, 1.0])
    n2 = theory.theoretical_phase_diagram(theory.N2M1_FAMILIES, importance, density)
    assert n2.labels[2, 0] == theory.NOT_LEARNED
    assert n2.labels[2, 3] == theory.DEDICATED
    assert list(n2.labels[0]) == [theory.SUPERPOSITION] * 4
    n3 = theory.theoretical_phase_diagram(theory.N3M2_FAMILIES, importance, density)
    assert n3.labels[2, 0] == theory.NOT_LEARNED
    assert n3.labels[2, 3] == theory.DEDICATED
    assert n3.labels[0, 0] == theory.SUPERPOSITION
    assert n3.labels[0, 3] == theory.DEDICATED
    assert n3.families[int(n3.winner[0, 3])].name == "W ⊥ [1, 1, 0]"
    assert np.all(n2.loss_gap >= 0) and np.all(n3.loss_gap >= 0)
    assert np.array_equal(n2.winner, n2.losses.argmin(-1))


def test_confused_family_needs_zero_biases() -> None:
    """McGrath (pp. 51–53) analyzed the model without biases, where W₁ ≈ W₂ can be optimal;
    with free biases the antipodal or single-feature solutions beat it."""
    families = (*theory.N2M1_FAMILIES, theory.N2M1_CONFUSED)
    importance, density = np.geomspace(0.1, 10, 8), np.geomspace(0.01, 1, 8)
    no_bias = theory.theoretical_phase_diagram(
        families, importance, density, settings=theory.OptimizerSettings(fit_bias=False)
    )
    with_bias = theory.theoretical_phase_diagram(families, importance, density)
    assert (no_bias.labels == theory.CONFUSED).any()
    assert not (with_bias.labels == theory.CONFUSED).any()


# --- other output activations (E12, p. 51; docs/DECISIONS.md D-601) --------------------------

SMOOTH = ("gelu", "silu", "softplus", "sigmoid")


def reference_activation_integral(activation: str, own: float, a: float, beta: float, c: float):
    """F_σ by nested adaptive quadrature, splitting where σ's argument crosses 0."""
    sigma = get_activation(activation)

    def f(u: float) -> float:
        return float(sigma(torch.tensor(u, dtype=torch.float64)))

    def inner(x: float) -> float:
        t = a * x + c
        kinks = [-t / beta] if beta != 0 and 0 < -t / beta < 1 else None
        value, _ = quad(
            lambda y: (own * x - f(t + beta * y)) ** 2,
            0,
            1,
            points=kinks,
            epsabs=1e-13,
            epsrel=1e-13,
            limit=200,
        )
        return value

    kinks = [x for x in ([-c / a, -(c + beta) / a] if a != 0 else []) if 0 < x < 1] or None
    value, _ = quad(inner, 0, 1, points=kinks, epsabs=1e-13, epsrel=1e-13, limit=200)
    return value


def test_activation_integral_is_exact_for_piecewise_linear_activations() -> None:
    """ReLU, leaky ReLU and identity give polynomial pieces: the quadrature is exact."""
    rng = np.random.default_rng(11)
    own = rng.integers(0, 2, 200).astype(float)
    a, beta, c = rng.normal(0.0, 3.0, (3, 200))
    relu = theory.activation_square_integral("relu", own, a, beta, c, nodes=4)
    assert np.allclose(relu, theory.relu_square_integral(own, a, beta, c), atol=1e-12)
    for activation in ("leaky_relu", "identity"):
        for k in range(5):
            exact = reference_activation_integral(activation, own[k], a[k], beta[k], c[k])
            value = theory.activation_square_integral(activation, own[k], a[k], beta[k], c[k], 4)
            assert float(value) == pytest.approx(exact, abs=1e-11)


@pytest.mark.parametrize("activation", SMOOTH)
def test_activation_integral_matches_adaptive_quadrature(activation: str) -> None:
    """D-601: 24 nodes per piece are accurate to 1e-9 across E12's search box (scale ≤ 4, so
    |α|, |β| ≤ 16, and bias in [−8, 4]); the 12-node search stays within 1e-4 there."""
    rng = np.random.default_rng(3)
    for _ in range(6):
        own = float(rng.integers(0, 2))
        a, beta, c = rng.uniform(0, 16), rng.uniform(-16, 16), rng.uniform(-8, 4)
        exact = reference_activation_integral(activation, own, a, beta, c)
        fine = float(theory.activation_square_integral(activation, own, a, beta, c, nodes=24))
        coarse = float(theory.activation_square_integral(activation, own, a, beta, c, nodes=12))
        assert fine == pytest.approx(exact, abs=1e-9)
        assert coarse == pytest.approx(exact, abs=1e-4)


def test_activation_1d_terms_are_special_cases_of_the_2d_integral() -> None:
    rng = np.random.default_rng(4)
    a, beta, c = rng.normal(0.0, 2.0, (3, 40))
    for activation in SMOOTH:
        own_error = theory.activation_error_1d(activation, 1.0, a, c, nodes=24)
        interference = theory.activation_error_1d(activation, 0.0, beta, c, nodes=24)
        assert np.allclose(
            own_error, theory.activation_square_integral(activation, 1.0, a, 0.0, c, 24)
        )
        assert np.allclose(
            interference, theory.activation_square_integral(activation, 0.0, 0.0, beta, c, 24)
        )


@pytest.mark.parametrize(
    ("activation", "W", "b", "p"),
    [
        ("gelu", [[1.1, -1.1]], [0.1, 0.1], 0.1),
        ("silu", [[0.9, 0.9]], [0.2, -0.3], 0.5),
        ("softplus", [[1.6, -1.2]], [-1.5, -1.0], 0.05),
        ("sigmoid", [[2.5, -2.5]], [-3.0, -2.5], 0.2),
        ("leaky_relu", [[1.0, 0.0]], [0.0, 0.25], 1.0),
    ],
)
def test_exact_loss_with_other_activations_matches_monte_carlo(
    activation: str, W: list, b: list, p: float
) -> None:
    importance = torch.tensor([1.0, 0.7])
    exact = theory.relu_output_expected_loss(
        np.array(W), np.array(b), importance.numpy(), p, activation
    )
    model = TiedModel(2, 1, 1, out_act=activation)
    with torch.no_grad():
        model.W.copy_(torch.tensor([W]))
        model.b.copy_(torch.tensor([b]))
    mean, se = monte_carlo_loss(model, p, importance, 1_000_000, seed=8)
    assert abs(exact - float(mean[0])) < N_SIGMA * float(se[0]) + 1e-9


def test_vectorized_exact_losses_match_one_at_a_time() -> None:
    rng = np.random.default_rng(6)
    W = rng.normal(0.0, 1.0, (5, 1, 2))
    b = rng.normal(0.0, 0.5, (5, 2))
    importance, p = np.array([[1.0, 0.3]] * 5), np.geomspace(0.02, 1.0, 5)
    for activation in ("relu", "gelu", "sigmoid"):
        batch = theory.tied_output_expected_losses(W, b, importance, p, activation)
        one = [
            theory.relu_output_expected_loss(W[k], b[k], importance[k], p[k], activation)
            for k in range(5)
        ]
        assert np.allclose(batch, one, rtol=1e-12)
    with pytest.raises(NotImplementedError):
        theory.tied_output_expected_losses(
            np.array([[[1.0, 0.5, 0.5], [0.0, 0.8, -0.6]]]), np.zeros((1, 3)), np.ones(3), 0.1
        )


def test_wider_box_leaves_the_relu_diagrams_unchanged() -> None:
    """D-602: E12's box and golden-section count give E02's ReLU labels on E02's grid."""
    importance, density = np.geomspace(0.1, 10, 12), np.geomspace(0.01, 1, 12)
    e12 = theory.OptimizerSettings(scale_max=4.0, bias_range=(-8.0, 4.0), golden_iterations=30)
    for paper, confused in (
        (theory.N2M1_FAMILIES, theory.N2M1_CONFUSED),
        (theory.N3M2_FAMILIES, theory.N3M2_CONFUSED),
    ):
        families = (*paper, confused)
        e02_diagram = theory.theoretical_phase_diagram(families, importance, density)
        e12_diagram = theory.theoretical_phase_diagram(families, importance, density, settings=e12)
        assert (e02_diagram.labels == e12_diagram.labels).all()
        assert np.allclose(e02_diagram.losses, e12_diagram.losses, rtol=1e-6, atol=1e-12)
