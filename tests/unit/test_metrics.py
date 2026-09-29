"""Unit and property tests for metrics.py (SPEC §6: required known answers and properties).

Geometric configurations are built by hand as W of shape (1, hidden, feat): column i is Wᵢ.
"""

import math

import networkx as nx
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from superposition import metrics
from superposition.data import IndependentSparse, geometric_importance
from superposition.models import TiedModel
from superposition.train import evaluate_loss
from superposition.utils import make_generator

ATOL = 1e-5


def polygon(k: int) -> torch.Tensor:
    """Regular k-gon of unit vectors in 2D, shape (1, 2, k)."""
    angles = 2 * math.pi * torch.arange(k, dtype=torch.float64) / k
    return torch.stack([angles.cos(), angles.sin()])[None].float()


def tetrahedron() -> torch.Tensor:
    vertices = torch.tensor([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=torch.float32)
    return (vertices / math.sqrt(3)).T[None]


def simplex(n: int) -> torch.Tensor:
    """Project the standard basis of Rⁿ onto the complement of (1, …, 1) (p. 21)."""
    return (torch.eye(n) - torch.full((n, n), 1 / n))[None]


def random_orthogonal(m: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(m, m, generator=g, dtype=torch.float64))
    return (q * torch.sign(torch.diagonal(r))).float()


def all_close(t: torch.Tensor, value: float) -> bool:
    return torch.allclose(t, torch.full_like(t, value), atol=ATOL)


# --- Known answers (SPEC §6 table) ---------------------------------------------------------


def test_pentagon() -> None:
    W = polygon(5)
    assert all_close(metrics.feature_dimensionality(W), 2 / 5)
    assert metrics.dims_per_feature(W).item() == pytest.approx(0.4, abs=ATOL)


def test_antipodal_pair() -> None:
    W_1d = torch.tensor([[[1.0, -1.0]]])  # two features in one dimension
    assert all_close(metrics.feature_dimensionality(W_1d), 1 / 2)
    W_2d = torch.tensor([[[0.6, -0.6], [0.8, -0.8]]])
    assert all_close(metrics.feature_dimensionality(W_2d), 1 / 2)
    assert all_close(metrics.superposition_score(W_2d), 1.0)


def test_equilateral_triangle() -> None:
    assert all_close(metrics.feature_dimensionality(polygon(3)), 2 / 3)


def test_regular_tetrahedron() -> None:
    assert all_close(metrics.feature_dimensionality(tetrahedron()), 3 / 4)


def test_orthonormal_basis() -> None:
    W = random_orthogonal(4, seed=0)[None]
    assert all_close(metrics.feature_dimensionality(W), 1.0)
    assert all_close(metrics.superposition_score(W), 0.0)
    assert metrics.dims_per_feature(W).item() == pytest.approx(1.0, abs=ATOL)


def test_zero_column() -> None:
    W = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]])
    D = metrics.feature_dimensionality(W)
    assert torch.allclose(D, torch.tensor([[1.0, 1.0, 0.0]]), atol=ATOL)
    assert metrics.superposition_score(W)[0, 2].item() == 0.0
    assert metrics.feature_dimensionality(torch.zeros(1, 2, 3)).abs().max() == 0


@pytest.mark.parametrize("n", [2, 3, 4, 5, 8, 13])
def test_regular_simplex(n: int) -> None:
    assert all_close(metrics.feature_dimensionality(simplex(n)), (n - 1) / n)


def test_feature_norms_and_gram() -> None:
    W = torch.tensor([[[3.0, 0.0], [4.0, 2.0]]])
    assert torch.allclose(metrics.feature_norms(W), torch.tensor([[5.0, 2.0]]))
    assert torch.allclose(metrics.gram(W)[0], W[0].T @ W[0])


def test_superposition_score_matches_definition() -> None:
    g = torch.Generator().manual_seed(1)
    W = torch.randn(3, 4, 7, generator=g)
    unit = W / W.norm(dim=1, keepdim=True)
    expected = torch.zeros(3, 7)
    for k in range(3):
        for i in range(7):
            for j in range(7):
                if i != j:
                    expected[k, i] += (unit[k, :, i] @ W[k, :, j]) ** 2
    assert torch.allclose(metrics.superposition_score(W), expected, atol=1e-4)


def test_represented_mask_threshold() -> None:
    W = torch.tensor([[[0.4, 0.6, 1.0]]])
    assert metrics.represented_mask(W).tolist() == [[False, True, True]]
    assert metrics.represented_mask(W, threshold=0.3).tolist() == [[True, True, True]]


def test_is_superposition() -> None:
    orthonormal = random_orthogonal(3, seed=2)[None]
    assert not metrics.is_superposition(orthonormal).item()
    assert metrics.is_superposition(polygon(5)).item()
    assert metrics.is_superposition(torch.tensor([[[1.0, -1.0]]])).item()
    # Two independent, non-orthogonal features in 2D: WᵀW is invertible, so formally no.
    assert not metrics.is_superposition(torch.tensor([[[1.0, 0.8], [0.0, 0.6]]])).item()
    # Unrepresented features are excluded from the test (SPEC §3).
    W = torch.tensor([[[1.0, 0.0, 0.01], [0.0, 1.0, 0.02]]])
    assert not metrics.is_superposition(W).item()


def test_classify_fraction() -> None:
    D = torch.tensor([1.0, 0.76, 0.66, 0.51, 0.40, 0.372, 0.01, 0.3, 0.58])
    names = [name for name, _ in metrics.CHARACTERISTIC_FRACTIONS]
    idx = metrics.classify_fraction(D, tol=0.02).tolist()
    assert [names[i] if i >= 0 else None for i in idx] == [
        "dedicated dimension",
        "tetrahedron",
        "triangle",
        "antipodal pair (digon)",
        "pentagon",
        "square antiprism",
        "not learned",
        None,
        None,
    ]


def test_angles_and_canonicalize_2d() -> None:
    W = polygon(5)
    angles = metrics.angles_2d(W)[0]
    assert angles[1].item() == pytest.approx(2 * math.pi / 5, abs=ATOL)
    canon = metrics.canonicalize_2d(W)
    # The most important represented feature points straight up ...
    assert torch.allclose(canon[0, :, 0], torch.tensor([0.0, 1.0]), atol=ATOL)
    # ... and the next one lies in the left half-plane.
    assert canon[0, 0, 1].item() < 0
    # Rotations and reflections of the input give the same canonical form.
    reflect = torch.tensor([[-1.0, 0.0], [0.0, 1.0]])
    for seed in range(5):
        Q = random_orthogonal(2, seed)
        for T in (Q, Q @ reflect):
            assert torch.allclose(metrics.canonicalize_2d(T @ W), canon, atol=1e-5)
    # Unrepresented features do not become the anchor.
    W2 = torch.tensor([[[0.1, 0.0, 1.0], [0.0, 1.0, 0.0]]])
    assert torch.allclose(metrics.canonicalize_2d(W2)[0, :, 1], torch.tensor([0.0, 1.0]), atol=ATOL)


def test_geometry_graph_components() -> None:
    # Two antipodal pairs on orthogonal axes: two tegum factors (a "square", p. 21).
    W = torch.tensor([[1.0, -1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0, 0.01]])
    graph, components = metrics.geometry_graph(W)
    assert isinstance(graph, nx.Graph)
    assert graph.number_of_nodes() == 5
    assert components == [{0, 1}, {2, 3}]
    assert not graph.nodes[4]["represented"]
    assert graph.nodes[0]["dimensionality"] == pytest.approx(0.5, abs=ATOL)
    _, pentagon_components = metrics.geometry_graph(polygon(5)[0])
    assert pentagon_components == [{0, 1, 2, 3, 4}]


def test_neuron_polysemanticity() -> None:
    W = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0],  # monosemantic
                [0.9, -0.8, 0.0, 0.1],  # two features (0.1 < 0.25 · 0.9 is ignored)
                [0.01, 0.0, 0.0, 0.0],  # dead neuron
            ]
        ]
    )
    assert metrics.neuron_polysemanticity(W).tolist() == [[1, 2, 0]]


def test_loss_by_num_active_recombines_to_total_loss() -> None:
    """Σₖ pᵏ(1 − p)ⁿ⁻ᵏ Lₖ equals the expected loss (Paper: p. 13)."""
    n, p = 5, 0.3
    model = TiedModel.from_kind("relu_output", n, 2, n_inst=2, seed=3)
    with torch.no_grad():
        model.b.copy_(torch.tensor([[0.2, -0.1, 0.3, -0.2, 0.1], [0.0, 0.1, -0.3, 0.2, 0.05]]))
    importance = geometric_importance(n, 0.8)
    terms = [
        metrics.loss_by_num_active(model, k, importance, 20_000, make_generator(10 + k))
        for k in range(n + 1)
    ]
    # L₀ is exact: Σᵢ Iᵢ ReLU(bᵢ)² for the ReLU output model.
    assert torch.allclose(terms[0], (importance * model.b.detach().clamp_min(0) ** 2).sum(-1))
    recombined = torch.stack([p**k * (1 - p) ** (n - k) * terms[k] for k in range(n + 1)]).sum(0)
    dist = IndependentSparse(p, n_inst=2, n_features=n)
    total = evaluate_loss(model, dist, importance, 200_000, make_generator(99))
    assert torch.allclose(recombined, total, rtol=0.02)


# --- Property tests (SPEC §6) ----------------------------------------------------------------


@st.composite
def harmonic_tight_frames(draw: st.DrawFn) -> tuple[torch.Tensor, int, int]:
    """A random unit-norm tight frame of n vectors in m dimensions.

    Real harmonic frame: rows √(2/m)·cos(2πkj/n) and √(2/m)·sin(2πkj/n) for distinct
    frequencies 1 ≤ k < n/2 (plus a constant row 1/√m when m is odd), then a random rotation,
    column permutation and sign flips (all of which preserve the unit-norm tight property).
    """
    m = draw(st.integers(min_value=1, max_value=6))
    n = draw(st.integers(min_value=m + 1, max_value=4 * m + 6))
    available = list(range(1, (n + 1) // 2))  # 1 ≤ k < n/2
    n_pairs = m // 2
    if len(available) < n_pairs:
        n = 2 * n_pairs + 2
        available = list(range(1, (n + 1) // 2))
    freqs = draw(st.permutations(available))[:n_pairs]
    j = torch.arange(n, dtype=torch.float64)
    rows = []
    if m % 2 == 1:
        rows.append(torch.full((n,), 1 / math.sqrt(m), dtype=torch.float64))
    for k in freqs:
        rows.append(math.sqrt(2 / m) * torch.cos(2 * math.pi * k * j / n))
        rows.append(math.sqrt(2 / m) * torch.sin(2 * math.pi * k * j / n))
    frame = torch.stack(rows)
    seed = draw(st.integers(min_value=0, max_value=10_000))
    g = torch.Generator().manual_seed(seed)
    frame = random_orthogonal(m, seed).double() @ frame
    frame = frame[:, torch.randperm(n, generator=g)]
    frame = frame * torch.where(torch.rand(n, generator=g) < 0.5, -1.0, 1.0).double()
    return frame[None].float(), m, n


@settings(max_examples=60, deadline=None)
@given(harmonic_tight_frames())
def test_unit_norm_tight_frames_have_uniform_dimensionality(
    frame: tuple[torch.Tensor, int, int],
) -> None:
    W, m, n = frame
    D = metrics.feature_dimensionality(W)
    assert torch.allclose(D, torch.full_like(D, m / n), atol=1e-4)
    assert D.sum().item() == pytest.approx(m, abs=1e-3)


@settings(max_examples=60, deadline=None)
@given(
    n_inst=st.integers(1, 3),
    m=st.integers(1, 6),
    n=st.integers(1, 12),
    seed=st.integers(0, 10_000),
    scale=st.floats(0.05, 2.0),
)
def test_metrics_are_rotation_invariant(
    n_inst: int, m: int, n: int, seed: int, scale: float
) -> None:
    """Rotating the hidden space by any orthogonal matrix leaves every geometric metric fixed.

    Excluded by design: ``angles_2d`` (it reports orientation) and ``neuron_polysemanticity``
    (it is defined relative to the neuron basis, which a rotation changes; that basis
    dependence is exactly what a privileged basis means, p. 6).
    """
    g = torch.Generator().manual_seed(seed)
    W = torch.randn(n_inst, m, n, generator=g, dtype=torch.float64).float() * scale
    Q = random_orthogonal(m, seed + 1)
    R = torch.einsum("ij,kjf->kif", Q, W)
    for fn in (
        metrics.feature_norms,
        metrics.superposition_score,
        metrics.feature_dimensionality,
        metrics.dims_per_feature,
        metrics.gram,
        metrics.cosine_similarity_matrix,
    ):
        assert torch.allclose(fn(W), fn(R), atol=1e-4, rtol=1e-4), fn.__name__
    assert torch.equal(metrics.represented_mask(W), metrics.represented_mask(R))
    assert torch.equal(metrics.is_superposition(W), metrics.is_superposition(R))
    D_w, D_r = metrics.feature_dimensionality(W), metrics.feature_dimensionality(R)
    far_from_edges = (
        (D_w[..., None] - torch.tensor([1, 0.75, 2 / 3, 0.5, 0.4, 0.375, 0])).abs() - 0.02
    ).abs().min(-1).values > 1e-3
    assert torch.equal(
        metrics.classify_fraction(D_w)[far_from_edges],
        metrics.classify_fraction(D_r)[far_from_edges],
    )
    if m == 2:
        assert torch.allclose(metrics.canonicalize_2d(W), metrics.canonicalize_2d(R), atol=1e-4)
    _, comp_w = metrics.geometry_graph(W[0])
    _, comp_r = metrics.geometry_graph(R[0])
    assert comp_w == comp_r
