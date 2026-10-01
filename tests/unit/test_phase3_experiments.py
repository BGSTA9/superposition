"""The acceptance logic of E02–E04, checked on hand-built weights (no training).

Paper: pp. 14–16, 51–53 (E02); pp. 16–21 (E03); pp. 21–25 (E04). As for E00 and E01, these
tests pin down what each criterion means: an ideal paper solution must pass, and a solution
that breaks the claim must fail.
"""

import math
from collections.abc import Sequence

import numpy as np
import pytest
import torch

from superposition import theory
from superposition.acceptance import FAIL, INFO, PASS
from superposition.experiments import nonuniform_geometry as e04
from superposition.experiments import phase_change as e02
from superposition.experiments import uniform_geometry as e03


def by_id(criteria: list) -> dict:
    return {c.id: c for c in criteria}


def unit_vectors(degrees: Sequence[float], norms: Sequence[float] | None = None) -> torch.Tensor:
    """(2, k) columns at the given angles (degrees), unit length unless ``norms`` is given."""
    rad = np.deg2rad(degrees)
    W = np.stack([np.cos(rad), np.sin(rad)]) * (1.0 if norms is None else np.asarray(norms))
    return torch.tensor(W, dtype=torch.float32)


# --- E02 phase_change ------------------------------------------------------------------------


def test_e02_labels_the_extra_feature() -> None:
    """The three regimes of p. 14 and McGrath's confused solution (pp. 51–53)."""
    W = torch.tensor(
        [
            [[1.0, 0.0]],  # W ⊥ [0, 1]: extra feature dropped
            [[0.0, 1.0]],  # W ⊥ [1, 0]: extra feature gets the dimension
            [[1.0, -1.0]],  # W ⊥ [1, 1]: antipodal pair
            [[0.9, 0.9]],  # W ⊥ [1, −1]: both share one direction with the same sign
            [[0.3, 1.0]],  # the other feature is too weak to count as represented
        ]
    )
    labels, norm, score = e02.extra_feature_regime(W, feature=1)
    assert list(labels) == [
        theory.NOT_LEARNED,
        theory.DEDICATED,
        theory.SUPERPOSITION,
        theory.CONFUSED,
        theory.DEDICATED,
    ]
    assert torch.allclose(norm, torch.tensor([0.0, 1.0, 1.0, 0.9, 1.0]))
    assert float(score[2]) == 1.0 and float(score[4]) == 0.0
    # n = 3, m = 2 (p. 16): the extra feature is feature 2.
    W3 = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, -1.0]],  # W ⊥ [0, 1, 1]: extra paired with feature 1
            [[1.0, -1.0, 0.0], [0.0, 0.0, 1.0]],  # W ⊥ [1, 1, 0]: extra dedicated, others paired
            [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],  # W ⊥ [1, 0, 0]: extra dedicated, 0 dropped
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],  # W ⊥ [0, 0, 1]: extra dropped
        ]
    )
    labels3, _, _ = e02.extra_feature_regime(W3, feature=2)
    assert list(labels3) == [
        theory.SUPERPOSITION,
        theory.DEDICATED,
        theory.DEDICATED,
        theory.NOT_LEARNED,
    ]


def test_e02_cell_majority_discards_the_worst_model() -> None:
    labels = np.array([["a", "a", "b", "b", "b"], ["a", "b", "b", "a", "c"]])
    losses = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 1.0, 2.0, 3.0, 0.5]])
    majority, counts = e02.cell_majority(labels, losses, discard_worst=1)
    # Row 0: the worst model ("b", loss 5) is dropped, leaving a tie a:2 b:2 broken by the
    # lowest loss (an "a"). Row 1: the worst ("a", loss 5) is dropped; b wins 2 to 1 to 1.
    assert list(majority) == ["a", "b"]
    assert counts[0] == {"a": 2, "b": 2} and counts[1] == {"b": 2, "a": 1, "c": 1}


def test_e02_near_boundary() -> None:
    labels = np.array([["a", "a", "a"], ["a", "a", "b"], ["a", "a", "b"]])
    expected = np.array([[False, True, True], [False, True, True], [False, True, True]])
    assert np.array_equal(e02.near_boundary(labels), expected)


def synthetic_grid(
    setting: e02.ProblemSetting, config: e02.PhaseChangeConfig, predicted: e02.TheoryResult
) -> e02.PhaseGridResult:
    """Every model of every cell set to the winning paper family's optimal weights; the last
    model of each cell has the highest loss (it is discarded)."""
    grid = config.grid
    n_dens, n_imp, models = grid.n_density, grid.n_importance, config.models_per_cell
    n_paper = len(predicted.family_labels) - 1
    winner = predicted.losses[..., :n_paper].argmin(-1)
    d, i = np.meshgrid(np.arange(n_dens), np.arange(n_imp), indexing="ij")
    W = predicted.W[winner, d, i].reshape(n_dens * n_imp, setting.n_hidden, setting.n_features)
    b = predicted.b[winner, d, i].reshape(n_dens * n_imp, setting.n_features)
    loss = np.tile(np.arange(models, dtype=np.float64), n_dens * n_imp)
    cell = torch.arange(n_dens * n_imp).repeat_interleave(models)
    return e02.PhaseGridResult(
        setting=setting.label,
        W=torch.tensor(W, dtype=torch.float32)[cell],
        b=torch.tensor(b, dtype=torch.float32)[cell],
        final_loss=torch.tensor(loss, dtype=torch.float32),
        extra_importance=torch.tensor(np.tile(grid.importance(), n_dens), dtype=torch.float32)[
            cell
        ],
        density=torch.tensor(np.repeat(grid.density(), n_imp), dtype=torch.float32)[cell],
        cell=cell,
        importance_grid=grid.importance(),
        density_grid=grid.density(),
        models_per_cell=models,
        wall_time_s=0.0,
    )


def test_e02_theory_weights_pass_and_broken_weights_fail() -> None:
    """Labeling the theory's own optimal weights must reproduce the theory's labels in every
    cell (so empirical and theoretical diagrams use one definition of each regime)."""
    config = e02.PhaseChangeConfig(
        grid=e02.PhaseGrid(n_importance=6, n_density=6), models_per_cell=3
    )
    for setting in config.settings:
        predicted = e02.compute_theory(setting, config)
        empirical = synthetic_grid(setting, config, predicted)
        frame, criteria = e02.evaluate_setting(config, setting, empirical, predicted)
        assert bool(frame["agree"].all()), frame[~frame["agree"]]
        result = by_id(criteria)
        for key in ("three_regions_empirical", "three_regions_theoretical"):
            assert result[key].status == PASS
        assert result["empirical_theory_agreement"].status == PASS
        assert result["empirical_theory_agreement"].value["agreement"] == 1.0
        assert result["disagreements"].status == INFO
        assert result["disagreements"].value["n_disagree"] == 0

        broken = synthetic_grid(setting, config, predicted)
        broken.W = torch.zeros_like(broken.W)  # nothing learned anywhere
        _, broken_criteria = e02.evaluate_setting(config, setting, broken, predicted)
        broken_result = by_id(broken_criteria)
        assert broken_result["three_regions_empirical"].status == FAIL
        assert broken_result["empirical_theory_agreement"].status == FAIL


# --- E03 uniform_geometry --------------------------------------------------------------------


def test_e03_plateau_span_takes_the_widest_run() -> None:
    x = np.geomspace(1, 10, 10)
    dstar = np.array([1.0, 0.9, 0.51, 0.7, 0.5, 0.49, 0.52, 0.48, 0.3, 0.2])
    assert e03.plateau_span(x, dstar, 0.5, 0.03) == (4, 7)
    assert e03.plateau_span(x, np.ones(10), 0.5, 0.03) is None


def test_e03_fraction_clusters() -> None:
    rng = np.random.default_rng(0)

    def near(value: float, k: int) -> np.ndarray:
        return value + rng.uniform(-0.005, 0.005, k)

    dims = [
        near(0.5, 30),
        np.concatenate([near(2 / 3, 12), near(0.4, 12), near(0.75, 5), np.zeros(11)]),
    ]
    represented = [np.ones(30, bool), np.array([True] * 29 + [False] * 11)]
    clusters = e03.fraction_clusters(dims, represented, 0.02, 0.2, 8)
    found = {name for name, c in clusters.items() if c["cluster"]}
    assert found == {"antipodal pair (digon) (0.5)", "triangle (0.667)", "pentagon (0.4)"}
    assert clusters["tetrahedron (0.75)"]["count"] == 5  # too few to count as a cluster
    # D-603: with each feature's tegum factor, a feature counts only in its fraction's polytope.
    factors = [
        ["digon (antipodal pair)"] * 30,
        ["triangle"] * 12 + ["7 in 3 (non-uniform)"] * 12 + ["tetrahedron"] * 5 + [None] * 11,
    ]
    clusters = e03.fraction_clusters(dims, represented, 0.02, 0.2, 8, factors)
    found = {name for name, c in clusters.items() if c["cluster"]}
    assert found == {"antipodal pair (digon) (0.5)", "triangle (0.667)"}
    assert clusters["pentagon (0.4)"]["count"] == 0


def test_e03_a_band_between_two_fractions_is_not_two_clusters() -> None:
    """R2-01: 2/5 and 3/8 are 0.025 apart, so their ±0.02 windows overlap. A single band of Dᵢ
    centred between them must not count as a pentagon cluster and an antiprism cluster at once
    unless the features really sit in pentagons and square antiprisms."""
    rng = np.random.default_rng(1)
    band = 0.3875 + rng.normal(0.0, 0.01, 76)
    represented = [np.ones(76, bool)]
    by_value = e03.fraction_clusters([band], represented, 0.02, 0.2, 8)
    assert by_value["pentagon (0.4)"]["cluster"] and by_value["square antiprism (0.375)"]["cluster"]
    one_factor = [["76 in 30 (non-uniform)"] * 76]
    strict = e03.fraction_clusters([band], represented, 0.02, 0.2, 8, one_factor)
    assert not any(c["cluster"] for c in strict.values())


def paper_like_summary(dstar_fn) -> e03.Summary:
    """A 20-point curve over 1/(1 − S) ∈ [1, 10] with Dᵢ that cluster at 1/2, 2/3 and 2/5."""
    x = np.geomspace(1, 10, 20)
    dstar = np.array([dstar_fn(v) for v in x])
    half = np.full(60, 0.5)
    mixed = np.concatenate([np.full(15, 2 / 3), np.full(20, 0.4), np.full(10, 0.5)])
    dims = [half if abs(d - 0.5) < 0.03 else mixed for d in dstar]
    represented = [np.ones(len(d), bool) for d in dims]
    polytope = {2 / 3: "triangle", 0.4: "pentagon", 0.5: "digon (antipodal pair)"}
    factors: list[list[str | None]] = [
        [polytope[min(polytope, key=lambda f: abs(f - v))] for v in d] for d in dims
    ]
    return e03.Summary(
        inverse_density=x,
        dstar=dstar,
        dimensionality=dims,
        represented=represented,
        census=[{} for _ in x],
        sum_dimensionality=np.array([d.sum() for d in dims]),
        factors=factors,
    )


def paper_curve(v: float) -> float:
    """Sticky at 1 up to 1.3, at 1/2 over [2.3, 3.8], then ≈ 0.25 at 8 (p. 17)."""
    if v <= 1.3:
        return 1.0
    if v < 2.3:
        return 1.0 - 0.48 * math.log(v / 1.3) / math.log(2.3 / 1.3)
    if v <= 3.8:
        return 0.5
    return 0.5 * (3.8 / v) ** 0.93


def test_e03_paper_curve_passes_and_flat_curve_fails() -> None:
    config = e03.UniformGeometryConfig()
    result = by_id(e03.evaluate(config, paper_like_summary(paper_curve)))
    for key in ("dense_dstar_is_1", "half_plateau", "falls_after_plateau", "fraction_clusters"):
        assert result[key].status == PASS, (key, result[key].value)
    assert result["sum_dimensionality"].status == INFO
    flat = by_id(e03.evaluate(config, paper_like_summary(lambda v: 1.0)))
    assert flat["dense_dstar_is_1"].status == PASS
    assert flat["half_plateau"].status == FAIL
    assert flat["falls_after_plateau"].status == FAIL
    # A plateau shorter than the minimum span fails.
    short = by_id(
        e03.evaluate(
            config, paper_like_summary(lambda v: 0.5 if 2.9 <= v <= 3.2 else paper_curve(v) - 0.1)
        )
    )
    assert short["half_plateau"].status == FAIL


# --- E04 nonuniform_geometry -----------------------------------------------------------------


def pentagon_with_gap(neighbour_angle: float) -> torch.Tensor:
    """Feature 0 at the top, its neighbours 1 and 4 at ± ``neighbour_angle``; 2 and 3 share the
    rest of the circle evenly (a stretched pentagon, p. 22)."""
    rest = (360.0 - 2 * neighbour_angle) / 3
    angles = [90.0, 90.0 + neighbour_angle, 90.0 + neighbour_angle + rest]
    angles += [angles[2] + rest, 90.0 - neighbour_angle]
    return unit_vectors(angles)


def two_digons() -> torch.Tensor:
    """Features 1–4 as two antipodal pairs on orthogonal axes; feature 0 at ≈ 0."""
    W = unit_vectors([0.0, 0.0, 90.0, 180.0, 270.0], norms=[0.01, 1, 1, 1, 1])
    return W


def test_e04_classify_perturbed() -> None:
    thr = e04.NonuniformThresholds()
    assert e04.classify_perturbed(pentagon_with_gap(72.0), 0, thr) == "pentagon"
    assert e04.classify_perturbed(two_digons(), 0, thr) == "digons (perturbed dropped)"
    other_dropped = unit_vectors([0, 0, 90, 180, 270], norms=[1, 0.01, 1, 1, 1])
    assert e04.classify_perturbed(other_dropped, 0, thr) == "digons (other dropped)"
    triangle = unit_vectors([90, 210, 330, 0, 0], norms=[1, 1, 1, 0, 0])
    assert e04.classify_perturbed(triangle, 0, thr) == "triangle"
    assert e04.neighbour_angle_deg(pentagon_with_gap(72.0), 0) == pytest.approx(72.0, abs=1e-4)
    assert e04.neighbour_angle_deg(pentagon_with_gap(80.0), 0) == pytest.approx(80.0, abs=1e-4)


def perturbed_state(angle_of_ratio, collapse_at_sparsest: bool) -> dict:
    """Two restarts per ratio: a stretched pentagon and two digons; the digons have the lower
    loss only at the sparsest ratio (when ``collapse_at_sparsest``)."""
    ratios = np.geomspace(0.1, 10, 5)
    weights, losses = [], []
    for k, r in enumerate(ratios):
        weights += [pentagon_with_gap(angle_of_ratio(r)), two_digons()]
        digon_wins = collapse_at_sparsest and k == 0
        losses += [1.0, 0.5] if digon_wins else [0.5, 1.0]
    return {
        "ratios": ratios,
        "condition": torch.arange(len(ratios)).repeat_interleave(2),
        "W": torch.stack(weights),
        "b": torch.zeros(len(weights), 5),
        "final_loss": torch.tensor(losses),
    }


def test_e04_perturbed_pentagon_passes_and_fails() -> None:
    config = e04.NonuniformGeometryConfig()
    config.perturbed.n_restarts = 2
    criteria, frame = e04.evaluate_perturbed(
        config, perturbed_state(lambda r: 72.0 + 6.0 * math.log10(r), True)
    )
    result = by_id(criteria)
    for key in ("denser_neighbours_repel", "sparser_neighbours_close_in", "collapse_to_digons"):
        assert result[key].status == PASS, (key, result[key].value)
    assert result["collapse_to_digons"].value["largest_ratio_with_collapse"] == 0.1
    assert len(frame) == 10
    rigid, _ = e04.evaluate_perturbed(config, perturbed_state(lambda r: 72.0, False))
    rigid_result = by_id(rigid)
    for key in ("denser_neighbours_repel", "sparser_neighbours_close_in", "collapse_to_digons"):
        assert rigid_result[key].status == FAIL, key


def correlation_states(ideal: bool) -> dict:
    """p. 24's three m = 2 panels and p. 25's local bases, as one restart each."""
    if ideal:
        corr_2x2 = unit_vectors([90, 0, 270, 180])  # within a pair: orthogonal
        anti_2x2 = unit_vectors([90, 270, 0, 180])  # within a pair: antipodal
        corr_3x2 = unit_vectors([0, 60, 120, 180, 240, 300])  # pairs side by side
    else:
        corr_2x2 = unit_vectors([90, 270, 0, 180])
        anti_2x2 = unit_vectors([90, 0, 270, 180])
        corr_3x2 = unit_vectors([0, 180, 60, 240, 120, 300])  # each pair antipodal
    g = torch.Generator().manual_seed(0)
    q1, _ = torch.linalg.qr(torch.randn(10, 10, generator=g))
    q2, _ = torch.linalg.qr(torch.randn(10, 10, generator=g))
    if ideal:  # each set is an orthonormal basis (p. 25); the two bases are unrelated
        local = torch.cat([q1, q2], dim=1)
    else:  # each set crowded around one direction, the two directions orthogonal
        noise = 0.1 * torch.randn(10, 20, generator=g)
        local = torch.cat([q1[:, :1].expand(10, 10), q1[:, 1:2].expand(10, 10)], dim=1) + noise

    def state(W: torch.Tensor, n_sets: int, set_size: int) -> dict:
        return {"W": W[None], "final_loss": torch.zeros(1), "n_sets": n_sets, "set_size": set_size}

    return {
        "correlated_2x2": state(corr_2x2, 2, 2),
        "anticorrelated_2x2": state(anti_2x2, 2, 2),
        "correlated_3x2": state(corr_3x2, 3, 2),
        "local_bases": state(local, 2, 10),
    }


def test_e04_correlated_sets_pass_and_fail() -> None:
    config = e04.NonuniformGeometryConfig()
    keys = (
        "anticorrelated_more_negative",
        "correlated_orthogonal_when_room",
        "correlated_adjacent_when_no_room",
        "local_almost_orthogonal_bases",
    )
    result = by_id(e04.evaluate_correlation(config, correlation_states(ideal=True)))
    for key in keys:
        assert result[key].status == PASS, (key, result[key].value)
    broken = by_id(e04.evaluate_correlation(config, correlation_states(ideal=False)))
    for key in keys:
        assert broken[key].status == FAIL, (key, broken[key].value)


def collapse_state(order: list[int]) -> dict:
    """Three correlated pairs at four densities: none, one, two, three pairs collapsed onto
    their principal components (p. 25); ``order`` permutes which density gets which."""
    hexagon = [0, 60, 120, 180, 240, 300]
    stages = []
    for n_collapsed in range(4):
        angles = list(hexagon)
        for pair in range(n_collapsed):
            mid = hexagon[2 * pair] + 30
            angles[2 * pair] = angles[2 * pair + 1] = mid
        norms = [1 / math.sqrt(2) if i // 2 < n_collapsed else 1.0 for i in range(6)]
        stages.append(unit_vectors(angles, norms))
    W = torch.stack([stages[k] for k in order])
    return {
        "densities": np.geomspace(0.01, 0.5, 4),
        "condition": torch.arange(4),
        "W": W,
        "final_loss": torch.zeros(4),
        "n_sets": 3,
        "set_size": 2,
    }


def test_e04_collapse_passes_and_fails() -> None:
    config = e04.NonuniformGeometryConfig()
    result = by_id(e04.evaluate_collapse(config, collapse_state([0, 1, 2, 3])))
    assert result["sparse_hexagon_pairs_adjacent"].status == PASS
    assert result["collapse_with_density"].status == PASS
    assert result["collapsed_sets"].value["0.5"]["collapsed_pairs"] == 3
    reversed_ = by_id(e04.evaluate_collapse(config, collapse_state([3, 2, 1, 0])))
    assert reversed_["sparse_hexagon_pairs_adjacent"].status == FAIL
    assert reversed_["collapse_with_density"].status == FAIL


def test_e04_spearman_matches_scipy_and_rejects_constant_input() -> None:
    from scipy.stats import spearmanr

    rng = np.random.default_rng(3)
    x, y = rng.integers(0, 5, 30).astype(float), rng.normal(size=30)  # x has ties
    expected = float(np.asarray(spearmanr(x, y))[0])
    assert e04.spearman(x, y) == pytest.approx(expected, abs=1e-12)
    assert math.isnan(e04.spearman(x, np.ones(30)))
