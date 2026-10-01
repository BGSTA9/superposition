"""E12's report logic on hand-built grids (no training).

Paper: Comments & Replications, p. 51 (Redwood: the phase diagrams depend on the activation);
Superposition as a Phase Change, pp. 14–16 (E02, whose labels and vote E12 reuses). E12 has no
pass/fail (SPEC §7 E12), so these tests pin down what each reported number means.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from superposition import theory
from superposition.acceptance import INFO, overall_status
from superposition.experiments import activation_functions as e12
from superposition.experiments import phase_change as e02


def small_config() -> e12.ActivationFunctionsConfig:
    """A 4 × 4 grid with a coarse, fast optimizer (the logic, not the numbers, is tested)."""
    config = e12.ActivationFunctionsConfig(
        activations=["relu", "gelu"],
        grid=e02.PhaseGrid(n_importance=4, n_density=4),
        models_per_cell=3,
    )
    config.theory.scale_grid, config.theory.bias_grid = 17, 49
    config.theory.golden_iterations, config.theory.quadrature_nodes = 20, 8
    config.theory.check_nodes = 12
    return config


def theory_models(
    setting: e02.ProblemSetting, config: e12.ActivationFunctionsConfig, predicted: e02.TheoryResult
) -> e02.PhaseGridResult:
    """Every model of every cell set to the winning paper family's optimum; the last model of
    each cell has the highest sampled loss (it does not vote)."""
    grid = config.grid
    n_dens, n_imp, models = grid.n_density, grid.n_importance, config.models_per_cell
    n_paper = len(predicted.family_labels) - 1
    winner = predicted.losses[..., :n_paper].argmin(-1)
    d, i = np.meshgrid(np.arange(n_dens), np.arange(n_imp), indexing="ij")
    W = predicted.W[winner, d, i].reshape(n_dens * n_imp, setting.n_hidden, setting.n_features)
    b = predicted.b[winner, d, i].reshape(n_dens * n_imp, setting.n_features)
    cell = torch.arange(n_dens * n_imp).repeat_interleave(models)
    return e02.PhaseGridResult(
        setting=setting.label,
        W=torch.tensor(W, dtype=torch.float32)[cell],
        b=torch.tensor(b, dtype=torch.float32)[cell],
        final_loss=torch.tensor(np.tile(np.arange(models), n_dens * n_imp), dtype=torch.float32),
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


@pytest.fixture(scope="module")
def frames_and_checks() -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    config, setting = small_config(), e02.ProblemSetting()
    frames, checks = {}, {}
    for activation in config.activations:
        predicted, checks[activation] = e12.compute_activation_theory(setting, config, activation)
        models = theory_models(setting, config, predicted)
        frames[activation] = e12.activation_frame(config, activation, models, predicted)
    return frames, checks


def by_key(criteria: list) -> dict:
    return {(c.group, c.id): c for c in criteria}


def test_theory_weights_reproduce_each_activations_theory(frames_and_checks) -> None:
    """Labeled by E02's rule, each activation's own optimal weights give its theory's labels,
    and their exact loss is the theory's optimum (the best model sits on the theory)."""
    frames, checks = frames_and_checks
    config, setting = small_config(), e02.ProblemSetting()
    criteria = e12.evaluate_activations(config, setting, frames, checks)
    assert all(c.status == INFO for c in criteria) and overall_status(criteria) == INFO
    result = by_key(criteria)
    for activation in config.activations:
        group = f"n2_m1 · {activation}"
        agreement = result[(group, "agreement_with_own_theory")].value
        assert agreement["paper_families"] == 1.0
        frame = frames[activation]
        assert (frame["best_loss_source"] == "exact").all()
        ratio = frame["best_loss"] / frame["theory_loss"]
        assert np.allclose(ratio, 1.0, rtol=1e-4)
        check = result[(group, "quadrature_check")].value
        assert check["cells_changing_label"] == 0
    # The ReLU is the reference: it has no comparison row of its own.
    assert ("n2_m1 · relu", "vs_reference") not in result
    assert ("n2_m1 · gelu", "vs_reference") in result


def test_relu_quadrature_check_is_exact(frames_and_checks) -> None:
    _, checks = frames_and_checks
    assert checks["relu"]["max_abs_change"] < 1e-12  # closed form both times


def test_changes_against_the_reference_are_counted(frames_and_checks) -> None:
    frames, checks = frames_and_checks
    config, setting = small_config(), e02.ProblemSetting()
    same = {"relu": frames["relu"], "gelu": frames["relu"].assign(activation="gelu")}
    result = by_key(e12.evaluate_activations(config, setting, same, checks))
    row = result[("n2_m1 · gelu", "vs_reference")].value
    assert row["empirical_same"] == 1.0 and row["empirical_changes"] == {}
    changed = frames["relu"].copy()
    first = changed.index[0]
    before = str(changed.loc[first, "empirical"])
    after = theory.CONFUSED if before != theory.CONFUSED else theory.DEDICATED
    changed.loc[first, "empirical"] = after
    result = by_key(
        e12.evaluate_activations(config, setting, {"relu": frames["relu"], "gelu": changed}, checks)
    )
    row = result[("n2_m1 · gelu", "vs_reference")].value
    assert row["empirical_changes"] == {f"{before} → {after}": 1}
    assert row["empirical_same"] == pytest.approx(1 - 1 / len(changed))


def test_superposition_share_and_ranking(frames_and_checks) -> None:
    frames, checks = frames_and_checks
    config, setting = small_config(), e02.ProblemSetting()
    more = frames["gelu"].copy()
    more["empirical"] = theory.SUPERPOSITION
    result = by_key(
        e12.evaluate_activations(config, setting, {"relu": frames["relu"], "gelu": more}, checks)
    )
    value = result[("n2_m1", "superposition_share")].value
    assert value["shares"]["gelu"]["empirical"] == 1.0
    expected = float((frames["relu"]["empirical"] == theory.SUPERPOSITION).mean())
    assert value["shares"]["relu"]["empirical"] == pytest.approx(expected, abs=1e-4)
    assert value["ranking_empirical"][0] == "gelu"


def test_exact_model_losses_fall_back_when_outputs_read_two_features() -> None:
    """n = 3 models in general position have no exact loss: the run's sampled losses are used."""
    setting = e02.ProblemSetting("n3_m2", 3, 2, "p. 16")
    grid = e02.PhaseGridResult(
        setting=setting.label,
        W=torch.tensor([[[1.0, 0.5, -0.5], [0.0, 0.8, 0.6]]]),
        b=torch.zeros(1, 3),
        final_loss=torch.zeros(1),
        extra_importance=torch.ones(1),
        density=torch.full((1,), 0.1),
        cell=torch.zeros(1, dtype=torch.long),
        importance_grid=np.ones(1),
        density_grid=np.full(1, 0.1),
        models_per_cell=1,
        wall_time_s=0.0,
    )
    assert e12.exact_model_losses(grid, 1.0, "gelu") is None
