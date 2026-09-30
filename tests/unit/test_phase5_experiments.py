"""Criteria and helpers of E09–E11 on synthetic inputs (no training).

Paper: pp. 38–39 (E09), 42–44 (E10), 42 (E11).
"""

import pandas as pd
import pytest
import torch

from superposition.experiments import l1_removes_superposition as e11
from superposition.experiments import mech_interp_abs as e09
from superposition.experiments import sae_recovery as e10
from superposition.models import TiedModel
from superposition.train import importance_weighted_loss
from superposition.utils import make_generator


def e09_summary(named: int, live: int, agree: int, motifs: int = 0) -> pd.DataFrame:
    row = {
        "shape": "n6_m10",
        "graded": True,
        "instance": 0,
        "loss": 0.01,
        "neurons": live,
        "live": live,
        "dead": 0,
        "named": named,
        "spec_named": named,
        "agree": agree,
        "motifs": motifs,
        "roles": ["positive-side(0)"] * live,
        "restarts_with_motif": f"{motifs}/20",
        "rebuild_recovered": 0.9,
        "rebuild_recovered_trained_bias": 0.9,
    }
    context = {**row, "graded": False, "named": 2, "agree": 1}
    return pd.DataFrame(
        [{**row, "feature_probability": 0.3}, {**row, "feature_probability": 0.2},
         {**context, "feature_probability": 0.1}]
    )  # fmt: skip


def motif_rows(n: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"shape": "n6_m10", "feature_probability": 0.3, "graded": True, "instance": k,
             "is_best": k == 0, "loss": 0.01, "asymmetric_neuron": 3, "inhibitor_neuron": 7,
             "heavy": 0, "victim": 5, "input_ratio": 3.5, "output_ratio": 3.4,
             "interference": 3.9, "inhibition": -3.95, "net_interference": -0.05}
            for k in range(n)
        ]
    )  # fmt: skip


def test_e09_criteria_pass_and_fail() -> None:
    config = e09.MechInterpConfig()
    by_id = {c.id: c for c in e09.evaluate(config, e09_summary(10, 10, 10, 2), motif_rows(3))}
    assert by_id["named_roles"].status == "PASS"
    assert by_id["ablations_agree"].status == "PASS"
    assert by_id["motif_found"].status == "PASS"
    assert by_id["motif_found"].value["models_with_motif"] == 3
    assert by_id["motif_details"].value["net_interference_negative"] == "1/1"
    assert by_id["context_densities"].status == "INFO"
    # Named fraction and agreement are pooled over graded models only (context ignored).
    by_id = {c.id: c for c in e09.evaluate(config, e09_summary(8, 10, 8), motif_rows(0))}
    assert by_id["named_roles"].status == "FAIL"  # 16/20 = 0.8 < 0.9
    assert by_id["named_roles"].value["fraction"] == pytest.approx(0.8)
    assert by_id["ablations_agree"].status == "PASS"  # 16/16
    assert by_id["motif_found"].status == "FAIL"


def test_e09_showcase_prefers_the_configured_shape() -> None:
    config = e09.MechInterpConfig()
    summary = e09_summary(10, 10, 10, 2)
    other = summary.iloc[[0]].assign(shape="n4_m6", motifs=4)
    chosen = e09.showcase_model(config, pd.concat([summary, other], ignore_index=True))
    assert chosen is not None and chosen["shape"] == "n6_m10"
    assert e09.showcase_model(config, e09_summary(10, 10, 10, 0)) is None


def test_e10_sweep_settings_and_best_row() -> None:
    sweep = e10.SweepSpec(dictionary_sizes=[5, 10], l1_values=[0.1, 0.3], resample=[False, True],
                          topk_values=[1])  # fmt: skip
    settings = e10.sweep_settings(sweep)
    assert len(settings) == 2 * (2 * 2 + 1)
    assert settings[4] == {"size": 5, "l1": 0.0, "topk": 1, "resample": True}
    frame = pd.DataFrame(
        {"recovered_0.95": [0.8, 1.0, 1.0], "mean_max_cos": [0.99, 0.97, 0.98],
         "size": [5, 5, 10], "l1": [0.1, 0.3, 0.1], "topk": [0, 0, 0],
         "resample": [False, False, True]}
    )  # fmt: skip
    best = e10.best_row(frame, 0.95)
    assert (best["size"], best["l1"]) == (10, 0.1)  # ties broken by mean max cos
    assert e10.setting_label(best) == "10 latents, L1 λ=0.1, resample on"


def test_e10_selection_without_ground_truth() -> None:
    frame = pd.DataFrame(
        {"instance": range(5), "size": [5, 5, 10, 10, 20], "l1": [0.1, 1.0, 0.1, 0.0, 0.3],
         "topk": [0, 0, 0, 1, 0], "resample": [False, True, False, True, True],
         "l0": [0.9, 0.3, 0.6, 1.0, 0.6], "fvu": [0.02, 0.4, 0.08, 0.01, 0.05],
         "mean_max_cos": [0.9, 0.99, 0.95, 1.0, 0.97],
         "recovered_0.95": [0.4, 1.0, 0.6, 1.0, 0.8]}
    )  # fmt: skip
    chosen, n_qualified = e10.select_without_ground_truth(frame, 0.1)
    # Instance 1 is sparsest and recovers everything, but fails the FVU bar; instances 2 and 4
    # tie on L0 and the lower FVU wins. Recovery plays no part in the choice.
    assert (int(chosen["instance"]), n_qualified) == (4, 4)
    shuffled = frame.assign(**{"recovered_0.95": frame["recovered_0.95"][::-1].to_numpy()})
    assert int(e10.select_without_ground_truth(shuffled, 0.1)[0]["instance"]) == 4
    # Nothing meets the bar: fall back to the lowest FVU.
    chosen, n_qualified = e10.select_without_ground_truth(frame, 0.005)
    assert (int(chosen["instance"]), n_qualified) == (3, 0)

    spec = e10.ToyModelSpec(label="toy", cos_threshold=0.95, min_fraction=1.0, min_represented=5)
    passing = e10.selection_criterion(spec, frame, 5, fvu_max=0.1)
    assert passing.status == "FAIL"  # the selected SAE (instance 4) recovers 80%
    assert passing.value["setting"] == "20 latents, L1 λ=0.3, resample on"
    assert passing.value["sae_meeting_fvu_bar"] == "4/5"
    lenient = e10.ToyModelSpec(label="toy", cos_threshold=0.95, min_fraction=0.8)
    assert e10.selection_criterion(lenient, frame, 5, fvu_max=0.1).status == "PASS"


def test_e11_shared_evaluation_matches_the_model() -> None:
    torch.manual_seed(0)
    W = torch.randn(3, 2, 5)
    b = torch.randn(3, 5) * 0.1
    importance = 0.9 ** torch.arange(5.0)
    loss, penalty = e11.shared_evaluation(
        W, b, importance, 0.3, 4000, make_generator(1), chunk=1000
    )
    # Same samples, scored by the TiedModel itself.
    from superposition.data import IndependentSparse

    x = IndependentSparse(0.3, 1, 5).sample(4000, make_generator(1))
    x_all = torch.cat([x[i : i + 1000] for i in range(0, 4000, 1000)]).expand(-1, 3, -1)
    model = TiedModel.from_kind("relu_output", 5, 2, 3)
    with torch.no_grad():
        model.W.copy_(W)
        model.b.copy_(b)
        expected = importance_weighted_loss(model(x_all), x_all, importance)
        h = model.hidden(x_all)
    assert torch.allclose(loss, expected, rtol=1e-5)
    assert torch.allclose(penalty, h.abs().sum(-1).mean(0), rtol=1e-5)


def e11_table(
    superposed: list[int], losses: list[float], formal: list[bool] | None = None
) -> pd.DataFrame:
    n = len(superposed)
    lambdas = [0.0, *[0.1 * (i + 1) for i in range(n - 1)]]
    formal = formal if formal is not None else [s > 0 for s in superposed]
    return pd.DataFrame(
        {
            "setup": "toy", "lambda": lambdas, "instance": range(n), "loss": losses,
            "penalty": 0.0, "objective": losses,
            "represented": [max(s, 2) for s in superposed], "superposed": superposed,
            "formal_superposition": formal, "max_score": 0.0, "dims_per_feature": 0.5,
            "represented_features": [[0, 1]] * n,
        }
    )  # fmt: skip


def test_e11_criteria() -> None:
    setup = e11.L1Setup(label="toy", n_hidden=2)
    thr = e11.L1Thresholds()
    good = e11_table([5, 4, 4, 0, 0], [0.02, 0.021, 0.025, 0.07, 0.12])
    assert all(c.status == "PASS" for c in e11.evaluate_setup(setup, good, thr))
    by_id = {c.id: c for c in e11.evaluate_setup(setup, good, thr)}
    assert by_id["toy_superposition_falls_to_none"].value["first_lambda"] == pytest.approx(0.3)
    # A rise of 2 superposed features, and a 30% loss drop, both fail.
    bad = e11_table([5, 2, 4, 0, 0], [0.02, 0.048, 0.034, 0.07, 0.12])
    by_id = {c.id: c for c in e11.evaluate_setup(setup, bad, thr)}
    assert by_id["toy_superposition_declines"].status == "FAIL"
    assert by_id["toy_loss_rises"].status == "FAIL"
    assert by_id["toy_loss_rises"].value["largest_relative_drop"] == pytest.approx(0.2917, abs=1e-3)
    # Never free of superposition: the falls-to-none criterion fails and so does loss_rises.
    never = e11_table([5, 5, 4, 4, 2], [0.02, 0.02, 0.03, 0.04, 0.05])
    by_id = {c.id: c for c in e11.evaluate_setup(setup, never, thr)}
    assert by_id["toy_superposition_falls_to_none"].status == "FAIL"
    assert by_id["toy_loss_rises"].status == "FAIL"
