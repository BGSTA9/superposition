"""SAE lab: sparse autoencoders scored against a toy model's true features, and the L1 route.

SPEC §8, page 9: "Pick a trained toy model. Train a small SAE live, or load one. Compare decoder
directions with the true features, in a 2D view when m = 2. Show the recovery metrics." Paper:
Approach 2: Finding an Overcomplete Basis, pp. 42–44 (E10), and Approach 1: Creating Models
Without Superposition, p. 42 (E11). Live training is small; sweeps are read from the latest
``sae_recovery`` and ``l1_removes_superposition`` runs.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
from lab_ui import missing_artifact, page_setup

from superposition import lab, metrics
from superposition.plotting.arrows import arrows_figure
from superposition.plotting.l1 import tradeoff_curve_figure, tradeoff_figure
from superposition.plotting.sae import dictionary_2d_figure, recovery_sweep_figure
from superposition.plotting.weights import WeightsPanel, weights_grid
from superposition.runs import RunDir, latest_run

page_setup("SAE lab")
st.title("SAE lab")
st.markdown(
    "A **sparse autoencoder** (SAE) tries to undo superposition after the fact: it rewrites each "
    "hidden activation h as a sparse combination of learned directions (its decoder columns). "
    "In a real model nobody knows the true features, but in a toy model we do (p. 44), so we "
    "can check whether the SAE's directions are the model's feature directions Wᵢ."
)


@st.cache_data(show_spinner="Training the toy model …")
def toy_model(preset: str) -> lab.PlaygroundResult:
    return lab.train_playground(lab.SAE_TOY_PRESETS[preset])


@st.cache_data(show_spinner="Training the SAE …")
def live_sae(preset: str, config: dict) -> lab.LiveSAEResult:
    return lab.train_live_sae(toy_model(preset), lab.LiveSAEConfig(**config))


@st.cache_data(show_spinner="Loading E10 …")
def load_e10(path: str) -> lab.SAERecoveryRun:
    return lab.load_sae_recovery(RunDir(Path(path)))


@st.cache_data(show_spinner="Loading E11 …")
def load_e11(path: str) -> lab.L1Run:
    return lab.load_l1(RunDir(Path(path)))


tab_live, tab_e10, tab_e11 = st.tabs(
    ["Train an SAE live", "E10 · SAE sweeps (pp. 42–44)", "E11 · L1 on h (p. 42)"]
)

with tab_live:
    left, right = st.columns([1, 2])
    with left:
        preset = st.selectbox("toy model", list(lab.SAE_TOY_PRESETS), key="sae_toy")
        n_latents = st.slider(
            "dictionary size (latents)", *lab.SAE_LATENTS_RANGE, 10, key="sae_latents"
        )
        topk = st.slider("TopK k (0 = ReLU + L1)", *lab.SAE_TOPK_RANGE, 0, key="sae_topk")
        l1 = st.select_slider(
            "L1 coefficient λ", [0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0], 0.3, key="sae_l1",
            disabled=topk > 0,
        )  # fmt: skip
        resample = st.checkbox("resample dead latents", True, key="sae_resample")
        steps = st.select_slider(
            "SAE steps", [500, 1000, 2000, 3000, 5000, 10000], 3000, key="sae_steps"
        )
        seed = st.number_input("seed", 0, 10_000, 0, key="sae_seed")
    config = lab.LiveSAEConfig(
        n_latents=int(n_latents), l1=0.0 if topk else float(l1), topk=int(topk),
        resample=bool(resample), steps=int(steps), seed=int(seed),
    )  # fmt: skip
    toy = toy_model(preset)
    result = live_sae(preset, asdict(config))
    with right:
        cols = st.columns(3)
        cols[0].metric("recovered at cos > 0.9", f"{result.recovered_09:.0%}")
        cols[1].metric("recovered at cos > 0.95", f"{result.recovered_095:.0%}")
        cols[2].metric("mean max cos", f"{result.mean_max_cos:.3f}")
        cols = st.columns(3)
        cols[0].metric("L0 (active latents / sample)", f"{result.l0:.3f}")
        cols[1].metric("fraction of variance unexplained", f"{result.fvu:.3f}")
        cols[2].metric("dead latents", f"{result.dead_fraction:.0%}")
        st.caption(
            f"Toy model: best of {toy.config.restarts} restarts, loss {toy.best_loss:.5f}, "
            f"{int(result.represented.sum())} features represented · SAE trained in "
            f"{result.train_seconds:.1f} s, {result.resampled} latents resampled"
        )
        if toy.config.n_hidden == 2:
            st.pyplot(
                dictionary_2d_figure(
                    toy.best_W,
                    [
                        (
                            "live SAE",
                            result.decoder,
                            np.ones_like(result.alive),
                            result.alive,
                            result.recovered_095,
                        )
                    ],
                    0.95,
                ),
                width="content",
            )
    st.subheader("Per feature")
    st.dataframe(
        pd.DataFrame(
            {
                "feature": range(len(result.max_cos)),
                "importance": toy.importance,
                "‖Wᵢ‖": np.linalg.norm(toy.best_W, axis=0).round(3),
                "represented": result.represented,
                "max cos with a live latent": np.round(result.max_cos, 4),
                "best latent": result.best_latent,
            }
        ),
        hide_index=True,
        width="stretch",
    )

with tab_e10:
    run = latest_run("sae_recovery")
    if run is None:
        missing_artifact("A complete run of E10 sae_recovery", lab.run_command("sae_recovery"))
    else:
        e10 = load_e10(str(run.path))
        st.caption(f"Run `{run.run_id}`")
        label = st.selectbox("toy model", list(e10.toys), key="e10_toy")
        rows = e10.table.loc[(e10.table["toy_model"] == label).to_numpy()]
        st.pyplot(recovery_sweep_figure(rows, e10.thresholds), width="stretch")
        toy = e10.toys[label]
        if toy["n_hidden"] == 2:
            names = [
                f"{int(r['size'])} latents, "
                + (f"TopK {int(r['topk'])}" if r["topk"] else f"λ = {r['l1']:g}")
                + (", resample" if r["resample"] else "")
                for r in rows.to_dict("records")
            ]
            pick = st.selectbox("SAE", names, key="e10_sae")
            k = int(rows.iloc[names.index(pick)]["instance"])
            col = f"recovered_{e10.thresholds[label]:g}"
            st.pyplot(
                dictionary_2d_figure(
                    toy["W"],
                    [
                        (
                            pick,
                            toy["decoder"][k],
                            toy["mask"][k],
                            toy["alive"][k],
                            float(rows.iloc[names.index(pick)][col]),
                        )
                    ],
                    e10.thresholds[label],
                ),
                width="content",
            )
        st.dataframe(rows, hide_index=True, width="stretch")
        with st.expander("Acceptance criteria"):
            st.dataframe(lab.criteria_frame(run), hide_index=True, width="stretch")

with tab_e11:
    run = latest_run("l1_removes_superposition")
    if run is None:
        missing_artifact(
            "A complete run of E11 l1_removes_superposition",
            lab.run_command("l1_removes_superposition"),
        )
    else:
        e11 = load_e11(str(run.path))
        st.caption(
            'Adding λ‖h‖₁ to the loss: "quite easy to get rid of superposition … albeit at the '
            f'cost of a higher loss" (p. 42) · run `{run.run_id}`'
        )
        st.pyplot(tradeoff_figure(e11.table, e11.n_hidden), width="stretch")
        st.pyplot(tradeoff_curve_figure(e11.table), width="stretch")
        setup = st.selectbox("setup", list(e11.weights), key="e11_setup")
        lam = st.select_slider("λ", list(e11.weights[setup]), key="e11_lambda")
        W, b = e11.weights[setup][lam]
        if e11.n_hidden[setup] == 2:
            st.pyplot(arrows_figure(W, title=f"λ = {lam:g} (raw hidden basis)"), width="content")
        else:
            Wt = torch.as_tensor(W)[None]
            panel = WeightsPanel(
                title=f"λ = {lam:g}",
                gram=metrics.gram(Wt)[0].numpy(),
                bias=b,
                norms=metrics.feature_norms(Wt)[0].numpy(),
                superposition=metrics.superposition_score(Wt)[0].numpy(),
            )
            st.pyplot(weights_grid([panel]), width="content")
        st.dataframe(
            e11.table.loc[(e11.table["setup"] == setup).to_numpy()],
            hide_index=True,
            width="stretch",
        )
        with st.expander("Acceptance criteria"):
            st.dataframe(lab.criteria_frame(run), hide_index=True, width="stretch")
