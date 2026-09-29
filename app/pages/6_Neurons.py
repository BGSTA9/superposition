"""Neurons: superposition in a privileged basis (E07) and computation in superposition (E08).

SPEC §8, page 7: "E07 and E08 stack plots across sparsities, plus the E09 role tables and
ablations" (the E09 part arrives in Phase 5). Paper: pp. 31–39. Results are read from the
latest ``privileged_basis`` and ``abs_computation`` runs; this page never trains.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from lab_ui import missing_artifact, page_setup

from superposition import lab
from superposition.plotting.neurons import (
    abs_weights_figure,
    capacity_figure,
    neuron_count_figure,
    neuron_rows,
)
from superposition.runs import RunDir, latest_run

page_setup("Neurons")
st.title("Neurons")
st.markdown(
    "Once the hidden layer has a ReLU, its neurons form a **privileged basis** and W can be read "
    "neuron by neuron (p. 31). In a **stack plot** each column is a neuron and each block one of "
    "its weights, colored by that feature's superposition score (dark = none, bright = ≥ 1); "
    "negative weights hang below the axis (p. 32). Lavender columns are monosemantic neurons, "
    "yellow ones polysemantic."
)


@st.cache_data(show_spinner="Loading E07 …")
def load_e07(path: str) -> tuple[list[lab.NeuronModel], pd.DataFrame, dict | None]:
    return lab.load_privileged_basis(RunDir(Path(path)))


@st.cache_data(show_spinner="Loading E08 …")
def load_e08(path: str) -> lab.AbsRun:
    return lab.load_abs_computation(RunDir(Path(path)))


tab_e07, tab_e08, tab_e09 = st.tabs(
    [
        "E07 · privileged basis (pp. 31–34)",
        "E08 · computing |x| (pp. 34–38)",
        "E09 · roles and ablations (Phase 5)",
    ]
)

with tab_e07:
    run = latest_run("privileged_basis")
    if run is None:
        missing_artifact(
            "A complete run of E07 privileged_basis", lab.run_command("privileged_basis")
        )
    else:
        models, summary, variant = load_e07(str(run.path))
        st.caption(f"ReLU hidden layer model, n = 10, m = 5, Iᵢ = 0.75ⁱ · run `{run.run_id}`")
        label = st.select_slider(
            "feature density 1 − S", options=[m.label for m in models], key="e07_density"
        )
        chosen = next(m for m in models if m.label == label)
        left, right = st.columns([3, 2])
        with left:
            st.pyplot(neuron_rows([chosen.panel]), width="content")
        with right:
            st.metric("monosemantic neurons", int(chosen.summary["monosemantic"]))
            st.metric("polysemantic neurons", int(chosen.summary["polysemantic"]))
            st.metric("features represented", int(chosen.summary["represented"]))
            st.caption(f"features per neuron: {list(chosen.summary['features_per_neuron'])}")
        st.pyplot(
            neuron_count_figure(
                summary["feature_probability"].to_numpy(),
                summary["monosemantic"].to_numpy(),
                summary["polysemantic"].to_numpy(),
                title="Monosemantic → polysemantic as sparsity rises (p. 32)",
            ),
            width="content",
        )
        if variant:
            st.markdown(
                "**Sidestepping the ReLU (p. 34).** Given a hidden bias, the model makes it "
                "positive so the neurons stay linear; without one it has to use the ReLU. "
                "“clipped” is the share of (input, neuron) pairs where the ReLU zeroes a "
                "negative pre-activation."
            )
            st.dataframe(pd.DataFrame(variant).T, width="stretch")
        with st.expander("Acceptance criteria"):
            st.dataframe(lab.criteria_frame(run), hide_index=True, width="stretch")

with tab_e08:
    run = latest_run("abs_computation")
    if run is None:
        missing_artifact(
            "A complete run of E08 abs_computation", lab.run_command("abs_computation")
        )
    else:
        abs_run = load_e08(str(run.path))
        st.caption(
            "Absolute-value model h = ReLU(W₁x), y′ = ReLU(W₂h + b), inputs U[−1, 1] when active, "
            f"target |x|; weights normalized as in footnote 15 · run `{run.run_id}`"
        )
        st.subheader("(a) n = 3, m = 6: one positive-side and one negative-side neuron each")
        st.pyplot(
            abs_weights_figure(
                abs_run.small_W1,
                abs_run.small_W2,
                order=pd.Series(range(len(abs_run.small_roles))).to_numpy(),
                roles=abs_run.small_roles,
            ),
            width="content",
        )
        st.subheader("(b) n = 100, m = 40, Iᵢ = 0.8ⁱ: |W₁| per neuron across sparsity")
        label = st.select_slider(
            "feature density 1 − S", options=[m.label for m in abs_run.sweep], key="e08_density"
        )
        chosen = next(m for m in abs_run.sweep if m.label == label)
        st.pyplot(neuron_rows([chosen.panel], absolute=True, show_matrix=False), width="stretch")
        s = chosen.summary
        cols = st.columns(4)
        cols[0].metric(
            "features computed well",
            int(s["computed_well"]),
            help="loss < 50% of predicting the mean",
        )
        cols[1].metric("monosemantic neurons", int(s["monosemantic"]))
        cols[2].metric("polysemantic neurons", int(s["polysemantic"]))
        cols[3].metric(
            "primary + secondary neurons", int(s["primary_plus_secondary"]), help="p. 38"
        )
        summary = abs_run.summary
        st.pyplot(
            capacity_figure(
                summary["feature_probability"].to_numpy(),
                summary["computed_well"].to_numpy(),
                summary["monosemantic"].to_numpy(),
                summary["polysemantic"].to_numpy(),
                abs_run.n_hidden,
                title="Computing |x| for more than m/2 features needs superposition",
            ),
            width="content",
        )
        with st.expander("Acceptance criteria"):
            st.dataframe(lab.criteria_frame(run), hide_index=True, width="stretch")

with tab_e09:
    st.info(
        "Neuron role tables, ablations and the asymmetric-superposition motif (pp. 38–39) come "
        "with `interp.py` in Phase 5 (E09 mech_interp_abs)."
    )
