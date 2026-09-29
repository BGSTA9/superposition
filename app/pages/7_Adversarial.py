"""Adversarial: superposition makes models vulnerable to small input perturbations (E06).

SPEC §8, page 8. Paper: Relationship to Adversarial Robustness, pp. 29–30. Results are read
from the latest ``adversarial`` run; this page never trains.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
from lab_ui import missing_artifact, page_setup

from superposition import lab
from superposition.plotting.adversarial import (
    VulnerabilityCurve,
    vulnerability_figure,
    vulnerability_scatter,
)
from superposition.runs import RunDir, latest_run

page_setup("Adversarial")
st.title("Adversarial")
st.markdown(
    "Superposition leaves small interference terms in WᵀW, (WᵀW)₀ = (1, ε, −ε, ε, …), and each "
    "one is a direction an adversary can push along (p. 29). The attack here is the paper's: "
    "for each feature i, move the input by ε along row i of WᵀW (the direction that raises "
    "output i the most), with ε = 0.1 × the average input norm, and keep whichever feature's "
    "attack hurts each input most. **Vulnerability** is the damage relative to a model without "
    "superposition at the same sparsity and budget, (L′ − L)/(L′₀ − L₀)."
)

NAME = "adversarial"
run = latest_run(NAME)
if run is None:
    missing_artifact("A complete run of E06 adversarial", lab.run_command(NAME))
    st.stop()


@st.cache_data(show_spinner="Loading E06 …")
def load(path: str) -> pd.DataFrame:
    return RunDir(Path(path)).read_table("vulnerability")


table = load(str(run.path))
st.caption(f"Run `{run.run_id}`")
curves = [
    VulnerabilityCurve(
        label=str(label),
        inverse_density=group["inverse_density"].to_numpy(),
        vulnerability=group["vulnerability"].to_numpy(),
        features_per_dimension=group["features_per_dimension"].to_numpy(),
    )
    for label, group in table.sort_values("inverse_density").groupby("setting", sort=False)
]
labels = [c.label for c in curves]
shown = st.multiselect("settings", labels, default=labels)
selected = [c for c in curves if c.label in shown]
if selected:
    st.pyplot(vulnerability_figure(selected), width="stretch")
    st.caption(
        "Top: vulnerability relative to the no-superposition model; bottom: features per "
        "dimension 1/D* = ‖W‖²_F/m. The paper's p. 29 figure uses the uniform setup of p. 17 over "
        "1/(1 − S) ∈ [1, 10]."
    )
    st.pyplot(vulnerability_scatter(selected), width="content")
setting = st.selectbox("per-sparsity table", labels)
columns = [
    "inverse_density",
    "features_per_dimension",
    "vulnerability",
    "epsilon",
    "clean_loss",
    "attacked_loss",
    "reference_damage",
    "most_attacked_feature",
    "most_attacked_share",
]
st.dataframe(
    table[table["setting"] == setting][columns],
    hide_index=True,
    width="stretch",
    column_config={c: st.column_config.NumberColumn(format="%.4g") for c in columns},
)
with st.expander("Acceptance criteria"):
    st.dataframe(lab.criteria_frame(run), hide_index=True, width="stretch")
