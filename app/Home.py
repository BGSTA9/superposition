"""Home page of the lab: reproduction status (live) and the paper map.

SPEC §8, page 1: "The paper map (section → pages → code → experiment → notebook) and the live
PASS/PARTIAL/FAIL table." Run with ``uv run superposition app``.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from lab_ui import PAPER_URL, missing_artifact, page_setup

from superposition import lab
from superposition.experiments import REGISTRY
from superposition.runs import latest_run, results_root

page_setup("Home")

st.title("superposition · lab")
st.markdown(
    f"An interactive companion to [*Toy Models of Superposition*]({PAPER_URL}) "
    "(Elhage et al., 2022). Train toy models live in the **Playground**, probe them feature by "
    "feature, and check every replicated claim below. Everything here calls the "
    "`superposition` library; experiment results are read from `results/`."
)

# --- live status ------------------------------------------------------------------------------
st.header("Reproduction status")
st.caption(
    "**Cited** is the run docs/REPRODUCTION.md is written from (docs/cited_runs.yaml). "
    "**Latest local run** is the newest complete run in your `results/` folder, read live. "
    f"Results folder: `{results_root()}`."
)


@st.cache_data(ttl=10, show_spinner=False)
def load_status() -> pd.DataFrame:
    return lab.status_frame()


status = load_status()
st.dataframe(
    status,
    hide_index=True,
    width="stretch",
    column_config={
        "clean tree": st.column_config.CheckboxColumn(
            "clean tree", help="Run made from a committed tree"
        ),
        "command": st.column_config.TextColumn("command to (re)run", width="large"),
    },
)
if st.button("Refresh status"):
    load_status.clear()
    st.rerun()

# --- latest figures ---------------------------------------------------------------------------
st.header("Latest figures")
for name, exp in REGISTRY.items():
    with st.expander(f"{exp.code} · {name} ({exp.paper_pages}) — {exp.title}"):
        run = latest_run(name)
        if run is None:
            missing_artifact(f"A complete run of {name}", f"uv run superposition run {name}")
            continue
        pngs = sorted(run.figures_dir.glob("*.png"))
        if not pngs:
            missing_artifact(f"Figures of run {run.run_id}", f"uv run superposition figures {name}")
            continue
        st.caption(f"Run `{run.run_id}`")
        for png in pngs:
            st.image(str(png), width="stretch")

# --- pages ------------------------------------------------------------------------------------
st.header("Lab pages")
st.markdown(
    """
| Page | What it does | Status |
|---|---|---|
| **Playground** | train a small model live; WᵀW, feature norms, Dᵢ, 2D/3D views; probe any input | available |
| Phase diagrams | E02 and E12 with the theory overlay | Phase 3 |
| Geometry | E03 curves, geometry graphs, Dᵢ histograms | Phase 3 |
| Non-uniform | E04 with a density slider for the perturbed feature | Phase 3 |
| Dynamics | E05 training-step scrubber and 3D trajectories | Phase 4 |
| Neurons | E07/E08 stack plots, E09 roles and ablations | Phase 4–5 |
| Adversarial | E06 | Phase 4 |
| SAE lab | train or load SAEs and compare them with the true features | Phase 5 |
"""
)

# --- paper map --------------------------------------------------------------------------------
st.header("Paper map")
st.caption(
    "From docs/PAPER_MAP.md: every paper section mapped to code, experiments, notebooks and lab pages."
)


@st.cache_data(show_spinner=False)
def load_paper_map() -> pd.DataFrame:
    return lab.paper_map()


paper_map = load_paper_map()
query = st.text_input("Filter the map", placeholder="e.g. geometry, p. 17, E03, metrics")
if query:
    mask = paper_map.apply(
        lambda row: row.str.contains(query, case=False, regex=False).any(), axis=1
    )
    paper_map = paper_map[mask]
st.dataframe(paper_map, hide_index=True, width="stretch")
