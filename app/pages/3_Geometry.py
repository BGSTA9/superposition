"""Geometry: dimensions per feature, feature dimensionality and polytopes (E03).

SPEC §8, page 4. Paper: Uniform Superposition, p. 17; Feature Dimensionality, pp. 18–19; Why
These Geometric Structures?, pp. 19–20. Results are read from the latest ``uniform_geometry``
run; this page never trains. Selecting a sparsity shows its geometry graph and Dᵢ histogram.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from lab_ui import missing_artifact, page_setup

from superposition import lab
from superposition.plotting.geometry import (
    dimensionality_figure,
    dstar_figure,
    geometry_graph_figure,
)
from superposition.runs import RunDir, latest_run

page_setup("Geometry")
st.title("Geometry")
st.markdown(
    "n = 400 equally important, equally sparse features in m = 30 dimensions (p. 17). "
    "**D\\* = m / ‖W‖²_F** is the number of dimensions per represented feature: it sticks at 1 "
    "and at 1/2 (antipodal pairs). Each feature's **dimensionality** "
    "Dᵢ = ‖Wᵢ‖² / Σⱼ(Ŵᵢ·Wⱼ)² clusters at fractions that name uniform polytopes: 3/4 "
    "tetrahedron, 2/3 triangle, 1/2 digon, 2/5 pentagon, 3/8 square antiprism (p. 19). In the "
    "**geometry graph** features are nodes joined when they are not orthogonal; its connected "
    "components are tegum factors (p. 20)."
)

NAME = "uniform_geometry"
run = latest_run(NAME)
if run is None:
    missing_artifact("A complete run of E03 uniform_geometry", lab.run_command(NAME))
    st.stop()


@st.cache_data(show_spinner="Loading E03 …")
def load(path: str) -> lab.UniformGeometryRun:
    return lab.load_uniform_geometry(RunDir(Path(path)))


geo = load(str(run.path))
st.caption(f"Run `{run.run_id}`")
x = geo.inverse_density
left, right = st.columns(2)
with left:
    st.pyplot(
        dstar_figure(x, geo.dstar, geo.plateau, title="D* = m / ‖W‖²_F (best restart)"),
        width="stretch",
    )
with right:
    st.pyplot(
        dimensionality_figure(x, geo.dimensionality, geo.dstar, title="every feature's Dᵢ"),
        width="stretch",
    )

default = int(np.argmin(np.abs(x - 3.0)))
point = st.select_slider(
    "sparsity 1/(1 − S)",
    options=list(range(len(x))),
    value=default,
    format_func=lambda k: f"{x[k]:.2f}",
    key="geo_point",
)
dims, represented = geo.dimensionality[point], geo.represented[point]
cols = st.columns(4)
cols[0].metric("D*", f"{geo.dstar[point]:.3f}")
cols[1].metric("represented features", f"{int(represented.sum())} / {geo.n_features}")
cols[2].metric("Σᵢ Dᵢ", f"{geo.sum_dimensionality[point]:.2f}", help=f"m = {geo.n_hidden} (p. 18)")
cols[3].metric("feature density 1 − S", f"{1 / x[point]:.3f}")

left, right = st.columns([3, 2])
with left:
    st.pyplot(
        geometry_graph_figure(
            [lab.geometry_graph_panel(geo, point)], title="Geometry graph (represented features)"
        ),
        width="stretch",
    )
    st.caption(
        "Nodes are colored by the tegum factor they belong to, named by its number of features, "
        "the dimensions it spans and whether every member has the same Dᵢ."
    )
with right:
    counts, edges = np.histogram(dims[represented], bins=np.linspace(0, 1, 41))
    st.markdown("**Dᵢ of the represented features**")
    st.bar_chart(
        pd.DataFrame({"features": counts}, index=[f"{e:.3f}" for e in edges[:-1]]), height=260
    )
    st.markdown("**Features per kind of tegum factor**")
    census = pd.DataFrame(
        sorted(geo.census[point].items(), key=lambda kv: -kv[1]), columns=["factor", "features"]
    )
    st.dataframe(census, hide_index=True, width="stretch")

with st.expander("The run's figures"):
    for png in sorted(run.figures_dir.glob("*.png")):
        st.image(str(png), width="stretch")
with st.expander("Acceptance criteria"):
    st.dataframe(lab.criteria_frame(run), hide_index=True, width="stretch")
