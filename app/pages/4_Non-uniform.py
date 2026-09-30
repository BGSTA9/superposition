"""Non-uniform: polytopes deform and snap as features stop being identical (E04).

SPEC §8, page 5. Paper: Non-Uniform Superposition, pp. 21–25. Results are read from the latest
``nonuniform_geometry`` run; this page never trains. The slider moves the perturbed feature's
density (part (a)); the tabs show correlated and anticorrelated sets (b), local
almost-orthogonal bases (c) and the collapse of correlated pairs (d).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch
from lab_ui import missing_artifact, page_setup
from matplotlib.patches import Circle

from superposition import lab, metrics
from superposition.experiments.nonuniform_geometry import (
    GEOMETRY_COLORS,
    SET_COLORS,
    classify_perturbed,
    mean_within_set_cos,
    neighbour_angle_deg,
    set_pairs,
)
from superposition.plotting.arrows import arrows_figure, plot_feature_arrows
from superposition.plotting.style import INK, INK_SECONDARY, paper_style
from superposition.plotting.weights import plot_gram
from superposition.runs import RunDir, latest_run

page_setup("Non-uniform")
st.title("Non-uniform superposition")
st.markdown(
    "Uniform superposition packs identical features into regular polytopes. When features "
    "differ in density or are correlated, the polytopes deform smoothly until they snap to "
    "another one (p. 21). Correlated features prefer to be orthogonal, then side by side, and "
    "finally collapse onto their principal component; anticorrelated features prefer to be "
    "antipodal (pp. 23–25)."
)

NAME = "nonuniform_geometry"
run = latest_run(NAME)
if run is None:
    missing_artifact("A complete run of E04 nonuniform_geometry", lab.run_command(NAME))
    st.stop()


@st.cache_data(show_spinner="Loading E04 …")
def load(path: str) -> lab.NonuniformRun:
    return lab.load_nonuniform(RunDir(Path(path)))


data = load(str(run.path))
st.caption(f"Run `{run.run_id}`")
tab_a, tab_b, tab_c, tab_d, tab_criteria = st.tabs(
    [
        "(a) perturbing one feature",
        "(b) correlated sets",
        "(c) local bases",
        "(d) collapse",
        "criteria",
    ]
)

with tab_a:
    state, thr = data.perturbed, data.thresholds
    ratios = np.asarray(state["ratios"])
    st.markdown(
        "Five features in two dimensions, all at density 0.05 except **feature 0**, whose "
        "density is scaled by the slider (p. 22). Denser, and its neighbours move away; sparser, "
        "and they close in; sparse enough, and the pentagon snaps to two digons."
    )
    one = int(np.argmin(np.abs(np.log(ratios))))
    c = st.select_slider(
        "feature 0's density relative to the others",
        options=list(range(len(ratios))),
        value=one,
        format_func=lambda k: f"{ratios[k]:.3g}×",
        key="nu_ratio",
    )
    best = lab.best_in_condition(state, c)
    W = state["W"][best]
    geometry = classify_perturbed(W, 0, thr)
    left, right = st.columns([2, 3])
    with left:
        st.pyplot(
            arrows_figure(lab.canonical_2d(W), title=f"best of the restarts: {geometry}"),
            width="stretch",
        )
        norm0 = float(metrics.feature_norms(W[None])[0, 0])
        st.caption(
            f"‖W₀‖ = {norm0:.3f}"
            + (
                f"; feature 0 and its neighbours are {neighbour_angle_deg(W, 0):.1f}° apart "
                "(72° in a regular pentagon)."
                if geometry == "pentagon"
                else "."
            )
        )
    with right:
        table = data.perturbed_table
        figure = go.Figure()
        for kind, color in GEOMETRY_COLORS.items():
            rows = table[table["geometry"] == kind]
            if len(rows):
                figure.add_trace(
                    go.Scatter(
                        x=rows["ratio"],
                        y=rows["loss"],
                        mode="markers",
                        name=kind,
                        marker={"color": color, "size": 6},
                    )
                )
        figure.add_vline(x=float(ratios[c]), line_dash="dash", line_color=INK_SECONDARY)
        figure.update_xaxes(type="log", title="feature 0's relative density")
        figure.update_yaxes(type="log", title="final loss (every restart)")
        figure.update_layout(
            height=420,
            margin={"l": 50, "r": 10, "t": 30, "b": 40},
            legend={"orientation": "h", "yanchor": "top", "y": -0.2, "x": 0},
        )
        st.plotly_chart(figure, width="stretch")
        st.caption(
            "Each geometry has its own loss curve; where the curves cross, the best solution "
            "changes (a first-order phase change, p. 22)."
        )
    restarts = pd.DataFrame(table[np.isclose(table["ratio"], ratios[c])])
    st.dataframe(
        restarts.sort_values(by="loss")[["restart", "geometry", "loss", "perturbed_norm"]],
        hide_index=True,
        width="stretch",
    )

with tab_b:
    rows = [
        ("correlated_2x2", "2 correlated pairs: orthogonal within a pair"),
        ("anticorrelated_2x2", "2 anticorrelated pairs: antipodal within a pair"),
        ("correlated_3x2", "3 correlated pairs: side by side"),
    ]
    for name, title in rows:
        s = data.correlation[name]
        sets = set_pairs(s["n_sets"], s["set_size"])
        colors = [SET_COLORS[k % 3] for k in range(s["n_sets"]) for _ in range(s["set_size"])]
        order = torch.argsort(s["final_loss"])[:5].tolist()
        st.markdown(f"**{title}** — the five lowest-loss restarts; color = set")
        with paper_style():
            fig, axes = plt.subplots(1, len(order), figsize=(2.2 * len(order), 2.4))
            for ax, k in zip(np.atleast_1d(axes), order, strict=True):
                plot_feature_arrows(
                    ax, lab.canonical_2d(s["W"][k]), annotate=False, colors=colors, limit=1.4
                )
                ax.set_title(
                    f"mean within-set cos {mean_within_set_cos(s['W'][k], sets):.2f}",
                    fontsize=7,
                    loc="left",
                )
            fig.tight_layout()
        st.pyplot(fig, width="stretch")

with tab_c:
    s = data.correlation["local_bases"]
    k = int(s["final_loss"].argmin())
    W = s["W"][k]
    st.markdown(
        f"Two correlated sets of {s['set_size']} features in m = {W.shape[0]} dimensions "
        "(pp. 24–25): within each set the features are nearly orthogonal, a local basis, even "
        "though the model as a whole is in superposition."
    )
    with paper_style():
        fig, ax = plt.subplots(figsize=(4.4, 4.2))
        plot_gram(ax, metrics.gram(W[None])[0].numpy())
        half = s["set_size"] - 0.5
        ax.axhline(half, color=INK, linewidth=0.8, linestyle=(0, (3, 2)))
        ax.axvline(half, color=INK, linewidth=0.8, linestyle=(0, (3, 2)))
        ax.set_title("WᵀW of the best restart (dashed: set boundary)", fontsize=9, loc="left")
        fig.tight_layout()
    st.pyplot(fig, width="content")

with tab_d:
    s = data.collapse
    densities = np.asarray(s["densities"])
    sets = set_pairs(s["n_sets"], s["set_size"])
    colors = [SET_COLORS[k % 3] for k in range(s["n_sets"]) for _ in range(s["set_size"])]
    st.markdown(
        "Three correlated pairs in two dimensions (p. 25). Very sparse: a hexagon with each "
        "pair side by side. Denser: pairs collapse, one after another, onto their principal "
        "component (a + b)/√2, drawn at radius 1/√2."
    )
    k = st.select_slider(
        "set density 1 − S",
        options=list(range(len(densities))),
        value=0,
        format_func=lambda j: f"{densities[j]:.3g}",
        key="nu_collapse",
    )
    best = lab.best_in_condition(s, k)
    W = s["W"][best]
    cos = metrics.cosine_similarity_matrix(W[None])[0]
    left, right = st.columns([2, 3])
    with left:
        with paper_style():
            fig, ax = plt.subplots(figsize=(3.4, 3.4))
            plot_feature_arrows(ax, lab.canonical_2d(W), annotate=False, colors=colors, limit=1.4)
            ax.add_patch(
                Circle(
                    (0, 0), 1 / np.sqrt(2), fill=False, linestyle=(0, (2, 2)), color=INK_SECONDARY
                )
            )
            fig.tight_layout()
        st.pyplot(fig, width="stretch")
    with right:
        st.dataframe(
            pd.DataFrame(
                {
                    "pair": [f"{a}, {b}" for a, b in sets],
                    "cos within the pair": [round(float(cos[a, b]), 3) for a, b in sets],
                    "norms": [
                        f"{float(W[:, a].norm()):.2f}, {float(W[:, b].norm()):.2f}" for a, b in sets
                    ],
                }
            ),
            hide_index=True,
            width="stretch",
        )
        curve = [
            mean_within_set_cos(s["W"][lab.best_in_condition(s, j)], sets)
            for j in range(len(densities))
        ]
        st.line_chart(
            pd.DataFrame({"mean within-pair cos": curve}, index=[f"{v:.3g}" for v in densities]),
            height=240,
        )

with tab_criteria:
    st.dataframe(lab.criteria_frame(run), hide_index=True, width="stretch")
