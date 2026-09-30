"""Phase diagrams: when is an extra feature dropped, superposed, or given its own dimension (E02)?

SPEC §8, page 3. Paper: Superposition as a Phase Change, pp. 14–16; Comments & Replications
(McGrath), pp. 51–53. Results are read from the latest ``phase_change`` run; this page never
trains. Clicking a cell (or moving the two sliders) shows the models trained there and every
theoretical family's optimum at that cell.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from lab_ui import missing_artifact, page_setup

from superposition import lab
from superposition.plotting.arrows import arrows_figure
from superposition.plotting.interactive import phase_map
from superposition.plotting.phase import REGIME_NAMES
from superposition.runs import RunDir, latest_run

page_setup("Phase diagrams")
st.title("Phase diagrams")
st.markdown(
    "Store n features in m < n dimensions and vary one **extra** feature: its importance relative "
    "to the others (x axis) and the density 1 − S of every feature (y axis). The extra feature "
    "ends up **not represented**, stored in **superposition** (an antipodal pair), or given a "
    "**dedicated dimension** (p. 14). The empirical diagram labels ten trained models per cell, "
    "drops the one with the highest loss and takes the majority of the other nine. The "
    "theoretical diagram minimizes the exact expected loss of each candidate weight family and "
    "keeps the cheapest (pp. 14–15; `superposition.theory`)."
)

NAME = "phase_change"
run = latest_run(NAME)
if run is None:
    missing_artifact("A complete run of E02 phase_change", lab.run_command(NAME))
    st.stop()


@st.cache_data(show_spinner="Loading E02 …")
def load(path: str) -> lab.PhaseChangeRun:
    return lab.load_phase_change(RunDir(Path(path)))


data = load(str(run.path))
st.caption(f"Run `{run.run_id}`")
titles = {label: f"n = {p.n_features}, m = {p.n_hidden}" for label, p in data.problems.items()}
left, right = st.columns([3, 2])
with left:
    label = st.radio(
        "problem", list(titles), format_func=titles.__getitem__, horizontal=True, key="pc_problem"
    )
    problem = data.problems[label]
    views = {
        "empirical (majority of 9)": problem.grid("empirical"),
        "theory: the paper's families": problem.grid("theory"),
        "theory + McGrath's confused family": problem.grid("theory_with_confused"),
        "theory with every bias fixed at 0": problem.grid("theory_no_bias"),
    }
    view = st.radio("show", list(views), horizontal=True, key="pc_view")
    mark = st.checkbox(
        "mark cells where the empirical label differs from the theory (×)",
        value=True,
        key="pc_mark",
    )

n_dens, n_imp = len(problem.density), len(problem.importance)
# A click on the map (handled below the map) leaves the clicked cell here for this rerun;
# widget state can only be set before the sliders are drawn.
pending = st.session_state.pop("pc_pending", None)
if pending is not None:
    st.session_state["pc_density"], st.session_state["pc_importance"] = pending
for key, size, default in (("pc_density", n_dens, 0), ("pc_importance", n_imp, n_imp // 2)):
    if st.session_state.get(key, size) >= size:
        st.session_state[key] = default
with right:
    d = st.select_slider(
        "feature density 1 − S",
        options=list(range(n_dens)),
        format_func=lambda k: f"{problem.density[k]:.3g}",
        key="pc_density",
    )
    i = st.select_slider(
        "relative importance of the extra feature",
        options=list(range(n_imp)),
        format_func=lambda k: f"{problem.importance[k]:.3g}×",
        key="pc_importance",
    )
    row = problem.cells.iloc[d * n_imp + i]
    gap = float(row["theory_relative_gap"])
    gap_text = f"{100 * gap:.1f}% above" if gap < 1 else f"{1 + gap:.3g}× as large as"
    st.markdown(
        f"**Empirical majority:** {REGIME_NAMES[str(row['empirical'])]}  \n"
        f"**Theory (paper families):** {REGIME_NAMES[str(row['theory'])]}"
    )
    st.caption(
        f"Votes of the nine kept models: {row['counts']}. The theory's second-best family's "
        f"loss is {gap_text} the best one's"
        + (" (next to a theoretical boundary)." if bool(row["near_theory_boundary"]) else ".")
    )

with left:
    figure = phase_map(
        views[view],
        problem.importance,
        problem.density,
        overlay=problem.grid("theory") if mark and view.startswith("empirical") else None,
        selected=(d, i),
        title=f"E02 · {titles[label]}: {view}",
    )
    event = st.plotly_chart(
        figure, key="pc_map", on_select="rerun", selection_mode="points", width="stretch"
    )
    points = event.selection.points if event and event.selection else []
    if points:
        clicked = (round(float(points[0]["y"])), round(float(points[0]["x"])))
        # The selection persists across reruns: act on a click only once, so that moving the
        # sliders afterwards is not undone by the old selection.
        if clicked != st.session_state.get("pc_last_click"):
            st.session_state["pc_last_click"] = clicked
            if clicked != (d, i):
                st.session_state["pc_pending"] = clicked
                st.rerun()

st.subheader(
    f"The cell at 1 − S = {problem.density[d]:.3g}, importance {problem.importance[i]:.3g}×"
)
models = lab.phase_cell_models(problem, d, i, data.discard_worst)
theory = lab.phase_cell_theory(problem, d, i)
cols = st.columns(2)
with cols[0]:
    st.markdown("**Trained models**, lowest loss first (the last one does not vote)")
    st.dataframe(
        models,
        hide_index=True,
        width="stretch",
        column_config={"loss": st.column_config.NumberColumn(format="%.3e")},
    )
with cols[1]:
    st.markdown("**Theory**: each family's optimal exact loss here")
    st.dataframe(
        theory,
        hide_index=True,
        width="stretch",
        column_config={
            "optimal loss": st.column_config.NumberColumn(format="%.3e"),
            "loss with b = 0": st.column_config.NumberColumn(format="%.3e"),
        },
    )

# Representative weights: the best voting model of each regime, next to the theory's winner.
kept = pd.DataFrame(models[models["votes"]])
representatives = pd.DataFrame(kept.groupby("regime", sort=False).head(1))
importance = np.ones(problem.n_features)
importance[-1] = problem.importance[i]
panels = [
    (f"trained · {REGIME_NAMES[r['regime']]} · loss {r['loss']:.2e}", np.array(r["W"]))
    for r in representatives.to_dict("records")
]
winner = theory[theory["paper family"]].iloc[0]
panels.append((f"theory · {winner['family']}", np.array(winner["W"])))
if problem.n_hidden == 1:
    st.markdown("**Representative weights** (m = 1: each feature is a signed number)")
    st.bar_chart(
        pd.DataFrame(
            {title: W[0] for title, W in panels},
            index=[f"feature {k}" + (" (extra)" if k == problem.n_features - 1 else "")
                   for k in range(problem.n_features)],
        ),
        stack=False,
        height=280,
    )  # fmt: skip
else:
    st.markdown("**Representative weights** (feature 2, the extra feature, is drawn last)")
    figure_cols = st.columns(len(panels))
    for col, (title, W) in zip(figure_cols, panels, strict=True):
        with col:
            st.pyplot(arrows_figure(W, title=title, importance=importance), width="stretch")
st.caption(
    "Trained models are shown as trained (not canonicalized); a model's regime is decided by the "
    "extra feature's norm (> 0.5) and its overlap with the other represented features."
)

with st.expander("The run's figures"):
    for png in sorted(run.figures_dir.glob(f"*{label}.png")):
        st.image(str(png), width="stretch")
with st.expander("Acceptance criteria"):
    st.dataframe(lab.criteria_frame(run), hide_index=True, width="stretch")
