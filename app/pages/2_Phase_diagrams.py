"""Phase diagrams: when is an extra feature dropped, superposed, or given its own dimension?

SPEC §8, page 3: E02 and E12 with the theory overlay. Paper: Superposition as a Phase Change,
pp. 14–16; Comments & Replications, pp. 51–53 (McGrath's confused feature; Redwood's remark
that the diagrams depend on the activation function, p. 51). Results are read from the latest
``phase_change`` and ``activation_functions`` runs; this page never trains. Two sliders pick a
cell (the map outlines it; hovering shows any cell's labels), and the page shows the models
trained there and every theoretical family's optimum at that cell. SPEC §8 asks for clicking a
cell; docs/DECISIONS.md D-311 records why the page uses sliders instead.
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


@st.cache_data(show_spinner="Loading E02 …")
def load_e02(path: str) -> lab.PhaseChangeRun:
    return lab.load_phase_change(RunDir(Path(path)))


@st.cache_data(show_spinner="Loading E12 …")
def load_e12(path: str) -> lab.ActivationFunctionsRun:
    return lab.load_activation_functions(RunDir(Path(path)))


def explore(
    problem: lab.PhaseProblem,
    views: dict[str, np.ndarray],
    title: str,
    discard_worst: int,
    prefix: str,
) -> None:
    """The phase map with a selected cell, the cell's models and the families' optima there."""
    left, right = st.columns([3, 2])
    with left:
        view = st.radio("show", list(views), horizontal=True, key=f"{prefix}_view")
        mark = st.checkbox(
            "mark cells where the empirical label differs from the theory (×)",
            value=True,
            key=f"{prefix}_mark",
        )
    n_dens, n_imp = len(problem.density), len(problem.importance)
    for key, size, default in (
        (f"{prefix}_density", n_dens, 0),
        (f"{prefix}_importance", n_imp, n_imp // 2),
    ):
        if st.session_state.get(key, size) >= size:
            st.session_state[key] = default
    with right:
        d = st.select_slider(
            "feature density 1 − S",
            options=list(range(n_dens)),
            format_func=lambda k: f"{problem.density[k]:.3g}",
            key=f"{prefix}_density",
        )
        i = st.select_slider(
            "relative importance of the extra feature",
            options=list(range(n_imp)),
            format_func=lambda k: f"{problem.importance[k]:.3g}×",
            key=f"{prefix}_importance",
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
            title=f"{title}: {view}",
        )
        st.plotly_chart(figure, key=f"{prefix}_map", width="stretch")
        st.caption(
            "Pick a cell with the two sliders; hover over the map to read any cell's labels."
        )

    st.subheader(
        f"The cell at 1 − S = {problem.density[d]:.3g}, importance {problem.importance[i]:.3g}×"
    )
    models = lab.phase_cell_models(problem, d, i, discard_worst)
    theory = lab.phase_cell_theory(problem, d, i)
    if bool(np.isnan(theory["loss with b = 0"].to_numpy(dtype=float)).all()):  # E12: no b = 0
        theory = theory.drop(columns=["loss with b = 0"])
    cols = st.columns(2)
    with cols[0]:
        exact = "exact loss" in models
        st.markdown(
            "**Trained models**, lowest "
            + ("exact loss" if exact else "loss")
            + " first (the model with the highest sampled loss does not vote)"
        )
        st.dataframe(
            models,
            hide_index=True,
            width="stretch",
            column_config={
                "loss": st.column_config.NumberColumn("sampled loss", format="%.3e"),
                "exact loss": st.column_config.NumberColumn(format="%.3e"),
            },
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
        (
            f"trained · {REGIME_NAMES[r['regime']]} · loss {r.get('exact loss', r['loss']):.2e}",
            np.array(r["W"]),
        )
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
        for col, (panel_title, W) in zip(figure_cols, panels, strict=True):
            with col:
                st.pyplot(
                    arrows_figure(W, title=panel_title, importance=importance), width="stretch"
                )
    st.caption(
        "Trained models are shown as trained (not canonicalized); a model's regime is decided by "
        "the extra feature's norm (> 0.5) and its overlap with the other represented features."
    )


def problem_title(problem: lab.PhaseProblem) -> str:
    return f"n = {problem.n_features}, m = {problem.n_hidden}"


e02_tab, e12_tab = st.tabs(["E02 · the ReLU output model", "E12 · other output activations"])

with e02_tab:
    run = latest_run("phase_change")
    if run is None:
        missing_artifact("A complete run of E02 phase_change", lab.run_command("phase_change"))
    else:
        data = load_e02(str(run.path))
        st.caption(f"Run `{run.run_id}`")
        titles = {label: problem_title(p) for label, p in data.problems.items()}
        label = st.radio(
            "problem", list(titles), format_func=titles.__getitem__, horizontal=True,
            key="pc_problem",
        )  # fmt: skip
        problem = data.problems[label]
        explore(
            problem,
            {
                "empirical (majority of 9)": problem.grid("empirical"),
                "theory: the paper's families": problem.grid("theory"),
                "theory + McGrath's confused family": problem.grid("theory_with_confused"),
                "theory with every bias fixed at 0": problem.grid("theory_no_bias"),
            },
            f"E02 · {titles[label]}",
            data.discard_worst,
            "pc",
        )
        with st.expander("The run's figures"):
            for png in sorted(run.figures_dir.glob(f"*{label}.png")):
                st.image(str(png), width="stretch")
        with st.expander("Acceptance criteria"):
            st.dataframe(lab.criteria_frame(run), hide_index=True, width="stretch")

with e12_tab:
    st.markdown(
        "Redwood Research found that the phase diagrams look quite different depending on the "
        "activation function (p. 51). E12 reruns E02 with the output ReLU replaced by another "
        "activation: same grid, same seeds, same labels. Each activation's theory minimizes the "
        "same families' exact loss with that activation (by quadrature, D-601). There is no "
        "pass/fail; docs/REPRODUCTION.md reports the differences."
    )
    run12 = latest_run("activation_functions")
    if run12 is None:
        missing_artifact(
            "A complete run of E12 activation_functions", lab.run_command("activation_functions")
        )
    else:
        data12 = load_e12(str(run12.path))
        st.caption(f"Run `{run12.run_id}`")
        settings = list(data12.problems)
        first = next(iter(data12.problems[settings[0]].values()))
        label12 = (
            st.radio(
                "problem",
                settings,
                format_func=lambda s: problem_title(next(iter(data12.problems[s].values()))),
                horizontal=True,
                key="e12_problem",
            )
            if len(settings) > 1
            else settings[0]
        )
        activation = st.radio(
            "output activation", data12.activations, horizontal=True, key="e12_activation"
        )
        problem12 = data12.problems[label12][activation]
        cells12 = pd.DataFrame(data12.cells[data12.cells["setting"] == label12])
        shares = st.columns(2)
        for col, (column, name) in zip(
            shares,
            (("empirical", "empirical majority"), ("theory_with_confused", "theory")),
            strict=True,
        ):
            with col:
                st.markdown(f"**Share of cells per regime** ({name}), by activation")
                table = lab.regime_shares(cells12, column)
                st.bar_chart(table.rename(columns=REGIME_NAMES), horizontal=True, height=260)
        explore(
            problem12,
            {
                "empirical (majority of 9)": problem12.grid("empirical"),
                "theory: the paper's families": problem12.grid("theory"),
                "theory + McGrath's confused family": problem12.grid("theory_with_confused"),
            },
            f"E12 · {problem_title(first)} · {activation}",
            data12.discard_worst,
            "e12",
        )
        with st.expander("The run's figures"):
            for png in sorted(run12.figures_dir.glob(f"*{label12}.png")):
                st.image(str(png), width="stretch")
        with st.expander("Report rows (all INFO)"):
            st.dataframe(lab.criteria_frame(run12), hide_index=True, width="stretch")
