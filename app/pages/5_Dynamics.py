"""Dynamics: how superposition forms over training (E05).

SPEC §8, page 6: "E05, with a training-step scrubber and the 3D trajectories." Paper:
Superposition and Learning Dynamics, pp. 26–28. Everything is read from the latest
``learning_dynamics`` run; this page never trains.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from lab_ui import missing_artifact, page_setup

from superposition import lab
from superposition.experiments.learning_dynamics import STAGES
from superposition.plotting.dynamics import dimensionality_lines, trajectories_3d
from superposition.runs import RunDir, latest_run

page_setup("Dynamics")
st.title("Dynamics")
st.markdown(
    "How do features arrive at their final geometry? Two phenomena from pp. 26–27: features "
    "**jump between dimensionalities** (with sudden drops in the loss), and small models learn "
    "through a **sequence of geometric transformations**. Drag the step slider to scrub through "
    "training."
)

NAME = "learning_dynamics"
run = latest_run(NAME)
if run is None:
    missing_artifact("A complete run of E05 learning_dynamics", lab.run_command(NAME))
    st.stop()


@st.cache_data(show_spinner="Loading trajectories …")
def load(path: str) -> lab.DynamicsRun:
    return lab.load_dynamics(RunDir(Path(path)))


dyn = load(str(run.path))
st.caption(f"Run `{dyn.run_id}` · results folder `{run.path.parent}`")

tab_a, tab_b, tab_c = st.tabs(
    ["Energy-level jumps (p. 26)", "Geometric transformations (p. 27)", "Acceptance"]
)

# --- (a) energy levels ------------------------------------------------------------------------
with tab_a:
    n_restarts = dyn.dimensionality.shape[1]
    st.markdown(
        f"n = {dyn.dimensionality.shape[2]} features of equal importance in m = {dyn.n_hidden} "
        f"dimensions at "
        f"1/(1 − S) = {dyn.inverse_density:g}, inside the D* = 1/2 plateau of p. 17. Each line is "
        "one feature's dimensionality Dᵢ; orange lines are features whose Dᵢ jumps from a level "
        "it held for a while, together with a drop in the loss."
    )
    restart = st.radio("restart", list(range(n_restarts)), horizontal=True, key="dyn_restart")
    steps = dyn.steps
    index = st.select_slider(
        "training step",
        options=list(range(len(steps))),
        value=len(steps) - 1,
        format_func=lambda i: f"{int(steps[i]):,}",
        key="dyn_step_a",
    )
    jumps = [j for j in dyn.jumps if j["restart"] == restart and j["coincides_with_drop"]]
    st.plotly_chart(
        dimensionality_lines(
            steps,
            dyn.dimensionality[:, restart],
            dyn.loss[:, restart],
            marker_step=int(steps[index]),
            jumpers=[int(j["feature"]) for j in jumps],
            title=f"restart {restart}",
        ),
        width="stretch",
    )
    dims_now = dyn.dimensionality[index, restart]
    cols = st.columns(4)
    cols[0].metric("features at Dᵢ ≈ 1/2", int((np.abs(dims_now - 0.5) <= 0.05).sum()))
    cols[1].metric("features at Dᵢ ≈ 0", int((dims_now <= 0.05).sum()))
    cols[2].metric(f"Σᵢ Dᵢ (≤ m = {dyn.n_hidden})", f"{dims_now.sum():.2f}")
    cols[3].metric("loss (fixed batch)", f"{dyn.loss[index, restart]:.4f}")
    hist = pd.DataFrame({"Dᵢ": np.round(dims_now, 2)}).value_counts().sort_index()
    st.bar_chart(hist.reset_index(name="features").set_index("Dᵢ"), height=180)
    st.caption("Distribution of Dᵢ at the selected step.")
    if jumps:
        st.dataframe(pd.DataFrame(jumps), hide_index=True, width="stretch")

# --- (b) transformations ----------------------------------------------------------------------
with tab_b:
    table = dyn.table
    st.markdown(
        "Six features in m = 3 dimensions, as two correlated sets of three (blue and green). "
        "Stages: **small** (near the origin, A) → **collapsed** (each set moves as one "
        "direction, the linear/PCA-like stage B) → **prism** (two triangles on opposite sides, "
        "C) → **octahedron** (the triangles rotate into an antiprism: antipodal pairs across "
        "sets, D)."
    )
    records = table.to_dict("records")
    c1, c2, c3 = st.columns(3)
    scale = c1.selectbox("init scale (× Xavier)", sorted({r["init_scale"] for r in records}))
    density = c2.selectbox("set density 1 − S", sorted({r["set_probability"] for r in records}))
    rows = [r for r in records if r["init_scale"] == scale and r["set_probability"] == density]
    restart_b = c3.selectbox("restart", sorted(r["restart"] for r in rows))
    row = next(r for r in rows if r["restart"] == restart_b)
    k = int(row["instance"])
    tsteps = dyn.trans_steps
    t_index = st.select_slider(
        "training step",
        options=list(range(len(tsteps))),
        value=len(tsteps) - 1,
        format_func=lambda i: f"{int(tsteps[i]):,}",
        key="dyn_step_b",
    )
    snaps = lab.trajectory_in_frame(dyn, k)
    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(
            trajectories_3d(
                snaps,
                tsteps,
                dyn.sets,
                upto=t_index,
                title=f"stage at step {int(tsteps[t_index]):,}: "
                f"{STAGES[int(dyn.stages[k, t_index])]}",
            ),
            width="stretch",
        )
    with right:
        st.markdown(f"**Trajectory type:** {row['trajectory_type']}")
        st.markdown(
            f"Final stage: **{row['final_stage']}** · first octahedron at step "
            f"{int(row['first_octahedron_step']):,} · time in the prism stage: "
            f"{int(row['prism_steps']):,} steps"
        )
        loss = pd.DataFrame({"step": np.maximum(tsteps, 1), "loss": dyn.trans_loss[k]})
        st.line_chart(loss.set_index("step"), height=220)
        st.caption("Loss on a fixed batch (linear steps); plateaus are the stages.")
    counts = Counter((r["init_scale"], r["set_probability"], r["trajectory_type"]) for r in records)
    st.dataframe(
        pd.DataFrame(
            [
                {"init scale": s, "set density 1 − S": p, "trajectory type": kind, "restarts": n}
                for (s, p, kind), n in sorted(counts.items())
            ]
        ),
        hide_index=True,
        width="stretch",
    )

with tab_c:
    st.dataframe(lab.criteria_frame(run), hide_index=True, width="stretch")
