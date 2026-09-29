"""Playground: train a small toy model live, look at it, and probe it one input at a time.

SPEC §8, page 2. Paper: The Model and Basic Results, pp. 10–12; Mathematical Understanding,
p. 13 (interference); Feature Dimensionality, pp. 18–19; Superposition in a Privileged Basis,
p. 31. All computation is in ``superposition.lab``; figures come from ``superposition.plotting``
so they match the paper-style figures of the experiments.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st
import torch
from lab_ui import page_setup

from superposition import lab, metrics
from superposition.plotting.arrows import arrows_figure
from superposition.plotting.interactive import (
    feature_vectors_3d,
    hidden_chart,
    loss_curves,
    probe_chart,
)
from superposition.plotting.weights import WeightsPanel, weight_matrix_figure, weights_grid
from superposition.theory import linear_optimum_loss

page_setup("Playground")

DENSITIES = (0.001, 0.0015, 0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07,
             0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0)  # fmt: skip
STEP_CHOICES = (500, 1_000, 2_000, 5_000, 10_000, 20_000)
MAX_RESTARTS = 16
KIND_LABELS = {
    "linear": "Linear: x′ = WᵀWx + b",
    "relu_output": "ReLU output: x′ = ReLU(WᵀWx + b)",
    "relu_hidden": "ReLU hidden: x′ = ReLU(Wᵀ ReLU(Wx) + b)",
}
DEFAULT = lab.PlaygroundConfig()
# Presets from the paper: (label, n, m, decay, 1 − S, model).
PRESETS = {
    "Custom": None,
    "p. 2 · dense: two dedicated dimensions (S = 0)": (5, 2, 0.9, 1.0, "relu_output"),
    "p. 2 · antipodal pairs (S = 0.8)": (5, 2, 0.9, 0.2, "relu_output"),
    "p. 2 · pentagon (S = 0.9)": (5, 2, 0.9, 0.1, "relu_output"),
    "p. 2 · the linear model at S = 0.9": (5, 2, 0.9, 0.1, "linear"),
    "p. 11 · n = 20, m = 5, 1 − S = 0.01": (20, 5, 0.7, 0.01, "relu_output"),
    "p. 32 · ReLU hidden layer, n = 10, m = 5, 1 − S = 0.1": (10, 5, 0.75, 0.1, "relu_hidden"),
    "3D · six features in m = 3 (1 − S = 0.05)": (6, 3, 1.0, 0.05, "relu_output"),
}
WIDGET_DEFAULTS = {
    "pg_n": DEFAULT.n_features,
    "pg_m": DEFAULT.n_hidden,
    "pg_decay": DEFAULT.importance_decay,
    "pg_p": DEFAULT.feature_probability,
    "pg_kind": DEFAULT.model_kind,
    "pg_steps": DEFAULT.steps,
    "pg_restarts": DEFAULT.restarts,
    "pg_seed": DEFAULT.seed,
    "pg_device": "cpu",
}
for key, value in WIDGET_DEFAULTS.items():
    st.session_state.setdefault(key, value)


def apply_preset() -> None:
    preset = PRESETS[st.session_state["pg_preset"]]
    if preset is None:
        return
    n, m, decay, p, kind = preset
    st.session_state.update(pg_n=n, pg_m=m, pg_decay=decay, pg_p=p, pg_kind=kind)


@st.cache_data(show_spinner=False, max_entries=64)
def train_cached(items: tuple[tuple[str, Any], ...], device: str) -> lab.PlaygroundResult:
    """Train once per (config, device); later visits to the same config are instant."""
    return lab.train_playground(lab.config_from_dict(dict(items)), device=device)


@st.cache_resource(show_spinner=False, max_entries=64)
def best_model(items: tuple[tuple[str, Any], ...], device: str) -> torch.nn.Module:
    """The best restart as a one-instance model, for the probe panel."""
    return train_cached(items, device).best_model()


def config_from_widgets() -> lab.PlaygroundConfig:
    s = st.session_state
    return lab.config_from_dict(
        {
            "n_features": int(s.pg_n),
            "n_hidden": int(s.pg_m),
            "importance_decay": float(s.pg_decay),
            "feature_probability": float(s.pg_p),
            "model_kind": s.pg_kind,
            "steps": int(s.pg_steps),
            "restarts": int(s.pg_restarts),
            "seed": int(s.pg_seed),
        }
    )


# --- controls ---------------------------------------------------------------------------------
with st.sidebar:
    st.header("Model and data")
    st.selectbox(
        "Start from a paper setting", list(PRESETS), key="pg_preset", on_change=apply_preset
    )
    with st.form("controls"):
        st.slider("features n", *lab.N_FEATURES_RANGE, key="pg_n")
        st.slider("hidden dimensions m", *lab.N_HIDDEN_RANGE, key="pg_m")
        st.slider(
            "importance decay (Iᵢ = decayⁱ)", *lab.IMPORTANCE_DECAY_RANGE, step=0.01, key="pg_decay"
        )
        st.select_slider(
            "feature density 1 − S (log scale)",
            options=DENSITIES,
            format_func=lambda p: f"{p:g}   (S = {1 - p:.4g})",
            key="pg_p",
        )
        st.radio("model", lab.MODEL_KINDS, format_func=KIND_LABELS.get, key="pg_kind")
        st.select_slider("training steps", options=STEP_CHOICES, key="pg_steps")
        st.slider("restarts (keep the lowest loss)", 1, MAX_RESTARTS, key="pg_restarts")
        st.number_input("seed", min_value=0, max_value=1_000_000, step=1, key="pg_seed")
        st.radio(
            "device",
            ["cpu", "auto"],
            horizontal=True,
            key="pg_device",
            help="auto = CUDA → MPS → CPU",
        )
        submitted = st.form_submit_button("Train", type="primary", width="stretch")
    st.caption(
        f"Defaults ({DEFAULT.steps:,} steps × {DEFAULT.restarts} restarts, n = 5, m = 2) train in a "
        "few seconds on a laptop CPU. Results are cached by configuration."
    )

if submitted or "pg_items" not in st.session_state:
    try:
        st.session_state["pg_items"] = tuple(
            sorted(lab.config_to_dict(config_from_widgets()).items())
        )
        st.session_state["pg_train_device"] = st.session_state.pg_device
    except ValueError as error:
        st.error(f"Invalid setting: {error}")
        st.stop()

items: tuple[tuple[str, Any], ...] = st.session_state["pg_items"]
device: str = st.session_state["pg_train_device"]
config = lab.config_from_dict(dict(items))

# --- train ------------------------------------------------------------------------------------
st.title("Playground")
st.markdown(
    "Train the paper's toy models (p. 10: x′ = act(WᵀWx + b); p. 31: ReLU on the hidden layer too) "
    "on sparse synthetic features, then look inside. Each feature xᵢ is 0 with probability S and "
    "otherwise uniform on [0, 1]; the loss weights feature i's squared error by Iᵢ. **Predict "
    "before you press Train**: how many features will the model represent, and in what geometry?"
)
start = time.perf_counter()
with st.spinner(f"Training {config.restarts} restart(s) × {config.steps:,} steps on {device} …"):
    result = train_cached(items, device)
elapsed = time.perf_counter() - start
cached = elapsed < 0.5 * result.train_seconds
st.caption(
    f"n = {config.n_features}, m = {config.n_hidden}, Iᵢ = {config.importance_decay:g}ⁱ, "
    f"1 − S = {config.feature_probability:g}, {KIND_LABELS[config.model_kind]} · "
    f"trained in {result.train_seconds:.1f} s on {result.device}"
    + (" (served from cache)" if cached else "")
)

# --- headline numbers -------------------------------------------------------------------------
numbers = lab.summary(result)
cols = st.columns(5)
cols[0].metric("best loss", f"{numbers['best loss']:.4g}")
cols[1].metric(
    "represented (‖Wᵢ‖ > 0.5)", f"{numbers['represented features']} / {config.n_features}"
)
cols[2].metric("D* (dims per feature)", f"{numbers['dimensions per feature D*']:.3f}")
cols[3].metric(
    "superposition (p. 8)",
    "yes" if numbers["superposition (WᵀW singular on represented)"] else "no",
)
if config.model_kind == "linear":
    optimum = linear_optimum_loss(
        torch.as_tensor(result.importance), config.n_hidden, config.feature_probability
    )
    cols[4].metric(
        "closed-form linear optimum", f"{optimum:.4g}", help="Σ_{i≥m} Iᵢ(p/3 − p²/4), p. 13"
    )
else:
    spread = float(result.final_loss.max() - result.final_loss.min())
    cols[4].metric("loss spread across restarts", f"{spread:.3g}")

# --- weights ----------------------------------------------------------------------------------
st.subheader("What the model learned")
W, b = result.best_W, result.best_b
Wt = torch.as_tensor(W)[None]
left, right = st.columns([3, 2])
with left:
    panel = WeightsPanel(
        title="best restart",
        gram=metrics.gram(Wt)[0].numpy(),
        bias=b,
        norms=metrics.feature_norms(Wt)[0].numpy(),
        superposition=metrics.superposition_score(Wt)[0].numpy(),
    )
    st.pyplot(weights_grid([panel]), width="content")
    st.caption(
        "Top: WᵀW (red +, blue −) with the bias b beside it (p. 10). Bottom: ‖Wᵢ‖ per feature, "
        "most important on top, colored by Σⱼ≠ᵢ(Ŵᵢ·Wⱼ)²: dark = orthogonal to all others, "
        "bright = in superposition (p. 11)."
    )
with right:
    if config.n_hidden == 2:
        canon = metrics.canonicalize_2d(Wt)[0].numpy()
        st.pyplot(
            arrows_figure(
                canon, title="Feature directions (canonicalized)", importance=result.importance
            ),
            width="content",
        )
        st.caption("Each line is one feature's direction Wᵢ; brighter = more important (p. 2).")
    elif config.n_hidden == 3:
        st.plotly_chart(feature_vectors_3d(W, result.importance), width="stretch")
        st.caption(
            "Drag to rotate. Each line is one feature's direction Wᵢ; brighter = more important."
        )
    elif config.n_hidden == 1:
        st.bar_chart(pd.DataFrame({"Wᵢ (1D)": W[0]}), height=260)
        st.caption(
            "With m = 1 every feature is a signed number: opposite signs form antipodal pairs."
        )
    else:
        st.info("No direct picture for m ≥ 4: WᵀW and the Dᵢ table describe the geometry.")
    if config.model_kind == "relu_hidden":
        st.pyplot(weight_matrix_figure(W), width="content")
        st.caption(
            "With a hidden ReLU the neuron basis is privileged, so W itself is readable (p. 31)."
        )

st.markdown("**Per-feature table** (Dᵢ = ‖Wᵢ‖² / Σⱼ(Ŵᵢ·Wⱼ)², p. 18; fractions of p. 19)")
st.caption(
    "“Nearest fraction” names the closest characteristic Dᵢ within ±0.02. It is a hint, not a "
    "geometry test: 2/5 (pentagon) and 3/8 (square antiprism) are only 0.025 apart, and unequal "
    "importances distort polytopes, so a pentagon's least important feature can land nearer 3/8."
)
table = lab.feature_table(W, b, result.importance)
st.dataframe(
    table,
    hide_index=True,
    width="stretch",
    column_config={
        col: st.column_config.NumberColumn(format="%.3f")
        for col in table.columns
        if table[col].dtype.kind == "f"
    },
)

# --- training ---------------------------------------------------------------------------------
with st.expander("Training curves and every restart"):
    st.plotly_chart(
        loss_curves(result.log_steps, result.loss_history, result.best), width="stretch"
    )
    restarts = pd.DataFrame(
        {
            "restart": np.arange(config.restarts),
            "final loss": result.final_loss,
            "represented": metrics.represented_mask(torch.as_tensor(result.W)).sum(-1).numpy(),
            "best": np.arange(config.restarts) == result.best,
        }
    )
    st.dataframe(restarts, hide_index=True, width="stretch")
    st.caption(
        "Small models, especially with m = 2, often stop in worse local optima; the paper fits "
        "several times and keeps the lowest loss (p. 23). Losses at different sparsities measure "
        "different tasks and are not comparable (footnote 17)."
    )

# --- probe ------------------------------------------------------------------------------------
st.subheader("Probe panel")
st.markdown(
    "Choose which features are active and set their values; see the hidden vector h, each "
    "feature's readout Wᵀh, the readout after adding the bias, the output after the activation, "
    "and the reconstruction next to the truth. Try one active feature, then two that share a "
    "direction, to see interference appear (pp. 7, 13)."
)
probe_key = "_".join(str(v) for _, v in items)
default_active = [0]
active = st.multiselect(
    "active features",
    options=list(range(config.n_features)),
    default=default_active,
    key=f"probe_active_{probe_key}",
)
x = np.zeros(config.n_features)
if active:
    slider_cols = st.columns(min(len(active), 6))
    for position, feature in enumerate(active):
        with slider_cols[position % len(slider_cols)]:
            x[feature] = st.slider(
                f"x{feature}", 0.0, 1.0, 1.0, 0.05, key=f"probe_value_{probe_key}_{feature}"
            )
model = best_model(items, device)
probed = lab.probe(model, x)  # type: ignore[arg-type]
output_label = "output x′ = ReLU(·)" if config.model_kind != "linear" else "output x′ (no ReLU)"
c1, c2 = st.columns([1, 2])
with c1:
    hidden_pre = probed.hidden_pre if config.model_kind == "relu_hidden" else None
    st.plotly_chart(hidden_chart(probed.hidden, hidden_pre), width="stretch")
with c2:
    st.plotly_chart(
        probe_chart(
            probed.target, probed.readout, probed.readout_plus_bias, probed.output, output_label
        ),
        width="stretch",
    )
frame = probed.feature_frame(result.importance)
st.dataframe(
    frame,
    hide_index=True,
    width="stretch",
    column_config={
        col: st.column_config.NumberColumn(format="%.3f")
        for col in frame.columns
        if col != "feature"
    },
)
st.caption(
    f"Importance-weighted squared error on this input: {frame['weighted sq. error'].sum():.4g}. "
    "Negative readouts are removed by the ReLU for free; the negative bias trims small positive "
    "interference, at the cost of shrinking every feature's own readout (p. 13)."
)
