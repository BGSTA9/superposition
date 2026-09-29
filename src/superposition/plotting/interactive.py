"""Interactive (plotly) figures for the lab: 3D feature vectors, loss curves, probe panel.

SPEC §8 (Playground outputs: "2D arrows when m = 2, or a 3D plot when m = 3"; the probe panel)
and §6 plotting ("3D trajectories (plotly)", p. 27). Colors come from the shared style module
so the lab matches the paper-style figures: importance uses the same yellow → green ramp as the
p. 2 arrows, and categorical series use the fixed-order palette. Every figure has hover
tooltips; functions take arrays and return figures.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from matplotlib.colors import to_hex

from superposition.plotting.arrows import importance_colors
from superposition.plotting.style import GRID, INK, INK_SECONDARY, SERIES, SURFACE

# Fixed-order categorical slots for the probe panel's four stages (validated palette order).
PROBE_SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")


def _layout(fig: go.Figure, title: str, height: int = 360) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0.0, "font": {"size": 14, "color": INK}},
        height=height,
        margin={"l": 50, "r": 20, "t": 50, "b": 90},
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"color": INK_SECONDARY, "size": 12},
        legend={"orientation": "h", "yanchor": "top", "y": -0.22, "x": 0, "xanchor": "left"},
        hoverlabel={"bgcolor": "white"},
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def feature_vectors_3d(
    W: np.ndarray, importance: np.ndarray | None = None, title: str = "Feature directions in 3D"
) -> go.Figure:
    """Each column of a (3, feat) W as a line from the origin to a dot, colored by importance.

    The 3D counterpart of the p. 2 arrows; hover shows the feature index and its norm.
    """
    if W.shape[0] != 3:
        raise ValueError(f"expected W of shape (3, feat), got {W.shape}")
    n = W.shape[1]
    colors = [to_hex(c) for c in importance_colors(n, importance)]
    fig = go.Figure()
    for i in reversed(range(n)):
        x, y, z = W[:, i]
        norm = float(np.linalg.norm(W[:, i]))
        fig.add_trace(
            go.Scatter3d(
                x=[0, x], y=[0, y], z=[0, z],
                mode="lines+markers",
                line={"color": colors[i], "width": 6},
                marker={"size": [0, 5], "color": colors[i]},
                name=f"feature {i}",
                hovertemplate=f"feature {i}<br>‖W‖ = {norm:.3f}<extra></extra>",
            )
        )  # fmt: skip
    limit = max(1.2, float(np.abs(W).max()) * 1.15)
    axis = {"range": [-limit, limit], "backgroundcolor": SURFACE, "gridcolor": GRID}
    fig.update_layout(scene={"xaxis": axis, "yaxis": axis, "zaxis": axis, "aspectmode": "cube"})
    fig = _layout(fig, title, height=460)
    fig.update_layout(
        legend={"orientation": "v", "yanchor": "top", "y": 0.9, "x": 1.0, "xanchor": "left"},
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
    )
    return fig


def loss_curves(
    log_steps: list[int], history: np.ndarray, best: int, title: str = "Training loss"
) -> go.Figure:
    """Training-batch loss of every restart (log y); the best restart is drawn on top."""
    fig = go.Figure()
    for r in range(history.shape[1]):
        is_best = r == best
        fig.add_trace(
            go.Scatter(
                x=log_steps,
                y=history[:, r],
                mode="lines",
                line={"color": SERIES[1] if is_best else "#b9b8b2", "width": 2.5 if is_best else 1},
                name=f"restart {r}" + (" (best)" if is_best else ""),
                showlegend=is_best,
                hovertemplate=f"restart {r}<br>step %{{x}}<br>loss %{{y:.4g}}<extra></extra>",
            )
        )
    fig.update_yaxes(type="log", title="loss (training batch)")
    fig.update_xaxes(title="step")
    return _layout(fig, title)


def probe_chart(
    target: np.ndarray,
    readout: np.ndarray,
    readout_plus_bias: np.ndarray,
    output: np.ndarray,
    output_label: str = "output x′",
) -> go.Figure:
    """Per feature: the true value, the readout Wᵀh, after + b, and the model output.

    Reading left to right within a feature shows what each stage of the model does to it.
    """
    features = [f"{i}" for i in range(len(target))]
    fig = go.Figure()
    series = [
        ("truth x", target),
        ("readout Wᵀh", readout),
        ("+ bias", readout_plus_bias),
        (output_label, output),
    ]
    for (name, values), color in zip(series, PROBE_SERIES, strict=True):
        fig.add_trace(
            go.Bar(
                x=features,
                y=values,
                name=name,
                marker={"color": color, "line": {"color": SURFACE, "width": 1}},
                hovertemplate=f"feature %{{x}}<br>{name}: %{{y:.3f}}<extra></extra>",
            )
        )
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.05)
    fig.update_xaxes(title="feature (most → least important)")
    fig.update_yaxes(title="value", zeroline=True, zerolinecolor=INK_SECONDARY)
    return _layout(fig, "Probe: each feature through the model")


def hidden_chart(hidden: np.ndarray, hidden_pre: np.ndarray | None = None) -> go.Figure:
    """The hidden vector h (and Wx before the ReLU, for the ReLU hidden model)."""
    dims = [f"h{j}" for j in range(len(hidden))]
    fig = go.Figure()
    if hidden_pre is not None:
        fig.add_trace(
            go.Bar(
                x=dims, y=hidden_pre, name="Wx (before ReLU)",
                marker={"color": SERIES[0]},
                hovertemplate="%{x}: %{y:.3f}<extra>Wx</extra>",
            )
        )  # fmt: skip
    fig.add_trace(
        go.Bar(
            x=dims, y=hidden, name="h",
            marker={"color": SERIES[1] if hidden_pre is not None else SERIES[0]},
            hovertemplate="%{x}: %{y:.3f}<extra>h</extra>",
        )
    )  # fmt: skip
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="activation", zeroline=True, zerolinecolor=INK_SECONDARY)
    return _layout(fig, "Hidden activations h", height=300)
