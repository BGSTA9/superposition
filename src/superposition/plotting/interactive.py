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


def phase_map(
    labels: np.ndarray,
    importance: np.ndarray,
    density: np.ndarray,
    overlay: np.ndarray | None = None,
    selected: tuple[int, int] | None = None,
    title: str = "",
) -> go.Figure:
    """A clickable phase diagram (pp. 15–16): one colored cell per (importance, density).

    ``labels`` is ``(n_density, n_importance)`` of regime names. Cells whose ``overlay`` label
    (e.g. the theory's) differs are marked with an ×, and ``selected`` (density index,
    importance index) is outlined. An invisible marker sits on every cell so that clicking a
    cell selects it (``st.plotly_chart(on_select=...)`` returns its ``customdata``).
    """
    from superposition.plotting.phase import REGIME_COLORS, REGIME_NAMES

    regimes = list(REGIME_COLORS)
    codes = np.vectorize(regimes.index)(np.asarray(labels, dtype=str))
    scale = []
    for k, regime in enumerate(regimes):
        scale += [
            [k / len(regimes), REGIME_COLORS[regime]],
            [(k + 1) / len(regimes), REGIME_COLORS[regime]],
        ]
    n_dens, n_imp = codes.shape
    fig = go.Figure(
        go.Heatmap(
            z=codes,
            zmin=-0.5,
            zmax=len(regimes) - 0.5,
            colorscale=scale,
            showscale=False,
            hoverinfo="skip",
        )
    )
    d, i = np.meshgrid(np.arange(n_dens), np.arange(n_imp), indexing="ij")
    names = np.vectorize(lambda r: REGIME_NAMES[r])(np.asarray(labels, dtype=str))
    hover = [
        f"importance {importance[c]:.3g}×<br>density {density[r]:.3g}<br>{names[r, c]}"
        + (f"<br>theory: {REGIME_NAMES[str(overlay[r, c])]}" if overlay is not None else "")
        for r, c in zip(d.ravel(), i.ravel(), strict=True)
    ]
    fig.add_trace(
        go.Scatter(
            x=i.ravel(),
            y=d.ravel(),
            mode="markers",
            marker={"size": 18, "symbol": "square", "opacity": 0.0},
            customdata=np.stack([d.ravel(), i.ravel()], axis=1),
            hovertext=hover,
            hoverinfo="text",
            showlegend=False,
            name="cells",
        )
    )
    if overlay is not None:
        rows, cols = np.nonzero(np.asarray(labels) != np.asarray(overlay))
        fig.add_trace(
            go.Scatter(
                x=cols,
                y=rows,
                mode="markers",
                marker={"symbol": "x-thin", "size": 9, "line": {"width": 1.6, "color": INK}},
                hoverinfo="skip",
                name="differs from the theory",
            )
        )
    if selected is not None:
        fig.add_trace(
            go.Scatter(
                x=[selected[1]],
                y=[selected[0]],
                mode="markers",
                marker={"symbol": "square-open", "size": 22, "line": {"width": 2.5, "color": INK}},
                hoverinfo="skip",
                name="selected cell",
            )
        )
    x_ticks = [c for c in range(n_imp) if c in (0, n_imp // 2, n_imp - 1)]
    y_ticks = [r for r in range(n_dens) if r in (0, n_dens // 2, n_dens - 1)]
    fig.update_xaxes(
        tickvals=x_ticks,
        ticktext=[f"{importance[c]:.2g}×" for c in x_ticks],
        title="relative importance of the extra feature",
        showgrid=False,
        zeroline=False,
    )
    fig.update_yaxes(
        tickvals=y_ticks,
        ticktext=[f"{density[r]:.2g}" for r in y_ticks],
        title="feature density 1 − S",
        showgrid=False,
        zeroline=False,
    )
    return _layout(fig, title, height=420)
