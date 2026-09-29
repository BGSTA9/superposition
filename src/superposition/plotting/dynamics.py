"""Learning-dynamics figures: dimensionality over training, stage rasters, 3D trajectories.

Paper: Superposition and Learning Dynamics, pp. 26–28. Figures:

* p. 26: each feature's dimensionality Dᵢ over training (log steps), with the loss curve below.
* p. 27: feature weight trajectories in a 3D hidden space (top and perspective views) above a
  loss curve whose regimes correspond to geometric transformations.

Static figures use matplotlib (saved as PNG/PDF with the shared style); the rotatable 3D
trajectories are plotly figures for the lab. Functions take arrays and return figures.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from plotly.subplots import make_subplots

from superposition.plotting.style import (
    GRID,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    SERIES,
    SURFACE,
    paper_style,
)

# Correlated-set colors, as on p. 27: blues for set A, greens for set B (three shades each).
SET_SHADES = (
    ("#1f3fb0", "#2a78d6", "#6fa8e8"),
    ("#0e7a52", "#1baf7a", "#73d2a8"),
)
# Stage colors (fixed order, categorical; small → octahedron reads left to right in time).
STAGE_COLORS = {
    "small": "#d9d8d3",
    "collapsed": "#eda100",
    "transition": "#8a8984",
    "prism": "#eb6834",
    "octahedron": "#2a78d6",
}
# Colors for feature lines on p. 26-style plots.
JUMP_COLOR = SERIES[1]
LEVEL_COLOR = "#6f8fe0"
DROPPED_COLOR = "#c9c8c2"


def dimensionality_over_training(
    steps: np.ndarray,
    dimensionality: np.ndarray,
    loss: np.ndarray,
    jumps: Sequence[tuple[int, int, int]] = (),
    title: str = "",
    levels: Sequence[tuple[float, str]] = ((0.5, "1/2"),),
) -> Figure:
    """p. 26 layout for one model: Dᵢ of every feature over training, loss curve below.

    Args:
        steps: ``(T,)`` training steps of the snapshots (the first may be 0).
        dimensionality: ``(T, feat)`` Dᵢ at each snapshot.
        loss: ``(T,)`` loss at each snapshot (same evaluation batch every time).
        jumps: ``(feature, start_step, end_step)`` of detected jumps; those features are drawn
            in the accent color and their windows shaded on both panels.
        title: panel title.
        levels: dotted reference lines ``(value, label)``, e.g. 1/2 for antipodal pairs.
    """
    x = np.maximum(steps, 1)
    jumpers = {f for f, _, _ in jumps}
    final = dimensionality[-1]
    with paper_style():
        fig, (ax_d, ax_l) = plt.subplots(
            2, 1, figsize=(6.4, 4.6), sharex=True, gridspec_kw={"height_ratios": [2.2, 1.0]}
        )
        for f in range(dimensionality.shape[1]):
            if f in jumpers:
                continue
            color = LEVEL_COLOR if final[f] > 0.25 else DROPPED_COLOR
            ax_d.plot(x, dimensionality[:, f], color=color, linewidth=0.5, alpha=0.5)
        for f in sorted(jumpers):
            ax_d.plot(x, dimensionality[:, f], color=JUMP_COLOR, linewidth=1.6)
        for value, label in levels:
            ax_d.axhline(value, color=INK_SECONDARY, linewidth=0.8, linestyle=(0, (1, 2)))
            ax_d.text(x[-1] * 1.05, value, label, fontsize=7.5, color=INK_SECONDARY, va="center")
        for _, start, end in jumps:
            for ax in (ax_d, ax_l):
                ax.axvspan(max(start, 1), max(end, 1), color=JUMP_COLOR, alpha=0.12, linewidth=0)
        ax_d.set_ylabel("feature dimensionality Dᵢ")
        ax_d.set_ylim(-0.02, max(0.75, float(np.nanmax(dimensionality)) + 0.03))
        ax_d.set_title(title, loc="left")
        ax_l.plot(x, loss, color=INK, linewidth=1.2)
        # Zoom to the last 90% of training, as p. 26 does: the early descent is off-scale, so
        # small drops on the plateau are visible.
        late = loss[x >= 0.1 * x[-1]]
        low, high = float(late.min()), float(late.max())
        pad = 0.15 * (high - low) if high > low else 1e-3 * max(abs(low), 1.0)
        ax_l.set_ylim(low - pad, high + pad)
        ax_l.set_ylabel("loss (zoomed)")
        ax_l.set_xscale("log")
        ax_l.set_xlabel("training step (log scale)")
        for ax in (ax_d, ax_l):
            ax.grid(axis="y")
        handles = [
            Patch(color=LEVEL_COLOR, label="ends represented"),
            Patch(color=DROPPED_COLOR, label="ends at 0"),
            Patch(color=JUMP_COLOR, label="jumps late (detected)"),
        ]
        ax_d.legend(handles=handles, loc="upper left", ncol=3)
        fig.tight_layout()
    return fig


def stage_raster(
    steps: np.ndarray,
    stages: np.ndarray,
    row_labels: Sequence[str],
    stage_names: Sequence[str],
    title: str = "",
) -> Figure:
    """One row per training run, colored by the geometric stage at each snapshot (log steps).

    Reading a row left to right gives that run's trajectory type (p. 27: "the learning
    trajectory … is one of a few different learning trajectories").

    Args:
        steps: ``(T,)`` snapshot steps.
        stages: ``(rows, T)`` integer stage codes indexing ``stage_names``.
        row_labels: one label per row.
        stage_names: names of the codes, looked up in :data:`STAGE_COLORS`.
    """
    x = np.maximum(steps, 1).astype(float)
    edges = np.concatenate([[x[0] / 1.2], np.sqrt(x[:-1] * x[1:]), [x[-1] * 1.2]])
    colors = [STAGE_COLORS.get(name, INK_MUTED) for name in stage_names]
    rows = stages.shape[0]
    with paper_style():
        fig, ax = plt.subplots(figsize=(6.6, 0.22 * rows + 1.4))
        for r in range(rows):
            codes = stages[r]
            start = 0
            for t in range(1, len(codes) + 1):
                if t == len(codes) or codes[t] != codes[start]:
                    ax.barh(
                        r,
                        edges[t] - edges[start],
                        left=edges[start],
                        height=0.8,
                        color=colors[codes[start]],
                        linewidth=0,
                    )
                    start = t
        ax.set_xscale("log")
        ax.set_yticks(range(rows))
        ax.set_yticklabels(row_labels, fontsize=7)
        ax.set_ylim(rows - 0.5, -0.5)
        ax.set_xlabel("training step (log scale)")
        ax.set_title(title, loc="left")
        handles = [Patch(color=c, label=n) for n, c in zip(stage_names, colors, strict=True)]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=5)
        fig.tight_layout()
    return fig


def trajectory_panels(
    snapshots: np.ndarray,
    steps: np.ndarray,
    loss: np.ndarray,
    stage_codes: np.ndarray,
    stage_names: Sequence[str],
    set_of_feature: np.ndarray,
    moments: Sequence[int],
    title: str = "",
) -> Figure:
    """p. 27 layout for one run: 3D weight trajectories at a few moments above the loss curve.

    Args:
        snapshots: ``(T, 3, feat)`` weights in a fixed (canonical) frame.
        steps: ``(T,)`` snapshot steps; ``loss``: ``(T,)`` loss at each snapshot.
        stage_codes: ``(T,)`` stage of each snapshot (colors the loss curve's background).
        set_of_feature: ``(feat,)`` correlated-set index of each feature (0 or 1).
        moments: snapshot indices to draw (trajectory up to that moment), at most 4.
    """
    colors = feature_set_colors(set_of_feature)
    x = np.maximum(steps, 1).astype(float)
    limit = max(1.1, float(np.abs(snapshots).max()) * 1.05)
    with paper_style():
        fig = plt.figure(figsize=(7.2, 4.8))
        grid = fig.add_gridspec(2, len(moments), height_ratios=[1.35, 1.0], hspace=0.12)
        for c, t in enumerate(moments):
            ax = fig.add_subplot(grid[0, c], projection="3d")
            for f in range(snapshots.shape[2]):
                path = snapshots[: t + 1, :, f]
                ax.plot(path[:, 0], path[:, 1], path[:, 2], color=colors[f], linewidth=0.8)
                ax.scatter(*snapshots[t, :, f], color=colors[f], s=14, depthshade=False)
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)
            ax.set_zlim(-limit, limit)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_zticks([])  # type: ignore[attr-defined]
            ax.view_init(elev=20, azim=-60)  # type: ignore[attr-defined]
            name = stage_names[int(stage_codes[t])]
            ax.set_title(f"{chr(65 + c)} · step {int(steps[t]):,}\n{name}", fontsize=8)
        ax_l = fig.add_subplot(grid[1, :])
        start = 0
        for t in range(1, len(stage_codes) + 1):
            if t == len(stage_codes) or stage_codes[t] != stage_codes[start]:
                left = x[start]
                right = x[min(t, len(x) - 1)]
                ax_l.axvspan(
                    left,
                    right,
                    color=STAGE_COLORS.get(stage_names[int(stage_codes[start])], INK_MUTED),
                    alpha=0.25,
                    linewidth=0,
                )
                start = t
        ax_l.plot(x, loss, color=INK, linewidth=1.2)
        for c, t in enumerate(moments):
            ax_l.axvline(x[t], color=INK_SECONDARY, linewidth=0.7, linestyle=(0, (2, 2)))
            ax_l.text(x[t], ax_l.get_ylim()[1], chr(65 + c), fontsize=8, ha="center", va="bottom")
        ax_l.set_xscale("log")
        ax_l.set_yscale("log")
        ax_l.set_xlabel("training step (log scale)")
        ax_l.set_ylabel("loss")
        handles = [
            Patch(color=STAGE_COLORS.get(n, INK_MUTED), alpha=0.5, label=n) for n in stage_names
        ]
        ax_l.legend(handles=handles, loc="lower left", ncol=5)
        fig.suptitle(title, x=0.02, ha="left", fontsize=10)
    return fig


def feature_set_colors(set_of_feature: np.ndarray) -> list[str]:
    """Blue shades for set 0 and green shades for set 1, as on p. 27."""
    counters = [0, 0]
    colors = []
    for s in set_of_feature.astype(int):
        shades = SET_SHADES[s % 2]
        colors.append(shades[counters[s % 2] % len(shades)])
        counters[s % 2] += 1
    return colors


def _nan_joined(x: np.ndarray, columns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Many lines as one trace: concatenate columns with NaN gaps (fast in plotly)."""
    n = columns.shape[1]
    xs = np.concatenate([np.append(x, np.nan) for _ in range(n)]) if n else np.array([])
    ys = np.concatenate([np.append(columns[:, i], np.nan) for i in range(n)]) if n else xs
    return xs, ys


def dimensionality_lines(
    steps: np.ndarray,
    dimensionality: np.ndarray,
    loss: np.ndarray,
    marker_step: int | None = None,
    jumpers: Sequence[int] = (),
    title: str = "Feature dimensionality over training",
) -> go.Figure:
    """Interactive p. 26 view: Dᵢ of every feature (top) and the loss (bottom), log steps.

    ``marker_step`` draws a vertical line (the lab's training-step scrubber); ``jumpers`` are
    highlighted features.
    """
    x = np.maximum(steps, 1).astype(float)
    final = dimensionality[-1]
    jump_set = set(int(f) for f in jumpers)
    others = [f for f in range(dimensionality.shape[1]) if f not in jump_set]
    kept = [f for f in others if final[f] > 0.25]
    dropped = [f for f in others if final[f] <= 0.25]
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.68, 0.32])
    for features, color, name, width in (
        (dropped, DROPPED_COLOR, "ends at 0", 0.6),
        (kept, LEVEL_COLOR, "ends represented", 0.6),
        (sorted(jump_set), JUMP_COLOR, "jumps late", 2.0),
    ):
        xs, ys = _nan_joined(x, dimensionality[:, features])
        fig.add_trace(
            go.Scattergl(
                x=xs, y=ys, mode="lines", line={"color": color, "width": width}, name=name,
                hoverinfo="skip",
            ),
            row=1, col=1,
        )  # fmt: skip
    fig.add_trace(
        go.Scatter(
            x=x, y=loss, mode="lines", line={"color": INK, "width": 1.5}, name="loss",
            hovertemplate="step %{x}<br>loss %{y:.5g}<extra></extra>",
        ),
        row=2, col=1,
    )  # fmt: skip
    # plotly's stubs type row/col as str; integers are what it accepts at runtime.
    if marker_step is not None:
        for row in (1, 2):
            fig.add_vline(
                x=max(marker_step, 1),
                line={"color": INK_SECONDARY, "dash": "dot"},
                row=row,  # type: ignore[arg-type]
                col=1,  # type: ignore[arg-type]
            )
    fig.add_hline(
        y=0.5,
        line={"color": INK_SECONDARY, "dash": "dot", "width": 1},
        row=1,  # type: ignore[arg-type]
        col=1,  # type: ignore[arg-type]
    )
    fig.update_xaxes(type="log", gridcolor=GRID)
    fig.update_yaxes(gridcolor=GRID)
    fig.update_yaxes(title_text="Dᵢ", row=1, col=1)
    fig.update_yaxes(title_text="loss", row=2, col=1)
    fig.update_xaxes(title_text="training step (log scale)", row=2, col=1)
    fig.update_layout(
        title={"text": title, "x": 0.0, "font": {"size": 14, "color": INK}},
        height=520,
        margin={"l": 50, "r": 20, "t": 50, "b": 40},
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font={"color": INK_SECONDARY, "size": 12},
        legend={"orientation": "h", "y": 1.08, "x": 0.4},
    )
    return fig


def trajectories_3d(
    snapshots: np.ndarray,
    steps: np.ndarray,
    set_of_feature: np.ndarray,
    upto: int | None = None,
    title: str = "Feature weight trajectories",
) -> go.Figure:
    """Rotatable 3D trajectories of every feature's weight vector (plotly; p. 27).

    Args:
        snapshots: ``(T, 3, feat)`` weights in a fixed frame.
        steps: ``(T,)`` snapshot steps (shown on hover).
        set_of_feature: ``(feat,)`` correlated-set index, for colors.
        upto: draw trajectories up to this snapshot index (default: all), with markers there.
    """
    t_end = snapshots.shape[0] - 1 if upto is None else int(upto)
    colors = feature_set_colors(set_of_feature)
    fig = go.Figure()
    for f in range(snapshots.shape[2]):
        path = snapshots[: t_end + 1, :, f]
        label = f"feature {f} (set {'AB'[int(set_of_feature[f]) % 2]})"
        fig.add_trace(
            go.Scatter3d(
                x=path[:, 0], y=path[:, 1], z=path[:, 2],
                mode="lines",
                line={"color": colors[f], "width": 3},
                name=label,
                customdata=steps[: t_end + 1],
                hovertemplate=label + "<br>step %{customdata}<extra></extra>",
            )
        )  # fmt: skip
        fig.add_trace(
            go.Scatter3d(
                x=[0, path[-1, 0]], y=[0, path[-1, 1]], z=[0, path[-1, 2]],
                mode="lines+markers",
                line={"color": colors[f], "width": 6},
                marker={"size": [0, 5], "color": colors[f]},
                showlegend=False,
                hovertemplate=label + f"<br>step {int(steps[t_end])}<extra></extra>",
            )
        )  # fmt: skip
    limit = max(1.1, float(np.abs(snapshots).max()) * 1.1)
    axis = {"range": [-limit, limit], "backgroundcolor": SURFACE, "gridcolor": GRID}
    fig.update_layout(
        title={"text": title, "x": 0.0, "font": {"size": 14, "color": INK}},
        scene={"xaxis": axis, "yaxis": axis, "zaxis": axis, "aspectmode": "cube"},
        height=480,
        margin={"l": 10, "r": 10, "t": 50, "b": 10},
        paper_bgcolor=SURFACE,
        font={"color": INK_SECONDARY, "size": 12},
        legend={"orientation": "v", "yanchor": "top", "y": 0.9, "x": 1.0, "xanchor": "left"},
    )
    return fig
