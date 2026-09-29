"""2D feature arrows colored by importance, after canonicalization.

Paper: the introduction figure, p. 2 ("As Sparsity Increases, Models Use 'Superposition' To
Represent More Features Than Dimensions"): each feature's direction Wᵢ in the 2D hidden space
drawn as a line from the origin to a dot, colored from most important (yellow) to least
important (green). Callers pass canonicalized W (metrics.canonicalize_2d, p. 22) so panels are
comparable.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from superposition.plotting.style import (
    GRID,
    IMPORTANCE_CMAP,
    INK_SECONDARY,
    PANEL,
    SURFACE,
    paper_style,
)


def importance_colors(n_features: int, importance: np.ndarray | None = None) -> np.ndarray:
    """RGBA colors for features 0 … n−1, brightest = most important.

    Without ``importance``, colors follow the feature index (features are ordered by
    importance). With it, colors follow the importance values, so equally important features
    share one color.
    """
    if importance is not None:
        values = np.asarray(importance, dtype=float)
        span = values.max() - values.min()
        shades = (values - values.min()) / span if span > 0 else np.ones(n_features)
    else:
        shades = np.linspace(1.0, 0.0, n_features) if n_features > 1 else np.ones(1)
    return IMPORTANCE_CMAP(shades)


def plot_feature_arrows(
    ax: Axes,
    W: np.ndarray,
    title: str = "",
    limit: float = 1.35,
    annotate: bool = True,
    subtitle: str = "",
    importance: np.ndarray | None = None,
    colors: Sequence[str] | None = None,
) -> None:
    """Draw each column of a (2, feat) W as a line from the origin to a dot (p. 2).

    Colors follow importance (brightest = most important) unless ``colors`` gives one color
    per feature (e.g. one per correlated set, p. 24).
    """
    if W.shape[0] != 2:
        raise ValueError(f"expected W of shape (2, feat), got {W.shape}")
    n = W.shape[1]
    palette = list(importance_colors(n, importance)) if colors is None else list(colors)
    ax.set_facecolor(PANEL)
    ax.axhline(0, color=GRID, linewidth=0.8, linestyle=(0, (3, 3)), zorder=0)
    ax.axvline(0, color=GRID, linewidth=0.8, linestyle=(0, (3, 3)), zorder=0)
    # Draw least important first so the important features sit on top.
    for i in reversed(range(n)):
        x, y = W[0, i], W[1, i]
        ax.plot([0, x], [0, y], color=palette[i], linewidth=2.2, solid_capstyle="round", zorder=2)
        ax.scatter([x], [y], s=46, color=palette[i], edgecolors=SURFACE, linewidths=1.5, zorder=3)
        if annotate and np.hypot(x, y) > 0.2:
            ax.annotate(
                str(i),
                (x, y),
                xytext=(1.18 * x, 1.18 * y),
                ha="center",
                va="center",
                fontsize=7.5,
                color=INK_SECONDARY,
            )
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, fontsize=9.5, loc="left")
    if subtitle:
        ax.text(
            0.0,
            -0.06,
            subtitle,
            transform=ax.transAxes,
            fontsize=7.5,
            color=INK_SECONDARY,
            va="top",
        )


def arrows_grid(
    rows: Sequence[tuple[str, Sequence[tuple[str, np.ndarray, str]]]],
    suptitle: str | None = None,
) -> Figure:
    """A grid of arrow panels: ``rows = [(row_label, [(title, W, subtitle), ...]), ...]``."""
    n_rows = len(rows)
    n_cols = max(len(panels) for _, panels in rows)
    limit = max(1.35, *(float(np.abs(W).max()) * 1.3 for _, panels in rows for _, W, _ in panels))
    with paper_style():
        fig, axes = plt.subplots(
            n_rows, n_cols, figsize=(2.6 * n_cols + 0.6, 2.75 * n_rows), squeeze=False
        )
        for r, (row_label, panels) in enumerate(rows):
            for c in range(n_cols):
                ax = axes[r, c]
                if c >= len(panels):
                    ax.set_visible(False)
                    continue
                title, W, subtitle = panels[c]
                plot_feature_arrows(ax, W, title=title, limit=limit, subtitle=subtitle)
            axes[r, 0].text(
                -0.12,
                0.5,
                row_label,
                transform=axes[r, 0].transAxes,
                rotation=90,
                ha="center",
                va="center",
                fontsize=9.5,
                fontweight="bold",
            )
        n = rows[0][1][0][1].shape[1]
        colors = importance_colors(n)
        handles = [
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                color=colors[i],
                markersize=6,
                label=f"feature {i}",
            )
            for i in range(n)
        ]
        fig.legend(
            handles=handles,
            loc="center right",
            title="importance ↓",
            title_fontsize=8,
            bbox_to_anchor=(1.0, 0.5),
        )
        fig.subplots_adjust(
            left=0.06, right=0.86, wspace=0.12, hspace=0.3, top=0.88 if suptitle else 0.95
        )
        if suptitle:
            fig.suptitle(suptitle, x=0.06, ha="left", fontsize=10.5)
    return fig


def arrows_figure(
    W: np.ndarray, title: str = "", subtitle: str = "", importance: np.ndarray | None = None
) -> Figure:
    """A single arrow panel as its own figure (used by the lab's Playground)."""
    limit = max(1.35, float(np.abs(W).max()) * 1.3)
    with paper_style():
        fig, ax = plt.subplots(figsize=(3.6, 3.6))
        plot_feature_arrows(
            ax, W, title=title, limit=limit, subtitle=subtitle, importance=importance
        )
        fig.tight_layout()
    return fig
