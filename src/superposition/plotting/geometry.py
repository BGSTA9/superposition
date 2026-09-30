"""Uniform-superposition figures: D* curve, Dᵢ scatter with fraction lines, geometry graphs.

Paper: Uniform Superposition, p. 17 (D* = m/‖W‖²_F against 1/(1 − S) on a log axis, "sticky"
at 1 and 1/2); Feature Dimensionality, pp. 18–19 (each feature's Dᵢ as a scatter over the
same axis, with lines at the characteristic fractions; feature geometry graphs whose nodes
are features and whose edges are |Wᵢ·Wⱼ|).

Functions take arrays (or a prepared networkx graph) and return figures.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

from superposition.plotting.style import GRID, INK, INK_MUTED, INK_SECONDARY, paper_style

# Fraction lines as on p. 19 (value, label, color). Colors follow the paper's legend hues.
FRACTION_LINES: tuple[tuple[float, str, str], ...] = (
    (1.0, "1 · dedicated dimension", "#c93fa5"),
    (3 / 4, "3/4 · tetrahedron", "#1ea5e0"),
    (2 / 3, "2/3 · triangle", "#3fbf7f"),
    (1 / 2, "1/2 · digon (antipodal pair)", "#d8c20a"),
    (2 / 5, "2/5 · pentagon", "#f08c1a"),
    (3 / 8, "3/8 · square antiprism", "#8a3fd1"),
    (0.0, "0 · not learned", "#d9453c"),
)
STRUCTURE_COLORS: dict[str, str] = {
    "dedicated dimension": "#c93fa5",
    "tetrahedron": "#1ea5e0",
    "triangle": "#3fbf7f",
    "digon (antipodal pair)": "#d8c20a",
    "pentagon": "#f08c1a",
    "square antiprism": "#8a3fd1",
}
OTHER_COLOR = "#8a8984"


def _log_axis(ax: Axes, xs: np.ndarray) -> None:
    ax.set_xscale("log")
    ticks = [t for t in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10) if xs.min() * 0.98 <= t <= xs.max() * 1.02]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("1 / (1 − S)   (log scale; dense → sparse)")


def dstar_figure(
    inverse_density: np.ndarray,
    dstar: np.ndarray,
    plateau: tuple[float, float] | None = None,
    title: str = "Dimensions per feature D* = m / ‖W‖²_F",
) -> Figure:
    """D* against 1/(1 − S), with reference lines at 1 and 1/2 (p. 17)."""
    with paper_style():
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        for level in (1.0, 0.5):
            ax.axhline(level, color=GRID, linewidth=1.0, zorder=0)
        if plateau is not None:
            ax.plot(plateau, (0.5, 0.5), color="#eda100", linewidth=2.2, linestyle=(0, (1, 1.2)),
                    label=f"plateau {plateau[0]:.2f}–{plateau[1]:.2f}", zorder=1)  # fmt: skip
        ax.plot(inverse_density, dstar, color=INK, linewidth=1.6, marker="o", markersize=3)
        _log_axis(ax, inverse_density)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "1/4", "1/2", "3/4", "1"])
        ax.set_ylabel("D*")
        ax.set_title(title, loc="left", fontsize=10)
        if plateau is not None:
            ax.legend(loc="upper right")
        fig.tight_layout()
    return fig


def dimensionality_figure(
    inverse_density: np.ndarray,
    dimensionalities: Sequence[np.ndarray],
    dstar: np.ndarray,
    title: str = "Feature dimensionality Dᵢ of every feature",
    jitter: float = 0.012,
    seed: int = 0,
) -> Figure:
    """Each feature's Dᵢ at each sparsity, over the D* curve, with fraction lines (p. 19)."""
    rng = np.random.default_rng(seed)
    with paper_style():
        fig, ax = plt.subplots(figsize=(7.4, 4.4))
        for value, _, color in FRACTION_LINES:
            ax.axhline(value, color=color, linewidth=2.0, alpha=0.55, zorder=0)
        ax.plot(inverse_density, dstar, color=INK_MUTED, linewidth=1.0, zorder=1)
        for x, dims in zip(inverse_density, dimensionalities, strict=True):
            xs = x * 10 ** (jitter * rng.standard_normal(len(dims)))
            ax.scatter(xs, dims, s=2.5, color=INK, alpha=0.35, linewidths=0, zorder=2)
        _log_axis(ax, inverse_density)
        ax.set_ylim(-0.04, 1.06)
        ax.set_yticks([v for v, _, _ in FRACTION_LINES],
                      ["1", "3/4", "2/3", "1/2", "2/5", "3/8", "0"])  # fmt: skip
        ax.set_ylabel("Dᵢ = ‖Wᵢ‖² / Σⱼ (Ŵᵢ·Wⱼ)²")
        ax.set_title(title, loc="left", fontsize=10)
        handles = [
            Line2D([], [], color=c, linewidth=2.0, alpha=0.7, label=lab)
            for _, lab, c in FRACTION_LINES
        ]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7.5)
        fig.tight_layout()
    return fig


def _structure_color(name: str) -> str:
    return STRUCTURE_COLORS.get(name, OTHER_COLOR)


def geometry_graph_figure(
    panels: Sequence[tuple[str, nx.Graph, dict[int, str]]],
    seed: int = 0,
    title: str = "Feature geometry graphs",
) -> Figure:
    """Small multiples of feature geometry graphs (p. 19, top row).

    Each panel is ``(title, graph, structure_by_node)``. Only represented features are drawn;
    nodes are colored by the tegum factor they belong to (dedicated, digon, triangle, …),
    edges by |Wᵢ·Wⱼ| (darker = larger).
    """
    with paper_style():
        # A lone panel (the lab's Geometry page) gets room for its legend below it.
        size = (2.6 * len(panels), 2.9) if len(panels) > 2 else (5.2, 4.6)
        fig, axes = plt.subplots(1, len(panels), figsize=size, squeeze=False)
        for ax, (panel_title, graph, structure) in zip(axes[0], panels, strict=True):
            nodes = [v for v, d in graph.nodes(data=True) if d.get("represented")]
            sub = graph.subgraph(nodes)
            layout = nx.spring_layout(sub, seed=seed, k=1.2 / max(1, len(nodes)) ** 0.5)
            weights = [d["weight"] for _, _, d in sub.edges(data=True)]
            if weights:
                nx.draw_networkx_edges(
                    sub, layout, ax=ax, width=0.4,
                    edge_color=np.clip(weights, 0, 1), edge_cmap=plt.get_cmap("Greys"),
                    edge_vmin=-0.3, edge_vmax=1.0,
                )  # fmt: skip
            colors = [_structure_color(structure.get(v, "")) for v in sub.nodes]
            nx.draw_networkx_nodes(sub, layout, ax=ax, node_size=9, node_color=colors, linewidths=0)
            ax.set_title(panel_title, fontsize=8.5, loc="left")
            ax.set_axis_off()
        names = [*STRUCTURE_COLORS, "other"]
        handles = [
            Line2D([], [], marker="o", linestyle="", color=STRUCTURE_COLORS.get(n, OTHER_COLOR),
                   markersize=5, label=n)
            for n in names
        ]  # fmt: skip
        ncol = len(handles) if len(panels) > 2 else 4
        fig.legend(handles=handles, loc="lower center", ncol=ncol, fontsize=7.5)
        fig.suptitle(title, x=0.01, ha="left", fontsize=10.5)
        fig.tight_layout(rect=(0, 0.1 if len(panels) > 2 else 0.14, 1, 0.92))
    return fig


def census_figure(
    inverse_density: np.ndarray,
    census: Sequence[dict[str, int]],
    n_features: int,
    title: str = "Which tegum factors hold the represented features",
) -> Figure:
    """Stacked share of all features in each kind of factor, per sparsity (pp. 19–20)."""
    names = list(STRUCTURE_COLORS)
    with paper_style():
        fig, ax = plt.subplots(figsize=(7.0, 3.4))
        bottom = np.zeros(len(inverse_density))
        widths = np.diff(np.log10(inverse_density)).mean() if len(inverse_density) > 1 else 0.05
        lefts = np.log10(inverse_density)
        for name in [*names, "other"]:
            share = np.array([
                sum(v for k, v in c.items() if (k == name) or (name == "other" and k not in names))
                for c in census
            ]) / n_features  # fmt: skip
            ax.bar(lefts, share, width=widths * 0.9, bottom=bottom,
                   color=STRUCTURE_COLORS.get(name, OTHER_COLOR), label=name, linewidth=0)  # fmt: skip
            bottom += share
        ticks = [
            t
            for t in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
            if inverse_density.min() * 0.98 <= t <= inverse_density.max() * 1.02
        ]
        ax.set_xticks(np.log10(ticks), [str(t) for t in ticks])
        ax.set_xlabel("1 / (1 − S)   (log scale)")
        ax.set_ylabel("share of all features")
        ax.set_ylim(0, 1)
        ax.set_title(title, loc="left", fontsize=10)
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=7.5)
        fig.text(0.01, 0.01, "white space = features not represented (‖Wᵢ‖ ≤ 0.5)",
                 fontsize=7.5, color=INK_SECONDARY)  # fmt: skip
        fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig
