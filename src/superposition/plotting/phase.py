"""Phase diagrams: extra-feature regime over (relative importance × feature density).

Paper: Superposition as a Phase Change, pp. 14–16. The paper draws each diagram twice: an
empirical version, colored by the norm and superposition of the extra feature (white = not
represented, blue = dedicated dimension, red = superposition), and a theoretical version,
colored by the winning weight configuration. Both axes are logarithmic: relative importance
of the extra feature from 0.1× to 10× (x) and feature density 1 − S from 0.01 to 1 (y).

Functions take arrays and return figures; nothing here computes a metric or a label.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from superposition.plotting.style import INK, INK_SECONDARY, SURFACE, paper_style

# Regime colors, following pp. 15–16: gray = not represented, blue = dedicated dimension,
# red = superposition (antipodal). McGrath's confused-feature regime (p. 52) gets amber.
REGIME_COLORS: dict[str, str] = {
    "not_learned": "#d9d8d2",
    "dedicated": "#2446b8",
    "superposition": "#c8262c",
    "confused": "#eda100",
}
REGIME_NAMES: dict[str, str] = {
    "not_learned": "not represented",
    "dedicated": "dedicated dimension",
    "superposition": "superposition (antipodal)",
    "confused": "confused (W₁ ≈ W₂)",
}
_BLUE = np.array(to_rgb("#2446b8"))
_RED = np.array(to_rgb("#c8262c"))
_WHITE = np.array(to_rgb(SURFACE))


def _log_edges(values: np.ndarray) -> np.ndarray:
    """Cell edges (log10) of a log-spaced grid: geometric midpoints, extended at both ends."""
    logs = np.log10(np.asarray(values, dtype=float))
    if logs.size == 1:
        return np.array([logs[0] - 0.5, logs[0] + 0.5])
    mids = (logs[1:] + logs[:-1]) / 2
    return np.concatenate([[2 * logs[0] - mids[0]], mids, [2 * logs[-1] - mids[-1]]])


def _format_axes(ax: Axes, importance: np.ndarray, density: np.ndarray) -> None:
    x_edges, y_edges = _log_edges(importance), _log_edges(density)
    ax.set_xlim(x_edges[0], x_edges[-1])
    ax.set_ylim(y_edges[0], y_edges[-1])
    x_ticks = [t for t in (0.1, 1.0, 10.0) if x_edges[0] <= np.log10(t) <= x_edges[-1]]
    y_ticks = [t for t in (0.01, 0.1, 1.0) if y_edges[0] <= np.log10(t) <= y_edges[-1]]
    ax.set_xticks(np.log10(x_ticks), [f"{t:g}×" for t in x_ticks])
    ax.set_yticks(np.log10(y_ticks), [f"{t:g}" for t in y_ticks])
    ax.set_xlabel("relative importance of the extra feature")
    ax.set_ylabel("feature density 1 − S")


def _show(ax: Axes, rgb: np.ndarray, importance: np.ndarray, density: np.ndarray) -> None:
    x_edges, y_edges = _log_edges(importance), _log_edges(density)
    ax.imshow(
        rgb,
        origin="lower",
        extent=(x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]),
        aspect="auto",
        interpolation="nearest",
    )
    _format_axes(ax, importance, density)


def regime_colors(labels: np.ndarray) -> np.ndarray:
    """RGB image ``(*labels.shape, 3)`` for an array of regime labels."""
    table = {k: np.array(to_rgb(v)) for k, v in REGIME_COLORS.items()}
    flat = np.array([table[str(label)] for label in np.asarray(labels).ravel()])
    return flat.reshape(*np.asarray(labels).shape, 3)


def norm_superposition_colors(norm: np.ndarray, superposition: np.ndarray) -> np.ndarray:
    """The pp. 15–16 empirical coloring: white → blue with ‖Wᵢ‖, blue → red with superposition.

    ``norm`` and ``superposition`` are clipped to [0, 1] (the paper's legend saturates at ≥ 1).
    """
    n = np.clip(np.asarray(norm, dtype=float), 0, 1)[..., None]
    s = np.clip(np.asarray(superposition, dtype=float), 0, 1)[..., None]
    return (1 - n) * _WHITE + n * ((1 - s) * _BLUE + s * _RED)


def plot_regimes(
    ax: Axes, labels: np.ndarray, importance: np.ndarray, density: np.ndarray, title: str = ""
) -> None:
    """A categorical phase diagram: one color per regime (p. 15, theoretical version)."""
    _show(ax, regime_colors(labels), importance, density)
    if title:
        ax.set_title(title, fontsize=9.5, loc="left")


def plot_norm_superposition(
    ax: Axes,
    norm: np.ndarray,
    superposition: np.ndarray,
    importance: np.ndarray,
    density: np.ndarray,
    title: str = "",
) -> None:
    """The empirical diagram colored by the extra feature's norm and superposition (p. 15)."""
    _show(ax, norm_superposition_colors(norm, superposition), importance, density)
    if title:
        ax.set_title(title, fontsize=9.5, loc="left")


def plot_agreement(
    ax: Axes,
    empirical: np.ndarray,
    theory: np.ndarray,
    importance: np.ndarray,
    density: np.ndarray,
    near_boundary: np.ndarray | None = None,
    title: str = "",
) -> None:
    """Agreeing cells in light gray; disagreeing cells in the empirical regime's color.

    ``near_boundary`` (optional boolean map) marks cells next to a theoretical boundary with a
    dot, so disagreements on the boundary are easy to tell from ones in a region's interior.
    """
    agree = np.asarray(empirical) == np.asarray(theory)
    rgb = regime_colors(empirical)
    rgb[agree] = np.array(to_rgb("#f1f0ec"))
    _show(ax, rgb, importance, density)
    if near_boundary is not None:
        logs_x, logs_y = np.log10(importance), np.log10(density)
        rows, cols = np.nonzero(near_boundary & ~agree)
        ax.scatter(logs_x[cols], logs_y[rows], s=6, color=INK, zorder=3, linewidths=0)
    if title:
        ax.set_title(title, fontsize=9.5, loc="left")


def phase_figure(
    panels: Sequence[tuple[str, str, dict]],
    importance: np.ndarray,
    density: np.ndarray,
    suptitle: str = "",
    regimes: Sequence[str] = ("not_learned", "superposition", "dedicated", "confused"),
) -> Figure:
    """A row of phase-diagram panels sharing axes.

    ``panels`` is a list of ``(kind, title, data)`` with kind ``"regimes"`` (data: labels),
    ``"norm_superposition"`` (data: norm, superposition) or ``"agreement"`` (data: empirical,
    theory, optional near_boundary).
    """
    with paper_style():
        fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels) + 0.4, 3.6))
        axes = np.atleast_1d(axes)
        for ax, (kind, title, data) in zip(axes, panels, strict=True):
            if kind == "regimes":
                plot_regimes(ax, data["labels"], importance, density, title)
            elif kind == "norm_superposition":
                plot_norm_superposition(
                    ax, data["norm"], data["superposition"], importance, density, title
                )
            elif kind == "agreement":
                plot_agreement(
                    ax,
                    data["empirical"],
                    data["theory"],
                    importance,
                    density,
                    data.get("near_boundary"),
                    title,
                )
            else:
                raise ValueError(f"unknown panel kind {kind!r}")
        for ax in axes[1:]:
            ax.set_ylabel("")
        handles = [
            Patch(facecolor=REGIME_COLORS[r], edgecolor="none", label=REGIME_NAMES[r])
            for r in regimes
        ]
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), fontsize=8)
        if suptitle:
            fig.suptitle(suptitle, x=0.02, ha="left", fontsize=10.5)
        fig.tight_layout(rect=(0, 0.08, 1, 0.93 if suptitle else 1.0))
    return fig


def family_loss_figure(
    density: np.ndarray,
    losses: np.ndarray,
    family_names: Sequence[str],
    family_labels: Sequence[str],
    importance_values: Sequence[float],
    suptitle: str = "",
) -> Figure:
    """Each family's optimal loss against density, one panel per relative importance.

    Paper: p. 15 ("there's crossover between the functions, causing a discontinuity in the
    derivative of the optimal loss"). ``losses`` is ``(len(importance_values), n_density,
    n_families)``; losses are divided by the density's feature variance p/3 − p²/4 so the
    curves share one scale (losses at different densities are not comparable, footnote 17).
    """
    variance = density / 3 - density**2 / 4
    # Families sharing a regime (n = 3 has two "dedicated" ones) get distinct line styles.
    styles = []
    for f, label in enumerate(family_labels):
        repeat = list(family_labels[:f]).count(label)
        styles.append((0, (4, 2)) if label == "confused" else ["-", (0, (1, 1.5))][repeat % 2])
    with paper_style():
        fig, axes = plt.subplots(
            1, len(importance_values), figsize=(3.4 * len(importance_values), 3.2), squeeze=False
        )
        for ax, imp, block in zip(axes[0], importance_values, losses, strict=True):
            for f, (name, label) in enumerate(zip(family_names, family_labels, strict=True)):
                ax.plot(
                    density,
                    block[:, f] / variance,
                    color=REGIME_COLORS[label],
                    linewidth=1.8,
                    label=name,
                    linestyle=styles[f],
                )
            ax.set_xscale("log")
            ax.set_xlabel("feature density 1 − S")
            ax.set_title(f"extra importance {imp:.3g}×", fontsize=9.5, loc="left")
            ax.grid(axis="y")
        axes[0, 0].set_ylabel("optimal loss / (p/3 − p²/4)")
        handles, names = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles,
            names,
            loc="lower left",
            bbox_to_anchor=(0.01, 0.0),
            ncol=min(3, len(names)),
            fontsize=8,
        )
        if suptitle:
            fig.suptitle(suptitle, x=0.02, ha="left", fontsize=10.5, color=INK)
        fig.text(
            0.99,
            0.02,
            "lowest curve wins; kinks in the minimum are first-order phase changes",
            ha="right",
            fontsize=7.5,
            color=INK_SECONDARY,
        )
        fig.tight_layout(rect=(0, 0.17, 1, 0.92 if suptitle else 1.0))
    return fig
