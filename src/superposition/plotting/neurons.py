"""Neuron-basis figures: W heatmaps and neuron stack plots.

Paper: Superposition in a Privileged Basis, pp. 31–33 (W as a matrix and as a stack plot) and
Computation in Superposition, pp. 36–37 (stack plots of |W₁| for the absolute-value model).

The p. 32 stack plot, exactly:

* each column visualizes one neuron (one column of W when W is drawn features × neurons);
* each rectangle is one weight entry, with height |weight|;
* its color is the feature it acts on (here: that feature's superposition score, dark = not
  in superposition, bright = ≥ 1, as the p. 33 and p. 36 figures color them);
* negative weights go below the x-axis; the order of rectangles is not significant (we stack
  the largest at the bottom).

Neurons are shaded lightly by whether they are monosemantic or polysemantic (p. 33). Functions
take arrays and return figures.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from superposition.plotting.style import (
    INK_SECONDARY,
    SERIES,
    SUPERPOSITION_CMAP,
    SURFACE,
    WEIGHT_CMAP,
    paper_style,
    superposition_norm,
    weight_norm,
)

MONO_SHADE = "#ece6f3"  # light lavender, as on p. 33
POLY_SHADE = "#fbf1c7"  # light yellow
DEAD_SHADE = "#f1f0ec"
MIN_BLOCK = 0.02  # display only: weights smaller than this are not drawn (slivers)
WEIGHT_COLOR_PERCENTILE = 98.0  # W heatmaps saturate beyond this percentile of |W|


@dataclass
class NeuronPanel:
    """One model as the p. 33 rows show it.

    Attributes:
        title: row label (e.g. the sparsity).
        W: ``(hidden, feat)`` weights (neurons × features); for the absolute-value model, W₁
            after footnote-15 normalization.
        superposition: ``(feat,)`` superposition score of each feature (colors the blocks).
        polysemantic: ``(hidden,)`` features per neuron (0 = dead, 1 = mono, ≥ 2 = poly).
        order: neuron display order.
    """

    title: str
    W: np.ndarray
    superposition: np.ndarray
    polysemantic: np.ndarray
    order: np.ndarray


def neuron_order_by_top_feature(W: np.ndarray) -> np.ndarray:
    """Neurons sorted by the index (importance) of their largest-|weight| feature (p. 36).

    Ties are broken by that weight's magnitude, largest first.
    """
    top = np.abs(W).argmax(axis=1)
    magnitude = np.abs(W).max(axis=1)
    return np.lexsort((-magnitude, top))


def plot_stack(
    ax: Axes,
    W: np.ndarray,
    superposition: np.ndarray,
    polysemantic: np.ndarray,
    order: np.ndarray,
    absolute: bool = False,
) -> None:
    """Draw a neuron stack plot of ``W`` (hidden, feat) on ``ax``.

    Args:
        absolute: stack |W| upward only (the p. 36 figures), instead of signed stacks.
    """
    colors = SUPERPOSITION_CMAP(superposition_norm()(superposition))
    for col, h in enumerate(order):
        shade = (
            DEAD_SHADE
            if polysemantic[h] == 0
            else MONO_SHADE
            if polysemantic[h] == 1
            else POLY_SHADE
        )
        ax.axvspan(col - 0.5, col + 0.5, color=shade, linewidth=0)
        weights = np.abs(W[h]) if absolute else W[h]
        up, down = 0.0, 0.0
        for i in np.argsort(-np.abs(weights)):
            w = float(weights[i])
            if abs(w) < MIN_BLOCK:
                continue
            if w > 0:
                ax.bar(col, w, bottom=up, width=0.78, color=colors[i], edgecolor=SURFACE, lw=0.4)
                up += w
            else:
                ax.bar(col, w, bottom=down, width=0.78, color=colors[i], edgecolor=SURFACE, lw=0.4)
                down += w
    ax.axhline(0.0, color=INK_SECONDARY, linewidth=0.8)
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_xticks([])
    for spine in ("top", "right", "bottom"):
        ax.spines[spine].set_visible(False)


def plot_weight_matrix(ax: Axes, W: np.ndarray, order: np.ndarray, limit: float) -> None:
    """W drawn features × neurons (feature 0 on top), neurons in ``order`` (p. 31)."""
    ax.imshow(
        W[order].T,
        cmap=WEIGHT_CMAP,
        norm=weight_norm(limit),
        interpolation="nearest",
        aspect="auto",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def neuron_rows(
    panels: Sequence[NeuronPanel],
    suptitle: str = "",
    absolute: bool = False,
    show_matrix: bool = True,
) -> Figure:
    """The p. 33 layout: one row per model, W on the left, its neuron stack plot on the right.

    With ``absolute`` (p. 36) stacks show |W| and the matrix column is usually left out.
    """
    k = len(panels)
    n_hidden = panels[0].W.shape[0]
    # Color scale: a robust maximum, so one outlier weight does not wash out every panel
    # (larger weights saturate; display only).
    limit = max(
        1.0,
        float(
            np.percentile(
                np.abs(np.concatenate([p.W.ravel() for p in panels])), WEIGHT_COLOR_PERCENTILE
            )
        ),
    )
    stack_top = max(float(np.clip(p.W, 0, None).sum(axis=1).max()) for p in panels)
    stack_bottom = min(float(np.clip(p.W, None, 0).sum(axis=1).min()) for p in panels)
    if absolute:
        stack_top = max(float(np.abs(p.W).sum(axis=1).max()) for p in panels)
        stack_bottom = 0.0
    width = 0.13 * n_hidden + 2.0
    with paper_style():
        fig = plt.figure(figsize=((1.6 if show_matrix else 0.0) + width + 1.6, 1.25 * k + 0.9))
        ratios = [0.55 * n_hidden / 5 + 0.6, width] if show_matrix else [width]
        grid = fig.add_gridspec(k, len(ratios), width_ratios=ratios, hspace=0.35, wspace=0.08)
        for r, panel in enumerate(panels):
            if show_matrix:
                ax_w = fig.add_subplot(grid[r, 0])
                plot_weight_matrix(ax_w, panel.W, panel.order, limit)
                ax_w.set_ylabel(panel.title, fontsize=8, rotation=0, ha="right", va="center")
            ax_s = fig.add_subplot(grid[r, -1])
            plot_stack(
                ax_s, panel.W, panel.superposition, panel.polysemantic, panel.order, absolute
            )
            if absolute:  # |W| stacks differ by an order of magnitude across sparsities
                ax_s.set_ylim(0.0, float(np.abs(panel.W).sum(axis=1).max()) * 1.08 + 0.05)
            else:
                ax_s.set_ylim(stack_bottom * 1.08 - 0.05, stack_top * 1.08 + 0.05)
            ax_s.tick_params(axis="y", labelsize=7)
            n_mono = int((panel.polysemantic == 1).sum())
            n_poly = int((panel.polysemantic >= 2).sum())
            label = panel.title if not show_matrix else ""
            ax_s.set_title(
                f"{label}{'  ·  ' if label else ''}{n_mono} monosemantic, {n_poly} polysemantic",
                fontsize=8,
                loc="left",
                pad=2,
            )
        fig.subplots_adjust(
            left=0.12, right=0.86, top=0.93 if suptitle else 0.97, bottom=0.05 + 0.35 / k
        )
        cax = fig.add_axes((0.885, 0.55, 0.012, 0.3))
        cbar = fig.colorbar(
            ScalarMappable(superposition_norm(), SUPERPOSITION_CMAP), cax=cax, ticks=[0, 1]
        )
        cbar.ax.set_yticklabels(["0", "≥1"])
        cbar.set_label("feature's Σⱼ≠ᵢ (Ŵᵢ·Wⱼ)²", fontsize=7.5)
        cbar.outline.set_visible(False)  # type: ignore[union-attr]
        cbar.ax.tick_params(length=2, labelsize=7)
        handles = [
            Patch(color=MONO_SHADE, label="monosemantic neuron"),
            Patch(color=POLY_SHADE, label="polysemantic neuron"),
            Patch(color=DEAD_SHADE, label="dead neuron"),
        ]
        fig.legend(
            handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=3, fontsize=7
        )
        if show_matrix:
            cax_w = fig.add_axes((0.885, 0.2, 0.012, 0.2))
            cbar_w = fig.colorbar(
                ScalarMappable(weight_norm(limit), WEIGHT_CMAP), cax=cax_w, ticks=[-limit, 0, limit]
            )
            cbar_w.set_label("weight", fontsize=7.5)
            cbar_w.outline.set_visible(False)  # type: ignore[union-attr]
            cbar_w.ax.tick_params(length=2, labelsize=7)
        if suptitle:
            fig.suptitle(suptitle, x=0.02, ha="left", fontsize=10.5)
    return fig


def abs_weights_figure(
    W1: np.ndarray, W2: np.ndarray, order: np.ndarray, roles: Sequence[str], title: str = ""
) -> Figure:
    """The p. 36 "raw weights" view of a small absolute-value model.

    Top: W₁ (input → neuron) weights, one group of bars per neuron, colored by input feature.
    Bottom: W₂ (neuron → output) weights, colored by output feature. Both after footnote-15
    normalization.

    Args:
        W1: ``(hidden, feat)``; W2: ``(feat, hidden)``; order: neuron display order;
        roles: one label per neuron (shown under the bars).
    """
    n_hidden, n_feat = W1.shape
    colors = [SERIES[i % len(SERIES)] for i in range(n_feat)]
    width = 0.8 / n_feat
    with paper_style():
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(0.75 * n_hidden + 2.4, 3.8), sharex=True)
        for ax, matrix, label in (
            (ax1, W1, "W₁ (input → neuron)"),
            (ax2, W2.T, "W₂ (neuron → output)"),
        ):
            for col, h in enumerate(order):
                for i in range(n_feat):
                    ax.bar(
                        col - 0.4 + (i + 0.5) * width,
                        matrix[h, i],
                        width=width,
                        color=colors[i],
                        label=f"feature {i}" if col == 0 else None,
                    )
            ax.axhline(0.0, color=INK_SECONDARY, linewidth=0.8)
            ax.set_ylabel(label)
            ax.set_ylim(-1.25, 1.25)
            ax.grid(axis="y")
        ax2.set_xticks(range(n_hidden))
        ax2.set_xticklabels([roles[h] for h in order], fontsize=7, rotation=30, ha="right")
        ax1.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=n_feat)
        if title:
            ax1.set_title(title, loc="left")
        fig.tight_layout()
    return fig


def neuron_count_figure(
    feature_probability: np.ndarray,
    monosemantic: np.ndarray,
    polysemantic: np.ndarray,
    title: str = "",
) -> Figure:
    """Monosemantic and polysemantic neuron counts against feature density (p. 32 inset)."""
    with paper_style():
        fig, ax = plt.subplots(figsize=(4.8, 3.0))
        ax.plot(
            feature_probability, monosemantic, marker="o", color="#6a4c9c", label="monosemantic"
        )
        ax.plot(
            feature_probability, polysemantic, marker="o", color="#d9a400", label="polysemantic"
        )
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("feature density 1 − S  (dense → sparse)")
        ax.set_ylabel("neurons (best model)")
        ax.grid(axis="y")
        ax.legend(loc="center left")
        if title:
            ax.set_title(title, loc="left")
        fig.tight_layout()
    return fig


def capacity_figure(
    feature_probability: np.ndarray,
    computed_well: np.ndarray,
    monosemantic: np.ndarray,
    polysemantic: np.ndarray,
    n_hidden: int,
    title: str = "",
) -> Figure:
    """Features computed well, and neuron counts, against feature density (pp. 36–37).

    Two neurons per feature is the capacity without superposition, so the m/2 line marks where
    computing more features requires superposition.
    """
    with paper_style():
        fig, ax = plt.subplots(figsize=(5.0, 3.2))
        ax.plot(
            feature_probability, computed_well, marker="o", color=SERIES[1],
            label="features computed well",
        )  # fmt: skip
        ax.plot(
            feature_probability, polysemantic, marker="s", color="#d9a400",
            label="polysemantic neurons",
        )  # fmt: skip
        ax.plot(
            feature_probability, monosemantic, marker="s", color="#6a4c9c",
            label="monosemantic neurons",
        )  # fmt: skip
        ax.axhline(
            n_hidden / 2, color=INK_SECONDARY, linewidth=1.0, linestyle=(0, (4, 3)), label="m/2"
        )
        ax.set_xscale("log")
        ax.invert_xaxis()
        ax.set_xlabel("feature density 1 − S  (dense → sparse)")
        ax.set_ylabel("count (best model)")
        if title:
            ax.set_title(title, loc="left")
        ax.grid(axis="y")
        ax.legend(loc="upper left", fontsize=7)
        fig.tight_layout()
    return fig
