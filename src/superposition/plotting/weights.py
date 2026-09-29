"""WᵀW heatmaps with the bias, and feature-norm bars colored by superposition.

Paper: Basic Results, p. 10 (WᵀW with b alongside, diverging colors centered at 0) and p. 11
(‖Wᵢ‖ bars colored by Σ_{j≠i}(Ŵᵢ·Wⱼ)², dark = orthogonal, bright = ≥ 1; the p. 11 grid of
one column per sparsity level). Features are ordered from most to least important, top to
bottom.

All functions take NumPy arrays and return matplotlib objects; no metric is computed here.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure

from superposition.plotting.style import (
    INK_SECONDARY,
    PANEL,
    SUPERPOSITION_CMAP,
    WEIGHT_CMAP,
    paper_style,
    superposition_norm,
    weight_norm,
)


@dataclass
class WeightsPanel:
    """Everything one column of the p. 11 figure shows, for one trained model."""

    title: str
    gram: np.ndarray  # (feat, feat), WᵀW
    bias: np.ndarray  # (feat,)
    norms: np.ndarray  # (feat,), ‖Wᵢ‖
    superposition: np.ndarray  # (feat,), Σ_{j≠i}(Ŵᵢ·Wⱼ)²


def plot_gram(ax: Axes, gram: np.ndarray, limit: float = 1.0) -> None:
    """WᵀW as a heatmap, diverging and centered at 0 (p. 10)."""
    ax.imshow(gram, cmap=WEIGHT_CMAP, norm=weight_norm(limit), interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_bias(ax: Axes, bias: np.ndarray, limit: float = 1.0) -> None:
    """The bias vector b as a one-column heatmap beside WᵀW (p. 10)."""
    ax.imshow(
        bias[:, None],
        cmap=WEIGHT_CMAP,
        norm=weight_norm(limit),
        aspect="auto",
        interpolation="nearest",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_feature_bars(
    ax: Axes, norms: np.ndarray, superposition: np.ndarray, xmax: float = 1.2
) -> None:
    """Horizontal bars of ‖Wᵢ‖, feature 0 on top, colored by superposition score (p. 11)."""
    n = len(norms)
    colors = SUPERPOSITION_CMAP(superposition_norm()(superposition))
    ax.set_facecolor(PANEL)
    ax.barh(np.arange(n), norms, height=0.82, color=colors, edgecolor="none")
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xlim(0, xmax)
    ax.axvline(1.0, color=INK_SECONDARY, linewidth=0.6, linestyle=(0, (2, 2)))
    ax.set_yticks([])
    ax.set_xticks([0, 1])
    ax.tick_params(axis="x", length=2)
    for spine in ("left", "top", "right"):
        ax.spines[spine].set_visible(False)


def weights_grid(
    panels: Sequence[WeightsPanel],
    suptitle: str | None = None,
    group_labels: Sequence[tuple[int, str]] = (),
) -> Figure:
    """The p. 11 layout: one column per model; WᵀW + b on top, ‖Wᵢ‖ bars below.

    Args:
        panels: one :class:`WeightsPanel` per column, left to right.
        suptitle: optional figure title.
        group_labels: ``(first_column, text)`` headers spanning columns, e.g. model names.
    """
    k = len(panels)
    n = len(panels[0].norms)
    limit = 1.0
    xmax = max(1.2, *(float(np.max(p.norms)) * 1.05 for p in panels))
    with paper_style():
        fig = plt.figure(figsize=(max(1.55 * k + 1.2, 4.2), 4.3))
        grid = fig.add_gridspec(
            2,
            2 * k,
            width_ratios=[1.0, 0.09] * k,
            height_ratios=[1.0, 1.0],
            wspace=0.08,
            hspace=0.18,
        )
        for c, panel in enumerate(panels):
            ax_gram = fig.add_subplot(grid[0, 2 * c])
            ax_bias = fig.add_subplot(grid[0, 2 * c + 1])
            ax_bars = fig.add_subplot(grid[1, 2 * c : 2 * c + 2])
            plot_gram(ax_gram, panel.gram, limit)
            plot_bias(ax_bias, panel.bias, limit)
            plot_feature_bars(ax_bars, panel.norms, panel.superposition, xmax)
            ax_gram.set_title(panel.title, fontsize=8.5, pad=4)
            if c == 0:
                ax_gram.set_ylabel("WᵀW", fontsize=8.5)
                ax_bars.set_ylabel("‖Wᵢ‖  (features ↓)", fontsize=8.5)
            ax_bias.set_title("b", fontsize=8, pad=4, color=INK_SECONDARY)
        fig.subplots_adjust(left=0.05, right=0.86, top=0.78 if group_labels else 0.86, bottom=0.08)
        cax_w = fig.add_axes((0.885, 0.56, 0.012, 0.26))
        cbar_w = fig.colorbar(
            ScalarMappable(weight_norm(limit), WEIGHT_CMAP), cax=cax_w, ticks=[-1, 0, 1]
        )
        cbar_w.set_label("weight / bias", fontsize=8)
        cax_s = fig.add_axes((0.885, 0.12, 0.012, 0.26))
        cbar_s = fig.colorbar(
            ScalarMappable(superposition_norm(), SUPERPOSITION_CMAP), cax=cax_s, ticks=[0, 1]
        )
        cbar_s.ax.set_yticklabels(["0", "≥1"])
        cbar_s.set_label("Σⱼ≠ᵢ (Ŵᵢ·Wⱼ)²", fontsize=8)
        for cbar in (cbar_w, cbar_s):
            cbar.outline.set_visible(False)  # type: ignore[union-attr]
            cbar.ax.tick_params(length=2, labelsize=7)
        for first_col, text in group_labels:
            left = fig.axes[3 * first_col].get_position().x0
            fig.text(left, 0.875, text, fontsize=9.5, fontweight="bold", ha="left", va="bottom")
        if suptitle:
            fig.suptitle(suptitle, x=0.05, ha="left", fontsize=10.5, y=1.0)
        fig.text(
            0.05,
            0.005,
            f"n = {n} features, ordered most → least important",
            fontsize=7.5,
            color=INK_SECONDARY,
        )
    return fig


def weight_matrix_figure(W: np.ndarray, title: str = "W (features × neurons)") -> Figure:
    """W itself, features as rows (most important on top) and neurons as columns.

    Paper: Superposition in a Privileged Basis, p. 31. With a ReLU on the hidden layer the
    neuron basis is privileged, so W can be read directly, unlike the rotation-invariant WᵀW.
    """
    n_hidden, n_features = W.shape
    limit = max(1.0, float(np.abs(W).max()))
    with paper_style():
        fig, ax = plt.subplots(figsize=(0.35 * n_hidden + 1.6, 0.22 * n_features + 1.2))
        image = ax.imshow(
            W.T, cmap=WEIGHT_CMAP, norm=weight_norm(limit), interpolation="nearest", aspect="auto"
        )
        ax.set_xlabel("neuron")
        ax.set_ylabel("feature")
        ax.set_xticks(range(n_hidden))
        ax.set_yticks(range(n_features) if n_features <= 20 else range(0, n_features, 5))
        ax.set_title(title, fontsize=9.5, loc="left")
        cbar = fig.colorbar(image, ax=ax, fraction=0.08, pad=0.04)
        cbar.outline.set_visible(False)  # type: ignore[union-attr]
        cbar.ax.tick_params(labelsize=7)
        fig.tight_layout()
    return fig
