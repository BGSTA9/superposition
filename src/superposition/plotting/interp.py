"""Figures for reverse-engineering absolute-value models (E09).

Paper: The Asymmetric Superposition Motif, pp. 38–39. The p. 39 figure draws each neuron as a
column with its input weights (W₁ row) above and its output weights (W₂ column) below, stacked
and colored by feature, positive weights above the axis and negative below; the motif's
neurons are picked out. :func:`role_weights_figure` draws the same view with each neuron's role
written under it. Functions take arrays and return figures.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from superposition.plotting.style import (
    CATEGORICAL,
    GRID,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    PANEL,
    SERIES,
    SURFACE,
    WEIGHT_CMAP,
    paper_style,
    weight_norm,
)

# Role kinds in display order (sides first, as p. 39 draws the "expected" neurons first).
KIND_ORDER = (
    "negative-side",
    "positive-side",
    "superposition",
    "asymmetric-superposition",
    "inhibitor",
    "unexplained",
    "dead",
)
# Roles take categorical slots in order; "unexplained" and "dead" are not roles, so neutral grays.
KIND_COLORS = {
    **dict(zip(KIND_ORDER[:5], CATEGORICAL, strict=False)),
    "unexplained": INK_MUTED,
    "dead": GRID,
}
WEIGHT_CLIP = 2.2  # output weights beyond this are drawn clipped, with an arrow (as p. 39)
MIN_BLOCK = 0.02  # weights smaller than this are not drawn (display only)


def motif_order(kinds: Sequence[str], labels: Sequence[str], W1n: np.ndarray) -> np.ndarray:
    """Neuron display order: by role kind (:data:`KIND_ORDER`), then by main feature."""
    rank = {k: i for i, k in enumerate(KIND_ORDER)}
    main = np.abs(W1n).argmax(axis=1)
    keys = [(rank.get(k, len(rank)), int(main[h]), labels[h]) for h, k in enumerate(kinds)]
    return np.array(sorted(range(len(kinds)), key=lambda h: keys[h]))


def _stack(ax: Axes, weights: np.ndarray, order: np.ndarray, clip: float | None) -> None:
    """Signed stacked bars, one column per neuron (rows of ``weights``), colored by feature.

    Smaller weights sit next to the axis and larger ones outside them, so that with ``clip``
    only the outermost block runs off the chart; it is cut at ``clip`` and marked with an arrow
    and its true value, as the p. 39 figure marks its large inhibitory weights.
    """
    for col, h in enumerate(order):
        row = weights[h]
        for sign in (1.0, -1.0):
            idx = [i for i in np.argsort(np.abs(row)) if sign * row[i] >= MIN_BLOCK]
            base = 0.0
            for i in idx:
                w = float(row[i])
                color = CATEGORICAL[i % len(CATEGORICAL)]
                height = abs(w) if clip is None else min(abs(w), max(clip - base, 0.0))
                ax.bar(
                    col, sign * height, bottom=sign * base, width=0.74, color=color,
                    edgecolor=SURFACE, lw=1.0,
                )  # fmt: skip
                if clip is not None and height < abs(w) - 1e-9:
                    tip = sign * (base + height)
                    ax.annotate(
                        f"{w:.1f}",
                        xy=(col, tip),
                        xytext=(col, tip + sign * 0.45),
                        ha="center",
                        va="bottom" if sign > 0 else "top",
                        fontsize=6.5,
                        color=INK_SECONDARY,
                        arrowprops={"arrowstyle": "<-", "color": color, "lw": 1.2},
                    )
                base += height
    ax.axhline(0.0, color=INK, linewidth=0.8)
    ax.set_xlim(-0.5, len(order) - 0.5)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)


def role_weights_figure(
    W1n: np.ndarray,
    W2n: np.ndarray,
    labels: Sequence[str],
    order: np.ndarray,
    highlight: set[int] | None = None,
    title: str = "",
) -> Figure:
    """The p. 39 view of one absolute-value model, with every neuron's role.

    Args:
        W1n: ``(hidden, feat)`` input weights after footnote-15 normalization.
        W2n: ``(feat, hidden)`` output weights, normalized the same way.
        labels: each neuron's role label.
        order: neuron display order.
        highlight: neurons to pick out (the motif's), shaded as on p. 39.
    """
    n_hidden, n_feat = W1n.shape
    highlight = highlight or set()
    with paper_style():
        fig, (ax_in, ax_out) = plt.subplots(
            2, 1, figsize=(0.62 * n_hidden + 2.2, 5.2), sharex=True, height_ratios=[1, 1.25]
        )
        for ax in (ax_in, ax_out):
            for col, h in enumerate(order):
                if h in highlight:
                    ax.axvspan(col - 0.5, col + 0.5, color=PANEL, linewidth=0, zorder=0)
        _stack(ax_in, W1n, order, clip=None)
        _stack(ax_out, W2n.T, order, clip=WEIGHT_CLIP)
        ax_in.set_ylabel("input weights W₁\n(feature → neuron)")
        ax_out.set_ylabel("output weights W₂\n(neuron → output)")
        ax_in.set_ylim(-1.35, 1.35)
        ax_out.set_ylim(-WEIGHT_CLIP - 0.9, WEIGHT_CLIP + 0.9)
        for ax in (ax_in, ax_out):
            ax.grid(axis="y")
        ax_out.set_xticks(range(n_hidden))
        ax_out.set_xticklabels(
            [f"{h}: {labels[h]}" for h in order], rotation=35, ha="right", fontsize=7
        )
        handles = [
            Patch(color=CATEGORICAL[i % len(CATEGORICAL)], label=f"feature {i}")
            for i in range(n_feat)
        ]
        if highlight:
            handles.append(Patch(color=PANEL, label="motif neurons"))
        fig.legend(
            handles=handles,
            loc="upper right",
            ncol=min(len(handles), 4),
            fontsize=7,
            bbox_to_anchor=(0.99, 1.0),
        )
        if title:
            fig.suptitle(title, x=0.02, ha="left", y=1.06, fontsize=10)
        fig.tight_layout()
    return fig


def ablation_figure(
    zero: np.ndarray,
    mean: np.ndarray,
    labels: Sequence[str],
    order: np.ndarray,
    title: str = "",
) -> Figure:
    """Zero- and mean-ablation matrices: Δ(per-feature loss) / loss of predicting the mean.

    Rows are neurons in ``order`` (labeled with their roles), columns features. Red = the
    feature's loss rises when the neuron is ablated, blue = it falls.
    """
    n_hidden, n_feat = zero.shape
    limit = max(1.0, float(np.percentile(np.abs(np.concatenate([zero, mean])), 95)))
    norm = weight_norm(limit)
    with paper_style():
        fig, axes = plt.subplots(
            1, 2, figsize=(2 * (0.45 * n_feat + 1.2) + 2.6, 0.34 * n_hidden + 1.4), sharey=True
        )
        for ax, matrix, name in ((axes[0], zero, "zero-ablated"), (axes[1], mean, "mean-ablated")):
            ax.imshow(matrix[order], cmap=WEIGHT_CMAP, norm=norm, aspect="auto")
            for r, h in enumerate(order):
                for i in range(n_feat):
                    value = float(matrix[h, i])
                    if abs(value) >= 0.01:
                        ax.text(
                            i,
                            r,
                            f"{value:.2g}",
                            ha="center",
                            va="center",
                            fontsize=6,
                            color=SURFACE if abs(value) > 0.6 * limit else INK,
                        )
            ax.set_xticks(range(n_feat))
            ax.set_xticklabels([str(i) for i in range(n_feat)])
            ax.set_xlabel("feature whose loss changes")
            ax.set_title(name, loc="left")
            for spine in ax.spines.values():
                spine.set_visible(False)
        axes[0].set_yticks(range(n_hidden))
        axes[0].set_yticklabels([f"{h}: {labels[h]}" for h in order], fontsize=7)
        cbar = fig.colorbar(
            ScalarMappable(norm, WEIGHT_CMAP),
            ax=axes,
            fraction=0.04,
            pad=0.02,
            ticks=[-limit, 0, limit],
        )
        cbar.set_label("Δ loss / loss of predicting the mean", fontsize=7.5)
        cbar.outline.set_visible(False)  # type: ignore[union-attr]
        if title:
            fig.suptitle(title, x=0.02, ha="left", fontsize=10)
    return fig


def _model_label(row: pd.Series | dict) -> str:
    """``n6_m10`` at 0.2 → "n = 6, m = 10, 1 − S = 0.2"."""
    n, m = (part[1:] for part in str(row["shape"]).split("_"))
    return f"n = {n}, m = {m}, 1 − S = {row['feature_probability']:g}"


def rebuild_figure(summary: pd.DataFrame, rebuild: pd.DataFrame, title: str = "") -> Figure:
    """Fraction of each model's loss reduction the hand-built rebuild recovers.

    One row per analyzed model (graded settings first); two bars: idealized biases and the
    trained biases. 1 = the rebuild is as good as the trained model; 0 = no better than
    predicting the mean.
    """
    rows = summary.sort_values(
        ["graded", "shape", "feature_probability"], ascending=[False, True, False]
    )
    labels = [
        _model_label(r) + ("" if bool(r["graded"]) else "  (context)") for _, r in rows.iterrows()
    ]
    y = np.arange(len(rows))
    with paper_style():
        fig, ax = plt.subplots(figsize=(6.2, 0.36 * len(rows) + 1.3))
        ax.barh(
            y - 0.19, rows["rebuild_recovered"], height=0.36, color=SERIES[0],
            label="rebuild (idealized biases)",
        )  # fmt: skip
        ax.barh(
            y + 0.19, rows["rebuild_recovered_trained_bias"], height=0.36, color=SERIES[1],
            label="rebuild (trained biases)",
        )  # fmt: skip
        # Values in a column right of the 1.0 reference line (idealized / trained biases).
        pairs = zip(rows["rebuild_recovered"], rows["rebuild_recovered_trained_bias"], strict=True)
        for yi, (ideal, trained) in zip(y, pairs, strict=True):
            ax.text(
                1.04, float(yi), f"{ideal:.2f} / {trained:.2f}", va="center", fontsize=6.5,
                color=INK_SECONDARY,
            )  # fmt: skip
        low = min(
            0.0, float(rows[["rebuild_recovered", "rebuild_recovered_trained_bias"]].min().min())
        )
        ax.set_xlim(low - 0.02, 1.25)
        ax.set_xticks([t for t in np.arange(0.0, 1.01, 0.2) if t >= low])
        ax.axvline(1.0, color=INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
        ax.axvline(0.0, color=INK_SECONDARY, linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlabel("share of the trained model's loss reduction recovered")
        ax.grid(axis="x")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16 - 0.4 / len(rows)), ncol=2)
        if title:
            ax.set_title(title, loc="left")
        fig.tight_layout()
    return fig


def role_census_figure(summary: pd.DataFrame, roles: pd.DataFrame, title: str = "") -> Figure:
    """Stacked counts of neuron role kinds for the best model of every setting."""
    rows = summary.sort_values(
        ["graded", "shape", "feature_probability"], ascending=[False, True, False]
    )
    labels, counts = [], []
    for _, r in rows.iterrows():
        table = roles[
            (roles["shape"] == r["shape"])
            & (roles["feature_probability"] == r["feature_probability"])
        ]
        counts.append([int((table["kind"] == k).sum()) for k in KIND_ORDER])
        labels.append(_model_label(r) + ("" if bool(r["graded"]) else "  (context)"))
    counts_arr = np.array(counts)
    y = np.arange(len(rows))
    with paper_style():
        fig, ax = plt.subplots(figsize=(6.4, 0.36 * len(rows) + 1.5))
        left = np.zeros(len(rows))
        for k, kind in enumerate(KIND_ORDER):
            if not counts_arr[:, k].any():
                continue
            ax.barh(
                y, counts_arr[:, k], left=left, height=0.62, color=KIND_COLORS[kind],
                edgecolor=SURFACE, lw=1.0, label=kind,
            )  # fmt: skip
            left += counts_arr[:, k]
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=7.5)
        ax.invert_yaxis()
        ax.set_xlabel("neurons")
        ax.grid(axis="x")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14 - 0.4 / len(rows)), ncol=3)
        if title:
            ax.set_title(title, loc="left")
        fig.tight_layout()
    return fig
