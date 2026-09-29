"""Figures for sparse autoencoders scored against a toy model's true features (E10).

Paper: Approach 2: Finding an Overcomplete Basis, pp. 42–44. Functions take arrays or tables
and return figures.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter

from superposition.plotting.style import (
    IMPORTANCE_CMAP,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    SERIES,
    SURFACE,
    paper_style,
)

# Single-hue sequential ramp (blue, light → dark) for "share recovered", 0 → 1.
RECOVERY_CMAP = LinearSegmentedColormap.from_list(
    "recovery", ["#eef4fc", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"]
)
DEAD_LATENT = "#c9c8c2"


def dictionary_2d_figure(
    W_true: np.ndarray,
    panels: Sequence[tuple[str, np.ndarray, np.ndarray, np.ndarray, float]],
    threshold: float,
    title: str = "",
) -> Figure:
    """True feature arrows (colored by importance, as p. 2) with SAE decoder directions.

    Args:
        W_true: ``(2, feat)`` the toy model's W.
        panels: ``(title, decoder (2, latent), mask (latent,), alive (latent,), recovered)``.
            Live latents are dark lines, dead ones light; masked latents are not drawn.
        threshold: the cos threshold the recovered share refers to.
    """
    n_feat = W_true.shape[1]
    colors = IMPORTANCE_CMAP(np.linspace(1.0, 0.0, max(n_feat, 2)))[:n_feat]
    radius = 1.15 * max(1.0, float(np.linalg.norm(W_true, axis=0).max()))
    with paper_style():
        fig, axes = plt.subplots(1, len(panels), figsize=(3.3 * len(panels), 3.7))
        axes = np.atleast_1d(axes)
        for ax, (name, decoder, mask, alive, recovered) in zip(axes, panels, strict=True):
            for j in np.flatnonzero(mask):
                d = decoder[:, j] / max(np.linalg.norm(decoder[:, j]), 1e-12) * radius
                ax.plot(
                    [0, d[0]], [0, d[1]], color=INK_SECONDARY if alive[j] else DEAD_LATENT,
                    lw=1.0, ls="-" if alive[j] else (0, (2, 2)), zorder=1,
                )  # fmt: skip
            for i in range(n_feat):
                ax.annotate(
                    "",
                    xy=(W_true[0, i], W_true[1, i]),
                    xytext=(0, 0),
                    arrowprops={"arrowstyle": "-|>", "color": colors[i], "lw": 2.4},
                    zorder=2,
                )
                ax.text(
                    1.12 * W_true[0, i], 1.12 * W_true[1, i], str(i), ha="center",
                    va="center", fontsize=8, color=INK,
                )  # fmt: skip
            ax.set_xlim(-radius * 1.1, radius * 1.1)
            ax.set_ylim(-radius * 1.1, radius * 1.1)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.set_title(
                f"{name}\nrecovered at cos > {threshold:g}: {recovered:.0%}", fontsize=8, loc="left"
            )
        handles = [
            Line2D([], [], color=colors[0], lw=2.4, label="true feature Wᵢ (bright = important)"),
            Line2D([], [], color=INK_SECONDARY, lw=1.0, label="live SAE latent (decoder column)"),
            Line2D([], [], color=DEAD_LATENT, lw=1.0, ls=(0, (2, 2)), label="dead latent"),
        ]
        fig.legend(
            handles=handles, loc="lower center", ncol=3, fontsize=7, bbox_to_anchor=(0.5, -0.02)
        )
        if title:
            fig.suptitle(title, x=0.02, ha="left", fontsize=10)
        fig.tight_layout(rect=(0, 0.06, 1, 0.9))
    return fig


def _setting_column(row: dict) -> str:
    if row["topk"]:
        return f"TopK {int(row['topk'])}"
    return f"λ={row['l1']:g}\n{'resample' if row['resample'] else 'no resample'}"


def recovery_sweep_figure(
    table: pd.DataFrame, thresholds: dict[str, float], title: str = ""
) -> Figure:
    """One heatmap per toy model: share of represented features recovered, by SAE setting.

    Rows are dictionary sizes; columns are (λ, resampling) pairs and TopK variants. Each
    model uses its own cos threshold (0.95 for the pentagon, 0.9 otherwise).
    """
    models = list(dict.fromkeys(table["toy_model"]))
    norm = Normalize(0.0, 1.0)
    n_sizes = [int(table.loc[table["toy_model"] == m, "size"].nunique()) for m in models]
    with paper_style():
        fig = plt.figure(figsize=(8.6, 0.42 * sum(n_sizes) + 1.25 * len(models) + 0.6))
        layout = fig.add_gridspec(len(models), 1, height_ratios=n_sizes, hspace=1.3)
        axes = [fig.add_subplot(layout[i, 0]) for i in range(len(models))]
        for ax, model in zip(axes, models, strict=True):
            rows = [dict(r) for r in table.to_dict("records") if r["toy_model"] == model]
            col = f"recovered_{thresholds[model]:g}"
            sizes = sorted({int(r["size"]) for r in rows})
            columns = list(dict.fromkeys(_setting_column(r) for r in rows))
            grid = np.full((len(sizes), len(columns)), np.nan)
            for r in rows:
                grid[sizes.index(int(r["size"])), columns.index(_setting_column(r))] = r[col]
            ax.imshow(grid, cmap=RECOVERY_CMAP, norm=norm, aspect="auto")
            for a in range(len(sizes)):
                for b in range(len(columns)):
                    if np.isfinite(grid[a, b]):
                        ax.text(
                            b, a, f"{grid[a, b]:.0%}", ha="center", va="center", fontsize=6.5,
                            color=SURFACE if grid[a, b] > 0.55 else INK,
                        )  # fmt: skip
            ax.set_yticks(range(len(sizes)))
            ax.set_yticklabels([f"{s} latents" for s in sizes], fontsize=7)
            ax.set_xticks(range(len(columns)))
            ax.set_xticklabels(columns, fontsize=6.5)
            ax.set_title(f"{model}  (cos > {thresholds[model]:g})", loc="left", fontsize=8.5)
            for spine in ax.spines.values():
                spine.set_visible(False)
        cbar = fig.colorbar(ScalarMappable(norm, RECOVERY_CMAP), ax=axes, fraction=0.025, pad=0.02)
        cbar.set_label("share of represented features recovered", fontsize=7.5)
        cbar.outline.set_visible(False)  # type: ignore[union-attr]
        if title:
            fig.suptitle(title, x=0.02, ha="left", fontsize=10)
    return fig


def reconstruction_vs_recovery_figure(
    table: pd.DataFrame, thresholds: dict[str, float], title: str = ""
) -> Figure:
    """Small multiples, one per toy model: fraction of variance unexplained vs share recovered.

    Circles are ReLU + L1 SAEs, triangles TopK SAEs. Points low on the left reconstruct well
    but do not find the features.
    """
    models = list(dict.fromkeys(table["toy_model"]))
    with paper_style():
        fig, axes = plt.subplots(
            1, len(models), figsize=(2.6 * len(models) + 0.6, 3.1), sharey=True, squeeze=False
        )
        for ax, model in zip(axes[0], models, strict=True):
            rows = [dict(r) for r in table.to_dict("records") if r["toy_model"] == model]
            col = f"recovered_{thresholds[model]:g}"
            for topk, marker, color, label in (
                (False, "o", SERIES[0], "ReLU + L1"),
                (True, "^", SERIES[1], "TopK"),
            ):
                chosen = [r for r in rows if bool(r["topk"]) == topk]
                ax.scatter(
                    [max(float(r["fvu"]), 1e-4) for r in chosen], [float(r[col]) for r in chosen],
                    marker=marker, s=34, color=color, edgecolor=SURFACE, linewidth=0.8, label=label,
                )  # fmt: skip
            ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel("fraction of variance unexplained")
            ax.set_title(f"{model}\n(cos > {thresholds[model]:g})", fontsize=8, loc="left")
            ax.grid(True, color="#ecebe6")
        axes[0][0].set_ylabel("share recovered")
        handles, labels = axes[0][0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.04))
        if title:
            fig.suptitle(title, x=0.02, ha="left", fontsize=10)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        for ax in axes[0]:
            ax.tick_params(axis="x", colors=INK_MUTED)
    return fig
