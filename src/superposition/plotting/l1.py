"""Figures for the L1 penalty on the hidden layer (E11).

Paper: Approach 1: Creating Models Without Superposition, p. 42. Functions take tables and
return figures.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from superposition.plotting.style import INK_MUTED, INK_SECONDARY, PANEL, SERIES, paper_style

LINTHRESH = 1e-3  # symlog x-axis: λ = 0 sits left of the logarithmic part


def _rows(table: pd.DataFrame, setup: str) -> list[dict]:
    return [dict(r) for r in table.to_dict("records") if r["setup"] == setup]


def _free(row: dict) -> bool:
    return (
        int(row["represented"]) >= 1
        and int(row["superposed"]) == 0
        and not bool(row["formal_superposition"])
    )


def tradeoff_figure(table: pd.DataFrame, n_hidden: dict[str, int], title: str = "") -> Figure:
    """Per setup (columns): reconstruction loss vs λ (top), and features represented and
    superposed vs λ (bottom, with m dashed). λ values without superposition are shaded."""
    setups = list(dict.fromkeys(table["setup"]))
    with paper_style():
        fig, axes = plt.subplots(
            2, len(setups), figsize=(4.4 * len(setups), 5.4), sharex="col", squeeze=False
        )
        for c, setup in enumerate(setups):
            rows = _rows(table, setup)
            lam = np.array([float(r["lambda"]) for r in rows])
            loss = np.array([float(r["loss"]) for r in rows])
            ax_l, ax_n = axes[0, c], axes[1, c]
            for ax in (ax_l, ax_n):
                for r in rows:
                    if _free(r):
                        x = float(r["lambda"])
                        ax.axvspan(x * 0.87, x * 1.15, color=PANEL, linewidth=0, zorder=0)
            ax_l.plot(lam, loss, marker="o", color=SERIES[0])
            ax_l.set_ylabel("reconstruction loss\n(no penalty)")
            ax_l.set_title(setup, loc="left")
            ax_n.plot(
                lam, [int(r["represented"]) for r in rows], marker="o", color=SERIES[0],
                label="represented (‖Wᵢ‖ > 0.5)",
            )  # fmt: skip
            ax_n.plot(
                lam, [int(r["superposed"]) for r in rows], marker="s", color=SERIES[1],
                label="superposed (score > 0.05)",
            )  # fmt: skip
            ax_n.axhline(n_hidden[setup], color=INK_MUTED, lw=1.0, ls=(0, (4, 3)), label="m")
            ax_n.set_ylabel("features")
            ax_n.set_xlabel("λ (L1 coefficient on h)")
            for ax in (ax_l, ax_n):
                ax.set_xscale("symlog", linthresh=LINTHRESH)
                ax.set_xlim(-0.3 * LINTHRESH, float(lam.max()) * 1.25)  # no negative λ
                ax.grid(axis="y")
        axes[1, 0].legend(loc="lower left")
        fig.text(
            0.99, 0.005, "shaded: no superposition", ha="right", va="bottom", fontsize=7,
            color=INK_SECONDARY,
        )  # fmt: skip
        if title:
            fig.suptitle(title, x=0.02, ha="left", fontsize=10.5)
        fig.tight_layout()
    return fig


def tradeoff_curve_figure(table: pd.DataFrame, title: str = "") -> Figure:
    """Per setup: reconstruction loss (x) against superposed features (y), in λ order.

    Labels mark λ = 0 and the first superposition-free λ.
    """
    setups = list(dict.fromkeys(table["setup"]))
    with paper_style():
        fig, axes = plt.subplots(1, len(setups), figsize=(4.2 * len(setups), 3.3), squeeze=False)
        for ax, setup in zip(axes[0], setups, strict=True):
            rows = _rows(table, setup)
            loss = [float(r["loss"]) for r in rows]
            sup = [int(r["superposed"]) for r in rows]
            ax.plot(loss, sup, marker="o", color=SERIES[0], lw=1.6)
            first_free = next(
                (i for i, r in enumerate(rows) if float(r["lambda"]) > 0 and _free(r)), None
            )
            labeled = [(0, (8, -2))] + ([(first_free, (-6, 8))] if first_free is not None else [])
            for i, offset in labeled:
                ax.annotate(
                    f"λ = {float(rows[i]['lambda']):g}", (loss[i], sup[i]), xytext=offset,
                    textcoords="offset points", fontsize=7, color=INK_SECONDARY,
                    ha="left" if offset[0] > 0 else "right",
                )  # fmt: skip
            ax.set_ylim(-0.4, max(sup) * 1.12 + 0.2)
            ax.set_xscale("log")
            ax.set_xlabel("reconstruction loss (log)")
            ax.set_ylabel("superposed features")
            ax.set_title(setup, loc="left")
            ax.grid(True, color="#ecebe6")
        if title:
            fig.suptitle(title, x=0.02, ha="left", fontsize=10.5)
        fig.tight_layout()
    return fig
