"""E12 figures: the output activations and how many phase-diagram cells each regime takes.

Paper: Comments & Replications, p. 51 (Redwood: the phase diagrams depend on the activation
function). Functions take arrays and return figures; nothing here computes a metric or a label.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from superposition.plotting.phase import REGIME_COLORS, REGIME_NAMES
from superposition.plotting.style import CATEGORICAL, GRID, INK_SECONDARY, paper_style


def activation_shares_figure(
    u: np.ndarray,
    curves: Mapping[str, np.ndarray],
    shares: Mapping[str, Mapping[str, Mapping[str, int]]],
    regimes: Sequence[str] = ("not_learned", "superposition", "dedicated", "confused"),
    suptitle: str = "",
) -> Figure:
    """Left: each activation σ(u). Right: cells per regime, empirical and theory, per activation.

    Args:
        u: ``(points,)`` the x axis of the curves.
        curves: activation name → σ(u).
        shares: activation name → source (``"empirical"``, ``"theory"``) → regime → cell count.
    """
    names = list(curves)
    with paper_style():
        fig, (left, right) = plt.subplots(
            1, 2, figsize=(11.0, 3.9), gridspec_kw={"width_ratios": [1.0, 1.5]}
        )
        left.axhline(0.0, color=GRID, linewidth=0.8, zorder=0)
        left.axvline(0.0, color=GRID, linewidth=0.8, zorder=0)
        for k, name in enumerate(names):
            left.plot(u, curves[name], color=CATEGORICAL[k % len(CATEGORICAL)], label=name)
        left.set_xlabel("pre-activation u = (WᵀWx + b)ᵢ")
        left.set_ylabel("output σ(u)")
        left.set_ylim(-0.6, 3.1)
        left.set_title("the output activations", fontsize=9.5, loc="left")
        left.legend(loc="upper left")

        sources = [s for s in ("empirical", "theory") if any(s in shares[n] for n in names)]
        height = 0.8 / len(sources)
        y = np.arange(len(names))
        for j, source in enumerate(sources):
            offset = (j - (len(sources) - 1) / 2) * height
            totals = np.array([sum(shares[n].get(source, {}).values()) for n in names], float)
            start = np.zeros(len(names))
            for regime in regimes:
                counts = np.array([shares[n].get(source, {}).get(regime, 0) for n in names], float)
                share = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
                right.barh(
                    y + offset,
                    share,
                    height=height * 0.92,
                    left=start,
                    color=REGIME_COLORS[regime],
                    label=REGIME_NAMES[regime] if j == 0 else None,
                    edgecolor="white",
                    linewidth=0.6,
                )
                start += share
            for k in range(len(names)):
                right.text(
                    1.01, y[k] + offset, source, va="center", fontsize=7, color=INK_SECONDARY
                )
        right.set_yticks(y, names)
        right.invert_yaxis()
        right.set_xlim(0, 1)
        right.set_xlabel("share of the phase diagram's cells")
        right.set_title(
            "regimes of the extra feature (top bar: empirical; bottom: theory)",
            fontsize=9.5,
            loc="left",
        )
        right.legend(loc="lower center", bbox_to_anchor=(0.5, -0.36), ncol=len(regimes))
        if suptitle:
            fig.suptitle(suptitle, x=0.02, ha="left", fontsize=10.5)
        fig.tight_layout(rect=(0, 0, 0.97, 0.93 if suptitle else 1.0))
    return fig
