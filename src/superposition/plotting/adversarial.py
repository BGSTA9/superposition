"""Adversarial-vulnerability figures.

Paper: Relationship to Adversarial Robustness, pp. 29–30:

* p. 29: vulnerability (L′ − L)/(L′₀ − L₀), relative to a non-superposition model, against
  1/(1 − S) on a log axis, above the features per dimension 1/D* = ‖W‖²_F/m on the same axis;
* p. 30: vulnerability plotted directly against features per dimension.

Functions take arrays and return figures.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from superposition.plotting.style import INK, INK_SECONDARY, SERIES, paper_style


@dataclass
class VulnerabilityCurve:
    """One sparsity sweep: x = 1/(1 − S), y = relative vulnerability and 1/D*."""

    label: str
    inverse_density: np.ndarray
    vulnerability: np.ndarray
    features_per_dimension: np.ndarray


def _times(value: float, _: object) -> str:
    return f"{value:g}×"


def _log_ticks(values: np.ndarray) -> list[float]:
    """1–2–5 ticks (1, 2, 3, 5, 10, 20, …) inside the range of ``values``, as on p. 29."""
    low, high = float(np.min(values)), float(np.max(values))
    ticks = [
        m * 10.0**e
        for e in range(-3, 6)
        for m in (1, 2, 3, 5)
        if low * 0.999 <= m * 10.0**e <= high * 1.001
    ]
    return ticks if len(ticks) >= 2 else [low, high]


def vulnerability_figure(curves: Sequence[VulnerabilityCurve], title: str = "") -> Figure:
    """The p. 29 layout: vulnerability over features per dimension, one column per setting."""
    k = len(curves)
    with paper_style():
        fig, axes = plt.subplots(2, k, figsize=(3.6 * k + 0.6, 4.6), sharex="col", squeeze=False)
        for c, curve in enumerate(curves):
            ax_v, ax_d = axes[0, c], axes[1, c]
            color = SERIES[c % len(SERIES)]
            ax_v.plot(curve.inverse_density, curve.vulnerability, marker="o", color=color)
            ax_v.axhline(1.0, color=INK_SECONDARY, linewidth=0.7, linestyle=(0, (2, 2)))
            ax_v.yaxis.set_major_formatter(FuncFormatter(_times))
            ax_v.set_title(curve.label, loc="left")
            ax_d.plot(curve.inverse_density, curve.features_per_dimension, marker="o", color=INK)
            ax_d.axhline(1.0, color=INK_SECONDARY, linewidth=0.7, linestyle=(0, (2, 2)))
            ax_d.set_xscale("log")
            ax_d.set_xlabel("1/(1 − S)  (log scale; dense → sparse)")
            for ax in (ax_v, ax_d):
                ax.grid(axis="y")
            ax_d.set_xticks(_log_ticks(curve.inverse_density))
            ax_d.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax_d.xaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
            if c == 0:
                ax_v.set_ylabel("vulnerability\n(L′ − L)/(L′₀ − L₀)")
                ax_d.set_ylabel("features per\ndimension 1/D*")
        if title:
            fig.suptitle(title, x=0.02, ha="left", fontsize=10.5)
        fig.tight_layout()
    return fig


def vulnerability_scatter(curves: Sequence[VulnerabilityCurve], title: str = "") -> Figure:
    """The p. 30 view: vulnerability against features per dimension, one series per setting."""
    with paper_style():
        fig, ax = plt.subplots(figsize=(4.4, 3.6))
        for c, curve in enumerate(curves):
            ax.plot(
                curve.features_per_dimension,
                curve.vulnerability,
                marker="o",
                linewidth=1.0,
                color=SERIES[c % len(SERIES)],
                label=curve.label,
            )
        ax.set_xlabel("features per dimension 1/D*")
        ax.set_ylabel("vulnerability")
        ax.yaxis.set_major_formatter(FuncFormatter(_times))
        ax.grid(axis="y")
        ax.legend(loc="upper left")
        if title:
            ax.set_title(title, loc="left", color=INK)
        fig.tight_layout()
    return fig
