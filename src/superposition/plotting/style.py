"""Shared style for every paper-style figure: colors, colormaps, rc settings, saving.

SPEC §6 (plotting: "Paper-style figures, saved as PNG and PDF, sharing one style module")
and §3 (Visualization conventions).

Color jobs follow the paper and a colorblind-aware method:

* **Weights / WᵀW (diverging, polarity):** blue (−1) → neutral light gray (0) → red (+1),
  centered at 0, as on p. 10.
* **Superposition score (sequential, magnitude):** viridis from dark (0, orthogonal) to
  bright (≥ 1), as on p. 11. Viridis is perceptually uniform and colorblind-safe.
* **Importance (sequential):** the yellow → green part of viridis, most important brightest,
  echoing p. 2.
* **Categorical series** (e.g. linear vs ReLU output): fixed-order slots from a validated
  palette; color follows the entity, never its rank.

Plotting functions take arrays and return figures; nothing here computes metrics.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, Normalize, TwoSlopeNorm
from matplotlib.figure import Figure

# Text and chrome (text wears neutral ink, never a series color).
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8984"
GRID = "#e4e3de"
SURFACE = "#fcfcfb"
PANEL = "#f4f3ef"

# Categorical slots, fixed order (validated palette: blue, orange, aqua).
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
MODEL_COLORS = {"linear": SERIES[0], "relu_output": SERIES[1], "relu_hidden": SERIES[2]}
MODEL_LABELS = {"linear": "Linear", "relu_output": "ReLU output", "relu_hidden": "ReLU hidden"}

# Diverging map for weights: two hues with a neutral light-gray midpoint.
WEIGHT_CMAP = LinearSegmentedColormap.from_list(
    "superposition_weights",
    ["#1f3fb0", "#6f8fe0", "#ecebe8", "#e0755f", "#b01f24"],
)
SUPERPOSITION_CMAP = mpl.colormaps["viridis"]
IMPORTANCE_CMAP = ListedColormap(
    mpl.colormaps["viridis"](
        [0.42 + 0.58 * i / 255 for i in range(256)]  # teal-green (least) → yellow (most)
    ),
    name="importance",
)


def weight_norm(limit: float = 1.0) -> TwoSlopeNorm:
    """Diverging normalization centered at 0 with symmetric limits."""
    return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)


def superposition_norm() -> Normalize:
    """Superposition score 0 (dark) → 1 (bright); scores ≥ 1 saturate."""
    return Normalize(vmin=0.0, vmax=1.0, clip=True)


RC: dict[str, Any] = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK_MUTED,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "font.size": 9,
    "font.family": "sans-serif",
    "lines.linewidth": 2.0,
    "lines.markersize": 5,
    "figure.dpi": 100,
}


@contextmanager
def paper_style() -> Iterator[None]:
    """Context manager applying the shared rc settings."""
    with mpl.rc_context(RC):  # type: ignore[arg-type]
        yield


def save_figure(
    fig: Figure, stem: Path, dpi: int = 150, formats: tuple[str, ...] = ("png", "pdf")
) -> list[Path]:
    """Save ``fig`` as ``stem.png`` and ``stem.pdf``; return the paths. Closes the figure."""
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = stem.with_suffix(f".{fmt}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths
