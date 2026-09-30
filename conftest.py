"""Repository-wide pytest configuration.

Test runs (replication tests, notebook smoke tests) must never write into the owner's
results/ directory, so unless SUPERPOSITION_RESULTS is already set, point it at a fresh
temporary directory for the whole session. Notebook kernels started by nbmake inherit it.

Test tiers (docs/DECISIONS.md D-310):

* unmarked: the fast unit and property tier (CI, ``make test``).
* ``slow``: trains small models or runs short notebooks; the whole tier stays under 15 minutes
  on a laptop CPU (``make test-slow``).
* ``paper``: re-runs an experiment's whole quick profile, or a notebook that does; minutes to
  tens of minutes each (``make test-paper``). This tier is opt-in: paper tests are deselected
  unless the ``-m`` expression names ``paper``, so ``pytest -m "not slow"`` and a bare
  ``pytest`` stay fast. Run one with ``pytest -m paper tests/replication/test_<name>.py``.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

# Headless tests draw with the non-interactive Agg backend. This must be set before anything
# imports matplotlib: a GUI backend (macOS) cannot create figures off the main thread, which
# is where streamlit's AppTest runs page scripts.
os.environ.setdefault("MPLBACKEND", "Agg")

import pytest
from hypothesis import settings

from superposition.runs import RESULTS_ENV

# Deterministic property tests in CI; random exploration locally.
settings.register_profile("ci", derandomize=True, deadline=None)
if os.environ.get("CI"):
    settings.load_profile("ci")


def pytest_configure(config: pytest.Config) -> None:
    if not os.environ.get(RESULTS_ENV):
        os.environ[RESULTS_ENV] = tempfile.mkdtemp(prefix="superposition-test-results-")
    _use_this_interpreter_as_python3_kernel()


def _use_this_interpreter_as_python3_kernel() -> None:
    """Make the notebooks' ``python3`` kernel run *this* interpreter (the project env).

    Jupyter searches JUPYTER_PATH before the user's kernel directory, so a user-level
    ``python3`` kernelspec pointing at another Python would otherwise win and the notebooks
    could not import ``superposition``.
    """
    kernel_dir = Path(tempfile.mkdtemp(prefix="superposition-jupyter-")) / "kernels" / "python3"
    kernel_dir.mkdir(parents=True)
    spec = {
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "Python 3 (superposition test env)",
        "language": "python",
    }
    (kernel_dir / "kernel.json").write_text(json.dumps(spec))
    existing = os.environ.get("JUPYTER_PATH")
    root = str(kernel_dir.parent.parent)
    os.environ["JUPYTER_PATH"] = root if not existing else os.pathsep.join([root, existing])


# Notebooks that finish in a minute or two on a laptop CPU with an empty results folder. Every
# other notebook trains quick profiles when it finds no runs, so it belongs to the paper tier.
SLOW_TIER_NOTEBOOKS = frozenset(
    {
        "00_orientation",
        "01_features_directions_superposition",
        "02_demonstrating_superposition",
        "03_the_math",
    }
)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Assign notebook smoke tests (collected by nbmake) to a tier, and keep the paper tier
    opt-in (see the module docstring)."""
    for item in items:
        if item.path.suffix == ".ipynb":
            tier = "slow" if item.path.stem in SLOW_TIER_NOTEBOOKS else "paper"
            item.add_marker(getattr(pytest.mark, tier))
    if "paper" in (config.option.markexpr or ""):
        return
    paper = [item for item in items if item.get_closest_marker("paper")]
    if paper:
        config.hook.pytest_deselected(items=paper)
        items[:] = [item for item in items if not item.get_closest_marker("paper")]
