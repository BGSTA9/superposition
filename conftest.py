"""Repository-wide pytest configuration.

Test runs (replication tests, notebook smoke tests) must never write into the owner's
results/ directory, so unless SUPERPOSITION_RESULTS is already set, point it at a fresh
temporary directory for the whole session. Notebook kernels started by nbmake inherit it.
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


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Notebook smoke tests (collected by nbmake) belong to the slow tier (SPEC §5)."""
    for item in items:
        if item.path.suffix == ".ipynb":
            item.add_marker(pytest.mark.slow)
