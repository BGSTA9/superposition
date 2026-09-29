"""Run directories, manifests, checkpoints and latest-run lookup.

SPEC §5 (Run artifacts) and §2 (Provenance: "Every number or figure in the docs traces to a
logged run: config, seed and git commit").

A run lives in ``results/<experiment>/<run_id>/`` where ``run_id`` is a UTC timestamp plus a
short config hash, e.g. ``20260929T101500Z-3fa2c1d0``. It contains::

    config.yaml        the resolved config
    manifest.json      git commit + dirty flag, package versions, device, seed, times, status
    metrics.parquet    per-instance metrics (experiments may add more *.parquet tables)
    acceptance.json    each criterion with its value, threshold and status
    checkpoints/*.pt   trained parameters
    figures/*.png|pdf  figures
    log.txt            the run log

The results root is ``$SUPERPOSITION_RESULTS`` if set, else ``<project root>/results``.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from superposition.config import config_hash, save_config

RESULTS_ENV = "SUPERPOSITION_RESULTS"
LOGGER_NAME = "superposition"
_TRACKED_PACKAGES = ("superposition", "torch", "numpy", "einops", "pandas", "matplotlib", "scipy")


def project_root() -> Path:
    """The repository root: the nearest ancestor of the CWD (or of this file) with a
    ``pyproject.toml`` declaring ``name = "superposition"``; falls back to the CWD."""
    candidates = [Path.cwd(), *Path.cwd().parents, *Path(__file__).resolve().parents]
    for directory in candidates:
        pyproject = directory / "pyproject.toml"
        if pyproject.is_file() and 'name = "superposition"' in pyproject.read_text():
            return directory
    return Path.cwd()


def results_root() -> Path:
    """Where runs are stored (``$SUPERPOSITION_RESULTS`` or ``<root>/results``)."""
    env = os.environ.get(RESULTS_ENV)
    return Path(env) if env else project_root() / "results"


def git_info(root: Path | None = None) -> dict[str, Any]:
    """Current git commit and whether the working tree has uncommitted changes."""
    root = root or project_root()

    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", *args], cwd=root, capture_output=True, text=True, check=True, timeout=10
            )
        except (OSError, subprocess.SubprocessError):
            return None
        return out.stdout.strip()

    commit = git("rev-parse", "HEAD")
    status = git("status", "--porcelain", "--untracked-files=no")
    return {"commit": commit, "dirty": None if status is None else bool(status)}


def package_versions() -> dict[str, str | None]:
    """Versions of Python and the packages that determine numerical results."""
    versions: dict[str, str | None] = {"python": platform.python_version()}
    for name in _TRACKED_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _now() -> datetime:
    return datetime.now(UTC)


@dataclass
class RunDir:
    """Paths and IO helpers for one run directory."""

    path: Path

    @property
    def run_id(self) -> str:
        return self.path.name

    @property
    def experiment(self) -> str:
        return self.path.parent.name

    @property
    def config_path(self) -> Path:
        return self.path / "config.yaml"

    @property
    def manifest_path(self) -> Path:
        return self.path / "manifest.json"

    @property
    def acceptance_path(self) -> Path:
        return self.path / "acceptance.json"

    @property
    def log_path(self) -> Path:
        return self.path / "log.txt"

    @property
    def checkpoints_dir(self) -> Path:
        return self.path / "checkpoints"

    @property
    def figures_dir(self) -> Path:
        return self.path / "figures"

    # -- manifest ---------------------------------------------------------------------------
    def read_manifest(self) -> dict[str, Any]:
        return json.loads(self.manifest_path.read_text())

    def update_manifest(self, **fields: Any) -> None:
        manifest = self.read_manifest() if self.manifest_path.exists() else {}
        manifest.update(fields)
        self.manifest_path.write_text(json.dumps(manifest, indent=2, default=str) + "\n")

    @property
    def status(self) -> str | None:
        if not self.manifest_path.exists():
            return None
        return self.read_manifest().get("status")

    # -- tables -----------------------------------------------------------------------------
    def write_table(self, frame: pd.DataFrame, name: str = "metrics") -> Path:
        path = self.path / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        return path

    def read_table(self, name: str = "metrics") -> pd.DataFrame:
        return pd.read_parquet(self.path / f"{name}.parquet")

    # -- checkpoints ------------------------------------------------------------------------
    def save_checkpoint(self, name: str, state: dict[str, Any]) -> Path:
        self.checkpoints_dir.mkdir(exist_ok=True)
        path = self.checkpoints_dir / f"{name}.pt"
        cpu_state = {
            k: v.detach().cpu() if isinstance(v, torch.Tensor) else v for k, v in state.items()
        }
        torch.save(cpu_state, path)
        return path

    def load_checkpoint(self, name: str) -> dict[str, Any]:
        return torch.load(self.checkpoints_dir / f"{name}.pt", weights_only=False)

    # -- acceptance -------------------------------------------------------------------------
    def write_acceptance(self, acceptance: dict[str, Any]) -> Path:
        self.acceptance_path.write_text(json.dumps(acceptance, indent=2, default=str) + "\n")
        return self.acceptance_path

    def read_acceptance(self) -> dict[str, Any]:
        return json.loads(self.acceptance_path.read_text())


def create_run(experiment: str, config: Any, device: torch.device | str, seed: int) -> RunDir:
    """Create a fresh run directory, write ``config.yaml`` and a ``running`` manifest."""
    started = _now()
    base_id = f"{started:%Y%m%dT%H%M%SZ}-{config_hash(config)}"
    exp_dir = results_root() / experiment
    exp_dir.mkdir(parents=True, exist_ok=True)
    run_id, suffix = base_id, 1
    while (exp_dir / run_id).exists():
        suffix += 1
        run_id = f"{base_id}-{suffix}"
    run = RunDir(exp_dir / run_id)
    run.path.mkdir()
    run.figures_dir.mkdir()
    save_config(config, run.config_path)
    run.update_manifest(
        experiment=experiment,
        run_id=run_id,
        status="running",
        seed=seed,
        device=str(device),
        git=git_info(),
        packages=package_versions(),
        platform=platform.platform(),
        command=" ".join(sys.argv),
        started_utc=started.isoformat(),
    )
    return run


def finish_run(run: RunDir, status: str = "complete", error: str | None = None) -> None:
    """Record end time, wall-clock seconds and final status in the manifest."""
    ended = _now()
    started = datetime.fromisoformat(run.read_manifest()["started_utc"])
    fields: dict[str, Any] = {
        "status": status,
        "ended_utc": ended.isoformat(),
        "wall_clock_s": round((ended - started).total_seconds(), 3),
    }
    if error is not None:
        fields["error"] = error
    run.update_manifest(**fields)


def list_runs(experiment: str, status: str | None = "complete") -> list[RunDir]:
    """Runs of ``experiment``, oldest first (run IDs start with a UTC timestamp)."""
    exp_dir = results_root() / experiment
    if not exp_dir.is_dir():
        return []
    runs = [RunDir(p) for p in sorted(exp_dir.iterdir()) if (p / "manifest.json").is_file()]
    return [r for r in runs if status is None or r.status == status]


def latest_run(experiment: str) -> RunDir | None:
    """The most recent *complete* run of ``experiment``, or None."""
    runs = list_runs(experiment)
    return runs[-1] if runs else None


def resolve_run(experiment: str, run: str = "latest") -> RunDir:
    """Resolve ``latest`` or an explicit run ID to a :class:`RunDir`.

    Raises:
        FileNotFoundError: with the CLI command that would create the missing run.
    """
    if run == "latest":
        found = latest_run(experiment)
        if found is None:
            raise FileNotFoundError(
                f"no complete run of {experiment!r} under {results_root()}; create one with "
                f"`uv run superposition run {experiment} --profile quick`"
            )
        return found
    path = results_root() / experiment / run
    if not (path / "manifest.json").is_file():
        raise FileNotFoundError(f"run {run!r} of {experiment!r} not found at {path}")
    return RunDir(path)


def run_logger(run: RunDir, echo: bool = True) -> logging.Logger:
    """A logger writing to the run's ``log.txt`` (and to stderr when ``echo``)."""
    logger = logging.getLogger(f"{LOGGER_NAME}.{run.experiment}.{run.run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%dT%H:%M:%S")
    file_handler = logging.FileHandler(run.log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    if echo:
        stream = logging.StreamHandler()
        stream.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream)
    return logger


def close_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)
