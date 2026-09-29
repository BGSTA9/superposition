"""Run an experiment end to end, redraw its figures, and publish them.

SPEC §5 (CLI: ``run``, ``figures --publish``) and §2 (Provenance). The CLI and the notebooks
both call these functions, so a notebook run produces the same run directory, manifest and
acceptance file as ``superposition run``.
"""

from __future__ import annotations

import shutil
from importlib import resources
from pathlib import Path
from typing import Any

from superposition.acceptance import acceptance_document
from superposition.config import load_config
from superposition.experiments.base import Experiment
from superposition.runs import (
    RunDir,
    close_logger,
    create_run,
    finish_run,
    project_root,
    resolve_run,
    run_logger,
)
from superposition.utils import resolve_device, seed_everything

# SPEC §5: only small published figures (≤ 500 KB each) go in docs/figures/.
MAX_PUBLISHED_BYTES = 500 * 1024


def get_experiment(name: str) -> Experiment:
    """Look up an experiment by name (``intro_pentagon``) or code (``E00``)."""
    from superposition.experiments import REGISTRY

    for exp in REGISTRY.values():
        if name in (exp.name, exp.code):
            return exp
    raise KeyError(f"unknown experiment {name!r}; registered: {sorted(REGISTRY)}")


def config_path(name: str) -> Path:
    """``configs/<name>.yaml`` in the project, or the copy packaged with the wheel."""
    local = project_root() / "configs" / f"{name}.yaml"
    if local.is_file():
        return local
    packaged = resources.files("superposition") / "configs" / f"{name}.yaml"
    if packaged.is_file():
        return Path(str(packaged))
    raise FileNotFoundError(f"no config file for {name!r} (looked in {local.parent})")


def load_experiment_config(
    name: str,
    profile: str = "quick",
    seed: int | None = None,
    device: str | None = None,
    overrides: dict[str, Any] | None = None,
) -> Any:
    """Resolve an experiment's config for a profile, with optional seed/device overrides."""
    exp = get_experiment(name)
    merged: dict[str, Any] = dict(overrides or {})
    if seed is not None:
        merged["seed"] = seed
    if device is not None:
        merged["device"] = device
    return load_config(exp.config_cls, config_path(exp.name), profile, merged)


def run_experiment(
    name: str,
    profile: str = "quick",
    seed: int | None = None,
    device: str | None = None,
    overrides: dict[str, Any] | None = None,
    echo: bool = True,
    figures: bool = True,
) -> RunDir:
    """Run an experiment and write a complete run directory (SPEC §5 run artifacts).

    Returns the :class:`RunDir`. On an exception the manifest is marked ``failed`` and the
    exception propagates.
    """
    exp = get_experiment(name)
    config = load_experiment_config(exp.name, profile, seed, device, overrides)
    torch_device = resolve_device(config.device)
    seed_everything(config.seed)
    run = create_run(exp.name, config, torch_device, config.seed)
    logger = run_logger(run, echo=echo)
    logger.info(
        f"{exp.code} {exp.name} · profile={profile} · device={torch_device} · run={run.run_id}"
    )
    try:
        criteria = exp.run(config, run, torch_device, logger)
        document = acceptance_document(
            experiment=exp.name,
            code=exp.code,
            title=exp.title,
            paper_pages=exp.paper_pages,
            run_id=run.run_id,
            profile=profile,
            criteria=criteria,
        )
        run.write_acceptance(document)
        for c in criteria:
            logger.info(f"[{c.status:>4}] {c.group + ' · ' if c.group else ''}{c.id}: {c.value}")
        logger.info(
            f"overall: {document['overall']} ({document['n_passed']}/{document['n_graded']})"
        )
        if figures:
            for path in exp.figures(run):
                logger.info(f"figure: {path.relative_to(run.path)}")
        finish_run(run, "complete")
    except BaseException as error:
        finish_run(run, "failed", repr(error))
        raise
    finally:
        close_logger(logger)
    return run


def make_figures(name: str, run: str = "latest", publish: bool = False) -> list[Path]:
    """Redraw figures for a saved run; with ``publish`` copy PNGs to docs/figures/<exp>/."""
    exp = get_experiment(name)
    run_dir = resolve_run(exp.name, run)
    paths = exp.figures(run_dir)
    if publish:
        return publish_figures(run_dir, paths)
    return paths


def publish_figures(run: RunDir, paths: list[Path]) -> list[Path]:
    """Copy a run's PNG figures to ``docs/figures/<experiment>/`` (≤ 500 KB each).

    A ``SOURCE.txt`` next to them records the run ID and git commit they came from.
    """
    target = project_root() / "docs" / "figures" / run.experiment
    target.mkdir(parents=True, exist_ok=True)
    published = []
    for path in paths:
        if path.suffix != ".png":
            continue
        size = path.stat().st_size
        if size > MAX_PUBLISHED_BYTES:
            raise ValueError(
                f"{path.name} is {size / 1024:.0f} KB; published figures must be ≤ 500 KB"
            )
        destination = target / path.name
        shutil.copy2(path, destination)
        published.append(destination)
    manifest = run.read_manifest()
    git = manifest.get("git", {})
    (target / "SOURCE.txt").write_text(
        f"experiment: {run.experiment}\nrun_id: {run.run_id}\n"
        f"profile: {_profile_of(run)}\ngit_commit: {git.get('commit')}\n"
        f"git_dirty: {git.get('dirty')}\nfigures: {', '.join(p.name for p in published)}\n"
    )
    return published


def _profile_of(run: RunDir) -> str:
    import yaml

    return str(yaml.safe_load(run.config_path.read_text()).get("profile"))
