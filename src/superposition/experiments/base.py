"""The experiment interface every module in ``experiments/`` implements.

SPEC §7: each experiment has a module, a config with ``quick`` and ``full`` profiles,
published figures and an ``acceptance.json`` per run. An :class:`Experiment` bundles the
experiment's identity (name, code, paper pages), its config dataclass, a ``run`` function that
trains and measures, and a ``figures`` function that redraws figures from a saved run.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from superposition.acceptance import Criterion
from superposition.config import ExperimentConfig
from superposition.runs import RunDir

RunFn = Callable[[Any, RunDir, torch.device, logging.Logger], list[Criterion]]
FiguresFn = Callable[[RunDir], list[Path]]


@dataclass(frozen=True)
class Experiment:
    """A registered experiment.

    Attributes:
        name: CLI name, e.g. ``intro_pentagon``; also the config file stem and results folder.
        code: short ID used in docs, e.g. ``E00``.
        title: one-line description.
        paper_pages: pages of the claims it checks, e.g. ``p. 2``.
        config_cls: the experiment's config dataclass (a subclass of ExperimentConfig).
        run: ``run(config, run_dir, device, logger) -> criteria``. It must save checkpoints and
            tables into ``run_dir`` so that ``figures`` can work from the saved run alone.
        figures: ``figures(run_dir) -> paths`` of saved figures (PNG and PDF).
    """

    name: str
    code: str
    title: str
    paper_pages: str
    config_cls: type[ExperimentConfig]
    run: RunFn
    figures: FiguresFn
