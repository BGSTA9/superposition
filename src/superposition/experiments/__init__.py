"""One module per experiment (E00–E12), plus a registry.

SPEC §7. Each module defines an ``EXPERIMENT`` (see :mod:`.base`); the registry maps CLI names
to them. Run one with ``superposition run <name>`` or
:func:`superposition.experiments.runner.run_experiment`.
"""

from superposition.experiments import (
    abs_computation,
    adversarial,
    basic_results,
    intro_pentagon,
    l1_removes_superposition,
    learning_dynamics,
    mech_interp_abs,
    privileged_basis,
    sae_recovery,
)
from superposition.experiments.base import Experiment

REGISTRY: dict[str, Experiment] = {
    exp.name: exp
    for exp in (
        intro_pentagon.EXPERIMENT,
        basic_results.EXPERIMENT,
        learning_dynamics.EXPERIMENT,
        adversarial.EXPERIMENT,
        privileged_basis.EXPERIMENT,
        abs_computation.EXPERIMENT,
        mech_interp_abs.EXPERIMENT,
        sae_recovery.EXPERIMENT,
        l1_removes_superposition.EXPERIMENT,
    )
}

__all__ = ["REGISTRY", "Experiment"]
