"""Seeding and device utilities shared by every module.

SPEC §3 (Initialization) and §5 (Compute, Code).

Two ideas live here:

* **Seed derivation.** One global ``seed`` fans out into independent random streams, keyed by
  small integers: the initialization of instance k, the training data, the evaluation data.
  ``derive_seed(seed, INIT_STREAM, param, k)`` depends only on its arguments, so adding
  instances or restarts never changes the streams of existing ones (SPEC §3).
* **Device resolution.** ``--device auto`` picks CUDA, then Apple MPS, then CPU (SPEC §5).
"""

from __future__ import annotations

import random

import numpy as np
import torch

# Stream keys for derive_seed(). Named constants, so every module agrees on them.
INIT_STREAM = 0
DATA_STREAM = 1
EVAL_STREAM = 2
PROBE_STREAM = 3

DEVICE_CHOICES = ("auto", "cpu", "mps", "cuda")


def seed_everything(seed: int) -> None:
    """Seed Python's, NumPy's and PyTorch's global generators.

    Library code never relies on global state (it derives explicit ``torch.Generator``s with
    :func:`derive_seed`); this is for scripts and notebooks that call unseeded functions.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 (seeding the legacy global RNG is the point here)
    torch.manual_seed(seed)


def derive_seed(seed: int, *keys: int) -> int:
    """Derive an independent 63-bit seed from a global seed and integer keys.

    Uses NumPy's ``SeedSequence`` with ``spawn_key=keys``, the standard way to build
    statistically independent child streams. The result depends only on ``(seed, *keys)``.

    Args:
        seed: The global, non-negative seed of a run.
        *keys: Non-negative integers naming the stream, e.g. ``(INIT_STREAM, 0, k)`` for
            parameter 0 of instance k.

    Returns:
        A non-negative integer below 2**63, valid for ``torch.Generator.manual_seed``.
    """
    if seed < 0 or any(k < 0 for k in keys):
        raise ValueError(f"seed and keys must be non-negative, got seed={seed}, keys={keys}")
    state = np.random.SeedSequence(entropy=seed, spawn_key=keys).generate_state(1, np.uint64)
    return int(state[0] >> np.uint64(1))


def make_generator(seed: int, device: torch.device | str = "cpu") -> torch.Generator:
    """Return a ``torch.Generator`` on ``device`` seeded with ``seed``."""
    return torch.Generator(device=torch.device(device)).manual_seed(seed)


def resolve_device(device: str | torch.device = "auto") -> torch.device:
    """Resolve a device string: ``auto`` picks CUDA, then MPS, then CPU (SPEC §5).

    Raises:
        ValueError: if an explicit accelerator is requested but not available.
    """
    if isinstance(device, torch.device):
        return device
    if device not in DEVICE_CHOICES:
        raise ValueError(f"device must be one of {DEVICE_CHOICES}, got {device!r}")
    if device == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("device='cuda' requested but CUDA is not available")
    if device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("device='mps' requested but MPS is not available")
    return torch.device(device)
