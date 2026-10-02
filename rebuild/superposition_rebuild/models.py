"""Toy models with a leading instance dimension (linear, ReLU output, ReLU hidden, abs).

Paper: Demonstrating Superposition, The Model, p. 10 (linear and ReLU output models);
Superposition in a Privileged Basis, p. 31 (ReLU hidden layer model); Computation in
Superposition, Experiment Setup, p. 35 (absolute-value model).

| Model            | Hidden layer         | Output                        |
|------------------|----------------------|-------------------------------|
| linear           | h = Wx               | x′ = Wᵀh + b = WᵀWx + b       |
| ReLU output      | h = Wx               | x′ = ReLU(WᵀWx + b)           |
| ReLU hidden      | h = ReLU(Wx), no bias| x′ = ReLU(Wᵀh + b)            |
| absolute value   | h = ReLU(W₁x)        | y′ = ReLU(W₂h + b), target |x| |

Every parameter carries a leading instance dimension, so ``n_inst`` independent models train
in one vectorized step. Shapes: ``W: (inst, hidden, feat)``, ``b: (inst, feat)``; column
``W[k, :, i]`` is feature i's direction Wᵢ in instance k.

Sign convention: we use ``+b`` everywhere. The phase-change section writes ReLU(WᵀWx − b)
(p. 14), so its b is our −b.
"""

# Rebuild kit, milestone M2 (rebuild/README.md). Signatures and docstrings are
# copied from src/superposition/models.py; every body raises NotImplementedError. Implement
# them until `uv run pytest -c rebuild/pytest.ini rebuild/tests/test_models.py` passes.

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
import torch.nn.functional as F
from jaxtyping import Float
from torch import Tensor, nn

Activation = Callable[[Tensor], Tensor]
TiedKind = Literal["linear", "relu_output", "relu_hidden"]


def _identity(x: Tensor) -> Tensor:
    raise NotImplementedError


# Pluggable activations (SPEC §6), so extensions such as E12 can swap them.
ACTIVATIONS: dict[str, Activation] = {
    "identity": _identity,
    "relu": F.relu,
    "gelu": F.gelu,
    "silu": F.silu,
    "softplus": F.softplus,
    "leaky_relu": F.leaky_relu,
    "sigmoid": torch.sigmoid,
}


def get_activation(name: str | None) -> Activation:
    """Look up an activation by name; ``None`` means identity."""
    raise NotImplementedError


def register_activation(name: str, fn: Activation) -> None:
    """Add an activation to the registry (for extensions)."""
    raise NotImplementedError


def xavier_normal_per_instance(
    n_inst: int, fan_out: int, fan_in: int, seed: int, param_key: int = 0
) -> Float[Tensor, "inst out in"]:
    """Xavier-normal weights, drawn independently for each instance.

    SPEC §3 (Initialization): std = √(2/(fan_in + fan_out)) = √(2/(n + m)), and instance k's
    weights depend only on ``(seed, param_key, k)``, so adding instances never changes
    existing ones.

    Why not ``nn.init.xavier_normal_``? On a 3-D ``(inst, m, n)`` tensor PyTorch computes
    fan-out as ``inst * m`` and the scale would silently change with the number of instances.

    Args:
        n_inst: number of instances.
        fan_out, fan_in: the matrix shape of one instance, ``(fan_out, fan_in)``.
        seed: the global seed.
        param_key: distinguishes parameters of one model (e.g. W₁ = 0, W₂ = 1).
    """
    raise NotImplementedError


class TiedModel(nn.Module):
    """The tied-weight toy model: linear, ReLU output, or ReLU hidden layer.

    Paper: The Model, p. 10; Superposition in a Privileged Basis, p. 31.

    The decoder is Wᵀ ("to recover the original vector, we'll use the transpose of the same
    matrix", p. 10). There is never a hidden bias: with one, the ReLU hidden model shifts its
    neurons into the linear regime and sidesteps the ReLU (p. 34).

    Args:
        n_features: n, the number of features.
        n_hidden: m, the number of hidden dimensions.
        n_inst: number of independent instances trained together.
        hidden_act: activation on h (``None`` for the linear and ReLU output models).
        out_act: activation on the output (``None`` for the linear model, ``"relu"`` otherwise).
        bias: learn the output bias b. If False, b is a fixed zero buffer.
        seed: global seed; instance k's init depends only on (seed, k).
    """

    def __init__(
        self,
        n_features: int,
        n_hidden: int,
        n_inst: int,
        hidden_act: str | None = None,
        out_act: str | None = None,
        bias: bool = True,
        seed: int = 0,
    ) -> None:
        raise NotImplementedError

    @classmethod
    def from_kind(
        cls, kind: TiedKind, n_features: int, n_hidden: int, n_inst: int, seed: int = 0
    ) -> TiedModel:
        """Build one of the three paper models by name (SPEC §3 table)."""
        raise NotImplementedError

    def hidden(self, x: Float[Tensor, "batch inst feat"]) -> Float[Tensor, "batch inst hidden"]:
        """h = act(Wx)."""
        raise NotImplementedError

    def readout(self, h: Float[Tensor, "batch inst hidden"]) -> Float[Tensor, "batch inst feat"]:
        """The pre-activation output Wᵀh + b (the probe panel shows it before the ReLU)."""
        raise NotImplementedError

    def forward_with_hidden(
        self, x: Float[Tensor, "batch inst feat"]
    ) -> tuple[Float[Tensor, "batch inst feat"], Float[Tensor, "batch inst hidden"]]:
        """Return ``(x′, h)``; the trainer needs h for the optional L1 penalty (p. 42)."""
        raise NotImplementedError

    def forward(self, x: Float[Tensor, "batch inst feat"]) -> Float[Tensor, "batch inst feat"]:
        """x′ = out_act(Wᵀ hidden_act(Wx) + b)."""
        raise NotImplementedError

    def target(self, x: Float[Tensor, "batch inst feat"]) -> Float[Tensor, "batch inst feat"]:
        """The reconstruction target: the input itself (p. 10)."""
        raise NotImplementedError

    def gram(self) -> Float[Tensor, "inst feat feat"]:
        """WᵀW, the feature-by-feature matrix visualized on p. 10."""
        raise NotImplementedError


class AbsModel(nn.Module):
    """The absolute-value model with untied weights: h = ReLU(W₁x), y′ = ReLU(W₂h + b).

    Paper: Computation in Superposition, Experiment Setup, p. 35. The target is y = |x| for
    inputs whose nonzero values are uniform on [−1, 1]. ``abs(x) = ReLU(x) + ReLU(−x)`` is what
    makes the hidden ReLU necessary, so this model cannot sidestep it.

    Shapes: ``W1: (inst, hidden, feat)``, ``W2: (inst, feat, hidden)``, ``b: (inst, feat)``.
    Both weights use per-instance Xavier-normal init with std √(2/(n + m)).
    """

    def __init__(self, n_features: int, n_hidden: int, n_inst: int, seed: int = 0) -> None:
        raise NotImplementedError

    def hidden(self, x: Float[Tensor, "batch inst feat"]) -> Float[Tensor, "batch inst hidden"]:
        """h = ReLU(W₁x)."""
        raise NotImplementedError

    def readout(self, h: Float[Tensor, "batch inst hidden"]) -> Float[Tensor, "batch inst feat"]:
        """The pre-activation output W₂h + b."""
        raise NotImplementedError

    def forward_with_hidden(
        self, x: Float[Tensor, "batch inst feat"]
    ) -> tuple[Float[Tensor, "batch inst feat"], Float[Tensor, "batch inst hidden"]]:
        """Return ``(y′, h)``."""
        raise NotImplementedError

    def forward(self, x: Float[Tensor, "batch inst feat"]) -> Float[Tensor, "batch inst feat"]:
        """y′ = ReLU(W₂ ReLU(W₁x) + b)."""
        raise NotImplementedError

    def target(self, x: Float[Tensor, "batch inst feat"]) -> Float[Tensor, "batch inst feat"]:
        """The target y = |x| (p. 35)."""
        raise NotImplementedError


ToyModel = TiedModel | AbsModel
