"""Vectorized multi-instance trainer, evaluation, and best-restart selection.

Paper: Demonstrating Superposition, The Loss, p. 10: L = Σₓ Σᵢ Iᵢ (xᵢ − x′ᵢ)².
Optimization notes: p. 23 (2D hidden spaces are hard; fit several times, keep the lowest
loss), p. 14 (phase diagrams average restarts), p. 32 (1,000 restarts per setting).

The loss is computed in three steps (SPEC §3):

1. sum the importance-weighted squared error over features;
2. take the mean over the batch, giving one loss per instance;
3. sum over instances for ``backward()``.

Summing (not averaging) over instances keeps each instance's gradient exactly what it would
be if it were trained alone: an instance's parameters appear only in its own loss term, and
Adam's update is elementwise, so the instances never interact.

The paper does not state its optimizer settings. The defaults here (Adam, lr 1e-3, batch 1024,
10,000 steps, constant schedule, fresh batch every step) are the ones verified for E00; see
docs/DECISIONS.md.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from jaxtyping import Float, Int
from torch import Tensor, nn
from tqdm.auto import tqdm

from superposition.config import Schedule, TrainConfig
from superposition.data import FeatureDistribution
from superposition.utils import DATA_STREAM, EVAL_STREAM, derive_seed, make_generator

# Cap on elements per evaluation chunk (batch · inst · feat), to bound memory.
_EVAL_CHUNK_ELEMENTS = 1 << 24


@dataclass
class TrainResult:
    """Everything a training run returns (SPEC §6).

    Attributes:
        log_steps: training steps at which the loss was logged.
        loss_history: per-instance training-batch loss at each logged step, ``(log, inst)``.
        final_loss: per-instance loss on a fresh evaluation batch, ``(inst,)``.
        snapshot_steps: number of optimizer updates completed at each snapshot.
        snapshots: parameter snapshots (CPU tensors), one dict per snapshot step.
        steps: optimizer steps taken.
        eval_samples: samples used for ``final_loss``.
        wall_time_s: training plus evaluation wall-clock time in seconds.
    """

    log_steps: list[int]
    loss_history: Float[Tensor, "log inst"]
    final_loss: Float[Tensor, " inst"]
    steps: int
    eval_samples: int
    wall_time_s: float
    snapshot_steps: list[int] = field(default_factory=list)
    snapshots: list[dict[str, Tensor]] = field(default_factory=list)


def importance_weighted_loss(
    output: Float[Tensor, "batch inst feat"],
    target: Float[Tensor, "batch inst feat"],
    importance: Float[Tensor, "*inst feat"],
) -> Float[Tensor, " inst"]:
    """Σᵢ Iᵢ (targetᵢ − outputᵢ)², summed over features and averaged over the batch.

    Paper: The Loss, p. 10; footnote 1 (importance multiplies each squared error).
    Returns one value per instance (steps 1 and 2 of the SPEC §3 recipe).
    """
    return (importance * (target - output) ** 2).sum(dim=-1).mean(dim=0)


def lr_multiplier(schedule: Schedule, steps: int, warmup_frac: float) -> Callable[[int], float]:
    """Learning-rate multiplier as a function of the step (for ``LambdaLR``).

    ``constant``: 1. ``cosine``: ½(1 + cos(π t/T)). ``warmup_cosine``: linear warm-up over the
    first ``warmup_frac`` of training, then cosine decay to 0.
    """
    if schedule == "constant":
        return lambda step: 1.0
    if schedule == "cosine":
        return lambda step: 0.5 * (1.0 + math.cos(math.pi * min(step, steps) / steps))
    if schedule == "warmup_cosine":
        warmup = max(1, round(warmup_frac * steps))

        def multiplier(step: int) -> float:
            if step < warmup:
                return (step + 1) / warmup
            progress = (step - warmup) / max(1, steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

        return multiplier
    raise ValueError(f"unknown schedule {schedule!r}")


def _snapshot(model: nn.Module) -> dict[str, Tensor]:
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters()}


@torch.no_grad()
def evaluate_loss(
    model: nn.Module,
    dist: FeatureDistribution,
    importance: Float[Tensor, "*inst feat"],
    n_samples: int,
    generator: torch.Generator,
) -> Float[Tensor, " inst"]:
    """Per-instance expected loss, estimated on ``n_samples`` fresh samples.

    Samples are drawn in chunks to bound memory. The value is the per-sample
    importance-weighted squared error summed over features, averaged over samples, i.e. the
    same quantity the trainer minimizes (without any L1 penalty).
    """
    target_fn: Callable[[Tensor], Tensor] = model.target  # type: ignore[assignment]
    importance = importance.to(generator.device)
    chunk = max(1, min(n_samples, _EVAL_CHUNK_ELEMENTS // (dist.n_inst * dist.n_features)))
    total = torch.zeros(dist.n_inst, device=generator.device)
    done = 0
    while done < n_samples:
        size = min(chunk, n_samples - done)
        x = dist.sample(size, generator)
        total += importance_weighted_loss(model(x), target_fn(x), importance) * size
        done += size
    return (total / n_samples).cpu()


def train(
    model: nn.Module,
    dist: FeatureDistribution,
    importance: Float[Tensor, "*inst feat"],
    steps: int = 10_000,
    batch_size: int = 1024,
    lr: float = 1e-3,
    schedule: Schedule = "constant",
    l1_hidden: float | Float[Tensor, " inst"] = 0.0,
    snapshot_every: int | None = None,
    log_every: int = 100,
    *,
    eval_samples: int = 20_000,
    warmup_frac: float = 0.05,
    seed: int = 0,
    progress: bool = False,
) -> TrainResult:
    """Train every instance of ``model`` on ``dist`` with Adam; return a :class:`TrainResult`.

    Paper: The Loss, p. 10. Approach 1 (p. 42) adds λ‖h‖₁ to the loss; ``l1_hidden`` is λ
    (a float, or one value per instance for sweeps).

    Args:
        model: a toy model with a leading instance dimension (``TiedModel`` or ``AbsModel``).
            It stays on its current device; data is generated on the same device.
        dist: the feature distribution; ``dist.n_inst`` must equal ``model.n_inst``.
        importance: Iᵢ, shape ``(feat,)`` or ``(inst, feat)``.
        steps, batch_size, lr, schedule: optimizer settings. A fresh batch every step.
        l1_hidden: L1 penalty coefficient on the hidden activations.
        snapshot_every: if set, snapshot all parameters at init and every this many steps.
        log_every: record the per-instance training loss every this many steps.
        eval_samples: fresh samples for the final per-instance loss (SPEC: ≥ 20,000).
        warmup_frac: warm-up fraction for ``warmup_cosine``.
        seed: derives the data stream and the evaluation stream (initialization is seeded
            when the model is built).
        progress: show a tqdm progress bar.
    """
    if dist.n_inst != model.n_inst:  # type: ignore[attr-defined]
        raise ValueError(f"dist.n_inst={dist.n_inst} != model.n_inst={model.n_inst}")  # type: ignore[attr-defined]
    device = next(model.parameters()).device
    importance = importance.to(device)
    l1 = torch.as_tensor(l1_hidden, dtype=torch.float32, device=device)
    use_l1 = bool((l1 != 0).any())
    data_gen = make_generator(derive_seed(seed, DATA_STREAM), device)
    eval_gen = make_generator(derive_seed(seed, EVAL_STREAM), device)
    forward: Callable[[Tensor], tuple[Tensor, Tensor]] = model.forward_with_hidden  # type: ignore[assignment]
    target_fn: Callable[[Tensor], Tensor] = model.target  # type: ignore[assignment]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_multiplier(schedule, steps, warmup_frac)
    )

    log_steps: list[int] = []
    logged: list[Tensor] = []
    snapshot_steps: list[int] = []
    snapshots: list[dict[str, Tensor]] = []
    if snapshot_every:
        snapshot_steps.append(0)
        snapshots.append(_snapshot(model))

    start = time.perf_counter()
    for step in tqdm(range(steps), disable=not progress, mininterval=1.0, leave=False):
        x = dist.sample(batch_size, data_gen)
        out, h = forward(x)
        per_instance = importance_weighted_loss(out, target_fn(x), importance)
        objective = per_instance
        if use_l1:
            objective = objective + l1 * h.abs().sum(dim=-1).mean(dim=0)
        optimizer.zero_grad(set_to_none=True)
        objective.sum().backward()  # step 3: sum over instances
        optimizer.step()
        scheduler.step()
        if step % log_every == 0 or step == steps - 1:
            log_steps.append(step)
            logged.append(per_instance.detach())
        if snapshot_every and ((step + 1) % snapshot_every == 0 or step == steps - 1):
            snapshot_steps.append(step + 1)
            snapshots.append(_snapshot(model))

    final_loss = evaluate_loss(model, dist, importance, eval_samples, eval_gen)
    history = torch.stack(logged).cpu() if logged else torch.empty(0, dist.n_inst)
    return TrainResult(
        log_steps=log_steps,
        loss_history=history,
        final_loss=final_loss,
        steps=steps,
        eval_samples=eval_samples,
        wall_time_s=time.perf_counter() - start,
        snapshot_steps=snapshot_steps,
        snapshots=snapshots,
    )


def train_from_config(
    model: nn.Module,
    dist: FeatureDistribution,
    importance: Float[Tensor, "*inst feat"],
    config: TrainConfig,
    seed: int,
    progress: bool = False,
) -> TrainResult:
    """:func:`train` with settings taken from a :class:`~superposition.config.TrainConfig`."""
    return train(
        model,
        dist,
        importance,
        steps=config.steps,
        batch_size=config.batch_size,
        lr=config.lr,
        schedule=config.schedule,
        l1_hidden=config.l1_hidden,
        snapshot_every=config.snapshot_every,
        log_every=config.log_every,
        eval_samples=config.eval_samples,
        warmup_frac=config.warmup_frac,
        seed=seed,
        progress=progress,
    )


def select_best(
    result: TrainResult | Float[Tensor, " inst"], groups: Sequence[int] | Int[Tensor, " inst"]
) -> Int[Tensor, " group"]:
    """Index of the lowest-loss instance in each group (experimental condition).

    Paper: p. 23 ("fitting each model multiple times and taking the solution with the lowest
    loss"). ``groups[k]`` is instance k's condition label; the result lists one instance index
    per distinct label, in increasing label order. NaN losses are never selected unless a
    whole group is NaN.
    """
    loss = result.final_loss if isinstance(result, TrainResult) else result
    loss_np = loss.detach().cpu().numpy().astype(np.float64)
    groups_np = np.asarray(groups.cpu() if isinstance(groups, Tensor) else groups)
    if loss_np.shape != groups_np.shape:
        raise ValueError(f"loss shape {loss_np.shape} != groups shape {groups_np.shape}")
    order = np.lexsort((loss_np, groups_np))  # by group, then by loss (NaN sorts last)
    _, first = np.unique(groups_np[order], return_index=True)
    return torch.as_tensor(order[first], dtype=torch.long)
