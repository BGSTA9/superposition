"""Analytic L2 adversarial attacks and adversarial vulnerability.

Paper: Relationship to Adversarial Robustness, pp. 29–30.

Why superposition invites attacks (p. 29): without superposition, the end-to-end weights of a
feature are (WᵀW)₀ = (1, 0, 0, …), so moving the input changes output 0 only through x₀. With
superposition they look like (1, ε, −ε, ε, …): every small interference term is a direction an
adversary can push along to raise output 0 while barely changing the input.

The attack (p. 29):

* **Budget.** An L2 perturbation δ with ‖δ‖₂ ≤ ε, where ε is 0.1 of the average input norm,
  ε = 0.1 · E‖x‖₂ (zero inputs included).
* **Candidates.** For each feature i, the optimal L2 attack on that feature's pre-activation
  (WᵀWx + b)ᵢ is the direction of row i of WᵀW: δᵢ = ε (WᵀW)ᵢ / ‖(WᵀW)ᵢ‖₂. It raises output i's
  pre-activation by ε‖(WᵀW)ᵢ‖₂, the most any budget-ε perturbation can. The paper derived
  these analytically because gradient attacks suffer from gradient masking when the output
  ReLUs are almost always off (sparse data).
* **Choice.** "taking the one of these attacks which most harms model performance": for every
  input we evaluate all n candidates and keep the one with the highest loss (a per-example
  worst case; docs/DECISIONS.md D-405).
* **Loss.** The attacked loss compares the model's output on x + δ with the *clean* target:
  the adversary perturbs the input, the right answer stays x.

**Vulnerability** (the p. 29 figure's y-axis) is the attack's damage relative to a model
without superposition, (L′ − L)/(L′₀ − L₀), where L and L′ are the clean and attacked losses.
Our reference (subscript 0) is the ideal no-superposition model at the same sparsity and the
same budget: the top-m features on orthonormal dedicated dimensions (W = [Iₘ | 0], b = 0).
For it every candidate raises exactly one output by ε, so L′₀ − L₀ = max_{i<m} Iᵢ · ε².
:func:`no_superposition_model` builds it, so both terms go through the same code.
"""

from __future__ import annotations

from dataclasses import dataclass

import einops
import torch
from jaxtyping import Float, Int
from torch import Tensor

from superposition.models import TiedModel

# Norms below this are treated as zero when normalizing attack directions.
_ZERO_NORM = 1e-12
# Cap on elements per chunk of attacked inputs (examples · candidates · inst · feat).
_CHUNK_ELEMENTS = 1 << 24


def attack_budget(
    x: Float[Tensor, "batch inst feat"], fraction: float = 0.1
) -> Float[Tensor, " inst"]:
    """ε = ``fraction`` · E‖x‖₂, the mean L2 input norm over the batch (Paper: p. 29).

    Zero inputs count toward the mean, so sparser data gets a smaller budget.
    """
    return fraction * torch.linalg.vector_norm(x, dim=-1).mean(dim=0)


def attack_directions(W: Float[Tensor, "inst hidden feat"]) -> Float[Tensor, "inst attack feat"]:
    """Unit attack directions (WᵀW)ᵢ / ‖(WᵀW)ᵢ‖₂, one per feature i (Paper: p. 29).

    Row i of the result is the input direction that raises feature i's pre-activation
    (WᵀWx + b)ᵢ the most per unit of L2 norm. A feature whose row of WᵀW is zero (not
    represented at all) gets a zero direction, i.e. no attack.
    """
    gram = einops.einsum(W, W, "inst hidden f1, inst hidden f2 -> inst f1 f2")
    norms = torch.linalg.vector_norm(gram, dim=-1, keepdim=True)
    return torch.where(norms > _ZERO_NORM, gram / norms.clamp_min(_ZERO_NORM), 0.0)


@dataclass
class AttackResult:
    """Clean and worst-case attacked losses of every instance.

    Attributes:
        clean_loss: L = E Σᵢ Iᵢ (targetᵢ − x′ᵢ(x))², ``(inst,)``.
        attacked_loss: L′ = E maxₐ Σᵢ Iᵢ (targetᵢ − x′ᵢ(x + δₐ))², ``(inst,)``.
        epsilon: the budget ε used, ``(inst,)``.
        chosen: how often each candidate (by attacked feature) was the worst case,
            ``(inst, attack)``.
        n_samples: examples averaged over.
    """

    clean_loss: Float[Tensor, " inst"]
    attacked_loss: Float[Tensor, " inst"]
    epsilon: Float[Tensor, " inst"]
    chosen: Int[Tensor, "inst attack"]
    n_samples: int

    @property
    def damage(self) -> Float[Tensor, " inst"]:
        """L′ − L, the loss increase caused by the attack."""
        return self.attacked_loss - self.clean_loss


@torch.no_grad()
def analytic_attack(
    model: TiedModel,
    x: Float[Tensor, "batch inst feat"],
    importance: Float[Tensor, "*inst feat"],
    epsilon: Float[Tensor, " inst"],
) -> AttackResult:
    """Evaluate the per-feature analytic L2 attacks on ``x`` and keep each input's worst one.

    Paper: Relationship to Adversarial Robustness, p. 29. For each input x and each candidate
    δₐ = ε (WᵀW)ₐ/‖(WᵀW)ₐ‖₂, the loss of the output on x + δₐ against the clean target is
    computed; the attacked loss of x is the maximum over candidates. The budget is a maximum
    ("a max L2 attack norm"), so δ = 0 is allowed too: the attacked loss is never below the
    clean loss.

    Args:
        model: a tied model (linear, ReLU output or ReLU hidden layer); the attack directions
            come from its WᵀW.
        x: inputs, ``(batch, inst, feat)``, on the model's device.
        importance: Iᵢ, ``(feat,)`` or ``(inst, feat)``.
        epsilon: the L2 budget per instance.
    """
    batch, n_inst, n_feat = x.shape
    importance = importance.to(x.device)
    directions = attack_directions(model.W)  # (inst, attack, feat)
    delta = epsilon.to(x.device)[:, None, None] * directions
    target = model.target(x)
    clean = (importance * (target - model(x)) ** 2).sum(dim=-1)  # (batch, inst)

    n_attack = delta.shape[1]
    per_example = max(1, _CHUNK_ELEMENTS // (n_attack * n_inst * n_feat))
    worst = torch.empty(batch, n_inst, device=x.device)
    argworst = torch.empty(batch, n_inst, dtype=torch.long, device=x.device)
    for start in range(0, batch, per_example):
        xs = x[start : start + per_example]  # (b, inst, feat)
        b = xs.shape[0]
        attacked = xs[:, None] + delta.transpose(0, 1)[None]  # (b, attack, inst, feat)
        out = model(attacked.reshape(b * n_attack, n_inst, n_feat))
        out = out.reshape(b, n_attack, n_inst, n_feat)
        loss = (importance * (target[start : start + b, None] - out) ** 2).sum(dim=-1)
        worst[start : start + b], argworst[start : start + b] = loss.max(dim=1)
    attacked_loss = torch.maximum(worst, clean)
    chosen = torch.zeros(n_inst, n_attack, dtype=torch.long, device=x.device)
    chosen.scatter_add_(1, argworst.T, torch.ones_like(argworst.T))
    return AttackResult(
        clean_loss=clean.mean(dim=0).cpu(),
        attacked_loss=attacked_loss.mean(dim=0).cpu(),
        epsilon=epsilon.cpu(),
        chosen=chosen.cpu(),
        n_samples=batch,
    )


def no_superposition_model(
    n_features: int, n_hidden: int, n_inst: int, kind: str = "relu_output"
) -> TiedModel:
    """The reference model for vulnerability: top-m features on dedicated dimensions.

    W = [Iₘ | 0] (feature i < m on hidden dimension i, the rest unrepresented) and b = 0, the
    same in every instance. It has no superposition, and with features ordered most important
    first it is the solution the linear model learns (p. 13). Under the analytic attack every
    candidate raises one output by exactly ε, so its damage is max_{i<m} Iᵢ · ε².
    """
    model = TiedModel.from_kind(kind, n_features, n_hidden, n_inst)  # type: ignore[arg-type]
    W = torch.zeros(n_inst, n_hidden, n_features)
    k = min(n_hidden, n_features)
    W[:, torch.arange(k), torch.arange(k)] = 1.0
    with torch.no_grad():
        model.W.copy_(W)
        model.b.zero_()
    return model


def vulnerability(attacked: AttackResult, reference: AttackResult) -> Float[Tensor, " inst"]:
    """(L′ − L) / (L′₀ − L₀): attack damage relative to the no-superposition reference.

    Paper: p. 29 figure ("Vulnerability to Adversarial Examples (Relative to Non-Superposition
    Model)"). Both results must use the same inputs' distribution and the same budget ε.
    """
    return attacked.damage / reference.damage
