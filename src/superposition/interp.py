"""Neuron roles, ablations, motif detection and rebuild tests for the absolute-value model.

Paper: Computation in Superposition, pp. 34–39 (side neurons, p. 36; primary and secondary
features, p. 38; The Asymmetric Superposition Motif, pp. 38–39); footnote 15 (p. 56).

The model (p. 35) is h = ReLU(W₁x), y′ = ReLU(W₂h + b) with target |x| and inputs that are 0
or uniform on [−1, 1]. Its hidden ReLU is necessary (|x| = ReLU(x) + ReLU(−x)), so its neurons
form a privileged basis and each one can be given a *role*. This module does what p. 39 does
by eye, as code a test can check.

**Path gains.** When only feature i is active, xᵢ = ±v, neuron h fires only if the sign of
xᵢ matches the sign of W₁[h, i] (the *side* h reads), and then adds
v · |W₁[h, i]| · W₂[j, h] to output j's pre-activation. We call

    G_h[i, j] = |W₁[h, i]| · W₂[j, h]

the path gain from feature i (on the side h reads) to output j. It does not change under the
footnote-15 rescaling (W₁ row × α, W₂ column × α⁻¹), and it is exactly what the loss sees on
1-sparse inputs, which dominate when features are sparse (p. 13):

* G_h[i, i] > 0: h computes that side of |xᵢ| (1 = exactly, on its own);
* G_h[i, j] > 0 for j ≠ i: positive interference of i on j, which the output ReLU passes;
* G_h[i, j] < 0: i *inhibits* output j, which is nearly free while j is inactive (p. 13).

**Roles** (thresholds in :class:`RoleThresholds`; D-501). With C_h = {i : G_h[i, i] ≥
``self_gain``} the features h computes:

| role | rule |
|---|---|
| dead | every \\|G_h[i, j]\\| < ``dead_gain``: the neuron affects no output |
| inhibitor(j) | its strongest effect is a negative gain onto output j: −minᵢ G_h[i, j] ≥ ``self_gain`` and ≥ ``inhibition_dominance`` × its largest positive gain. Features it reads only to inhibit j are its *triggers* (the "small amount of" the heavy feature, p. 39); what the rest of the neuron does is its secondary role (``also``) |
| positive-side(i) / negative-side(i) | C_h = {i} and every other \\|G_h\\| entry < ``clean`` × G_h[i, i] (p. 36); the side is the sign of W₁[h, i] |
| asymmetric-superposition(i, j) | C_h = {i, j}, nothing else above ``clean`` × the block, and "sizes flipped" (p. 39): \\|W₁[h, i]\\| ≥ ``magnitude_ratio`` × \\|W₁[h, j]\\| while W₂[j, h] ≥ ``magnitude_ratio`` × W₂[i, h]. i is the *heavy* feature (it interferes with j), j the *victim* |
| superposition(i, j) | the same two-feature neuron without the flip: normal superposition, W₁ = [1, −1], W₂ = [1, 1] (p. 39) |
| unexplained | anything else |

SPEC §6 lists positive-side, negative-side, asymmetric-superposition, inhibitor and
unexplained; *superposition* (normal, named on p. 39) and *dead* are added (D-501).

**Motif** (pp. 38–39): neuron h stores features i and j with unequal input weights and
reciprocal output weights (the asymmetric condition above, checked on the weights directly),
so i interferes heavily with j but not the reverse; a second neuron h′ reads i on the same side
(a "small amount of" i) and writes strongly negative to j, cancelling at least ``cancel`` of the
positive interference: G_h′[i, j] ≤ −``cancel`` × G_h[i, j]. "This essentially converts positive
interference … into negative interference".

**Ablations**: the change in each feature's loss when a neuron is zeroed (h_k = 0) or
mean-ablated (h_k = E[h_k]). **Rebuild**: an idealized model built by hand from the roles; the
fraction of the trained model's loss reduction (relative to predicting the mean) it recovers.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from jaxtyping import Float
from torch import Tensor

from superposition import metrics
from superposition.data import IndependentSparse
from superposition.models import AbsModel
from superposition.theory import uniform_feature_variance

# Nonzero input values of the absolute-value task are U[−1, 1] (p. 35).
ABS_LOW, ABS_HIGH = -1.0, 1.0

# Role kinds. SPEC §6 names the first four plus "unexplained"; "superposition" (normal,
# symmetric superposition, p. 39) and "dead" are added (D-501).
POSITIVE_SIDE = "positive-side"
NEGATIVE_SIDE = "negative-side"
ASYMMETRIC = "asymmetric-superposition"
INHIBITOR = "inhibitor"
SUPERPOSITION = "superposition"
DEAD = "dead"
UNEXPLAINED = "unexplained"
SPEC_NAMED_KINDS = (POSITIVE_SIDE, NEGATIVE_SIDE, ASYMMETRIC, INHIBITOR)
NAMED_KINDS = (*SPEC_NAMED_KINDS, SUPERPOSITION)


@dataclass
class RoleThresholds:
    """Thresholds for roles, motifs and ablation agreement (D-501). Gains are per unit |xᵢ|."""

    dead_gain: float = 0.05  # every |G_h[i, j]| below this: the neuron is dead
    self_gain: float = 0.1  # G_h[i, i] ≥ this: h computes (a side of) feature i
    clean: float = 0.25  # other gains below this fraction of the main one are ignored
    magnitude_ratio: float = 2.0  # "sizes flipped" by at least this factor, in and out
    inhibition_dominance: float = 1.0  # inhibition ≥ this × the largest positive gain
    cancel: float = 0.5  # a motif's inhibitor cancels ≥ this share of the interference
    ablation_rise: float = 0.05  # a loss "rises" by ≥ this fraction of predicting the mean


@dataclass(frozen=True)
class NeuronRole:
    """The role of one neuron.

    Attributes:
        neuron: hidden-unit index.
        kind: one of the role kinds above.
        features: side: ``(i,)``; (asymmetric) superposition: ``(i, j)``, heavy feature first
            for the asymmetric kind; inhibitor: ``(j,)``, the inhibited feature.
        also_kind, also_features: for an inhibitor, the role of the rest of the neuron (e.g.
            negative-side of feature 3); ``unexplained`` with the features it computes if the
            rest fits no role; empty if the rest does nothing.
        triggers: for an inhibitor, the features it reads only to inhibit (it does not compute
            them): the "small amount of" the heavy feature on p. 39.
        computes: features whose side the neuron computes (G_h[i, i] ≥ ``self_gain``).
    """

    neuron: int
    kind: str
    features: tuple[int, ...] = ()
    also_kind: str = ""
    also_features: tuple[int, ...] = ()
    triggers: tuple[int, ...] = ()
    computes: tuple[int, ...] = ()

    @staticmethod
    def _name(kind: str, features: tuple[int, ...]) -> str:
        return f"{kind}({', '.join(map(str, features))})" if features else kind

    @property
    def label(self) -> str:
        base = self._name(self.kind, self.features)
        return (
            f"{base} + {self._name(self.also_kind, self.also_features)}" if self.also_kind else base
        )

    @property
    def named(self) -> bool:
        """A live neuron with a role other than unexplained."""
        return self.kind in NAMED_KINDS

    @property
    def fully_named(self) -> bool:
        """Named, with nothing unexplained left: an inhibitor whose other content fits no role
        does not count (docs/DECISIONS.md D-604; on random weights almost any neuron whose most
        negative gain dominates is an inhibitor)."""
        return self.named and not (self.kind == INHIBITOR and self.also_kind == UNEXPLAINED)

    def has_role(self, kind: str, features: tuple[int, ...]) -> bool:
        """Whether this neuron carries ``kind(features)`` as its role or its secondary role."""
        return (self.kind, self.features) == (kind, features) or (
            self.also_kind,
            self.also_features,
        ) == (kind, features)

    @property
    def predicted_rise(self) -> tuple[int, ...]:
        """Features whose loss must rise when this neuron is ablated: the ones it computes,
        plus the one it inhibits (its interference is exposed without it)."""
        extra = self.features if self.kind == INHIBITOR else ()
        return tuple(sorted(set(self.computes) | set(extra)))


@dataclass(frozen=True)
class Motif:
    """One instance of asymmetric superposition with inhibition (pp. 38–39).

    Attributes:
        asymmetric_neuron: h, storing ``heavy`` with a large and ``victim`` with a small weight.
        inhibitor_neuron: h′, reading ``heavy`` on the same side and inhibiting ``victim``.
        heavy, victim: feature indices.
        input_ratio: |W₁[h, heavy]| / |W₁[h, victim]|.
        output_ratio: W₂[victim, h] / W₂[heavy, h] (reciprocal weights make it ≈ input_ratio).
        interference: G_h[heavy, victim] > 0, the positive interference per unit |x_heavy|.
        inhibition: G_h′[heavy, victim] < 0.
        net_interference: output ``victim``'s pre-activation when only ``heavy`` is active at
            |x| = 1 on h's side, summed over every neuron, plus b_victim. Negative means the
            positive interference was converted into negative interference.
    """

    asymmetric_neuron: int
    inhibitor_neuron: int
    heavy: int
    victim: int
    input_ratio: float
    output_ratio: float
    interference: float
    inhibition: float
    net_interference: float


# --- gains and roles -----------------------------------------------------------------------------


def path_gains(
    W1: Float[Tensor, "hidden feat"], W2: Float[Tensor, "feat hidden"]
) -> Float[Tensor, "hidden feat feat"]:
    """G[h, i, j] = |W₁[h, i]| · W₂[j, h], the 1-sparse gain from feature i to output j via h."""
    return W1.abs()[:, :, None] * W2.T[:, None, :]


def _two_feature_role(
    h: int, G: Tensor, W1: Tensor, W2: Tensor, computed: list[int], thr: RoleThresholds
) -> tuple[str, tuple[int, ...]] | None:
    """Side or (asymmetric) superposition role from one neuron's gains, else None."""
    if len(computed) == 1:
        i = computed[0]
        others = G.abs().clone()
        others[i, i] = 0.0
        if float(others.max()) < thr.clean * float(G[i, i]):
            return (POSITIVE_SIDE if float(W1[h, i]) > 0 else NEGATIVE_SIDE), (i,)
        return None
    if len(computed) == 2:
        i, j = computed
        block = torch.zeros_like(G, dtype=torch.bool)
        block[np.ix_([i, j], [i, j])] = True
        outside = torch.where(block, torch.zeros_like(G), G.abs())
        if float(outside.max()) >= thr.clean * float(G[block].max()):
            return None
        heavy, victim = (i, j) if abs(float(W1[h, i])) >= abs(float(W1[h, j])) else (j, i)
        flipped_in = abs(float(W1[h, heavy])) >= thr.magnitude_ratio * abs(float(W1[h, victim]))
        flipped_out = float(W2[victim, h]) >= thr.magnitude_ratio * float(W2[heavy, h])
        if flipped_in and flipped_out:
            return ASYMMETRIC, (heavy, victim)
        return SUPERPOSITION, (i, j)
    return None


def neuron_roles(
    W1: Float[Tensor, "hidden feat"],
    W2: Float[Tensor, "feat hidden"],
    thr: RoleThresholds | None = None,
) -> list[NeuronRole]:
    """Assign every neuron of one absolute-value model a role (rules in the module docstring).

    Works on raw or footnote-15-normalized weights alike, since path gains are invariant to
    the rescaling.
    """
    thr = thr or RoleThresholds()
    gains = path_gains(W1, W2)
    roles = []
    for h in range(W1.shape[0]):
        G = gains[h]
        if float(G.abs().max()) < thr.dead_gain:
            roles.append(NeuronRole(h, DEAD))
            continue
        computed = [i for i in range(G.shape[0]) if float(G[i, i]) >= thr.self_gain]
        column_min = G.min(dim=0).values
        target = int(column_min.argmin())
        inhibition = -float(column_min[target])
        largest_positive = float(G.clamp_min(0.0).max())
        if (
            inhibition >= thr.self_gain
            and inhibition >= thr.inhibition_dominance * largest_positive
        ):
            # Triggers: features read only to inhibit the target (not computed here).
            triggers = tuple(
                i
                for i in range(G.shape[0])
                if i != target and i not in computed and float(G[i, target]) <= -thr.dead_gain
            )
            rest = G.clone()
            rest[:, target] = 0.0
            rest[list(triggers), :] = 0.0
            rest_computed = [i for i in computed if i != target]
            rest_role = _two_feature_role(h, rest, W1, W2, rest_computed, thr)
            if rest_role is None and rest_computed:
                rest_role = (UNEXPLAINED, tuple(rest_computed))
            also_kind, also_features = rest_role or ("", ())
            roles.append(
                NeuronRole(
                    h,
                    INHIBITOR,
                    (target,),
                    also_kind,
                    also_features,
                    triggers,
                    tuple(rest_computed),
                )
            )
            continue
        role = _two_feature_role(h, G, W1, W2, computed, thr)
        if role is None:
            roles.append(NeuronRole(h, UNEXPLAINED, computes=tuple(computed)))
        else:
            roles.append(NeuronRole(h, role[0], role[1], computes=tuple(computed)))
    return roles


def find_motifs(
    W1: Float[Tensor, "hidden feat"],
    W2: Float[Tensor, "feat hidden"],
    b: Float[Tensor, " feat"],
    thr: RoleThresholds | None = None,
) -> list[Motif]:
    """Every (asymmetric neuron, inhibitor neuron) pair in one model (pp. 38–39).

    Checked on the weights directly, independently of :func:`neuron_roles`: h computes both
    features (G_h[i, i], G_h[j, j] ≥ ``self_gain``) with input and output sizes flipped by at
    least ``magnitude_ratio``; h′ ≠ h reads i with the same sign as h and has
    G_h′[i, j] ≤ −``cancel`` × G_h[i, j].
    """
    thr = thr or RoleThresholds()
    gains = path_gains(W1, W2)
    n_hidden, n_feat = W1.shape
    motifs = []
    for h in range(n_hidden):
        G = gains[h]
        for i in range(n_feat):
            for j in range(n_feat):
                if i == j or float(G[i, i]) < thr.self_gain or float(G[j, j]) < thr.self_gain:
                    continue
                w_in_i, w_in_j = abs(float(W1[h, i])), abs(float(W1[h, j]))
                w_out_i, w_out_j = float(W2[i, h]), float(W2[j, h])
                if w_in_i < thr.magnitude_ratio * w_in_j or w_out_j < thr.magnitude_ratio * w_out_i:
                    continue
                side = float(np.sign(float(W1[h, i])))
                same_side = torch.sign(W1[:, i]) == side  # neurons that fire on this side of i
                net = float(gains[same_side, i, j].sum() + b[j])
                for k in range(n_hidden):
                    if k == h or not bool(same_side[k]):
                        continue
                    if float(gains[k, i, j]) <= -thr.cancel * float(G[i, j]):
                        motifs.append(
                            Motif(
                                asymmetric_neuron=h,
                                inhibitor_neuron=k,
                                heavy=i,
                                victim=j,
                                input_ratio=w_in_i / w_in_j,
                                output_ratio=w_out_j / w_out_i,
                                interference=float(G[i, j]),
                                inhibition=float(gains[k, i, j]),
                                net_interference=net,
                            )
                        )
    return motifs


def role_consistent_motifs(motifs: list[Motif], roles: list[NeuronRole]) -> list[Motif]:
    """The motifs whose neurons carry the motif's roles (pp. 38–39; D-604).

    :func:`find_motifs` checks the weight pattern only. Here the asymmetric neuron must have
    the role asymmetric-superposition(heavy, victim) (as its role or its secondary role), so it
    stores those two features and nothing else, and the partner must be an inhibitor of the
    victim. A neuron storing three features with a flipped pair, or a partner whose negative
    gain is incidental, is not the paper's motif.
    """
    return [
        m
        for m in motifs
        if roles[m.asymmetric_neuron].has_role(ASYMMETRIC, (m.heavy, m.victim))
        and roles[m.inhibitor_neuron].kind == INHIBITOR
        and roles[m.inhibitor_neuron].features == (m.victim,)
    ]


# --- ablations -----------------------------------------------------------------------------------


def _abs_model(W1: Tensor, W2: Tensor, b: Tensor) -> AbsModel:
    """An :class:`AbsModel` holding the given ``(inst, …)`` weights (CPU)."""
    n_inst, n_hidden, n_feat = W1.shape
    model = AbsModel(n_feat, n_hidden, n_inst)
    with torch.no_grad():
        model.W1.copy_(W1)
        model.W2.copy_(W2)
        model.b.copy_(b)
    return model


@torch.no_grad()
def hidden_means(
    W1: Float[Tensor, "hidden feat"],
    feature_probability: float,
    n_samples: int,
    generator: torch.Generator,
) -> Float[Tensor, " hidden"]:
    """E[hₖ] = E[ReLU(W₁x)ₖ] on the task's input distribution (for mean ablation)."""
    dist = IndependentSparse(feature_probability, 1, W1.shape[1], ABS_LOW, ABS_HIGH)
    x = dist.sample(n_samples, generator)[:, 0]
    return torch.relu(x @ W1.T.to(x.device)).mean(dim=0).cpu()


@dataclass
class Ablation:
    """Per-feature loss changes when each neuron is ablated (one model).

    Attributes:
        baseline: ``(feat,)`` per-feature loss E[(|xᵢ| − y′ᵢ)²] of the intact model.
        mean_prediction: ``(feat,)`` loss of predicting the mean, Var(|xᵢ|) = p/3 − p²/4.
        zero: ``(hidden, feat)`` change in per-feature loss with hₖ set to 0.
        mean: ``(hidden, feat)`` change with hₖ set to its mean E[hₖ].
        hidden_mean: ``(hidden,)`` E[hₖ].
    """

    baseline: Tensor
    mean_prediction: Tensor
    zero: Tensor
    mean: Tensor
    hidden_mean: Tensor

    def relative(self, kind: str) -> Tensor:
        """``zero`` or ``mean`` divided by each feature's mean-prediction loss."""
        return getattr(self, kind) / self.mean_prediction[None, :]


def ablation_matrix(
    W1: Float[Tensor, "hidden feat"],
    W2: Float[Tensor, "feat hidden"],
    b: Float[Tensor, " feat"],
    feature_probability: float,
    n_samples: int,
    generator: torch.Generator,
) -> Ablation:
    """Zero- and mean-ablate every neuron and measure each feature's loss change (SPEC §6).

    One multi-instance model holds the intact model and its 2m ablated variants: zeroing hₖ is
    W₂[:, k] = 0, and mean-ablating it is W₂[:, k] = 0 with b += W₂[:, k] E[hₖ]. All variants
    are scored on the same inputs (``shared_samples``), with the exact conditional estimator
    :func:`superposition.metrics.per_feature_loss`.
    """
    n_hidden, n_feat = W1.shape
    mu = hidden_means(W1, feature_probability, n_samples, generator)
    keep = 1.0 - torch.eye(n_hidden)  # row k zeroes neuron k
    W2_ablated = W2[None] * keep[:, None, :]  # (hidden, feat, hidden)
    b_mean = b[None] + (W2 * mu[None, :]).T  # (hidden, feat): b + W₂[:, k] E[hₖ]
    n_inst = 1 + 2 * n_hidden
    model = _abs_model(
        W1[None].expand(n_inst, -1, -1),
        torch.cat([W2[None], W2_ablated, W2_ablated]),
        torch.cat([b[None], b[None].expand(n_hidden, -1), b_mean]),
    ).to(generator.device)
    loss = metrics.per_feature_loss(
        model,
        torch.full((n_inst,), feature_probability),
        n_samples,
        generator,
        low=ABS_LOW,
        high=ABS_HIGH,
        shared_samples=True,
    ).cpu()
    baseline = loss[0]
    return Ablation(
        baseline=baseline,
        mean_prediction=torch.full((n_feat,), float(uniform_feature_variance(feature_probability))),
        zero=loss[1 : 1 + n_hidden] - baseline,
        mean=loss[1 + n_hidden :] - baseline,
        hidden_mean=mu,
    )


def ablation_agrees(role: NeuronRole, relative: Float[Tensor, " feat"], rise: float) -> bool | None:
    """Do one neuron's ablation effects (relative to the mean-prediction loss) fit its role?

    * a role that computes or inhibits features: each predicted feature's loss rises by at
      least ``rise``, and the largest relative rise is on a predicted feature;
    * dead: no feature's loss moves by ``rise`` or more;
    * unexplained: no prediction (None).
    """
    if role.kind == UNEXPLAINED:
        return None
    if role.kind == DEAD:
        return bool(relative.abs().max() < rise)
    predicted = list(role.predicted_rise)
    if not predicted:
        return None
    rises = bool((relative[predicted] >= rise).all())
    return rises and int(relative.argmax()) in predicted


# --- rebuild -------------------------------------------------------------------------------------


def rebuild_weights(
    roles: Sequence[NeuronRole],
    W1n: Float[Tensor, "hidden feat"],
    W2n: Float[Tensor, "feat hidden"],
) -> tuple[Float[Tensor, "hidden feat"], Float[Tensor, "feat hidden"]]:
    """An idealized model built by hand from the roles (SPEC §6 rebuild test, step 1).

    In normalized units (each neuron's largest |W₁| = 1):

    * side(i): W₁[h, i] = ±1, W₂[i, h] = 1 (p. 36);
    * superposition(i, j): W₁ = ±1 on both (signs as trained), W₂ = 1 on both (p. 39);
    * asymmetric-superposition(i, j): the trained input magnitudes aᵢ, aⱼ (signs as trained)
      and reciprocal outputs W₂ = 1/aᵢ, 1/aⱼ, so each feature's own gain is exactly 1;
    * inhibitor(j): its secondary role as above, the trained inhibition weight W₂[j, h] and the
      trained trigger weights W₁[h, t] (the heavy features it reads a little of, p. 39);
    * dead and unexplained: zero.

    Everything a role does not name is zero. Only the numbers that define a role's shape (an
    asymmetric neuron's input magnitudes, an inhibitor's strength and triggers) come from the
    trained model.
    """
    W1 = torch.zeros_like(W1n)
    W2 = torch.zeros_like(W2n)

    def place(h: int, kind: str, features: tuple[int, ...]) -> None:
        if kind in (POSITIVE_SIDE, NEGATIVE_SIDE):
            (i,) = features
            W1[h, i] = 1.0 if kind == POSITIVE_SIDE else -1.0
            W2[i, h] = 1.0
        elif kind == SUPERPOSITION:
            for i in features:
                W1[h, i] = torch.sign(W1n[h, i])
                W2[i, h] = 1.0
        elif kind == ASYMMETRIC:
            for i in features:
                W1[h, i] = W1n[h, i]
                W2[i, h] = 1.0 / W1n[h, i].abs()

    for role in roles:
        h = role.neuron
        if role.kind == INHIBITOR:
            place(h, role.also_kind, role.also_features)
            (j,) = role.features
            W2[j, h] = W2n[j, h]
            for t in role.triggers:
                W1[h, t] = W1n[h, t]
        else:
            place(h, role.kind, role.features)
    return W1, W2


def idealized_bias(
    W2: Float[Tensor, "feat hidden"], feature_probability: float
) -> Float[Tensor, " feat"]:
    """b = 0 for features some neuron writes to, else the mean E|xᵢ| = p/2 (the best constant)."""
    written = (W2 > 0).any(dim=1)
    mean = torch.full(written.shape, feature_probability / 2, dtype=W2.dtype)
    return torch.where(written, torch.zeros_like(mean), mean)


@dataclass
class Rebuild:
    """The rebuild test for one model (SPEC §6).

    Per-feature losses are ``(feat,)``; ``fraction_recovered`` is
    (L_mean − L_rebuild) / (L_mean − L_trained) on the importance-weighted total, and
    ``fraction_by_feature`` the same per feature (NaN where the trained model does not beat
    the mean). ``*_trained_bias`` is the variant that keeps the trained output bias.
    """

    trained: Tensor
    rebuild: Tensor
    rebuild_trained_bias: Tensor
    mean_prediction: Tensor
    fraction_recovered: float
    fraction_recovered_trained_bias: float
    fraction_by_feature: Tensor
    W1: Tensor = field(repr=False)
    W2: Tensor = field(repr=False)
    b: Tensor = field(repr=False)


def _recovered(mean: Tensor, trained: Tensor, rebuild: Tensor, importance: Tensor) -> float:
    gain = float((importance * (mean - trained)).sum())
    return float((importance * (mean - rebuild)).sum()) / gain if gain > 0 else float("nan")


def rebuild_test(
    roles: Sequence[NeuronRole],
    W1: Float[Tensor, "hidden feat"],
    W2: Float[Tensor, "feat hidden"],
    b: Float[Tensor, " feat"],
    importance: Float[Tensor, " feat"],
    feature_probability: float,
    n_samples: int,
    generator: torch.Generator,
) -> Rebuild:
    """Build the idealized model, compare per-feature losses, report the recovered fraction."""
    W1n, W2n = metrics.normalize_hidden_units(W1[None], W2[None])
    W1r, W2r = rebuild_weights(roles, W1n[0], W2n[0])
    b_ideal = idealized_bias(W2r, feature_probability)
    model = _abs_model(
        torch.stack([W1, W1r, W1r]), torch.stack([W2, W2r, W2r]), torch.stack([b, b_ideal, b])
    ).to(generator.device)
    loss = metrics.per_feature_loss(
        model,
        torch.full((3,), feature_probability),
        n_samples,
        generator,
        low=ABS_LOW,
        high=ABS_HIGH,
        shared_samples=True,
    ).cpu()
    mean = torch.full_like(loss[0], float(uniform_feature_variance(feature_probability)))
    improvement = mean - loss[0]
    by_feature = torch.where(
        improvement > 1e-3 * mean, (mean - loss[1]) / improvement, torch.full_like(mean, np.nan)
    )
    return Rebuild(
        trained=loss[0],
        rebuild=loss[1],
        rebuild_trained_bias=loss[2],
        mean_prediction=mean,
        fraction_recovered=_recovered(mean, loss[0], loss[1], importance),
        fraction_recovered_trained_bias=_recovered(mean, loss[0], loss[2], importance),
        fraction_by_feature=by_feature,
        W1=W1r,
        W2=W2r,
        b=b_ideal,
    )


# --- the role table ------------------------------------------------------------------------------


def role_table(
    roles: Sequence[NeuronRole],
    W1n: Float[Tensor, "hidden feat"],
    W2n: Float[Tensor, "feat hidden"],
    ablation: Ablation | None = None,
    thr: RoleThresholds | None = None,
    digits: int = 2,
) -> pd.DataFrame:
    """One row per neuron: role, footnote-15-normalized weights and (optionally) ablations.

    With an :class:`Ablation`, the table adds the feature whose loss rises most under each
    ablation (relative to predicting the mean) and whether the effects agree with the role.
    """
    thr = thr or RoleThresholds()
    rows = []
    for role in roles:
        h = role.neuron
        row: dict[str, object] = {
            "neuron": h,
            "role": role.label,
            "kind": role.kind,
            "W1 (in)": np.round(W1n[h].double().numpy(), digits).tolist(),
            "W2 (out)": np.round(W2n[:, h].double().numpy(), digits).tolist(),
            "predicted rise": list(role.predicted_rise),
        }
        if ablation is not None:
            for kind in ("zero", "mean"):
                relative = ablation.relative(kind)[h]
                row[f"{kind}: top feature"] = int(relative.argmax())
                row[f"{kind}: top rise"] = round(float(relative.max()), 3)
                row[f"{kind}: agrees"] = ablation_agrees(role, relative, thr.ablation_rise)
        rows.append(row)
    return pd.DataFrame(rows)
