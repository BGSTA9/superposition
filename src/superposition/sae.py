"""Sparse autoencoders on toy-model activations, evaluated against ground truth.

Paper: Approach 2: Finding an Overcomplete Basis, pp. 42–44. "Take a regular model, which has
superposition, and find an overcomplete basis describing how features are embedded after the
fact. This appears to be a relatively standard sparse coding problem" (p. 43); footnote 20
(p. 57): given hidden activations H ∼ [d, m], find A ∼ [d, n] and B ∼ [n, m] with H ≈ AB and A
sparse. Toy models are the place to test this, because "we do know what the features are!"
(p. 44).

The sparse autoencoder (SAE) here is the standard way to do that factorization:

    f = ReLU(W_enc (h − b_dec) + b_enc)          (the sparse code, a row of A)
    ĥ = W_dec f + b_dec                          (W_dec's columns are the dictionary, B)
    loss = ‖h − ĥ‖² + λ ‖f‖₁                    (per sample: summed over dims, mean over batch)

* **Unit-norm decoder.** Each column of W_dec is renormalized to norm 1 after every optimizer
  step; otherwise the L1 penalty could be dodged by shrinking f and growing W_dec.
* **TopK variant.** Instead of an L1 penalty, keep only the k largest activations per sample
  (``topk`` > 0, usually with λ = 0).
* **Dead-latent resampling.** A latent that never fires gets no gradient and never recovers.
  Every ``resample_every`` steps (until ``resample_until`` of training), each latent that fired
  on fewer than ``dead_frequency`` of the samples since the last check counts as dead (a
  frequency rather than "never", because with very sparse inputs latents can fire once in
  millions of samples and still be useless; D-503), and is
  re-initialized toward an input the SAE reconstructs badly: an input is drawn with
  probability ∝ its squared reconstruction error, the decoder column is set to its direction,
  the encoder row to the same direction scaled to a fraction of the live rows' norm, the
  encoder bias to 0, and the latent's Adam moments are reset.

Everything has a leading instance dimension, like the toy models: one :class:`SparseAutoencoder`
trains a sweep (λ, TopK k, resampling on/off, dictionary sizes via ``latent_mask``) in one
vectorized model. **Ground-truth evaluation** (:func:`recovery`): for each represented true
feature direction Wᵢ of the toy model, the largest cosine similarity with any *live* decoder
column (a dead latent's direction is never used, so it cannot count), and the share of
features recovered above 0.9 and 0.95. Plus L0, reconstruction MSE (and the fraction of
variance unexplained), and the dead-latent fraction.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import einops
import torch
from jaxtyping import Bool, Float, Int
from torch import Tensor, nn

from superposition.utils import DATA_STREAM, INIT_STREAM, derive_seed, make_generator

# derive_seed stream key for SAE initialization (distinct from the toy models' keys).
SAE_INIT_KEY = 7
_ZERO_NORM = 1e-12
# Cap on elements per evaluation chunk (samples · inst · latent), to bound memory.
_EVAL_CHUNK_ELEMENTS = 1 << 24


@dataclass
class SAETrainConfig:
    """SAE optimizer and resampling settings (D-503)."""

    steps: int = 10_000
    batch_size: int = 4096
    lr: float = 1e-3
    resample_every: int = 2_000
    resample_until: float = 0.6  # no resampling after this fraction of training
    dead_frequency: float = 1e-5  # "dead": fired on fewer than this share of the window's samples
    resample_scale: float = 0.2  # new encoder rows: this × the mean norm of live rows
    eval_samples: int = 100_000
    log_every: int = 100


class SparseAutoencoder(nn.Module):
    """A multi-instance sparse autoencoder on m-dimensional activations.

    Paper: pp. 42–44 and footnote 20. Parameters: ``W_enc (inst, latent, hidden)``,
    ``b_enc (inst, latent)``, ``W_dec (inst, hidden, latent)`` with unit-norm columns,
    ``b_dec (inst, hidden)``.

    Args:
        n_hidden: m, the dimension of the activations being decomposed.
        n_latents: dictionary size (the largest one when ``latent_mask`` pads smaller ones).
        n_inst: number of SAEs trained together.
        l1: λ, one per instance (or a scalar).
        topk: k per instance; 0 means no TopK (plain ReLU + L1).
        resample: whether each instance resamples dead latents.
        latent_mask: ``(inst, latent)``, False for latents an instance does not have, so
            several dictionary sizes share one model. Masked latents are always 0.
        seed: instance k's initialization depends only on (seed, k).
    """

    # Per-instance settings, registered as buffers (they move with the model's device).
    l1: Tensor
    top_k: Tensor
    resample_dead: Tensor
    latent_mask: Tensor

    def __init__(
        self,
        n_hidden: int,
        n_latents: int,
        n_inst: int,
        l1: float | Sequence[float] | Tensor = 0.0,
        topk: int | Sequence[int] | Tensor = 0,
        resample: bool | Sequence[bool] | Tensor = True,
        latent_mask: Bool[Tensor, "inst latent"] | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.n_hidden = n_hidden
        self.n_latents = n_latents
        self.n_inst = n_inst
        decoder = torch.empty(n_inst, n_hidden, n_latents)
        for k in range(n_inst):  # initialization only, not a hot path
            g = torch.Generator().manual_seed(derive_seed(seed, INIT_STREAM, SAE_INIT_KEY, k))
            decoder[k] = torch.randn(n_hidden, n_latents, generator=g)
        decoder = decoder / decoder.norm(dim=1, keepdim=True).clamp_min(_ZERO_NORM)
        self.W_dec = nn.Parameter(decoder)
        self.W_enc = nn.Parameter(decoder.transpose(1, 2).clone())
        self.b_enc = nn.Parameter(torch.zeros(n_inst, n_latents))
        self.b_dec = nn.Parameter(torch.zeros(n_inst, n_hidden))
        mask = (
            torch.ones(n_inst, n_latents, dtype=torch.bool) if latent_mask is None else latent_mask
        )
        self.register_buffer(
            "l1", torch.broadcast_to(torch.as_tensor(l1, dtype=torch.float32), (n_inst,)).clone()
        )
        self.register_buffer(
            "top_k", torch.broadcast_to(torch.as_tensor(topk, dtype=torch.long), (n_inst,)).clone()
        )
        self.register_buffer(
            "resample_dead",
            torch.broadcast_to(torch.as_tensor(resample, dtype=torch.bool), (n_inst,)).clone(),
        )
        self.register_buffer("latent_mask", mask.clone())

    def encode(self, h: Float[Tensor, "batch inst hidden"]) -> Float[Tensor, "batch inst latent"]:
        """f = ReLU(W_enc (h − b_dec) + b_enc), masked, then TopK where ``topk`` > 0."""
        centered = h - self.b_dec
        pre = einops.einsum(
            self.W_enc, centered, "inst latent hidden, batch inst hidden -> batch inst latent"
        )
        f = torch.relu(pre + self.b_enc) * self.latent_mask
        k_max = int(self.top_k.max())
        if k_max > 0:
            k_max = min(k_max, self.n_latents)
            top = f.topk(k_max, dim=-1).values  # (batch, inst, k_max), descending
            k = self.top_k.clamp(1, k_max)
            kth = top.gather(-1, (k - 1)[None, :, None].expand(f.shape[0], -1, 1))
            keep = (f >= kth) | (self.top_k == 0)[None, :, None]
            f = f * keep
        return f

    def decode(self, f: Float[Tensor, "batch inst latent"]) -> Float[Tensor, "batch inst hidden"]:
        """ĥ = W_dec f + b_dec."""
        out = einops.einsum(
            self.W_dec, f, "inst hidden latent, batch inst latent -> batch inst hidden"
        )
        return out + self.b_dec

    def forward(
        self, h: Float[Tensor, "batch inst hidden"]
    ) -> tuple[Float[Tensor, "batch inst hidden"], Float[Tensor, "batch inst latent"]]:
        """Return ``(ĥ, f)``."""
        f = self.encode(h)
        return self.decode(f), f

    def loss_terms(
        self, h: Float[Tensor, "batch inst hidden"]
    ) -> tuple[Float[Tensor, " inst"], Float[Tensor, " inst"], Float[Tensor, "batch inst latent"]]:
        """Per-instance MSE (summed over dims, mean over batch), L1 of f, and f itself."""
        h_hat, f = self(h)
        mse = ((h - h_hat) ** 2).sum(dim=-1).mean(dim=0)
        return mse, f.abs().sum(dim=-1).mean(dim=0), f

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        """Rescale every decoder column to unit norm (after every optimizer step)."""
        self.W_dec.div_(self.W_dec.norm(dim=1, keepdim=True).clamp_min(_ZERO_NORM))

    def unit_decoder(self) -> Float[Tensor, "inst hidden latent"]:
        return self.W_dec.detach() / self.W_dec.detach().norm(dim=1, keepdim=True).clamp_min(
            _ZERO_NORM
        )


ActivationSampler = Callable[[int, torch.Generator], Float[Tensor, "batch hidden"]]


@dataclass
class SAETrainResult:
    """What :func:`train_sae` returns (CPU tensors)."""

    log_steps: list[int]
    mse_history: Tensor  # (log, inst)
    l1_history: Tensor  # (log, inst)
    resampled: Tensor  # (inst,) latents resampled in total
    wall_time_s: float
    resample_steps: list[int] = field(default_factory=list)


@torch.no_grad()
def resample_dead_latents(
    sae: SparseAutoencoder,
    optimizer: torch.optim.Adam,
    h: Float[Tensor, "batch inst hidden"],
    dead: Bool[Tensor, "inst latent"],
    scale: float,
    generator: torch.Generator,
) -> Int[Tensor, " inst"]:
    """Re-initialize dead latents toward badly reconstructed inputs (module docstring).

    Only instances with ``sae.resample_dead`` and only unmasked latents are touched. Returns the
    number of latents resampled per instance.
    """
    dead = dead & sae.latent_mask & sae.resample_dead[:, None]
    counts = dead.sum(dim=-1)
    if not bool(dead.any()):
        return counts.cpu()
    h_hat, _ = sae(h)
    error = ((h - h_hat) ** 2).sum(dim=-1)  # (batch, inst)
    alive = sae.latent_mask & ~dead
    row_norm = sae.W_enc.norm(dim=-1)  # (inst, latent)
    mean_alive = (row_norm * alive).sum(-1) / alive.sum(-1).clamp_min(1)
    mean_alive = torch.where(alive.any(-1), mean_alive, torch.ones_like(mean_alive))
    for k in torch.nonzero(counts).flatten().tolist():  # a few instances, a few times per run
        latents = torch.nonzero(dead[k]).flatten()
        weights = error[:, k]
        if float(weights.sum()) <= 0:
            directions = torch.randn(
                len(latents), sae.n_hidden, generator=generator, device=h.device
            )
        else:
            picks = torch.multinomial(weights, len(latents), replacement=True, generator=generator)
            directions = h[picks, k] - sae.b_dec[k]
        directions = directions / directions.norm(dim=-1, keepdim=True).clamp_min(_ZERO_NORM)
        sae.W_dec[k][:, latents] = directions.T
        sae.W_enc[k][latents] = directions * scale * mean_alive[k]
        sae.b_enc[k][latents] = 0.0
        for param, index in (
            (sae.W_dec, (k, slice(None), latents)),
            (sae.W_enc, (k, latents)),
            (sae.b_enc, (k, latents)),
        ):
            state = optimizer.state.get(param, {})
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state:
                    state[key][index] = 0.0
    return counts.cpu()


def train_sae(
    sae: SparseAutoencoder,
    sample: ActivationSampler,
    config: SAETrainConfig,
    seed: int = 0,
    progress: bool = False,
) -> SAETrainResult:
    """Train every instance of ``sae`` with Adam on activations drawn by ``sample``.

    ``sample(batch, generator)`` returns ``(batch, hidden)`` activations of the frozen toy
    model; every instance sees the same batch. The objective per instance is
    MSE + λ·L1 (step 1), summed over instances for ``backward()`` (instances never interact).
    """
    from tqdm import tqdm

    device = sae.W_dec.device
    data_gen = make_generator(derive_seed(seed, DATA_STREAM, SAE_INIT_KEY, 0), device)
    resample_gen = make_generator(derive_seed(seed, DATA_STREAM, SAE_INIT_KEY, 1), device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=config.lr)
    fires = torch.zeros(sae.n_inst, sae.n_latents, device=device)  # since the last check
    window = 0
    resampled = torch.zeros(sae.n_inst, dtype=torch.long)
    log_steps: list[int] = []
    mse_log: list[Tensor] = []
    l1_log: list[Tensor] = []
    resample_steps: list[int] = []
    last = int(config.resample_until * config.steps)
    start = time.perf_counter()
    for step in tqdm(range(config.steps), disable=not progress, mininterval=1.0, leave=False):
        h = sample(config.batch_size, data_gen)[:, None, :].expand(-1, sae.n_inst, -1)
        mse, l1, f = sae.loss_terms(h)
        optimizer.zero_grad(set_to_none=True)
        (mse + sae.l1 * l1).sum().backward()
        optimizer.step()
        sae.normalize_decoder()
        fires += (f.detach() > 0).sum(dim=0)
        window += config.batch_size
        if step % config.log_every == 0 or step == config.steps - 1:
            log_steps.append(step)
            mse_log.append(mse.detach().cpu())
            l1_log.append(l1.detach().cpu())
        done = step + 1
        if config.resample_every and done % config.resample_every == 0 and done <= last:
            dead = fires < config.dead_frequency * window
            resampled += resample_dead_latents(
                sae, optimizer, h, dead, config.resample_scale, resample_gen
            )
            resample_steps.append(done)
            fires.zero_()
            window = 0
    return SAETrainResult(
        log_steps=log_steps,
        mse_history=torch.stack(mse_log),
        l1_history=torch.stack(l1_log),
        resampled=resampled,
        wall_time_s=time.perf_counter() - start,
        resample_steps=resample_steps,
    )


# --- evaluation --------------------------------------------------------------------------------


@dataclass
class SAEStats:
    """Reconstruction statistics per instance on held-out activations.

    Attributes:
        l0: mean number of active latents per sample (all samples, zeros included).
        mse: mean ‖h − ĥ‖².
        fvu: fraction of variance unexplained, mse / E‖h − E[h]‖².
        alive: ``(inst, latent)`` latents that fired at least once (masked latents: False).
        dead_fraction: share of each instance's (unmasked) latents that never fired.
    """

    l0: Tensor
    mse: Tensor
    fvu: Tensor
    alive: Tensor
    dead_fraction: Tensor


@torch.no_grad()
def sae_stats(sae: SparseAutoencoder, h: Float[Tensor, "samples hidden"]) -> SAEStats:
    """L0, MSE, FVU and dead latents on held-out activations, in memory-bounded chunks."""
    n = h.shape[0]
    chunk = max(1, min(n, _EVAL_CHUNK_ELEMENTS // (sae.n_inst * sae.n_latents)))
    l0 = torch.zeros(sae.n_inst, device=h.device)
    sq = torch.zeros(sae.n_inst, device=h.device)
    alive = torch.zeros(sae.n_inst, sae.n_latents, dtype=torch.bool, device=h.device)
    for s in range(0, n, chunk):
        hb = h[s : s + chunk][:, None, :].expand(-1, sae.n_inst, -1)
        h_hat, f = sae(hb)
        active = f > 0
        l0 += active.sum(dim=-1).sum(dim=0)
        sq += ((hb - h_hat) ** 2).sum(dim=-1).sum(dim=0)
        alive |= active.any(dim=0)
    variance = ((h - h.mean(dim=0)) ** 2).sum(dim=-1).mean()
    mse = sq / n
    n_latents = sae.latent_mask.sum(dim=-1).clamp_min(1)
    return SAEStats(
        l0=(l0 / n).cpu(),
        mse=mse.cpu(),
        fvu=(mse / variance.clamp_min(_ZERO_NORM)).cpu(),
        alive=alive.cpu(),
        dead_fraction=(1 - (alive & sae.latent_mask).sum(-1) / n_latents).cpu(),
    )


@dataclass
class Recovery:
    """Ground-truth recovery per SAE instance (p. 44).

    Attributes:
        max_cos: ``(inst, feat)`` largest cos(Wᵢ, live decoder column) per represented feature
            (NaN for features that are not represented).
        best_latent: ``(inst, feat)`` the latent achieving it (−1 if none is alive).
        mean_max_cos: ``(inst,)`` mean over represented features.
        fraction: ``{threshold: (inst,)}`` share of represented features with max cos above it.
    """

    max_cos: Tensor
    best_latent: Tensor
    mean_max_cos: Tensor
    fraction: dict[float, Tensor]


def recovery(
    decoder: Float[Tensor, "inst hidden latent"],
    alive: Bool[Tensor, "inst latent"],
    true_directions: Float[Tensor, "hidden feat"],
    represented: Bool[Tensor, " feat"],
    thresholds: Sequence[float] = (0.9, 0.95),
) -> Recovery:
    """Compare an SAE's decoder with the toy model's true feature directions.

    Uses the signed cosine: SAE codes are non-negative, so a latent recovers feature i only if
    its decoder column points along +Wᵢ. Only live latents count.
    """
    unit_true = true_directions / true_directions.norm(dim=0, keepdim=True).clamp_min(_ZERO_NORM)
    unit_dec = decoder / decoder.norm(dim=1, keepdim=True).clamp_min(_ZERO_NORM)
    cos = einops.einsum(unit_true, unit_dec, "hidden feat, inst hidden latent -> inst feat latent")
    cos = torch.where(alive[:, None, :], cos, torch.full_like(cos, -torch.inf))
    best = cos.max(dim=-1)
    max_cos = torch.where(
        represented[None, :], best.values, torch.full_like(best.values, torch.nan)
    )
    max_cos = torch.where(torch.isinf(max_cos), torch.full_like(max_cos, -1.0), max_cos)
    best_latent = torch.where(torch.isinf(best.values), -1, best.indices)
    rep = represented[None, :].expand_as(max_cos)
    n_rep = max(int(represented.sum()), 1)
    fraction = {t: ((max_cos > t) & rep).sum(dim=-1).float() / n_rep for t in thresholds}
    mean = torch.where(rep, max_cos, torch.zeros_like(max_cos)).sum(dim=-1) / n_rep
    return Recovery(max_cos=max_cos, best_latent=best_latent, mean_max_cos=mean, fraction=fraction)
