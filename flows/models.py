"""
flows/models.py — Flow model definitions and loading utilities.

Exports: DiscreteMediator, ContinuousMediatorFlow, load_flow_models,
         combined_loss, evaluate, build_tensors (re-export).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from nflows import flows
from nflows.transforms import (
    CompositeTransform,
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform,
    RandomPermutation,
)
from nflows.distributions import StandardNormal

# Re-export for backward compat
from data.build_tensors import build_tensors  # noqa: F401


class DiscreteMediator(nn.Module):
    """Conditional model for discrete mediators.

    With ``autoregressive=False`` this retains the original conditionally
    independent heads, which keeps old checkpoints loadable.  New experiments
    should use ``autoregressive=True`` and configured mediator blocks:

        p(w_d | x, z) = product_j p(w_j | pa(w_j), x, z).

    Only mediators in strictly earlier blocks are parents.  Thus an internal
    tensor/flow coordinate order is never interpreted as a causal relation
    between variables in the same block.
    """

    def __init__(self, dim_x: int, dim_z: int, vocab: dict,
                 hidden_dim: int = 64, n_layers: int = 2,
                 autoregressive: bool = False, embed_dim: int = 4,
                 layers=None):
        super().__init__()
        self.vocab     = vocab
        self.col_names = list(vocab.keys())
        self.autoregressive = autoregressive
        n_cats = list(vocab.values())
        self.cardinalities = n_cats
        in_dim = dim_x + dim_z

        # None is intentionally the historical left-to-right factorization,
        # preserving compatibility with old checkpoints.  New checkpoints
        # always record explicit layers.
        if layers is None:
            self.layers = [[name] for name in self.col_names]
            self._legacy_sequential = True
        else:
            self.layers = [list(block) for block in layers]
            flattened = [name for block in self.layers for name in block]
            if (set(flattened) != set(self.col_names)
                    or len(flattened) != len(set(flattened))):
                raise ValueError(
                    "Discrete mediator layers must contain every discrete "
                    "mediator exactly once."
                )
            self._legacy_sequential = False

        layer_by_name = {
            name: layer_idx
            for layer_idx, block in enumerate(self.layers)
            for name in block
        }
        self.parent_indices = []
        for j, name in enumerate(self.col_names):
            if self._legacy_sequential:
                parents = list(range(j))
            else:
                parents = [
                    k for k, parent in enumerate(self.col_names)
                    if layer_by_name[parent] < layer_by_name[name]
                ]
                if any(k >= j for k in parents):
                    raise ValueError(
                        "Discrete vocabulary columns must follow topological "
                        "mediator-block order."
                    )
            self.parent_indices.append(parents)

        if in_dim == 0 or not vocab:
            self.backbone = self.heads = None
            return

        if not autoregressive:
            layers, d = [], in_dim
            for _ in range(n_layers):
                layers += [nn.Linear(d, hidden_dim), nn.ReLU()]
                d = hidden_dim
            self.backbone = nn.Sequential(*layers)
            self.heads = nn.ModuleList([nn.Linear(hidden_dim, k) for k in n_cats])
            self.embeddings = None
        else:
            self.backbone = None
            self.embeddings = nn.ModuleList(
                [nn.Embedding(k, embed_dim) for k in n_cats]
            )
            self.heads = nn.ModuleList()
            for j, k in enumerate(n_cats):
                head_layers, d = [], in_dim + len(self.parent_indices[j]) * embed_dim
                for _ in range(n_layers):
                    head_layers += [nn.Linear(d, hidden_dim), nn.ReLU()]
                    d = hidden_dim
                head_layers.append(nn.Linear(d, k))
                self.heads.append(nn.Sequential(*head_layers))

    def _cond(self, x, z):
        parts = [p for p in [x, z] if p.shape[1] > 0]
        return torch.cat(parts, dim=1) if parts else None

    def log_prob(self, w_disc: torch.Tensor, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if not self.vocab:
            return torch.zeros(x.shape[0], device=x.device)
        total = torch.zeros(x.shape[0], device=x.device)
        if not self.autoregressive:
            h = self.backbone(self._cond(x, z))
            logits = [head(h) for head in self.heads]
        else:
            context = self._cond(x, z)
            logits = []
            observed = [
                self.embeddings[i](w_disc[:, i])
                for i in range(len(self.col_names))
            ]
            for i, head in enumerate(self.heads):
                parent_embeddings = [observed[j] for j in self.parent_indices[i]]
                head_input = torch.cat([context] + parent_embeddings, dim=1)
                logits.append(head(head_input))
        for i, values in enumerate(logits):
            total -= F.cross_entropy(values, w_disc[:, i], reduction="none")
        return total

    @torch.no_grad()
    def sample(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        values = torch.full(
            (x.shape[0], len(self.col_names)), -1,
            dtype=torch.long, device=x.device,
        )
        return self.sample_with_clamps(x, z, values)

    @torch.no_grad()
    def sample_with_clamps(self, x: torch.Tensor, z: torch.Tensor,
                           values: torch.Tensor) -> torch.Tensor:
        """Sample mechanisms while preserving non-negative clamped values.

        ``values`` has one column per discrete mediator.  An entry of ``-1``
        is generated from its conditional mechanism; a non-negative entry is
        an observed or intervened value that is held fixed.
        """
        if not self.vocab:
            return torch.empty(x.shape[0], 0, dtype=torch.long, device=x.device)
        if values.shape != (x.shape[0], len(self.col_names)):
            raise ValueError(
                "values must have shape [batch, number of discrete mediators]"
            )
        context = self._cond(x, z)
        if not self.autoregressive:
            h = self.backbone(context)
            cols = []
            for i, head in enumerate(self.heads):
                proposed = torch.multinomial(torch.softmax(head(h), dim=-1), 1)[:, 0]
                cols.append(torch.where(values[:, i] >= 0, values[:, i], proposed))
        else:
            cols = []
            for i, head in enumerate(self.heads):
                parent_embeddings = [
                    self.embeddings[j](cols[j]) for j in self.parent_indices[i]
                ]
                head_input = torch.cat([context] + parent_embeddings, dim=1)
                proposed = torch.multinomial(
                    torch.softmax(head(head_input), dim=-1), 1
                )[:, 0]
                cols.append(torch.where(values[:, i] >= 0, values[:, i], proposed))
        for i, col in enumerate(cols):
            if torch.any((col < 0) | (col >= self.cardinalities[i])):
                raise ValueError(f"Invalid clamp value for {self.col_names[i]}")
        return torch.stack(cols, dim=1)


class ContinuousMediatorFlow(nn.Module):
    """P(W_cont | W_disc, X, Z) via a conditional Neural Spline Flow (NSF).

    NSF replaces the affine coupling of MAF with a piecewise rational-quadratic
    spline, which is essential when dim_wc=1: stacked affine transforms on a
    1-D variable compose to a single affine map (= conditional Gaussian), giving
    a flat NLL after epoch 1. NSF remains expressive for any dim_wc >= 1.
    """

    def __init__(self, dim_wc: int, dim_x: int, dim_z: int, vocab: dict,
                 embed_dim: int = 4, hidden_features: int = 64,
                 n_flow_layers: int = 4, n_blocks: int = 2,
                 num_bins: int = 8, tail_bound: float = 3.0):
        super().__init__()
        self.dim_wc = dim_wc
        self.embeddings = nn.ModuleList([
            nn.Embedding(n_cats, embed_dim) for n_cats in vocab.values()
        ])
        self.dim_context = len(vocab) * embed_dim + dim_x + dim_z

        if dim_wc == 0:
            self.flow = None
            return

        transforms_ = []
        for _ in range(n_flow_layers):
            transforms_.append(
                MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                    features=dim_wc,
                    hidden_features=hidden_features,
                    context_features=self.dim_context if self.dim_context > 0 else None,
                    num_bins=num_bins,
                    num_blocks=n_blocks,
                    use_residual_blocks=True,
                    activation=F.relu,
                    tails="linear",
                    tail_bound=tail_bound,
                )
            )
            if dim_wc > 1:
                transforms_.append(RandomPermutation(features=dim_wc))

        self.flow = flows.Flow(
            transform=CompositeTransform(transforms_),
            distribution=StandardNormal(shape=[dim_wc]),
        )

    def _context(self, w_disc, x, z):
        parts = [emb(w_disc[:, i]) for i, emb in enumerate(self.embeddings)]
        parts += [p for p in [x, z] if p.shape[1] > 0]
        return torch.cat(parts, dim=1) if parts else None

    def log_prob(self, w_cont, w_disc, x, z):
        if self.dim_wc == 0 or self.flow is None:
            return torch.zeros(x.shape[0], device=x.device)
        return self.flow.log_prob(w_cont, context=self._context(w_disc, x, z))

    @torch.no_grad()
    def sample(self, n: int, w_disc, x, z):
        if self.dim_wc == 0 or self.flow is None:
            return torch.empty(x.shape[0], 0, device=x.device)
        if n != 1:
            raise ValueError(
                "ContinuousMediatorFlow.sample expects n=1; repeat the "
                "conditioning rows to obtain multiple draws."
            )
        context = self._context(w_disc, x, z)
        return self._sample_context_robust(context).squeeze(1)

    def _sample_context_robust(self, context, leaf_retries: int = 8):
        """Isolate rare float32 roundoff in the analytic spline inverse.

        ``nflows`` asserts that the inverse quadratic discriminant is
        non-negative.  It can very rarely become slightly negative in
        float32.  When a large Monte Carlo batch contains such a draw, split
        the batch recursively so only affected rows are resampled.  A bounded
        single-row retry prevents an actual invalid transform from being
        hidden.
        """
        try:
            return self.flow.sample(1, context=context)
        except AssertionError as error:
            if context.shape[0] > 1:
                midpoint = context.shape[0] // 2
                return torch.cat([
                    self._sample_context_robust(context[:midpoint], leaf_retries),
                    self._sample_context_robust(context[midpoint:], leaf_retries),
                ], dim=0)
            for _ in range(leaf_retries):
                try:
                    return self.flow.sample(1, context=context)
                except AssertionError:
                    continue
            raise RuntimeError(
                "Inverse spline remained numerically invalid after bounded "
                "single-row resampling."
            ) from error

    @torch.no_grad()
    def log_prob_no_grad(self, w_cont, w_disc, x, z):
        return self.log_prob(w_cont, w_disc, x, z)


class ConditionalGaussian(nn.Module):
    """Simple diagonal Gaussian baseline for P(W_cont | W_disc,X,Z)."""

    def __init__(self, dim_wc: int, dim_x: int, dim_z: int, vocab: dict,
                 embed_dim: int = 4, hidden_features: int = 64, **unused):
        super().__init__()
        self.dim_wc = dim_wc
        self.embeddings = nn.ModuleList([
            nn.Embedding(n_cats, embed_dim) for n_cats in vocab.values()
        ])
        context_dim = len(vocab) * embed_dim + dim_x + dim_z
        self.network = None if dim_wc == 0 else nn.Sequential(
            nn.Linear(context_dim, hidden_features), nn.ReLU(),
            nn.Linear(hidden_features, 2 * dim_wc),
        )

    def _context(self, w_disc, x, z):
        parts = [emb(w_disc[:, i]) for i, emb in enumerate(self.embeddings)]
        parts += [p for p in (x, z) if p.shape[1] > 0]
        return torch.cat(parts, dim=1)

    def _params(self, w_disc, x, z):
        mean, log_scale = self.network(self._context(w_disc, x, z)).chunk(2, 1)
        return mean, log_scale.clamp(-5.0, 3.0)

    def log_prob(self, w_cont, w_disc, x, z):
        if self.dim_wc == 0:
            return torch.zeros(x.shape[0], device=x.device)
        mean, log_scale = self._params(w_disc, x, z)
        standardized = (w_cont - mean) * torch.exp(-log_scale)
        return (-0.5 * standardized.square() - log_scale
                - 0.5 * math.log(2.0 * math.pi)).sum(1)

    @torch.no_grad()
    def sample(self, n, w_disc, x, z):
        del n
        if self.dim_wc == 0:
            return torch.empty(x.shape[0], 0, device=x.device)
        mean, log_scale = self._params(w_disc, x, z)
        return mean + torch.randn_like(mean) * torch.exp(log_scale)

    @torch.no_grad()
    def log_prob_no_grad(self, w_cont, w_disc, x, z):
        return self.log_prob(w_cont, w_disc, x, z)


def load_flow_models(checkpoint_path: str, device: torch.device):
    """Reload trained g_phi and f_theta from a checkpoint file."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    g_phi_ = DiscreteMediator(**ckpt["g_phi_cfg"]).to(device)
    family = ckpt.get("continuous_family", "spline")
    continuous_cls = (ConditionalGaussian if family == "gaussian"
                      else ContinuousMediatorFlow)
    f_theta_ = continuous_cls(**ckpt["f_theta_cfg"]).to(device)
    g_phi_.load_state_dict(ckpt["g_phi_state"])
    f_theta_.load_state_dict(ckpt["f_theta_state"])
    g_phi_.eval(); f_theta_.eval()
    return g_phi_, f_theta_, ckpt["scaler"], ckpt["sfm_cfg"]


def combined_loss(batch, g_phi, f_theta, device):
    x, z, w_disc, w_cont = [t.to(device) for t in batch[:4]]
    ll_disc = g_phi.log_prob(w_disc, x, z)
    ll_cont = f_theta.log_prob(w_cont, w_disc, x, z)
    return -(ll_disc + ll_cont).mean()


@torch.no_grad()
def evaluate(loader, g_phi, f_theta, device):
    total_disc, total_cont, n = 0.0, 0.0, 0
    for batch in loader:
        x, z, w_disc, w_cont = [t.to(device) for t in batch[:4]]
        ll_disc = g_phi.log_prob(w_disc, x, z)
        ll_cont = f_theta.log_prob(w_cont, w_disc, x, z)
        bs = x.shape[0]
        total_disc += (-ll_disc.mean().item()) * bs
        total_cont += (-ll_cont.mean().item()) * bs
        n += bs
    return total_disc / n, total_cont / n
