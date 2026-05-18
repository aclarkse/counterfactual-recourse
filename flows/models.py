"""
flows/models.py — Flow model definitions and loading utilities.

Exports: DiscreteMediator, ContinuousMediatorFlow, load_flow_models,
         combined_loss, evaluate, build_tensors (re-export).
"""

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
    """P(W_disc | X, Z) as a product of independent categoricals via a shared MLP."""

    def __init__(self, dim_x: int, dim_z: int, vocab: dict,
                 hidden_dim: int = 64, n_layers: int = 2):
        super().__init__()
        self.vocab     = vocab
        self.col_names = list(vocab.keys())
        n_cats = list(vocab.values())
        in_dim = dim_x + dim_z

        if in_dim == 0 or not vocab:
            self.backbone = self.heads = None
            return

        layers, d = [], in_dim
        for _ in range(n_layers):
            layers += [nn.Linear(d, hidden_dim), nn.ReLU()]
            d = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.heads = nn.ModuleList([nn.Linear(hidden_dim, k) for k in n_cats])

    def _cond(self, x, z):
        parts = [p for p in [x, z] if p.shape[1] > 0]
        return torch.cat(parts, dim=1) if parts else None

    def log_prob(self, w_disc: torch.Tensor, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if not self.vocab:
            return torch.zeros(x.shape[0], device=x.device)
        h = self.backbone(self._cond(x, z))
        total = torch.zeros(x.shape[0], device=x.device)
        for i, head in enumerate(self.heads):
            total += F.cross_entropy(head(h), w_disc[:, i], reduction="none") * -1.0
        return total

    @torch.no_grad()
    def sample(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        if not self.vocab:
            return torch.empty(x.shape[0], 0, dtype=torch.long, device=x.device)
        h = self.backbone(self._cond(x, z))
        cols = [torch.multinomial(torch.softmax(head(h), dim=-1), 1) for head in self.heads]
        return torch.cat(cols, dim=1)


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
        return self.flow.sample(1, context=self._context(w_disc, x, z)).squeeze(1)

    @torch.no_grad()
    def log_prob_no_grad(self, w_cont, w_disc, x, z):
        return self.log_prob(w_cont, w_disc, x, z)


def load_flow_models(checkpoint_path: str, device: torch.device):
    """Reload trained g_phi and f_theta from a checkpoint file."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    g_phi_ = DiscreteMediator(**ckpt["g_phi_cfg"]).to(device)
    f_theta_ = ContinuousMediatorFlow(**ckpt["f_theta_cfg"]).to(device)
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
