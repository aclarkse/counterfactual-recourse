"""
flows/diagnostics.py — Sampling and log-prob closures for flow models.

Used by evaluation/estimate_gap.py.
"""

import numpy as np
import torch


def make_sample_fns(g_phi, f_theta, scaler, device):
    """Return (sample_w_given_xz, log_prob_w_given_xz) closures."""

    WKHP_CLIP = (1, 60)

    def _to_original_scale(w_cont: torch.Tensor) -> torch.Tensor:
        """Invert QuantileTransform → clip to WKHP_CLIP → round to nearest hour."""
        if scaler is None or w_cont.shape[1] == 0:
            return w_cont
        arr = scaler.inverse_transform(w_cont.numpy())
        arr = np.clip(arr, *WKHP_CLIP)
        arr = np.round(arr).astype(np.float32)
        return torch.tensor(arr)

    @torch.no_grad()
    def sample_w_given_xz(x, z, K=500):
        g_phi.eval(); f_theta.eval()
        x, z = x.to(device), z.to(device)
        x_rep = x.repeat_interleave(K, dim=0)
        z_rep = z.repeat_interleave(K, dim=0)
        w_disc_s = g_phi.sample(x_rep, z_rep)
        w_cont_s = f_theta.sample(1, w_disc_s, x_rep, z_rep)
        ll_disc  = g_phi.log_prob(w_disc_s, x_rep, z_rep)
        ll_cont  = f_theta.log_prob_no_grad(w_cont_s, w_disc_s, x_rep, z_rep)
        return {
            "w_disc":      w_disc_s.cpu(),
            "w_cont":      w_cont_s.cpu(),
            "w_cont_orig": _to_original_scale(w_cont_s.cpu()),
            "log_prob":    (ll_disc + ll_cont).cpu(),
            "x_rep":       x_rep.cpu(),
            "z_rep":       z_rep.cpu(),
        }

    @torch.no_grad()
    def log_prob_w_given_xz(w_disc, w_cont, x, z):
        g_phi.eval(); f_theta.eval()
        w_disc, w_cont, x, z = [t.to(device) for t in [w_disc, w_cont, x, z]]
        ll_disc = g_phi.log_prob(w_disc, x, z)
        ll_cont = f_theta.log_prob_no_grad(w_cont, w_disc, x, z)
        return (ll_disc + ll_cont).cpu()

    return sample_w_given_xz, log_prob_w_given_xz
