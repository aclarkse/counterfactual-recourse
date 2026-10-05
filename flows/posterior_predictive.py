"""Held-out posterior-predictive checks for categorical mediator models."""

from __future__ import annotations

import torch


def _probabilities(values: torch.Tensor, cardinality: int) -> torch.Tensor:
    counts = torch.bincount(values.long(), minlength=cardinality).float()
    return counts / counts.sum().clamp_min(1.0)


def _total_variation(real: torch.Tensor, generated: torch.Tensor,
                     cardinality: int) -> float:
    p = _probabilities(real, cardinality)
    q = _probabilities(generated, cardinality)
    return float(0.5 * torch.abs(p - q).sum())


def _ordered_wasserstein(real: torch.Tensor, generated: torch.Tensor,
                         cardinality: int) -> float:
    """Wasserstein-1 on ordered category codes, normalized to [0, 1]."""
    if cardinality <= 1:
        return 0.0
    p = _probabilities(real, cardinality)
    q = _probabilities(generated, cardinality)
    return float(torch.abs(torch.cumsum(p, 0)[:-1]
                           - torch.cumsum(q, 0)[:-1]).sum()
                 / (cardinality - 1))


def _joint_codes(values: torch.Tensor, cardinalities: list[int]) -> torch.Tensor:
    codes = torch.zeros(len(values), dtype=torch.long)
    multiplier = 1
    for column, cardinality in reversed(list(enumerate(cardinalities))):
        codes += values[:, column].long() * multiplier
        multiplier *= cardinality
    return codes


@torch.no_grad()
def categorical_posterior_predictive(
    model,
    X: torch.Tensor,
    Z: torch.Tensor,
    W_disc: torch.Tensor,
    vocab: dict,
    *,
    samples_per_context: int = 8,
    max_rows: int = 4096,
    seed: int = 0,
    ordered_names=(),
    min_group_rows: int = 100,
) -> dict:
    """Compare generated categorical mediators with held-out observations.

    The validation contexts (X,Z) remain fixed. Metrics include marginal
    total variation overall and within each sensitive group, normalized
    Wasserstein distance for explicitly ordered mediators, and joint TV as a
    diagnostic. Joint TV is intentionally not used as a default gate because
    sparse high-dimensional tables have substantial finite-sample variation.
    """
    names = list(vocab)
    cardinalities = [int(vocab[name]) for name in names]
    if not names:
        return {
            "n_contexts": min(len(X), int(max_rows)),
            "samples_per_context": int(samples_per_context),
            "mediators": {},
            "joint_tv": 0.0,
            "max_overall_marginal_tv": 0.0,
            "max_group_marginal_tv": 0.0,
            "max_ordered_wasserstein": 0.0,
        }
    if W_disc.shape[1] != len(names):
        raise ValueError("W_disc columns do not match the configured vocabulary.")
    unknown_ordered = set(ordered_names) - set(names)
    if unknown_ordered:
        raise ValueError("Unknown ordered mediators: {}".format(
            sorted(unknown_ordered)
        ))

    n = min(len(X), int(max_rows))
    generator = torch.Generator().manual_seed(int(seed))
    indices = torch.randperm(len(X), generator=generator)[:n].sort().values
    x = X[indices]
    z = Z[indices]
    observed = W_disc[indices].cpu()
    device = next(model.parameters()).device
    x_device = x.to(device).repeat_interleave(samples_per_context, dim=0)
    z_device = z.to(device).repeat_interleave(samples_per_context, dim=0)
    cuda_devices = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda" else []
    )
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(int(seed))
        generated = model.sample(x_device, z_device).cpu()

    groups_observed = x[:, 0].long().cpu()
    groups_generated = groups_observed.repeat_interleave(samples_per_context)
    ordered = set(ordered_names)
    mediator_metrics = {}
    overall_tvs, group_tvs, ordered_distances = [], [], []

    for column, (name, cardinality) in enumerate(zip(names, cardinalities)):
        metric = {
            "overall_tv": _total_variation(
                observed[:, column], generated[:, column], cardinality
            ),
            "group_tv": {},
        }
        overall_tvs.append(metric["overall_tv"])
        if name in ordered:
            metric["ordered_wasserstein"] = _ordered_wasserstein(
                observed[:, column], generated[:, column], cardinality
            )
            ordered_distances.append(metric["ordered_wasserstein"])

        for group in torch.unique(groups_observed).tolist():
            observed_mask = groups_observed == group
            if int(observed_mask.sum()) < int(min_group_rows):
                continue
            generated_mask = groups_generated == group
            value = _total_variation(
                observed[observed_mask, column],
                generated[generated_mask, column],
                cardinality,
            )
            metric["group_tv"][str(int(group))] = value
            group_tvs.append(value)
            if name in ordered:
                value = _ordered_wasserstein(
                    observed[observed_mask, column],
                    generated[generated_mask, column],
                    cardinality,
                )
                metric.setdefault("group_ordered_wasserstein", {})[
                    str(int(group))
                ] = value
                ordered_distances.append(value)
        mediator_metrics[name] = metric

    joint_cardinality = int(torch.tensor(cardinalities).prod())
    joint_tv = _total_variation(
        _joint_codes(observed, cardinalities),
        _joint_codes(generated, cardinalities),
        joint_cardinality,
    )
    return {
        "n_contexts": int(n),
        "samples_per_context": int(samples_per_context),
        "mediators": mediator_metrics,
        "joint_tv": joint_tv,
        "max_overall_marginal_tv": max(overall_tvs, default=0.0),
        "max_group_marginal_tv": max(group_tvs, default=0.0),
        "max_ordered_wasserstein": max(ordered_distances, default=0.0),
    }

def _normalized_quantile_wasserstein(
    real: torch.Tensor, generated: torch.Tensor
) -> float:
    """Approximate marginal W1, normalized by the observed standard deviation."""
    quantiles = torch.linspace(0.0, 1.0, 101)
    distance = torch.abs(
        torch.quantile(real.float(), quantiles)
        - torch.quantile(generated.float(), quantiles)
    ).mean()
    scale = real.float().std(unbiased=False).clamp_min(1e-6)
    return float(distance / scale)


def _correlation_matrix(values: torch.Tensor) -> torch.Tensor:
    centered = values.float() - values.float().mean(dim=0, keepdim=True)
    covariance = centered.T @ centered
    norms = torch.sqrt(torch.diag(covariance).clamp_min(1e-12))
    return covariance / (norms[:, None] * norms[None, :]).clamp_min(1e-12)


def _max_correlation_error(
    real: torch.Tensor, generated: torch.Tensor
) -> float:
    if real.shape[1] < 2:
        return 0.0
    difference = torch.abs(
        _correlation_matrix(real) - _correlation_matrix(generated)
    )
    upper = torch.triu(
        torch.ones_like(difference, dtype=torch.bool), diagonal=1
    )
    return float(difference[upper].max())


@torch.no_grad()
def mediator_posterior_predictive(
    discrete_model,
    continuous_model,
    X: torch.Tensor,
    Z: torch.Tensor,
    W_disc: torch.Tensor,
    W_cont: torch.Tensor,
    vocab: dict,
    continuous_names,
    **kwargs,
) -> dict:
    """Held-out checks for both categorical and joint continuous mediators."""
    diagnostics = categorical_posterior_predictive(
        discrete_model, X, Z, W_disc, vocab, **kwargs
    )
    names = list(continuous_names)
    diagnostics.update({
        "continuous_mediators": {},
        "max_overall_continuous_w1": 0.0,
        "max_group_continuous_w1": 0.0,
        "max_continuous_correlation_error": 0.0,
    })
    if not names:
        return diagnostics
    if W_cont.shape[1] != len(names):
        raise ValueError(
            "W_cont columns do not match the configured continuous mediators."
        )

    max_rows = int(kwargs.get("max_rows", 4096))
    samples_per_context = int(kwargs.get("samples_per_context", 8))
    min_group_rows = int(kwargs.get("min_group_rows", 100))
    seed = int(kwargs.get("seed", 0))
    n = min(len(X), max_rows)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(X), generator=generator)[:n].sort().values
    x, z = X[indices], Z[indices]
    observed = W_cont[indices].cpu()
    observed_groups = x[:, 0].long().cpu()
    generated_groups = observed_groups.repeat_interleave(samples_per_context)

    parameter = next(discrete_model.parameters(), None)
    if parameter is None:
        parameter = next(continuous_model.parameters())
    device = parameter.device
    x_device = x.to(device).repeat_interleave(samples_per_context, dim=0)
    z_device = z.to(device).repeat_interleave(samples_per_context, dim=0)
    cuda_devices = (
        [device.index if device.index is not None
         else torch.cuda.current_device()]
        if device.type == "cuda" else []
    )
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        generated_disc = discrete_model.sample(x_device, z_device)
        generated = continuous_model.sample(
            1, generated_disc, x_device, z_device
        ).cpu()

    overall_distances, group_distances = [], []
    for column, name in enumerate(names):
        metric = {
            "overall_w1": _normalized_quantile_wasserstein(
                observed[:, column], generated[:, column]
            ),
            "group_w1": {},
        }
        overall_distances.append(metric["overall_w1"])
        for group in torch.unique(observed_groups).tolist():
            observed_mask = observed_groups == group
            if int(observed_mask.sum()) < min_group_rows:
                continue
            generated_mask = generated_groups == group
            value = _normalized_quantile_wasserstein(
                observed[observed_mask, column],
                generated[generated_mask, column],
            )
            metric["group_w1"][str(int(group))] = value
            group_distances.append(value)
        diagnostics["continuous_mediators"][name] = metric

    correlation_errors = [_max_correlation_error(observed, generated)]
    for group in torch.unique(observed_groups).tolist():
        observed_mask = observed_groups == group
        if int(observed_mask.sum()) < min_group_rows:
            continue
        generated_mask = generated_groups == group
        correlation_errors.append(_max_correlation_error(
            observed[observed_mask], generated[generated_mask]
        ))

    diagnostics.update({
        "max_overall_continuous_w1": max(overall_distances, default=0.0),
        "max_group_continuous_w1": max(group_distances, default=0.0),
        "max_continuous_correlation_error": max(
            correlation_errors, default=0.0
        ),
    })
    return diagnostics
