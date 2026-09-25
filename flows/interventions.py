"""Interventional sampling for partially ordered mediator mechanisms.

The learned models estimate observational conditional mechanisms.  This
module supplies the separate surgery needed for a mediator intervention:
directly acted-on coordinates are clamped, variables in later blocks are
regenerated ancestrally, and same-block coordinates are never treated as
descendants merely because of a tensor or flow coordinate order.
"""

from dataclasses import dataclass

import torch

from flows.schema import MediatorSchema


@dataclass(frozen=True)
class InterventionSamples:
    x: torch.Tensor
    z: torch.Tensor
    w_disc: torch.Tensor
    w_cont: torch.Tensor
    plan_index: torch.Tensor


def _validate_batch_tensor(name, value, expected_shape):
    if value.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {value.shape}")


@torch.no_grad()
def sample_intervention_batch(
    g_phi,
    f_theta,
    schema: MediatorSchema,
    x: torch.Tensor,
    z: torch.Tensor,
    factual_w_disc: torch.Tensor,
    factual_w_cont: torch.Tensor,
    action_w_disc: torch.Tensor,
    action_w_cont: torch.Tensor,
    action_mask_disc: torch.Tensor,
    action_mask_cont: torch.Tensor,
    n_samples: int = 1,
    same_level: str = "preserve_factual",
) -> InterventionSamples:
    """Draw mediator values under a batch of possibly different action plans.

    Continuous values are on the flow's transformed scale.  Each input row is
    one person/action plan and is repeated ``n_samples`` times.

    ``same_level='preserve_factual'`` gives an individual intervention: an
    unacted coordinate in the same unordered block retains that individual's
    factual value.  ``same_level='resample_marginal'`` instead draws the joint
    block from its conditional observational distribution and overwrites only
    the acted coordinates; this preserves the unacted coordinate's population
    marginal without asserting a within-block causal direction.
    """
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1")
    if same_level not in {"preserve_factual", "resample_marginal"}:
        raise ValueError(
            "same_level must be 'preserve_factual' or 'resample_marginal'"
        )

    batch = x.shape[0]
    n_disc = len(schema.discrete_names)
    n_cont = len(schema.continuous_names)
    _validate_batch_tensor("factual_w_disc", factual_w_disc, (batch, n_disc))
    _validate_batch_tensor("factual_w_cont", factual_w_cont, (batch, n_cont))
    _validate_batch_tensor("action_w_disc", action_w_disc, (batch, n_disc))
    _validate_batch_tensor("action_w_cont", action_w_cont, (batch, n_cont))
    _validate_batch_tensor("action_mask_disc", action_mask_disc, (batch, n_disc))
    _validate_batch_tensor("action_mask_cont", action_mask_cont, (batch, n_cont))

    action_mask_disc = action_mask_disc.bool()
    action_mask_cont = action_mask_cont.bool()
    plan_index = torch.arange(batch, device=x.device).repeat_interleave(n_samples)
    repeat = lambda value: value.repeat_interleave(n_samples, dim=0)
    x_rep, z_rep = repeat(x), repeat(z)
    factual_wd_rep, factual_wc_rep = repeat(factual_w_disc), repeat(factual_w_cont)
    action_wd_rep, action_wc_rep = repeat(action_w_disc), repeat(action_w_cont)
    mask_wd_rep, mask_wc_rep = repeat(action_mask_disc), repeat(action_mask_cont)

    layer_by_name = schema.layer_by_name
    no_action_layer = len(schema.layers)
    first_action_layer = torch.full(
        (batch,), no_action_layer, dtype=torch.long, device=x.device
    )
    for j, name in enumerate(schema.discrete_names):
        layer = torch.full_like(first_action_layer, layer_by_name[name])
        first_action_layer = torch.where(
            action_mask_disc[:, j], torch.minimum(first_action_layer, layer),
            first_action_layer,
        )
    for j, name in enumerate(schema.continuous_names):
        layer = torch.full_like(first_action_layer, layer_by_name[name])
        first_action_layer = torch.where(
            action_mask_cont[:, j], torch.minimum(first_action_layer, layer),
            first_action_layer,
        )
    first_action_rep = first_action_layer.repeat_interleave(n_samples)

    # Preserve all observed discrete values except strict descendants of the
    # earliest action.  Direct interventions override both cases.
    discrete_clamps = factual_wd_rep.clone()
    for j, name in enumerate(schema.discrete_names):
        is_descendant = layer_by_name[name] > first_action_rep
        should_sample = is_descendant & ~mask_wd_rep[:, j]
        discrete_clamps[should_sample, j] = -1
        discrete_clamps[mask_wd_rep[:, j], j] = action_wd_rep[mask_wd_rep[:, j], j]
    sampled_wd = g_phi.sample_with_clamps(x_rep, z_rep, discrete_clamps)

    sampled_wc = factual_wc_rep.clone()
    if n_cont:
        cont_layer = layer_by_name[schema.continuous_names[0]]
        draw_downstream = cont_layer > first_action_rep
        draw_same_level = torch.zeros_like(draw_downstream)
        if same_level == "resample_marginal":
            has_cont_action = mask_wc_rep.any(dim=1)
            draw_same_level = (cont_layer == first_action_rep) & has_cont_action
        should_draw_block = draw_downstream | draw_same_level
        if should_draw_block.any():
            natural_draw = f_theta.sample(1, sampled_wd, x_rep, z_rep)
            sampled_wc[should_draw_block] = natural_draw[should_draw_block]
        sampled_wc = torch.where(mask_wc_rep, action_wc_rep, sampled_wc)

    return InterventionSamples(
        x=x_rep,
        z=z_rep,
        w_disc=sampled_wd,
        w_cont=sampled_wc,
        plan_index=plan_index,
    )
