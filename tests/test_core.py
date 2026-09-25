import unittest

import numpy as np
import torch

from evaluation.recourse_metrics import (
    add_candidate_geometry,
    add_sampled_candidate_geometry,
    attach_mediated_disparities,
    select_best,
)
from evaluation.compute_recourse import _bootstrap_absolute_closure
from evaluation.synthetic_validation import run as run_synthetic
from flows.interventions import sample_intervention_batch
from flows.models import (
    ConditionalGaussian,
    ContinuousMediatorFlow,
    DiscreteMediator,
)
from flows.schema import MediatorSchema


class _DeterministicDiscrete:
    def sample_with_clamps(self, x, z, values):
        del x, z
        result = values.clone()
        # A missing first mediator becomes 1; a missing second mediator is a
        # deterministic child of the first.  This exposes which blocks were
        # actually regenerated without relying on random neural weights.
        if result.shape[1] >= 1:
            result[:, 0] = torch.where(result[:, 0] < 0, 1, result[:, 0])
        if result.shape[1] >= 2:
            child = 1 - result[:, 0]
            result[:, 1] = torch.where(result[:, 1] < 0, child, result[:, 1])
        return result


class _DeterministicContinuous:
    def __init__(self, values):
        self.values = values

    def sample(self, n, w_disc, x, z):
        del n, w_disc, z
        return self.values.to(x.device).expand(x.shape[0], -1).clone()


class _TransientSplineFailure(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def sample(self, n, context):
        self.calls += 1
        if self.calls == 1:
            raise AssertionError("simulated float32 inverse roundoff")
        return torch.zeros(context.shape[0], n, 1)


class RecourseObjectiveTests(unittest.TestCase):
    def test_closure_uses_absolute_disparity_and_penalizes_overshoot(self):
        closure = _bootstrap_absolute_closure(
            np.array([0.2, 0.2]), np.array([-0.1, -0.1]), n_boot=20
        )
        self.assertAlmostEqual(closure[0], 0.5)

    def test_eta_and_lambda_select_different_terms(self):
        candidates = [
            {"cost": 0.0, "shortfall": 1.0, "direct_effect": 0.0,
             "transport": 0.0, "p_xi": 0.0, "p_xcf": 0.0},
            {"cost": 0.2, "shortfall": 0.0, "direct_effect": 1.0,
             "transport": 0.0, "p_xi": 0.0, "p_xcf": 1.0},
            {"cost": 0.4, "shortfall": 0.0, "direct_effect": 0.0,
             "transport": 0.0, "p_xi": 0.5, "p_xcf": 0.5},
        ]
        self.assertIs(select_best(candidates, eta=1.0, lambda_invariance=0.0),
                      candidates[1])
        self.assertIs(select_best(candidates, eta=1.0, lambda_invariance=1.0),
                      candidates[2])

    def test_post_disparity_is_reference_minus_post_prediction(self):
        record = {"p_xi": 0.7, "p_xcf": 0.8}
        attach_mediated_disparities(record, reference=0.6,
                                    natural_disadvantaged=0.4,
                                    factual_prediction=0.3)
        self.assertAlmostEqual(
            record["pre_recourse_mediated_prediction_disparity"], 0.2
        )
        self.assertAlmostEqual(
            record["pre_recourse_factual_mediated_prediction_disparity"], 0.3
        )
        self.assertAlmostEqual(
            record["post_recourse_mediated_prediction_disparity"], -0.1
        )

    def test_post_prediction_is_observed_group_for_disadvantaged_code_one(self):
        record = {"p_xi": 0.7, "p_xcf": 0.2}
        attach_mediated_disparities(
            record, reference=0.6, natural_disadvantaged=0.4,
            factual_prediction=0.3, disadvantaged_value=1,
        )
        self.assertAlmostEqual(
            record["post_recourse_mediated_prediction_disparity"], -0.1
        )

    def test_mixed_metric_does_not_treat_category_codes_as_euclidean(self):
        candidates = [{"cost": 0.0, "shortfall": 0.0, "p_xi": 0.4,
                       "p_xcf": 0.5, "delta_cat": 2.0}]
        add_candidate_geometry(
            candidates, factual_wd=np.array([0]), factual_wc=np.empty(0),
            reference_wd=np.array([[1], [2], [2], [3]]),
            reference_wc=np.empty((4, 0)), disc_names=["cat"], cont_names=[],
            mediator_specs={"cat": {"actionable": "free"}},
        )
        self.assertAlmostEqual(candidates[0]["transport"], 0.5)
        self.assertAlmostEqual(candidates[0]["direct_effect"], 0.1)

    def test_stochastic_transport_is_labelled_as_coupling_upper_bound(self):
        candidates = [{"cost": 0.0, "shortfall": 0.0,
                       "p_xi": 0.5, "p_xcf": 0.5}]
        add_sampled_candidate_geometry(
            candidates,
            sampled_wd=np.array([[[0], [1]]]),
            sampled_wc=np.array([[[0.0], [1.0]]]),
            reference_wd=np.array([[0], [1]]),
            reference_wc=np.array([[0.0], [1.0]]),
            disc_names=["cat"], cont_names=["cont"],
            mediator_specs={"cat": {}, "cont": {"clip": [0.0, 1.0]}},
        )
        self.assertAlmostEqual(candidates[0]["transport_marginal_w1"], 0.0)
        self.assertAlmostEqual(candidates[0]["transport_upper_bound"], 0.5)
        self.assertEqual(candidates[0]["transport_estimand"],
                         "independent_coupling_upper_bound")


class MediationTests(unittest.TestCase):
    def test_synthetic_ground_truth_with_interaction(self):
        result = run_synthetic(n=100, K=300, seed=7)
        for effect in ("nde", "nie", "te", "interaction"):
            self.assertLess(result["effects"][effect]["absolute_error"], 0.03)

    def test_autoregressive_discrete_model_shapes(self):
        model = DiscreteMediator(1, 1, {"a": 3, "b": 4}, hidden_dim=8,
                                 n_layers=1, autoregressive=True, embed_dim=2)
        x, z = torch.zeros(5, 1), torch.ones(5, 1)
        sample = model.sample(x, z)
        self.assertEqual(tuple(sample.shape), (5, 2))
        self.assertEqual(tuple(model.log_prob(sample, x, z).shape), (5,))

    def test_conditional_gaussian_baseline_shapes(self):
        model = ConditionalGaussian(1, 1, 1, {"a": 3}, hidden_features=8)
        x, z = torch.zeros(5, 1), torch.ones(5, 1)
        wd = torch.zeros(5, 1, dtype=torch.long)
        sample = model.sample(1, wd, x, z)
        self.assertEqual(tuple(sample.shape), (5, 1))
        self.assertEqual(tuple(model.log_prob(sample, wd, x, z).shape), (5,))

    def test_spline_sampling_isolates_transient_inverse_roundoff(self):
        model = ContinuousMediatorFlow(
            dim_wc=1, dim_x=1, dim_z=1, vocab={}, hidden_features=4,
            n_flow_layers=1, n_blocks=1, num_bins=4,
        )
        transient_flow = _TransientSplineFailure()
        model.flow = transient_flow

        actual = model._sample_context_robust(torch.zeros(4, 2))

        self.assertEqual(tuple(actual.shape), (4, 1, 1))
        self.assertEqual(transient_flow.calls, 3)

    def test_ordered_intervention_resamples_only_strict_descendants(self):
        schema = MediatorSchema.build(
            [["education"], ["occupation"], ["hours"]],
            ["education", "occupation"], ["hours"],
        )
        x, z = torch.zeros(1, 1), torch.zeros(1, 1)
        factual_wd = torch.tensor([[0, 1]])
        factual_wc = torch.tensor([[35.0]])
        samples = sample_intervention_batch(
            _DeterministicDiscrete(), _DeterministicContinuous(torch.tensor([[40.0]])),
            schema, x, z, factual_wd, factual_wc,
            action_w_disc=torch.tensor([[1, 0]]),
            action_w_cont=torch.zeros(1, 1),
            action_mask_disc=torch.tensor([[True, False]]),
            action_mask_cont=torch.tensor([[False]]),
            n_samples=3,
        )
        self.assertTrue(torch.equal(samples.w_disc, torch.tensor([[1, 0]]).repeat(3, 1)))
        self.assertTrue(torch.equal(samples.w_cont, torch.tensor([[40.0]]).repeat(3, 1)))

    def test_same_level_joint_block_has_no_implied_causal_order(self):
        schema = MediatorSchema.build([["lsat", "gpa"]], [], ["lsat", "gpa"])
        empty_wd = torch.empty(1, 0, dtype=torch.long)
        factual_wc = torch.tensor([[0.2, -0.7]])
        common = dict(
            g_phi=_DeterministicDiscrete(),
            f_theta=_DeterministicContinuous(torch.tensor([[1.5, 2.5]])),
            schema=schema,
            x=torch.zeros(1, 1), z=torch.zeros(1, 2),
            factual_w_disc=empty_wd, factual_w_cont=factual_wc,
            action_w_disc=empty_wd,
            action_w_cont=torch.tensor([[0.9, 0.0]]),
            action_mask_disc=torch.empty(1, 0, dtype=torch.bool),
            action_mask_cont=torch.tensor([[True, False]]),
        )
        individual = sample_intervention_batch(**common, same_level="preserve_factual")
        population = sample_intervention_batch(**common, same_level="resample_marginal")
        self.assertTrue(torch.equal(individual.w_cont, torch.tensor([[0.9, -0.7]])))
        self.assertTrue(torch.equal(population.w_cont, torch.tensor([[0.9, 2.5]])))


if __name__ == "__main__":
    unittest.main()
