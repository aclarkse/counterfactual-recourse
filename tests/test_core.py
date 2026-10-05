import unittest

import numpy as np
import pandas as pd
import torch
from scipy.stats import wasserstein_distance

from data.adult import (
    VOCAB_ADULT, _OCCUPATION_CODES, _bucket_education, _bucket_hours,
)
from data.german import (
    PURPOSE_COLUMNS,
    SFM_CONFIG_GERMAN,
    VOCAB_GERMAN,
    _CHECKING_CODES,
    _FEMALE_PERSONAL_STATUS_CODES,
    _HOUSING_CODES,
    _PURPOSE_CODES,
    _SAVINGS_CODES,
)
from evaluation.recourse_metrics import (
    add_candidate_geometry,
    add_sampled_candidate_geometry,
    attach_mediated_disparities,
    select_best,
    select_distribution_constrained,
    select_validity_constrained,
)
from evaluation.compute_recourse import _bootstrap_absolute_closure
from evaluation.ablate_recourse import (
    compatibility_threshold,
    compute_constraint_diagnostics,
)
from evaluation.synthetic_validation import run as run_synthetic
from flows.interventions import sample_intervention_batch
from flows.models import (
    ConditionalGaussian,
    ContinuousMediatorFlow,
    DiscreteMediator,
)
from flows.schema import MediatorSchema
from flows.posterior_predictive import mediator_posterior_predictive


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

class _EmptyDiscrete(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    def sample(self, x, z):
        del z
        return torch.empty(len(x), 0, dtype=torch.long, device=x.device)


class _ContextContinuous:
    def sample(self, n, w_disc, x, z):
        del n, w_disc, x
        return z[:, :2]


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

    def test_distribution_constraint_selects_minimum_cost_feasible_plan(self):
        candidates = [
            {"cost": 0.0, "success_probability_xi": 0.2,
             "outcome_wasserstein": 0.02},
            {"cost": 0.4, "success_probability_xi": 0.9,
             "outcome_wasserstein": 0.08},
            {"cost": 0.7, "success_probability_xi": 1.0,
             "outcome_wasserstein": 0.04},
        ]
        selected = select_distribution_constrained(candidates, 0.8, 0.1)
        self.assertEqual(selected["cost"], 0.4)
        self.assertTrue(selected["constraint_feasible"])
        self.assertFalse(selected["constraint_abstained"])

    def test_validity_only_baseline_uses_the_same_chance_constraint(self):
        candidates = [
            {"cost": 0.0, "success_probability_xi": 0.2},
            {"cost": 0.4, "success_probability_xi": 0.9},
            {"cost": 0.7, "success_probability_xi": 1.0},
        ]
        selected = select_validity_constrained(candidates, 0.8)
        self.assertEqual(selected["cost"], 0.4)
        self.assertTrue(selected["constraint_feasible"])

    def test_constraint_diagnostics_separate_reference_compatibility(self):
        candidates = [
            [
                {"success_probability_xi": 0.9, "outcome_wasserstein": 0.04},
                {"success_probability_xi": 0.2, "outcome_wasserstein": 0.01},
            ],
            [
                {"success_probability_xi": 1.0, "outcome_wasserstein": 0.20},
            ],
        ]
        diagnostics = compute_constraint_diagnostics(
            candidates,
            np.asarray([[0.6, 0.7], [0.1, 0.2]]),
            threshold=0.5,
            min_success_probability=0.8,
            epsilon_grid=[0.1, 0.3],
        )
        self.assertEqual(diagnostics["reference_validity_rate"], 0.5)
        self.assertEqual(diagnostics["validity_only_coverage"], 1.0)
        self.assertAlmostEqual(
            diagnostics[
                "minimum_attainable_w1_among_valid_actions"]["median"],
            0.12,
        )
        at_point_one = diagnostics["coverage_vs_epsilon"][0]
        self.assertEqual(at_point_one["coverage"], 0.5)
        self.assertEqual(at_point_one["coverage_reference_valid"], 1.0)
        self.assertEqual(at_point_one["coverage_reference_invalid"], 0.0)

    def test_distribution_constraint_abstains_with_factual_plan(self):
        candidates = [
            {"cost": 0.0, "success_probability_xi": 0.2,
             "outcome_wasserstein": 0.02},
            {"cost": 0.4, "success_probability_xi": 0.9,
             "outcome_wasserstein": 0.2},
        ]
        selected = select_distribution_constrained(candidates, 0.8, 0.1)
        self.assertEqual(selected["cost"], 0.0)
        self.assertFalse(selected["constraint_feasible"])
        self.assertTrue(selected["constraint_abstained"])

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

    def test_mediator_context_standardizes_only_configured_confounder(self):
        model = DiscreteMediator(
            1, 2, {"a": 2}, hidden_dim=8, n_layers=1,
            z_mean=[40.0, 0.0], z_scale=[10.0, 1.0],
        )
        context = model._cond(torch.tensor([[1.0]]), torch.tensor([[50.0, 1.0]]))
        torch.testing.assert_close(context, torch.tensor([[1.0, 1.0, 1.0]]))

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

    def test_same_level_discrete_block_uses_statistical_factorization_only(self):
        schema = MediatorSchema.build(
            [["checking", "savings", "housing"], ["amount", "duration"]],
            ["checking", "savings", "housing"], ["amount", "duration"],
        )
        model = DiscreteMediator(
            1, 1, {"checking": 4, "savings": 5, "housing": 3},
            autoregressive=True, layers=schema.discrete_layers,
            within_block_autoregressive=True,
        )
        self.assertEqual(model.parent_indices, [[], [0], [0, 1]])
        self.assertEqual(schema.descendants_of({"checking"}), {"amount", "duration"})

    def test_continuous_mechanism_can_exclude_sensitive_attribute(self):
        model = ConditionalGaussian(
            2, 1, 1, {"status": 3}, hidden_features=8,
            condition_on_x=False,
        )
        wd = torch.tensor([[1], [1]])
        z = torch.tensor([[0.5], [0.5]])
        context = model._context(wd, torch.tensor([[0.0], [1.0]]), z)
        torch.testing.assert_close(context[0], context[1])

    def test_continuous_posterior_predictive_uses_fixed_contexts(self):
        x = torch.tensor([[0.0], [0.0], [1.0], [1.0]])
        z = torch.tensor([
            [1.0, 2.0], [2.0, 4.0], [3.0, 6.0], [4.0, 8.0],
        ])
        diagnostics = mediator_posterior_predictive(
            _EmptyDiscrete(), _ContextContinuous(), x, z,
            torch.empty(4, 0, dtype=torch.long), z.clone(), {},
            ["hours", "weeks"], samples_per_context=1, max_rows=4,
            min_group_rows=2, seed=3,
        )

        self.assertLess(diagnostics["max_overall_continuous_w1"], 0.1)
        self.assertLess(diagnostics["max_group_continuous_w1"], 0.1)
        self.assertLess(
            diagnostics["max_continuous_correlation_error"], 1e-6
        )

class AdultDataTests(unittest.TestCase):
    def test_education_bucketing_matches_acs_six_level_structure(self):
        values = pd.Series([1, 8, 9, 10, 12, 13, 14, 15, 16])
        actual = _bucket_education(values).tolist()
        self.assertEqual(actual, [0, 0, 1, 2, 2, 3, 4, 5, 5])

    def test_hours_binning_is_ordered_and_isolates_forty_hours(self):
        values = pd.Series(
            [1, 19, 20, 29, 30, 34, 35, 39, 40, 41, 49, 50, 59, 60, 99]
        )
        expected = [0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 6, 7, 7]
        self.assertEqual(_bucket_hours(values).tolist(), expected)

    def test_occupation_encoding_is_complete_and_contiguous(self):
        self.assertEqual(len(_OCCUPATION_CODES), 14)
        self.assertEqual(sorted(_OCCUPATION_CODES.values()), list(range(14)))
        self.assertEqual(VOCAB_ADULT["OCCUPATION_GRP"], 14)
        self.assertEqual(VOCAB_ADULT["HOURS_GRP"], 8)


class GermanCreditDataTests(unittest.TestCase):
    def test_sensitive_sex_codes_match_uci_documentation(self):
        self.assertEqual(_FEMALE_PERSONAL_STATUS_CODES, {"A92", "A95"})

    def test_mediator_encodings_are_complete_and_contiguous(self):
        for mapping, size in (
            (_SAVINGS_CODES, 5),
            (_CHECKING_CODES, 4),
            (_HOUSING_CODES, 3),
            (_PURPOSE_CODES, 10),
        ):
            self.assertEqual(sorted(mapping.values()), list(range(size)))
        self.assertEqual(VOCAB_GERMAN, {
            "CHECKING_GRP": 4,
            "SAVINGS_GRP": 5,
            "HOUSING_GRP": 3,
        })

    def test_acsr_roles_include_purpose_and_repayment_terms(self):
        self.assertEqual(len(PURPOSE_COLUMNS), 10)
        self.assertEqual(
            SFM_CONFIG_GERMAN["confounders"], ["AGE", *PURPOSE_COLUMNS]
        )
        self.assertEqual(
            SFM_CONFIG_GERMAN["mediators_cont"], ["CREDIT_AMOUNT", "DURATION"]
        )



class CompatibilityThresholdTests(unittest.TestCase):
    def test_zero_when_reference_is_valid(self):
        reference = np.array([[0.6, 0.7, 0.8, 0.9]])
        self.assertEqual(compatibility_threshold(reference, 0.5, 0.75)[0], 0.0)

    def test_closed_form_on_step_quantile(self):
        # Upper half of the quantile function sits at 0.2 and 0.4; lifting
        # it to tau = 0.5 costs (0.3 + 0.1) / 4.
        reference = np.array([[0.0, 0.1, 0.2, 0.4]])
        self.assertAlmostEqual(
            compatibility_threshold(reference, 0.5, 0.5)[0], 0.1)

    def test_lower_bounds_every_valid_plan(self):
        rng = np.random.default_rng(0)
        reference = rng.beta(2.0, 4.0, size=(1, 200))
        bound = compatibility_threshold(reference, 0.5, 0.8)[0]
        for _ in range(100):
            valid = np.concatenate([rng.uniform(0.5, 1.0, 26),
                                    rng.uniform(0.0, 1.0, 6)])
            self.assertGreaterEqual(
                wasserstein_distance(valid, reference[0]), bound - 1e-12)


if __name__ == "__main__":
    unittest.main()
