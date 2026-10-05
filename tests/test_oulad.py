import unittest

import pandas as pd
import torch

from data.oulad import (
    SFM_CONFIG_OULAD,
    Z_COLUMNS_OULAD,
    _imd_values,
    _one_hot,
)
from flows.models import ConditionalGaussian, DiscreteMediator
from flows.posterior_predictive import mediator_posterior_predictive


class OULADDataTests(unittest.TestCase):
    def test_imd_midpoints_and_missing_indicator(self):
        midpoint, missing = _imd_values(pd.Series(["0-10%", "90-100%", "?"]))
        self.assertEqual(midpoint.tolist(), [5.0, 95.0, 45.0])
        self.assertEqual(missing.tolist(), [0, 0, 1])

    def test_one_hot_uses_stable_declared_categories(self):
        encoded = _one_hot(
            pd.Series(["b", "a", "b"]), ("a", "b", "c"),
            ("A", "B", "C"),
        )
        self.assertEqual(encoded.columns.tolist(), ["A", "B", "C"])
        self.assertEqual(encoded.sum().tolist(), [1.0, 2.0, 0.0])

    def test_sfm_roles_use_joint_continuous_engagement_block(self):
        self.assertEqual(len(Z_COLUMNS_OULAD), 33)
        self.assertEqual(SFM_CONFIG_OULAD["mediators_disc"], [])
        self.assertEqual(
            SFM_CONFIG_OULAD["mediators_cont"],
            ["LOG_CLICKS_60", "ACTIVE_DAYS_60", "RESOURCES_60"],
        )


class AllContinuousPosteriorPredictiveTests(unittest.TestCase):
    def test_diagnostics_do_not_require_discrete_model_parameters(self):
        x = torch.tensor([[0.0], [1.0], [0.0], [1.0]])
        z = torch.zeros(4, 1)
        w_disc = torch.empty(4, 0, dtype=torch.long)
        w_cont = torch.randn(4, 2)
        discrete = DiscreteMediator(1, 1, {}, layers=())
        continuous = ConditionalGaussian(2, 1, 1, {})
        diagnostics = mediator_posterior_predictive(
            discrete, continuous, x, z, w_disc, w_cont, {}, ("a", "b"),
            samples_per_context=2, max_rows=4, min_group_rows=1,
        )
        self.assertIn("max_overall_continuous_w1", diagnostics)


if __name__ == "__main__":
    unittest.main()
