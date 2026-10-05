import unittest

import numpy as np
import yaml

from data.german_synth import (
    AMOUNT_RANGE,
    DURATION_RANGE,
    SFM_CONFIG_GERMAN_SYNTH,
    VOCAB_GERMAN_SYNTH,
    make_oracle_sample_fn,
    oracle_mediation_effects,
    sample_german_synthetic,
)
from flows.schema import MediatorSchema


class GermanSyntheticSCMTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = sample_german_synthetic(20000, seed=0)
        cls.effects = oracle_mediation_effects(n_samples=100_000, seed=1)

    def test_sampling_is_reproducible(self):
        again = sample_german_synthetic(20000, seed=0)
        self.assertTrue(self.df.equals(again))

    def test_columns_and_ranges(self):
        cfg = SFM_CONFIG_GERMAN_SYNTH
        expected = (cfg["sensitive"] + cfg["confounders"]
                    + cfg["mediators_disc"] + cfg["mediators_cont"]
                    + [cfg["outcome"]])
        self.assertEqual(sorted(self.df.columns), sorted(expected))
        for name, size in VOCAB_GERMAN_SYNTH.items():
            self.assertEqual(set(self.df[name].unique()), set(range(size)))
        self.assertTrue(self.df["CREDIT_AMOUNT"].between(*AMOUNT_RANGE).all())
        self.assertTrue(self.df["DURATION"].between(*DURATION_RANGE).all())
        self.assertTrue(self.df["AGE"].between(19, 75).all())
        self.assertAlmostEqual(self.df["GOOD_CREDIT"].mean(), 0.7, delta=0.05)

    def test_mediated_effect_dominates(self):
        e = self.effects
        self.assertGreater(e["nie"], 0.15)
        self.assertLess(e["nde"], 0.08)
        self.assertGreater(e["nde"], 0.0)
        self.assertGreater(e["nie"] / e["te"], 0.7)

    def test_config_layers_match_generating_scm(self):
        with open("conf/dataset/german_synth.yaml") as file:
            sfm = yaml.safe_load(file)["sfm"]
        for key, value in SFM_CONFIG_GERMAN_SYNTH.items():
            self.assertEqual(sfm[key], value)
        schema = MediatorSchema.from_sfm_config(sfm)
        self.assertEqual(
            schema.layers,
            (("EDUCATION",), ("JOB",), ("SAVINGS_GRP",),
             ("CREDIT_AMOUNT", "DURATION")),
        )

    def test_sex_shifts_every_mediator_in_same_direction(self):
        means = self.df.groupby("SEX")[
            ["EDUCATION", "JOB", "SAVINGS_GRP"]].mean()
        self.assertTrue(np.all(means.loc[1] > means.loc[0]))

    def test_unit_mediation_scale_is_the_default(self):
        scaled = sample_german_synthetic(20000, seed=0, mediation_scale=1.0)
        self.assertTrue(self.df.equals(scaled))

    def test_mediation_scale_controls_indirect_effect_only(self):
        effects = [oracle_mediation_effects(n_samples=100_000, seed=1,
                                            mediation_scale=scale)
                   for scale in (0.0, 0.5, 1.0, 2.0)]
        self.assertLess(abs(effects[0]["nie"]), 0.01)
        nies = [e["nie"] for e in effects]
        self.assertTrue(all(a < b for a, b in zip(nies, nies[1:])))
        for e in effects[1:]:
            self.assertAlmostEqual(e["nde"], effects[0]["nde"], places=3)

    def test_zero_mediation_scale_removes_sex_mediator_shift(self):
        df = sample_german_synthetic(50000, seed=0, mediation_scale=0.0)
        means = df.groupby("SEX")[["EDUCATION", "JOB", "SAVINGS_GRP"]].mean()
        self.assertTrue(np.all(np.abs(means.loc[1] - means.loc[0]) < 0.03))

    def test_oracle_sample_fn_matches_flow_sampler_interface(self):
        import torch
        from sklearn.preprocessing import QuantileTransformer

        scaler = QuantileTransformer(output_distribution="normal",
                                     random_state=0)
        scaler.fit(self.df[["CREDIT_AMOUNT", "DURATION"]].to_numpy())
        sample_fn = make_oracle_sample_fn(scaler, seed=0)
        x = torch.tensor([[0.0], [1.0]])
        z = torch.tensor([[30.0], [50.0]])
        draws = sample_fn(x, z, K=7)
        self.assertEqual(tuple(draws["w_disc"].shape), (14, 3))
        self.assertEqual(draws["w_disc"].dtype, torch.long)
        self.assertEqual(tuple(draws["w_cont"].shape), (14, 2))
        self.assertTrue(torch.equal(draws["z_rep"][:7, 0],
                                    torch.full((7,), 30.0)))
        back = scaler.inverse_transform(draws["w_cont"].numpy())
        np.testing.assert_allclose(back, draws["w_cont_orig"].numpy(),
                                   rtol=1e-2)


if __name__ == "__main__":
    unittest.main()
