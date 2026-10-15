import copy
from io import BytesIO
import unittest

import numpy as np
import pandas as pd

from inverse_analysis import recommendation_parameters
from sensitivity_analysis import (
    MATERIAL, OUTPUTS, STUDIES, default_ranges, evaluate_samples, excel_report,
    material_ranges, reference_baseline, run_analysis,
)


class SensitivityMethodsTests(unittest.TestCase):
    def setUp(self):
        self.p = reference_baseline(STUDIES[1])
        self.diffusion = {"mode": "Fixed D"}

    def test_material_bounds_follow_each_nominal_and_keep_units(self):
        original = copy.deepcopy(self.p)
        first = material_ranges(MATERIAL, self.p).set_index("Parameter")
        second = material_ranges(MATERIAL, dict(self.p, E=self.p["E"] * 2)).set_index("Parameter")
        self.assertEqual(self.p, original)
        self.assertEqual(first.loc["E", "Unit"], "MPa")
        np.testing.assert_allclose(first.loc["D", ["Lower", "Upper"]].astype(float),
                                   [self.p["D"] / 2, self.p["D"] * 2])
        np.testing.assert_allclose(second.loc["E", ["Lower", "Upper"]].astype(float),
                                   first.loc["E", ["Lower", "Upper"]].astype(float) * 2)
        self.assertEqual(first.loc["D", "Distribution"], "log-uniform")
        self.assertEqual(first.loc["Csat", "Distribution"], "uniform")

    def test_material_bounds_handle_negative_cte_and_near_limit_poisson_ratio(self):
        table = material_ranges(["alpha_lam", "nu"], dict(self.p, alpha_lam=-3., nu=.49)).set_index("Parameter")
        np.testing.assert_allclose(table.loc["alpha_lam", ["Lower", "Upper"]].astype(float), [-3.3, -2.7])
        self.assertLess(table.loc["nu", "Upper"], .5)
        with self.assertRaisesRegex(ValueError, "zero"):
            material_ranges(["D"], dict(self.p, D=0))

    def test_morris_recovers_analytic_linear_water_effect_and_ignores_modulus(self):
        ranges = material_ranges(["Csat", "E"], self.p).to_dict("records")
        result = run_analysis(self.p, ranges, self.diffusion, "W_t", method="Morris", trajectories=40)
        table = result["indices"].set_index("Parameter")
        uptake = float(evaluate_samples(self.p, self.diffusion)["Mt_Minf"])
        expected = uptake * .018015 * (ranges[0]["Upper"] - ranges[0]["Lower"])
        self.assertAlmostEqual(table.loc["Csat", "mu_star"], expected)
        self.assertAlmostEqual(table.loc["Csat", "mu"], expected)
        self.assertAlmostEqual(table.loc["Csat", "sigma"], 0)
        self.assertEqual(table.loc["E", "mu_star"], 0)
        samples = result["samples"]
        self.assertEqual(len(samples), 40 * 3)
        for _, path in samples.groupby("Trajectory"):
            changes = np.diff(path[["Csat", "E"]].to_numpy(), axis=0)
            np.testing.assert_array_equal((np.abs(changes) > 0).sum(axis=1), [1, 1])

    def test_morris_is_reproducible_and_reports_nonlinear_effect_variation(self):
        ranges = material_ranges(["D", "Csat", "beta"], self.p).to_dict("records")
        one = run_analysis(self.p, ranges, self.diffusion, "eps_h", method="Morris", seed=42)
        two = run_analysis(self.p, ranges, self.diffusion, "eps_h", method="Morris", seed=42)
        pd.testing.assert_frame_equal(one["indices"], two["indices"])
        self.assertTrue((one["indices"]["sigma"] > 0).all())
        with self.assertRaisesRegex(ValueError, "total-effect"):
            recommendation_parameters(dict(config=dict(method="Morris"), result=one))

    def test_additional_responses_and_morris_export_have_correct_meaning(self):
        self.assertIn("eps_h", OUTPUTS)
        self.assertIn("W_t", OUTPUTS)
        ranges = default_ranges(["Csat"]).to_dict("records")
        for target in ("W_t", "eps_h"):
            result = run_analysis(self.p, ranges, self.diffusion, target, N=512)
            np.testing.assert_allclose(result["indices"][["S1", "ST"]], 1, atol=.02)
        config = dict(baseline=self.p, ranges=ranges, diffusion=self.diffusion,
                      target="W_t", method="Morris", trajectories=40)
        result = run_analysis(self.p, ranges, self.diffusion, "W_t", method="Morris")
        workbook = pd.ExcelFile(BytesIO(excel_report(result, config)))
        self.assertIn("Morris effects", workbook.sheet_names)
        self.assertNotIn("Sobol S1 ST", workbook.sheet_names)
        nominal = workbook.parse("Nominal inputs").set_index("Parameter")
        self.assertEqual(nominal.loc["Csat", "Value"], self.p["Csat"])
        notes = workbook.parse("How to read")["How to read"].str.cat(sep=" ")
        self.assertIn("normalized input", notes)
        self.assertNotIn("bootstrap", notes)

    def test_both_methods_validate_the_same_input_ranges(self):
        for method in ("Sobol", "Morris"):
            with self.subTest(method=method):
                with self.assertRaisesRegex(ValueError, "Poisson's ratio"):
                    run_analysis(self.p, [dict(Parameter="nu", Lower=.1, Upper=.5, Distribution="uniform")],
                                 self.diffusion, method=method, N=512)
                with self.assertRaisesRegex(ValueError, "temperature"):
                    run_analysis(self.p, default_ranges(["D"]).to_dict("records"),
                                 {"mode": "Arrhenius"}, method=method, N=512)


if __name__ == "__main__":
    unittest.main()
