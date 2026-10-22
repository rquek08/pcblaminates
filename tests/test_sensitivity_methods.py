import copy
from io import BytesIO
import unittest

import numpy as np
import pandas as pd

from sensitivity_analysis import (
    MATERIAL, OUTPUTS, STUDIES, default_ranges, evaluate_samples, excel_report,
    material_ranges, reference_baseline, run_analysis,
)


class SobolSensitivityTests(unittest.TestCase):
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

    def test_moisture_response_identifies_capacity_and_ignores_thermal_expansion(self):
        ranges = material_ranges(["Csat", "alpha_lam"], self.p).to_dict("records")
        result = run_analysis(self.p, ranges, self.diffusion, "sigma_hygro", N=1024)
        table = result["indices"].set_index("Parameter")
        np.testing.assert_allclose(table.loc["Csat", ["S1", "ST"]].astype(float), 1, atol=.01)
        np.testing.assert_allclose(table.loc["alpha_lam", ["S1", "ST"]].astype(float), 0, atol=1e-12)
        self.assertEqual(len(result["samples"]), 1024 * 4)
        self.assertEqual(set(result["samples"]["Block"]), {"A", "B", "AB_Csat", "AB_alpha_lam"})

    def test_thermal_response_ignores_hygroscopic_expansion(self):
        ranges = material_ranges(["alpha_lam", "beta"], self.p).to_dict("records")
        result = run_analysis(self.p, ranges, self.diffusion, "sigma_thermal", N=1024)
        table = result["indices"].set_index("Parameter")
        np.testing.assert_allclose(table.loc["alpha_lam", ["S1", "ST"]].astype(float), 1, atol=.01)
        np.testing.assert_allclose(table.loc["beta", ["S1", "ST"]].astype(float), 0, atol=1e-12)

    def test_default_signed_response_matches_analytic_additive_variance(self):
        ranges = [dict(Parameter="Csat", Lower=0., Upper=400., Distribution="uniform"),
                  dict(Parameter="alpha_lam", Lower=8., Upper=23., Distribution="uniform")]
        result = run_analysis(self.p, ranges, self.diffusion, N=2048)
        # With other inputs fixed, signed stress is linear in both inputs.
        # Independent uniform contributions have variance (slope * width)²/12.
        uptake = float(evaluate_samples(self.p, self.diffusion)["Mt_Minf"])
        modulus = self.p["E"] / (1 - self.p["nu"])
        moisture_variance = (modulus * self.p["beta"] * uptake * .018015 * 400)**2 / 12
        thermal_variance = (modulus * 1e-6 * self.p["delta_T"] * 15)**2 / 12
        expected = np.array([moisture_variance, thermal_variance]) / (moisture_variance + thermal_variance)
        table = result["indices"]
        np.testing.assert_allclose(table["S1"], expected, atol=.01)
        np.testing.assert_allclose(table["ST"], expected, atol=.01)
        response = result["samples"]["sigma_index"]
        self.assertLess(response.min(), 0)
        self.assertGreater(response.max(), 0)

    def test_sobol_samples_indices_and_intervals_are_reproducible(self):
        ranges = material_ranges(["D", "Csat", "beta"], self.p).to_dict("records")
        one = run_analysis(self.p, ranges, self.diffusion, "sigma_index", N=512, seed=42)
        two = run_analysis(self.p, ranges, self.diffusion, "sigma_index", N=512, seed=42)
        pd.testing.assert_frame_equal(one["indices"], two["indices"])
        pd.testing.assert_frame_equal(one["samples"], two["samples"])
        self.assertTrue(np.isfinite(one["indices"][["S1_low", "S1_high", "ST_low", "ST_high"]]).all().all())

    def test_three_stress_responses_and_sobol_export(self):
        self.assertEqual(list(OUTPUTS), ["sigma_index", "sigma_hygro", "sigma_thermal"])
        self.assertEqual(OUTPUTS["sigma_index"][0], "Stress index")
        for target, parameter in (("sigma_index", "Csat"), ("sigma_hygro", "Csat"),
                                  ("sigma_thermal", "alpha_lam")):
            ranges = default_ranges([parameter]).to_dict("records")
            result = run_analysis(self.p, ranges, self.diffusion, target, N=512)
            np.testing.assert_allclose(result["indices"][["S1", "ST"]], 1, atol=.02)
        config = dict(baseline=self.p, ranges=ranges, diffusion=self.diffusion,
                      target="sigma_thermal", method="Sobol", N=512, seed=12345)
        workbook = pd.ExcelFile(BytesIO(excel_report(result, config)))
        self.assertIn("Sobol S1 ST", workbook.sheet_names)
        nominal = workbook.parse("Nominal inputs").set_index("Parameter")
        self.assertEqual(nominal.loc["Csat", "Value"], self.p["Csat"])
        notes = workbook.parse("How to read")["How to read"].str.cat(sep=" ")
        self.assertIn("bootstrap", notes)
        self.assertIn("Saltelli", notes)
        samples = workbook.parse("Samples")
        self.assertIn("Block", samples)
        self.assertTrue({"W_t", "eps_h", "Mt_Minf", "sigma_abs"}.issubset(samples.columns))

    def test_intermediate_outputs_are_not_sensitivity_responses(self):
        ranges = default_ranges(["Csat"]).to_dict("records")
        for target in ("sigma_abs", "Mt_Minf", "W_t", "eps_h"):
            with self.subTest(target=target):
                with self.assertRaisesRegex(ValueError, "supported sensitivity output"):
                    run_analysis(self.p, ranges, self.diffusion, target, N=512)

    def test_sobol_validates_input_ranges_and_diffusion_mode(self):
        with self.assertRaisesRegex(ValueError, "Poisson's ratio"):
            run_analysis(self.p, [dict(Parameter="nu", Lower=.1, Upper=.5, Distribution="uniform")],
                         self.diffusion, N=512)
        with self.assertRaisesRegex(ValueError, "temperature"):
            run_analysis(self.p, default_ranges(["D"]).to_dict("records"),
                         {"mode": "Arrhenius"}, N=512)


if __name__ == "__main__":
    unittest.main()
