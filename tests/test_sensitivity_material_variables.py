import unittest

import numpy as np

from sensitivity_analysis import (
    MATERIAL, STUDIES, default_ranges, evaluate_samples, reference_baseline,
    run_analysis, saltelli_samples,
)


class PoissonRatioSensitivityTests(unittest.TestCase):
    def test_poisson_ratio_changes_stress_without_changing_free_strain(self):
        baseline = reference_baseline(STUDIES[1])
        output = evaluate_samples(dict(baseline, nu=np.array([0.18, 0.30])), {"mode": "Fixed D"})
        for quantity in ("Mt_Minf", "eps_h", "eps_t"):
            self.assertEqual(output[quantity][0], output[quantity][1])
        self.assertGreater(output["sigma_abs"][1], output["sigma_abs"][0])
        self.assertAlmostEqual(output["sigma_abs"][1] / output["sigma_abs"][0], 0.82 / 0.70)

    def test_poisson_ratio_can_be_the_only_varied_material_input(self):
        self.assertIn("nu", MATERIAL)
        result = run_analysis(reference_baseline(STUDIES[1]),
                              default_ranges(["nu"]).to_dict("records"),
                              {"mode": "Fixed D"}, N=2048)
        self.assertEqual(result["indices"]["Parameter"].tolist(), ["nu"])
        np.testing.assert_allclose(result["indices"][["S1", "ST"]], 1.0, atol=0.01)
        self.assertGreater(result["samples"]["nu"].std(), 0)

    def test_sampling_rejects_singular_or_nonphysical_poisson_ratio_bounds(self):
        for lower, upper in ((-1.0, 0.3), (0.2, 0.5), (0.5, 0.6)):
            with self.subTest(lower=lower, upper=upper):
                with self.assertRaisesRegex(ValueError, "Poisson's ratio bounds"):
                    saltelli_samples([dict(Parameter="nu", Lower=lower, Upper=upper,
                                           Distribution="uniform")], 8)


if __name__ == "__main__":
    unittest.main()
