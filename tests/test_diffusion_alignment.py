"""Check one-sided diffusion and agreement between the three app calculations."""
import ast
import json
from pathlib import Path
import unittest

import numpy as np

from inverse_analysis import APPLICATION_DEFAULTS, recommend_materials
from sensitivity_analysis import STUDIES, evaluate_samples, reference_baseline


ROOT = Path(__file__).resolve().parents[1]


class DiffusionAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the calculator's numerical functions without rendering its Streamlit UI.
        tree = ast.parse((ROOT / "app.py").read_text())
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in {"compute_fickian_uptake", "run_analytical_stress_pipeline"}]
        constants = [node for node in tree.body if isinstance(node, ast.Assign)
                     and any(isinstance(target, ast.Name) and target.id == "MOLAR_MASS_WATER"
                             for target in node.targets)]
        cls.calculator = {"np": np}
        exec(compile(ast.Module(body=constants + functions, type_ignores=[]), "app.py", "exec"), cls.calculator)

    def calculator_result(self, inputs):
        return self.calculator["run_analytical_stress_pipeline"](
            inputs["D"], inputs["Csat"], inputs["h_mm"] / 1000, inputs["t_hours"] * 3600,
            inputs["beta"], inputs["alpha_lam"], inputs["alpha_cu"], inputs["delta_T"],
            inputs["E"], inputs["nu"],
        )

    def test_early_uptake_matches_one_exposed_face(self):
        inputs = dict(reference_baseline(STUDIES[1]), D=1e-12, h_mm=1., t_hours=100 / 3600)
        result = evaluate_samples(inputs, {"mode": "Fixed D"})
        expected = 2 * np.sqrt(inputs["D"] * 100 / np.pi) / .001
        self.assertAlmostEqual(float(result["Mt_Minf"]), expected)
        self.assertAlmostEqual(float(result["Fo"]), inputs["D"] * 100 / (.002**2))

    def test_both_branches_and_saturation_match_calculator(self):
        inputs = reference_baseline(STUDIES[1])
        fourier = np.array([0., 1e-7, .001, .00499, .00501, .1, 10.])
        seconds = fourier * (2 * inputs["h_mm"] / 1000)**2 / inputs["D"]
        inputs["t_hours"] = seconds / 3600
        actual = evaluate_samples(inputs, {"mode": "Fixed D"})
        expected = self.calculator_result(inputs)
        for key in expected.keys() & actual.keys():
            np.testing.assert_allclose(actual[key], expected[key], rtol=1e-13, atol=1e-14, err_msg=key)
        self.assertEqual(actual["Mt_Minf"][0], 0.)
        self.assertAlmostEqual(actual["Mt_Minf"][-1], 1.)

    def test_arrhenius_uses_the_same_one_sided_boundary(self):
        inputs = dict(reference_baseline(STUDIES[0]), delta_T=np.array([0., 30., 60.]))
        diffusion = dict(mode="Arrhenius", D0=6.31e-7, Ea=37560., T_ref_C=25.)
        result = evaluate_samples(inputs, diffusion)
        inputs["D"] = diffusion["D0"] * np.exp(-diffusion["Ea"] / (8.314 * (298.15 + inputs["delta_T"])))
        expected = [self.calculator_result(dict(inputs, D=float(d), delta_T=float(dt)))
                    for d, dt in zip(inputs["D"], inputs["delta_T"])]
        for key in expected[0].keys() & result.keys():
            np.testing.assert_allclose(result[key], [row[key] for row in expected],
                                       rtol=1e-13, atol=1e-14, err_msg=key)

    def test_recommender_results_match_calculator_for_real_presets(self):
        records = json.loads((ROOT / "material_presets.json").read_text())["presets"]
        result = recommend_materials(records, "85°C/85%RH", APPLICATION_DEFAULTS)
        self.assertFalse(result["candidates"].empty)
        for row in result["candidates"].to_dict("records"):
            expected = self.calculator_result(row)
            for key in ("Mt_Minf", "W_t", "eps_h", "sigma_hygro", "sigma_thermal", "sigma_index"):
                self.assertAlmostEqual(row[key], float(expected[key]), places=12, msg=f"{row['name']}: {key}")


if __name__ == "__main__":
    unittest.main()
