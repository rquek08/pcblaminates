import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from inverse_analysis import recommendation_parameters, recommend_materials, records_from_csv
from sensitivity_analysis import MATERIAL, STUDIES, default_ranges, evaluate_samples, reference_baseline, run_analysis


class MaterialRecommenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = json.loads((Path(__file__).parents[1] / "material_presets.json").read_text())["presets"]
        cls.config = dict(baseline=reference_baseline(STUDIES[1]), ranges=default_ranges(MATERIAL).to_dict("records"),
                          diffusion={"mode": "Fixed D"}, target="sigma_abs", N=512, seed=12345)
        cls.saved = dict(config=cls.config, result=run_analysis(cls.config["baseline"], cls.config["ranges"],
                         cls.config["diffusion"], cls.config["target"], cls.config["N"], cls.config["seed"]))
        cls.condition = "85°C/85%RH"

    def test_parameters_are_top_three_material_total_effects(self):
        table = recommendation_parameters(self.saved)
        expected = self.saved["result"]["indices"].nlargest(3, "ST")["Parameter"].tolist()
        self.assertEqual(table["Parameter"].tolist(), expected)
        self.assertAlmostEqual(table["Screening weight"].sum(), 1.)

    def test_large_library_evaluates_only_ten_and_ranks_verified_magnitude(self):
        source = next(r for r in self.records if r["condition"] == self.condition)
        records = [dict(source, name=f"Material {i}") for i in range(200)]
        # Put this library within all sampled bounds, including diffusivity.
        records = [dict(r, properties=dict(r["properties"], D=1e-12)) for r in records]
        with patch("inverse_analysis.evaluate_samples", wraps=evaluate_samples) as model:
            result = recommend_materials(records, self.condition, self.saved)
        model.assert_called_once()
        self.assertEqual(len(model.call_args.args[0]["E"]), 10)
        self.assertEqual(result["total"], 200)
        self.assertEqual(result["evaluated"], 10)
        table = result["candidates"]
        self.assertTrue(table["sigma_abs"].is_monotonic_increasing)
        np.testing.assert_allclose(table["sigma_abs"], np.abs(table["sigma_index"]))

    def test_verification_uses_all_material_properties_and_saved_exposure(self):
        result = recommend_materials(self.records, self.condition, self.saved)
        self.assertGreater(result["evaluated"], 0)
        for _, candidate in result["candidates"].iterrows():
            original = next(r for r in self.records if r["name"] == candidate["name"])
            p = dict(self.config["baseline"], **{key: original["properties"][key] for key in MATERIAL})
            expected = evaluate_samples(p, self.config["diffusion"])
            self.assertAlmostEqual(candidate["sigma_index"], float(expected["sigma_index"]))
            self.assertEqual(candidate["delta_T"], self.config["baseline"]["delta_T"])

    def test_limit_changes_flags_without_refilling_or_rescreening_shortlist(self):
        strict = recommend_materials(self.records, self.condition, self.saved, limit=0.)["candidates"]
        relaxed = recommend_materials(self.records, self.condition, self.saved, limit=1000.)["candidates"]
        self.assertEqual(strict["name"].tolist(), relaxed["name"].tolist())
        self.assertFalse(strict["within_limit"].any())
        self.assertTrue(relaxed["within_limit"].all())

    def test_invalid_and_out_of_bounds_materials_are_excluded_without_model_calls(self):
        source = next(r for r in self.records if r["condition"] == self.condition)
        key = recommendation_parameters(self.saved).iloc[0]["Parameter"]
        upper = next(r["Upper"] for r in self.config["ranges"] if r["Parameter"] == key)
        invalid = dict(source, name="Invalid", properties=dict(source["properties"], nu=.5))
        outside = dict(source, name="Outside", properties=dict(source["properties"], **{key: upper * 2}))
        with patch("inverse_analysis.evaluate_samples") as model:
            result = recommend_materials([invalid, outside], self.condition, self.saved)
        model.assert_not_called()
        self.assertEqual(result["invalid"], 1)
        self.assertEqual(result["outside"], 1)
        self.assertEqual(result["evaluated"], 0)

    def test_unusable_studies_and_invalid_constraints_are_rejected(self):
        for change in ({"target": "Mt_Minf"}, {"diffusion": {"mode": "Arrhenius"}},
                       {"ranges": default_ranges(["E", "beta"]).to_dict("records")}):
            saved = copy.deepcopy(self.saved)
            saved["config"].update(change)
            if len(saved["config"]["ranges"]) == 2:
                saved["result"]["indices"] = saved["result"]["indices"].iloc[:2]
            with self.assertRaises(ValueError):
                recommend_materials(self.records, self.condition, saved)
        saved = copy.deepcopy(self.saved)
        saved["config"]["ranges"] += default_ranges(["delta_T"]).to_dict("records")
        with self.assertRaisesRegex(ValueError, "fixed exposure"):
            recommendation_parameters(saved)
        for limit in (-1., np.inf, np.nan):
            with self.assertRaises(ValueError):
                recommend_materials(self.records, self.condition, self.saved, limit=limit)
        with self.assertRaises(ValueError):
            recommend_materials(self.records, self.condition, self.saved, shortlist_size=11)

    def test_csv_roundtrip_and_missing_fields(self):
        frame = pd.DataFrame([dict(name=r["name"], condition=r["condition"],
                                   **{key: r["properties"][key] for key in MATERIAL}) for r in self.records])
        result = recommend_materials(records_from_csv(frame), self.condition, self.saved)
        original = recommend_materials(self.records, self.condition, self.saved)
        self.assertEqual(result["candidates"]["name"].tolist(), original["candidates"]["name"].tolist())
        with self.assertRaisesRegex(ValueError, "Missing CSV columns"):
            records_from_csv(frame.drop(columns="D"))
        with self.assertRaisesRegex(ValueError, "empty"):
            records_from_csv(frame.iloc[:0])

    def test_screening_uses_log_distance_and_low_magnitude_not_most_negative_index(self):
        config = copy.deepcopy(self.config)
        keys = ["D", "E", "alpha_lam"]
        config["ranges"] = default_ranges(keys).to_dict("records")
        config["ranges"][0].update(Lower=1e-14, Upper=1e-10)
        config["target"] = "sigma_index"
        baseline = config["baseline"]
        samples = pd.DataFrame(dict(D=[1e-12] + [1e-13] * 9, E=[baseline["E"]] * 10,
                                    alpha_lam=[baseline["alpha_lam"]] * 10,
                                    sigma_abs=[.2] + [999.] * 9, sigma_index=[.2] + [-999.] * 9,
                                    Block=["A"] * 5 + ["B"] * 5))
        indices = pd.DataFrame(dict(Parameter=keys, ST=[1., .5, .3]))
        saved = dict(config=config, result=dict(samples=samples, indices=indices))
        records = [dict(name=name, condition=self.condition, properties=dict(baseline, D=d))
                   for name, d in (("One decade below", 1e-13), ("Closer logarithmically", 3e-12))]
        result = recommend_materials(records, self.condition, saved, shortlist_size=1)
        self.assertEqual(result["candidates"].iloc[0]["name"], "Closer logarithmically")


if __name__ == "__main__":
    unittest.main()
