import copy
from io import StringIO
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from inverse_analysis import (
    APPLICATION_DEFAULTS, OBJECTIVES, prepare_library, recommend_materials, records_from_csv,
)
from sensitivity_analysis import MATERIAL, evaluate_samples


class MaterialRecommenderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = json.loads((Path(__file__).parents[1] / "material_presets.json").read_text())["presets"]
        cls.frame = prepare_library(cls.records)
        cls.condition = "85°C/85%RH"
        cls.application = dict(APPLICATION_DEFAULTS, t_hours=48., h_mm=.8, delta_T=35.)
        rng = np.random.default_rng(12)
        n = 137
        cls.large = pd.DataFrame(dict(name=[f"Material {i}" for i in range(n)], condition=[cls.condition] * n,
                                      material_class=["PCB laminate"] * n, alpha_lam=rng.uniform(8,23,n),
                                      E=rng.uniform(7500,28000,n), nu=rng.uniform(.175,.32,n),
                                      D=10**rng.uniform(-14,-11,n), Csat=rng.uniform(90,370,n),
                                      beta=rng.uniform(1e-4,4e-4,n)))
        cls.large = records_from_csv(cls.large)

    def brute_force(self, frame, objective, output_limits=None):
        output = evaluate_samples(dict(self.application, **{key: frame[key].to_numpy() for key in MATERIAL}),
                                  {"mode": "Fixed D"})
        expected = frame.copy()
        for key, value in output.items():
            expected[key] = value
        for key, limit in (output_limits or {}).items():
            expected = expected.loc[expected[key] <= limit]
        return expected.sort_values(objective, kind="stable").head(10)

    def test_all_objectives_match_exhaustive_ranking_across_batches(self):
        for objective in OBJECTIVES:
            with self.subTest(objective=objective):
                result = recommend_materials(self.large, self.condition, self.application,
                                             objective=objective, batch_size=17)
                expected = self.brute_force(self.large, objective)
                self.assertEqual(result["candidates"]["name"].tolist(), expected["name"].tolist())
                self.assertEqual(result["evaluated"], len(self.large))
                self.assertEqual(result["qualifying"], len(self.large))
                self.assertEqual(len(result["candidates"]), 10)

    def test_model_calls_are_bounded_and_every_eligible_record_is_evaluated_once(self):
        with patch("inverse_analysis.evaluate_samples", wraps=evaluate_samples) as model:
            result = recommend_materials(self.large, self.condition, self.application, batch_size=16)
        sizes = [len(call.args[0]["E"]) for call in model.call_args_list]
        self.assertEqual(sum(sizes), len(self.large))
        self.assertLessEqual(max(sizes), 16)
        self.assertEqual(result["batches"], len(sizes))

    def test_class_environment_and_property_filters_apply_before_model_calls(self):
        bounds = {"E": [10000., 20000.], "alpha_lam": [9., 14.]}
        source = self.records[0]
        records = [dict(source, name="Eligible"),
                   dict(source, name="Other condition", condition="different"),
                   dict(source, name="Other class", material_class="Other"),
                   dict(source, name="Outside", properties=dict(source["properties"], E=50000.)),
                   dict(source, name="Invalid", properties=dict(source["properties"], nu=.5))]
        records[0]["properties"] = dict(source["properties"], E=15000., alpha_lam=12.)
        with patch("inverse_analysis.evaluate_samples", wraps=evaluate_samples) as model:
            result = recommend_materials(records, self.condition, self.application, material_class=source["material_class"],
                                         property_bounds=bounds)
        self.assertEqual(result["total"], 3)
        self.assertEqual(result["invalid"], 1)
        self.assertEqual(result["property_excluded"], 1)
        self.assertEqual(result["eligible"], 1)
        self.assertEqual(result["candidates"]["name"].tolist(), ["Eligible"])
        model.assert_called_once()

    def test_output_limits_filter_before_top_ten_and_refill_from_entire_library(self):
        limits = {"W_t": .5, "sigma_abs": 15.}
        result = recommend_materials(self.large, self.condition, self.application, objective="eps_h",
                                     output_limits=limits, batch_size=19)
        expected = self.brute_force(self.large, "eps_h", limits)
        self.assertEqual(result["candidates"]["name"].tolist(), expected["name"].tolist())
        self.assertTrue(result["candidates"]["meets_requirements"].all())
        self.assertGreater(result["requirements_excluded"], 0)
        self.assertEqual(result["qualifying"] + result["requirements_excluded"], len(self.large))

    def test_application_inputs_override_preset_exposure_and_use_all_material_properties(self):
        result = recommend_materials(self.frame, self.condition, self.application)
        for _, row in result["candidates"].iterrows():
            original = next(r for r in self.records if r["name"] == row["name"])
            expected = evaluate_samples(dict(self.application, **{key: original["properties"][key] for key in MATERIAL}),
                                        {"mode": "Fixed D"})
            for key in ("sigma_abs", "sigma_index", "sigma_hygro", "sigma_thermal", "eps_h", "eps_t", "W_t"):
                self.assertAlmostEqual(row[key], float(expected[key]))
            for key, value in self.application.items():
                self.assertEqual(row[key], value)

    def test_zero_soak_and_diffusivity_give_no_water_or_swelling(self):
        frame = self.large.iloc[:1].copy()
        for change in ({"t_hours": 0.}, {}):
            if not change:
                frame["D"] = 0.
            result = recommend_materials(frame, self.condition, dict(self.application, **change))
            row = result["candidates"].iloc[0]
            self.assertEqual(row["W_t"], 0.)
            self.assertEqual(row["eps_h"], 0.)
            self.assertEqual(row["sigma_index"], row["sigma_thermal"])

    def test_signed_contributions_and_absolute_magnitude_are_retained(self):
        result = recommend_materials(self.large, self.condition, self.application)
        table = result["candidates"]
        np.testing.assert_allclose(table["sigma_abs"], np.abs(table["sigma_index"]))
        np.testing.assert_allclose(table["sigma_index"], table["sigma_hygro"] + table["sigma_thermal"])
        self.assertTrue(table["sigma_abs"].is_monotonic_increasing)

    def test_tied_rankings_are_stable_across_batch_sizes(self):
        source = self.records[0]
        records = [dict(source, name=f"Record {i}") for i in range(30)]
        for size in (1, 7, 100):
            result = recommend_materials(records, self.condition, self.application, batch_size=size)
            self.assertEqual(result["candidates"]["name"].tolist(), [f"Record {i}" for i in range(10)])

    def test_no_matches_invalid_rows_and_unmet_requirements_return_no_fallback(self):
        with patch("inverse_analysis.evaluate_samples") as model:
            none = recommend_materials(self.frame, "unknown condition", self.application)
            invalid = self.frame.copy()
            invalid["E"] = -1.
            bad = recommend_materials(invalid, self.condition, self.application)
        model.assert_not_called()
        self.assertTrue(none["candidates"].empty)
        self.assertTrue(bad["candidates"].empty)
        self.assertEqual(bad["invalid"], bad["total"])
        unmet = recommend_materials(self.frame, self.condition, self.application, output_limits={"W_t": 0.})
        self.assertTrue(unmet["candidates"].empty)
        self.assertEqual(unmet["requirements_excluded"], unmet["evaluated"])

    def test_csv_metadata_and_numeric_invalid_rows(self):
        raw = self.large.copy()
        raw["D"] = raw["D"].astype(object)
        raw.loc[0, "D"] = "not numeric"
        parsed = records_from_csv(raw)
        self.assertTrue(pd.isna(parsed.loc[0, "D"]))
        self.assertEqual(recommend_materials(parsed, self.condition, self.application)["invalid"], 1)
        legacy = records_from_csv(raw.drop(columns=["material_class", "supplier", "grade"]))
        self.assertTrue(legacy["material_class"].eq("Unspecified").all())
        roundtrip = records_from_csv(pd.read_csv(StringIO(self.frame.to_csv(index=False))))
        a = recommend_materials(roundtrip, self.condition, self.application)
        b = recommend_materials(self.frame, self.condition, self.application)
        self.assertEqual(a["candidates"]["name"].tolist(), b["candidates"]["name"].tolist())
        for frame, message in ((raw.drop(columns="D"), "Missing CSV"), (raw.iloc[:0], "empty"),
                               (raw.assign(name=" "), "material name")):
            with self.assertRaisesRegex(ValueError, message):
                records_from_csv(frame)

    def test_nonfinite_model_results_are_excluded_and_counted(self):
        frame = self.large.iloc[:2].copy()
        frame.loc[frame.index[0], "E"] = 1e308
        frame.loc[frame.index[0], "alpha_lam"] = 1e308
        result = recommend_materials(frame, self.condition, self.application)
        self.assertEqual(result["numerical_invalid"], 1)
        self.assertEqual(result["qualifying"], 1)

    def test_invalid_application_constraints_and_batch_configuration_are_rejected(self):
        for changed in ({"h_mm": 0.}, {"t_hours": -1.}, {"delta_T": float('inf')}, {"alpha_cu": float('nan')}):
            with self.assertRaises(ValueError):
                recommend_materials(self.frame, self.condition, dict(self.application, **changed))
        for kwargs in ({"objective": "unknown"}, {"shortlist_size": 11}, {"batch_size": 0},
                       {"property_bounds": {"E": [2., 1.]}}, {"output_limits": {"sigma_abs": -1.}},
                       {"output_limits": {"eps_h": float('nan')}}):
            with self.assertRaises(ValueError):
                recommend_materials(self.frame, self.condition, self.application, **kwargs)

    def test_does_not_modify_library_or_application(self):
        original = self.large.copy(deep=True)
        application = copy.deepcopy(self.application)
        recommend_materials(self.large, self.condition, self.application)
        pd.testing.assert_frame_equal(original, self.large)
        self.assertEqual(application, self.application)


if __name__ == "__main__":
    unittest.main()
