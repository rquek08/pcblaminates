import json
import unittest
from pathlib import Path

import numpy as np

from inverse_analysis import expansion_state, library_ranges, search_design_space
from inverse_view import expansion_figure


class InverseExplorationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = json.loads((Path(__file__).parents[1] / "material_presets.json").read_text())["presets"]
        cls.condition = "85°C/85%RH"
        cls.exposure = dict(h_mm=1.6, t_hours=24., delta_T=60., alpha_cu=16.5)

    def search(self, limit, exposure=None, ranges=None):
        return search_design_space(self.records, self.condition, exposure or self.exposure, limit,
                                   ranges or library_ranges(self.records, self.condition).to_dict("records"), N=256)

    def test_absolute_limit_applies_to_negative_and_positive_indices(self):
        result = self.search(10.)
        space = result["space"]
        self.assertTrue((space["sigma_index"] < 0).any())
        np.testing.assert_array_equal(space["feasible"], np.abs(space["sigma_index"]) <= 10.)
        candidates = result["candidates"]
        self.assertEqual(len(candidates), 10)
        np.testing.assert_array_equal(candidates["feasible"], np.abs(candidates["sigma_index"]) <= 10.)
        # Relaxing the constraint cannot remove a previously feasible design.
        relaxed = self.search(20.)
        self.assertTrue((~space["feasible"] | relaxed["space"]["feasible"]).all())

    def test_zero_exposure_and_temperature_change_give_zero_index(self):
        exposure = dict(self.exposure, t_hours=0., delta_T=0.)
        result = self.search(0., exposure=exposure)
        self.assertTrue(result["space"]["feasible"].all())
        self.assertTrue(result["candidates"]["feasible"].all())
        np.testing.assert_array_equal(result["space"]["sigma_index"], 0.)

    def test_library_candidates_must_also_meet_property_bounds(self):
        ranges = library_ranges(self.records, self.condition).to_dict("records")
        next(row for row in ranges if row["Parameter"] == "E")["Upper"] = 24000.
        result = self.search(100., ranges=ranges)
        outside = result["candidates"][result["candidates"]["E"] > 24000.]
        self.assertFalse(outside.empty)
        self.assertFalse(outside["feasible"].any())
        self.assertTrue((outside["status"] == "Outside property bounds").all())

    def test_geometry_and_strain_are_scaled_without_inventing_curvature(self):
        properties = self.records[0]["properties"]
        state = expansion_state(properties, self.exposure, 20., 12.)
        longer = expansion_state(properties, self.exposure, 40., 12.)
        self.assertAlmostEqual(longer["delta_length_laminate"], 2 * state["delta_length_laminate"])
        self.assertEqual(longer["sigma_index"], state["sigma_index"])
        self.assertAlmostEqual(state["sigma_index"],
                               properties["E"] / (1 - properties["nu"]) * state["mismatch_strain"])
        figure = expansion_figure(state, 20., 12., 1.6, 35., 1)
        laminate, copper = figure.data[:2]
        self.assertAlmostEqual(max(laminate.x) - min(laminate.x), 20. + state["delta_length_laminate"])
        self.assertAlmostEqual(max(laminate.z) - min(laminate.z), 1.6)
        self.assertAlmostEqual(max(copper.z) - min(copper.z), .035)
        self.assertAlmostEqual(max(figure.frames[0].data[0].x) - min(figure.frames[0].data[0].x), 20.)
        magnified = expansion_figure(state, 20., 12., 1.6, 35., 100)
        self.assertAlmostEqual(max(magnified.data[0].x) - min(magnified.data[0].x),
                               20. + 100 * state["delta_length_laminate"])

    def test_invalid_constraints_are_rejected(self):
        for limit, exposure in ((-1., self.exposure), (10., dict(self.exposure, h_mm=0.)),
                                (10., dict(self.exposure, t_hours=-1.))):
            with self.assertRaises(ValueError):
                self.search(limit, exposure=exposure)

    def test_thickness_and_exposure_time_change_moisture_expansion(self):
        properties = self.records[0]["properties"]
        baseline = expansion_state(properties, self.exposure, 20., 12.)
        thicker = expansion_state(properties, dict(self.exposure, h_mm=3.2), 20., 12.)
        longer = expansion_state(properties, dict(self.exposure, t_hours=48.), 20., 12.)
        self.assertLess(thicker["delta_length_laminate"], baseline["delta_length_laminate"])
        self.assertGreater(longer["delta_length_laminate"], baseline["delta_length_laminate"])
        self.assertEqual(thicker["copper_strain"], baseline["copper_strain"])
        self.assertEqual(longer["copper_strain"], baseline["copper_strain"])


if __name__ == "__main__":
    unittest.main()
