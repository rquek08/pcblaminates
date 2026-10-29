"""Check shortlist heatmap scaling and its integration with the current recommendations."""
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from inverse_view import recommendation_heatmap


ROOT = Path(__file__).resolve().parents[1]
APP = f"""
import json
from pathlib import Path
from inverse_view import render_recommender
render_recommender(json.loads(Path({str(ROOT / 'material_presets.json')!r}).read_text())["presets"])
"""


class RecommendationHeatmapTests(unittest.TestCase):
    def test_signs_raw_values_units_and_rank_order_are_preserved(self):
        candidates = pd.DataFrame([
            dict(rank=1, name="Material", material_class="PCB laminate", sigma_abs=.2, sigma_index=.2,
                 sigma_hygro=5., sigma_thermal=-4.8, eps_h=1e-4, W_t=.5, Mt_Minf=.25),
            dict(rank=2, name="Material", material_class="PCB laminate", sigma_abs=1., sigma_index=-1.,
                 sigma_hygro=2., sigma_thermal=-3., eps_h=2e-4, W_t=1., Mt_Minf=.5),
        ])
        original = candidates.copy(deep=True)
        figure = recommendation_heatmap(candidates)
        self.assertEqual(len(figure.data), 7)
        for trace in figure.data:
            self.assertEqual(list(trace.y), ["1. Material", "2. Material"])
        # Every coloured column must span exactly its matching header position.
        for column, trace in enumerate(figure.data):
            np.testing.assert_allclose(trace.x, [column - .5, column + .5])
        self.assertEqual(figure.layout.xaxis.type, "linear")
        self.assertEqual(list(figure.layout.xaxis.tickvals), list(range(7)))
        self.assertEqual(list(figure.layout.xaxis.range), [-.5, 6.5])
        signed = figure.data[1]
        np.testing.assert_allclose(signed.z, [[.2], [-1.]])
        self.assertEqual(signed.coloraxis, "coloraxis2")
        self.assertEqual(signed.customdata[1][0][2], -1.)
        self.assertEqual(signed.customdata[1][0][3], "MPa")
        np.testing.assert_allclose(figure.data[2].z, [[1.], [.4]])
        np.testing.assert_allclose(figure.data[4].z, [[.5], [1.]])
        self.assertEqual(figure.data[4].customdata[0][0][2], 1e-4)
        self.assertEqual(figure.data[6].customdata[0][0][2], 25.)
        self.assertEqual(figure.data[6].customdata[0][0][3], "%")
        self.assertEqual(figure.layout.coloraxis2.cmin, -1)
        self.assertEqual(figure.layout.coloraxis2.cmax, 1)
        self.assertEqual(figure.layout.yaxis.autorange, "reversed")
        pd.testing.assert_frame_equal(candidates, original)

    def test_single_candidate_with_zero_outputs_has_finite_neutral_cells(self):
        candidates = pd.DataFrame([dict(rank=1, name="Dry", material_class="PCB laminate",
                                       **dict.fromkeys(["sigma_abs", "sigma_index", "sigma_hygro",
                                                        "sigma_thermal", "eps_h", "W_t", "Mt_Minf"], 0.))])
        figure = recommendation_heatmap(candidates)
        for trace in figure.data:
            np.testing.assert_array_equal(trace.z, [[0.]])
            self.assertEqual(trace.text[0][0], "0")

    def test_heatmap_uses_current_shortlist_and_disappears_when_inputs_change(self):
        app = AppTest.from_string(APP, default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get("plotly_chart")), 0)
        app.selectbox(key="inverse_objective").set_value("eps_h")
        app.number_input(key="inverse_app_delta_T").set_value(35.)
        app.number_input(key="inverse_app_t_hours").set_value(48.)
        app.number_input(key="inverse_result_count").set_value(3)
        app.button(key="inverse_recommend").click().run()
        self.assertFalse(app.exception)
        self.assertIn("Compare recommended materials", [item.label for item in app.expander])
        stored = app.session_state["inverse_recommendation"]
        candidates = stored["result"]["candidates"]
        self.assertEqual(len(candidates), 3)
        self.assertTrue(candidates["eps_h"].is_monotonic_increasing)
        self.assertTrue(candidates["delta_T"].eq(35.).all())
        spec = json.loads(app.get("plotly_chart")[0].proto.spec)
        for trace in spec["data"]:
            self.assertEqual([data[0][0] for data in trace["customdata"]], candidates["name"].tolist())
        np.testing.assert_allclose([data[0][2] for data in spec["data"][0]["customdata"]], candidates["sigma_abs"])
        # A changed scenario must not display the previous table or heatmap.
        app.number_input(key="inverse_app_delta_T").set_value(80.).run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get("plotly_chart")), 0)


if __name__ == "__main__":
    unittest.main()
