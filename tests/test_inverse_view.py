"""Check shortlist heatmap scaling and its integration with the current recommendations."""
import json
from pathlib import Path
import unittest

import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from inverse_view import property_scatterplot, recommendation_heatmap


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
        self.assertEqual(len(app.get("dialog")), 0)
        self.assertEqual(len(app.get("plotly_chart")), 0)
        app.button(key="inverse_open_heatmap").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get("dialog")), 1)
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
        for view in ("exploration", "recommendations", "heatmap"):
            self.assertTrue(app.button(key=f"inverse_open_{view}").disabled)

    def test_views_open_only_on_request_and_upload_waits_for_a_library(self):
        app = AppTest.from_string(APP, default_timeout=30).run()
        for view in ("exploration", "recommendations", "heatmap"):
            self.assertTrue(app.button(key=f"inverse_open_{view}").disabled)
        app.button(key="inverse_recommend").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get("dialog")), 0)
        self.assertEqual(len(app.get("plotly_chart")), 0)
        app.button(key="inverse_open_recommendations").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get("dialog")), 1)
        self.assertEqual(app.get("dialog")[0].proto.dialog.title, "Recommended Materials")
        self.assertNotIn("Recommended Materials", [item.value for item in app.subheader])
        # A normal app rerun closes the dialog; the next result uses the same shortlist.
        app.run()
        app.button(key="inverse_open_exploration").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get("dialog")), 1)
        spec = json.loads(app.get("plotly_chart")[0].proto.spec)
        self.assertEqual([trace["type"] for trace in spec["data"]], ["scatter3d", "scatter3d"])
        self.assertEqual(spec["data"][1]["name"], "Recommended shortlist")
        app.radio(key="inverse_library_source").set_value("Upload CSV").run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get("plotly_chart")), 0)
        self.assertTrue(app.button(key="inverse_open_exploration").disabled)

    def test_summary_uses_current_application_and_top_material_for_each_objective(self):
        app = AppTest.from_string(APP, default_timeout=30).run()
        metrics = {item.label: item.value for item in app.metric}
        self.assertEqual(metrics["Temperature change"], "+60 K")
        self.assertEqual(metrics["Soak time"], "24 h")
        self.assertNotIn("Signed stress", metrics)
        for objective, label, unit in [("sigma_abs", "Stress magnitude", " MPa"),
                                       ("eps_h", "Swelling strain", ""),
                                       ("W_t", "Absorbed water", " kg/m³")]:
            app.selectbox(key="inverse_objective").set_value(objective).run()
            app.button(key="inverse_recommend").click().run()
            self.assertFalse(app.exception)
            row = app.session_state["inverse_recommendation"]["result"]["candidates"].iloc[0]
            metrics = {item.label: item.value for item in app.metric}
            self.assertEqual(metrics[label], f"{row[objective]:.3g}{unit}")
            self.assertEqual(metrics["Signed stress"], f"{row['sigma_index']:+.2f} MPa")
            self.assertEqual(metrics["Moisture saturation"], f"{row['Mt_Minf'] * 100:.2f} %")
        # Changing conditions replaces the previous leader with the current application.
        app.number_input(key="inverse_app_delta_T").set_value(35.).run()
        metrics = {item.label: item.value for item in app.metric}
        self.assertEqual(metrics["Temperature change"], "+35 K")
        self.assertNotIn("Signed stress", metrics)
        # An empty shortlist must never retain a previous recommendation.
        app.checkbox(key="inverse_enable_W_t").check().run()
        app.number_input(key="inverse_max_W_t").set_value(0.).run()
        app.button(key="inverse_recommend").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.session_state["inverse_recommendation"]["result"]["candidates"].empty)
        self.assertNotIn("Signed stress", {item.label: item.value for item in app.metric})
        self.assertTrue(any("No materials meet" in item.value for item in app.caption))

    def test_scatter_uses_property_values_and_shortlist_without_changing_them(self):
        points = pd.DataFrame([dict(name="A", material_class="Laminate", alpha_lam=12., beta=.0002, E=20000.),
                               dict(name="B", material_class="Laminate", alpha_lam=15., beta=.0003, E=25000.)])
        shortlist = points.iloc[[1]].copy()
        original = points.copy(deep=True)
        figure = property_scatterplot(points, shortlist, ["E", "alpha_lam", "beta"])
        np.testing.assert_array_equal(figure.data[0].x, points["E"])
        np.testing.assert_array_equal(figure.data[0].y, points["alpha_lam"])
        np.testing.assert_array_equal(figure.data[0].z, points["beta"])
        np.testing.assert_array_equal(figure.data[1].x, shortlist["E"])
        self.assertEqual(figure.data[1].customdata[0][0], "B")
        self.assertIn("Young's modulus (MPa)", figure.layout.scene.xaxis.title.text)
        pd.testing.assert_frame_equal(points, original)

    def test_inspector_cards_and_contributions_follow_selected_material(self):
        inspector_app = APP + """
from inverse_view import render_recommendations
saved = st.session_state.get("inverse_recommendation")
if saved:
    render_recommendations(saved["result"], dict(application=dict(delta_T=60., t_hours=24., h_mm=1.6, alpha_cu=16.5),
                           objective="sigma_abs", condition="85°C/85%RH"), "test-library", saved["signature"])
"""
        app = AppTest.from_string("import streamlit as st\n" + inspector_app, default_timeout=30).run()
        app.button(key="inverse_recommend").click().run()
        self.assertFalse(app.exception)
        stored = app.session_state["inverse_recommendation"]
        candidates = stored["result"]["candidates"]
        selector_key = f"inverse_selected_{stored['signature'][:12]}"
        for selected in (0, 1):
            app.selectbox(key=selector_key).set_value(selected).run()
            self.assertFalse(app.exception)
            self.assertNotIn("Inspect a recommended material", [item.label for item in app.expander])
            self.assertIn("Inspect a recommended material", [item.value for item in app.subheader])
            inspector = app
            row = candidates.loc[selected]
            cards = {item.label: item.value for item in inspector.metric}
            self.assertEqual(cards["Moisture Saturation"], f"{row['Mt_Minf'] * 100:.2f} %")
            self.assertEqual(cards["Hygroscopic Strain"], f"{row['eps_h']:.2e}")
            self.assertEqual(cards["Thermal Strain"], f"{row['eps_t']:.2e}")
            self.assertEqual(cards["Net Stress Index"], f"{row['sigma_index']:.2f} MPa")
            chart_column = next(column for column in inspector.columns
                                if any(item.value == "Stress Contributions and Net Index" for item in column.subheader))
            results_column = next(column for column in inspector.columns
                                  if any(item.value == "Results" for item in column.subheader))
            self.assertEqual(len(chart_column.get("plotly_chart")), 1)
            self.assertEqual(len(results_column.get("plotly_chart")), 0)
            chart = json.loads(chart_column.get("plotly_chart")[0].proto.spec)
            self.assertEqual(chart["data"][0]["text"],
                             [f"{row[key]:+.2f} MPa" for key in ("sigma_hygro", "sigma_thermal", "sigma_index")])
            self.assertTrue(any(f"{row['sigma_index']:+.2f} MPa" in item.value for item in results_column.markdown))
            properties = next(item for item in inspector.expander if item.label == "Material properties")
            self.assertFalse(properties.proto.expanded)
            self.assertEqual(len(properties.dataframe), 1)


if __name__ == "__main__":
    unittest.main()
