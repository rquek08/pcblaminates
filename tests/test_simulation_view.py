"""Check the compact analyser, input state, and calculation results dialog."""
from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "app.py"


class SimulationViewTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_file(str(APP), default_timeout=30).run()
        self.app.pills(key="main_navigation").set_value("Stress Analyser").run()
        self.assertFalse(self.app.exception)

    def test_initial_view_has_inputs_on_right_and_no_inline_results(self):
        self.assertEqual(len(self.app.number_input), 10)
        self.assertEqual(len(self.app.metric), 0)
        self.assertEqual(len(self.app.get("plotly_chart")), 0)
        self.assertEqual(len(self.app.get("dialog")), 0)
        self.assertEqual([widget.label for widget in self.app.selectbox], ["Laminate"])
        self.assertEqual(len(self.app.segmented_control), 0)
        structure = next(block for block in self.app.get("flex_container")
                         if block.key == "simulation_structure")
        self.assertEqual(len(structure.number_input), 0)
        self.assertEqual(len(structure.selectbox), 0)
        self.assertEqual(len(structure.caption), 0)

    def test_calculate_opens_a_dialog_using_current_inputs(self):
        self.app.number_input(key="field_delta_T").set_value(35.)
        self.app.number_input(key="field_h_mm").set_value(.8)
        self.app.button(key="calculate_simulation").click().run()
        self.assertFalse(self.app.exception)
        self.assertEqual(len(self.app.get("dialog")), 1)
        self.assertEqual(len(self.app.get("plotly_chart")), 2)
        saved = self.app.session_state["simulation_calculation"]
        self.assertEqual(saved["scenario"]["inputs"]["delta_T"], 35.)
        self.assertEqual(saved["scenario"]["inputs"]["h_mm"], .8)
        cards = {metric.label: metric.value for metric in self.app.metric}
        self.assertEqual(cards["Net Stress Index"], f"{saved['results']['sigma_index']:.2f} MPa")
        self.assertEqual(cards["Thermal Strain"], f"{saved['results']['eps_t']:.2e}")
        self.assertEqual([tab.label for tab in self.app.tabs], ["Stress contributions", "Moisture uptake"])

    def test_property_edits_survive_navigation_and_invalidate_previous_results(self):
        self.app.number_input(key="field_D").set_value(1e-12).run()
        self.app.button(key="calculate_simulation").click().run()
        self.assertIn("simulation_calculation", self.app.session_state)
        self.app.pills(key="main_navigation").set_value("Home").run()
        self.app.pills(key="main_navigation").set_value("Stress Analyser").run()
        self.assertEqual(self.app.number_input(key="field_D").value, 1e-12)
        self.app.number_input(key="field_delta_T").set_value(80.).run()
        self.assertFalse(self.app.exception)
        self.assertNotIn("simulation_calculation", self.app.session_state)
        self.assertEqual(len(self.app.get("dialog")), 0)


if __name__ == "__main__":
    unittest.main()
