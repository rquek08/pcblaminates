"""Exercise editable sensitivity inputs and modal results through Streamlit."""
import json
from pathlib import Path
import unittest

from streamlit.testing.v1 import AppTest

from sensitivity_analysis import STUDIES


PRESETS = Path(__file__).resolve().parents[1] / "material_presets.json"
APP = f"""
import json
from pathlib import Path
from sensitivity_view import render_sensitivity
render_sensitivity(json.loads(Path({str(PRESETS)!r}).read_text())["presets"])
"""


class SensitivityViewTests(unittest.TestCase):
    def setUp(self):
        self.app = AppTest.from_string(APP, default_timeout=30).run()
        self.assertFalse(self.app.exception)

    def edit(self, changes, material=0):
        editor = [item for item in self.app.dataframe if item.key and item.key.startswith("sens_inputs_")][material]
        rows = editor.value["Parameter"].tolist()
        states = self.app._tree.get_widget_states()
        widget = states.widgets.add()
        widget.id = editor.proto.id
        widget.string_value = json.dumps({
            "edited_rows": {str(rows.index(key)): values for key, values in changes.items()},
            "added_rows": [], "deleted_rows": [],
        })
        self.app._run(states)
        self.assertFalse(self.app.exception)

    def run_analysis(self):
        self.app.button(key="sens_run").click().run()
        self.assertFalse(self.app.exception)
        self.assertFalse(self.app.warning)

    def results_modal(self):
        return self.app.get("dialog")[0]

    def test_results_open_in_modal_and_sampling_controls_are_hidden(self):
        labels = [item.label for item in self.app.number_input]
        self.assertNotIn("Base sample size N", labels)
        self.assertNotIn("Sampling seed", labels)
        self.assertEqual(len(self.app.get("dialog")), 0)
        self.assertEqual(len(self.app.get("plotly_chart")), 0)
        self.run_analysis()
        self.assertEqual(len(self.app.get("dialog")), 1)
        self.assertEqual(len(self.results_modal().get("plotly_chart")), 1)
        config = self.app.session_state["sensitivity_result"]["config"]
        self.assertEqual(config["N"], 4096)
        self.assertEqual(config["seed"], 12345)

    def test_nominal_edits_update_bounds_and_fixed_inputs_reach_analysis(self):
        self.edit({"E": {"Nominal": 30000.}, "t_hours": {"Nominal": 48.},
                   "D": {"Nominal": 1e-12}})
        table = self.app.dataframe[0].value.set_index("Parameter")
        self.assertEqual(table.loc["E", "Lower"], 27000.)
        self.assertEqual(table.loc["E", "Upper"], 33000.)
        self.assertEqual(table.loc["D", "Lower"], 5e-13)
        self.assertEqual(table.loc["D", "Upper"], 2e-12)
        self.assertEqual(table.loc["t_hours", "Treatment"], "Fixed")
        self.run_analysis()
        config = self.app.session_state["sensitivity_result"]["config"]
        self.assertEqual(config["baseline"]["t_hours"], 48.)
        self.assertEqual(config["baseline"]["E"], 30000.)
        self.assertEqual(config["baseline"]["D"], 1e-12)

    def test_manual_ranges_survive_nominal_edits_and_results_become_stale(self):
        self.edit({"E": {"Lower": 20000., "Upper": 40000.}})
        self.run_analysis()
        self.edit({"E": {"Nominal": 32000.}})
        table = self.app.dataframe[0].value.set_index("Parameter")
        self.assertEqual(table.loc["E", "Lower"], 20000.)
        self.assertEqual(table.loc["E", "Upper"], 40000.)
        self.assertNotIn("Distribution", table.columns)
        self.assertEqual(len(self.app.get("plotly_chart")), 0)
        self.assertEqual(len(self.app.get("dialog")), 0)
        self.run_analysis()
        config = self.app.session_state["sensitivity_result"]["config"]
        bounds = next(row for row in config["ranges"] if row["Parameter"] == "E")
        self.assertEqual(bounds["Upper"], 40000.)

    def test_comparison_materials_keep_separate_inputs_across_range_modes(self):
        self.app.pills(key="sens_scope").set_value("Compare materials").run()
        self.edit({"E": {"Nominal": 30000., "Lower": 20000., "Upper": 40000.}})
        editors = [item for item in self.app.dataframe if item.key and item.key.startswith("sens_inputs_")]
        self.assertEqual(editors[0].value.set_index("Parameter").loc["E", "Nominal"], 30000.)
        self.assertNotEqual(editors[1].value.set_index("Parameter").loc["E", "Nominal"], 30000.)
        self.app.radio(key="sens_range_basis").set_value("Broad exploratory ranges").run()
        table = self.app.dataframe[0].value.set_index("Parameter")
        self.assertEqual(table.loc["E", "Nominal"], 30000.)
        self.assertEqual(table.loc["E", "Lower"], 7500.)
        self.app.radio(key="sens_range_basis").set_value("Around each material").run()
        self.assertEqual(self.app.dataframe[0].value.set_index("Parameter").loc["E", "Lower"], 20000.)
        self.run_analysis()
        self.assertEqual(len(self.app.session_state["sensitivity_batch"]["runs"]), 2)
        self.assertEqual(len(self.results_modal().get("plotly_chart")), 2)

    def test_empty_selection_invalid_bounds_and_arrhenius(self):
        self.app.multiselect(key="sens_parameters_1").set_value([]).run()
        self.assertTrue(self.app.button(key="sens_run").disabled)
        self.app.multiselect(key="sens_parameters_1").set_value(["E"]).run()
        self.edit({"E": {"Lower": 40000., "Upper": 20000.}})
        self.app.button(key="sens_run").click().run()
        self.assertFalse(self.app.exception)
        self.assertTrue(any("upper bound" in item.value for item in self.app.warning))
        self.app.selectbox(key="sens_study").set_value(STUDIES[0]).run()
        self.app.radio(key="sens_diffusion_model").set_value("Arrhenius").run()
        self.assertEqual(self.app.dataframe[0].value.set_index("Parameter").loc["D", "Treatment"],
                         "From temperature")
        self.run_analysis()


if __name__ == "__main__":
    unittest.main()
