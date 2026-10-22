"""Validate 3D constituent geometry and the visualiser's interactive controls."""
from collections import Counter
import unittest

import numpy as np
from streamlit.testing.v1 import AppTest

from laminate_showcase import (
    COPPER_ROUTES, COPPER_VIAS, LAYOUTS, Mesh, add_weave, filler_particles, structure_figure,
)


class LaminateGeometryTests(unittest.TestCase):
    def test_homogeneous_cross_section_has_aligned_layers_and_adjustable_opacity(self):
        figure = structure_figure("Homogeneous laminate", resin_opacity=.05)
        solids = [trace for trace in figure.data if trace.type == "mesh3d"]
        self.assertEqual(len(solids), 3)
        for trace in solids:
            self.assertEqual((min(trace.x), max(trace.x)), (0., 10.))
            self.assertEqual((min(trace.y), max(trace.y)), (2.2, 7.))
            self.assertEqual(trace.opacity, .05 if trace.name == "Homogeneous resin core" else 1.)
            self.assertTrue(trace.flatshading)
        top = next(trace for trace in solids if trace.name == "Top copper foil")
        core = next(trace for trace in solids if trace.name == "Homogeneous resin core")
        bottom = next(trace for trace in solids if trace.name == "Bottom copper foil")
        self.assertEqual(min(top.z), max(core.z))
        self.assertEqual(max(bottom.z), min(core.z))
        transparent = structure_figure("Homogeneous laminate", view="See-through", resin_opacity=.35)
        core = next(trace for trace in transparent.data if trace.name == "Homogeneous resin core")
        self.assertEqual(core.opacity, .35)

    def test_all_structures_and_views_have_valid_closed_solid_meshes(self):
        for layout in LAYOUTS:
            for view in ("Cross section", "See-through", "Lift outer foils"):
                with self.subTest(layout=layout, view=view):
                    figure = structure_figure(layout, view=view)
                    self.assertEqual(figure.layout.scene.dragmode, "orbit")
                    for trace in figure.data:
                        if trace.type != "mesh3d":
                            continue
                        points = np.column_stack([trace.x, trace.y, trace.z])
                        faces = np.column_stack([trace.i, trace.j, trace.k])
                        self.assertTrue(np.isfinite(points).all())
                        self.assertGreaterEqual(faces.min(), 0)
                        self.assertLess(faces.max(), len(points))
                        triangles = points[faces]
                        area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                                       triangles[:, 2] - triangles[:, 0]), axis=1)
                        self.assertTrue((area > 1e-10).all())
                        edges = Counter(tuple(sorted(edge)) for a, b, c in faces
                                        for edge in ((a, b), (b, c), (c, a)))
                        self.assertTrue(all(count == 2 for count in edges.values()))

    def test_all_cross_sections_share_one_plane_and_respect_resin_opacity(self):
        for layout in LAYOUTS:
            with self.subTest(layout=layout):
                figure = structure_figure(layout, view="Cross section", resin_opacity=.4)
                for trace in figure.data:
                    if trace.type != "mesh3d":
                        continue
                    self.assertGreaterEqual(min(trace.y), 2.2 - 1e-10)
                    if trace.legendgroup == "Resin matrix":
                        self.assertEqual(trace.opacity, .4)
                        self.assertEqual((min(trace.y), max(trace.y)), (2.2, 7.))
                    if trace.name in ("Top copper foil", "Bottom copper foil"):
                        self.assertEqual((min(trace.y), max(trace.y)), (2.2, 7.))
                    if trace.name in ("Glass yarns · fill", "Dispersed filler particles"):
                        points = np.column_stack([trace.x, trace.y, trace.z])
                        faces = points[np.column_stack([trace.i, trace.j, trace.k])]
                        cap = np.isclose(faces[:, :, 1], 2.2).all(axis=1)
                        self.assertTrue(cap.any(), f"{trace.name} must be sliced, not omitted")

    def test_slice_retains_and_caps_an_intersected_filler_particle(self):
        particle = Mesh()
        particle.particle(np.array([5., 2.2, 1.2]), .3)
        section = particle.clip_y(2.2)
        points = np.vstack(section.vertices)
        faces = points[np.vstack(section.triangles)]
        self.assertGreaterEqual(points[:, 1].min(), 2.2 - 1e-10)
        self.assertAlmostEqual(points[:, 1].max(), 2.5)
        cap = faces[np.isclose(faces[:, :, 1], 2.2).all(axis=1)]
        area = np.linalg.norm(np.cross(cap[:, 1] - cap[:, 0], cap[:, 2] - cap[:, 0]), axis=1).sum() / 2
        self.assertGreater(area, .25)
        self.assertLess(area, .29)

    def test_structure_choices_have_their_own_constituents(self):
        expected = {
            "Homogeneous laminate": (False, False, False),
            "Glass fibre weave": (True, False, False),
            "Dispersed fillers": (False, True, False),
            "Multilayer copper": (False, False, True),
        }
        for layout, (glass, fillers, internal) in expected.items():
            names = [trace.name or "" for trace in structure_figure(layout).data]
            self.assertEqual(any("Glass yarns" in name for name in names), glass)
            self.assertEqual(any("filler particles" in name for name in names), fillers)
            self.assertEqual(any("Internal copper layer" in name for name in names), internal)
            self.assertTrue(any("Top copper" in name for name in names))
            self.assertTrue(any("Bottom copper" in name for name in names))

    def test_plain_weave_alternates_without_intersecting_at_crossings(self):
        warp, fill = Mesh(), Mesh()
        add_weave(warp, fill, 0.)
        heights = []
        for row in range(6):
            row_heights = []
            wx, wy, wz = warp.vertices[row][:-2].reshape(-1, 10, 3).mean(axis=1).T
            for col in range(8):
                fx, fy, fz = fill.vertices[col][:-2].reshape(-1, 10, 3).mean(axis=1).T
                separation = np.interp(fx[0], wx, wz) - np.interp(wy[0], fy, fz)
                self.assertGreater(abs(separation), .20)
                row_heights.append(separation)
            heights.append(row_heights)
        signs = np.sign(heights)
        self.assertTrue((signs[:, 1:] == -signs[:, :-1]).all())
        self.assertTrue((signs[1:] == -signs[:-1]).all())

    def test_fillers_stay_inside_core_without_overlapping(self):
        particles = filler_particles()
        self.assertEqual(len(particles), 180)
        for index, (center, radius) in enumerate(particles):
            self.assertTrue((center - radius >= 0).all())
            self.assertTrue((center + radius <= [10, 7, 2.4]).all())
            for other, other_radius in particles[:index]:
                self.assertGreater(np.linalg.norm(center - other), radius + other_radius)

    def test_vias_span_both_interfaces_and_internal_copper_is_buried(self):
        figure = structure_figure("Multilayer copper")
        vias = next(trace for trace in figure.data if trace.name == "Plated through-vias")
        self.assertLess(min(vias.z), 0)
        self.assertGreater(max(vias.z), 2.4)
        layers = [trace for trace in figure.data if (trace.name or "").startswith("Internal copper layer")]
        self.assertEqual(len(layers), 2)
        for layer in layers:
            self.assertGreater(min(layer.z), 0)
            self.assertLess(max(layer.z), 2.4)

    def test_buried_copper_is_routed_across_the_board_with_angled_sides(self):
        figure = structure_figure("Multilayer copper", view="See-through")
        for trace in figure.data:
            if not (trace.name or "").startswith("Internal copper layer"):
                continue
            points = np.column_stack([trace.x, trace.y, trace.z])
            self.assertGreater(np.ptp(points[:, 0]), 8.)
            self.assertGreater(np.ptp(points[:, 1]), 4.)
            faces = points[np.column_stack([trace.i, trace.j, trace.k])]
            normals = np.cross(faces[:, 1] - faces[:, 0], faces[:, 2] - faces[:, 0])
            angled_side = (np.abs(normals[:, 0]) > 1e-6) & (np.abs(normals[:, 1]) > 1e-6)
            self.assertTrue(angled_side.any())
            # Thin tracks leave most resin exposed, rather than filling a layer.
            top = np.isclose(faces[:, :, 2], points[:, 2].max()).all(axis=1)
            copper_area = np.linalg.norm(normals[top], axis=1).sum() / 2
            self.assertLess(copper_area, 15.)

    def test_central_copper_layer_spans_the_core_at_its_midplane(self):
        for view in ("Cross section", "See-through", "Lift outer foils"):
            figure = structure_figure("Multilayer copper", view=view)
            middle = next(trace for trace in figure.data if trace.name == "Central copper layer")
            self.assertEqual((min(middle.x), max(middle.x)), (0., 10.))
            self.assertEqual((min(middle.y), max(middle.y)), (2.2 if view == "Cross section" else 0., 7.))
            self.assertAlmostEqual((min(middle.z) + max(middle.z)) / 2, 1.2)
            self.assertGreater(max(middle.z), min(middle.z))

    def test_every_trace_end_reaches_a_board_edge_or_a_via(self):
        for layer, routes in COPPER_ROUTES.items():
            for route in routes:
                for endpoint in (route[0], route[-1]):
                    x, y = endpoint
                    at_edge = np.isclose(x, 0) or np.isclose(x, 10) or np.isclose(y, 0) or np.isclose(y, 7)
                    at_via = any(np.allclose(endpoint, center) for center in COPPER_VIAS)
                    self.assertTrue(at_edge or at_via, f"{layer}: unconnected trace end at {endpoint}")
        figure = structure_figure("Multilayer copper", view="See-through")
        for trace in figure.data:
            if trace.type == "mesh3d":
                self.assertGreaterEqual(min(trace.x), -1e-10)
                self.assertLessEqual(max(trace.x), 10. + 1e-10)
                self.assertGreaterEqual(min(trace.y), -1e-10)
                self.assertLessEqual(max(trace.y), 7. + 1e-10)

    def test_cross_section_clips_every_routed_layer_at_the_exposed_core_face(self):
        figure = structure_figure("Multilayer copper", view="Cross section")
        tracks = [trace for trace in figure.data if trace.type == "mesh3d" and
                  ("copper traces" in trace.name or "Internal copper layer" in trace.name)]
        self.assertEqual(len(tracks), 4)
        for trace in tracks:
            self.assertGreaterEqual(min(trace.y), 2.2 - 1e-10)
        self.assertTrue(any(np.isclose(trace.y, 2.2).any() for trace in tracks))

    def test_lifted_foils_and_highlighting_preserve_internal_geometry(self):
        original = structure_figure("Dispersed fillers", view="See-through")
        lifted = structure_figure("Dispersed fillers", focus="Fillers and additives", view="Lift outer foils")
        before = {trace.name: trace for trace in original.data}
        after = {trace.name: trace for trace in lifted.data}
        self.assertGreater(min(after["Top copper foil"].z), max(before["Top copper foil"].z))
        self.assertLess(max(after["Bottom copper foil"].z), min(before["Bottom copper foil"].z))
        for axis in ("x", "y", "z"):
            np.testing.assert_array_equal(getattr(before["Dispersed filler particles"], axis),
                                          getattr(after["Dispersed filler particles"], axis))
        self.assertEqual(after["Dispersed filler particles"].opacity, 1)
        self.assertLess(after["Top copper foil"].opacity, .1)


class LaminateVisualiserTests(unittest.TestCase):
    def test_homogeneous_cross_section_and_see_through_offer_matching_controls(self):
        app = AppTest.from_string("from laminate_showcase import render_showcase\nrender_showcase()").run()
        self.assertFalse(app.exception)
        self.assertEqual(len(app.slider), 1)
        app.selectbox(key="inverse_structure_view").set_value("See-through").run()
        app.slider(key="inverse_structure_opacity").set_value(.4).run()
        self.assertFalse(app.exception)
        app.selectbox(key="inverse_structure_view").set_value("Cross section").run()
        self.assertEqual(len(app.slider), 1)
        self.assertFalse(app.exception)
        app.selectbox(key="inverse_structure_view").set_value("See-through").run()
        self.assertEqual(app.slider(key="inverse_structure_opacity").value, .4)

    def test_all_concepts_and_view_controls_render_with_property_cards_on_right(self):
        app = AppTest.from_string("from laminate_showcase import render_showcase\nrender_showcase()",
                                  default_timeout=30).run()
        self.assertFalse(app.exception)
        self.assertIn("3D Laminate Visualiser", [item.value for item in app.subheader])
        for layout in LAYOUTS:
            app.radio(key="inverse_structure_layout").set_value(layout).run()
            self.assertFalse(app.exception)
            self.assertEqual(len(app.get("plotly_chart")), 1)
            self.assertEqual(len(app.columns[0].get("plotly_chart")), 1)
            self.assertEqual(len(app.columns[-1].get("plotly_chart")), 0)
            self.assertTrue(any(item.value.startswith("**") for item in app.columns[-1].markdown))
        app.selectbox(key="inverse_structure_view").set_value("Lift outer foils").run()
        app.slider(key="inverse_structure_opacity").set_value(.4).run()
        app.selectbox(key="inverse_structure_focus").set_value("Copper foil interfaces").run()
        self.assertFalse(app.exception)
        app.radio(key="inverse_structure_layout").set_value("Dispersed fillers").run()
        app.selectbox(key="inverse_structure_focus").set_value("Fillers and additives").run()
        app.radio(key="inverse_structure_layout").set_value("Homogeneous laminate").run()
        self.assertFalse(app.exception)
        self.assertEqual(app.selectbox(key="inverse_structure_focus").value, "All constituents")

    def test_previous_structure_selection_migrates(self):
        app = AppTest.from_string("from laminate_showcase import render_showcase\nrender_showcase()")
        app.session_state["inverse_structure_layout"] = "Copper-asymmetric"
        app.session_state["inverse_structure_view"] = "Cutaway"
        app.run()
        self.assertFalse(app.exception)
        self.assertEqual(app.radio(key="inverse_structure_layout").value, "Multilayer copper")
        self.assertEqual(app.selectbox(key="inverse_structure_view").value, "Cross section")


if __name__ == "__main__":
    unittest.main()
