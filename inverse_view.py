"""Interactive property-space exploration and an in-plane expansion preview."""
import json

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from inverse_analysis import condition_records, expansion_state, library_ranges, search_design_space


def box_vertices(length, width, bottom, thickness):
    return np.array([
        [-length / 2, -width / 2, bottom], [length / 2, -width / 2, bottom],
        [length / 2, width / 2, bottom], [-length / 2, width / 2, bottom],
        [-length / 2, -width / 2, bottom + thickness],
        [length / 2, -width / 2, bottom + thickness],
        [length / 2, width / 2, bottom + thickness],
        [-length / 2, width / 2, bottom + thickness],
    ])


def layer_mesh(vertices, name, color):
    faces = np.array([[0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
                      [0, 1, 5], [0, 5, 4], [1, 2, 6], [1, 6, 5],
                      [2, 3, 7], [2, 7, 6], [3, 0, 4], [3, 4, 7]])
    return go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                     i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], name=name,
                     color=color, flatshading=True, showlegend=True,
                     lighting=dict(ambient=0.65, diffuse=0.7),
                     hovertemplate=f"{name}<br>x: %{{x:.4f}} mm<br>y: %{{y:.4f}} mm<extra></extra>")


def expansion_figure(state, length, width, thickness, copper_um, magnification):
    """Animate separate layer dimensions; never infer curvature from mismatch strain."""
    gap = max(thickness * 0.25, 0.15)
    copper_bottom = thickness + gap
    copper_thickness = copper_um / 1000

    def meshes(progress):
        laminate_scale = 1 + progress * magnification * state["laminate_strain"]
        copper_scale = 1 + progress * magnification * state["copper_strain"]
        if min(laminate_scale, copper_scale) <= 0:
            raise ValueError("Choose a lower display magnification for this contraction.")
        return [
            layer_mesh(box_vertices(length * laminate_scale, width * laminate_scale, 0, thickness),
                       "Laminate", "#648d76"),
            layer_mesh(box_vertices(length * copper_scale, width * copper_scale,
                                   copper_bottom, copper_thickness), "Copper foil", "#c87d55"),
        ]

    frames = [go.Frame(data=meshes(progress), traces=[0, 1], name=f"expansion_{i}")
              for i, progress in enumerate(np.linspace(0, 1, 21))]
    # Start at the current result; Replay shows the transition from original dimensions.
    figure = go.Figure(data=meshes(1), frames=frames)
    outline_x = [-length / 2, length / 2, length / 2, -length / 2, -length / 2]
    outline_y = [-width / 2, -width / 2, width / 2, width / 2, -width / 2]
    figure.add_trace(go.Scatter3d(x=outline_x, y=outline_y, z=[thickness] * 5,
                                 mode="lines", line=dict(color="#a8b8b0", width=3, dash="dash"),
                                 name="Original outline", hoverinfo="skip"))
    max_scale = max(1, 1 + magnification * state["laminate_strain"],
                    1 + magnification * state["copper_strain"])
    figure.update_layout(
        height=420, margin=dict(l=0, r=0, b=0, t=40),
        legend=dict(orientation="h", y=-0.05),
        scene=dict(aspectmode="data", camera=dict(eye=dict(x=1.5, y=-1.8, z=1.2)),
                   xaxis=dict(title="Length (mm)", range=[-length * max_scale * .65, length * max_scale * .65]),
                   yaxis=dict(title="Width (mm)", range=[-width * max_scale * .65, width * max_scale * .65]),
                   zaxis=dict(title="Thickness (mm)", range=[-.1, copper_bottom + copper_thickness + .1])),
        updatemenus=[dict(type="buttons", direction="left", x=0, y=1.1, showactive=False,
                         buttons=[dict(label="Replay expansion", method="animate",
                                       args=[[f"expansion_{i}" for i in range(21)],
                                             dict(mode="immediate", frame=dict(duration=65, redraw=True),
                                                  transition=dict(duration=0))]),
                                  dict(label="Original dimensions", method="animate",
                                       args=[["expansion_0"], dict(mode="immediate",
                                             frame=dict(duration=0, redraw=True), transition=dict(duration=0))])])],
    )
    return figure


def property_space_figure(space, candidates):
    figure = go.Figure()
    for feasible, label, color, opacity in ((False, "Outside index limit", "#bcc7c1", .22),
                                           (True, "Within index limit", "#368575", .8)):
        points = space[space["feasible"] == feasible]
        if points.empty:
            continue
        figure.add_trace(go.Scatter3d(
            x=points["alpha_lam"], y=points["beta"], z=points["E"], mode="markers",
            name=label, marker=dict(size=3, color=color, opacity=opacity),
            customdata=points[["sigma_index", "nu", "D", "Csat"]].to_numpy(),
            hovertemplate="CTE: %{x:.2f} ppm/K<br>CHE: %{y:.3e} m³/kg<br>E: %{z:.0f} MPa"
                          "<br>Index: %{customdata[0]:+.3f} MPa<br>ν: %{customdata[1]:.3f}"
                          "<br>D: %{customdata[2]:.3e} m²/s<br>Csat: %{customdata[3]:.2f} mol/m³<extra></extra>",
        ))
    figure.add_trace(go.Scatter3d(
        x=candidates["alpha_lam"], y=candidates["beta"], z=candidates["E"],
        mode="markers", name="Library comparisons",
        marker=dict(size=7, symbol="diamond", color="#c87d55", line=dict(width=1, color="#875637")),
        customdata=candidates[["grade", "sigma_index", "status"]].to_numpy(),
        hovertemplate="%{customdata[0]}<br>Index: %{customdata[1]:+.3f} MPa<br>%{customdata[2]}<extra></extra>",
    ))
    figure.update_layout(height=440, margin=dict(l=0, r=0, b=0, t=10),
                         legend=dict(orientation="h", y=-.08),
                         scene=dict(xaxis_title="Laminate CTE (ppm/K)",
                                    yaxis=dict(title="Moisture expansion (m³/kg)", tickformat=".1e"),
                                    zaxis_title="Modulus (MPa)"))
    return figure


def render_inverse(records):
    st.caption("Identify material-property combinations within a chosen stress-index limit, "
               "then compare library materials with this model-based design space.")
    controls, preview = st.columns([1, 1.05], gap="large")
    with controls:
        with st.container(border=True):
            st.subheader("Design constraints")
            limit = st.number_input("Maximum stress index magnitude (MPa)", min_value=0.0, value=10.0,
                                    step=1.0, key="inverse_limit", persist_state="session")
            environments = list(dict.fromkeys(record["condition"] for record in records))
            condition = st.selectbox("Conditioning environment", environments, key="inverse_condition", persist_state="session")
            matching = condition_records(records, condition)
            exposure_temperature = float(condition.split("°C")[0])
            reference_temperature = st.number_input("Reference temperature (°C)", min_value=-100.0,
                                                    max_value=200.0, value=25.0, key="inverse_reference_temperature", persist_state="session")
            time, thickness_input = st.columns(2)
            hours = time.number_input("Exposure time (hours)", min_value=0.0, value=24.0,
                                      key="inverse_hours", persist_state="session")
            thickness = thickness_input.number_input("Laminate thickness (mm)", min_value=.001,
                                                      value=1.6, key="inverse_thickness", persist_state="session")
            st.caption(f"ΔT = {exposure_temperature - reference_temperature:+.1f} K. "
                       "Diffusivity and saturation concentration use the selected library conditioning environment.")
            with st.expander("Geometry and copper properties"):
                length = st.number_input("Board length (mm)", min_value=1.0, value=20.0, key="inverse_length", persist_state="session")
                width = st.number_input("Board width (mm)", min_value=1.0, value=12.0, key="inverse_width", persist_state="session")
                copper_um = st.number_input("Copper foil thickness (µm)", min_value=1.0, value=35.0,
                                            key="inverse_copper_thickness", persist_state="session")
                copper_cte = st.number_input("Copper CTE (ppm/K)", value=16.5, key="inverse_copper_cte", persist_state="session")
                st.caption("Length, width and copper thickness control the preview geometry. "
                           "The current index model uses laminate thickness and copper CTE.")
            preview_names = [r["name"] for r in matching]
            if st.session_state.get("inverse_preview_material") not in preview_names:
                st.session_state["inverse_preview_material"] = preview_names[0]
            selected = st.selectbox("Preview material", preview_names, key="inverse_preview_material", persist_state="session")
            st.caption("This material supplies the preview properties; it does not restrict the property-space search.")
    exposure = dict(h_mm=thickness, t_hours=hours, delta_T=exposure_temperature - reference_temperature,
                    alpha_cu=copper_cte)
    record = next(r for r in matching if r["name"] == selected)
    with preview:
        st.subheader("Copper–laminate expansion preview")
        display = st.radio("Display scale", ["True scale (1×)", "Magnified strain (100×)"],
                            horizontal=True, key="inverse_display_scale", persist_state="session")
        magnification = 1 if display == "True scale (1×)" else 100
        state = expansion_state(record["properties"], exposure, length, width)
        try:
            st.plotly_chart(expansion_figure(state, length, width, thickness, copper_um, magnification),
                            width="stretch", key="inverse_expansion_preview")
        except ValueError as exc:
            st.error(str(exc))
        lam, cu = st.columns(2)
        lam.metric("Laminate length change", f"{state['delta_length_laminate'] * 1000:+.2f} µm")
        cu.metric("Copper length change", f"{state['delta_length_copper'] * 1000:+.2f} µm")
        st.caption(f"Preview stress index: {state['sigma_index']:+.2f} MPa. "
                   "Displayed dimensions use the selected geometry; 100× magnifies strain only. "
                   "Thickness stays fixed because through-thickness expansion coefficients are not provided. "
                   "The layer gap is for visibility.")
        st.caption("This is unconstrained in-plane free expansion. The model does not predict bending, "
                   "warpage or interface separation.")

    with st.expander("Material-property search bounds", expanded=False):
        st.caption("Starting bounds span the library materials at the chosen environment. "
                   "Edit them to explore other combinations; each property varies independently.")
        saved_ranges = st.session_state.setdefault("inverse_property_ranges", {})
        ranges = st.data_editor(saved_ranges.get(condition, library_ranges(records, condition)), hide_index=True, width="stretch",
                                disabled=["Parameter", "Label", "Unit"], key=f"inverse_ranges_{condition}",
                                column_config={"Parameter": None, "Label": "Property",
                                               "Lower": st.column_config.NumberColumn(required=True, format="%.4g"),
                                               "Upper": st.column_config.NumberColumn(required=True, format="%.4g"),
                                               "Distribution": st.column_config.SelectboxColumn(
                                                   options=["uniform", "log-uniform"], required=True)})
    config = dict(condition=condition, exposure=exposure, limit=limit, ranges=ranges.to_dict("records"))
    signature = json.dumps(config, sort_keys=True)
    if st.button("Explore feasible space", type="primary", key="inverse_explore"):
        saved_ranges[condition] = ranges.copy()
        try:
            with st.spinner("Exploring property combinations and comparing the material library…"):
                result = search_design_space(records, **config)
            st.session_state["inverse_result"] = dict(signature=signature, result=result)
        except (ValueError, TypeError, OverflowError) as exc:
            st.session_state.pop("inverse_result", None)
            st.error(str(exc))
    saved = st.session_state.get("inverse_result")
    if not saved or saved["signature"] != signature:
        st.caption("Choose your constraints, then explore the feasible space. Re-run after changing constraints or bounds.")
        return
    result = saved["result"]
    space, candidates = result["space"], result["candidates"]
    st.subheader("Feasible property space")
    samples_metric, library_metric = st.columns(2)
    samples_metric.metric("Samples within index limit", f"{int(space['feasible'].sum()):,} / {len(space):,}")
    library_metric.metric("Library entries within chosen limits", f"{int(candidates['feasible'].sum())} / {len(candidates)}")
    if not space["feasible"].any():
        st.info("No sampled combinations meet this limit within the chosen bounds. "
                "Try adjusting the constraints or exploring different property bounds.")
    st.plotly_chart(property_space_figure(space, candidates), width="stretch", key="inverse_property_space")
    st.caption("A three-dimensional projection of a six-property search. Poisson's ratio, diffusivity "
               "and saturation concentration also vary; hover over points to inspect them. "
               "Feasible samples are combinations within the index limit, not guaranteed manufacturable materials. "
               "Sampling does not establish the entire feasible boundary.")
    st.subheader("Candidate library comparisons")
    st.caption("Entries are compared with your index limit and property bounds, using your selected "
               "thickness and exposure time. These are preliminary model comparisons, not material recommendations.")
    table = candidates[["grade", "sigma_index", "margin", "status"]].rename(
        columns={"grade": "Material", "sigma_index": "Signed index (MPa)",
                 "margin": "Index-limit margin (MPa)", "status": "Comparison"})
    st.dataframe(table, hide_index=True, width="stretch",
                 column_config={"Signed index (MPa)": st.column_config.NumberColumn(format="%.3f"),
                                "Index-limit margin (MPa)": st.column_config.NumberColumn(format="%.3f")})
    st.caption("Model validation and physical measurements of the interface are needed before qualifying "
               "a material or assessing delamination risk. A small net index can also reflect cancellation "
               "between hygroscopic and thermal contributions.")
    with st.expander("Feasibility rule and sampled properties"):
        st.latex(r"\left|\frac{E}{1-\nu}\left[\beta W(t)+(\alpha_{\mathrm{lam}}-\alpha_{\mathrm{Cu}})10^{-6}\Delta T\right]\right|"
                 r"\leq \sigma_{\mathrm{limit}}")
        st.dataframe(space.head(500), hide_index=True, width="stretch")
        st.download_button("Download sampled property space (CSV)", space.to_csv(index=False).encode(),
                           "feasible_property_space.csv", "text/csv", key="inverse_space_download")
