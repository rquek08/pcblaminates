"""Compact material recommendations with exploration and results in dialogs."""
import hashlib
from html import escape
from io import BytesIO
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from inverse_analysis import (
    APPLICATION_DEFAULTS, OBJECTIVES, prepare_library, recommend_materials, records_from_csv,
)
from sensitivity_analysis import MATERIAL, MODEL_VERSION, PARAMETERS
from stress_results import render_contributions, render_summary


@st.cache_data(show_spinner=False, max_entries=4)
def load_library(records, uploaded_bytes=None):
    if uploaded_bytes is None:
        payload = json.dumps(records, sort_keys=True).encode()
        frame = prepare_library(records)
    else:
        payload = uploaded_bytes
        frame = records_from_csv(pd.read_csv(BytesIO(uploaded_bytes)))
    return frame, hashlib.sha256(payload).hexdigest()


@st.cache_data(show_spinner=False, max_entries=8)
def calculate_recommendations(library_digest, config, _library):
    # The digest covers the complete source, including uploads larger than the
    # DataFrame hashing sample. Cached results retain at most ten candidates.
    return recommend_materials(_library, config["condition"], config["application"],
                               objective=config["objective"], material_class=config["material_class"],
                               property_bounds=config["property_bounds"], output_limits=config["output_limits"],
                               shortlist_size=config["shortlist_size"])


def render_inverse(records):
    render_recommender(records)


def select_available(label, options, key):
    if st.session_state.get(key) not in options:
        st.session_state.pop(key, None)
    return st.selectbox(label, options, key=key, persist_state="session")


def render_recommender(records):
    with st.container(key="inverse_workspace"):
        controls, results = st.columns([1.05, 1], gap="medium")
        with controls.container(border=True, key="inverse_controls"):
            context = render_recommender_controls(records)
        with results.container(border=False, key="inverse_result_options"):
            render_result_options(context)


def render_recommender_controls(records):
    with st.container(key="inverse_controls_header"):
        title, help_button = st.columns([1.3, 1], gap="small", vertical_alignment="center")
        title.html('<p class="compact-section-label">Material recommender</p>')
        guidance = help_button.popover("How ranking works", width="stretch")
    with guidance:
        st.write("Conditioning selects each material's tabulated properties. Diffusivity stays fixed at that condition. "
                 "Enter the temperature change for thermal mismatch separately.")
        st.write("The recommender filters by material class, conditioning and your property bounds, then calculates "
                 "moisture uptake, swelling and stress for every eligible record using the same application conditions. "
                 "Materials meeting your output requirements are ranked by the selected objective, smallest first.")
        st.write("Large libraries are calculated in batches. Repeated requests are cached, and only the best "
                 "results are retained for display. The ranking uses all six material properties.")
        st.write("Moisture enters through one exposed face of the laminate. The opposite face interfaces "
                 "with copper foil and is treated as impermeable; the diffusion term uses (2h)² for laminate thickness h.")
    with st.container(key="inverse_inputs"):
        source = st.radio("Material library", ["Included presets", "Upload CSV"], horizontal=True,
                          key="inverse_library_source", persist_state="session", label_visibility="collapsed")
        upload = None
        if source == "Upload CSV":
            with st.container(key="inverse_library_picker"):
                with st.popover("Upload material library", width="stretch"):
                    upload = st.file_uploader("Material library (CSV)", type=["csv"], key="inverse_library_upload")
                    st.caption("Required: name, condition, alpha_lam, beta, E, nu, Csat, D. Optional: material_class, supplier, grade. "
                               "Units: CTE ppm/K; CHE m³/kg; E MPa; Csat mol/m³; D m²/s; ν dimensionless. "
                               "Rows without a class are labelled Unspecified. Invalid numeric property rows are excluded and counted.")
                    example, _ = load_library(records)
                    st.download_button("Download library CSV template", example.to_csv(index=False).encode(),
                                       "material_library.csv", "text/csv", key="inverse_library_template")
            if upload is None:
                st.info("Upload a CSV library to configure the recommendation.")
                return
        try:
            frame, digest = load_library(records, upload.getvalue() if upload is not None else None)
        except (ValueError, UnicodeError, pd.errors.ParserError) as exc:
            st.error(f"Could not read the material library: {exc}")
            return
        classes = list(dict.fromkeys(frame["material_class"]))
        all_classes = "All material classes"
        class_input, condition_input = st.columns(2, gap="small")
        with class_input:
            material_class = select_available("Material class", [*classes, all_classes], "inverse_material_class")
        class_rows = frame if material_class == all_classes else frame.loc[frame["material_class"].eq(material_class)]
        with condition_input:
            condition = select_available("Conditioning environment", list(dict.fromkeys(class_rows["condition"])),
                                         "inverse_recommend_condition")

        st.html('<hr class="compact-section-divider">')
        st.html('<p class="compact-section-label">Application conditions</p>')
        temperature, time = st.columns(2, gap="small")
        geometry, copper = st.columns(2, gap="small")
        application = dict(
            delta_T=temperature.number_input("Temperature change (°C or K)", value=APPLICATION_DEFAULTS["delta_T"],
                                              step=5.0, key="inverse_app_delta_T", persist_state="session"),
            t_hours=time.number_input("Soak time (hours)", min_value=0., value=APPLICATION_DEFAULTS["t_hours"],
                                      step=1.0, key="inverse_app_t_hours", persist_state="session"),
            h_mm=geometry.number_input("Laminate thickness (mm)", min_value=.001, value=APPLICATION_DEFAULTS["h_mm"],
                                        step=.1, key="inverse_app_h_mm", persist_state="session"),
            alpha_cu=copper.number_input("Copper foil CTE (ppm/K)", value=APPLICATION_DEFAULTS["alpha_cu"],
                                          step=.5, key="inverse_app_alpha_cu", persist_state="session"),
        )
        objective_col, count_col = st.columns([2, 1], gap="small")
        objective = objective_col.selectbox("Rank materials by", list(OBJECTIVES),
                                            format_func=lambda key: OBJECTIVES[key][0],
                                            key="inverse_objective", persist_state="session")
        count = count_col.number_input("Maximum results", min_value=1, max_value=10, value=10,
                                        key="inverse_result_count", persist_state="session")
        property_bounds, output_limits = {}, {}
        with st.container(key="inverse_requirements"):
            st.html('<hr class="compact-section-divider">')
            st.html('<p class="compact-section-label">Optional requirements</p>')
            keys = st.multiselect("Material properties to constrain", MATERIAL,
                                  format_func=lambda key: PARAMETERS[key][0], key="inverse_property_filters", persist_state="session",
                                  label_visibility="collapsed", placeholder="Constrain material properties (optional)",
                                  help="Only materials meeting all enabled bounds and output limits appear in the results. "
                                       "These are application requirements, not safety limits.")
            available = class_rows.loc[class_rows["condition"].eq(condition)]
            errors = []
            for key in keys:
                label, unit, default_lo, default_hi, _ = PARAMETERS[key]
                numeric = available[key].replace([float("inf"), -float("inf")], float("nan")).dropna()
                lo = float(numeric.min()) if len(numeric) else default_lo
                hi = float(numeric.max()) if len(numeric) else default_hi
                lower, upper = st.columns(2, gap="small")
                field_scope = hashlib.sha256(f"{digest}:{material_class}:{condition}:{key}".encode()).hexdigest()[:12]
                options = (dict(format="%.3e", step=1e-14 if key == "D" else 1e-5)
                           if key in ("D", "beta") else dict(format="%.4g"))
                lo = lower.number_input(f"Minimum {label} ({unit})", value=lo,
                                         key=f"inverse_lower_{field_scope}", persist_state="session", **options)
                hi = upper.number_input(f"Maximum {label} ({unit})", value=hi,
                                         key=f"inverse_upper_{field_scope}", persist_state="session", **options)
                property_bounds[key] = [lo, hi]
                if hi < lo:
                    errors.append(f"{label}: maximum must be at least the minimum.")
            limit_defaults = {"sigma_abs": 10.0, "eps_h": .001, "W_t": 1.0}
            limit_labels = {"sigma_abs": "stress-index magnitude", "eps_h": "hygroscopic strain", "W_t": "absorbed water concentration"}
            limit_columns = st.columns(3, gap="small")
            short_labels = {"sigma_abs": "stress", "eps_h": "swelling", "W_t": "water"}
            for (key, default), column in zip(limit_defaults.items(), limit_columns):
                with column, st.container(key=f"inverse_limit_{key}"):
                    if st.checkbox(f"Limit {short_labels[key]}", key=f"inverse_enable_{key}", persist_state="session",
                                   help=f"Set a maximum {limit_labels[key]} ({OBJECTIVES[key][1]})."):
                        unit = OBJECTIVES[key][1]
                        output_limits[key] = st.number_input(f"Maximum {short_labels[key]} ({unit})", min_value=0.,
                                                             value=default, format="%.3g", step=.0001 if key == "eps_h" else .1,
                                                             key=f"inverse_max_{key}", persist_state="session",
                                                             label_visibility="collapsed")
        config = dict(condition=condition, material_class=None if material_class == all_classes else material_class,
                      application=application, objective=objective, property_bounds=property_bounds,
                      output_limits=output_limits, shortlist_size=int(count), library_source=source, model_version=MODEL_VERSION)
        signature = hashlib.sha256(json.dumps(dict(library=digest, config=config), sort_keys=True).encode()).hexdigest()
        for error in errors:
            st.error(error)
    with st.container(key="inverse_actions"):
        submitted = st.button("Find recommended materials", type="primary", width="stretch",
                              key="inverse_recommend", disabled=bool(errors))
    if submitted:
        try:
            with st.spinner("Calculating eligible materials and ranking the results…"):
                result = calculate_recommendations(digest, config, frame)
            st.session_state["inverse_recommendation"] = dict(signature=signature, result=result)
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            st.session_state.pop("inverse_recommendation", None)
            st.error(str(exc))
    saved = st.session_state.get("inverse_recommendation")
    result = saved["result"] if saved and saved["signature"] == signature else None
    return dict(library=class_rows.loc[class_rows["condition"].eq(condition)], digest=digest,
                config=config, signature=signature, result=result)


def render_result_options(context):
    result = context["result"] if context else None
    st.html('<p class="compact-section-label">Results</p>')
    if result is None:
        st.caption("Set your application conditions and run the recommender to unlock these views.")
    else:
        st.caption(f"{len(result['candidates']):,} recommendations · {result['qualifying']:,} meet requirements · "
                   f"{result['evaluated']:,} materials calculated.")
    views = [
        ("recommendations", "Recommended materials", ":material/table_view:",
         "View the ranked results, inspect a material's stress contributions, and download the shortlist."),
        ("exploration", "Explore material properties", ":material/scatter_plot:",
         "See the library in a 3D property space, with the recommended shortlist highlighted. Choose the three axes."),
        ("heatmap", "Materials heatmap", ":material/grid_on:",
         "Compare stress, swelling and moisture uptake across the recommended materials."),
    ]
    for view, label, icon, description in views:
        with st.container(key=f"inverse_result_{view}"):
            if st.button(label, icon=icon, width="stretch", key=f"inverse_open_{view}", disabled=result is None):
                show_inverse_results(view, context)
            st.caption(description)
    render_result_summary(context)


def render_result_summary(context):
    result = context["result"] if context else None
    with st.container(key="inverse_result_summary"):
        if result is not None and not result["candidates"].empty:
            row = result["candidates"].iloc[0]
            objective = context["config"]["objective"]
            label = {"sigma_abs": "Stress magnitude", "eps_h": "Swelling strain", "W_t": "Absorbed water"}[objective]
            unit = OBJECTIVES[objective][1]
            value = f"{row[objective]:.3g}" + (f" {unit}" if unit != "dimensionless" else "")
            st.html('<p class="compact-section-label">Top recommendation</p>')
            st.html(f'<p class="inverse-summary-name">{escape(str(row["name"]))}</p>')
            ranked, stress, saturation = st.columns(3, gap="small")
            ranked.metric(label, value)
            stress.metric("Signed stress", f"{row['sigma_index']:+.2f} MPa")
            saturation.metric("Moisture saturation", f"{row['Mt_Minf'] * 100:.2f} %")
            st.caption(f"Ranked by {OBJECTIVES[objective][0].lower()} · {result['qualifying']:,} materials meet requirements.")
        else:
            st.html('<p class="compact-section-label">Current application</p>')
            if context is None:
                st.caption("Choose or upload a material library to configure your application.")
                return
            config = context["config"]
            application = config["application"]
            st.caption(config["condition"])
            temperature, soak, thickness, copper = st.columns(4, gap="small")
            temperature.metric("Temperature change", f"{application['delta_T']:+g} K")
            soak.metric("Soak time", f"{application['t_hours']:g} h")
            thickness.metric("Thickness", f"{application['h_mm']:g} mm")
            copper.metric("Copper CTE", f"{application['alpha_cu']:g} ppm/K")
            if result is not None:
                st.caption("No materials meet these requirements. Adjust the filters or application conditions and run again.")


def show_inverse_results(view, context):
    titles = {"exploration": "Explore material properties", "recommendations": "Recommended Materials",
              "heatmap": "Materials heatmap"}

    @st.dialog(titles[view], width="large")
    def render_dialog():
        with st.container(key="inverse_results_modal"):
            if view == "exploration":
                render_property_exploration(context["library"], context["result"]["candidates"])
            elif view == "recommendations":
                render_recommendations(context["result"], context["config"], context["digest"], context["signature"],
                                       show_heading=False)
            else:
                render_shortlist_heatmap(context["result"]["candidates"])

    render_dialog()


def render_property_exploration(library, candidates):
    axes = st.columns(3, gap="small")
    selected = [column.selectbox(f"{axis}-axis", MATERIAL, index=index,
                                format_func=lambda key: f"{PARAMETERS[key][0]} ({PARAMETERS[key][1]})",
                                key=f"inverse_explore_{axis.lower()}")
                for index, (axis, column) in enumerate(zip("XYZ", axes))]
    finite = np.isfinite(library[selected].to_numpy(dtype=float)).all(axis=1)
    points = library.loc[finite]
    total = len(points)
    if total > 3000:
        points = points.sample(3000, random_state=12345)
    if points.empty:
        st.info("No material records have finite values for all three selected properties.")
        return
    st.plotly_chart(property_scatterplot(points, candidates, selected), width="stretch", key="inverse_property_scatter")
    st.caption("Points show material records for the selected class and conditioning environment. "
               "Property bounds and output requirements determine the highlighted shortlist. "
               "Distances reflect the chosen properties and units; this view does not determine the ranking.")
    if total > len(points):
        st.caption(f"Showing a reproducible sample of {len(points):,} of {total:,} library records, plus the full shortlist.")
    if (~finite).any():
        st.caption(f"{int((~finite).sum()):,} records without finite values for the selected axes are omitted.")


def property_scatterplot(points, candidates, axes):
    figure = go.Figure()
    labels = [f"{PARAMETERS[key][0]} ({PARAMETERS[key][1]})" for key in axes]
    for data, name, color, size, symbol in [
        (points, "Library", "#92b8ad", 4, "circle"),
        (candidates, "Recommended shortlist", "#e3ac88", 7, "diamond"),
    ]:
        if data.empty:
            continue
        figure.add_trace(go.Scatter3d(
            x=data[axes[0]], y=data[axes[1]], z=data[axes[2]], name=name, mode="markers",
            marker=dict(color=color, size=size, symbol=symbol, opacity=.85),
            customdata=data[["name", "material_class"]].to_numpy(),
            hovertemplate=("<b>%{customdata[0]}</b><br>%{customdata[1]}<br>" +
                           f"{labels[0]}: %{{x:.4g}}<br>{labels[1]}: %{{y:.4g}}<br>{labels[2]}: %{{z:.4g}}<extra></extra>"),
        ))
    axis_titles = {"beta": "CHE (m³/kg)", "D": "Diffusivity (m²/s)", "Csat": "Csat (mol/m³)"}
    scene_axes = {
        f"{axis}axis": dict(title=dict(text=label, font=dict(size=11)), tickfont=dict(size=10),
                            backgroundcolor="#173239", gridcolor="rgba(214,239,229,.15)",
                            zerolinecolor="rgba(214,239,229,.25)")
        for axis, label in zip("xyz", [axis_titles.get(key, label) for key, label in zip(axes, labels)])
    }
    figure.update_layout(height=360, margin=dict(l=10, r=10, b=25, t=40),
                         legend=dict(orientation="h", x=0, y=1.04, yanchor="bottom"),
                         scene=dict(**scene_axes, aspectmode="cube", uirevision="inverse-property-space",
                                    camera=dict(eye=dict(x=1.8, y=1.8, z=1.5))))
    return figure


def render_shortlist_heatmap(candidates):
    if candidates.empty:
        st.info("No materials meet these conditions and requirements. Review the filters or choose another library.")
        return
    st.caption("Compare the same shortlist in ranking order. Each column has its own scale: "
               "darker green means a larger nonnegative value; signed stresses use orange for negative "
               "and green for positive values. Hover over a cell for its value and units.")
    st.plotly_chart(recommendation_heatmap(candidates), width="stretch", key="inverse_shortlist_heatmap")
    st.caption("A small net stress index can result from moisture and thermal contributions cancelling. "
               "Colours show relative values, not material suitability or safety.")


def render_recommendations(result, config, digest, signature, show_heading=True):
    candidates = result["candidates"]
    matched, calculated, qualifying, shown = st.columns(4)
    matched.metric("Matching class / environment", f"{result['total']:,}")
    calculated.metric("Materials calculated", f"{result['evaluated']:,}")
    qualifying.metric("Meet requirements", f"{result['qualifying']:,}")
    shown.metric("Results shown", len(candidates))
    st.caption(f"{result['invalid']:,} invalid property records · {result['property_excluded']:,} outside property bounds · "
               f"{result['requirements_excluded']:,} fail output requirements · {result['numerical_invalid']:,} nonfinite model results excluded.")
    if candidates.empty:
        st.info("No materials meet these conditions and requirements. Review the filters or choose another library.")
        return
    application = config["application"]
    with st.container(key="inverse_recommendation_heading"):
        if show_heading:
            st.subheader("Recommended Materials")
        st.caption(f"Ranked by {OBJECTIVES[config['objective']][0].lower()} · {config['condition']} · "
                   f"ΔT {application['delta_T']:+g} K · soak {application['t_hours']:g} h · thickness {application['h_mm']:g} mm · "
                   f"copper CTE {application['alpha_cu']:g} ppm/K.")
    labels = {"rank": "Rank", "name": "Material", "material_class": "Class", "sigma_abs": "Magnitude (MPa)",
              "sigma_index": "Signed index (MPa)", "sigma_hygro": "Moisture contribution (MPa)",
              "sigma_thermal": "Thermal contribution (MPa)", "eps_h": "Hygroscopic strain",
              "W_t": "Absorbed water (kg/m³)", "meets_requirements": "Meets requirements"}
    st.dataframe(candidates[list(labels)].rename(columns=labels), hide_index=True, width="stretch", column_config={
        **{labels[key]: st.column_config.NumberColumn(format="%.3f")
           for key in ("sigma_abs", "sigma_index", "sigma_hygro", "sigma_thermal", "W_t")},
        "Hygroscopic strain": st.column_config.NumberColumn(format="%.3e"),
    })
    with st.container(key="inverse_material_inspector"):
        st.subheader("Inspect a recommended material")
        # The signature gives changing result sets their own inspection state.
        selected = st.selectbox("Recommended material", candidates.index.tolist(),
                                format_func=lambda index: f"{index + 1}. {candidates.loc[index, 'name']}",
                                key=f"inverse_selected_{signature[:12]}", label_visibility="collapsed")
        row = candidates.loc[selected]
        render_summary(row, key="inverse_material")
        render_contributions(row, key="inverse_material")
        with st.expander("Material properties"):
            st.dataframe(pd.DataFrame([dict(Property=PARAMETERS[key][0], Value=row[key], Unit=PARAMETERS[key][1])
                                       for key in MATERIAL]), hide_index=True, width="stretch")
    st.download_button("Download recommendations (CSV)", candidates.to_csv(index=False).encode(),
                       "recommended_materials.csv", "text/csv", key="inverse_shortlist_download")


def recommendation_heatmap(candidates):
    """Compare current results with independent column scales and raw-value hover details."""
    metrics = [
        ("sigma_abs", "Stress<br>magnitude<br>(MPa)", "MPa", False),
        ("sigma_index", "Signed<br>index<br>(MPa)", "MPa", True),
        ("sigma_hygro", "Moisture<br>stress<br>(MPa)", "MPa", False),
        ("sigma_thermal", "Thermal<br>stress<br>(MPa)", "MPa", True),
        ("eps_h", "Swelling<br>strain<br>(−)", "dimensionless", False),
        ("W_t", "Absorbed<br>water<br>(kg/m³)", "kg/m³", False),
        ("Mt_Minf", "Moisture<br>uptake<br>(%)", "%", False),
    ]
    rows = candidates.to_dict("records")
    row_labels = [f"{row['rank']}. {row['name']}" for row in rows]
    figure = go.Figure()
    for column, (key, label, unit, signed) in enumerate(metrics):
        values = [float(row[key]) * (100 if key == "Mt_Minf" else 1) for row in rows]
        scale = max((abs(value) for value in values), default=0) or 1.0
        text = [[f"{value:.3g}"] for value in values]
        details = [[[row["name"], row["material_class"], value, unit, row["rank"], label.replace("<br>", " ")]]
                   for row, value in zip(rows, values)]
        figure.add_trace(go.Heatmap(
            # Explicit cell edges keep separate traces on the same column grid.
            x=[column - .5, column + .5], y=row_labels, z=[[value / scale] for value in values],
            coloraxis="coloraxis2" if signed else "coloraxis",
            text=text, texttemplate="%{text}", textfont=dict(size=11),
            customdata=details, xgap=3, ygap=3,
            hovertemplate=("<b>%{customdata[0]}</b><br>Rank: %{customdata[4]}"
                           "<br>Class: %{customdata[1]}<br>%{customdata[5]}: %{customdata[2]:.4g} %{customdata[3]}<extra></extra>"),
        ))
    figure.update_layout(
        height=max(300, 140 + 42 * len(rows)), margin=dict(l=10, r=100, t=75, b=15),
        xaxis=dict(type="linear", range=[-.5, len(metrics) - .5], side="top", tickangle=0,
                   fixedrange=True, tickmode="array", tickvals=list(range(len(metrics))),
                   ticktext=[metric[1] for metric in metrics], tickfont=dict(size=11),
                   showgrid=False, zeroline=False),
        yaxis=dict(autorange="reversed", fixedrange=True, automargin=True, showgrid=False),
        coloraxis=dict(cmin=0, cmax=1, colorscale=[[0, "#f0f6f2"], [1, "#286553"]],
                       colorbar=dict(title="Magnitude", tickvals=[0, 1], ticktext=["0", "Max"],
                                     len=.45, y=.76, thickness=12)),
        coloraxis2=dict(cmin=-1, cmax=1,
                        colorscale=[[0, "#a7643b"], [.5, "#f7f8f5"], [1, "#368575"]],
                        colorbar=dict(title="Stress sign", tickvals=[-1, 0, 1], ticktext=["−Max", "0", "+Max"],
                                      len=.45, y=.23, thickness=12)),
    )
    return figure
