"""Independent application-based material recommendations and a laminate gallery."""
import hashlib
from io import BytesIO
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from inverse_analysis import (
    APPLICATION_DEFAULTS, OBJECTIVES, prepare_library, recommend_materials, records_from_csv,
)
from laminate_showcase import render_showcase
from sensitivity_analysis import MATERIAL, MODEL_VERSION, PARAMETERS


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
    st.caption("Find materials for your application, then explore the structures behind their properties.")
    render_recommender(records)
    st.divider()
    render_showcase()


def select_available(label, options, key):
    if st.session_state.get(key) not in options:
        st.session_state.pop(key, None)
    return st.selectbox(label, options, key=key, persist_state="session")


def render_recommender(records):
    st.header("Material recommender")
    st.caption("Set your application conditions and requirements. Rank the library by predicted stress, swelling or moisture content.")
    with st.container(border=True):
        source = st.radio("Material library", ["Included presets", "Upload CSV"], horizontal=True,
                          key="inverse_library_source", persist_state="session")
        upload = None
        if source == "Upload CSV":
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
        st.caption(f"{len(frame):,} material/condition records loaded.")
        classes = list(dict.fromkeys(frame["material_class"]))
        all_classes = "All material classes"
        material_class = select_available("Material class", [*classes, all_classes], "inverse_material_class")
        class_rows = frame if material_class == all_classes else frame.loc[frame["material_class"].eq(material_class)]
        condition = select_available("Conditioning environment", list(dict.fromkeys(class_rows["condition"])),
                                     "inverse_recommend_condition")
        st.caption("Conditioning selects each material's tabulated properties. Diffusivity stays fixed at that condition. "
                   "Enter the temperature change for thermal mismatch separately below.")

        st.markdown("#### Application conditions")
        temperature, time = st.columns(2)
        geometry, copper = st.columns(2)
        application = dict(
            delta_T=temperature.number_input("Change in Temperature (°C or K)", value=APPLICATION_DEFAULTS["delta_T"],
                                              step=5.0, key="inverse_app_delta_T", persist_state="session"),
            t_hours=time.number_input("Soak Time (hours)", min_value=0., value=APPLICATION_DEFAULTS["t_hours"],
                                      step=1.0, key="inverse_app_t_hours", persist_state="session"),
            h_mm=geometry.number_input("Laminate Thickness (mm)", min_value=.001, value=APPLICATION_DEFAULTS["h_mm"],
                                        step=.1, key="inverse_app_h_mm", persist_state="session"),
            alpha_cu=copper.number_input("Copper Foil CTE (ppm/K)", value=APPLICATION_DEFAULTS["alpha_cu"],
                                          step=.5, key="inverse_app_alpha_cu", persist_state="session"),
        )
        objective_col, count_col = st.columns([2, 1])
        objective = objective_col.selectbox("Rank materials by", list(OBJECTIVES),
                                            format_func=lambda key: OBJECTIVES[key][0],
                                            key="inverse_objective", persist_state="session")
        count = count_col.number_input("Maximum results", min_value=1, max_value=10, value=10,
                                        key="inverse_result_count", persist_state="session")
        property_bounds, output_limits = {}, {}
        with st.expander("Optional requirements"):
            st.caption("Only materials meeting all enabled requirements appear in the results. These are application requirements, not safety limits.")
            keys = st.multiselect("Material properties to constrain", MATERIAL,
                                  format_func=lambda key: PARAMETERS[key][0], key="inverse_property_filters", persist_state="session")
            available = class_rows.loc[class_rows["condition"].eq(condition)]
            errors = []
            for key in keys:
                label, unit, default_lo, default_hi, _ = PARAMETERS[key]
                numeric = available[key].replace([float("inf"), -float("inf")], float("nan")).dropna()
                lo = float(numeric.min()) if len(numeric) else default_lo
                hi = float(numeric.max()) if len(numeric) else default_hi
                lower, upper = st.columns(2)
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
            for key, default in limit_defaults.items():
                enabled, value = st.columns(2)
                if enabled.checkbox(f"Limit {limit_labels[key]}", key=f"inverse_enable_{key}", persist_state="session"):
                    unit = OBJECTIVES[key][1]
                    output_limits[key] = value.number_input(f"Maximum {limit_labels[key]} ({unit})", min_value=0.,
                                                             value=default, format="%.3g", step=.0001 if key == "eps_h" else .1,
                                                             key=f"inverse_max_{key}", persist_state="session")
        with st.expander("How ranking works"):
            st.write("The recommender filters by material class, conditioning and your property bounds, then calculates "
                     "moisture uptake, swelling and stress for every eligible record using the same application conditions. "
                     "Materials meeting your output requirements are ranked by the selected objective, smallest first.")
            st.write("Large libraries are calculated in batches. Repeated requests are cached, and only the best "
                     "results are retained for display. The ranking uses all six material properties.")
            st.write("Moisture enters through one exposed face of the laminate. The opposite face interfaces "
                     "with copper foil and is treated as impermeable; the diffusion term uses (2h)² for laminate thickness h.")
        config = dict(condition=condition, material_class=None if material_class == all_classes else material_class,
                      application=application, objective=objective, property_bounds=property_bounds,
                      output_limits=output_limits, shortlist_size=int(count), library_source=source, model_version=MODEL_VERSION)
        signature = hashlib.sha256(json.dumps(dict(library=digest, config=config), sort_keys=True).encode()).hexdigest()
        for error in errors:
            st.error(error)
        if st.button("Find recommended materials", type="primary", key="inverse_recommend", disabled=bool(errors)):
            try:
                with st.spinner("Calculating eligible materials and ranking the results…"):
                    result = calculate_recommendations(digest, config, frame)
                st.session_state["inverse_recommendation"] = dict(signature=signature, result=result)
            except (ValueError, TypeError, KeyError, OverflowError) as exc:
                st.session_state.pop("inverse_recommendation", None)
                st.error(str(exc))
    recommendation = st.session_state.get("inverse_recommendation")
    if not recommendation or recommendation["signature"] != signature:
        st.caption("Run the recommender to see results for these inputs.")
        return
    render_recommendations(recommendation["result"], config, digest, signature)


def render_recommendations(result, config, digest, signature):
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
    with st.expander("Compare recommended materials"):
        st.caption("Compare the same shortlist in ranking order. Each column has its own scale: "
                   "darker green means a larger nonnegative value; signed stresses use orange for negative "
                   "and green for positive values. Hover over a cell for its value and units.")
        st.plotly_chart(recommendation_heatmap(candidates), width="stretch", key="inverse_shortlist_heatmap")
        st.caption("A small net stress index can result from moisture and thermal contributions cancelling. "
                   "Colours show relative values, not material suitability or safety.")
    with st.expander("Inspect a recommended material"):
        # The signature gives changing result sets their own inspection state.
        selected = st.selectbox("Recommended material", candidates.index.tolist(),
                                format_func=lambda index: f"{index + 1}. {candidates.loc[index, 'name']}",
                                key=f"inverse_selected_{signature[:12]}")
        row = candidates.loc[selected]
        figure = go.Figure(go.Bar(x=["Moisture", "Thermal", "Net index"],
                                  y=[row["sigma_hygro"], row["sigma_thermal"], row["sigma_index"]],
                                  marker_color=["#368575", "#c87d55", "#7f8165"]))
        figure.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="MPa")
        st.plotly_chart(figure, width="stretch", key="inverse_shortlist_contributions")
        uptake, water, swelling = st.columns(3)
        uptake.metric("Fractional moisture uptake", f"{row['Mt_Minf']:.2%}")
        water.metric("Absorbed water (kg/m³)", f"{row['W_t']:.3g}")
        swelling.metric("Hygroscopic strain", f"{row['eps_h']:.3e}")
        st.dataframe(pd.DataFrame([dict(Property=PARAMETERS[key][0], Value=row[key], Unit=PARAMETERS[key][1])
                                   for key in MATERIAL]), hide_index=True, width="stretch")
    results_col, settings_col = st.columns(2)
    results_col.download_button("Download recommendations (CSV)", candidates.to_csv(index=False).encode(),
                                "recommended_materials.csv", "text/csv", key="inverse_shortlist_download")
    settings_col.download_button("Download application settings (JSON)",
                                 json.dumps(dict(library_digest=digest, **config), indent=2).encode(),
                                 "recommendation_settings.json", "application/json", key="inverse_settings_download")


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
