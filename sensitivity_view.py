"""Sobol sensitivity controls and plots; rendered only in tab 2."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sensitivity_analysis import (
    EXPOSURE, FIXED_META, MATERIAL, OUTPUTS as RESPONSE_OPTIONS, PARAMETERS, STUDIES,
    default_ranges, excel_report, material_ranges, run_analysis,
)
from typography import document_fonts


@st.cache_data(show_spinner=False, max_entries=6)
def calculate(config):
    return run_analysis(config["baseline"], config["ranges"], config["diffusion"],
                        target=config["target"], N=config["N"], seed=config["seed"])


def render_parameter_overview():
    """Explain the nine varied inputs in an isolated, animated guide."""
    st.header("What are we varying?")
    st.caption("Follow the two strain pathways, then see how stiffness turns their combined mismatch into a stress index.")
    overview = (Path(__file__).parent / "assets" / "sensitivity_overview.html").read_text(encoding="utf-8")
    st.iframe(document_fonts(overview), height="content")


def render_nominal_inputs(name, values):
    """Own widget namespace: no Stress Analyser state is read or changed."""
    scope = hashlib.sha256(name.encode()).hexdigest()[:12]
    p = dict(values)
    columns = st.columns(3)
    for i, (key, (label, unit)) in enumerate(FIXED_META.items()):
        options = dict(value=float(p[key]), key=f"sens_nominal_{scope}_{key}", persist_state="session")
        if key in ("D", "beta"):
            options.update(min_value=0.0, format="%.3e")
        elif key in ("Csat", "t_hours"):
            options.update(min_value=0.0)
        elif key in ("E", "h_mm"):
            options.update(min_value=0.001)
        elif key == "nu":
            options.update(min_value=-0.99, max_value=0.499, format="%.3f", step=0.01)
        p[key] = columns[i % 3].number_input(f"{label} ({unit})", **options)
    return p


def render_sensitivity(records):
    st.session_state["sensitivity_current_signature"] = None
    with st.container(horizontal=True, wrap=False, vertical_alignment="center",
                      gap="small", key="sensitivity_analysis_header"):
        st.markdown(
            '<div class="exploration-icon" aria-hidden="true">'
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
            '<path d="M4 3v17h17M9 16V9M14 16V5M19 16v-4" />'
            '</svg></div>', unsafe_allow_html=True, width="content",
        )
        st.title("Sensitivity Analysis")
    st.caption("Choose materials and study how exposure conditions or property variations affect moisture uptake, swelling and stress.")
    render_parameter_overview()
    st.header("Which parameters matter most?")
    st.caption("Choose a response and parameter ranges, then use Sobol analysis to see which inputs "
               "contribute most to variation in that response.")
    presets = {r["name"]: r for r in records}
    custom = "Custom material / manual entry"
    custom_values = dict(E=25000., nu=.2, alpha_lam=12., alpha_cu=16.5,
                         D=1e-12, Csat=150., beta=1e-4, h_mm=1.6, t_hours=24., delta_T=60.)
    with st.container(border=True):
        scope = st.segmented_control("Material selection mode", ["One material", "Compare materials"],
                                     default="One material", required=True, label_visibility="collapsed",
                                     key="sens_scope", persist_state="session")
        if scope == "One material":
            names = [st.selectbox("Material preset", [*presets, custom], key="sens_preset", persist_state="session")]
        else:
            names = st.multiselect("Material presets to compare (up to four)", list(presets),
                                   default=list(presets)[:2], max_selections=4,
                                   key="sens_compare_presets", persist_state="session")
        st.caption("Each preset identifies its material and conditioning environment.")
        if not names:
            st.info("Choose at least one material.")
            return
        study = st.radio("Study", STUDIES, index=1, horizontal=True, key="sens_study", persist_state="session")
        if st.session_state.get("sens_output") not in RESPONSE_OPTIONS:
            st.session_state["sens_output"] = "sigma_index"
        target = st.selectbox("Response to analyse", list(RESPONSE_OPTIONS),
                              format_func=lambda key: f"{RESPONSE_OPTIONS[key][0]} ({RESPONSE_OPTIONS[key][1]})",
                              key="sens_output", persist_state="session")
        st.caption("Sobol analysis estimates how much each input and its interactions contribute to response variance.")
        range_mode = st.radio("Parameter ranges", ["Around each material", "Broad exploratory ranges"],
                              horizontal=True, key="sens_range_basis", persist_state="session")
        percent, factor = 10.0, 2.0
        if range_mode == "Around each material":
            width_col, factor_col = st.columns(2)
            percent = width_col.number_input("Variation around nominal values (±%)", min_value=0.1,
                                             max_value=99.0, value=10.0, key="sens_percent", persist_state="session")
            factor = factor_col.number_input("Diffusivity factor (divide / multiply)", min_value=1.01,
                                              value=2.0, key="sens_D_factor", persist_state="session")
            st.caption("Assumed screening ranges, not measured tolerances. D uses a log-uniform range; "
                       "other selected inputs use uniform ranges. Bounds remain editable below.")
        else:
            st.caption("Explore the same broad bounds for every material. Varied properties use those bounds, "
                       "rather than the preset's nominal values.")
        group = STUDIES.index(study)
        allowed = EXPOSURE if group == 0 else MATERIAL if group == 1 else list(PARAMETERS)
        defaults = [key for key in allowed if key != "nu"] if range_mode == "Around each material" else allowed
        keys = st.multiselect("Parameters to vary", allowed, default=defaults,
                              format_func=lambda key: PARAMETERS[key][0],
                              key=f"sens_parameters_{group}_{range_mode}", persist_state="session")
        st.caption("Unselected inputs remain fixed at their nominal values. Copper CTE stays fixed. "
                   "The selected input distributions are independent.")
        mode = "Fixed D"
        if group == 0:
            mode = st.radio("Diffusivity model", ["Fixed D", "Arrhenius"], horizontal=True,
                            key="sens_diffusion_model", persist_state="session")
        diffusion = {"mode": mode}
        if mode == "Arrhenius":
            with st.expander("Temperature-dependent diffusion constants", expanded=True):
                d0, ea, tref = st.columns(3)
                diffusion["D0"] = d0.number_input("Pre-exponential factor D₀ (m²/s)", min_value=1e-30,
                                                 value=6.31e-7, format="%.3e", key="sens_D0", persist_state="session")
                diffusion["Ea"] = ea.number_input("Activation energy (J/mol)", min_value=0.0,
                                                 value=37560.0, step=100.0, key="sens_Ea", persist_state="session")
                diffusion["T_ref_C"] = tref.number_input("Reference temperature (°C)", min_value=-273.14,
                                                        value=25.0, key="sens_T_ref_C", persist_state="session")
                st.caption("D is calculated from reference temperature + ΔT, replacing the preset's D. "
                           "These editable diffusion constants apply to every material in this run; they are illustrative.")
        else:
            st.caption("D uses the preset value or its edited bounds. ΔT affects thermal mismatch only; "
                       "humidity is represented by the chosen conditioning preset.")

        baselines = {}
        with st.expander("Nominal material and exposure inputs", expanded=custom in names):
            st.caption("Edit nominal values here. Selected parameters vary over the bounds below; all others stay fixed. "
                       "For comparisons, each material has its own inputs and bounds.")
            for name, tab in zip(names, st.tabs(names)):
                with tab:
                    values = custom_values if name == custom else presets[name]["properties"]
                    baselines[name] = render_nominal_inputs(name, values)
        if not keys:
            st.info("Select at least one parameter to set up a sensitivity study.")
            return
        initial = {}
        for name in names:
            try:
                initial[name] = (material_ranges(keys, baselines[name], percent, factor)
                                 if range_mode == "Around each material" else default_ranges(keys))
            except ValueError as exc:
                st.info(f"{name}: {exc}")
                return
        st.markdown("#### Parameter ranges")
        st.caption("CTE: ppm/K · modulus: MPa · exposure: hours · thickness: mm. "
                   "Comparison rankings depend on each material's displayed ranges and fixed inputs.")
        saved_ranges = st.session_state.setdefault("sensitivity_range_values", {})
        editors, ranges = {}, {}
        with st.form("sens_run_form"):
            for name, tab in zip(names, st.tabs(names)):
                with tab:
                    fingerprint = json.dumps(dict(name=name, study=study, basis=range_mode,
                                                  initial=initial[name].to_dict("records")), sort_keys=True)
                    editor_key = "sens_ranges_" + hashlib.sha256(fingerprint.encode()).hexdigest()[:16]
                    editors[name] = editor_key
                    ranges[name] = st.data_editor(
                        saved_ranges.get(editor_key, initial[name]), hide_index=True, width="stretch",
                        disabled=["Parameter", "Label", "Unit"],
                        column_config={
                            "Parameter": None,
                            "Label": st.column_config.TextColumn("Parameter"),
                            "Lower": st.column_config.NumberColumn("Lower bound", format="%.4g", required=True),
                            "Upper": st.column_config.NumberColumn("Upper bound", format="%.4g", required=True),
                            "Distribution": st.column_config.SelectboxColumn(options=["uniform", "log-uniform"], required=True),
                        }, key=editor_key,
                    )
            size, scramble = st.columns(2)
            N = size.select_slider("Base sample size N", options=[512, 1024, 2048, 4096, 8192],
                                   value=4096, key="sens_N", persist_state="session")
            cost = N * (len(keys) + 2)
            note = "500 bootstrap resamples per material · 95% intervals"
            seed = scramble.number_input("Sampling seed", min_value=0, max_value=1000000,
                                          value=12345, step=1, key="sens_seed", persist_state="session")
            st.caption(f"{cost * len(names):,} model evaluations across {len(names)} material(s) · {note}")
            submitted = st.form_submit_button("Run sensitivity analysis", type="primary")

    configs = [dict(study=study, material=name, condition=presets[name]["condition"] if name != custom else "Manual inputs",
                    baseline_source="Material preset" if name != custom else "Manual inputs",
                    baseline=baselines[name], range_basis=range_mode, ranges=ranges[name].to_dict("records"),
                    diffusion=diffusion, target=target, method="Sobol", N=int(N),
                    seed=int(seed), bootstrap_resamples=500) for name in names]
    signature = json.dumps(configs, sort_keys=True)
    if submitted:
        for name in names:
            saved_ranges[editors[name]] = ranges[name].copy()
        runs = []
        with st.spinner(f"Calculating Sobol sensitivity for {len(names)} material(s)…"):
            for config in configs:
                try:
                    runs.append(dict(config=config, result=calculate(config)))
                except (ValueError, TypeError, OverflowError) as exc:
                    runs.append(dict(config=config, error=str(exc)))
        st.session_state["sensitivity_batch"] = dict(signature=signature, runs=runs)
        st.session_state.pop("sensitivity_excel", None)
    saved = st.session_state.get("sensitivity_batch")
    if not saved:
        st.info("Set your ranges and run the analysis to compare parameter influence.")
        return
    if saved["signature"] != signature:
        st.info("Your settings have changed. Run the analysis to update the results.")
        return
    for run in saved["runs"]:
        if "error" in run:
            st.warning(f"{run['config']['material']}: {run['error']}")
    successes = [run for run in saved["runs"] if "result" in run]
    if not successes:
        return
    if len(successes) > 1:
        render_comparison(successes, configs)
        available = [run["config"]["material"] for run in successes]
        if st.session_state.get("sens_inspect_material") not in available:
            st.session_state.pop("sens_inspect_material", None)
        inspected = st.selectbox("Inspect material results", available, key="sens_inspect_material")
        index = available.index(inspected)
    else:
        index = 0
    active = successes[index]
    config = active["config"]
    active_signature = signature + config["material"]
    st.session_state["sensitivity_current_signature"] = active_signature
    st.session_state["sensitivity_result"] = dict(signature=active_signature, **active)
    st.caption(f"Results for {config['material']} · Sobol · {RESPONSE_OPTIONS[target][0]}")
    render_results(active["result"], config, active_signature)


def render_comparison(runs, configs):
    st.subheader("Material Comparison")
    table = pd.concat([run["result"]["indices"].assign(Material=run["config"]["material"],
                                                     Condition=run["config"]["condition"]) for run in runs], ignore_index=True)
    fig = go.Figure()
    colors = ["#368575", "#c87d55", "#7f8165", "#92b8ad"]
    for run, color in zip(runs, colors):
        data = run["result"]["indices"]
        fig.add_trace(go.Bar(x=data["Label"], y=data["ST"], name=run["config"]["material"], marker_color=color))
    fig.update_layout(barmode="group", height=400, margin=dict(t=20, b=20),
                      yaxis_title="Total effect · ST",
                      legend=dict(orientation="h", y=-.35))
    st.plotly_chart(fig, width="stretch", key="sens_comparison_chart")
    st.caption("Each material is analysed separately with the same sampling seed and response. "
               "Its own nominal inputs and displayed bounds determine the ranking. "
               "ST describes a share of each material's own response variance, not its absolute stress level.")
    with st.expander("Comparison table and downloads"):
        st.dataframe(table.drop(columns="Parameter"), hide_index=True, width="stretch")
        st.download_button("All material rankings (CSV)", table.to_csv(index=False).encode(),
                           "material_sensitivity_comparison.csv", "text/csv", key="sens_comparison_csv")
        st.download_button("All material settings (JSON)", json.dumps(configs, indent=2).encode(),
                           "material_sensitivity_settings.json", "application/json", key="sens_comparison_settings")


def render_results(result, config, signature):
    with st.expander("Inputs and ranges used"):
        st.dataframe(pd.DataFrame([dict(Parameter=label, Nominal=config["baseline"][key], Unit=unit,
                                       Treatment="Varied" if key in {r["Parameter"] for r in config["ranges"]} else
                                       "Calculated from temperature" if key == "D" and config["diffusion"]["mode"] == "Arrhenius" else "Fixed")
                                   for key, (label, unit) in FIXED_META.items()]), hide_index=True, width="stretch")
        st.dataframe(pd.DataFrame(config["ranges"]).drop(columns="Parameter"), hide_index=True, width="stretch")
    table = result["indices"].sort_values("ST", ascending=False)
    unit = RESPONSE_OPTIONS[config["target"]][1]
    st.subheader("Parameter Importance")
    first, total, interactions = st.columns(3)
    with first.container(border=True):
        st.markdown("**S1 · Main effect**")
        st.write("The share of response variance associated with one input's main effect.")
    with total.container(border=True):
        st.markdown("**ST · Total effect**")
        st.write("The main effect plus every interaction involving that input. Ranked from largest to smallest below.")
    with interactions.container(border=True):
        st.markdown("**ST − S1 · Interactions**")
        st.write("A gap between ST and S1 suggests the input acts in combination with other inputs.")

    fig = go.Figure()
    for index, label, color in [("S1", "Main effect · S1", "#92b8ad"), ("ST", "Total effect · ST", "#368575")]:
        values = table[index].to_numpy()
        fig.add_trace(go.Bar(
            y=table["Label"], x=values, orientation="h", name=label, marker_color=color,
            error_x=dict(type="data", symmetric=False,
                         array=np.maximum(0, table[f"{index}_high"] - values),
                         arrayminus=np.maximum(0, values - table[f"{index}_low"])),
        ))
    fig.update_layout(barmode="group", height=max(340, 65 * len(table)),
                      xaxis_title="Fraction of selected response variance", yaxis=dict(autorange="reversed"),
                      margin=dict(t=25, b=30, l=15, r=25), legend=dict(orientation="h", y=1.12))
    st.plotly_chart(fig, width="stretch")
    top = table.iloc[0]
    st.info(f"Largest estimated total effect: **{top['Label']}** (ST ≈ {top['ST']:.3f}) for these ranges and fixed inputs.")
    st.caption("Error bars show paired bootstrap 95% intervals. Sampling noise can put estimates outside 0–1. "
               "Total effects overlap and should not be added as independent percentages.")
    with st.expander("Further information", expanded=False):
        st.markdown("#### Detailed parameter indices")
        st.dataframe(table.drop(columns="Parameter"), hide_index=True, width="stretch")
        st.caption(f"Largest change in ST between N/2 and N: {table['ST change'].max():.3f}. "
                   "Increase N or change the sampling seed to check uncertain rankings. This comparison does not guarantee convergence.")

        st.markdown("#### Response and intermediate output summary")
        st.dataframe(result["summary"], hide_index=True, width="stretch")
        st.caption(f"Statistics use the independent A and B base samples. Selected response units: {unit}.")
        st.markdown("#### Sampled inputs and calculated outputs")
        st.dataframe(result["samples"].head(500), hide_index=True, width="stretch")
        st.caption(f"Preview of the first 500 of {len(result['samples']):,} model evaluations. Downloads include all rows.")
        st.markdown("#### Model and method")
        st.latex(r"\sigma_{\mathrm{index}}=\frac{E}{1-\nu}\left[\beta W_t+(\alpha_{\mathrm{lam}}-\alpha_{\mathrm{Cu}})\Delta T\right]")
        st.write("The stress index retains the sign of the combined moisture and thermal contributions. Uptake starts dry and uses "
                 "one-dimensional plane-sheet diffusion with both faces exposed. Moisture uptake is calculated "
                 "from diffusivity, time, and thickness; Csat sets equilibrium concentration.")
        st.write("A 60-term Fickian series is used with a short-time approximation near zero uptake. "
                 "Saturation concentration and expansion coefficients remain constant with temperature. "
                 "The model calculates an indicator, not interface traction or delamination safety.")
        st.markdown("Scrambled Sobol sequence, Saltelli cross-sampling, centered first-order and Jansen "
                    "total-order estimators. [Method reference](https://salib.readthedocs.io/en/latest/_modules/SALib/analyze/sobol.html).")

    render_downloads(result, config, signature)


def render_downloads(result, config, signature):
    table = result["indices"].sort_values("ST", ascending=False)
    with st.expander("Download the study", expanded=False):
        indices, samples, settings = st.columns(3)
        indices.download_button("Sobol rankings (CSV)", table.to_csv(index=False).encode(), "sobol_indices.csv", "text/csv", key="sens_indices_download")
        samples.download_button("All samples (CSV)", result["samples"].to_csv(index=False).encode(), "sobol_samples.csv", "text/csv", key="sens_samples_download")
        settings.download_button("Study settings (JSON)", json.dumps(config, indent=2).encode(), "sobol_settings.json", "application/json", key="sens_settings_download")
        if st.button("Prepare Excel workbook", key="sens_prepare_excel"):
            try:
                with st.spinner("Preparing workbook with settings, samples, and results…"):
                    st.session_state["sensitivity_excel"] = (signature, excel_report(result, config))
            except ImportError:
                st.error("Excel export requires openpyxl. CSV downloads are available above.")
        workbook = st.session_state.get("sensitivity_excel")
        if workbook and workbook[0] == signature:
            st.download_button("Download Excel workbook", workbook[1], "sensitivity_study.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="sens_excel_download")
