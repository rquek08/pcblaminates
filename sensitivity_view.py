"""Sensitivity section controls and plots; rendered only in tab 2."""
import hashlib
import json

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from sensitivity_analysis import (
    EXPOSURE, FIXED_META, MATERIAL, OUTPUTS, PARAMETERS, STUDIES,
    default_ranges, excel_report, reference_baseline, run_analysis,
)


@st.cache_data(show_spinner=False, max_entries=6)
def calculate(config):
    return run_analysis(config["baseline"], config["ranges"], config["diffusion"],
                        config["target"], config["N"], config["seed"])


def render_sensitivity(simulation_scenario):
    with st.container(horizontal=True, wrap=False, vertical_alignment="center",
                      gap="small", key="sensitivity_analysis_header"):
        st.markdown(
            '<div class="exploration-icon" aria-hidden="true">'
            '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
            '<path d="M4 3v17h17M9 16V9M14 16V5M19 16v-4" />'
            '</svg></div>',
            unsafe_allow_html=True, width="content",
        )
        st.title("Sensitivity Analysis")
    st.caption("Explore how material properties and exposure conditions influence the calculated stress index using Sobol sensitivity analysis.")
    if st.session_state.get("sens_source") == "Notebook reference values":
        st.session_state["sens_source"] = "Default reference scenario"
    with st.container(border=True):
        study = st.radio("Study", STUDIES, horizontal=True, key="sens_study", persist_state="session")
        source, output = st.columns(2)
        with source:
            baseline_source = st.selectbox("Starting scenario",
                                           ["Default reference scenario", "Stress analyser scenario"], key="sens_source", persist_state="session",
                                           help="Use a built-in example scenario or your most recent Stress Analyser inputs. "
                                                "Selected parameters vary over their ranges; the other inputs stay fixed.")
        with output:
            target = st.selectbox("Response to analyse", list(OUTPUTS),
                                  format_func=lambda key: f"{OUTPUTS[key][0]} ({OUTPUTS[key][1]})",
                                  key="sens_output", persist_state="session")
        p = reference_baseline(study) if baseline_source == "Default reference scenario" else dict(simulation_scenario)
        if baseline_source == "Default reference scenario":
            st.caption("The default reference scenario is a built-in example. "
                       "View or edit its unvaried properties under Fixed inputs for this study.")
        else:
            st.caption("Start from your most recent Stress Analyser inputs. "
                       "Selected parameters vary over their ranges; all other inputs remain fixed and editable below.")
        group = STUDIES.index(study)
        allowed = EXPOSURE if group == 0 else MATERIAL if group == 1 else list(PARAMETERS)
        keys = st.multiselect("Parameters to vary", allowed, default=allowed,
                              format_func=lambda key: PARAMETERS[key][0], key=f"sens_parameters_{group}", persist_state="session")
        st.caption("Moisture uptake is represented by saturation concentration Csat and diffusivity D. "
                   "Copper CTE remains fixed. Poisson's ratio can be varied in material and combined studies. "
                   "Input distributions are independent.")

        if group == 0:
            mode = st.radio("Diffusivity model", ["Arrhenius", "Fixed D"], horizontal=True, key="sens_diffusion", persist_state="session")
            st.caption("With Arrhenius diffusion, temperature changes both diffusivity and thermal mismatch. "
                       "Fixed D changes only the thermal branch when ΔT varies.")
        else:
            mode = "Fixed D"
            st.caption("Effective diffusivity is varied directly. "
                       "Temperature affects the thermal branch; D has no additional temperature dependence in this study.")
        diffusion = {"mode": mode}
        if mode == "Arrhenius":
            with st.expander("Temperature-dependent diffusion constants", expanded=True):
                d0, ea, tref = st.columns(3)
                diffusion["D0"] = d0.number_input("Pre-exponential factor D₀ (m²/s)", min_value=1e-30,
                                                 value=6.31e-7, format="%.3e", key="sens_D0", persist_state="session")
                diffusion["Ea"] = ea.number_input("Activation energy (J/mol)", min_value=0.0,
                                                 value=37560.0, step=100.0, key="sens_Ea", persist_state="session")
                diffusion["T_ref_C"] = tref.number_input("Reference temperature (°C)", min_value=-273.14,
                                                        value=25.0, step=1.0, key="sens_T_ref_C", persist_state="session")
                st.caption("Exposure temperature = reference temperature + ΔT. "
                           "The diffusion constants are editable starting values.")

        scope = hashlib.sha256(json.dumps(dict(study=study, source=baseline_source, baseline=p),
                                          sort_keys=True).encode()).hexdigest()[:12]
        fixed = [key for key in FIXED_META if key not in keys and not (key == "D" and mode == "Arrhenius")]
        with st.expander("Fixed inputs for this study"):
            columns = st.columns(3)
            for i, key in enumerate(fixed):
                label, unit = FIXED_META[key]
                options = dict(value=float(p[key]), key=f"sens_fixed_{scope}_{key}", persist_state="session")
                if key in ("D", "beta"):
                    options.update(min_value=0.0, format="%.3e")
                elif key in ("Csat", "t_hours"):
                    options.update(min_value=0.0)
                elif key in ("E", "h_mm"):
                    options.update(min_value=0.001)
                elif key == "nu":
                    options.update(min_value=-0.99, max_value=0.499, format="%.3f", step=0.01)
                p[key] = columns[i % 3].number_input(f"{label} ({unit})", **options)
            st.caption("Inputs excluded from the varied parameter list use these fixed values.")

        st.markdown("#### Parameter ranges")
        st.caption("Ranges use ppm/K for CTE, MPa for modulus, hours for exposure and mm for thickness; "
                   "Poisson's ratio is dimensionless. Edit these exploratory bounds for your intended study.")
        if not keys:
            st.info("Select at least one parameter to set up a sensitivity study.")
            return
        editor_key = f"sens_ranges_{group}_{'_'.join(keys)}"
        saved_ranges = st.session_state.setdefault("sensitivity_range_values", {})
        with st.form("sens_run_form"):
            ranges = st.data_editor(
                saved_ranges.get(editor_key, default_ranges(keys)), hide_index=True, width="stretch",
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
            with size:
                N = st.select_slider("Base sample size N", options=[512, 1024, 2048, 4096, 8192],
                                     value=4096, key="sens_N", persist_state="session")
            with scramble:
                seed = st.number_input("Sampling seed", min_value=0, max_value=1000000,
                                       value=12345, step=1, key="sens_seed", persist_state="session")
            st.caption(f"{N * (len(keys) + 2):,} model evaluations · 500 bootstrap resamples · 95% intervals")
            submitted = st.form_submit_button("Run sensitivity analysis", type="primary")

    config = dict(study=study, baseline_source=baseline_source, baseline=p, ranges=ranges.to_dict("records"),
                  diffusion=diffusion, target=target, N=int(N), seed=int(seed), bootstrap_resamples=500)
    signature = json.dumps(config, sort_keys=True)
    if submitted:
        saved_ranges[editor_key] = ranges.copy()
        try:
            with st.spinner("Calculating Sobol indices and confidence intervals…"):
                result = calculate(config)
            st.session_state["sensitivity_result"] = dict(signature=signature, config=config, result=result)
            st.session_state.pop("sensitivity_excel", None)
        except (ValueError, TypeError, OverflowError) as exc:
            st.error(str(exc))
    saved = st.session_state.get("sensitivity_result")
    if not saved:
        st.info("Set your ranges and run the analysis to compare parameter influence.")
        return
    if saved["signature"] != signature:
        st.info("Your settings have changed. Run the analysis to update the results.")
        return
    render_results(saved["result"], config, signature)


def render_results(result, config, signature):
    table = result["indices"].sort_values("ST", ascending=False)
    unit = OUTPUTS[config["target"]][1]
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
        st.latex(r"\sigma_{\mathrm{index}}=\frac{E}{1-\nu}\left[\beta W(t)+(\alpha_{\mathrm{lam}}-\alpha_{\mathrm{Cu}})\Delta T\right]")
        st.write("The default response is the magnitude of this signed index. Uptake starts dry and uses "
                 "one-dimensional plane-sheet diffusion with both faces exposed. Moisture uptake is calculated "
                 "from diffusivity, time, and thickness; Csat sets equilibrium concentration.")
        st.write("A 60-term Fickian series is used with a short-time approximation near zero uptake. "
                 "Saturation concentration and expansion coefficients remain constant with temperature. "
                 "The model calculates an indicator, not interface traction or delamination safety.")
        st.markdown("Scrambled Sobol sequence, Saltelli cross-sampling, centered first-order and Jansen "
                    "total-order estimators. [Method reference](https://salib.readthedocs.io/en/latest/_modules/SALib/analyze/sobol.html).")

    with st.expander("Download the study", expanded=False):
        indices, samples, settings = st.columns(3)
        indices.download_button("Sobol indices (CSV)", table.to_csv(index=False).encode(), "sobol_indices.csv", "text/csv", key="sens_indices_download")
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
