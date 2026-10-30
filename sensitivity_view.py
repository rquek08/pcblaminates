"""Sobol sensitivity controls and plots for the sensitivity section."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from sensitivity_analysis import (
    EXPOSURE, FIXED_META, MATERIAL, MODEL_VERSION, OUTPUTS as RESPONSE_OPTIONS, PARAMETERS, STUDIES,
    default_ranges, excel_report, material_ranges, run_analysis,
)
from typography import document_fonts


SAMPLE_SIZE = 4096
SAMPLING_SEED = 12345


@st.cache_data(show_spinner=False, max_entries=6)
def calculate(config):
    return run_analysis(config["baseline"], config["ranges"], config["diffusion"],
                        target=config["target"], N=config["N"], seed=config["seed"])


def render_parameter_overview():
    """Show the three animated pathways with parameter names in a compact frame."""
    overview = (Path(__file__).parent / "assets" / "sensitivity_overview.html").read_text(encoding="utf-8")
    st.iframe(document_fonts(overview), height=560)


def parameter_input_table(baseline, keys, range_mode, percent, factor, overrides, diffusion):
    """Combine all nominal inputs with bounds for the selected parameters."""
    rows = []
    for key, (label, unit) in FIXED_META.items():
        varied = key in keys
        row = dict(Parameter=key, Label=label, Unit=unit, Nominal=baseline[key],
                   Lower=None, Upper=None, Distribution=None,
                   Treatment="Varied" if varied else "Fixed")
        if key == "D" and diffusion["mode"] == "Arrhenius":
            row["Treatment"] = "From temperature"
        if varied:
            if range_mode == "Around each material":
                try:
                    bounds = material_ranges([key], baseline, percent, factor)
                except (ValueError, TypeError):
                    # Zero or temporarily invalid nominals must remain editable.
                    bounds = default_ranges([key])
            else:
                bounds = default_ranges([key])
            row.update(bounds.iloc[0][["Lower", "Upper", "Distribution"]].to_dict())
            row.update(overrides.get(key, {}))
        rows.append(row)
    return pd.DataFrame(rows)


def save_parameter_edits(editor_key, name, range_scope, table):
    """Commit editor changes before rerendering bounds and result signatures."""
    state = st.session_state["sensitivity_input_values"][name]
    overrides = state["ranges"].setdefault(range_scope, {})
    for index, changes in st.session_state[editor_key].get("edited_rows", {}).items():
        row = table.iloc[int(index)]
        key = row["Parameter"]
        if "Nominal" in changes:
            state["baseline"][key] = changes["Nominal"]
        if row["Treatment"] == "Varied":
            for column in ("Lower", "Upper"):
                if column in changes:
                    overrides.setdefault(key, {})[column] = changes[column]


def render_sensitivity(records):
    st.session_state["sensitivity_current_signature"] = None
    with st.container(key="sensitivity_workspace"):
        overview, controls = st.columns([1.05, 1], gap="medium")
        with overview.container(key="sensitivity_guide"):
            render_parameter_overview()
        with controls.container(border=True, key="sensitivity_controls"):
            setup = render_sensitivity_controls(records)
    if setup is None:
        return
    configs, submitted = setup
    if submitted:
        calculate_sensitivity_runs(configs)
        show_sensitivity_results(configs)


@st.dialog("Sensitivity results", width="large")
def show_sensitivity_results(configs):
    with st.container(key="sensitivity_results_modal"):
        render_sensitivity_runs(configs, submitted=False)


def render_sensitivity_controls(records):
    presets = {r["name"]: r for r in records}
    custom = "Custom material / manual entry"
    custom_values = dict(E=25000., nu=.2, alpha_lam=12., alpha_cu=16.5,
                         D=1e-12, Csat=150., beta=1e-4, h_mm=1.6, t_hours=24., delta_T=60.)
    with st.container(horizontal=True, horizontal_alignment="center", key="sens_scope_row"):
        scope = st.pills("Material selection mode", ["One material", "Compare materials"],
                         default="One material", required=True, label_visibility="collapsed",
                         key="sens_scope", persist_state="session")
    if scope == "One material":
        names = [st.selectbox("Laminate Preset", [*presets, custom], key="sens_preset", persist_state="session",
                              help="The selected material and conditioning environment supply the nominal inputs.")]
    else:
        names = st.multiselect("Laminate Presets to compare (up to four)", list(presets),
                               default=list(presets)[:2], max_selections=4,
                               key="sens_compare_presets", persist_state="session")
    if not names:
        st.info("Choose at least one material.")
        return None
    study_input, response_input = st.columns(2, gap="small")
    study = study_input.selectbox("Study", STUDIES, index=1,
                                   format_func=lambda value: ["Exposure & geometry", "Material properties", "All parameters"][STUDIES.index(value)],
                                   key="sens_study", persist_state="session")
    group = STUDIES.index(study)
    allowed = EXPOSURE if group == 0 else MATERIAL if group == 1 else list(PARAMETERS)
    if st.session_state.get("sens_output") not in RESPONSE_OPTIONS:
        st.session_state["sens_output"] = "sigma_index"
    target = response_input.selectbox("Response to analyse", list(RESPONSE_OPTIONS),
                                      format_func=lambda key: f"{RESPONSE_OPTIONS[key][0]} ({RESPONSE_OPTIONS[key][1]})",
                                      key="sens_output", persist_state="session")
    keys = st.multiselect("Parameters to vary", allowed, default=[key for key in allowed if key != "nu"],
                          format_func=lambda key: PARAMETERS[key][0],
                          key=f"sens_parameters_{group}", persist_state="session",
                          help="Selected parameters vary independently. Other inputs stay at their nominal values; copper CTE stays fixed.")
    mode = "Fixed D"
    diffusion = {"mode": mode}
    if group == 0:
        model_input, constants = st.columns([1.3, 1], gap="small", vertical_alignment="bottom")
        mode = model_input.radio("Diffusivity model", ["Fixed D", "Arrhenius"], horizontal=True,
                                  key="sens_diffusion_model", persist_state="session")
        diffusion["mode"] = mode
        if mode == "Arrhenius":
            with constants.popover("Diffusion constants", width="stretch"):
                diffusion["D0"] = st.number_input("Pre-exponential factor D₀ (m²/s)", min_value=1e-30,
                                                 value=6.31e-7, format="%.3e", key="sens_D0", persist_state="session")
                diffusion["Ea"] = st.number_input("Activation energy (J/mol)", min_value=0.0,
                                                 value=37560.0, step=100.0, key="sens_Ea", persist_state="session")
                diffusion["T_ref_C"] = st.number_input("Reference temperature (°C)", min_value=-273.14,
                                                      value=25.0, key="sens_T_ref_C", persist_state="session")
                st.caption("D is calculated from reference temperature + ΔT, replacing nominal D. "
                           "These illustrative diffusion constants apply to every material in this run.")

    with st.container(key="sensitivity_footer"):
        st.html('<hr class="compact-section-divider">')
        ranges_input, variation_input = st.columns([1.6, 1], gap="small", vertical_alignment="top")
        with ranges_input:
            st.html('<p class="sensitivity-field-label">Parameter ranges</p>')
            with st.container(horizontal=True, gap="small", vertical_alignment="top", key="sens_range_options"):
                range_mode = st.radio("Parameter ranges", ["Around each material", "Broad exploratory ranges"],
                                      key="sens_range_basis", persist_state="session", label_visibility="collapsed")
                with st.container(width="content", key="sens_range_option_help"):
                    st.markdown("", help="Compare sensitivity near the actual material presets by varying inputs "
                                "around each material's nominal values.")
                    st.markdown("", help="Explore the influence of properties across a wider set of possible values, "
                                "using broad bounds rather than ranges around each material preset.")
        percent, factor = 10.0, 2.0
        if range_mode == "Around each material":
            percent = variation_input.number_input("Variation (±%)", min_value=0.1,
                                                   max_value=99.0, value=10.0, key="sens_percent", persist_state="session",
                                                   help="Illustrative variation around nominal inputs. Diffusivity uses D/2 to 2D instead.")
        input_values = st.session_state.setdefault("sensitivity_input_values", {})
        range_scope = json.dumps(dict(study=study, basis=range_mode, percent=percent, factor=factor), sort_keys=True)
        baselines, ranges = {}, {}
        with st.popover("Parameter ranges table", width="stretch"):
            st.caption("Edit nominal values and bounds. Bounds apply only to rows marked Varied. "
                       "Each comparison material has its own table; scroll horizontally to see all columns.")
            with st.container(key="sens_range_tables"):
                material_tabs = st.tabs(names) if len(names) > 1 else [st.container()]
                for name, tab in zip(names, material_tabs):
                    with tab:
                        values = custom_values if name == custom else presets[name]["properties"]
                        state = input_values.setdefault(name, dict(baseline=dict(values), ranges={}))
                        table = parameter_input_table(state["baseline"], keys, range_mode, percent, factor,
                                                      state["ranges"].get(range_scope, {}), diffusion)
                        fingerprint = json.dumps(dict(name=name, scope=range_scope, table=table.to_dict("records")), sort_keys=True)
                        editor_key = "sens_inputs_" + hashlib.sha256(fingerprint.encode()).hexdigest()[:16]
                        st.data_editor(
                            table.drop(columns="Distribution"), hide_index=True, width="stretch", height=340,
                            disabled=["Parameter", "Label", "Unit", "Treatment"],
                            column_order=["Label", "Unit", "Nominal", "Lower", "Upper", "Treatment"],
                            column_config={
                                "Parameter": None,
                                "Label": st.column_config.TextColumn("Parameter", width="medium"),
                                "Nominal": st.column_config.NumberColumn("Nominal", format="%.6g", required=True),
                                "Lower": st.column_config.NumberColumn("Lower bound", format="%.6g"),
                                "Upper": st.column_config.NumberColumn("Upper bound", format="%.6g"),
                            }, key=editor_key, on_change=save_parameter_edits,
                            args=(editor_key, name, range_scope, table),
                        )
                        varied = table.loc[table["Treatment"].eq("Varied")]
                        log_uniform = varied.loc[varied["Distribution"].eq("log-uniform"), "Label"].tolist()
                        if log_uniform:
                            st.caption(f"Log-uniform sampling: {', '.join(log_uniform)}. All other varied parameters use uniform sampling.")
                        elif not varied.empty:
                            st.caption("All varied parameters use uniform sampling.")
                        baselines[name] = dict(state["baseline"])
                        ranges[name] = table.loc[table["Parameter"].isin(keys),
                                                 ["Parameter", "Label", "Unit", "Lower", "Upper", "Distribution"]]
        with st.container(key="sensitivity_actions"):
            submitted = st.button("Run sensitivity analysis", type="primary", width="stretch", key="sens_run", disabled=not keys)
    if not keys:
        return None
    configs = [dict(study=study, material=name, condition=presets[name]["condition"] if name != custom else "Manual inputs",
                    baseline_source="Material preset" if name != custom else "Manual inputs",
                    baseline=baselines[name], range_basis=range_mode, ranges=ranges[name].to_dict("records"),
                    diffusion=diffusion, target=target, method="Sobol", N=SAMPLE_SIZE,
                    seed=SAMPLING_SEED, bootstrap_resamples=500, model_version=MODEL_VERSION) for name in names]
    return configs, submitted


def calculate_sensitivity_runs(configs):
    runs = []
    with st.spinner(f"Calculating Sobol sensitivity for {len(configs)} material(s)…"):
        for config in configs:
            try:
                runs.append(dict(config=config, result=calculate(config)))
            except (ValueError, TypeError, OverflowError) as exc:
                runs.append(dict(config=config, error=str(exc)))
    st.session_state["sensitivity_batch"] = dict(signature=json.dumps(configs, sort_keys=True), runs=runs)
    st.session_state.pop("sensitivity_excel", None)


def render_sensitivity_runs(configs, submitted):
    target = configs[0]["target"]
    signature = json.dumps(configs, sort_keys=True)
    if submitted:
        calculate_sensitivity_runs(configs)
    saved = st.session_state.get("sensitivity_batch")
    if not saved:
        st.info("Set your inputs and ranges on the left, then run the analysis to see which parameters matter most.")
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
        comparison, individual = st.tabs(["Material comparison", "Individual results"])
        with comparison:
            render_comparison(successes, configs)
        with individual:
            render_material_result(successes, signature, target)
    else:
        render_material_result(successes, signature, target)


def render_material_result(successes, signature, target):
    if len(successes) > 1:
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
    st.caption(f"{config['material']} · {RESPONSE_OPTIONS[target][0]}")
    render_results(active["result"], config, active_signature)


def render_comparison(runs, configs):
    table = pd.concat([run["result"]["indices"].assign(Material=run["config"]["material"],
                                                     Condition=run["config"]["condition"]) for run in runs], ignore_index=True)
    fig = go.Figure()
    colors = ["#368575", "#c87d55", "#7f8165", "#92b8ad"]
    for run, color in zip(runs, colors):
        data = run["result"]["indices"]
        fig.add_trace(go.Bar(x=data["Label"], y=data["ST"], name=run["config"]["material"], marker_color=color))
    legend_space = 24 * len(runs)
    fig.update_layout(barmode="group", height=280 + legend_space,
                      margin=dict(t=30 + legend_space, b=20),
                      yaxis_title="Total effect · ST",
                      legend=dict(orientation="h", x=0, xanchor="left", y=1.04, yanchor="bottom"))
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
    importance, study_details = st.tabs(["Parameter importance", "Study details"])
    with importance:
        table = result["indices"].sort_values("ST", ascending=False)
        unit = RESPONSE_OPTIONS[config["target"]][1]
        st.caption("Ranked by total effect (ST), including interactions. Main effect (S1) shows each input's individual contribution.")

        fig = go.Figure()
        for index, label, color in [("S1", "Main effect · S1", "#92b8ad"), ("ST", "Total effect · ST", "#368575")]:
            values = table[index].to_numpy()
            fig.add_trace(go.Bar(
                y=table["Label"], x=values, orientation="h", name=label, marker_color=color,
                error_x=dict(type="data", symmetric=False,
                             array=np.maximum(0, table[f"{index}_high"] - values),
                             arrayminus=np.maximum(0, values - table[f"{index}_low"])),
            ))
        fig.update_layout(barmode="group", height=max(295, min(355, 38 * len(table) + 35)),
                          xaxis_title="Fraction of selected response variance", yaxis=dict(autorange="reversed"),
                          margin=dict(t=60, b=30, l=15, r=25),
                          legend=dict(orientation="h", x=0, xanchor="left", y=1.04, yanchor="bottom",
                                      entrywidth=180, entrywidthmode="pixels"))
        st.plotly_chart(fig, width="stretch")
        top = table.iloc[0]
        st.info(f"Largest estimated total effect: **{top['Label']}** (ST ≈ {top['ST']:.3f}) for these ranges and fixed inputs.")
        st.caption("Error bars show paired bootstrap 95% intervals. Sampling noise can put estimates outside 0–1. "
                   "Total effects overlap and should not be added as independent percentages.")
    with study_details:
        with st.expander("Inputs and ranges used"):
            st.dataframe(pd.DataFrame([dict(Parameter=label, Nominal=config["baseline"][key], Unit=unit,
                                           Treatment="Varied" if key in {r["Parameter"] for r in config["ranges"]} else
                                           "Calculated from temperature" if key == "D" and config["diffusion"]["mode"] == "Arrhenius" else "Fixed")
                                       for key, (label, unit) in FIXED_META.items()]), hide_index=True, width="stretch")
            st.dataframe(pd.DataFrame(config["ranges"]).drop(columns="Parameter"), hide_index=True, width="stretch")
        with st.expander("Further information", expanded=False):
            st.markdown("**S1 · Main effect:** the share of response variance associated with one input's main effect.\n\n"
                        "**ST · Total effect:** the main effect plus every interaction involving that input.\n\n"
                        "**ST − S1 · Interactions:** a gap suggests the input acts in combination with other inputs.")
            st.markdown("#### Detailed parameter indices")
            st.dataframe(table.drop(columns="Parameter"), hide_index=True, width="stretch")
            st.caption(f"Largest change in ST between N/2 and N: {table['ST change'].max():.3f}. "
                       "Large changes suggest uncertain rankings. This comparison does not guarantee convergence.")

            st.markdown("#### Response and intermediate output summary")
            st.dataframe(result["summary"], hide_index=True, width="stretch")
            st.caption(f"Statistics use the independent A and B base samples. Selected response units: {unit}.")
            st.markdown("#### Sampled inputs and calculated outputs")
            st.dataframe(result["samples"].head(500), hide_index=True, width="stretch")
            st.caption(f"Preview of the first 500 of {len(result['samples']):,} model evaluations. Downloads include all rows.")
            st.markdown("#### Model and method")
            st.latex(r"\sigma_{\mathrm{index}}=\frac{E}{1-\nu}\left[\beta W_t+(\alpha_{\mathrm{lam}}-\alpha_{\mathrm{Cu}})\Delta T\right]")
            st.write("The stress index retains the sign of the combined moisture and thermal contributions. Uptake starts dry and uses "
                     "one-dimensional plane-sheet diffusion through one exposed face, with zero flux at the copper interface. "
                     "The laminate thickness is h and the diffusion term uses (2h)². Moisture uptake is calculated "
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
