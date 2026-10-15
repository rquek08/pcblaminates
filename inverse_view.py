"""Sensitivity-driven recommendations and a conceptual laminate gallery."""
import hashlib
import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from inverse_analysis import recommendation_parameters, recommend_materials, records_from_csv
from laminate_showcase import render_showcase
from sensitivity_analysis import MATERIAL, PARAMETERS


def open_sensitivity():
    st.session_state["main_navigation"] = "Sensitivity Analysis"
    st.session_state["scroll_to_top"] = True


def render_inverse(records):
    st.caption("Use sensitivity results to shortlist materials, then explore the structures behind their properties.")
    render_recommender(records)
    st.divider()
    render_showcase()


def render_recommender(records):
    st.subheader("Material recommender")
    #st.caption("Sensitivity analysis → three important material parameters → shortlist up to 10 → verify the shortlist")
    saved = st.session_state.get("sensitivity_result")
    current = st.session_state.get("sensitivity_current_signature")
    if not saved or current != saved["signature"]:
        st.info("Sensitivity settings have changed. Run the study again to update recommendations." if saved else
                "Run Sensitivity Analysis with at least three material parameters and a stress-index response to get started.")
        st.button("Open Sensitivity Analysis →", on_click=open_sensitivity, key="inverse_open_sensitivity")
        return
    try:
        top = recommendation_parameters(saved)
    except ValueError as exc:
        st.info(str(exc))
        st.button("Open Sensitivity Analysis →", on_click=open_sensitivity, key="inverse_fix_sensitivity")
        return
    for column, (_, row) in zip(st.columns(3, gap="medium", border=True), top.iterrows()):
        with column:
            st.markdown(f"**{row['Label']}**")
            st.metric("Total effect · ST", f"{row['ST']:.3f}")
            st.caption(f"Screening weight: {row['Screening weight']:.1%}")
            bounds = next(r for r in saved["config"]["ranges"] if r["Parameter"] == row["Parameter"])
            st.caption(f"Study bounds: {bounds['Lower']:.4g}–{bounds['Upper']:.4g} {bounds['Unit']}")
    baseline = saved["config"]["baseline"]
    st.caption(f"Saved study: ΔT {baseline['delta_T']:+.1f} K · exposure {baseline['t_hours']:g} h · "
               f"thickness {baseline['h_mm']:g} mm · copper CTE {baseline['alpha_cu']:g} ppm/K. "
               "Normalised screening weights are not variance shares.")
    with st.expander("Use a larger material dataset"):
        upload = st.file_uploader("Material library (CSV)", type=["csv"], key="inverse_library_upload")
        st.caption("Required columns: name, condition, alpha_lam, beta, E, nu, Csat, D. "
                   "Units: CTE ppm/K; CHE m³/kg; modulus MPa; Poisson's ratio dimensionless; Csat mol/m³; D m²/s. "
                   "Exposure and thickness come from the saved study.")
        example = pd.DataFrame([dict(name=r["name"], condition=r["condition"],
                                     **{key: r["properties"][key] for key in MATERIAL}) for r in records])
        st.download_button("Download library CSV template", example.to_csv(index=False).encode(),
                           "material_library.csv", "text/csv", key="inverse_library_template")
    if upload is not None:
        try:
            records = records_from_csv(pd.read_csv(upload))
        except (ValueError, UnicodeError, pd.errors.ParserError) as exc:
            st.error(f"Could not read the material library: {exc}")
            return
    environment, constraint = st.columns(2)
    condition = environment.selectbox("Library conditioning environment", list(dict.fromkeys(r["condition"] for r in records)),
                                      key="inverse_recommend_condition", persist_state="session")
    limit = constraint.number_input("Stress-index magnitude limit (MPa)", min_value=0., value=10.,
                                    key="inverse_recommend_limit", persist_state="session")
    st.caption("Conditioning selects the library's measured D and Csat. ΔT stays at the saved study value; "
               "set the intended thermal exposure in Sensitivity Analysis.")
    with st.expander("How screening works"):
        st.write("The three largest material total effects (ST) set the screening weights. Materials outside those "
                 "three study bounds are excluded. The rest are ranked by weighted distance to low-stress samples "
                 "already calculated in the study. Log-uniform properties use logarithmic distance.")
        st.write("Only the closest 10 undergo the full stress calculation using all six material properties, then "
                 "are ordered by stress-index magnitude. The limit flags shortlisted materials; it does not trigger "
                 "calculations for the remaining dataset.")
        st.caption("Screening can miss better materials because three parameters do not capture every effect or "
                   "interaction. A small net index may reflect cancellation between moisture and thermal contributions.")
    signature = hashlib.sha256(json.dumps(dict(study=saved["signature"], records=records,
                                              condition=condition, limit=limit), sort_keys=True).encode()).hexdigest()
    if st.button("Recommend up to 10 materials", type="primary", key="inverse_recommend"):
        try:
            with st.spinner("Screening properties and verifying the shortlist…"):
                result = recommend_materials(records, condition, saved, limit)
            st.session_state["inverse_recommendation"] = dict(signature=signature, result=result)
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            st.session_state.pop("inverse_recommendation", None)
            st.error(str(exc))
    recommendation = st.session_state.get("inverse_recommendation")
    if not recommendation or recommendation["signature"] != signature:
        st.caption("Run the recommender to build a shortlist for these settings.")
        return
    result = recommendation["result"]
    library, shortlist, savings = st.columns(3)
    library.metric("Materials in environment", f"{result['total']:,}")
    shortlist.metric("Full model evaluations", result["evaluated"])
    savings.metric("Materials without full evaluation", f"{result['total'] - result['evaluated']:,}")
    st.caption(f"{result['eligible']:,} eligible · {result['outside']:,} outside the three screening bounds · "
               f"{result['invalid']:,} invalid property records excluded.")
    candidates = result["candidates"]
    if candidates.empty:
        st.info("No materials have valid properties within the three screening bounds. "
                "Check the dataset or broaden the sensitivity ranges and rerun that study.")
        return
    if result["total"] <= 10:
        st.caption("This environment has at most 10 entries. A larger dataset will demonstrate savings "
                   "beyond the property-bound exclusions.")
    labels = {"rank": "Rank", "name": "Material", "sigma_abs": "Magnitude (MPa)",
              "sigma_index": "Signed index (MPa)", "within_limit": "Within limit", "screening_distance": "Screening distance"}
    st.dataframe(candidates[list(labels)].rename(columns=labels), hide_index=True, width="stretch", column_config={
        "Magnitude (MPa)": st.column_config.NumberColumn(format="%.3f"),
        "Signed index (MPa)": st.column_config.NumberColumn(format="%.3f"),
        "Screening distance": st.column_config.NumberColumn(format="%.4f"),
    })
    if not candidates["within_limit"].any():
        st.info("None of the shortlisted materials meets the selected index limit.")
    st.caption("Rank uses full-model magnitude within the shortlist. Smaller screening distances mean closer "
               "to the saved low-stress sample regions.")
    with st.expander("Inspect a recommended material"):
        selected = st.selectbox("Shortlisted material", candidates.index.tolist(),
                                format_func=lambda index: candidates.loc[index, "name"], key="inverse_selected_candidate")
        row = candidates.loc[selected]
        figure = go.Figure(go.Bar(x=["Moisture", "Thermal", "Net index"],
                                  y=[row["sigma_hygro"], row["sigma_thermal"], row["sigma_index"]],
                                  marker_color=["#368575", "#c87d55", "#7f8165"]))
        figure.update_layout(height=260, margin=dict(l=10, r=10, t=10, b=10), yaxis_title="MPa")
        st.plotly_chart(figure, width="stretch", key="inverse_shortlist_contributions")
        st.dataframe(pd.DataFrame([dict(Property=PARAMETERS[key][0], Value=row[key], Unit=PARAMETERS[key][1],
                                       Screening="Top three" if key in top["Parameter"].tolist() else "Verification only")
                                  for key in MATERIAL]), hide_index=True, width="stretch")
    st.download_button("Download verified shortlist (CSV)", candidates.to_csv(index=False).encode(),
                       "recommended_materials.csv", "text/csv", key="inverse_shortlist_download")
