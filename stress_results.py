"""Shared summary cards and contribution explanation for calculated stress results."""
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st


def interpret_stress_contributions(hygroscopic, thermal):
    """Describe signed contributions using their magnitudes to determine dominance."""
    if hygroscopic == 0 and thermal == 0:
        return "Neither contribution is active", "Both contributions are zero, so the net stress index is zero."
    if hygroscopic == 0:
        return "Thermal contribution dominates", "There is no hygroscopic contribution at these inputs. The net stress index equals the thermal contribution."
    if thermal == 0:
        return "Hygroscopic contribution dominates", "There is no thermal mismatch contribution at these inputs. The net stress index equals the hygroscopic contribution."

    balanced = np.isclose(abs(hygroscopic), abs(thermal), rtol=1e-6, atol=1e-12)
    opposing = np.sign(hygroscopic) != np.sign(thermal)
    if balanced:
        headline = "Contributions have equal magnitude"
        effect = ("The contributions have opposite signs and nearly cancel, giving a near-zero net stress index."
                  if opposing else
                  "The contributions have the same sign and reinforce each other, increasing the magnitude of the net stress index.")
    else:
        dominant = "Hygroscopic" if abs(hygroscopic) > abs(thermal) else "Thermal"
        smaller = "Thermal mismatch" if dominant == "Hygroscopic" else "Moisture swelling"
        headline = f"{dominant} contribution dominates"
        effect = (f"The contributions have opposite signs. {smaller} offsets part of the larger contribution, "
                  "reducing the magnitude of the net stress index. The net index keeps the sign of the larger contribution."
                  if opposing else
                  "The contributions have the same sign and reinforce each other, increasing the magnitude of the net stress index.")
    return headline, effect


def render_summary(results, key, compact=False):
    """Show the same four result cards for a calculator scenario or a shortlisted material."""
    st.html(Path(__file__).parent / "assets" / "stress_results.css")
    with st.container(key=f"stress_results_{key}"):
        columns = st.columns(4, border=True)
        suffixes = ("saturation", "hygroscopic", "thermal", "net")
        uptake, swelling, thermal, net = [column.container(border=False, key=f"result_{suffix}_{key}")
                                          for column, suffix in zip(columns, suffixes)]
        cards = [
            (uptake, "Moisture Saturation", f"{results['Mt_Minf'] * 100:.2f} %",
             "Fractional equilibrium uptake (Mt/M∞) achieved under 1D Fickian diffusion at the current exposure duration."),
            (swelling, "Hygroscopic Strain", f"{results['eps_h']:.2e}",
             "Moisture-driven in-plane swelling strain: positive indicates matrix dilatation from absorbed water."),
            (thermal, "Thermal Strain", f"{results['eps_t']:.2e}",
             "Unconstrained differential thermal strain: negative indicates copper expands more during heating."),
            (net, "Net Stress Index", f"{results['sigma_index']:.2f} MPa",
             "Constrained equibiaxial stress: positive denotes in-plane tension; negative denotes compression."),
        ]
        for column, label, value, explanation in cards:
            column.metric(label, value, help=explanation if compact else None)
            if not compact:
                column.caption(explanation)


def render_contributions(results, key, height=350):
    """Place the signed contribution chart beside its explanation."""
    chart, commentary = st.columns([1.1, .9], gap="small")
    with chart:
        st.subheader("Stress Contributions and Net Index")
        values = [results["sigma_hygro"], results["sigma_thermal"], results["sigma_index"]]
        figure = go.Figure(go.Bar(
            x=["Hygroscopic", "Thermal", "Net Stress Index"], y=values,
            marker_color=["#368575", "#c87d55", "#7f8165"],
            text=[f"{value:+.2f} MPa" for value in values], textposition="auto",
            hovertemplate="%{x}<br>%{y:+.4f} MPa<extra></extra>",
        ))
        figure.update_layout(
            height=height, margin=dict(l=10, r=10, t=15, b=10), bargap=.5, showlegend=False,
            yaxis=dict(title="Stress-index value (MPa)", zeroline=True,
                       zerolinecolor="#78988e", zerolinewidth=1.5),
        )
        st.plotly_chart(figure, width="stretch", key=f"{key}_stress_contributions_chart")
    with commentary:
        st.subheader("Results")
        with st.container(border=True, key=f"stress_interpretation_{key}"):
            headline, effect = interpret_stress_contributions(results["sigma_hygro"], results["sigma_thermal"])
            st.markdown(f"**{headline}**")
            st.write(f"Hygroscopic contribution: **{results['sigma_hygro']:+.2f} MPa**  \n"
                     f"Thermal contribution: **{results['sigma_thermal']:+.2f} MPa**")
            st.write(effect)
            st.markdown(f"The resulting net stress index is **{results['sigma_index']:+.2f} MPa**.")
