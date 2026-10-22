"""Educational 2D structure concepts; no geometry-to-property simulation."""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from sensitivity_analysis import MATERIAL, PARAMETERS


LAYOUTS = {
    "Homogeneous baseline": "A uniform laminate core without glass reinforcement, weave, fillers or additives. "
                            "Copper foil is shown separately on the outer faces.",
    "Balanced": "Two woven-glass plies between similar copper skins: a reinforced structure for comparison.",
    "Resin-rich": "An open weave leaves several resin-filled pockets between yarns, where reinforcement is absent.",
    "Glass-dense": "More glass bundles and closer weave spacing illustrate increased reinforcement.",
    "Copper interfaces": "A flat laminate cross-section shows small copper tabs on the top and bottom surfaces, "
                         "with narrow vertical copper features extending into the laminate.",
    "Filled matrix": "Dispersed particles illustrate a resin formulation containing fillers or additives.",
}
COLORS = {"Glass fiber reinforcement": "#368575", "Resin matrix": "#e7c887",
          "Glass weave architecture": "#648d76", "Copper foil interfaces": "#c87d55",
          "Fillers and additives": "#7384b0"}
FACTORS = {
    "Glass fiber reinforcement": dict(
        properties=["E", "alpha_lam"],
        effect="Glass volume fraction and fibre orientation change how strongly the reinforcement constrains the resin. "
               "This changes the laminate's effective directional stiffness and thermal expansion. In the model, with "
               "Poisson's ratio held fixed, greater modulus increases the stress-index magnitude for the same net mismatch "
               "strain. Changing laminate CTE changes the thermal mismatch with copper.\n\n"
               "Lower laminate CTE does not automatically mean lower mismatch: the relevant quantity is its difference "
               "from copper's CTE.",
        parameter_note="Related model parameters: E and αlam. Use values for the relevant material direction.",
        evidence="Veldhuijzen van Zanten et al. (2018) combined PCB experiments with micromechanical modelling. "
                 "Across the constructions studied, effective Young's modulus ranged from 11–31 GPa and CTE from "
                 "10–28 ppm/K, depending on reinforcement and fibre volume fraction.",
        scope="These are study-specific ranges, not universal material bounds.",
        title="Veldhuijzen van Zanten et al. (2018) · Method to determine thermoelastic material properties of constituent "
              "and copper-patterned layers of multilayer printed circuit boards",
        source="https://doi.org/10.1007/s10854-017-8449-2",
        limit="The model uses effective material properties; it does not calculate them from glass volume fraction or resolve local fibre restraint."),
    "Resin matrix": dict(
        properties=["D", "Csat", "beta", "E", "alpha_lam"],
        effect="Resin chemistry and the cured network influence water mobility, equilibrium absorption and swelling. "
               "These affect D, Csat and β, changing the time-dependent hygroscopic contribution. Resin chemistry "
               "and conditioning can also change stiffness and thermal behaviour.\n\n"
               "The direction of these changes must be established experimentally. Crosslink density alone does not "
               "reliably predict diffusivity.",
        parameter_note="Related model parameters: D, Csat, β, E and αlam.",
        evidence="Toscano et al. (2016) studied DGEBA/DDS and DGEBF/DDS epoxies in water at 50°C and 80°C, "
                 "using gravimetry, dynamic mechanical analysis and photoelastic stress measurements. The more highly "
                 "crosslinked DGEBA system had higher diffusivity and equilibrium uptake. The two systems developed "
                 "comparable peak swelling stresses but different stress-development kinetics.",
        scope="These were neat-resin specimens. The findings support mechanisms rather than directly supplying "
              "properties for a glass-reinforced laminate or a resin-rich pocket.",
        title="Toscano et al. (2016) · Water diffusion and swelling stresses in highly crosslinked epoxy matrices",
        source="https://doi.org/10.1016/j.polymdegradstab.2016.09.004",
        limit="The constant-property model does not capture moisture-induced stiffness changes or local stress gradients around resin-rich regions."),
    "Glass weave architecture": dict(
        properties=["E", "nu", "alpha_lam"],
        effect="Warp/fill yarn arrangement and fibre crimp change the reinforcement in each direction. This produces "
               "directional differences in effective modulus, Poisson's ratio and CTE. Consequently, the calculated "
               "stress index can differ between the laminate's x and y directions when direction-specific inputs are used.\n\n"
               "The evidence supports directional thermoelastic properties. A specific weave should not be claimed "
               "to reduce diffusivity without supporting transport measurements.",
        parameter_note="Related model parameters: direction-specific E, ν and αlam.",
        evidence="Sottos, Ockers and Swindeman (1999) characterised woven glass/epoxy boards using optical microscopy, "
                 "tensile testing and thermomechanical analysis above and below the glass-transition temperature Tg. "
                 "They measured elastic moduli, Poisson's ratios and CTEs, compared them with analytical models, "
                 "and investigated the effect of fibre crimp.",
        scope="The study concerns woven glass/epoxy circuit-board substrates; its thermoelastic results do not establish a weave-dependent diffusion ranking.",
        title="Sottos, Ockers & Swindeman (1999) · Thermoelastic properties of plain weave composites for multilayer circuit board applications",
        source="https://doi.org/10.1115/1.2792659",
        limit="The effective-property model does not resolve local stresses around individual yarns or resin pockets."),
    "Copper foil interfaces": dict(
        properties=["alpha_cu"],
        effect="Copper coverage changes the surfaces available for moisture entry, affecting moisture distribution "
               "and subsequent swelling. This is primarily a diffusion-boundary-condition effect, rather than a change "
               "in the laminate's intrinsic diffusivity.\n\n"
               "Interface condition also affects adhesion and resistance to separation. This is a separate pathway "
               "to delamination that the current stress index does not calculate.",
        parameter_note="Related model input: copper CTE (αCu). Moisture-exposure geometry is a boundary-condition consideration.",
        evidence="Weide-Zaage, Horaud and Frémont (2005) compared PCB moisture measurements with finite-element "
                 "simulations. Copper layers acted as diffusion barriers, changing moisture distributions and drying behaviour.",
        title="Weide-Zaage, Horaud & Frémont (2005) · Moisture diffusion in printed circuit boards: Measurements and finite-element simulations",
        source="https://doi.org/10.1016/j.microrel.2005.07.077",
        additional_research=[dict(
            evidence="Chan and Yuen (2009) examined epoxy–copper specimens conditioned for up to 1,000 hours at "
                     "85°C/85% RH. Interfacial moisture measurements and fracture-surface analysis linked transport "
                     "along the interface and changes in its cuprous oxide region to adhesion degradation in that system.",
            title="Chan & Yuen (2009) · Study of interfacial moisture diffusion at epoxy/Cu interface",
            source="https://doi.org/10.1163/156856109X433982",
            scope="The interfacial findings apply to the tested epoxy/copper system; they are not a universal adhesion-degradation law.",
        )],
        limit="The uptake model assumes a sheet exposed on both faces. Copper coverage barriers, adhesion degradation, "
              "copper load sharing and local interface stresses are outside the current stress index."),
    "Fillers and additives": dict(
        properties=["E", "alpha_lam", "D", "Csat"],
        effect="Filler chemistry, loading and dispersion can change stiffness, CTE and moisture transport. "
               "In suitable silica-filled epoxies, changes in CTE and water uptake alter the thermal and hygroscopic "
               "contributions, while increased stiffness raises the stress scale for a given net mismatch.\n\n"
               "These effects compete, so more filler does not guarantee a smaller combined stress index. "
               "Lower CTE is beneficial for thermal mismatch only when it brings the laminate CTE closer to copper's CTE.",
        parameter_note="Related model parameters: E, αlam, D and Csat. A change in β requires separate swelling evidence.",
        evidence="Dittanet, Pearson and Kongkachuichay (2017) tested approximately 20 nm silica particles in DGEBF "
                 "and cycloaliphatic epoxies. Increasing filler content reduced CTE, moisture uptake and diffusivity, "
                 "and increased modulus in those formulations. At 85°C/85% RH, DGEBF equilibrium mass uptake "
                 "decreased from approximately 1.25% without filler to 0.84% at 35 wt% silica.",
        scope="The findings concern those silica-filled systems, not every filler or additive. Mass-uptake percentages "
              "require a consistent density and concentration basis before use as Csat.",
        title="Dittanet, Pearson & Kongkachuichay (2017) · Thermo-mechanical behaviors and moisture absorption of silica nanoparticle reinforcement in epoxy resins",
        source="https://doi.org/10.1016/j.ijadhadh.2017.06.006",
        limit="The model requires measured effective properties for a filled formulation; it does not infer them from filler loading."),
}

def structure_figure(layout, focus="All constituents"):
    """Cross-section and plan views with schematic, deliberately unitless geometry."""
    homogeneous = layout == "Homogeneous baseline"
    copper_interface = layout == "Copper interfaces"
    figure = make_subplots(rows=1, cols=2, column_widths=[.66, .34],
                           subplot_titles=["Laminate cross-section", "Laminate · plan view" if homogeneous or copper_interface else "Weave · plan view"],
                           horizontal_spacing=.08)

    def opacity(factor):
        return 1 if focus in ("All constituents", factor) else .3

    def polygon(x, y, factor, label, col=1, color=None, outline=None):
        color = color or COLORS[factor]
        figure.add_trace(go.Scatter(x=x, y=y, mode="lines", fill="toself",
                                    fillcolor=color, opacity=opacity(factor),
                                    line=outline or dict(color=color, width=0), name=label,
                                    hovertemplate=label + "<extra></extra>", showlegend=False), row=1, col=col)

    def rectangle(x0, x1, y0, y1, factor, label, col=1):
        polygon([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], factor, label, col=col)

    def finish(legend_colors):
        for factor, color in legend_colors.items():
            figure.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=factor,
                                        marker=dict(color=color, size=10)))
        figure.update_xaxes(visible=False, range=[-.15, 10.15], row=1, col=1)
        figure.update_yaxes(visible=False, range=[0, 5.65], row=1, col=1)
        figure.update_xaxes(visible=False, range=[-.2, 10.2], row=1, col=2)
        figure.update_yaxes(visible=False, range=[-.2, 6.2], scaleanchor="x2", scaleratio=1, row=1, col=2)
        figure.update_layout(height=390, margin=dict(l=10, r=10, t=45, b=80),
                              legend=dict(orientation="h", y=-.05), dragmode=False,
                              plot_bgcolor="rgba(0,0,0,0)")
        return figure

    rectangle(0, 10, .6, 4.7, "Resin matrix", "Homogeneous laminate" if homogeneous or copper_interface else "Resin matrix")
    if copper_interface:
        top_tabs = ((.8, 2.0), (2.8, 3.9), (4.6, 5.8), (7.0, 8.2), (9.0, 9.7))
        for left, right in top_tabs:
            rectangle(left, right, 4.7, 4.95, "Copper foil interfaces", "Top surface copper tab")
        for left, right in ((.5, 1.55), (2.5, 3.7), (4.6, 5.8), (7.1, 8.3), (9.1, 9.8)):
            rectangle(left, right, .35, .6, "Copper foil interfaces", "Bottom surface copper tab")
        rectangle(1.32, 1.47, 3.2, 4.7, "Copper foil interfaces", "Vertical copper feature from top surface")
        rectangle(7.65, 7.80, .6, 2.0, "Copper foil interfaces", "Vertical copper feature from bottom surface")
        rectangle(5.12, 5.28, .6, 4.7, "Copper foil interfaces", "Through-thickness copper connection")
        rectangle(0, 10, 0, 6, "Resin matrix", "Homogeneous laminate · top view", col=2)
        for left, right in top_tabs:
            rectangle(left, right, 2.25, 3.75, "Copper foil interfaces", "Top surface copper tab · plan view", col=2)
        figure.add_trace(go.Scatter(x=[0, 10], y=[3, 3], mode="lines",
                                    line=dict(color="#78988e", width=1, dash="dash"),
                                    showlegend=False, hovertemplate="Cross-section position<extra></extra>"), row=1, col=2)
        figure.add_trace(go.Scatter(x=[1.395, 5.2], y=[3, 3], mode="markers",
                                    marker=dict(size=7, color="#8d5036", symbol="circle-open", line=dict(width=1.5)),
                                    opacity=opacity("Copper foil interfaces"), showlegend=False,
                                    hovertemplate="Vertical copper feature beneath this tab<extra></extra>"), row=1, col=2)
        return finish({"Homogeneous laminate": COLORS["Resin matrix"],
                       "Copper foil interfaces": COLORS["Copper foil interfaces"]})
    rectangle(0, 10, .35, .6, "Copper foil interfaces", "Bottom copper foil")
    rectangle(0, 10, 4.7, 4.95, "Copper foil interfaces", "Top copper foil")
    if homogeneous:
        polygon([0, 10, 10, 0, 0], [0, 0, 6, 6, 0], "Resin matrix", "Homogeneous laminate", col=2)
        return finish({"Homogeneous laminate": COLORS["Resin matrix"],
                       "Copper foil interfaces": COLORS["Copper foil interfaces"]})
    # Both views share the same yarn pitch and x positions. Use data-space
    # ribbons, so crossing patches stay aligned when the chart is resized.
    pitch = 1.8 if layout == "Resin-rich" else 1.0 if layout == "Glass-dense" else 1.5
    half_width = pitch * (.14 if layout == "Resin-rich" else .22)
    positions = (np.arange(int(10 / pitch)) + .5) * pitch + (10 % pitch) / 2
    row_positions = (np.arange(int(6 / pitch)) + .5) * pitch + (6 % pitch) / 2
    x = np.linspace(0, 10, 500)
    centers = [1.35, 2.65, 3.95] if layout == "Glass-dense" else [1.65, 3.65]
    amplitude = .3

    def yarn_height(xs, ply):
        return centers[ply] - amplitude * np.cos(np.pi * (xs - positions[0]) / pitch + ply * np.pi)

    for i, center in enumerate(centers):
        y = yarn_height(x, i)
        polygon(np.r_[x, x[::-1]], np.r_[y - .11, (y + .11)[::-1]],
                "Glass fiber reinforcement", "Glass yarn running along the section")
        # Each transverse yarn sits opposite the longitudinal yarn at that
        # crossing, alternating above/below rather than floating at one height.
        theta = np.linspace(0, 2 * np.pi, 50)
        for crossing, position in enumerate(positions):
            transverse_height = center + amplitude * (-1) ** (crossing + i)
            polygon(position + half_width * np.cos(theta), transverse_height + .12 * np.sin(theta),
                    "Glass fiber reinforcement", "Cross-section of a transverse glass yarn")
    if layout == "Filled matrix":
        particle_x, particle_y = np.meshgrid(np.arange(.4, 10, .65), [1.05, 2.35, 2.85, 4.3])
        figure.add_trace(go.Scatter(x=particle_x.ravel(), y=particle_y.ravel(), mode="markers",
                                    marker=dict(size=6, color=COLORS["Fillers and additives"]),
                                    opacity=opacity("Fillers and additives"), showlegend=False,
                                    hovertemplate="Dispersed filler particle<extra></extra>"), row=1, col=1)
    for interface in (.6, 4.7):
        figure.add_trace(go.Scatter(x=x, y=interface + .035 * np.sin(20 * x), mode="lines",
                                    line=dict(color="#8d5036", width=2), opacity=opacity("Copper foil interfaces"),
                                    showlegend=False, hovertemplate="Copper/resin interface<extra></extra>"), row=1, col=1)
    # Draw the resin matrix beneath the plan-view weave so its open cells
    # clearly contain resin, rather than looking like empty holes.
    plan_factor = "Glass weave architecture"
    polygon([0, 10, 10, 0, 0], [0, 0, 6, 6, 0], "Resin matrix", "Resin between glass yarns", col=2)
    for position in positions:
        polygon([position - half_width, position + half_width, position + half_width, position - half_width],
                [0, 0, 6, 6], plan_factor, "Warp glass yarn", col=2, color="#368575")
    for position in row_positions:
        polygon([0, 10, 10, 0],
                [position - half_width, position - half_width, position + half_width, position + half_width],
                plan_factor, "Fill glass yarn", col=2, color="#86aa9d")
    for i, px in enumerate(positions):
        for j, py in enumerate(row_positions):
            if (i + j) % 2 == 0:
                # Restore the warp over the fill at alternate intersections.
                # Its edges are exactly the edges of the underlying warp.
                h = half_width + .03 * pitch
                polygon([px - half_width, px + half_width, px + half_width, px - half_width],
                        [py - h, py - h, py + h, py + h], plan_factor,
                        "Warp over fill · alternating plain weave", col=2, color="#368575")
    if layout == "Resin-rich":
        outline = dict(color="#9b712c", width=1.2, dash="dot")
        # These outlines occupy gaps between adjacent yarns. Their curved
        # section boundaries follow the crimp, without covering reinforcement.
        for number, gap in enumerate((0, 1, 3), start=1):
            ply = (number - 1) % 2
            side = 1 if ply == 0 else -1
            midpoint = (positions[gap] + positions[gap + 1]) / 2
            pocket_x = np.linspace(midpoint - .25 * pitch, midpoint + .25 * pitch, 50)
            near = yarn_height(pocket_x, ply) + side * .17
            far = np.full_like(pocket_x, centers[ply] + side * .65)
            label = f"R{number} · Resin-rich pocket; no glass reinforcement inside"
            polygon(np.r_[pocket_x, pocket_x[::-1]], np.r_[near, far[::-1]],
                    "Resin matrix", label, color="#f4dda8", outline=outline)
            figure.add_annotation(x=midpoint, y=centers[ply] + side * .46, text=f"R{number}",
                                  showarrow=False, font=dict(size=11, color="#735121"), row=1, col=1)
            # Mark open plan-view cells at the same x gaps.
            row_gap = ply
            pocket_y = (row_positions[row_gap] + row_positions[row_gap + 1]) / 2
            h = .25 * pitch
            polygon([midpoint - h, midpoint + h, midpoint + h, midpoint - h],
                    [pocket_y - h, pocket_y - h, pocket_y + h, pocket_y + h],
                    "Resin matrix", label, col=2, color="#f4dda8", outline=outline)
            figure.add_annotation(x=midpoint, y=pocket_y, text=f"R{number}", showarrow=False,
                                  font=dict(size=10, color="#735121"), row=1, col=2)
    return finish(COLORS)


def select_structure_context():
    if st.session_state.get("inverse_structure_layout") == "Copper interfaces":
        st.session_state["inverse_structure_factor"] = "Copper foil interfaces"


def render_showcase():
    st.subheader("Inhomogeneity in application")
    st.caption("Explore constituent layouts and the research connecting structure, properties and stress.")
    if st.session_state.get("inverse_structure_layout") == "Copper-asymmetric":
        st.session_state["inverse_structure_layout"] = "Copper interfaces"
        st.session_state["inverse_structure_factor"] = "Copper foil interfaces"
    layout = st.radio("Structure concept", list(LAYOUTS), horizontal=True,
                      key="inverse_structure_layout", persist_state="session", on_change=select_structure_context)
    st.write(LAYOUTS[layout])
    homogeneous = layout == "Homogeneous baseline"
    focus_options = (["All constituents", "Resin matrix", "Copper foil interfaces"] if homogeneous or layout == "Copper interfaces" else
                     ["All constituents", *FACTORS])
    if st.session_state.get("inverse_structure_focus") not in focus_options:
        st.session_state.pop("inverse_structure_focus", None)
    focus = st.selectbox("Highlight a constituent", focus_options, key="inverse_structure_focus")
    st.plotly_chart(structure_figure(layout, focus), width="stretch", key="inverse_structure_figure",
                    config=dict(displayModeBar=False))
    st.caption("Conceptual schematics, not to scale. Yarn shapes, particle size and foil thickness are illustrative. "
               "These layouts do not change the recommender's material properties or predict a stress field.")
    if homogeneous:
        st.caption("The baseline core is uniform throughout. Select another structure concept to introduce reinforcement, "
                   "resin-rich regions, fillers or patterned copper interfaces.")
    if layout == "Resin-rich":
        st.caption("R1–R3 mark resin-filled gaps between the glass yarns. Their boundaries follow the weave; "
                   "no reinforcement passes through the highlighted pockets.")
    if layout == "Copper interfaces":
        st.caption("The plan view shows top-surface copper tabs; the dashed line marks the cross-section position. "
                   "The cross-section also exposes bottom tabs and vertical copper features. The laminate is uniform, without glass weave.")
    factor = st.radio("Explore a factor to add" if homogeneous else "Explore a factor", list(FACTORS),
                      format_func=lambda name: "Resin matrix and resin-rich regions" if name == "Resin matrix" else name,
                      horizontal=True, key="inverse_structure_factor")
    note = FACTORS[factor]
    behaviour, research = st.columns(2, gap="large", border=True)
    with behaviour:
        st.markdown("**Structure → property → stress**")
        st.write(note["effect"])
        for point in note.get("mechanism", []):
            st.markdown(f"- {point}")
        st.caption(note["parameter_note"])
        saved = st.session_state.get("sensitivity_result")
        if (saved and saved["signature"] == st.session_state.get("sensitivity_current_signature")
                and "ST" in saved["result"]["indices"]):
            # Display the bridge to the ranking without translating structure into invented values.
            material = saved["result"]["indices"]
            top_keys = material[material["Parameter"].isin(MATERIAL)].nlargest(3, "ST")["Parameter"].tolist()
            relevant = [key for key in note["properties"] if key in top_keys]
            if relevant:
                st.info("Related to the leading sensitivity parameters: " + ", ".join(PARAMETERS[key][0] for key in relevant))
        st.caption(note["limit"])
        if note.get("units"):
            st.caption(note["units"])
    with research:
        st.markdown("**Research evidence**")
        for study in [note, *note.get("additional_research", [])]:
            st.write(study["evidence"])
            st.markdown(f"[{study['title']}]({study['source']})")
            if study.get("scope"):
                st.caption(study["scope"])
    with st.expander("Reference Links"):
        for name, entry in FACTORS.items():
            label = "Resin matrix and resin-rich regions" if name == "Resin matrix" else name
            st.markdown(f"**{label}**")
            for study in [entry, *entry.get("additional_research", [])]:
                st.markdown(f"[{study['title']}]({study['source']})")
    st.caption("For quantitative layout comparison, the next step is measured constituent properties plus "
               "a laminate or interface model that resolves direction, layer sequence and local geometry.")
