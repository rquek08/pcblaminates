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
        properties=["E", "alpha_lam", "nu"],
        effect="Glass content and direction change the laminate's effective stiffness and thermal expansion. "
               "More stiffness can also raise the stress index for the same mismatch strain.",
        evidence="PCB experiments and modelling found that reinforcement type and fiber volume fraction "
                 "change the measured Young's modulus and CTE.",
        source="https://doi.org/10.1007/s10854-017-8449-2", title="Veldhuijzen van Zanten et al. · PCB constituent properties (2018)",
        limit="Bulk directional properties must be measured for the actual glass-to-resin ratio."),
    "Resin matrix": dict(
        properties=["beta", "Csat", "D", "E"],
        effect="At equal matrix moisture content, less fiber restraint can permit greater local swelling.",
        mechanism=[
            "Epoxy absorbs water and swells; glass is treated as non-swelling. Fiber restraint suppresses axial swelling much more than transverse swelling.",
        ],
        evidence="Experiments and models show strongly directional swelling in glass/epoxy.",
        complications=[
            "Restraint produces interface mismatch stresses; uneven moisture uptake may promote microcracking.",
            "Fiber arrangement changes local swelling; bulk averages can hide it.",
            "The introduction also links swelling to reduced matrix strength.",
            "Interpretation: greater free swelling alone does not establish greater stress or damage.",
        ],
        scope="60°C water immersion; not a PCB-pocket test. Linear elasticity; interface imperfections neglected.",
        units="Paper coefficients use moisture mass percentages, not the app's kg/m³ basis.",
        source="https://www.mdpi.com/2504-477X/3/1/10",
        title="Krauklis, Gagani & Echtermeyer · Orthotropic hygroscopic swelling (2019)",
        limit="The app does not resolve local restraint or directional swelling."),
    "Glass weave architecture": dict(
        properties=["E", "alpha_lam", "nu"],
        effect="Warp/fill spacing and yarn crimp change directional reinforcement and the local stress distribution.",
        evidence="Finite-element research on woven glass/epoxy found weave-dependent interior and edge stresses "
                 "under cryogenic conditions. Those results do not establish the ranking of PCB weaves at service temperatures.",
        source="https://doi.org/10.1520/STP10414S", title="Kriz · Edge stresses in woven laminates (1989)",
        limit="The app uses one effective in-plane modulus and CTE; it does not resolve warp/fill anisotropy or crimp."),
    "Copper foil interfaces": dict(
        properties=["alpha_lam", "E"],
        effect="Copper traces bonded to the laminate transfer thermal-expansion mismatch through their interfaces. "
               "Copper surface treatment and resin bonding also govern adhesion.",
        evidence="90° peel tests on copper/glass-epoxy specimens showed that foil roughness affected peel strength "
                 "and local load variation through mechanical anchoring.",
        source="https://doi.org/10.1163/156855402320257320",
        title="Ikegawa et al. · Copper/laminate adhesion (2002)",
        limit="The stress index includes copper CTE, but trace geometry, local edge stresses, roughness and adhesion strength "
              "need a separate interface model."),
    "Fillers and additives": dict(
        properties=["alpha_lam", "E", "Csat", "D"],
        effect="Filler type, loading and dispersion change matrix properties. Benefits and tradeoffs depend on the formulation.",
        evidence="Experiments on two silica-filled epoxy systems found lower CTE and higher stiffness and toughness. "
                 "Moisture absorption was also measured; results cannot be transferred directly to a commercial PCB grade.",
        source="https://doi.org/10.1016/j.ijadhadh.2017.06.006",
        title="Dittanet et al. · Silica-filled epoxy properties (2017)",
        limit="The app has no filler-to-property conversion. Use measured properties for a specific formulation."),
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
    st.subheader("What about inhomogeneous laminates and copper interfaces?")
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
                      horizontal=True, key="inverse_structure_factor")
    note = FACTORS[factor]
    behaviour, research = st.columns(2, gap="large", border=True)
    with behaviour:
        st.markdown("**Structure → property → stress**")
        st.write(note["effect"])
        for point in note.get("mechanism", []):
            st.markdown(f"- {point}")
        st.caption("Related app parameters: " + ", ".join(note["properties"]))
        saved = st.session_state.get("sensitivity_result")
        if saved and saved["signature"] == st.session_state.get("sensitivity_current_signature"):
            # Display the bridge to the ranking without translating structure into invented values.
            material = saved["result"]["indices"]
            top_keys = material[material["Parameter"].isin(MATERIAL)].nlargest(3, "ST")["Parameter"].tolist()
            relevant = [key for key in note["properties"] if key in top_keys]
            if relevant:
                st.info("Related to your leading sensitivity parameters: " + ", ".join(PARAMETERS[key][0] for key in relevant))
        st.caption(note["limit"])
        if note.get("units"):
            st.caption(note["units"])
    with research:
        st.markdown("**Research evidence**")
        st.write(note["evidence"])
        if note.get("complications"):
            st.markdown("**Complications of swelling**")
            for point in note["complications"]:
                st.markdown(f"- {point}")
            st.caption(note["scope"])
        st.markdown(f"[{note['title']}]({note['source']})")
    with st.expander("Research across all five factors"):
        for name, entry in FACTORS.items():
            st.markdown(f"**{name}** — {entry['evidence']} [{entry['title']}]({entry['source']})")
    st.caption("For quantitative layout comparison, the next step is measured constituent properties plus "
               "a laminate or interface model that resolves direction, layer sequence and local geometry.")
