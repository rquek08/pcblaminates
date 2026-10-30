"""Hygrothermal mismatch stress explorer.

Includes:
  1. Numerical stress pipeline: water molar mass, diffusion, strain, stress index.
  2. Material presets: inputs supplied to the calculation.
  3. User interface: module loading, equation labels, charts, styling, navigation.

The numerical functions use metres, seconds, MPa, and CTE values in ppm/K.
The UI converts entered thickness (mm) and soak time (hours) before calling them.
"""

import streamlit as st
import json
import re
import sys
from importlib.util import module_from_spec, spec_from_file_location
import numpy as np
import plotly.graph_objects as go
from pathlib import Path


# -----------------------------------------------------------------------------
# 1. STRESS CALCULATION PIPELINE
# -----------------------------------------------------------------------------
MOLAR_MASS_WATER = 0.018015  # kg/mol


def compute_fickian_uptake(
    moisture_diffusivity,
    laminate_thickness_meters,
    exposure_time_seconds,
    number_of_series_terms=50,
):
    """Return the fraction of equilibrium moisture capacity absorbed, from 0 to 1.

    Inputs:
        moisture_diffusivity: moisture diffusion coefficient, in m²/s.
        laminate_thickness_meters: FULL sheet thickness, in m.
        exposure_time_seconds: a single exposure time or an array, in seconds.
        number_of_series_terms: number of terms in the diffusion series.

    Assumes an initially dry homogeneous sheet, constant diffusivity, and
    constant moisture concentration at the exposed face, and zero moisture
    flux at the opposite face interfacing with copper foil.

    The diffusion Fourier number is diffusivity multiplied by exposure time,
    divided by (2h)², where h is the laminate thickness. Below 0.005, use the
    short-time approximation; otherwise use the odd-term sheet series.
    """
    exposure_times_seconds = np.asarray(exposure_time_seconds, dtype=float)
    diffusion_fourier_number = (
        moisture_diffusivity
        * np.maximum(exposure_times_seconds, 0.0)
        / (2 * laminate_thickness_meters)**2
    )
    moisture_uptake_fraction = np.zeros_like(diffusion_fourier_number)
    # Avoid a truncated-series offset near zero exposure time or diffusivity.
    short_time_mask = (diffusion_fourier_number > 0) & (diffusion_fourier_number < 0.005)
    moisture_uptake_fraction[short_time_mask] = 4 * np.sqrt(
        diffusion_fourier_number[short_time_mask] / np.pi
    )
    series_solution_mask = diffusion_fourier_number >= 0.005
    odd_mode_numbers = 2 * np.arange(number_of_series_terms) + 1
    diffusion_series_terms = np.exp(
        -np.pi**2
        * diffusion_fourier_number[series_solution_mask, None]
        * odd_mode_numbers**2
    ) / odd_mode_numbers**2
    moisture_uptake_fraction[series_solution_mask] = (
        1.0 - (8.0 / np.pi**2) * diffusion_series_terms.sum(axis=1)
    )
    moisture_uptake_fraction = np.clip(moisture_uptake_fraction, 0.0, 1.0)
    return (
        float(moisture_uptake_fraction)
        if exposure_times_seconds.ndim == 0
        else moisture_uptake_fraction
    )


def run_analytical_stress_pipeline(
    moisture_diffusivity,
    saturation_moisture_concentration,
    laminate_thickness_meters,
    exposure_time_seconds,
    hygroscopic_expansion_coefficient,
    laminate_thermal_expansion_coefficient,
    copper_thermal_expansion_coefficient,
    temperature_change,
    youngs_modulus,
    poisson_ratio,
):
    """Compute uptake → average moisture → swelling → mismatch → stress index.

    Input units:
        moisture_diffusivity: m²/s.
        saturation_moisture_concentration: mol/m³.
        laminate_thickness_meters: full laminate thickness, in m.
        exposure_time_seconds: seconds.
        hygroscopic_expansion_coefficient: linear in-plane swelling per water
            mass concentration, in m³/kg.
        laminate_thermal_expansion_coefficient: ppm/K.
        copper_thermal_expansion_coefficient: ppm/K.
        temperature_change: K or °C difference.
        youngs_modulus: MPa.
        poisson_ratio: dimensionless.

    Output units:
        Moisture uptake fraction and strains: dimensionless.
        Average moisture concentration: mol/m³.
        Absorbed water mass concentration: kg/m³.
        Biaxial modulus and stress indices: MPa.

    This preserves the existing signed mismatch index and its convention.
    It does not solve layer load sharing, interface shear/peel, or curvature.
    """
    # 1. Determine the fraction of equilibrium moisture capacity absorbed.
    moisture_uptake_fraction = compute_fickian_uptake(
        moisture_diffusivity, laminate_thickness_meters, exposure_time_seconds
    )

    # 2. Scale the saturation concentration and convert mol/m³ to kg/m³.
    average_moisture_concentration = moisture_uptake_fraction * saturation_moisture_concentration
    absorbed_water_mass_concentration = average_moisture_concentration * MOLAR_MASS_WATER

    # 3. Multiply absorbed water concentration by the expansion coefficient.
    hygroscopic_strain = hygroscopic_expansion_coefficient * absorbed_water_mass_concentration

    # 4. Multiply the thermal expansion difference by the temperature change.
    #    The factor 1e-6 converts coefficients from ppm/K to 1/K.
    thermal_mismatch_strain = (
        (laminate_thermal_expansion_coefficient - copper_thermal_expansion_coefficient)
        * 1e-6
        * temperature_change
    )

    # 5. Multiply each strain contribution by the biaxial modulus, in MPa.
    biaxial_modulus = youngs_modulus / (1.0 - poisson_ratio)
    hygroscopic_stress_index = biaxial_modulus * hygroscopic_strain
    thermal_stress_index = biaxial_modulus * thermal_mismatch_strain
    combined_stress_index = biaxial_modulus * (hygroscopic_strain + thermal_mismatch_strain)

    # Preserve the existing result keys used by the app's charts and saved state.
    return {
        "Mt_Minf": moisture_uptake_fraction,
        "C_t": average_moisture_concentration,
        "W_t": absorbed_water_mass_concentration,
        "eps_h": hygroscopic_strain,
        "eps_t": thermal_mismatch_strain,
        "eps_net": hygroscopic_strain + thermal_mismatch_strain,
        "biaxial_modulus": biaxial_modulus,
        "sigma_hygro": hygroscopic_stress_index,
        "sigma_thermal": thermal_stress_index,
        "sigma_index": combined_stress_index
    }


# -----------------------------------------------------------------------------
# 2. MATERIAL PRESETS AND INPUT DATA
# -----------------------------------------------------------------------------
# Condition-specific laminate inputs extracted from Formula_sheet_V2.xlsx.
# Store the inputs locally so running the app does not require the source workbook.
MATERIAL_LIBRARY = json.loads(
    (Path(__file__).parent / "material_presets.json").read_text(encoding="utf-8")
)
PRESET_METADATA = {record["name"]: record for record in MATERIAL_LIBRARY["presets"]}
PRESETS = {name: record["properties"] for name, record in PRESET_METADATA.items()}
PRESETS.update({
    # Custom entry option
    "Custom / Manual Entry": {
        "E": 25000.0, "nu": 0.200, "alpha_lam": 12.0,
        "D": 1.00e-12, "Csat": 150.00, "beta": 1.00e-4, "h_mm": 1.6
    }
})
LEGACY_PRESET_NAMES = {
    "Isola 370HR (High-Tg Core)": "Isola 370HR — 85°C/85%RH",
    "Isola G200 (BT-Epoxy System)": "Isola G200 — 85°C/85%RH",
    "Isola IS410 (Lead-Free FR-4)": "Isola IS410 — 85°C/85%RH",
    "Shengyi SI10US (Halogen-Free CCL)": "Shengyi SI10US/SI13U — 85°C/85%RH",
}


# -----------------------------------------------------------------------------
# 3. USER INTERFACE: EXPLORATION MODULES
# -----------------------------------------------------------------------------
# Load the exploration modules from this app's folder, including when older
# copies from the parent folder are already in Python's module cache.
for module_name, filename in (("typography", "typography.py"),
                              ("stress_results", "stress_results.py"),
                              ("sensitivity_analysis", "sensitivity_analysis.py"),
                              ("_pcb_sensitivity_view", "sensitivity_view.py"),
                              ("inverse_analysis", "inverse_analysis.py"),
                              ("laminate_showcase", "laminate_showcase.py"),
                              ("_pcb_stack_view", "pcb_stack_view.py"),
                              ("_pcb_inverse_view", "inverse_view.py")):
    sensitivity_spec = spec_from_file_location(module_name, Path(__file__).parent / filename)
    sensitivity_module = module_from_spec(sensitivity_spec)
    sys.modules[module_name] = sensitivity_module
    sensitivity_spec.loader.exec_module(sensitivity_module)
render_sensitivity = sys.modules["_pcb_sensitivity_view"].render_sensitivity
render_inverse = sys.modules["_pcb_inverse_view"].render_inverse
render_stack = sys.modules["_pcb_stack_view"].render_stack
typography = sys.modules["typography"]
stress_results = sys.modules["stress_results"]


# -----------------------------------------------------------------------------
# 4. USER INTERFACE: EQUATION LABELS
# -----------------------------------------------------------------------------
# LaTeX, rendered symbol, meaning and units shared by all home-page equations.
EQUATION_SYMBOLS = {
    "D": ("D", "D", "Moisture diffusivity", "m²/s"),
    "h": ("h", "h", "Laminate thickness", "m"),
    "t": ("t", "t", "Exposure time", "s"),
    "z": ("z", "z", "Position through the laminate thickness", "m"),
    "C": ("C", "C", "Local moisture concentration in the diffusion equation", "mol/m³"),
    "uptake": (r"\frac{M_t}{M_\infty}", "M∞Mt", "Fractional moisture uptake relative to saturation", "Dimensionless"),
    "n": ("n", "n", "Series summation index, starting at zero", "Dimensionless"),
    "Csat": (r"C_{\mathrm{sat}}", "Csat", "Saturation moisture concentration", "mol/m³"),
    "Ct": ("C_t", "Ct", "Average moisture concentration at time t", "mol/m³"),
    "W": ("W_t", "Wt", "Absorbed moisture concentration by mass at time t", "kg/m³"),
    "water_mass": (r"M_{\mathrm{H_2O}}", "MH2O", "Molar mass of water, 0.018015", "kg/mol"),
    "beta": (r"\beta", "β", "Coefficient of hygroscopic expansion", "m³/kg"),
    "alpha_lam": (r"\alpha_{\mathrm{lam}}", "αlam", "Laminate in-plane coefficient of thermal expansion", "ppm/K"),
    "alpha_cu": (r"\alpha_{\mathrm{Cu}}", "αCu", "Copper coefficient of thermal expansion", "ppm/K"),
    "T": ("T", "T", "Exposure temperature", "°C or K"),
    "Tref": (r"T_{\mathrm{ref}}", "Tref", "Reference temperature, in the same units as T", "°C or K"),
    "delta_T": (r"\Delta T", "ΔT", "Temperature change from the reference", "K or °C difference"),
    "E": (r"E_{\mathrm{eff}}", "Eeff", "Effective modulus", "MPa"),
    "nu": (r"\nu_{\mathrm{eff}}", "νeff", "Effective Poisson's ratio", "Dimensionless"),
    "eps_h": (r"\varepsilon_h", "εh", "Hygroscopic swelling strain", "Dimensionless"),
    "eps_T": (r"\varepsilon_T", "εT", "Thermal mismatch strain", "Dimensionless"),
    "sigma_h": (r"\sigma_h", "σh", "Hygroscopic contribution to the stress index", "MPa"),
    "sigma_T": (r"\sigma_T", "σT", "Thermal contribution to the stress index", "MPa"),
    "sigma_index": (r"\sigma_{\mathrm{index}}", "σindex", "Combined stress index", "MPa"),
}


# -----------------------------------------------------------------------------
# 5. USER INTERFACE: CHARTS AND RESULT EXPLANATIONS
# -----------------------------------------------------------------------------
def render_moisture_uptake(
    moisture_diffusivity,
    laminate_thickness_meters,
    exposure_time_hours,
    current_moisture_uptake_fraction,
    *, height=370, compact=False,
):
    if compact:
        window_days = st.segmented_control("Time window (days)", [30, 120, 365], default=30, required=True,
                                           format_func=lambda days: f"{days} days", key="uptake_window_days",
                                           label_visibility="collapsed")
    else:
        st.subheader("Moisture Uptake Over Time")
        window_days = st.radio("Time window (days)", [30, 120, 365], horizontal=True,
                               key="uptake_window_days")
    soak_days = exposure_time_hours / 24.0
    end_days = max(float(window_days), soak_days * 1.1)
    # Sample early times more densely, and include the exact selected soak time.
    days = np.unique(np.r_[0.0, np.geomspace(end_days * 1e-7, end_days, 500), soak_days])
    moisture_uptake_fraction = compute_fickian_uptake(
        moisture_diffusivity, laminate_thickness_meters, days * 86400.0
    )
    figure = go.Figure(go.Scatter(
        x=days, y=moisture_uptake_fraction, mode="lines", name="Predicted uptake",
        line=dict(color="#368575", width=3), fill="tozeroy",
        fillcolor="rgba(54,133,117,.08)",
        hovertemplate="Time: %{x:.3g} days<br>Equilibrium uptake: %{y:.2%}<extra></extra>",
    ))
    figure.add_hline(y=1.0, line_dash="dash", line_color="#78988e")
    figure.add_vline(x=soak_days, line_dash="dot", line_color="#c87d55")
    figure.add_trace(go.Scatter(
        x=[soak_days], y=[current_moisture_uptake_fraction], mode="markers", name="Selected soak time",
        marker=dict(color="#c87d55", size=11, line=dict(color="white", width=2)),
        hovertemplate=f"Soak time: {exposure_time_hours:g} h<br>Equilibrium uptake: %{{y:.2%}}<extra></extra>",
    ))
    figure.update_layout(
        height=height, margin=dict(l=10, r=10, t=15, b=10),
        xaxis=dict(title="Exposure time (days)", range=[0, end_days]),
        yaxis=dict(title="Moisture uptake (%)" if compact else "Fractional moisture uptake (Mt/M∞)",
                   range=[0, 1.05], tickformat=".0%"),
        legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0),
    )
    st.plotly_chart(figure, width="stretch", key="moisture_uptake_chart")
    if compact:
        st.caption(f"Selected soak: {exposure_time_hours:g} h · {current_moisture_uptake_fraction:.2%} of equilibrium uptake. "
                   "One exposed face; impermeable copper interface. Diffusivity stays fixed.")
    else:
        st.caption(f"Selected soak time: {exposure_time_hours:g} h · {current_moisture_uptake_fraction:.2%} of equilibrium uptake. "
                   "100% means this material's own equilibrium capacity, not a mass percentage of water.")
        st.caption("Prediction for an initially dry, homogeneous laminate with one exposed face and an impermeable copper interface, using the current "
                   "diffusivity and thickness. Diffusivity stays fixed throughout the curve; "
                   "Change in Temperature affects thermal strain only.")
    if end_days > window_days:
        st.caption("The time window has been extended to include the selected soak time.")


def render_3d_deformation(
    hygroscopic_strain, thermal_mismatch_strain, net_mismatch_strain, n_frames=20
):
    """Generates an interactive animated 3D visual with a Play/Pause button."""
    L_nom = 20.0
    x = np.linspace(-L_nom / 2.0, L_nom / 2.0, 25)
    y = np.linspace(-L_nom / 2.0, L_nom / 2.0, 25)
    X, Y = np.meshgrid(x, y)

    scale_factor = 2500.0  # Fixed visually distinct exaggeration factor
    
    # Pre-generate frame surfaces (interpolating from 0 deformation to full strain)
    frames = []
    alphas = np.linspace(0.0, 1.0, n_frames)
    
    for k, alpha in enumerate(alphas):
        # Progressively apply strain from initial undeformed state to calculated state
        curv = (alpha * net_mismatch_strain) * scale_factor * 0.04
        Z_dielectric = curv * (X**2 + Y**2) / L_nom
        Z_copper = Z_dielectric + 0.35

        frames.append(
            go.Frame(
                data=[
                    go.Surface(x=X, y=Y, z=Z_copper),
                    go.Surface(x=X, y=Y, z=Z_dielectric - 0.8)
                ],
                name=f"frame_{k}"
            )
        )

    # Base initial frame (state at t = 0 or initial reference)
    Z_init_d = np.zeros_like(X)
    Z_init_cu = Z_init_d + 0.35

    fig = go.Figure(
        data=[
            go.Surface(
                x=X, y=Y, z=Z_init_cu,
                colorscale=[[0, '#C87D55'], [1, '#E2976D']],
                showscale=False,
                name='Copper Foil',
                opacity=0.95
            ),
            go.Surface(
                x=X, y=Y, z=Z_init_d - 0.8,
                colorscale=[[0, '#2C3E50'], [1, '#4A6572']],
                showscale=False,
                name='Dielectric Core',
                opacity=0.85
            )
        ],
        frames=frames
    )

    # Add Play and Pause buttons directly inside the 3D viewer
    fig.update_layout(
        updatemenus=[{
            "type": "buttons",
            "showactive": False,
            "direction": "left",
            "x": 0.05,
            "y": 1.15,
            "buttons": [
                {
                    "label": "▶ Play Deformation",
                    "method": "animate",
                    "args": [
                        None,
                        {
                            "frame": {"duration": 60, "redraw": True},
                            "fromcurrent": True,
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }
                    ]
                },
                {
                    "label": "❚❚ Pause",
                    "method": "animate",
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0}
                        }
                    ]
                }
            ]
        }],
        scene=dict(
            xaxis=dict(title="Length (mm)", showbackground=True),
            yaxis=dict(title="Width (mm)", showbackground=True),
            zaxis=dict(title="Deflection (Scaled)", showbackground=True, range=[-1.8, 1.8]),
            aspectratio=dict(x=1.2, y=1.2, z=0.4),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=1.0))
        ),
        margin=dict(l=0, r=0, b=0, t=30),
        height=520
    )
    return fig


# -----------------------------------------------------------------------------
# 6. USER INTERFACE: NAVIGATION, STYLING, AND PAGE LAYOUTS
# -----------------------------------------------------------------------------
HOME_PAGE = "Home"
NAVIGATION = ["Stress Analyser", "Sensitivity Analysis", "Inverse Exploration"]


def navigate(section=None):
    st.session_state["main_navigation"] = section or HOME_PAGE
    request_navigation_scroll()


def request_navigation_scroll():
    st.session_state["scroll_to_top"] = True


def scroll_to_top_after_navigation():
    if not st.session_state.pop("scroll_to_top", False):
        return
    # st.html executes in the app document, where Streamlit's main scroll
    # container lives. Wait for the destination's elements to finish mounting.
    st.html("""
    <script>
    requestAnimationFrame(() => {
        const main = document.querySelector('[data-testid="stMain"]');
        if (main) main.scrollTo({top: 0, left: 0, behavior: "instant"});
        window.scrollTo({top: 0, left: 0, behavior: "instant"});
    });
    </script>
    """, unsafe_allow_javascript=True)


def render_style():
    typography.configure_charts()
    st.html("<style>" + typography.font_css() + "</style>")
    st.html(Path(__file__).parent / "assets" / "site_theme.css")
    st.html(Path(__file__).parent / "assets" / "compact_toolbar.html", unsafe_allow_javascript=True)


def render_symbol_equation(equation):
    # Group compound symbols so the entire term is one hover target in KaTeX.
    equation = re.sub(r"\[\[(\w+)\]\]",
                      lambda match: r"\mathord{" + EQUATION_SYMBOLS[match[1]][0] + "}", equation)
    st.latex(equation)


def render_equation_symbol_hints():
    meanings = {symbol: {"meaning": meaning, "units": units}
                for _, symbol, meaning, units in EQUATION_SYMBOLS.values()}
    script = (Path(__file__).parent / "assets" / "equation_tooltips.js").read_text(encoding="utf-8")
    st.html(Path(__file__).parent / "assets" / "equation_hovercards.css")
    st.html("<script>\n(() => {\nconst meanings = " + json.dumps(meanings)
            + ";\n" + script + "\n})();\n</script>", unsafe_allow_javascript=True)


def render_home_stress_flow():
    st.subheader("Combined Stress Index")
    st.write(
        "Moisture causes the laminate to swell, while temperature changes produce different degrees of "
        "expansion in the copper and laminate. Bonding restricts this relative movement, "
        "creating mechanical loading at their interface. "
    )
    moisture, thermal = st.columns(2, gap="large", border=True)
    with moisture.container(border=False, key="home_moisture_path"):
        with st.container(border=False, key="home_moisture_heading"):
            st.markdown("#### Moisture contribution")
        with st.container(border=False, key="home_moisture_step_1"):
            st.markdown('<p class="flow-step">01 · MOISTURE UPTAKE</p>', unsafe_allow_html=True)
            st.markdown('<p class="flow-desc">Diffusion sets the moisture absorbed at exposure time t.</p>', unsafe_allow_html=True)
            st.markdown(
                """
                    <style>
                    .flow-step {
                    font-size: 0.75rem !important;
                    font-weight: 600;
                    letter-spacing: 0.05em;
                    }
                    .flow-desc {
                    font-size: 0.8rem !important;  /* Adjust size as needed */
                    margin-top: -0.25rem;
                    margin-bottom: 0.5rem;
                }
                </style>
            """,
            unsafe_allow_html=True,
        )
            render_symbol_equation(r"[[Ct]] = [[uptake]]\,[[Csat]], \qquad [[W]] = [[Ct]] [[water_mass]]")
        st.markdown('<p class="flow-down" aria-hidden="true">↓</p>', unsafe_allow_html=True)
        with st.container(border=False, key="home_moisture_step_2"):
            st.markdown('<p class="flow-step" style="font-size: 0.75rem;">02 · SWELLING STRAIN</p>', unsafe_allow_html=True)
            st.markdown('<p class="flow-desc">Absorbed water causes the laminate to expand.</p>', unsafe_allow_html=True)
            st.markdown(
                    """
                        <style>
                        .flow-step {
                        font-size: 0.75rem !important;
                        font-weight: 600;
                        letter-spacing: 0.05em;
                        }
                        .flow-desc {
                        font-size: 0.8rem !important;  /* Adjust size as needed */
                        margin-top: -0.25rem;
                        margin-bottom: 0.5rem;
                    }
                    </style>
                """,
                unsafe_allow_html=True,
            )
            render_symbol_equation(r"[[eps_h]] = [[beta]]\,[[W]]")
        st.markdown('<p class="flow-down" aria-hidden="true">↓</p>', unsafe_allow_html=True)
        with st.container(border=False, key="home_moisture_step_3"):
            st.markdown('<p class="flow-step" style="font-size: 0.75rem;">03 · STRESS-INDEX CONTRIBUTION</p>', unsafe_allow_html=True)
            render_symbol_equation(r"[[sigma_h]] = \frac{[[E]]}{1-[[nu]]}\,[[eps_h]]")
    with thermal.container(border=False, key="home_thermal_path"):
        with st.container(border=False, key="home_thermal_heading"):
            st.markdown("#### Thermal contribution")
        with st.container(border=False, key="home_thermal_step_1"):
            st.markdown('<p class="flow-step" style="font-size: 0.75rem;">01 · TEMPERATURE CHANGE</p>', unsafe_allow_html=True)
            st.markdown('<p class="flow-desc">Heating or cooling changes each material\'s dimensions.</p>', unsafe_allow_html=True)
            st.markdown(
                    """
                        <style>
                        .flow-step {
                        font-size: 0.75rem !important;
                        font-weight: 600;
                        letter-spacing: 0.05em;
                        }
                        .flow-desc {
                            font-size: 0.8rem !important;  /* Adjust size as needed */
                            margin-top: -0.25rem;
                            margin-bottom: 0.5rem;
                        }
                            </style>
                """,
                        unsafe_allow_html=True,
                )
            render_symbol_equation(r"[[delta_T]] = [[T]] - [[Tref]]")
        st.markdown('<p class="flow-down" aria-hidden="true">↓</p>', unsafe_allow_html=True)
        with st.container(border=False, key="home_thermal_step_2"):
            st.markdown('<p class="flow-step" style="font-size: 0.75rem;">02 · EXPANSION MISMATCH STRAIN</p>', unsafe_allow_html=True)
            st.markdown('<p class="flow-desc">Different CTEs create a relative expansion strain.</p>', unsafe_allow_html=True)
            st.markdown(
                    """
                        <style>
                        .flow-step {
                        font-size: 0.75rem !important;
                        font-weight: 600;
                        letter-spacing: 0.05em;
                        }
                        .flow-desc {
                            font-size: 0.8rem !important;  /* Adjust size as needed */
                            margin-top: -0.25rem;
                            margin-bottom: 0.5rem;
                        }
                            </style>
                """,
                        unsafe_allow_html=True,
                )
            render_symbol_equation(r"[[eps_T]] = ([[alpha_lam]]-[[alpha_cu]])\,[[delta_T]]")
        st.markdown('<p class="flow-down" aria-hidden="true">↓</p>', unsafe_allow_html=True)
        with st.container(border=False, key="home_thermal_step_3"):
            st.markdown('<p class="flow-step" style="font-size: 0.75rem;">03 · STRESS-INDEX CONTRIBUTION</p>', unsafe_allow_html=True)
            render_symbol_equation(r"[[sigma_T]] = \frac{[[E]]}{1-[[nu]]}\,[[eps_T]]")

    st.markdown('''
        <div class="flow-merge">
          <svg viewBox="0 0 1000 80" preserveAspectRatio="none" role="img"
               aria-label="Hygroscopic and thermal contributions merge into the combined stress index">
            <g fill="none" stroke="#78988e" stroke-width="2" vector-effect="non-scaling-stroke">
              <path d="M250 0 V20 Q250 32 262 32 H488 Q500 32 500 44" />
              <path d="M750 0 V20 Q750 32 738 32 H512 Q500 32 500 44 V70" />
              <path d="M492 60 L500 70 L508 60" />
            </g>
          </svg>
          <span class="flow-merge-mobile">↓ Combine both contributions</span>
        </div>
    ''', unsafe_allow_html=True)
    with st.container(border=True, key="home_combined_index"):
        st.markdown("#### Combined stress index")
        st.markdown('<p class="flow-desc">A signed measure of the constrained in-plane response for comparing materials and exposure conditions.</p>', unsafe_allow_html=True)
        st.markdown(
                """
                    <style>
                    .flow-step {
                    font-size: 0.75rem !important;
                    font-weight: 600;
                    letter-spacing: 0.05em;
                    }
                    .flow-desc {
                    font-size: 0.8rem !important;  /* Adjust size as needed */
                    margin-top: -0.8rem;
                    margin-bottom: 0.1rem;
                    }
                    </style>
                    """,
                    unsafe_allow_html=True,
            )
        render_symbol_equation(
            r"[[sigma_index]] = [[sigma_h]] + [[sigma_T]]"
            r" = \frac{[[E]]}{1-[[nu]]}"
            r"\left([[eps_h]] + [[eps_T]]\right)"
        )
        #st.write("A signed measure of the constrained in-plane response for comparing materials and exposure conditions.")
        
    with st.expander("Moisture uptake model"), st.container(border=False, key="home_uptake_model"):
        st.write("Fick's second law gives the fractional moisture uptake used in the swelling path; "
                 "the (2h)² term accounts for this one-sided diffusion boundary condition.")
        render_symbol_equation(r"\frac{\partial [[C]]}{\partial [[t]]} = [[D]]\frac{\partial^2 [[C]]}{\partial [[z]]^2}")
        render_symbol_equation(
            r"[[uptake]] = 1-\frac{8}{\pi^2}"
            r"\sum_{[[n]]=0}^{\infty}\frac{1}{(2[[n]]+1)^2}"
            r"\exp\!\left[-\frac{(2[[n]]+1)^2\pi^2 [[D]] [[t]]}{(2[[h]])^2}\right]"
        )
        st.caption("The plane-sheet solution assumes an initially dry laminate, constant diffusivity, "
                   "constant moisture concentration at the exposed face, and zero moisture flux at the copper interface.")
    render_equation_symbol_hints()


STRESS_ANALYSER_ICON = (
    '<path d="m12 3 10 5-10 5L2 8 12 3Z" />'
    '<path d="m2 12 10 5 10-5M2 16l10 5 10-5" />'
)
SENSITIVITY_ANALYSIS_ICON = '<path d="M4 3v17h17M9 16V9M14 16V5M19 16v-4" />'
INVERSE_EXPLORATION_ICON = (
    '<circle cx="11" cy="13" r="9" />'
    '<circle cx="11" cy="13" r="5" />'
    '<path d="m11 13 9-9M16 4h4v4" />'
)


def render_exploration_icon(shapes):
    st.markdown(
        '<div class="exploration-icon" aria-hidden="true">'
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
        f'{shapes}</svg></div>',
        unsafe_allow_html=True,
        width="content",
    )


def render_home():
    text, illustration = st.columns([1.05, 1], gap="large", vertical_alignment="top")
    with text.container(key="home_intro", height="stretch"):
        st.html(
            '<h1 class="hero-title">Hygrothermal Effects on Copper-Laminate Interfaces</h1>',
        )
        st.html(
            '<p class="hero-copy">In the field of electronics, printed circuit boards (PCBs) '
            'are essential components that connect various electronic parts. One of the most '
            'critical issues that can arise during the life of a PCB is delamination, a '
            'condition that compromises the integrity and reliability of the board.</p>'
        )
        st.html(
            '<p class="hero-copy">Temperature and moisture are two contributing factors. '
            'During manufacturing, PCBs are exposed to various heating cycles, including '
            'soldering processes. Each of these cycles can create stresses that result in '
            'separation between layers. Moisture absorption can also lead to delamination, '
            'particularly if the PCB material is prone to moisture ingress. When moisture '
            'penetrates the PCB, it can cause the internal layers to expand. Under high '
            'temperatures, trapped moisture can turn into steam, creating pressure that '
            'leads to layer separation.</p>',
        )
    with illustration.container(key="home_illustration", height="stretch"):
        animation = (Path(__file__).parent / "assets" / "laminate_animation.html").read_text(encoding="utf-8")
        st.iframe(typography.document_fonts(animation), height="stretch")

    st.divider()
    render_home_stress_flow()

    st.divider()
    st.subheader("Laminate Stress Explorer")
    simulation, sensitivity, inverse = st.columns(3, gap="medium", border=True)
    with simulation:
        with st.container(border=False, height="stretch", key="home_simulation_card"):
            render_exploration_icon(STRESS_ANALYSER_ICON)
            st.caption("01 · STRESS ANALYSER")
            st.markdown("#### Follow the physics")
            st.write("Input properties of your chosen laminate and its exposure conditions. Calculate moisture uptake, "
                     "swelling, and thermal mismatch to evaluate the resulting stress.")
        st.button("View section →", on_click=navigate, args=(NAVIGATION[0],), key="home_simulation")
    with sensitivity:
        with st.container(border=False, height="stretch", key="home_sensitivity_card"):
            render_exploration_icon(SENSITIVITY_ANALYSIS_ICON)
            st.caption("02 · SENSITIVITY ANALYSIS")
            st.markdown("#### Understand what matters")
            st.write("Explore the influence of material properties "
                     "and exposure conditions on the relevant strains and stresses. Compare their main effects and interactions with Sobol analysis.")
        st.button("View section →", on_click=navigate, args=(NAVIGATION[1],), key="home_sensitivity")
    with inverse:
        with st.container(border=False, height="stretch", key="home_inverse_card"):
            render_exploration_icon(INVERSE_EXPLORATION_ICON)
            st.caption("03 · INVERSE EXPLORATION")
            st.markdown("#### Find materials and explore structures")
            st.write("Find up to 10 materials that meet your application requirements, ranked by stress, swelling or moisture content. ")
        st.button("View section →", on_click=navigate, args=(NAVIGATION[2],), key="home_inverse")
    st.caption("The explorer calculates an analytical mismatch stress index. "
               "Delamination assessment requires specific geometry and interface adhesion data.")


def load_preset_values():
    for key, value in PRESETS[st.session_state["preset_selector"]].items():
        st.session_state[f"field_{key}"] = value


def render_simulation_inputs():
    # Keep a non-widget snapshot so leaving this section does not erase edits.
    defaults = {"preset_selector": next(iter(PRESETS)), "field_delta_T": 60.0,
                "field_t_hours": 24.0, "field_alpha_cu": 16.5,
                **{f"field_{key}": value for key, value in PRESETS[next(iter(PRESETS))].items()}}
    saved = st.session_state.get("simulation_inputs", defaults)
    for key, value in saved.items():
        st.session_state.setdefault(key, value)
    selected = st.session_state["preset_selector"]
    if selected not in PRESETS:
        st.session_state["preset_selector"] = LEGACY_PRESET_NAMES.get(selected, "Custom / Manual Entry")

    with st.container(key="simulation_workspace"):
        structure, inputs = st.columns([1.05, 1], gap="medium")
        with inputs.container(border=True, key="simulation_scenario"):
            st.html('<p class="compact-section-label">Laminate</p>')
            st.selectbox("Laminate", list(PRESETS), key="preset_selector", on_change=load_preset_values,
                         label_visibility="collapsed",
                         help="Load the selected laminate's properties. All inputs remain editable.")
            st.html('<p class="compact-section-label">Exposure & geometry</p>')
            temperature, time = st.columns(2, gap="small")
            temperature_change = temperature.number_input("Temperature change ΔT (°C / K)", key="field_delta_T", step=5.0)
            exposure_time_hours = time.number_input("Soak time (h)", min_value=0.0, key="field_t_hours", step=1.0)
            geometry, copper = st.columns(2, gap="small")
            laminate_thickness_millimeters = geometry.number_input("Thickness h (mm)", min_value=0.001,
                                                                   key="field_h_mm", step=0.1)
            copper_thermal_expansion_coefficient = copper.number_input("Copper CTE (ppm/K)", key="field_alpha_cu", step=0.5)

            st.html('<hr class="compact-section-divider">')
            st.html('<p class="compact-section-label">Material properties</p>')
            stiffness, poisson = st.columns(2, gap="small")
            youngs_modulus = stiffness.number_input("Young's modulus E (MPa)", min_value=0.001, key="field_E", step=500.0)
            poisson_ratio = poisson.number_input("Poisson's ratio ν", min_value=-0.99, max_value=0.499,
                                                 key="field_nu", step=0.01, format="%.3f")
            expansion, diffusion = st.columns(2, gap="small", vertical_alignment="bottom")
            laminate_thermal_expansion_coefficient = expansion.number_input("Laminate CTE (ppm/K)", key="field_alpha_lam", step=0.5)
            moisture_diffusivity = diffusion.number_input("Diffusivity D (m²/s)", min_value=0.0, key="field_D",
                                                         step=1e-14, format="%.2e",
                                                         help="Effective Fickian diffusivity at chamber temperature.")
            capacity, swelling = st.columns(2, gap="small")
            saturation_moisture_concentration = capacity.number_input("Saturation Csat (mol/m³)", min_value=0.0,
                                                                      key="field_Csat", step=10.0,
                                                                      help="Csat ≈ (WA% / 100) × (Density / 0.018015)")
            hygroscopic_expansion_coefficient = swelling.number_input("Moisture expansion β (m³/kg)", min_value=0.0,
                                                                      key="field_beta", step=1e-5, format="%.2e",
                                                                      help="In-plane coefficient of hygroscopic expansion.")
            with st.container(key="simulation_actions"):
                reference, action = st.columns([1, 1.5], gap="small", vertical_alignment="bottom")
                metadata = PRESET_METADATA.get(st.session_state["preset_selector"])
                if metadata:
                    with reference.popover("Material reference", width="stretch"):
                        st.caption(f"{metadata['material_class']} · {metadata['condition']}.")
                        st.caption("Preset properties are compiled from publicly available manufacturer datasets and technical "
                                   "datasheets for reference. Original sources retain their respective rights.")
                calculate = action.button("Calculate", type="primary", key="calculate_simulation", width="stretch")

        with structure.container(key="simulation_structure"):
            render_stack(st.session_state["preset_selector"], laminate_thickness_millimeters,
                         laminate_thermal_expansion_coefficient, copper_thermal_expansion_coefficient,
                         compact=True)

    st.session_state["simulation_inputs"] = {key: st.session_state[key] for key in defaults}
    return dict(D=moisture_diffusivity, Csat=saturation_moisture_concentration,
                h_mm=laminate_thickness_millimeters, t_hours=exposure_time_hours,
                beta=hygroscopic_expansion_coefficient, alpha_lam=laminate_thermal_expansion_coefficient,
                alpha_cu=copper_thermal_expansion_coefficient, delta_T=temperature_change,
                E=youngs_modulus, nu=poisson_ratio), calculate


@st.dialog("Calculation results", width="large")
def render_simulation_results(calculation):
    inputs = calculation["scenario"]["inputs"]
    results = calculation["results"]
    with st.container(key="simulation_results_modal"):
        st.caption(f"{calculation['scenario']['preset']} · Soak {inputs['t_hours']:g} h · "
                   f"Thickness {inputs['h_mm']:g} mm · ΔT {inputs['delta_T']:+g} K")
        stress_results.render_summary(results, key="simulation", compact=True)
        contributions, uptake = st.tabs(["Stress contributions", "Moisture uptake"])
        with contributions:
            stress_results.render_contributions(results, key="simulation", height=240)
        with uptake:
            render_moisture_uptake(inputs["D"], inputs["h_mm"] / 1000, inputs["t_hours"], results["Mt_Minf"],
                                   height=240, compact=True)
        st.caption("The calculated stress index is an indicator. Assessing delamination risk requires physical measurements, "
                   "including interface adhesion tests under the relevant exposure conditions.")


def render_simulation():
    inputs, calculate = render_simulation_inputs()
    scenario = {"preset": st.session_state["preset_selector"], "inputs": inputs}
    if calculate:
        results = run_analytical_stress_pipeline(
            moisture_diffusivity=inputs["D"], saturation_moisture_concentration=inputs["Csat"],
            laminate_thickness_meters=inputs["h_mm"] / 1000, exposure_time_seconds=inputs["t_hours"] * 3600,
            hygroscopic_expansion_coefficient=inputs["beta"], laminate_thermal_expansion_coefficient=inputs["alpha_lam"],
            copper_thermal_expansion_coefficient=inputs["alpha_cu"], temperature_change=inputs["delta_T"],
            youngs_modulus=inputs["E"], poisson_ratio=inputs["nu"],
        )
        st.session_state["simulation_calculation"] = dict(scenario=scenario, results=results)
        render_simulation_results(st.session_state["simulation_calculation"])
    elif st.session_state.get("simulation_calculation", {}).get("scenario") != scenario:
        st.session_state.pop("simulation_calculation", None)


def render_inverse_exploration():
    render_inverse(MATERIAL_LIBRARY["presets"])


def main():
    st.set_page_config(page_title="Hygrothermal Effects on Copper-Laminate Interfaces", page_icon="⚡", layout="wide")
    render_style()
    if st.session_state.get("main_navigation") not in [HOME_PAGE, *NAVIGATION]:
        st.session_state["main_navigation"] = HOME_PAGE
    current_page = st.session_state["main_navigation"]
    with st.container(key="site_header"):
        title = current_page if current_page != HOME_PAGE else ""
        st.html(f'<h1 class="site-page-title">{title}</h1>')
        with st.container(horizontal=True, horizontal_alignment="center", key="site_navigation_shell"):
            section = st.pills(
                "Explore", [HOME_PAGE, *NAVIGATION], key="main_navigation", required=True,
                label_visibility="collapsed", width="content", wrap=True,
                format_func=lambda page: "◈ Bonded by Stress" if page == HOME_PAGE else page,
                on_change=request_navigation_scroll,
            )
    if section == NAVIGATION[0]:
        render_simulation()
    elif section == NAVIGATION[1]:
        render_sensitivity(MATERIAL_LIBRARY["presets"])
    elif section == NAVIGATION[2]:
        render_inverse_exploration()
    else:
        render_home()
    scroll_to_top_after_navigation()


if __name__ == "__main__":
    main()
