import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pathlib import Path
from sensitivity_view import render_sensitivity

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
MOLAR_MASS_WATER = 0.018015  # kg/mol

# Verified Material Presets (using industry baseline 1.6 mm for rigid cores)
PRESETS = {
    "Isola 370HR (High-Tg Core)": {
        "E": 25814.0, "nu": 0.177, "alpha_lam": 13.0,
        "D": 1.65e-12, "Csat": 166.53, "beta": 1.15e-4, "h_mm": 1.6
    },
    "Isola G200 (BT-Epoxy System)": {
        "E": 24056.0, "nu": 0.182, "alpha_lam": 13.0,
        "D": 2.10e-12, "Csat": 222.04, "beta": 1.40e-4, "h_mm": 1.6
    },
    "Isola IS410 (Lead-Free FR-4)": {
        "E": 25352.0, "nu": 0.175, "alpha_lam": 11.0,
        "D": 1.85e-12, "Csat": 222.04, "beta": 1.30e-4, "h_mm": 1.6
    },
    "Shengyi SI10US (Halogen-Free CCL)": {
        "E": 24000.0, "nu": 0.200, "alpha_lam": 10.0,
        "D": 8.80e-14, "Csat": 110.00, "beta": 1.20e-4, "h_mm": 1.6
    },
    "Mitsubishi BT Substrate": {
        "E": 27000.0, "nu": 0.180, "alpha_lam": 10.0,
        "D": 1.65e-12, "Csat": 574.00, "beta": 3.80e-5, "h_mm": 1.6
    },
    "Custom / Manual Entry": {
        "E": 25000.0, "nu": 0.200, "alpha_lam": 12.0,
        "D": 1.00e-12, "Csat": 150.00, "beta": 1.00e-4, "h_mm": 1.6
    }
}

# -----------------------------------------------------------------------------
# Physics Pipeline Functions
# -----------------------------------------------------------------------------
def compute_fickian_uptake(D, h, t, n_terms=50):
    """Crank's 1D analytical series solution for plane-sheet diffusion."""
    if t <= 0:
        return 0.0
    n = np.arange(n_terms)
    exponent = -((2 * n + 1) ** 2) * (np.pi ** 2) * D * t / (h ** 2)
    terms = (1.0 / (2 * n + 1) ** 2) * np.exp(exponent)
    uptake = 1.0 - (8.0 / np.pi ** 2) * np.sum(terms)
    return float(np.clip(uptake, 0.0, 1.0))

def run_analytical_stress_pipeline(D, Csat, h, t, beta, alpha_lam, alpha_cu, delta_T, E, nu):
    """Executes the 5-equation hygro-thermo-mechanical analytical pipeline."""
    # 1. Crank's moisture ratio
    mt_minf = compute_fickian_uptake(D, h, t)
    
    # 2. Moisture mass uptake (kg/m^3)
    c_t = mt_minf * Csat
    w_t = c_t * MOLAR_MASS_WATER
    
    # 3. Hygroscopic strain
    eps_h = beta * w_t
    
    # 4. Thermal mismatch strain (in-plane plane-stress constraint)
    eps_t = (alpha_lam - alpha_cu) * 1e-6 * delta_T
    
    # 5. Combined interfacial stress index
    biaxial_modulus = E / (1.0 - nu)
    sigma_hygro = biaxial_modulus * eps_h
    sigma_thermal = biaxial_modulus * eps_t
    sigma_index = biaxial_modulus * (eps_h + eps_t)
    
    return {
        "Mt_Minf": mt_minf,
        "C_t": c_t,
        "W_t": w_t,
        "eps_h": eps_h,
        "eps_t": eps_t,
        "eps_net": eps_h + eps_t,
        "biaxial_modulus": biaxial_modulus,
        "sigma_hygro": sigma_hygro,
        "sigma_thermal": sigma_thermal,
        "sigma_index": sigma_index
    }

def render_3d_deformation(eps_h, eps_t, eps_net, n_frames=20):
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
        curv = (alpha * eps_net) * scale_factor * 0.04
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

NAVIGATION = ["Stress Analyser", "Sensitivity Analysis", "Inverse Exploration"]


def navigate(section=None):
    st.session_state["main_navigation"] = section


def render_style():
    st.markdown("""
    <style>
    .block-container { max-width: 1440px; padding-top: 2rem; padding-bottom: 3rem; }
    .st-key-home_link button { border: none; background: transparent; padding-left: 0; }
    .st-key-home_link button p { font-size: 1.15rem; font-weight: 700; }
    .st-key-main_navigation button {
        border: none; border-radius: 0; background: transparent;
        padding: .7rem .9rem; border-bottom: 2px solid transparent;
    }
    .st-key-main_navigation button[aria-pressed="true"] {
        color: #368575; border-bottom-color: #368575; background: transparent;
    }
    .st-key-main_navigation button:hover { background: rgba(54,133,117,.08); }
    .st-key-main_navigation button:focus-visible { outline: 2px solid #368575; }
    [data-testid="stBaseButton-primary"] { background-color: #368575; border-color: #368575; }
    [data-testid="stBaseButton-primary"]:hover { background-color: #2b6a5d; border-color: #2b6a5d; }
    .eyebrow { color: #368575; font-size: .78rem; font-weight: 650; letter-spacing: .16em; }
    .hero-title { font-size: clamp(2.25rem, 4vw, 3.8rem); line-height: 1.15;
        text-align: left; font-weight: 700; letter-spacing: -.035em;
        max-width: 640px; margin: 0 0 1.2rem; text-wrap: balance; }
    .hero-copy { font-size: 1rem; line-height: 1.75; max-width: 640px; opacity: .85; }
    [data-testid="stColumn"]:has(.st-key-home_moisture_path) {
        background: rgba(54,133,117,.045); border-radius: .75rem; }
    [data-testid="stColumn"]:has(.st-key-home_thermal_path) {
        background: rgba(200,125,85,.045); border-radius: .75rem; }
    .st-key-home_combined_index { background: rgba(54,133,117,.07); border-radius: .75rem; }
    .flow-step { font-size: .75rem; font-weight: 650; letter-spacing: .08em;
        color: #368575; margin: 0 0 .3rem; }
    .flow-down { text-align: center; color: #78988e; font-size: 1.5rem; margin: .25rem 0; }
    .flow-merge { height: 80px; }
    .flow-merge svg { display: block; width: 100%; height: 80px; }
    .flow-merge-mobile { display: none; }
    .exploration-icon { display: flex; align-items: center; justify-content: center;
        width: 48px; height: 48px; margin-bottom: .5rem; border-radius: 12px;
        background: rgba(54,133,117,.1); color: #368575; }
    .exploration-icon svg { width: 28px; height: 28px; }
    [data-testid="stMetricValue"] { font-size: 1.8rem; }
    @media (max-width: 760px) {
        .block-container { padding-top: 1rem; }
        .st-key-main_navigation button { padding: .6rem .5rem; }
        .flow-merge svg { display: none; }
        .flow-merge-mobile { display: block; text-align: center; padding: 1.25rem 0;
            color: #368575; }
    }
    </style>
    """, unsafe_allow_html=True)


def render_home_stress_flow():
    st.subheader("Moisture and Temperature Contributions to the Stress Index")
    st.write(
        "Moisture causes the laminate to swell, while temperature changes produce different degrees of "
        "expansion in the copper and laminate. Bonding restricts this relative movement, "
        "creating mechanical loading at their interface. The calculation below combines "
        "the two strain contributions through the laminate's effective in-plane stiffness."
    )
    moisture, thermal = st.columns(2, gap="large", border=True)
    with moisture.container(border=False, key="home_moisture_path"):
        st.markdown("#### Moisture contribution")
        st.markdown('<p class="flow-step">01 · MOISTURE UPTAKE</p>', unsafe_allow_html=True)
        st.write("Diffusion sets the moisture absorbed at exposure time t.")
        st.latex(r"C_t = \frac{M_t}{M_\infty}\,C_{\mathrm{sat}}, \qquad W(t) = C_t M_{\mathrm{H_2O}}")
        st.markdown('<p class="flow-down" aria-hidden="true">↓</p>', unsafe_allow_html=True)
        st.markdown('<p class="flow-step">02 · SWELLING STRAIN</p>', unsafe_allow_html=True)
        st.write("Absorbed water causes the laminate to expand.")
        st.latex(r"\varepsilon_h = \beta\,W(t)")
        st.markdown('<p class="flow-down" aria-hidden="true">↓</p>', unsafe_allow_html=True)
        st.markdown('<p class="flow-step">03 · STRESS-INDEX CONTRIBUTION</p>', unsafe_allow_html=True)
        st.latex(r"\sigma_h = \frac{E_{\mathrm{eff}}}{1-\nu_{\mathrm{eff}}}\,\varepsilon_h")
    with thermal.container(border=False, key="home_thermal_path"):
        st.markdown("#### Thermal contribution")
        st.markdown('<p class="flow-step">01 · TEMPERATURE CHANGE</p>', unsafe_allow_html=True)
        st.write("Heating or cooling changes each material's dimensions.")
        st.latex(r"\Delta T = T - T_{\mathrm{ref}}")
        st.markdown('<p class="flow-down" aria-hidden="true">↓</p>', unsafe_allow_html=True)
        st.markdown('<p class="flow-step">02 · EXPANSION MISMATCH STRAIN</p>', unsafe_allow_html=True)
        st.write("Different CTEs create a relative expansion strain.")
        st.latex(r"\varepsilon_T = (\alpha_{\mathrm{lam}}-\alpha_{\mathrm{Cu}})\,\Delta T")
        st.markdown('<p class="flow-down" aria-hidden="true">↓</p>', unsafe_allow_html=True)
        st.markdown('<p class="flow-step">03 · STRESS-INDEX CONTRIBUTION</p>', unsafe_allow_html=True)
        st.latex(r"\sigma_T = \frac{E_{\mathrm{eff}}}{1-\nu_{\mathrm{eff}}}\,\varepsilon_T")

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
        st.latex(
            r"\sigma_{\mathrm{index}} = \sigma_h + \sigma_T"
            r" = \frac{E_{\mathrm{eff}}}{1-\nu_{\mathrm{eff}}}"
            r"\left(\varepsilon_h + \varepsilon_T\right)"
        )
        st.write("A signed measure of the constrained in-plane response for comparing materials and exposure conditions.")
        st.caption("The two contributions can reinforce or offset each other, depending on the direction of the thermal mismatch.")

    with st.expander("Moisture uptake model & equation symbols"):
        st.write("For one-dimensional diffusion through a plane sheet of thickness h, "
                 "Fick's second law gives the fractional moisture uptake used in the swelling path.")
        st.latex(r"\frac{\partial C}{\partial t} = D\frac{\partial^2 C}{\partial z^2}")
        st.latex(
            r"\frac{M_t}{M_\infty} = 1-\frac{8}{\pi^2}"
            r"\sum_{n=0}^{\infty}\frac{1}{(2n+1)^2}"
            r"\exp\!\left[-\frac{(2n+1)^2\pi^2 D t}{h^2}\right]"
        )
        st.caption("The plane-sheet solution assumes an initially dry laminate, "
                   "constant diffusivity and constant surface moisture concentration.")
        st.markdown(r"""
| Symbol | Meaning | Units |
| :--- | :--- | :--- |
| $D$ | Moisture diffusivity | m²/s |
| $h$ | Laminate thickness | m |
| $t$ | Exposure time | s |
| $z$ | Position through the laminate thickness | m |
| $C$ | Local moisture concentration in the diffusion equation | mol/m³ |
| $M_t/M_\infty$ | Fractional moisture uptake relative to saturation | Dimensionless |
| $n$ | Series summation index, starting at zero | Dimensionless |
| $C_{\mathrm{sat}}$ | Saturation moisture concentration | mol/m³ |
| $C_t$ | Average moisture concentration at time $t$ | mol/m³ |
| $W(t)$ | Absorbed moisture concentration by mass at time $t$ | kg/m³ |
| $M_{\mathrm{H_2O}}$ | Molar mass of water, 0.018015 | kg/mol |
| $\beta$ | Coefficient of hygroscopic expansion | m³/kg |
| $\alpha_{\mathrm{lam}}$ | Laminate in-plane coefficient of thermal expansion | ppm/K |
| $\alpha_{\mathrm{Cu}}$ | Copper coefficient of thermal expansion | ppm/K |
| $T$ | Exposure temperature | °C or K |
| $T_{\mathrm{ref}}$ | Reference temperature, in the same units as $T$ | °C or K |
| $\Delta T$ | Temperature change from the reference | K or °C difference |
| $E_{\mathrm{eff}}$ | Effective modulus | MPa |
| $\nu_{\mathrm{eff}}$ | Effective Poisson's ratio | Dimensionless |
| $\varepsilon_h$ | Hygroscopic swelling strain | Dimensionless |
| $\varepsilon_T$ | Thermal mismatch strain | Dimensionless |
| $\sigma_h$ | Hygroscopic contribution to the stress index | MPa |
| $\sigma_T$ | Thermal contribution to the stress index | MPa |
| $\sigma_{\mathrm{index}}$ | Combined stress index | MPa |
""")
        st.caption("The factor 10⁻⁶ converts CTE values from ppm/K to K⁻¹.")
        st.caption("The stiffness factor Eeff/(1 − νeff) represents an effective linear-elastic, equibiaxial in-plane constraint.")


def render_exploration_icon(shapes):
    st.markdown(
        '<div class="exploration-icon" aria-hidden="true">'
        '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
        f'{shapes}</svg></div>',
        unsafe_allow_html=True,
    )


def render_home():
    text, illustration = st.columns([1.05, 1], gap="large", vertical_alignment="top")
    with text:
        st.markdown(
            '<h1 class="hero-title">Hygrothermal Effects on Copper-Laminate Interfaces</h1>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="hero-copy">In the field of electronics, printed circuit boards (PCBs) '
            'are essential components that connect various electronic parts. One of the most '
            'critical issues that can arise during the life of a PCB is delamination, a '
            'condition that compromises the integrity and reliability of the board.</p>'
            '<p class="hero-copy">Temperature and moisture are two contributing factors. '
            'During manufacturing, PCBs are exposed to various heating cycles, including '
            'soldering processes. Each of these cycles can create stresses that result in '
            'separation between layers. Moisture absorption can also lead to delamination, '
            'particularly if the PCB material is prone to moisture ingress. When moisture '
            'penetrates the PCB, it can cause the internal layers to expand. Under high '
            'temperatures, trapped moisture can turn into steam, creating pressure that '
            'leads to layer separation.</p>',
            unsafe_allow_html=True,
        )
        st.button("Open the stress analyser →", type="primary", on_click=navigate,
                  args=(NAVIGATION[0],), key="open_simulation")
    with illustration:
        st.iframe(Path(__file__).parent / "assets" / "laminate_animation.html", height="content")

    st.write("")
    render_home_stress_flow()

    st.write("")
    st.subheader("Laminate Stress Explorer")
    simulation, sensitivity, inverse = st.columns(3, gap="medium", border=True)
    with simulation:
        with st.container(border=False, height="stretch", key="home_simulation_card"):
            render_exploration_icon(
                '<path d="m12 3 10 5-10 5L2 8 12 3Z" />'
                '<path d="m2 12 10 5 10-5M2 16l10 5 10-5" />'
            )
            st.caption("01 · STRESS ANALYSER")
            st.markdown("#### Follow the physics")
            st.write("Input properties of your chosen laminate and its exposure conditions. Calculate moisture uptake, "
                     "swelling, and thermal mismatch to evaluate the resulting stress.")
        st.button("View section →", on_click=navigate, args=(NAVIGATION[0],), key="home_simulation")
    with sensitivity:
        with st.container(border=False, height="stretch", key="home_sensitivity_card"):
            render_exploration_icon(
                '<path d="M4 3v17h17M9 16V9M14 16V5M19 16v-4" />'
            )
            st.caption("02 · SENSITIVITY ANALYSIS")
            st.markdown("#### Understand what matters")
            st.write("Explore the influence of material properties "
                     "and exposure conditions on the relevant strains and stresses. Compare their main effects and interactions with Sobol analysis.")
        st.button("View section →", on_click=navigate, args=(NAVIGATION[1],), key="home_sensitivity")
    with inverse:
        with st.container(border=False, height="stretch", key="home_inverse_card"):
            render_exploration_icon(
                '<circle cx="11" cy="13" r="9" />'
                '<circle cx="11" cy="13" r="5" />'
                '<path d="m11 13 9-9M16 4h4v4" />'
            )
            st.caption("03 · INVERSE EXPLORATION")
            st.markdown("#### Design for target response")
            st.write("Design a laminate to meet specific performance requirements. Start with a desired response and explore "
                     "possible inputs.")
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

    st.title("Stress Analyser")
    st.caption("Choose your material and exposure conditions, then follow the calculation from uptake to stress.")
    with st.container(border=True):
        preset, detail = st.columns([1.25, 2])
        with preset:
            st.selectbox("Select Material Preset", list(PRESETS), key="preset_selector",
                         on_change=load_preset_values,
                         help="Loads the mechanical, moisture, and thickness fields below. All remain editable.")
        with detail:
            st.markdown("**Your simulation scenario**")
            st.caption("Start with a preset or enter your own properties. Results update as you change an input.")

        st.markdown("#### Exposure & geometry")
        temperature, duration, thickness, copper = st.columns(4)
        delta_T = temperature.number_input("Delta T (°C or K)", key="field_delta_T", step=5.0)
        t_hours = duration.number_input("Soak Time (hours)", min_value=0.0, key="field_t_hours", step=1.0)
        h_mm = thickness.number_input("Dielectric Thickness (mm)", min_value=0.001, key="field_h_mm", step=0.1)
        alpha_cu = copper.number_input("Copper Foil CTE (ppm/K)", key="field_alpha_cu", step=0.5)

        with st.expander("Material properties · mechanical & moisture", expanded=True):
            mechanical, hygrothermal = st.columns(2, gap="large")
            with mechanical:
                st.markdown("**Mechanical properties**")
                E = st.number_input("Young's Modulus E (MPa)", min_value=0.001, key="field_E", step=500.0)
                nu = st.number_input("Poisson's Ratio ν", min_value=-0.99, max_value=0.499,
                                     key="field_nu", step=0.01, format="%.3f")
                alpha_lam = st.number_input("In-Plane CTE α (ppm/K)", key="field_alpha_lam", step=0.5)
            with hygrothermal:
                st.markdown("**Hygrothermal properties**")
                D = st.number_input("Diffusion Coefficient D (m²/s)", min_value=0.0, key="field_D",
                                    step=1e-14, format="%.2e",
                                    help="Effective Fickian diffusivity at chamber temperature.")
                Csat = st.number_input("Saturation Concentration Csat (mol/m³)", min_value=0.0,
                                       key="field_Csat", step=10.0,
                                       help="Csat ≈ (WA% / 100) * (Density / 0.018015)")
                beta = st.number_input("In-Plane CHE β (m³/kg)", min_value=0.0,
                                       key="field_beta", step=1e-5, format="%.2e",
                                       help="In-plane coefficient of hygroscopic expansion.")

    st.session_state["simulation_inputs"] = {key: st.session_state[key] for key in defaults}
    return dict(D=D, Csat=Csat, h_mm=h_mm, t_hours=t_hours, beta=beta,
                alpha_lam=alpha_lam, alpha_cu=alpha_cu, delta_T=delta_T, E=E, nu=nu)


def render_simulation():
    p = render_simulation_inputs()
    D = p["D"]
    Csat = p["Csat"]
    h_mm = p["h_mm"]
    t_hours = p["t_hours"]
    beta = p["beta"]
    alpha_lam = p["alpha_lam"]
    alpha_cu = p["alpha_cu"]
    delta_T = p["delta_T"]
    E = p["E"]
    nu = p["nu"]

    h_meters = h_mm / 1000.0
    t_seconds = t_hours * 3600.0

    res = run_analytical_stress_pipeline(
        D=D, Csat=Csat, h=h_meters, t=t_seconds,
        beta=beta, alpha_lam=alpha_lam, alpha_cu=alpha_cu,
        delta_T=delta_T, E=E, nu=nu
    )

    st.caption(f"Active Scenario: **{st.session_state['preset_selector']}** | Exposure: **{t_hours:.1f} h** | Thickness: **{h_mm:.2f} mm**")

    # Top KPI Metric Cards
    col1, col2, col3, col4 = st.columns(4)
    col1, col2, col3, col4 = [column.container(border=True) for column in (col1, col2, col3, col4)]
    col1.metric("Moisture Saturation", f"{res['Mt_Minf'] * 100:.2f} %")
    col1.caption("Fractional equilibrium uptake (Mt/M∞) achieved under 1D Fickian diffusion at the current exposure duration.")
    col2.metric("Hygroscopic Strain (ε_h)", f"{res['eps_h']:.2e}")
    col2.caption("Moisture-driven in-plane swelling strain (ε_h = β·W). Positive indicates matrix dilatation from absorbed water.")
    col3.metric("Thermal Strain (ε_T)", f"{res['eps_t']:.2e}")
    col3.caption("Unconstrained differential thermal strain: (α_lam − α_Cu)ΔT. Negative indicates copper expands more during heating.")
    col4.metric("Net Stress Index (σ_index)", f"{res['sigma_index']:.2f} MPa")
    col4.caption("Constrained equibiaxial stress [E/(1−ν)]·(ε_h + ε_T). Positive denotes in-plane tension; negative denotes compression.")

    st.markdown("---")

    # Delamination Risk Assessment Card
    sigma_val = res["sigma_index"]
    screening_magnitude_mpa = 15.0  # Illustrative only; calibrate against interface tests.
    if abs(sigma_val) <= screening_magnitude_mpa:
        mismatch_direction = "positive" if sigma_val > 0 else "negative" if sigma_val < 0 else "zero"
        st.info(
            f"**Low mismatch index on an illustrative scale: {sigma_val:.2f} MPa**  \n"
            f"The net in-plane mismatch is {mismatch_direction}. This result alone cannot establish "
            "whether the copper-laminate interface is safe from delamination."
        )
    elif sigma_val < 0:
        st.warning(
            f"**Elevated negative mismatch index: {sigma_val:.2f} MPa**  \n"
            "The laminate's predicted free in-plane strain is less than copper's. "
            "Assess actual stresses, buckling, and interfacial adhesion for the geometry and exposure."
        )
    else:
        st.error(
            f"**Elevated positive mismatch index: {sigma_val:.2f} MPa**  \n"
            "The laminate's predicted free in-plane strain exceeds copper's. "
            "Assess local edge opening and shear against measured adhesion under the expected conditions."
        )
    st.caption(
        "The ±15 MPa screening band is illustrative, not a validated delamination limit. "
        "The index sign does not identify the actual stress in either material or the peel traction; "
        "qualify safety with geometry-specific "
        "buckling or fracture analysis and adhesion tests, including moisture and thermal exposure."
    )

    st.markdown("---")

    # Main Content Layout: Step-by-Step Breakdown vs. Visual Stress Contribution
    left_col, right_col = st.columns([1.1, 0.9])

    with left_col:
        st.subheader("Deterministic Pipeline Breakdown")
        st.caption("Intermediate physical values calculated sequentially across the 5 equations:")
    
        breakdown_df = pd.DataFrame([
            {"Step": "1. Fickian Mass Ratio (Mt/Minf)", "Value": f"{res['Mt_Minf']:.4f}", "Unit": "dimensionless"},
            {"Step": "2. Absorbed Moisture Density (W_t)", "Value": f"{res['W_t']:.4f}", "Unit": "kg/m³"},
            {"Step": "3. Hygroscopic Swelling Strain (ε_h)", "Value": f"{res['eps_h']:.6f}", "Unit": "dimensionless"},
            {"Step": "4. Thermal Mismatch Strain (ε_T)", "Value": f"{res['eps_t']:.6f}", "Unit": "dimensionless"},
            {"Step": "5. Net In-Plane Mismatch Strain", "Value": f"{res['eps_net']:.6f}", "Unit": "dimensionless"},
            {"Step": "6. Biaxial Modulus [E / (1 - ν)]", "Value": f"{res['biaxial_modulus']:.1f}", "Unit": "MPa"},
            {"Step": "7. Final Net Stress Index (σ_index)", "Value": f"{res['sigma_index']:.3f}", "Unit": "MPa"}
        ])
        st.table(breakdown_df)

    with right_col:
        st.subheader("Stress Component Decomposition")
        st.caption("Visualizing the competing effects of moisture swelling vs. thermal mismatch:")
    
        fig, ax = plt.subplots(figsize=(6, 3.8))
        components = ['Hygroscopic Contribution', 'Thermal Mismatch', 'Net Stress Index']
        values = [res['sigma_hygro'], res['sigma_thermal'], res['sigma_index']]
        net_color = '#4c78a8' if abs(sigma_val) <= screening_magnitude_mpa else '#e6a23c' if sigma_val < 0 else '#d9534f'
        colors = ['#d95f02', '#7570b3', net_color]
    
        ax.bar(components, values, color=colors, width=0.5)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.set_ylabel("Stress Contribution (MPa)")
        plt.xticks(rotation=15, ha='right')
        ax.grid(axis='y', linestyle=':', alpha=0.6)
    
        st.pyplot(fig)
        plt.close(fig)

    # -----------------------------------------------------------------------------
    # 3D Hygrothermal Deformation Visualizer
    # -----------------------------------------------------------------------------
    st.markdown("---")
    st.subheader("3D Bi-layer Deformation & Interfacial Behavior")
    st.caption(
        "Click **Play Deformation** to view a scaled illustration of the calculated mismatch strain. "
        "This schematic does not calculate physical warpage or interfacial tractions."
    )

    fig_3d = render_3d_deformation(
        eps_h=res["eps_h"],
        eps_t=res["eps_t"],
        eps_net=res["eps_net"]
    )

    st.plotly_chart(fig_3d, width="stretch")



def render_placeholder(section):
    st.title(section)
    with st.container(border=True):
        st.markdown("### Coming next")
        st.write("This section is ready for the workflow and controls you specify next.")
        st.button("Open the stress explorer →", on_click=navigate, args=(NAVIGATION[0],),
                  key="placeholder_simulation")


def main():
    st.set_page_config(page_title="Hygrothermal Effects on Copper-Laminate Interfaces", page_icon="⚡", layout="wide")
    render_style()
    brand, navigation = st.columns([1, 3], vertical_alignment="center")
    with brand:
        st.button("◈ Copper–Laminate Interfaces", key="home_link", on_click=navigate,
                  help="Return to the front page")
    with navigation:
        section = st.pills("Explore", NAVIGATION, key="main_navigation", required=True,
                           label_visibility="collapsed")
    st.divider()
    if section == NAVIGATION[0]:
        render_simulation()
    elif section == NAVIGATION[1]:
        snapshot = st.session_state.get("simulation_inputs", {})
        scenario = dict(PRESETS[next(iter(PRESETS))], delta_T=60.0, t_hours=24.0, alpha_cu=16.5)
        for key in scenario:
            scenario[key] = snapshot.get(f"field_{key}", scenario[key])
        render_sensitivity(scenario)
    elif section == NAVIGATION[2]:
        render_placeholder(section)
    else:
        render_home()


if __name__ == "__main__":
    main()
