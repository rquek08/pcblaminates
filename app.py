import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# Configuration & Constants
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Bonded by Stress - Interfacial Stress Analyzer",
    page_icon="⚡",
    layout="wide"
)

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

# -----------------------------------------------------------------------------
# Session State: Preset Syncing
# -----------------------------------------------------------------------------
def load_preset_values():
    choice = st.session_state["preset_selector"]
    vals = PRESETS[choice]
    for k, v in vals.items():
        st.session_state[f"field_{k}"] = v

if "initialized" not in st.session_state:
    st.session_state["initialized"] = True
    default_vals = PRESETS["Isola 370HR (High-Tg Core)"]
    for k, v in default_vals.items():
        st.session_state[f"field_{k}"] = v

# -----------------------------------------------------------------------------
# Sidebar: User Inputs
# -----------------------------------------------------------------------------
st.sidebar.title("Parameters")
st.sidebar.selectbox(
    "Select Material Preset",
    options=list(PRESETS.keys()),
    index=0,
    key="preset_selector",
    on_change=load_preset_values,
    help="Selecting a preset automatically updates all mechanical and hygrothermal fields below."
)

st.sidebar.markdown("---")
st.sidebar.subheader("1. Environmental & Geometry")
delta_T = st.sidebar.number_input("Delta T (°C or K)", value=60.0, step=5.0)
t_hours = st.sidebar.number_input("Soak Time (hours)", value=24.0, step=1.0)
h_mm = st.sidebar.number_input(
    "Dielectric Thickness (mm)",
    key="field_h_mm",
    step=0.1,
    help="1.6 mm is IPC-4101 standard core thickness."
)
alpha_cu = st.sidebar.number_input("Copper Foil CTE (ppm/K)", value=16.5, step=0.5)

st.sidebar.markdown("---")
st.sidebar.subheader("2. Mechanical Properties")
E = st.sidebar.number_input("Young's Modulus E (MPa)", key="field_E", step=500.0)
nu = st.sidebar.number_input("Poisson's Ratio ν", key="field_nu", step=0.01, format="%.3f")
alpha_lam = st.sidebar.number_input("In-Plane CTE α (ppm/K)", key="field_alpha_lam", step=0.5)

st.sidebar.markdown("---")
st.sidebar.subheader("3. Hygrothermal Properties")
D = st.sidebar.number_input(
    "Diffusion Coefficient D (m²/s)",
    key="field_D",
    step=1e-14,
    format="%.2e",
    help="Effective Fickian diffusivity at chamber temperature."
)
Csat = st.sidebar.number_input(
    "Saturation Concentration Csat (mol/m³)",
    key="field_Csat",
    step=10.0,
    help="Csat ≈ (WA% / 100) * (Density / 0.018015)"
)
beta = st.sidebar.number_input(
    "In-Plane CHE β (m³/kg)",
    key="field_beta",
    step=1e-5,
    format="%.2e",
    help="In-plane coefficient of hygroscopic expansion."
)

# -----------------------------------------------------------------------------
# Calculation & Main View
# -----------------------------------------------------------------------------
h_meters = h_mm / 1000.0
t_seconds = t_hours * 3600.0

res = run_analytical_stress_pipeline(
    D=D, Csat=Csat, h=h_meters, t=t_seconds,
    beta=beta, alpha_lam=alpha_lam, alpha_cu=alpha_cu,
    delta_T=delta_T, E=E, nu=nu
)

st.title("Laminate-Copper Interfacial Stress Calculator")
st.caption(f"Active Scenario: **{st.session_state['preset_selector']}** | Exposure: **{t_hours:.1f} h** | Thickness: **{h_mm:.2f} mm**")

# Top KPI Metric Cards
col1, col2, col3, col4 = st.columns(4)
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

# -----------------------------------------------------------------------------
# 3D Hygrothermal Deformation Visualizer
# -----------------------------------------------------------------------------
st.markdown("---")
st.subheader("3D Bi-layer Deformation & Interfacial Behavior")
st.caption(
    "Click **Play Deformation** below to watch the simulated strain accumulation and warpage "
    "between the copper foil (top) and dielectric substrate (bottom)."
)

fig_3d = render_3d_deformation(
    eps_h=res["eps_h"],
    eps_t=res["eps_t"],
    eps_net=res["eps_net"]
)

st.plotly_chart(fig_3d, use_container_width=True)