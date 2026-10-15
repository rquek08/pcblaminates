"""Notebook-based Sobol studies, with engineering units used by app.py.

The notebook's A exposure study uses Arrhenius diffusivity; B material uses
effective diffusivity directly. Sampling blocks are A, B, AB_0, ..., AB_k-1.
No notebook cells, plotting side effects, or command-line prompts are executed.
"""
from io import BytesIO

import numpy as np
import pandas as pd
from scipy.stats import qmc

PARAMETERS = {
    "alpha_lam": ("Laminate CTE", "ppm/K", 8.0, 23.0, "uniform"),
    "beta": ("Moisture expansion (CHE)", "m³/kg", 1.15e-4, 3.8e-4, "uniform"),
    "E": ("Young's modulus", "MPa", 7500.0, 28000.0, "uniform"),
    "nu": ("Poisson's ratio", "dimensionless", 0.175, 0.32, "uniform"),
    "Csat": ("Saturation moisture concentration", "mol/m³", 92.5, 366.0, "uniform"),
    "D": ("Effective diffusivity", "m²/s", 2.15e-14, 2.0e-12, "log-uniform"),
    "delta_T": ("Temperature change ΔT", "K", 10.0, 100.0, "uniform"),
    "t_hours": ("Exposure time", "h", 24.0, 8760.0, "log-uniform"),
    "h_mm": ("Laminate thickness", "mm", 0.2, 3.2, "uniform"),
}
EXPOSURE = ["delta_T", "t_hours", "h_mm"]
MATERIAL = [key for key in PARAMETERS if key not in EXPOSURE]
STUDIES = ["A · Exposure conditions", "B · Material properties", "Combined · All parameters"]
FIXED_META = {key: value[:2] for key, value in PARAMETERS.items()}
FIXED_META.update(alpha_cu=("Copper CTE", "ppm/K"))
OUTPUTS = {
    "sigma_abs": ("Stress-index magnitude", "MPa"),
    "sigma_index": ("Signed stress index", "MPa"),
    "sigma_hygro": ("Moisture contribution", "MPa"),
    "sigma_thermal": ("Thermal contribution", "MPa"),
    "Mt_Minf": ("Fractional moisture uptake", "dimensionless"),
    "W_t": ("Absorbed water concentration", "kg/m³"),
    "eps_h": ("Hygroscopic strain", "dimensionless"),
}


def reference_baseline(study):
    return dict(E=24056.0, nu=0.182, alpha_lam=13.5,
                alpha_cu=17.0 if study == STUDIES[0] else 16.5,
                D=2.1e-12, Csat=222.04, beta=1.4e-4, h_mm=1.6,
                t_hours=476500.0 / 3600, delta_T=50.0)


# Preserve compatibility with existing analysis scripts.
notebook_baseline = reference_baseline


def default_ranges(keys):
    return pd.DataFrame([
        dict(Parameter=key, Label=PARAMETERS[key][0], Unit=PARAMETERS[key][1],
             Lower=PARAMETERS[key][2], Upper=PARAMETERS[key][3],
             Distribution=PARAMETERS[key][4]) for key in keys
    ])


def material_ranges(keys, baseline, percent=10.0, diffusion_factor=2.0):
    """Illustrative bounds around one material, not measured tolerances."""
    if not 0 < percent < 100 or diffusion_factor <= 1:
        raise ValueError("Use a percentage between 0 and 100 and a diffusivity factor above 1.")
    table = default_ranges(keys)
    for i, key in enumerate(keys):
        nominal = float(baseline[key])
        if nominal == 0:
            raise ValueError(f"{PARAMETERS[key][0]} is zero. Enter explicit bounds using broad ranges, "
                             "or give this input a nonzero nominal value.")
        if key == "D":
            lo, hi = nominal / diffusion_factor, nominal * diffusion_factor
        else:
            width = abs(nominal) * percent / 100
            lo, hi = nominal - width, nominal + width
        if key == "nu":
            lo, hi = max(lo, -0.999), min(hi, 0.499)
        table.loc[i, ["Lower", "Upper", "Distribution"]] = [lo, hi, "log-uniform" if key == "D" else "uniform"]
    return table


def validate_baseline(p):
    for key in FIXED_META:
        x = np.asarray(p[key], dtype=float)
        if not np.all(np.isfinite(x)):
            raise ValueError(f"{FIXED_META[key][0]} must be finite.")
        if key in ("h_mm", "E") and np.any(x <= 0):
            raise ValueError(f"{FIXED_META[key][0]} must be greater than zero.")
        if key in ("D", "Csat", "beta", "t_hours") and np.any(x < 0):
            raise ValueError(f"{FIXED_META[key][0]} cannot be negative.")
        if key == "nu" and np.any((x <= -1) | (x >= 0.5)):
            raise ValueError("Poisson's ratio must be between −1 and 0.5.")


def evaluate_samples(p, diffusion):
    """Vectorized dry plane-sheet model; E in MPa, alpha in ppm/K.

    The 60-term notebook series is used at Fo >= .005. Its short-time
    asymptote below that avoids a truncation offset near t=0 or D=0.
    """
    validate_baseline(p)
    if diffusion["mode"] == "Arrhenius":
        d0, ea, tref = (float(diffusion[key]) for key in ("D0", "Ea", "T_ref_C"))
        if not np.all(np.isfinite([d0, ea, tref])) or d0 <= 0 or ea < 0:
            raise ValueError("Arrhenius requires a positive D₀ and nonnegative activation energy.")
        temperature = tref + 273.15 + np.asarray(p["delta_T"])
        if np.any(temperature <= 0):
            raise ValueError("Reference temperature + ΔT must be above absolute zero.")
        D = d0 * np.exp(-ea / (8.314 * temperature))
    elif diffusion["mode"] == "Fixed D":
        D = np.asarray(p["D"])
    else:
        raise ValueError("Select a supported diffusion model.")
    d, h, t, csat, beta, alpha, copper, dt, E, nu = np.broadcast_arrays(
        D, np.asarray(p["h_mm"]) / 1000, np.asarray(p["t_hours"]) * 3600,
        p["Csat"], p["beta"], p["alpha_lam"], p["alpha_cu"], p["delta_T"], p["E"], p["nu"])
    shape = d.shape
    fo = (d * t / h**2).ravel()
    uptake = np.zeros_like(fo)
    short = (fo > 0) & (fo < 0.005)
    uptake[short] = 4 * np.sqrt(fo[short] / np.pi)
    indices = np.flatnonzero(fo >= 0.005)
    odd = 2 * np.arange(60) + 1
    for start in range(0, len(indices), 4096):
        idx = indices[start:start + 4096]
        terms = np.exp(-fo[idx, None] * np.pi**2 * odd[None, :]**2) / odd[None, :]**2
        uptake[idx] = 1 - 8 / np.pi**2 * terms.sum(axis=1)
    uptake = np.clip(uptake.reshape(shape), 0, 1)
    water = uptake * csat * 0.018015
    eh, et = beta * water, (alpha - copper) * 1e-6 * dt
    modulus = E / (1 - nu)
    sigma_h, sigma_t = modulus * eh, modulus * et
    sigma = sigma_h + sigma_t
    return dict(D_effective=d, Fo=fo.reshape(shape), Mt_Minf=uptake, W_t=water,
                eps_h=eh, eps_t=et, sigma_hygro=sigma_h, sigma_thermal=sigma_t,
                sigma_index=sigma, sigma_abs=np.abs(sigma))


def scale_samples(unit, ranges):
    """Validate physical bounds and transform independent unit coordinates."""
    keys = [row["Parameter"] for row in ranges]
    if not keys or len(set(keys)) != len(keys) or any(key not in PARAMETERS for key in keys):
        raise ValueError("Choose at least one distinct supported parameter.")
    scaled = np.empty_like(unit)
    for i, row in enumerate(ranges):
        lo, hi = float(row["Lower"]), float(row["Upper"])
        if not np.all(np.isfinite([lo, hi])) or hi <= lo:
            raise ValueError(f"{PARAMETERS[keys[i]][0]}: upper bound must exceed a finite lower bound.")
        if keys[i] in ("E", "h_mm") and lo <= 0:
            raise ValueError(f"{PARAMETERS[keys[i]][0]} bounds must be positive.")
        if keys[i] in ("D", "Csat", "beta", "t_hours") and lo < 0:
            raise ValueError(f"{PARAMETERS[keys[i]][0]} bounds cannot be negative.")
        if keys[i] == "nu" and (lo <= -1 or hi >= 0.5):
            raise ValueError("Poisson's ratio bounds must lie strictly between −1 and 0.5.")
        for j in range(i, unit.shape[1], len(keys)):
            if row["Distribution"] == "log-uniform":
                if lo <= 0:
                    raise ValueError("Log-uniform bounds must be positive.")
                scaled[:, j] = np.exp(np.log(lo) + unit[:, j] * (np.log(hi) - np.log(lo)))
            elif row["Distribution"] == "uniform":
                scaled[:, j] = lo + unit[:, j] * (hi - lo)
            else:
                raise ValueError("Choose uniform or log-uniform distributions.")
    return scaled


def saltelli_samples(ranges, N, seed=12345):
    if not isinstance(N, int) or N < 2 or N & (N - 1):
        raise ValueError("Base sample size must be a power of two, at least 2.")
    if not ranges:
        raise ValueError("Choose at least one distinct supported parameter.")
    keys = [row["Parameter"] for row in ranges]
    unit = qmc.Sobol(d=2 * len(keys), scramble=True, seed=seed).random_base2(int(np.log2(N)))
    scaled = scale_samples(unit, ranges)
    A, B = scaled[:, :len(keys)], scaled[:, len(keys):]
    blocks = [A, B]
    for i in range(len(keys)):
        AB = A.copy()
        AB[:, i] = B[:, i]
        blocks.append(AB)
    return np.vstack(blocks)


def estimate_indices(y, N, k, n_boot=500, seed=7):
    """Centered Saltelli S1 / Jansen ST; paired percentile bootstrap CIs."""
    y = np.asarray(y, dtype=float)
    if y.shape != (N * (k + 2),) or not np.all(np.isfinite(y)):
        raise ValueError("Model output must be finite and match the sampling blocks.")
    base = y[:2 * N]
    scale = float(np.std(base))
    if scale <= np.finfo(float).eps * max(1.0, float(np.abs(base).max())):
        raise ValueError("The selected output has no measurable variance for this study. "
                         "Choose a different output, parameter set, or range.")
    y = (y - base.mean()) / scale
    A, B = y[:N], y[N:2 * N]
    AB = np.column_stack([y[(i + 2) * N:(i + 3) * N] for i in range(k)])

    def calculate(a, b, ab):
        variance = np.var(np.r_[a, b])
        if variance <= np.finfo(float).eps:
            return np.full((2, k), np.nan)
        return np.array([np.mean(b[:, None] * (ab - a[:, None]), axis=0) / variance,
                         .5 * np.mean((a[:, None] - ab)**2, axis=0) / variance])

    s1, st = calculate(A, B, AB)
    result = dict(S1=s1, ST=st)
    if n_boot:
        rng = np.random.default_rng(seed)
        estimates = []
        for _ in range(n_boot):
            idx = rng.integers(0, N, N)
            estimate = calculate(A[idx], B[idx], AB[idx])
            if np.all(np.isfinite(estimate)):
                estimates.append(estimate)
        if len(estimates) < 2:
            raise ValueError("Too few variable bootstrap samples; increase the base sample size.")
        low, high = np.percentile(estimates, [2.5, 97.5], axis=0)
        result.update(S1_low=low[0], S1_high=high[0], ST_low=low[1], ST_high=high[1])
    return result


def run_analysis(p, ranges, diffusion, target="sigma_abs", N=4096, seed=12345,
                 method="Sobol", trajectories=40):
    if target not in OUTPUTS:
        raise ValueError("Choose a supported sensitivity output.")
    if diffusion["mode"] == "Arrhenius" and any(row["Parameter"] == "D" for row in ranges):
        raise ValueError("Effective D is calculated from temperature in the Arrhenius study; "
                         "use fixed-D mode to vary diffusivity independently.")
    if method == "Morris":
        return run_morris(p, ranges, diffusion, target, trajectories, seed)
    if method != "Sobol":
        raise ValueError("Choose Morris or Sobol analysis.")
    X = saltelli_samples(ranges, N, seed)
    keys = [row["Parameter"] for row in ranges]
    varied = dict(p, **{key: X[:, i] for i, key in enumerate(keys)})
    out = evaluate_samples(varied, diffusion)
    result = estimate_indices(out[target], N, len(keys), seed=seed + 1)
    table = pd.DataFrame({"Parameter": keys, "Label": [PARAMETERS[key][0] for key in keys], **result})
    blocks = np.asarray(out[target]).reshape(len(keys) + 2, N)
    half = estimate_indices(blocks[:, :N // 2].ravel(), N // 2, len(keys), n_boot=0)
    table["ST at N/2"] = half["ST"]
    table["ST change"] = np.abs(table["ST"] - table["ST at N/2"])
    table["ST - S1"] = table["ST"] - table["S1"]
    summary = output_summary({key: values[:2 * N] for key, values in out.items()})
    samples = pd.DataFrame({key: X[:, i] for i, key in enumerate(keys)})
    for key, values in out.items():
        samples[key] = values
    samples["Block"] = np.repeat(["A", "B"] + [f"AB_{key}" for key in keys], N)
    return dict(indices=table, summary=summary, samples=samples)


def output_summary(out):
    return pd.DataFrame([
        {"Quantity": label, "Unit": unit, "Minimum": float(out[key].min()),
         "Mean": float(out[key].mean()), "Maximum": float(out[key].max()),
         "Std deviation": float(out[key].std())}
        for key, label, unit in [
            ("D_effective", "Effective diffusivity", "m²/s"), ("Fo", "Fourier number", "dimensionless"),
            ("Mt_Minf", "Fractional moisture uptake", "dimensionless"), ("W_t", "Absorbed moisture density", "kg/m³"),
            ("eps_h", "Hygroscopic strain", "dimensionless"), ("eps_t", "Thermal mismatch strain", "dimensionless"),
            ("sigma_abs", "Stress-index magnitude", "MPa"), ("sigma_index", "Signed stress index", "MPa")]
    ])


def run_morris(p, ranges, diffusion, target, trajectories=40, seed=12345):
    """Six-level Morris design; elementary effects in normalized input space."""
    if not isinstance(trajectories, int) or trajectories < 2:
        raise ValueError("Use at least two Morris trajectories.")
    keys = [row["Parameter"] for row in ranges]
    if not keys:
        raise ValueError("Choose at least one distinct supported parameter.")
    rng = np.random.default_rng(seed)
    k, delta = len(keys), 0.6
    paths, orders, signs = [], [], []
    for _ in range(trajectories):
        low = rng.choice([0.0, 0.2, 0.4], size=k)
        sign = rng.choice([-1, 1], size=k)
        x = np.where(sign < 0, low + delta, low)
        order = rng.permutation(k)
        path = [x.copy()]
        for j in order:
            x = x.copy()
            x[j] += sign[j] * delta
            path.append(x)
        paths.extend(path)
        orders.append(order)
        signs.append(sign)
    X = scale_samples(np.clip(np.asarray(paths), 0, 1), ranges)
    out = evaluate_samples(dict(p, **{key: X[:, i] for i, key in enumerate(keys)}), diffusion)
    response = np.asarray(out[target])
    if not np.all(np.isfinite(response)):
        raise ValueError("Model output must be finite.")
    y = response.reshape(trajectories, k + 1)
    effects = np.empty((trajectories, k))
    for i in range(trajectories):
        for step, j in enumerate(orders[i]):
            effects[i, j] = (y[i, step + 1] - y[i, step]) / (signs[i][j] * delta)
    table = pd.DataFrame(dict(Parameter=keys, Label=[PARAMETERS[key][0] for key in keys],
                              mu=effects.mean(axis=0), mu_star=np.abs(effects).mean(axis=0),
                              sigma=effects.std(axis=0, ddof=1)))
    samples = pd.DataFrame({key: X[:, i] for i, key in enumerate(keys)})
    for key, values in out.items():
        samples[key] = values
    samples["Trajectory"] = np.repeat(np.arange(1, trajectories + 1), k + 1)
    samples["Step"] = np.tile(np.arange(k + 1), trajectories)
    return dict(indices=table, summary=output_summary(out), samples=samples)


def excel_report(result, config):
    """Build an in-memory workbook; no files written to the user's project."""
    setup = [{"Setting": key, "Value": str(value)} for key, value in config.items()
             if key not in ("baseline", "ranges", "diffusion")]
    setup += [{"Setting": key, "Value": str(value)} for key, value in config["diffusion"].items()]
    fixed = [{"Parameter": key, "Label": FIXED_META[key][0], "Value": value,
              "Unit": FIXED_META[key][1]} for key, value in config["baseline"].items()
             if key not in [row["Parameter"] for row in config["ranges"]]
             and not (key == "D" and config["diffusion"]["mode"] == "Arrhenius")]
    morris = config.get("method", "Sobol") == "Morris"
    notes = pd.DataFrame({"How to read": ([
        "Morris mu_star: mean absolute elementary effect; sigma: variation from nonlinearity and/or interactions.",
        "Effects use normalized input steps on a six-level grid (delta=0.6), in the selected output's units, not variance shares.",
        "Summary statistics describe trajectory samples, not independent distribution draws or uncertainty bounds.",
    ] if morris else [
        "S1: main contribution to output variance. ST: main effect plus every interaction involving the input.",
        "ST - S1: interaction contribution involving this input. Total-order indices overlap; do not add them as independent shares.",
        "Intervals are paired bootstrap 95% percentiles (500 resamples), not material or model validation uncertainty.",
        "Output summary uses A and B base samples only. Samples includes all Saltelli cross-sampling blocks.",
    ]) + [
        "Inputs use app engineering units. E is MPa, CTE ppm/K, time h, thickness mm; all samples start dry.",
        "Arrhenius: D = D0 exp[-Ea/(8.314 T)], T = T_ref_C + 273.15 + delta_T.",
        "Independent input distributions and bounds determine the rankings. No interface failure is predicted.",
    ]})
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        pd.DataFrame(setup).to_excel(writer, sheet_name="Setup", index=False)
        pd.DataFrame([dict(Parameter=key, Label=FIXED_META[key][0], Value=value, Unit=FIXED_META[key][1])
                      for key, value in config["baseline"].items()]).to_excel(
            writer, sheet_name="Nominal inputs", index=False)
        pd.DataFrame(config["ranges"]).to_excel(writer, sheet_name="Variable inputs", index=False)
        pd.DataFrame(fixed).to_excel(writer, sheet_name="Fixed inputs", index=False)
        result["indices"].sort_values("mu_star" if morris else "ST", ascending=False).to_excel(
            writer, sheet_name="Morris effects" if morris else "Sobol S1 ST", index=False)
        result["summary"].to_excel(writer, sheet_name="Output summary", index=False)
        result["samples"].to_excel(writer, sheet_name="Samples", index=False)
        notes.to_excel(writer, sheet_name="How to read", index=False)
        for sheet in writer.book.worksheets:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
    return output.getvalue()
