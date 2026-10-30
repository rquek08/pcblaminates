"""Filter material libraries and rank every eligible record in bounded batches."""
import numpy as np
import pandas as pd

from sensitivity_analysis import MATERIAL, evaluate_samples


APPLICATION_DEFAULTS = dict(delta_T=60.0, t_hours=24.0, h_mm=1.6, alpha_cu=16.5)
OBJECTIVES = {
    "sigma_abs": ("Smallest stress-index magnitude", "MPa"),
    "eps_h": ("Lowest hygroscopic swelling", "dimensionless"),
    "W_t": ("Lowest absorbed water concentration", "kg/m³"),
}
RESULT_OUTPUTS = ["sigma_abs", "sigma_index", "sigma_hygro", "sigma_thermal", "Mt_Minf", "eps_h", "eps_t", "W_t"]


def records_from_csv(frame):
    """Read engineering-unit inputs; preserve invalid numeric rows for reporting."""
    required = ["name", "condition", *MATERIAL]
    missing = [key for key in required if key not in frame]
    if missing:
        raise ValueError("Missing CSV columns: " + ", ".join(missing))
    if frame.empty:
        raise ValueError("The uploaded material library is empty.")
    result = frame.copy()
    for key in ("name", "condition"):
        if result[key].isna().any() or result[key].astype(str).str.strip().eq("").any():
            raise ValueError("Every library row needs a material name and conditioning environment.")
        result[key] = result[key].astype(str).str.strip()
    for key, fallback in (("material_class", "Unspecified"), ("supplier", ""), ("grade", "")):
        if key not in result:
            result[key] = fallback
        result[key] = result[key].fillna(fallback).astype(str).str.strip()
    result["material_class"] = result["material_class"].replace("", "Unspecified")
    result[MATERIAL] = result[MATERIAL].apply(pd.to_numeric, errors="coerce")
    return result[["name", "condition", "material_class", "supplier", "grade", *MATERIAL]].reset_index(drop=True)


def prepare_library(records):
    """Accept bundled preset records or a flat uploaded library."""
    if isinstance(records, pd.DataFrame):
        return records_from_csv(records)
    rows = [dict(name=r.get("name"), condition=r.get("condition"),
                 material_class=r.get("material_class", "Unspecified"),
                 supplier=r.get("supplier", ""), grade=r.get("grade", ""),
                 **{key: r.get("properties", {}).get(key, np.nan) for key in MATERIAL}) for r in records]
    return records_from_csv(pd.DataFrame(rows, columns=["name", "condition", "material_class", "supplier", "grade", *MATERIAL]))


def validate_application(application):
    for key in APPLICATION_DEFAULTS:
        if key not in application or not np.isfinite(application[key]):
            raise ValueError("Enter finite temperature change, soak time, thickness and copper CTE values.")
    if application["h_mm"] <= 0:
        raise ValueError("Laminate thickness must be greater than zero.")
    if application["t_hours"] < 0:
        raise ValueError("Soak time cannot be negative.")


def recommend_materials(library, condition, application, objective="sigma_abs", material_class=None,
                        property_bounds=None, output_limits=None, shortlist_size=10, batch_size=4096):
    """Return the exact best K qualifying records, evaluating each eligible row once.

    Library loading is separate so callers can cache the prepared numeric table.
    Model arrays are bounded by batch_size; only the best K outputs are retained.
    No sensitivity rankings, surrogate predictions or three-property screen are used.
    """
    validate_application(application)
    if objective not in OBJECTIVES:
        raise ValueError("Choose a supported ranking objective.")
    if not isinstance(shortlist_size, int) or not 1 <= shortlist_size <= 10:
        raise ValueError("Show between one and ten materials.")
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("Batch size must be a positive integer.")
    property_bounds = property_bounds or {}
    output_limits = output_limits or {}
    for key, (lo, hi) in property_bounds.items():
        if key not in MATERIAL or not np.all(np.isfinite([lo, hi])) or hi < lo:
            raise ValueError("Property bounds must be finite, ordered ranges for supported material properties.")
    for key, limit in output_limits.items():
        if key not in OBJECTIVES or not np.isfinite(limit) or limit < 0:
            raise ValueError("Output requirements must be nonnegative, finite limits for stress, swelling or water concentration.")
    frame = library if isinstance(library, pd.DataFrame) else prepare_library(library)
    selected = frame["condition"].eq(condition)
    if material_class is not None:
        selected &= frame["material_class"].eq(material_class)
    candidates = frame.loc[selected].copy()
    candidates["_order"] = np.flatnonzero(selected.to_numpy())
    valid = np.isfinite(candidates[MATERIAL].to_numpy(dtype=float)).all(axis=1)
    valid &= (candidates["E"] > 0).to_numpy()
    valid &= (candidates[["D", "Csat", "beta"]] >= 0).all(axis=1).to_numpy()
    valid &= ((candidates["nu"] > -1) & (candidates["nu"] < .5)).to_numpy()
    inside = np.ones(len(candidates), dtype=bool)
    for key, (lo, hi) in property_bounds.items():
        inside &= candidates[key].between(lo, hi).to_numpy()
    summary = dict(library_total=len(frame), total=len(candidates), invalid=int((~valid).sum()),
                   property_excluded=int((valid & ~inside).sum()), eligible=int((valid & inside).sum()),
                   evaluated=0, numerical_invalid=0, requirements_excluded=0, qualifying=0, batches=0)
    candidates = candidates.loc[valid & inside]
    best = pd.DataFrame()
    for start in range(0, len(candidates), batch_size):
        batch = candidates.iloc[start:start + batch_size].copy()
        params = {key: batch[key].to_numpy() for key in MATERIAL}
        params.update({key: float(application[key]) for key in APPLICATION_DEFAULTS})
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            output = evaluate_samples(params, {"mode": "Fixed D"})
        summary["evaluated"] += len(batch)
        summary["batches"] += 1
        for key in RESULT_OUTPUTS:
            batch[key] = output[key]
        finite = np.isfinite(batch[RESULT_OUTPUTS].to_numpy()).all(axis=1)
        summary["numerical_invalid"] += int((~finite).sum())
        meets = np.ones(len(batch), dtype=bool)
        for key, limit in output_limits.items():
            meets &= batch[key].le(limit).to_numpy()
        summary["requirements_excluded"] += int((finite & ~meets).sum())
        batch = batch.loc[finite & meets].copy()
        summary["qualifying"] += len(batch)
        if batch.empty:
            continue
        best = pd.concat([best, batch], ignore_index=True)
        best = best.sort_values([objective, "_order"], kind="stable").head(shortlist_size)
    if not best.empty:
        best = best.drop(columns="_order").reset_index(drop=True)
        best.insert(0, "rank", np.arange(1, len(best) + 1))
        best["meets_requirements"] = True
        best["ranking_objective"] = objective
        for key in APPLICATION_DEFAULTS:
            best[key] = application[key]
    summary["candidates"] = best
    return summary
