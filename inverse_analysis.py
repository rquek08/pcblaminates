"""Screen a library using saved sensitivity samples; verify only the shortlist."""
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from sensitivity_analysis import EXPOSURE, MATERIAL, evaluate_samples, validate_baseline


def recommendation_parameters(saved):
    config = saved["config"]
    if config["target"] not in ("sigma_abs", "sigma_index"):
        raise ValueError("Analyse stress-index magnitude or signed stress index before recommending materials.")
    if config["diffusion"]["mode"] != "Fixed D" or any(
            row["Parameter"] in EXPOSURE for row in config["ranges"]):
        raise ValueError("Run a material-properties study with fixed exposure inputs to recommend materials.")
    table = saved["result"]["indices"]
    table = table[table["Parameter"].isin(MATERIAL)].copy()
    if len(table) < 3:
        raise ValueError("Vary at least three material parameters in Sensitivity Analysis.")
    if not np.isfinite(table["ST"].to_numpy(dtype=float)).all():
        raise ValueError("Sensitivity rankings must be finite. Run a new sensitivity study.")
    table = table.sort_values("ST", ascending=False, kind="stable").head(3)
    weights = np.maximum(table["ST"].to_numpy(dtype=float), 0)
    if weights.sum() <= 0:
        raise ValueError("The study has no positive total-effect estimates to guide screening.")
    table["Screening weight"] = weights / weights.sum()
    return table


def records_from_csv(frame):
    required = ["name", "condition", *MATERIAL]
    missing = [key for key in required if key not in frame]
    if missing:
        raise ValueError("Missing CSV columns: " + ", ".join(missing))
    records = []
    for row in frame.to_dict("records"):
        if pd.isna(row["name"]) or pd.isna(row["condition"]):
            raise ValueError("Every CSV row needs a material name and conditioning environment.")
        records.append(dict(name=str(row["name"]), condition=str(row["condition"]),
                            supplier=str(row.get("supplier", "")), grade=str(row.get("grade", row["name"])),
                            properties={key: row[key] for key in MATERIAL}))
    if not records:
        raise ValueError("The uploaded material library is empty.")
    return records


def _normalise(values, row):
    lo, hi = float(row["Lower"]), float(row["Upper"])
    if row["Distribution"] == "log-uniform":
        return (np.log(values) - np.log(lo)) / (np.log(hi) - np.log(lo))
    return (values - lo) / (hi - lo)


def recommend_materials(records, condition, saved, limit=10.0, shortlist_size=10):
    """Rank by weighted proximity to low-|stress| samples, then verify up to ten.

    Screening makes no model calls. Projection onto three properties loses
    omitted effects and interactions, so this is not a global-optimum guarantee.
    """
    if not np.isfinite(limit) or limit < 0:
        raise ValueError("The stress-index limit must be finite and nonnegative.")
    if not isinstance(shortlist_size, int) or not 1 <= shortlist_size <= 10:
        raise ValueError("The shortlist must contain between one and ten materials.")
    top = recommendation_parameters(saved)
    keys = top["Parameter"].tolist()
    config = saved["config"]
    ranges = {row["Parameter"]: row for row in config["ranges"]}
    samples = saved["result"]["samples"]
    # A/B are independent base blocks; cross-sampling blocks reuse their values.
    base = samples[samples["Block"].isin(["A", "B"])].copy()
    valid = np.isfinite(base[[*keys, "sigma_abs"]].to_numpy(dtype=float)).all(axis=1)
    for key in keys:
        valid &= base[key].between(ranges[key]["Lower"], ranges[key]["Upper"])
    base = base[valid]
    if base.empty:
        raise ValueError("No valid independent sensitivity samples are available for screening.")
    # A spread of the best decile allows more than one favourable combination.
    good = base.nsmallest(max(1, int(np.ceil(len(base) * .1))), "sigma_abs")
    if len(good) > 128:
        good = good.iloc[np.linspace(0, len(good) - 1, 128).astype(int)]
    scale = np.sqrt(top["Screening weight"].to_numpy())
    targets = np.column_stack([_normalise(good[key].to_numpy(), ranges[key]) for key in keys]) * scale
    tree = cKDTree(targets)
    baseline = dict(config["baseline"])
    validate_baseline(baseline)
    matches = [record for record in records if record["condition"] == condition]
    if not matches:
        raise ValueError("No library materials are available for this conditioning environment.")
    candidates = pd.DataFrame([
        dict(name=r["name"], supplier=r.get("supplier", ""), grade=r.get("grade", r["name"]),
             **{key: r.get("properties", {}).get(key, np.nan) for key in MATERIAL}) for r in matches])
    candidates[MATERIAL] = candidates[MATERIAL].apply(pd.to_numeric, errors="coerce")
    # Batch the physical-domain checks used by validate_baseline; avoid one
    # NumPy validation call per property per record in a large library.
    valid = np.isfinite(candidates[MATERIAL].to_numpy(dtype=float)).all(axis=1)
    valid &= candidates["E"] > 0
    valid &= (candidates[["D", "Csat", "beta"]] >= 0).all(axis=1)
    valid &= (candidates["nu"] > -1) & (candidates["nu"] < .5)
    inside = np.ones(len(candidates), dtype=bool)
    for key in keys:
        inside &= candidates[key].between(float(ranges[key]["Lower"]), float(ranges[key]["Upper"]))
    invalid_count = int((~valid).sum())
    outside_count = int((valid & ~inside).sum())
    candidates = candidates[valid & inside].copy()
    for key in baseline:
        if key not in MATERIAL:
            candidates[key] = baseline[key]
    summary = dict(top_parameters=top, total=len(matches), eligible=len(candidates),
                   invalid=invalid_count, outside=outside_count, evaluated=0,
                   reference_samples=len(good), candidates=candidates)
    if candidates.empty:
        return summary
    features = np.column_stack([_normalise(candidates[key].to_numpy(), ranges[key]) for key in keys]) * scale
    candidates["screening_distance"] = tree.query(features)[0]
    shortlist = candidates.sort_values("screening_distance", kind="stable").head(shortlist_size).copy()
    shortlist["screening_rank"] = np.arange(1, len(shortlist) + 1)
    output = evaluate_samples({key: shortlist[key].to_numpy() for key in baseline}, config["diffusion"])
    for key in ("sigma_abs", "sigma_index", "sigma_hygro", "sigma_thermal"):
        shortlist[key] = output[key]
    shortlist["within_limit"] = shortlist["sigma_abs"] <= limit
    shortlist = shortlist.sort_values(["sigma_abs", "screening_distance"], kind="stable").reset_index(drop=True)
    shortlist.insert(0, "rank", np.arange(1, len(shortlist) + 1))
    summary.update(candidates=shortlist, evaluated=len(shortlist))
    return summary
