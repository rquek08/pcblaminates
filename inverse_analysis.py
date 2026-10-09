"""Explore a bounded property space using the existing mismatch-index model."""
import numpy as np
import pandas as pd

from sensitivity_analysis import MATERIAL, PARAMETERS, evaluate_samples, saltelli_samples


def condition_records(records, condition):
    matches = [record for record in records if record["condition"] == condition]
    if not matches:
        raise ValueError("No library materials are available for this conditioning environment.")
    return matches


def library_ranges(records, condition):
    matches = condition_records(records, condition)
    rows = []
    for key in MATERIAL:
        values = [record["properties"][key] for record in matches]
        lower, upper = min(values), max(values)
        if lower == upper:
            # A constant library field is held fixed rather than inventing a range.
            continue
        rows.append(dict(Parameter=key, Label=PARAMETERS[key][0], Unit=PARAMETERS[key][1],
                         Lower=lower, Upper=upper,
                         Distribution="log-uniform" if key == "D" else "uniform"))
    return pd.DataFrame(rows)


def validate_exposure(exposure, limit):
    required = ("h_mm", "t_hours", "delta_T", "alpha_cu")
    if not np.isfinite(limit) or limit < 0:
        raise ValueError("The stress-index limit must be finite and nonnegative.")
    if any(not np.isfinite(exposure[key]) for key in required):
        raise ValueError("Exposure and geometry inputs must be finite.")
    if exposure["h_mm"] <= 0 or exposure["t_hours"] < 0:
        raise ValueError("Laminate thickness must be positive and exposure time nonnegative.")


def search_design_space(records, condition, exposure, limit, ranges, N=4096):
    """Return sampled combinations and independent library comparisons.

    Feasibility means |index| <= limit AND all chosen property bounds are met.
    Sampling is a projection of a bounded space, not a material recommendation.
    """
    validate_exposure(exposure, limit)
    matches = condition_records(records, condition)
    if not ranges or any(row["Parameter"] not in MATERIAL for row in ranges):
        raise ValueError("Choose ranges for supported material properties.")
    keys = [row["Parameter"] for row in ranges]
    # Reuse the sensitivity sampler's range validation; its A block is a Sobol design.
    samples = saltelli_samples(ranges, N, seed=12345)[:N]
    baseline = dict(matches[0]["properties"], **exposure)
    properties = dict(baseline, **{key: samples[:, i] for i, key in enumerate(keys)})
    output = evaluate_samples(properties, {"mode": "Fixed D"})
    space = pd.DataFrame({key: properties[key] for key in MATERIAL})
    space["sigma_index"] = output["sigma_index"]
    space["sigma_hygro"] = output["sigma_hygro"]
    space["sigma_thermal"] = output["sigma_thermal"]
    space["feasible"] = np.abs(space["sigma_index"]) <= limit

    comparisons = []
    for record in matches:
        p = dict(record["properties"], **exposure)
        result = evaluate_samples(p, {"mode": "Fixed D"})
        index = float(result["sigma_index"])
        inside_bounds = all(float(row["Lower"]) <= p[row["Parameter"]] <= float(row["Upper"])
                            for row in ranges)
        within_limit = abs(index) <= limit
        status = ("Within chosen limits" if inside_bounds and within_limit else
                  "Outside property bounds" if not inside_bounds else "Outside index limit")
        comparisons.append(dict(name=record["name"], grade=f"{record['supplier']} {record['grade']}",
                                sigma_index=index, margin=limit - abs(index),
                                feasible=inside_bounds and within_limit, status=status, **p))
    return dict(space=space, candidates=pd.DataFrame(comparisons))


def expansion_state(properties, exposure, length_mm, width_mm):
    """Physical in-plane free expansion; through-thickness strain is unspecified."""
    validate_exposure(exposure, 0.0)
    if not np.all(np.isfinite([length_mm, width_mm])) or min(length_mm, width_mm) <= 0:
        raise ValueError("Length and width must be finite and positive.")
    p = dict(properties, **exposure)
    result = evaluate_samples(p, {"mode": "Fixed D"})
    copper_strain = p["alpha_cu"] * 1e-6 * p["delta_T"]
    laminate_strain = float(result["eps_h"]) + p["alpha_lam"] * 1e-6 * p["delta_T"]
    return dict(laminate_strain=laminate_strain, copper_strain=copper_strain,
                delta_length_laminate=length_mm * laminate_strain,
                delta_length_copper=length_mm * copper_strain,
                delta_width_laminate=width_mm * laminate_strain,
                delta_width_copper=width_mm * copper_strain,
                mismatch_strain=laminate_strain - copper_strain,
                sigma_index=float(result["sigma_index"]))
