"""
monte_carlo.py
------------------------------------
Reads a ranges CSV (year, material, technology, min, max) and draws one
uniform random sample per (year, material, technology) pair, returning a
DataFrame in the same wide format as the original material intensities file:

    index 0 : year
    index 1 : material
    columns : technologies

Usage
-----
    from monte_carlo_material_intensities import load_ranges, sample_intensities

    ranges = load_ranges("material_intensities_ranges.csv")

    # Monte Carlo loop
    results = []
    for i in range(n_iterations):
        mi = sample_intensities(ranges)           # same shape as original CSV
        result = run_model(mi, ...)               # your model here
        results.append(result)

The sampled DataFrame can be written back to CSV with:
    mi.to_csv("sampled.csv")   # produces the same format as the original
"""

import numpy as np
import pandas as pd
import xarray as xr

from imagematerials.concepts import (
    create_electricity_graph, 
    create_region_graph, 
    create_vehicle_graph_2,
    create_building_graph
)
from imagematerials.electricity.utils import (
    logistic, 
    quadratic,
    interpolate_xr,
)
from imagematerials.vehicles.constants import maintenance_modes

dict_knowledge_graph = {
    "vehicles": create_vehicle_graph_2(),
    "electricity": create_electricity_graph(),
    "buildings": create_building_graph(),
}

dict_sector_types = {
    "vehicles": maintenance_modes,
}

# ---------------------------------------------------------------------------
# Sampling distributions
# ---------------------------------------------------------------------------
# Extend this dict to add other distribution types.
# Each callable receives (rng, row) and returns a scalar.

def _uniform(rng: np.random.Generator, row: pd.Series) -> float:
    lo, hi = row["min"], row["max"]
    if lo == hi:
        return lo
    return float(rng.uniform(lo, hi))


def _triangular(rng: np.random.Generator, row: pd.Series) -> float:
    """Requires columns 'min', 'mode', 'max'."""
    lo, mode, hi = row["min"], row["mode"], row["max"]
    if lo == hi:
        return lo
    return float(rng.triangular(lo, mode, hi))


DISTRIBUTIONS = {
    "uniform": _uniform,
    "triangular": _triangular,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_ranges_materials(filepath: str) -> pd.DataFrame:
    """Load the ranges CSV.

    Expected columns: year, material, technology, min, max
    Optional columns: mode  (needed for triangular distribution)
                      distribution  (defaults to 'uniform')

    Returns the raw long-form DataFrame; pass it directly to sample_intensities.

    """
    df = pd.read_csv(filepath)

    aliases = {
        "Cohort": ["time", "Time", "year", "years", "Year", "cohort", "Cohort"],
        "Type": ["technology", "Technology", "type", "Type"],
        "material": ["material", "Material"],
    }

    rename_map = {}

    for target, candidates in aliases.items():
        found = [c for c in candidates if c in df.columns]

        if not found:
            raise ValueError(
                f"Missing required column for '{target}'. "
                f"Expected one of: {candidates}"
            )

        rename_map[found[0]] = target

    df = df.rename(columns=rename_map)

    if "distribution" not in df.columns:
        df["distribution"] = "uniform"

    return df


def sample_intensities(
        sector: str,
        ranges: pd.DataFrame,
        rng: np.random.Generator | None = None,
        seed: int | None = None,
        year_start: int | None = 1971,
        year_end: int | None = 2100
    ) -> xr.DataArray:
    """Sample material intensities by drawing one random sample from each (Time, material, Type) range.

    Parameters
    ----------
    ranges : pd.DataFrame
        Output of load_ranges().
    rng : np.random.Generator, optional
        Pass an existing Generator for reproducibility inside a loop.
    seed : int, optional
        Seed a fresh Generator (ignored when rng is supplied).

    Returns
    -------
    xr.DataArray

    """
    if rng is None:
        rng = np.random.default_rng(seed)

    rows = []
    for _, row in ranges.iterrows():
        dist_name = row.get("distribution", "uniform")
        sampler = DISTRIBUTIONS.get(dist_name)
        if sampler is None:
            raise ValueError(
                f"Unknown distribution '{dist_name}'. "
                f"Available: {list(DISTRIBUTIONS)}"
            )
        value = sampler(rng, row)
        rows.append({"Cohort": row["Cohort"], "material": row["material"],
                     "Type": row["Type"], "value": value})

    long_df = pd.DataFrame(rows)

    da = long_df.set_index(["Cohort", "Type", "material"])["value"].to_xarray()
    da_interp = interpolate_xr(da, year_start, year_end)
    kg = dict_knowledge_graph[sector]
    da_interp = kg.rebroadcast_xarray(da_interp, output_coords=dict_sector_types[sector])

    return da_interp

def load_ranges_lifetimes(filepath: str, default_distribution: str = "uniform") -> pd.DataFrame: #### NOT WORKING YET - CHECK!
    """Load the wide lifetime-parameter matrix -> tidy long-form, one row per (Cohort, Type).

    Output columns: Cohort, Type, mean, mean_min, mean_max, stdev, shape, scale, distribution.
    `distribution` governs how the *mean* lifetime is drawn between mean_min and mean_max
    (default uniform); it is NOT the survival distribution (that's stdev / shape / scale).
    """
    df = pd.read_csv(filepath)

    year_aliases = ["time", "Time", "year", "years", "Year", "cohort", "Cohort"]
    stat_aliases = ["data", "Data", "stat", "statistic", "Statistic", "parameter"]
    year_col = next((c for c in year_aliases if c in df.columns), None)
    stat_col = next((c for c in stat_aliases if c in df.columns), None)
    if year_col is None:
        raise ValueError(f"Missing year column. Expected one of: {year_aliases}")
    if stat_col is None:
        raise ValueError(f"Missing statistic column. Expected one of: {stat_aliases}")

    tech_cols = [c for c in df.columns if c not in (year_col, stat_col)]
    if not tech_cols:
        raise ValueError("No technology columns found besides the id columns.")

    long = df.melt(id_vars=[year_col, stat_col], value_vars=tech_cols,
                   var_name="Type", value_name="value")
    wide = (long.set_index([year_col, "Type", stat_col])["value"]
                .unstack(stat_col).reset_index())
    wide.columns.name = None
    wide = wide.rename(columns={year_col: "Cohort"})

    for c in wide.columns:                       # melt makes stats object if a string row is mixed in
        if c not in {"Cohort", "Type", "distribution"}:
            wide[c] = pd.to_numeric(wide[c], errors="coerce")

    if "distribution" not in wide.columns:
        wide["distribution"] = default_distribution
    else:
        wide["distribution"] = wide["distribution"].fillna(default_distribution)
    return wide


def _validate_bounds(ranges: pd.DataFrame) -> None: #### NOT WORKING YET - CHECK!
    lo, mid, hi = ranges["mean_min"], ranges["mean"], ranges["mean_max"]
    bad = ranges[~((lo <= mid) & (mid <= hi) & (lo <= hi))]
    if len(bad):
        rows = bad[["Cohort", "Type", "mean_min", "mean", "mean_max"]].to_string(index=False)
        raise ValueError(
            "mean_min <= mean <= mean_max is violated for these rows "
            "(check the source file):\n" + rows
        )


def sample_lifetimes( #### NOT WORKING YET - CHECK!
    ranges: pd.DataFrame,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    keep_stats=("mean", "stdev", "shape", "scale"),
    validate: bool = True,
):
    """Draw one mean lifetime per (Cohort, Type) from its `distribution`, then run the
    same transformation the deterministic code uses. stdev / shape / scale pass through
    unchanged (they parameterise the survival distribution, which the MC does not randomise).
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    if validate:
        _validate_bounds(ranges)

    sampled_means = []
    for _, row in ranges.iterrows():
        lo, hi = row["mean_min"], row["mean_max"]
        if lo == hi:                                     # degenerate range -> point value
            sampled_means.append(lo)
            continue
        dist_name = row.get("distribution", "uniform")
        sampler = DISTRIBUTIONS.get(dist_name)
        if sampler is None:
            raise ValueError(f"Unknown distribution '{dist_name}'. Available: {list(DISTRIBUTIONS)}")
        sampled_means.append(sampler(rng, {"min": lo, "max": hi, "mode": row.get("mean")}))

    sampled = ranges.copy()
    sampled["mean"] = sampled_means

    # rebuild the (year, data) x mode matrix the deterministic path starts from
    stats = [s for s in keep_stats if s in sampled.columns]
    tidy = sampled[["Cohort", "Type"] + stats].melt(
        id_vars=["Cohort", "Type"], var_name="data", value_name="value")
    matrix = tidy.pivot(index=["Cohort", "data"], columns="Type", values="value")
    matrix.index = matrix.index.set_names(["year", "data"])
    matrix.columns.name = "mode"

    # --- identical to the original deterministic transformation ---
    s = matrix.stack()
    s = s[s != 0]
    wide = s.unstack(["mode", "data"])
    wide = interpolate(pd.DataFrame(wide))
    data_xarray = pandas_to_xarray(wide, unit_mapping)
    data_xarray = convert_lifetime(data_xarray)
    return data_xarray

# def sample_lifetimes(
#         sector: str,
#         ranges: pd.DataFrame,
#         rng: np.random.Generator | None = None,
#         seed: int | None = None,
#         year_start: int | None = 1971,
#         year_end: int | None = 2100
#     ) -> dict:

#     #...

#     return
