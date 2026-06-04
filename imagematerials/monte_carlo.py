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

def load_ranges(filepath: str) -> pd.DataFrame:
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

    # required = {"time", "material", "technology", "min", "max"}
    # missing = required - set(df.columns)
    # if missing:
    #     raise ValueError(f"Ranges file is missing columns: {missing}")

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
