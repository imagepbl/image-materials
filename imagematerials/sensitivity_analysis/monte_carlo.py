"""
Monte Carlo input pipeline, split into three stages.

    load_*(filepath)      -> tidy parameter table       (one per data type)
    sample_values(...)    -> same table + 'sampled'     (ONE function, shared by all)
    process_*(...)        -> xr.DataArray for the model (one per data type)

Every loader must produce the same *canonical* table:

    <id columns, free>  |  value  |  min  |  max  | distribution | [extras...]

sample_values() only ever looks at 'distribution', 'min', 'max' (and 'value', depending on the 
distribution), and writes its result to a new column called 'sampled'. The id columns and any extra
columns - unit, Region, category, ... - pass through untouched. Thereffore it can be used for 
different data types and sectors.

Note on the name 'value': it is the standard value used for normal model runs. It is used here as 
the central tendency for distributions.
"""

import numpy as np
import pandas as pd
import xarray as xr

import prism

from imagematerials.concepts import (
    create_electricity_graph, 
    create_region_graph, 
    create_vehicle_graph_2,
    create_building_graph
)
from imagematerials.util import (
    dataset_to_array,
    convert_lifetime,
    pandas_to_xarray
)
from imagematerials.electricity.utils import (
    logistic, 
    quadratic,
    interpolate_xr,
)
from imagematerials.electricity.constants import (
    YEAR_FIRST_GRID,
    EPG_TECHNOLOGIES,
    EPG_TECHNOLOGIES_FINAL,
    EPG_SUB_TECHNOLOGIES,
    TECH_STATIONARY_STORAGE,
)
from imagematerials.vehicles.constants import (
    vehicles_modes_sensitivity_analysis,
    unit_mapping,
    EV_BATTERY_TYPES,
    END_YEAR, 
    FIRST_YEAR
)
from imagematerials.vehicles.modelling_functions import (
    interpolate,
    scenario_change
)



dict_sector_knowledge_graph = {
    "vehicles": create_vehicle_graph_2(),
    "generation": create_electricity_graph(),
    "storage_phs": create_electricity_graph(),
    "storage_other": create_electricity_graph(),
    "ev_battery": create_electricity_graph(),
    "buildings": create_building_graph(),
}
dict_sector_types = {
    "vehicles": vehicles_modes_sensitivity_analysis,
    "generation": EPG_SUB_TECHNOLOGIES,
    "storage_phs": ["PHS"],
    "storage_other": TECH_STATIONARY_STORAGE,
    "ev_battery": EV_BATTERY_TYPES,
}
dict_sector_start_year = {
    "vehicles": FIRST_YEAR,
    "generation": YEAR_FIRST_GRID,
    "storage_phs": YEAR_FIRST_GRID,
    "storage_other": YEAR_FIRST_GRID,
    "ev_battery": YEAR_FIRST_GRID,
}
dict_sector_end_year = {
    "vehicles": END_YEAR,
    "generation": 2100,
    "storage_phs": 2100,
    "storage_other": 2100,
    "ev_battery": 2100,
}
dict_sector_separator = {
    "vehicles":         " - ",
    "electricity":      "_",
    "generation":       "_",
    "storage_phs":      "_",
    "storage_other":    "_",
    "ev_battery":     "_",
}

# ===========================================================================
# 0. Helpers
# ===========================================================================

def _merge_type_subtype(df: pd.DataFrame, sector: str) -> pd.DataFrame:
    """Merge 'Type' and 'SubType' columns into 'Type', separator depends on sector.

    If 'SubType' is not present in df, df is returned unchanged.
    Rows with an empty/NaN SubType keep 'Type' unmodified (no dangling separator).
    """
    if "SubType" not in df.columns:
        return df

    if sector not in dict_sector_separator:
        raise ValueError(
            f"Unknown sector '{sector}'; expected one of {list(dict_sector_separator)}"
        )
    separator = dict_sector_separator[sector]

    df = df.copy()
    has_sub = df["SubType"].notna() & (df["SubType"].astype(str).str.strip() != "")
    df["Type"] = df["Type"].astype(str)
    df.loc[has_sub, "Type"] = (
        df.loc[has_sub, "Type"] + separator + df.loc[has_sub, "SubType"].astype(str)
    )
    return df.drop(columns="SubType")

# ===========================================================================
# 1. Distribution registry
# ===========================================================================
# Each sampler gets (rng, block) where `block` is a DataFrame containing ALL the
# rows that use that distribution. numpy broadcasts over the columns, so one call
# draws one value per row - no Python loop over rows.
#
# If you already have a DISTRIBUTIONS dict with (rng, row) signatures, this is
# the one change you need to make: take a frame instead of a single row.


def _uniform(rng: np.random.Generator, block: pd.DataFrame) -> np.ndarray:
    """Uniform between min and max. min == max is fine - numpy returns that value."""
    return rng.uniform(block["min"], block["max"])
 
 
def _triangular(rng: np.random.Generator, block: pd.DataFrame) -> np.ndarray: #TODO: check this!!!
    """Triangular with the most-likely value at 'value'.
 
    Written as a full function rather than a lambda because it needs two guards:
    numpy raises error if peak falls outside [min, max], and also if min == max.
    """
    lo   = block["min"].to_numpy(dtype=float)
    peak = block["value"].to_numpy(dtype=float)
    hi   = block["max"].to_numpy(dtype=float)
 
    bad = (peak < lo) | (peak > hi)
    if bad.any():
        raise ValueError(
            f"'peak' lies outside [min, max] in {bad.sum()} row(s):\n"
            + block[bad].to_string()
        )
 
    out = np.empty(len(block), dtype=float)
    flat = lo == hi                       # zero-width range -> no randomness possible
    out[flat] = lo[flat]
    if (~flat).any():
        out[~flat] = rng.triangular(lo[~flat], peak[~flat], hi[~flat])
    return out
 
 
def _fixed(rng: np.random.Generator, block: pd.DataFrame) -> np.ndarray:
    """No randomness. Lets you hold parameters constant without special-casing them."""
    return block["value"].to_numpy(dtype=float)
 
 
DISTRIBUTIONS = {
    "uniform": _uniform,
    "triangular": _triangular,
    "fixed": _fixed,
}

# ===========================================================================
# 2. Loaders - one per data type
# ===========================================================================

def _rename_to_canonical(df: pd.DataFrame,
                         aliases: dict,
                         optional: set = frozenset()
                         ) -> pd.DataFrame: #TODO: discuss if strict guidelines on input files header names are the better way
    """Rename columns to canonical names using an alias table.

    aliases maps canonical_name -> list of accepted spellings in the file.
    optional is a set of canonical_names that may be absent from the file without raising an error 
    (e.g. sub_technology).
    Raises ValueError if a required (non-optional) column is missing entirely.
    """
    rename_map = {}
    for canonical, candidates in aliases.items():
        found = [c for c in candidates if c in df.columns]
        if not found:
            if canonical in optional:
                continue
            raise ValueError(
                f"Missing required column for '{canonical}'. "
                f"Expected one of: {candidates}"
            )
        rename_map[found[0]] = canonical      # first match wins
    return df.rename(columns=rename_map)


def _add_distribution_column(df: pd.DataFrame, default_distribution: str = "uniform") -> pd.DataFrame:
    """Shared last step for every loader: fill the distribution column, make sure
    the index is unique (sample_values assigns by index), and sanity-check bounds.

    """
    if "distribution" not in df.columns:
        df["distribution"] = default_distribution
    else:
        df["distribution"] = df["distribution"].fillna(default_distribution)

    df = df.reset_index(drop=True)            # guarantees a unique index

    bad = df[df["min"] > df["max"]]
    if len(bad):
        raise ValueError(
            "Found rows where min > max (check the source file):\n"
            + bad.to_string()
        )
    return df


def load_material_intensities(filepath: str,
                              sector: str
                              ) -> pd.DataFrame:
    """Long-form intensity ranges -> canonical table.

    Expected columns: year, material, technology, min, max
    Optional:         value, distribution, sub_technology

    """
    df = pd.read_csv(filepath)

    df = _rename_to_canonical(df, {
        "Cohort":   ["time", "Time", "year", "years", "Year", "cohort", "Cohort"],
        "Type":     ["technology", "Technology", "type", "Type"],
        "SubType":  ["SubType", "sub_technology", "sub-technology", "sub_type", "Sub-type", "subtype"],
        "material": ["material", "Material"],
        "min":      ["min", "Min", "minimum"],
        "max":      ["max", "Max", "maximum"],
    }, optional={"SubType"})

    df = _merge_type_subtype(df, sector)
    
    return _add_distribution_column(df)


def load_lifetimes(filepath: str) -> pd.DataFrame:
    """Wide lifetime-parameter matrix -> canonical table, one row per (Cohort, Type).

    Input is wide: one column per technology, one row per (year, statistic),
    where statistic is mean / mean_min / mean_max / stdev / shape / scale / ...

    Output: Cohort, Type, distribution, min, max, peak, plus stdev/shape/scale
    passed through unchanged. min/max come from mean_min/mean_max and peak from
    mean, i.e. the MC randomises the MEAN lifetime. stdev/shape/scale parameterise
    the survival curve and are NOT randomised - they just ride along.

    """
    df = pd.read_csv(filepath)

    df = _rename_to_canonical(df, {
        "Cohort": ["time", "Time", "year", "years", "Year", "cohort", "Cohort"],
        "parameter":   ["data", "Data", "stat", "statistic", "Statistic", "parameter"],
    })

    return _add_distribution_column(df)


# ===========================================================================
# 3. The sampler - ONE function, reused everywhere.
# ===========================================================================

def sample_values(
        params: pd.DataFrame,
        rng: np.random.Generator,
        out_col: str = "sampled",
    ) -> pd.DataFrame:
    """Draw one random value per row of `params`.

    Parameters
    ----------
    params : pd.DataFrame
        Output of any load_* function above. Need to have the columns "min" and "max" (and "value" 
        depending on the distribution).
    rng : np.random.Generator
        Pass the same generator on every iteration for a reproducible sequence.
    out_col : str
        Name of the column the drawn values are written to. Deliberately NOT
        'value', because input files often already carry a 'value' column
        holding the deterministic central estimate, which must survive.

    Returns
    -------
    pd.DataFrame
        A copy of `params` with one extra column, named by `out_col`.
        All original columns are left untouched.

    """
    # Fail loudly rather than overwriting an existing column of the same name.
    if out_col in params.columns:
        raise ValueError(
            f"Column '{out_col}' already exists in the input table. Rename it in "
            f"the loader, or call sample_values(..., out_col='something_else')."
        )

    out = params.copy()
    out[out_col] = np.nan

    # One call per distribution type rather than one per row.
    for dist_name, block in params.groupby("distribution", sort=False):
        sampler = DISTRIBUTIONS.get(dist_name)
        if sampler is None:
            raise ValueError(
                f"Unknown distribution '{dist_name}'. "
                f"Available: {list(DISTRIBUTIONS)}"
            )
        # np.asarray turns a scalar into shape (), so this one check catches both
        # a wrong-length return and a sampler that returned a single number
        # (which pandas would otherwise silently broadcast to every row).
        values = np.asarray(sampler(rng, block), dtype=float)
        if values.shape != (len(block),):
            raise ValueError(
                f"Sampler '{dist_name}' returned shape {values.shape}, "
                f"expected ({len(block)},) - one value per row."
            )

        # block.index and the returned array are in the same order, so this
        # writes each drawn value back onto its own row.
        out.loc[block.index, out_col] = values

    return out


# ===========================================================================
# 4. Processors - one per data type. These are your existing deterministic
#    transformations, with the sampled table as input instead of the CSV.
# ===========================================================================

def process_material_intensities(
        sampled: pd.DataFrame,
        sector: str,
        year_start: int | None = None,
        year_end: int | None = None,
    ) -> xr.DataArray:
    """Sampled intensity table -> model-ready DataArray.

    Only Cohort / Type / material are used as dimensions; every other column
    (unit, Region, category, min, max, peak, ...) is ignored here. If your file
    ever contains more than one row per (Cohort, Type, material) - e.g. several
    Regions - this needs an extra dimension or a filter first.

    """
    da = sampled.set_index(["Cohort", "Type", "material"])["sampled"].to_xarray()
    unique_units = sampled["unit"].unique()
    if len(unique_units) != 1:
        raise ValueError(
            f"Expected a single unit for all rows, but found multiple: {unique_units}"
        )
    unit = unique_units[0]
    da = prism.Q_(da, unit)
    da.name = f"{sector}Materials"

    knowledge_graph = dict_sector_knowledge_graph[sector] #TODO: is this needed?
    da = knowledge_graph.rebroadcast_xarray(da, output_coords=dict_sector_types[sector])
    da = da.sortby("Type")

    if not year_start:
        year_start = dict_sector_start_year[sector]
    if not year_end:
        year_end = dict_sector_end_year[sector]

    da = interpolate_xr(da, year_start, year_end)

    return da


def process_lifetimes(
        sampled: pd.DataFrame,
        sector: str,
        year_start: int | None = None,
        year_end: int | None = None,
    ) -> xr.DataArray:
    """Sampled lifetime table -> model-ready DataArray.

    The drawn value IS the mean lifetime; stdev/shape/scale are unchanged due to entry "fixed" in 
    distribution column.
    
    vehicles: lifetimes are defined on type level, not sub-type/engine (for Cars and not Cars - BEV, etc.)

    """
    da = sampled.copy()

    da = (
        da.set_index(["Cohort", "Type", "parameter"])["sampled"]
        .to_xarray()
        .rename({"parameter": "DistributionParams"})
    )
    da.name = f"{sector}Lifetime"

    if not year_start:
        year_start = dict_sector_start_year[sector]
    if not year_end:
        year_end = dict_sector_end_year[sector]

    da = interpolate_xr(da, year_start, year_end)

    da = convert_lifetime(da)

    return da