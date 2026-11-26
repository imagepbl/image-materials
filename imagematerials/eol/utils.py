"""Utilities for the end of life sector."""

import warnings
import pandas as pd
import prism
import xarray as xr

# define interpolation method
def interpolate_eol_rates(
    ds: xr.DataArray,
    anchor_year: int = 2020,
    target_year: int = 2060,
    min_value: float = 0,
    max_value: float = 1,
    full_time: list[int] = None,
    ) -> xr.DataArray:
    """
    Interpolate rates from anchor_year to target_year.
    - Constant before anchor_year
    - Linear interpolation between anchor_year and target_year
    - Constant after target_year
    - Clipped between min_value and max_value
    """
    # Fill pre-anchor with anchor value
    ds = ds.reindex(time=full_time)
    ds_before = ds.sel(time=anchor_year)
    ds.loc[dict(time=slice(None, anchor_year - 1))] = ds_before

    # Interpolate between anchor and target
    ds = ds.interpolate_na(dim="time", method="linear")

    # Fill post-target with target value
    ds_after = ds.sel(time=target_year)
    ds.loc[dict(time=slice(target_year + 1, None))] = ds_after

    # Clip to valid range
    return ds.clip(min=min_value, max=max_value)

def build_region_array(df: pd.DataFrame,
                       regions: list,
                       default="World") -> xr.DataArray:
    """DataFrame with columns time, Region, [Sector], Type, <materials...>
    -> DataArray (time, Region, Type, material) over `regions`.
    A region uses its own rows if present, otherwise the `default` ('World') rows.

    """
    # create xarray
    da = (
        df.drop(columns="Sector", errors="ignore")
          .set_index(["time", "Region", "Type"])  # set_index([...]) moves the id columns into the index, so the remaining columns (steel, concrete, and so on) are the materials
          .rename_axis(columns="material")   # all other columns = materials -> name the column axis materials
          .stack()                           # columns -> 'material' index level
          .to_xarray()                       # dims: time, Region, Type, material
    )
    da.name = "value"

    # check if there are unkown region names in the data
    labels = set(da.Region.values.tolist())
    unknown = labels - set(regions) - {default}
    if unknown:
        raise ValueError(f"Regions in file not in model regions: {sorted(unknown)}")

    # expand dimension Region to include all regions (all IMAGE regions)
    # if regions have individual values (they appear in the input data specifically, e.g. 2020,CAN,Electricity,grid,0.7650,...)
    # these values are used, if they do not appear in the input data, the general value for "World"
    # is used
    out = da.reindex(Region=regions)                           # NaN where a region has no rows
    if default in labels:
        out = out.fillna(da.sel(Region=default, drop=True))    # fill from World, broadcast

    if out.isnull().any():
        raise ValueError(f"Missing values after fallback to '{default}'")

    return out.transpose("time", "Region", "Type", "material")