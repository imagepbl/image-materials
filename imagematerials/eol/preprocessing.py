import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path
from imagematerials.constants import IMAGE_REGIONS, ALL_MATERIALS
from imagematerials.eol.constants import SCENARIO_SELECT, start_year, end_year, full_time
from imagematerials.util import (
    flag_enabled,
    overwrite_future_rates,
    read_circular_economy_data,
    read_resource_efficiency_flags,
)
from imagematerials.concepts import create_region_graph


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


def eol_preprocessing(base_dir, circular_economy_data_file=None, resource_efficiency_flags_file=None):
    circular_economy_config = read_circular_economy_data(circular_economy_data_file)
    resource_efficiency_flags = read_resource_efficiency_flags(resource_efficiency_flags_file)

    collection_data = pd.read_csv(Path(base_dir, "end_of_life", "standard_data", "collection.csv"))
    reuse_data = pd.read_csv(Path(base_dir, "end_of_life", "standard_data", "reuse.csv"))
    recycling_data = pd.read_csv(Path(base_dir, "end_of_life","standard_data","recycling.csv"))

    xr_collection = (
        collection_data
        .drop(columns="Sector")
        .set_index(["time", "Region", "Type"]) # set_index([...]) moves the id columns into the index, so the remaining columns (steel, concrete, and so on) are the materials
        .rename_axis(columns="material")   # name the column axis
        .stack()                           # columns -> 'material' index level
        .to_xarray()                       # dims: time, Region, Type, material
    )
    xr_collection.name = "value"

    xr_reuse = (
        reuse_data
        .drop(columns="Sector")
        .set_index(["time", "Region", "Type"])
        .rename_axis(columns="material")
        .stack()
        .to_xarray()
    )
    xr_reuse.name = "value"

    xr_recycling = (
        recycling_data
        .drop(columns="Sector")
        .set_index(["time", "Region", "Type"])
        .rename_axis(columns="material")
        .stack()
        .to_xarray()
    )
    xr_recycling.name = "value"
    
    # add other materials, fill w/ 0 and reorder
    xr_collection = xr_collection.reindex(material=ALL_MATERIALS, fill_value=0)
    xr_reuse = xr_reuse.reindex(material=ALL_MATERIALS, fill_value=0)
    xr_recycling = xr_recycling.reindex(material=ALL_MATERIALS, fill_value=0)
    
    # set targets for 2060 instead of 2050
    xr_collection = xr_collection.assign_coords(time = [2060 if t == 2050 else t for t in xr_collection.time.values])
    xr_reuse = xr_reuse.assign_coords(time = [2060 if t == 2050 else t for t in xr_reuse.time.values])
    xr_recycling = xr_recycling.assign_coords(time = [2060 if t == 2050 else t for t in xr_recycling.time.values])

    # scenario implementation
    building_supertypes = ["urban", "rural", "commercial"]
    vehicles_supertypes = ["passenger", "freight"]
    electricity_supertypes = ["generation", "grid", "storage"]

    if flag_enabled(resource_efficiency_flags, "end_of_life", "FlagLifetimeExtension"):
        eol_lifetime_extension = circular_economy_config["end_of_life"]["FlagLifetimeExtension"]
        reuse_rate_buildings = eol_lifetime_extension["buildings"]["eol_reuse_rate_2060"]
        reuse_rate_vehicles = eol_lifetime_extension["vehicles"]["eol_reuse_rate_2060"]
        reuse_rate_electricity = eol_lifetime_extension["electricity"]["eol_reuse_rate_2060"]

        xr_reuse = overwrite_future_rates(xr_reuse,2060,building_supertypes,reuse_rate_buildings)
        print("implemented FlagLifetimeExtension for Buildings EoL")
        xr_reuse = overwrite_future_rates(xr_reuse,2060,vehicles_supertypes,reuse_rate_vehicles)
        print("implemented FlagLifetimeExtension for Vehicles EoL")
        xr_reuse = overwrite_future_rates(xr_reuse,2060,electricity_supertypes,reuse_rate_electricity)
        print("implemented FlagLifetimeExtension for Electricity EoL")

    if flag_enabled(resource_efficiency_flags, "end_of_life", "FlagRecycling"):
        eol_recycling = circular_economy_config["end_of_life"]["FlagRecycling"]
        recycling_rate_buildings = eol_recycling["buildings"]["eol_recycling_rate_2060"]
        recycling_rate_vehicles = eol_recycling["vehicles"]["eol_recycling_rate_2060"]
        recycling_rate_electricity = eol_recycling["electricity"]["eol_recycling_rate_2060"]

        xr_recycling = overwrite_future_rates(xr_recycling,2060,building_supertypes,recycling_rate_buildings)
        print("implemented FlagRecycling for Buildings EoL")
        xr_recycling = overwrite_future_rates(xr_recycling,2060,vehicles_supertypes,recycling_rate_vehicles)
        print("implemented FlagRecycling for Vehicles EoL")
        xr_recycling = overwrite_future_rates(xr_recycling,2060,electricity_supertypes,recycling_rate_electricity)
        print("implemented FlagRecycling for Electricity EoL")

    # inter and extrapolated collection, reuse and recycling xr 
    collection = interpolate_eol_rates(xr_collection, anchor_year=2020, target_year=2060, min_value=0, max_value=1, full_time=full_time)
    reuse = interpolate_eol_rates(xr_reuse, anchor_year=2020, target_year=2060, min_value=0, max_value=1, full_time=full_time)
    recycling = interpolate_eol_rates(xr_recycling, anchor_year=2020, target_year=2060, min_value=0, max_value=1, full_time=full_time)

    # convert region coords to str for consistency
    collection.coords["Region"] = [str(x.values) for x in collection.coords["Region"]]
    reuse.coords["Region"] = [str(x.values) for x in reuse.coords["Region"]]
    recycling.coords["Region"] = [str(x.values) for x in recycling.coords["Region"]]

    # convert region names to the standard IMAGE regions
    knowledge_graph_region = create_region_graph()

    collection  = knowledge_graph_region.rebroadcast_xarray(collection, output_coords=IMAGE_REGIONS, dim="Region")
    reuse       = knowledge_graph_region.rebroadcast_xarray(reuse, output_coords=IMAGE_REGIONS, dim="Region")
    recycling   = knowledge_graph_region.rebroadcast_xarray(recycling, output_coords=IMAGE_REGIONS, dim="Region")

    return {
        "collection": collection,
        "reuse": reuse,
        "recycling": recycling,
    }
