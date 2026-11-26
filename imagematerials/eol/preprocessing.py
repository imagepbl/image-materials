import pandas as pd
import xarray as xr
import numpy as np
from pathlib import Path

from imagematerials.constants import IMAGE_REGIONS, ALL_MATERIALS
from imagematerials.util import (
    flag_enabled,
    overwrite_future_rates,
    read_circular_economy_data,
    read_resource_efficiency_flags,
)
from imagematerials.eol.constants import SCENARIO_SELECT, start_year, end_year, full_time
from imagematerials.eol.utils import (
    interpolate_eol_rates,
    build_region_array,
)

def eol_preprocessing(base_dir, circular_economy_data_file=None, resource_efficiency_flags_file=None):
    circular_economy_config = read_circular_economy_data(circular_economy_data_file)
    resource_efficiency_flags = read_resource_efficiency_flags(resource_efficiency_flags_file)

    # input data with columns time,Region,Sector,Type,steel,concrete,...
    # Region can contain values for specific IMAGE regions (CAN, USA, ...), values for a generalized
    # "World" region or a mix (if data are available for some regions, while others will be assigned
    # default values)
    collection_data = pd.read_csv(Path(base_dir, "end_of_life", "standard_data", "collection.csv"))
    reuse_data = pd.read_csv(Path(base_dir, "end_of_life", "standard_data", "reuse.csv"))
    recycling_data = pd.read_csv(Path(base_dir, "end_of_life","standard_data","recycling.csv"))

    # build DataArray (time, Region, Type, material) over all IMAGE regions
    xr_collection = build_region_array(collection_data, IMAGE_REGIONS)
    xr_reuse =      build_region_array(reuse_data, IMAGE_REGIONS)
    xr_recycling =  build_region_array(recycling_data, IMAGE_REGIONS)
    
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

    return {
        "collection": collection,
        "reuse": reuse,
        "recycling": recycling,
    }