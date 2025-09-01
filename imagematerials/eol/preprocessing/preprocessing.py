import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from imagematerials.eol.constants import SCENARIO_SELECT, start_year, end_year, full_time
from imagematerials.util import overwrite_future_rates, read_circular_economy_config
 

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
    ds = ds.reindex(Time=full_time)
    ds_before = ds.sel(Time=anchor_year)
    ds.loc[dict(Time=slice(None, anchor_year - 1))] = ds_before

    # Interpolate between anchor and target
    ds = ds.interpolate_na(dim="Time", method="linear")

    # Fill post-target with target value
    ds_after = ds.sel(Time=target_year)
    ds.loc[dict(Time=slice(target_year + 1, None))] = ds_after

    # Clip to valid range
    return ds.clip(min=min_value, max=max_value)


def eol_preprocessing(base_dir, circular_economy_scenario_dirs):
    circular_economy_config = read_circular_economy_config(circular_economy_scenario_dirs)

    collection_in = pd.read_csv(Path(base_dir, "end_of_life","SSP2_2D_RE", "collection.csv"))
    reuse_in = pd.read_csv(Path(base_dir, "end_of_life", "SSP2_CP", "reuse.csv"))
    recycling_in = pd.read_csv(Path(base_dir, "end_of_life","SSP2_CP","recycling.csv"))

    # renaming columns for consistency
    collection_in = collection_in.rename(columns={'time':'Time','sector': 'Sector', 'regions': 'Region', 'category':'Type'})
    reuse_in = reuse_in.rename(columns={ 'Element': 'Type'} )
    recycling_in = recycling_in.rename(columns={'Element': 'Type'} )

    # dropping redundant 'Sector' level
    collection_in = collection_in.drop(columns='Sector')
    reuse_in = reuse_in.drop(columns='Sector')
    recycling_in = recycling_in.drop(columns='Sector')

    # melting material columns
    id_vars = ['Region', 'Time', 'Type'] 
    value_vars = ['Steel','Concrete','Wood','Cu','Aluminium', 'Glass']

    collection_df = pd.melt(
        collection_in,
        id_vars = id_vars,
        value_vars = value_vars,
        var_name = 'material',
        value_name = 'value'
    )

    reuse_df = pd.melt(
        reuse_in,
        id_vars = id_vars,
        value_vars = value_vars,
        var_name = 'material',
        value_name = 'value'
    )

    recycling_df = pd.melt(
        recycling_in,
        id_vars = id_vars,
        value_vars = value_vars,
        var_name = 'material',
        value_name = 'value'
    )
   
    # renaming material coordinates
    material_rename = {
        'Cu': 'Copper'
    }

    collection_df['material'] = collection_df['material'].replace(material_rename)
    reuse_df['material'] = reuse_df['material'].replace(material_rename)
    recycling_df['material'] = recycling_df['material'].replace(material_rename)

    # creating xarrays from EoL dfs
    xr_collection = collection_df.set_index(['Time', 'Region', 'Type', 'material']) \
                    .to_xarray()['value']

    xr_reuse = reuse_df.set_index(['Time', 'Region','Type', 'material']) \
                    .to_xarray()['value']

    xr_recycling = recycling_df.set_index(['Time', 'Region','Type', 'material']) \
                    .to_xarray()['value']
    
    # add Brick and Cement to materials dim, fill w/ 0 and reorder
    outflows_materials = ['Aluminium', 'Brick', 'Cement', 'Concrete', 'Copper', 'Glass', 'Steel', 'Wood']
    xr_collection = xr_collection.reindex(material=outflows_materials, fill_value=0)
    xr_reuse = xr_reuse.reindex(material=outflows_materials, fill_value=0)
    xr_recycling = xr_recycling.reindex(material=outflows_materials, fill_value=0)
    
    # set targets for 2060 instead of 2050
    xr_collection = xr_collection.assign_coords(Time = [2060 if t == 2050 else t for t in xr_collection.Time.values])
    xr_reuse = xr_reuse.assign_coords(Time = [2060 if t == 2050 else t for t in xr_reuse.Time.values])
    xr_recycling = xr_recycling.assign_coords(Time = [2060 if t == 2050 else t for t in xr_recycling.Time.values])

    # scenario implementation
    building_supertypes = ["urban", "rural", "commercial"]
    vehicles_supertypes = ["passenger", "freight"]

    if "slow" in circular_economy_config.keys():
        reuse_rate_buildings = circular_economy_config["slow"]["buildings"]["eol_reuse_rate_2060"]
        reuse_rate_vehicles = circular_economy_config["slow"]["vehicles"]["eol_reuse_rate_2060"]

        xr_reuse = overwrite_future_rates(xr_reuse,2060,building_supertypes,reuse_rate_buildings)
        print("implemented 'slow' for Buildings EoL")
        xr_reuse = overwrite_future_rates(xr_reuse,2060,vehicles_supertypes,reuse_rate_vehicles)
        print("implemented 'slow' for Vehicles EoL")

    if "close" in circular_economy_config.keys():
        recycling_rate_buildings = circular_economy_config["close"]["buildings"]["eol_recycling_rate_2060"]
        recycling_rate_vehicles = circular_economy_config["close"]["vehicles"]["eol_recycling_rate_2060"]
        
        xr_recycling = overwrite_future_rates(xr_recycling,2060,building_supertypes,recycling_rate_buildings)
        print("implemented 'close' for Buildings EoL")    
        xr_recycling = overwrite_future_rates(xr_recycling,2060,vehicles_supertypes,recycling_rate_vehicles)
        print("implemented 'close' for Vehicles EoL")    

    # inter and extrapolated collection, reuse and recycling xr 
    collection = interpolate_eol_rates(xr_collection, anchor_year=2020, target_year=2060, min_value=0, max_value=1, full_time=full_time)
    reuse = interpolate_eol_rates(xr_reuse, anchor_year=2020, target_year=2060, min_value=0, max_value=1, full_time=full_time)
    recycling = interpolate_eol_rates(xr_recycling, anchor_year=2020, target_year=2060, min_value=0, max_value=1, full_time=full_time)

    collection.coords["Region"] = [str(x.values) for x in collection.coords["Region"]]
    reuse.coords["Region"] = [str(x.values) for x in reuse.coords["Region"]]
    recycling.coords["Region"] = [str(x.values) for x in recycling.coords["Region"]]

    
    return {
        "collection": collection,
        "reuse": reuse,
        "recycling": recycling,
    }
