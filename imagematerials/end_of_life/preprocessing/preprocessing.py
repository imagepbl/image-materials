import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from end_of_life.preprocessing.constants import start_year, end_year, full_time, SCENARIO_SELECT, EolTypes


def get_eol_rates():

    collection_in = pd.read_csv(Path("data", "raw", "end_of_life","SSP2_2D_RE", "collection.csv"))
    reuse_in = pd.read_csv(Path("data", "raw", "end_of_life", "SSP2_2D_RE", "reuse.csv"))
    recycling_in = pd.read_csv(Path("data","raw", "end_of_life","SSP2_2D_RE","recycling.csv"))

    # renaming columns for consistency
    collection_in = collection_in.rename(columns={'regions': 'region'})
    reuse_in = reuse_in.rename(columns={'Time': 'time', 'Region':'region', 'Sector': 'sector', 'Element': 'category'} )
    recycling_in = recycling_in.rename(columns={'Time': 'time', 'Region':'region', 'Sector': 'sector', 'Element': 'category'} )

    # melting material columns
    id_vars = ['region', 'time', 'sector', 'category'] 
    value_vars = ['Steel',	'Concrete',	'Wood',	'Cu','Aluminium', 'Glass']

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
    return collection_df, reuse_df, recycling_df

# converting melted dataframes to xarrays
def get_eol_xr(collection_df, reuse_df, recycling_df):

    xr_collection = collection_df.set_index(['time', 'region', 'sector', 'category', 'material']) \
                    .to_xarray()['value']

    xr_reuse = reuse_df.set_index(['time', 'region', 'sector', 'category', 'material']) \
                    .to_xarray()['value']

    xr_recycling = recycling_df.set_index(['time', 'region', 'sector', 'category', 'material']) \
                    .to_xarray()['value']
    
    return xr_collection, xr_reuse, xr_recycling

# define interpolation method
def compute_eol(ds, start_year, end_year, min_value = 0, max_value = 1, method = 'linear'):              # collection, reuse, and recycling rate cannot be lower than zero or higher than 1

    reindexed = ds.reindex(time=full_time)
    interpolated = reindexed.interpolate_na(dim='time', method=method, fill_value='extrapolate')
    clipped = interpolated.clip(min=min_value, max = max_value)

    return clipped