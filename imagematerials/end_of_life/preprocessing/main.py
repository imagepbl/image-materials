import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from imagematerials.end_of_life.constants import SCENARIO_SELECT, start_year, end_year
from imagematerials.end_of_life.preprocessing.preprocessing import compute_eol

def eol_preprocessing(base_dir):
    collection_in = pd.read_csv(Path(base_dir, "end_of_life","SSP2_2D_RE", "collection.csv"))
    reuse_in = pd.read_csv(Path(base_dir, "end_of_life", "SSP2_2D_RE", "reuse.csv"))
    recycling_in = pd.read_csv(Path(base_dir, "end_of_life","SSP2_2D_RE","recycling.csv"))

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

    xr_collection = collection_df.set_index(['time', 'region', 'sector', 'category', 'material']) \
                    .to_xarray()['value']

    xr_reuse = reuse_df.set_index(['time', 'region', 'sector', 'category', 'material']) \
                    .to_xarray()['value']

    xr_recycling = recycling_df.set_index(['time', 'region', 'sector', 'category', 'material']) \
                    .to_xarray()['value']
    
        # inter and extrapolated collection, reuse and recycling xr 
    collection = compute_eol(xr_collection, start_year= start_year, end_year=end_year,min_value = 0,max_value = 1)
    reuse = compute_eol(xr_reuse, start_year= start_year, end_year=end_year,min_value = 0,max_value = 1)
    recycling = compute_eol(xr_recycling, start_year= start_year, end_year=end_year,min_value = 0,max_value = 1)

    return {
        "collection": collection,
        "reuse": reuse,
        "recycling": recycling,
    }