import pandas as pd
import xarray as xr

from imagematerials.buildings.constants import FLAG_NORMAL, SCENARIO_SELECT, YEARS
from imagematerials.util import dataset_to_array, merge_dims


def compute_lifetimes(base_directory, commercial_types, flag_normal=FLAG_NORMAL):

    lifetimes_commercial = pd.read_csv(base_directory / 'files_lifetimes' / SCENARIO_SELECT / 'lifetimes_comm.csv', index_col = [0,1])  # Weibull parameter database for commercial buildings (shape & scale parameters given by region, area & building-type)
    # TODO originally lifetimes_commercial was only read in with flag_normal == 0 

    if flag_normal == 0:
        lifetimes_residential = pd.read_csv(base_directory / 'files_lifetimes' / SCENARIO_SELECT / 'lifetimes.csv', index_col = [0,1,2,3])   # Weibull parameter database for residential buildings (shape & scale parameters given by region, area & building-type)
    else:
        lifetimes_residential = pd.read_csv(base_directory / 'files_lifetimes' / 'lifetimes_normal.csv')  # Normal distribution database (Mean & StDev parameters given by region, area & building-type, though only defined by region for now)

    lifetimes_commercial_interpolated = (
        lifetimes_commercial.unstack(level='Region')
        .reindex(YEARS)
        .interpolate(method='linear', limit=300, limit_direction='both')
        .stack(level='Region')
    )
    # Interpolate for the 'Shape' column for residential data
    lifetimes_res_shape_interpolated = (
        lifetimes_residential['Shape']
        .unstack(level='time')  # Temporarily unstack 'time' level
        .reindex(columns=YEARS)  # Reindex to include all years in the specified range
        .interpolate(method='linear', limit=300, limit_direction='both', axis=1)  # Interpolate missing years
        .stack(level='time')  # Stack 'time' level back to its original position
    )

    # Interpolate for the 'Scale' column for residential data
    lifetimes_res_scale_interpolated = (
        lifetimes_residential['Scale']
        .unstack(level='time')
        .reindex(columns=YEARS)
        .interpolate(method='linear', limit=300, limit_direction='both', axis=1)
        .stack(level='time')
    )

    # Combine both interpolated 'Shape' and 'Scale' columns back into a single DataFrame
    lifetimes_residential_interpolated = pd.concat(
        [lifetimes_res_shape_interpolated, lifetimes_res_scale_interpolated],
        axis=1,
        keys=['Shape', 'Scale']
    )

    # Convert commercial lifetimes to xarray and add Type dimension.
    xr_lifetimes_commercial = dataset_to_array(lifetimes_commercial_interpolated.to_xarray(),
                                               ["Time", "Region"],
                                               ["Parameter"])
    xr_lifetimes_commercial = xr_lifetimes_commercial.expand_dims({"Type": commercial_types})

    # Convert residential lifetimes to xarray and merge Type - Area
    xr_lifetimes_residential = dataset_to_array(lifetimes_residential_interpolated.to_xarray(),
                                                ["Region", "Type", "Area", "Time"],
                                                ["Parameter"])
    xr_lifetimes_residential = merge_dims(xr_lifetimes_residential, "Type", "Area")
    xr_lifetimes_residential = xr_lifetimes_residential.transpose("Time", "Region", "Type", "Parameter")
    return xr.concat((xr_lifetimes_commercial, xr_lifetimes_residential), dim="Type")

