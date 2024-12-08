import pandas as pd

from imagematerials.buildings.constants import FLAG_NORMAL, SCENARIO_SELECT, YEARS


def compute_lifetimes(base_directory, flag_normal=FLAG_NORMAL):

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
    return lifetimes_commercial_interpolated, lifetimes_residential_interpolated
