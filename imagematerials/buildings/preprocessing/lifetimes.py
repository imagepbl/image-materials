import pandas as pd
import xarray as xr

from imagematerials.buildings.constants import FLAG_NORMAL, SCENARIO_SELECT, ALL_YEARS
from imagematerials.util import dataset_to_array, merge_dims



def compute_lifetimes(base_directory, commercial_types, distribution_type="weibull"):

    lifetimes_commercial = pd.read_csv(base_directory / 'files_lifetimes' / SCENARIO_SELECT / 'lifetimes_comm.csv', index_col = [0,1])  # Weibull parameter database for commercial buildings (shape & scale parameters given by region, area & building-type)
    # TODO originally lifetimes_commercial was only read in with flag_normal == 0

    if distribution_type == "weibull":
        lifetimes_residential = pd.read_csv(base_directory / 'files_lifetimes' / SCENARIO_SELECT / 'lifetimes.csv', index_col = [0,1,2,3])   # Weibull parameter database for residential buildings (shape & scale parameters given by region, area & building-type)
    elif distribution_type == "folded_norm":
        lifetimes_residential = pd.read_csv(base_directory / 'files_lifetimes' / 'lifetimes_normal.csv')  # Normal distribution database (Mean & StDev parameters given by region, area & building-type, though only defined by region for now)
    else:
        raise ValueError(f"Unknown distribution type {distribution_type}, "
                         "available: [weibull, folded_norm]")
    lifetimes_commercial_interpolated = (
        lifetimes_commercial.unstack(level='Region')
        .reindex(ALL_YEARS)
        .interpolate(method='linear', limit=300, limit_direction='both')
        .stack(level='Region')
    )
    # Interpolate for the 'Shape' column for residential data
    lifetimes_res_shape_interpolated = (
        lifetimes_residential['Shape']
        .unstack(level='time')  # Temporarily unstack 'time' level
        .reindex(columns=ALL_YEARS)  # Reindex to include all years in the specified range
        .interpolate(method='linear', limit=300, limit_direction='both', axis=1)  # Interpolate missing years
        .stack(level='time')  # Stack 'time' level back to its original position
    )

    # Interpolate for the 'Scale' column for residential data
    lifetimes_res_scale_interpolated = (
        lifetimes_residential['Scale']
        .unstack(level='time')
        .reindex(columns=ALL_YEARS)
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
    fixed_coords = [x if x != "Appartments" else "Appartment" for x in xr_lifetimes_residential.coords["Type"].values]
    xr_lifetimes_residential.coords["Type"] = fixed_coords
    xr_lifetimes_residential = merge_dims(xr_lifetimes_residential, "Type", "Area")
    xr_lifetimes_residential = xr_lifetimes_residential.transpose("Time", "Region", "Type", "Parameter")
    lifetimes_array = xr.concat((xr_lifetimes_commercial, xr_lifetimes_residential), dim="Type")
    return convert_lifetimes_buildings(lifetimes_array, distribution_type)

def convert_lifetimes_buildings(lifetimes, distribution_type="folded_norm"):
    if distribution_type == "folded_norm":
        c = lifetimes.sel(Parameter="Scale")/lifetimes.sel(Parameter="Shape")
        scale = lifetimes.sel(Parameter="Scale")
        loc = 0
    elif distribution_type == "weibull":
        c = lifetimes.sel(Parameter="Shape")
        scale = lifetimes.sel(Parameter="Scale")
        loc = 0
    scipy_lifetimes = xr.concat((c, scale), dim="ScipyParam")
    scipy_lifetimes.attrs["loc"] = loc
    scipy_lifetimes.coords["ScipyParam"] = ["c", "scale"]
    scipy_lifetimes.coords["Region"] = [str(x) for x in scipy_lifetimes.coords["Region"].values]
    return xr.Dataset({distribution_type: scipy_lifetimes.drop("Parameter").transpose(
        "Time", "Region", "Type", "ScipyParam", transpose_coords=True)})

