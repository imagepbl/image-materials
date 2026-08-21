import pandas as pd
import numpy as np
import xarray as xr
import prism

from pathlib import Path

from imagematerials.constants import IMAGE_REGIONS
from imagematerials.concepts import create_region_graph

from imagematerials.factory import ModelFactory
from imagematerials.read_mym import read_mym_df
from imagematerials.buildings.preprocessing.population import compute_population

from imagematerials.rest_of.preprocessing.regressions_all_materials import (fit_models_all_materials, get_X_max_scaling_factor,
                                                                            make_gompertz_coefs_da, 
                                                                            mean_historic_other_fraction_consumption_to_xr, 
                                                                            historic_other_fraction_consumption_to_xr)

from imagematerials.rest_of.util import sum_inflows_for_all_sectors, save_sum_as_csv
from imagematerials.util import flag_enabled, read_resource_efficiency_flags

def read_gompertz_values(base_directory, scenario: str, resource_efficiency_flags: dict = None):
    """
    Reads Gompertz coefficient values from a NetCDF file and adapts them according to the specified scenario.

    This function loads the Gompertz coefficients from the file
    '<base_directory>/rest-of/gompertz_values/coefs_gompertz.nc', converts them to an xarray DataArray,
    and sorts the regions numerically. The coefficients can be adapted using the 'adapt_gompertz_regional' function for resource efficiency measures between 2030 and 2050.

    Parameters
    ----------
    base_directory : Path or str
        The base directory containing the 'rest-of/gompertz_values/coefs_gompertz.nc' file.
    scenario : str
        The scenario name.
    resource_efficiency_flags : dict, optional
        Nested dictionary as returned by read_resource_efficiency_flags. If the
        'rest_of' sector's 'FlagResourceEfficiency' flag is enabled, the Gompertz
        coefficients adapted for resource efficiency measures are used.

    Returns
    -------
    xr.DataArray
        The Gompertz coefficients as an xarray DataArray, possibly adapted for the scenario.

    Notes
    -----
    - The returned DataArray has dimensions including 'coef', 'material', 'Region', and 'time'.
    - Only the 'a' coefficient is adapted for resource efficiency scenarios.
    - The function expects regions to be numeric strings for sorting.
    """

    if flag_enabled(resource_efficiency_flags, "rest_of", "FlagResourceEfficiency"):
        print('Using Gompertz coefficients for resource efficiency measures')
        name = "coefs_gompertz_eff.nc"
    else:
        name = "coefs_gompertz.nc"

    xr_gompertz = xr.open_dataset(base_directory / "rest-of" / "gompertz_values" / name, engine="netcdf4")
    xr_gompertz = xr_gompertz.to_array().isel(variable=0).drop_vars("variable")

    return xr_gompertz


def read_historic_diff_cons_data_mean(base_directory):

    diff_consumption_mean = xr.open_dataset(base_directory / "rest-of" / "gompertz_values" / "diff_cons_all_mean.nc", engine="netcdf4")
    diff_consumption_mean = diff_consumption_mean.to_array().isel(variable=0).drop_vars("variable")
    diff_consumption_mean = prism.Q_(diff_consumption_mean, 't')

    return diff_consumption_mean


def read_historic_diff_cons_data(base_directory):

    diff_consumption = xr.open_dataset(base_directory / "rest-of" / "gompertz_values" / "diff_cons_all.nc", engine="netcdf4")
    diff_consumption = diff_consumption.to_array().isel(variable=0).drop_vars("variable")
    diff_consumption = prism.Q_(diff_consumption, 't')

    return diff_consumption


def read_image_gdp_cap_data(base_directory, image_scenario_directory):
    max_x = xr.open_dataset(base_directory / "rest-of" / "gompertz_values" / "max_x_regressor.nc", engine="netcdf4")
    max_x = max_x.to_array().isel(variable=0).drop_vars("variable")
    image_directory = Path(image_scenario_directory)
    gdp_per_capita: pd.DataFrame = read_mym_df(image_directory.joinpath("Socioeconomic", "gdp_pc.scn"))

    # to xarry
    # drop empty and global region
    gdp_per_capita = gdp_per_capita.loc[:, :26]
    gdp_per_capita = gdp_per_capita.rename_axis(index = "time", columns = "Region")

    gdp_per_capita_xr = xr.DataArray(
        data=gdp_per_capita.values,                # Data values from the DataFrame
        dims=["time", "Region"],            # Names for the two dimensions
        coords={"time": gdp_per_capita.index,      # time coordinates from the DataFrame index
                "Region": gdp_per_capita.columns}  # Region coordinates from the DataFrame columns
    )
    gdp_per_capita_xr.coords["Region"]  = [str(x.values) for x in gdp_per_capita_xr.coords["Region"]]
    knowledge_graph_region = create_region_graph()
    gdp_per_capita_xr = knowledge_graph_region.rebroadcast_xarray(gdp_per_capita_xr, output_coords=IMAGE_REGIONS, dim="Region")

    downscaled_gdp_per_capita_xr = gdp_per_capita_xr/max_x # downscale gdp to avoid numerical issues
    return downscaled_gdp_per_capita_xr


def fit_all_materials_save_corrseponding_input_data(path_input_data, path_input_data_image):
    results, regions_grouping = fit_models_all_materials(
        path_input_data=path_input_data,
        path_input_data_image=path_input_data_image
        )
    gompertz = make_gompertz_coefs_da(results)
    mean_historic_other_fraction_consumption_to_xr(results)
    all_historic_data_xr = historic_other_fraction_consumption_to_xr(results)
    max_x = get_X_max_scaling_factor(results, save=True)


def rest_of_preprocessing(base_directory, image_scenario_directory, scenario: str,
                          resource_efficiency_flags_file=None, refit = False):

    if refit == True:
        fit_all_materials_save_corrseponding_input_data()
        print('Materials regression refitted and preprocessing data updated.')

    resource_efficiency_flags = read_resource_efficiency_flags(resource_efficiency_flags_file)
    gompertz_values = read_gompertz_values(base_directory, scenario, resource_efficiency_flags)
    gdp_per_capita = read_image_gdp_cap_data(base_directory, image_scenario_directory)
    historic_diff_consumption_mean = read_historic_diff_cons_data_mean(base_directory)
    historic_diff_consumption_total = read_historic_diff_cons_data(base_directory)
    population = compute_population(image_scenario_directory, base_directory)
    # Filter to total population from 1971 onward.
    population = population.sel(Area="Total", time=slice(1971, None)).drop_vars("Area")
    # Total is duplicated across Quintile; collapse to a single time x Region series.
    if "Quintile" in population.dims:
        population = population.mean("Quintile")
    # from nr to Region abbreviations
    knowledge_graph_region = create_region_graph()
    population = knowledge_graph_region.rebroadcast_xarray(population, output_coords=IMAGE_REGIONS, dim="Region")

    preprocessing_dict = {
        "gompertz_coefs": gompertz_values,
        "gdp_per_capita": gdp_per_capita,
        "population": population,
        "historic_diff_consumption_mean": historic_diff_consumption_mean,
        "historic_diff_consumption_total": historic_diff_consumption_total
    }
    
    return preprocessing_dict