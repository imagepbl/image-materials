import pandas as pd
import numpy as np
import xarray as xr
import prism

from pathlib import Path

from imagematerials.concepts import create_class_region_graph


from imagematerials.constants import IMAGE_REGIONS
from imagematerials.concepts import create_region_graph

from imagematerials.read_mym import read_mym_df
from imagematerials.buildings.preprocessing.population import compute_population

from imagematerials.rest_of.const import IAI_TO_IMAGE_CLASSES
from imagematerials.rest_of.preprocessing.resource_efficiency_measures import adapt_gompertz_regional

from imagematerials.rest_of.preprocessing.regressions_all_materials import (fit_models_all_materials,
                                                                            make_gompertz_coefs_da, 
                                                                            mean_historic_other_fraction_consumption_to_xr, 
                                                                            historic_other_fraction_consumption_to_xr)

from imagematerials.rest_of.util import sum_inflows_for_all_sectors, save_sum_as_csv


def sum_and_save(model_run, list_sum_sectors: list, save_path: Path, mfa_data = 'inflow_materials'):
    """
    Sums inflows across specified sectors and saves the results for steel, aluminium, and copper
    Parameters
    ----------
    model_run :
        ImageMaterials model instance providing sector attributes (buildings, vehicles, generation, grid, ...)
        each exposing `.get(get_mfa_data).to_array()` returning an xarray.DataArray.
    list_sum_sectors : list
        List of sector names to include in the sum (e.g. ["buildings", "vehicles", "generation", "grid"]).
    mfa_data : str, optional
        Key/name of the MFA data to extract from each sector (default is 'inflow_materials').
        Could also be stock, or outflow_materials.
    Returns
    -------
    xarray.DataArray
        Elementwise sum of the sector inflows for the specified sectors.
    """

    total_inflow_combined = sum_inflows_for_all_sectors(model_run, mfa_data, list_sum_sectors)

    # save materials
    save_sum_as_csv(total_inflow_combined, "steel", save_path)
    save_sum_as_csv(total_inflow_combined, "aluminium", save_path)
    save_sum_as_csv(total_inflow_combined, "copper", save_path)
    save_sum_as_csv(total_inflow_combined, "cement", save_path)
    save_sum_as_csv(total_inflow_combined, "sand_gravel_crushed_rock", save_path)

    print("Materials saved successfully.")
    return total_inflow_combined


def read_gompertz_values(base_directory, scenario: str):
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
        The scenario name. If it matches 'SSP2_VLLO_LifeTech', resource efficiency adaptation is applied.

    Returns
    -------
    xr.DataArray
        The Gompertz coefficients as an xarray DataArray, possibly adapted for the scenario.

    Notes
    -----
    - The returned DataArray has dimensions including 'coef', 'material', 'Region', and 'Time'.
    - Only the 'a' coefficient is adapted for resource efficiency scenarios.
    - The function expects regions to be numeric strings for sorting.
    """

    if scenario in ["SSP2_VLLO_LifeTech", "SSP2_narrow_activity","SSP2_narrow", "SSP2_narrow_slow_close", 
                    "SSP2_narrow_act_26_tax","SSP2_narrow_26_tax", "SSP2_narrow_slow_close_26_tax",
                    "SSP2_narrow_slow_close_19_tax"]:
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

# TODO: read in max X data


def read_image_gdp_cap_data(base_directory, image_scenario_directory):
    max_x = xr.open_dataset(base_directory / "rest-of" / "gompertz_values" / "max_x_regressor.nc", engine="netcdf4")
    max_x = max_x.to_array().isel(variable=0).drop_vars("variable")
    image_directory = Path(image_scenario_directory)
    gdp_per_capita: pd.DataFrame = read_mym_df(image_directory.joinpath("Socioeconomic", "gdp_pc.scn"))

    # to xarry
    # drop empty and global region
    gdp_per_capita = gdp_per_capita.loc[:, :26]
    gdp_per_capita = gdp_per_capita.rename_axis(index = "Time", columns = "Region")

    gdp_per_capita_xr = xr.DataArray(
        data=gdp_per_capita.values,                # Data values from the DataFrame
        dims=["Time", "Region"],            # Names for the two dimensions
        coords={"Time": gdp_per_capita.index,      # Time coordinates from the DataFrame index
                "Region": gdp_per_capita.columns}  # Region coordinates from the DataFrame columns
    )
    gdp_per_capita_xr.coords["Region"]  = [str(x.values) for x in gdp_per_capita_xr.coords["Region"]]
    knowledge_graph_region = create_region_graph()
    gdp_per_capita_xr = knowledge_graph_region.rebroadcast_xarray(gdp_per_capita_xr, output_coords=IMAGE_REGIONS, dim="Region")

    downscaled_gdp_per_capita_xr = gdp_per_capita_xr/max_x # downscale gdp to avoid numerical issues
    return downscaled_gdp_per_capita_xr


def rest_of_preprocessing(base_directory, image_scenario_directory, scenario: str, 
                          refit = False):
    
    if refit == True:
        results = fit_models_all_materials()
        make_gompertz_coefs_da(results)
        mean_historic_other_fraction_consumption_to_xr(results)
        historic_other_fraction_consumption_to_xr(results)
        print('Materials regression refitted and preprocessing data updated.')

    gompertz_values = read_gompertz_values(base_directory, scenario)
    gdp_per_capita = read_image_gdp_cap_data(base_directory, image_scenario_directory)
    historic_diff_consumption_mean = read_historic_diff_cons_data_mean(base_directory)
    historic_diff_consumption_total = read_historic_diff_cons_data(base_directory)
    population = compute_population(image_scenario_directory, base_directory)
    # Filter population data to start from 1971 & only total population needed
    population = population.sel(Area = 'Total').loc[1971:]
    # drop Area coords
    population = population.drop_vars('Area')
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