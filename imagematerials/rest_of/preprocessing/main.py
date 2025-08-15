import pandas as pd
import xarray as xr

from pathlib import Path

from imagematerials.read_mym import read_mym_df
from imagematerials.buildings.preprocessing.population import compute_population


def read_gompertz_values(base_directory):

    xr_gompertz = xr.open_dataset(base_directory / "rest-of" / "gompertz_values" / "coefs_gompertz.nc", engine="netcdf4")

    # make this dataset an xarray DataArray
    gompertz_coefs_xr = xr.DataArray(
        xr_gompertz['__xarray_dataarray_variable__'].values,
        dims=('Region', 'material', 'coef'),
        coords={
            'Region': list(map(str, xr_gompertz['Region'].values)),
            'material': [str(m) for m in xr_gompertz['material'].values],
            'coef': ['a', 'b', 'c']
    }
)

    return gompertz_coefs_xr


def read_image_gdp_cap_data(image_scenario_directory):

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
    # gdp_per_capita_xr.coords["Region"] = [str(x.values) for x in gdp_per_capita_xr.coords["Region"]]
    gdp_per_capita_xr.coords["Region"]  = [str(x.values) for x in gdp_per_capita_xr.coords["Region"]]

    return gdp_per_capita_xr


def rest_of_preprocessing(base_directory, image_scenario_directory):
    gompertz_values = read_gompertz_values(base_directory)
    gdp_per_capita = read_image_gdp_cap_data(image_scenario_directory)
    population = compute_population(image_scenario_directory, base_directory)
    # Filter population data to start from 1971 & only total population needed
    population = population.sel(Area = 'Total').loc[1971:]
    # drop Area coords
    population = population.drop_vars('Area')



    preprocessing_dict = {
        "gompertz_coefs": gompertz_values,
        "gdp_per_capita": gdp_per_capita,
        "population": population,
    }
    return preprocessing_dict