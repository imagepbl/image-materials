from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from imagematerials.buildings.constants import SCENARIO_SELECT, YEARS
from imagematerials.util import dataset_to_array

far_start_year = 1721
start_year = 1820
end_year = 1970

years_1721_1820 = xr.DataArray(np.arange(far_start_year, start_year), dims=["Year"], coords={"Year": np.arange(far_start_year, start_year)})
years_1820_1970 = xr.DataArray(np.arange(start_year, end_year), dims=["Year"], coords={"Year": np.arange(start_year, end_year)})

def compute_population(base_directory = Path("..", "IMAGE-Mat_old_version", "IMAGE-Mat", "BUMA")):

    # Set the correct directories
    database_directory = base_directory / "files_DB" / SCENARIO_SELECT
    image_directory = base_directory / "files_IMAGE" / SCENARIO_SELECT
    assert database_directory.is_dir(), database_directory
    assert image_directory.is_dir()

    # Compute total/rural/urban populations
    tot_population = compute_total_population(image_directory, base_directory)
    rurpop_share, urbpop_share = compute_rurpop_share(image_directory)
    
    # Merge into one xarray DataArray.
    all_population = xr.concat((tot_population, rurpop_share*tot_population, urbpop_share*tot_population),
                               dim="Area")
    all_population = all_population.assign_coords({"Area": ["Total", "Rural", "Urban"]})
    all_population = all_population.transpose("Year", "Region", "Area")

    return all_population


def compute_total_population(image_directory, base_directory):
    # Pop; unit: million of people; meaning: global population (over time, by region)             
    population_1970_future_df: pd.DataFrame = pd.read_csv(image_directory.joinpath('pop.csv'), index_col = [0])

    # initial population as a percentage of the 1970 population; unit: %; according to the Maddison Project Database (MPD) 2018 (Groningen University)
    historic_population_df = pd.read_csv(base_directory / 'files_initial_stock' /'hist_pop.csv', index_col = [0])  

    population_1970_future_df = population_1970_future_df.reindex(YEARS).interpolate(method='cubic')
    population_1970_future_df = population_1970_future_df.rename_axis(index = "Year", columns = "Region")

    # Deriving historic population tail based on fraction for 1970
    population_1970 = population_1970_future_df.loc[1970]
    historic_population = historic_population_df.multiply(population_1970, axis=1)

    population_1820_future_df = pd.concat([historic_population.loc[:1969], population_1970_future_df])
    population_1820_future_df = population_1820_future_df.rename_axis(index = "Year", columns = "Region")


    population_1820_future = xr.DataArray(
        data=population_1820_future_df.values,                # Data values from the DataFrame
        dims=["Year", "Region"],                  # Names for the two dimensions
        coords={"Year": population_1820_future_df.index,      # Year coordinates from the DataFrame index
                "Region": population_1820_future_df.columns}  # Region coordinates from the DataFrame columns
    )

    # Extrapolate population to zero from 1720 (population at 1721 > 0)
    population_1721_1820 = (population_1820_future[0]*(1-(start_year - years_1721_1820)/(start_year-far_start_year+1))).transpose()
    population = xr.concat((population_1721_1820, population_1820_future), dim="Year")

    return population

    

def compute_rurpop_share(image_directory):
    # rurpop; unit: %; meaning: the share of people living in rural areas (over time, by region)
    # TODO: seems to be a fraction, so no % units, check!
    raw_rural_population: pd.DataFrame = pd.read_csv(image_directory. joinpath('rurpop.csv'), index_col = [0])

    # Interpolate population and rural population data (fills in missing years with cubic interpolation)
    rural_population = raw_rural_population.reindex(YEARS).interpolate(method='cubic')

    maximum_rural_population = rural_population.values.max()

    # Get the rural population share from 1970-future directly from the data
    rurpop_share_1970_future = dataset_to_array(rural_population.to_xarray(), ["Year"], ["Region"])

    # Get the growth or rural population by year (average for the first 10 years of IMAGE data)
    # TODO: This trend computation seems wrong, find yearly increase with log instead.
    # TODO: Also there is unneccessary multiplication by 100
    rural_population_trend = ((1-rural_population.loc[1980]/rural_population.loc[1970])/10)*100
    rurpop_trend_xr = rural_population_trend.to_xarray().rename({"index": "Region"})
    rurpop_share_1820_1970 = (rurpop_share_1970_future.loc[1970] * ((100+rurpop_trend_xr)/100)**(1970-years_1820_1970)).transpose()
    rurpop_share_1820_1970 = rurpop_share_1820_1970.where(rurpop_share_1820_1970 <= maximum_rural_population,
                                                          maximum_rural_population)

    # Compute the rural population share for the first part.
    # TODO: Rural population share goes to zero (took it from the original), but probably should go to one.
    rurpop_share_1721_1820 = 0*years_1721_1820 + rurpop_share_1820_1970.loc[1820]
    # rurpop_share_1721_1820 = rurpop_share_1820_1970.loc[1820] - (rurpop_share_1820_1970.loc[1820])*(1820-years_1721_1820)/100
    # rurpop_share_1721_1820 = rurpop_share_1721_1820.transpose()
    # rurpop_share_1721_1820 = rurpop_share_1721_1820.where(rurpop_share_1721_1820 >= 0, 0)

    # Concatenate timeline together
    # TODO: I think there is a bug in the old code where urbpop_share + rurpop_share != 1, check this.
    rurpop_share = xr.concat((rurpop_share_1721_1820, rurpop_share_1820_1970, rurpop_share_1970_future), dim="Year")
    urbpop_share = 1-rurpop_share

    # TODO: Remove this BUG
    rurpop_share.loc[1721:1819] = rurpop_share.loc[1721:1819] * (years_1721_1820-1720)/100
    urbpop_share.loc[1721:1819] = urbpop_share.loc[1721:1819] * (years_1721_1820-1720)/100

    return rurpop_share, urbpop_share
