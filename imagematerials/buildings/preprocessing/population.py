from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import prism
from importlib.resources import files


from imagematerials.buildings.constants import (
    urban_share_1820)

from imagematerials.buildings.constants import (
    YEARS, 
    far_start_year, 
    start_year,
    end_year)

from imagematerials.util import dataset_to_array
from imagematerials.read_mym import read_mym_df

years_1721_1820 = xr.DataArray(np.arange(far_start_year, start_year), dims=["Time"], coords={"Time": np.arange(far_start_year, start_year)})
years_1820_1970 = xr.DataArray(np.arange(start_year, end_year), dims=["Time"], coords={"Time": np.arange(start_year, end_year)})

def compute_population(image_directory, base_directory):
    # Compute total/rural/urban populations
    tot_population_xr, _ = compute_total_population(image_directory, base_directory)
    urbpop_total, rurpop_total = compute_rur_urb_pop(image_directory, base_directory)

    #TODO: use function from util if possible? problem: extra_dims?
    rurpop_total_xr = xr.DataArray(
    data=rurpop_total.values,                # Data values from the DataFrame
    dims=["Time", "Region"],                 # Names for the two dimensions
    coords={"Time": rurpop_total.index,      # Time coordinates from the DataFrame index
            "Region": rurpop_total.columns}  # Region coordinates from the DataFrame columns
        )

    urbpop_total_xr = xr.DataArray(
    data=urbpop_total.values,                # Data values from the DataFrame
    dims=["Time", "Region"],                 # Names for the two dimensions
    coords={"Time": urbpop_total.index,      # Time coordinates from the DataFrame index
            "Region": urbpop_total.columns}  # Region coordinates from the DataFrame columns
        )

    all_population = xr.concat((tot_population_xr, rurpop_total_xr, urbpop_total_xr), dim="Area")
    all_population = all_population.assign_coords({"Area": ["Total", "Rural", "Urban"]})
    all_population = all_population.transpose("Time", "Region", "Area")
    all_population = all_population * 1e6 # data from TIMER comes in million persons
    all_population = prism.Q_(all_population, "person")

    return all_population


def compute_total_population(image_directory, base_directory):

    # import total population 1971 - 2100 from IMAGE
    population_1970_future_df: pd.DataFrame = read_mym_df(image_directory.joinpath("Socioeconomic", "pop.scn"))
    # drop empty and global region
    population_1970_future_df = population_1970_future_df.loc[:, :26]

    # initial population as a percentage of the 1970 population; unit: %; according to the Maddison Project Database (MPD) 2018 (Groningen University)
    historic_population_df = pd.read_csv(base_directory / 'buildings' / 'files_initial_stock' /'hist_pop.csv', index_col = [0])  
    historic_population_df.columns = historic_population_df.columns.astype(int) # for multiplication both indexes must be int

    population_1970_future_df = population_1970_future_df.rename_axis(index = "Time", columns = "Region")

    # Deriving historic population tail based on fraction for 1971
    population_1971 = population_1970_future_df.loc[1971]

    # multiply shares from historic populatin with last year from IMAGE data 
    historic_population = historic_population_df.multiply(population_1971, axis=1)

    population_1820_future_df = pd.concat([historic_population.loc[:1970], population_1970_future_df])
    population_1820_future_df = population_1820_future_df.rename_axis(index = "Time", columns = "Region")

    population_1820_future = xr.DataArray(
        data=population_1820_future_df.values,                # Data values from the DataFrame
        dims=["Time", "Region"],                  # Names for the two dimensions
        coords={"Time": population_1820_future_df.index,      # Time coordinates from the DataFrame index
                "Region": population_1820_future_df.columns}  # Region coordinates from the DataFrame columns
    )


    return regionalized_total_pop_history_future_xr, regionalized_total_pop_history_future 

    

def compute_rurpop_share(image_directory):
    # rurpop; unit: %; meaning: the share of people living in rural areas (over time, by region)
    # TODO: seems to be a fraction, so no % units, check!

    # total 

    raw_rural_population: pd.DataFrame = pd.read_csv(image_directory. joinpath('rurpop.csv'), index_col = [0])

    #rural population total meaning [Million]: the total of people living in rural areas (over time, by region)
    rural_population: pd.DataFrame = read_mym_df(image_directory.joinpath("Socioeconomic", "RURPOPTOT.out"))
    rural_population.columns = rural_population.columns.astype(str)

    #urban population total meaning [Million]: the total of people living in urban areas (over time, by region)
    urban_population: pd.DataFrame = read_mym_df(image_directory.joinpath("Socioeconomic", "URBPOPTOT.out"))
    urban_population.columns = urban_population.columns.astype(str)

    # remove emty and global region
    rural_population, urban_population = rural_population.loc[:, :"26"], urban_population.loc[:, :"26"]

    # split up in rural and urban
    # get urban share for base year
    urban_share = urban_population/regionalized_total_pop_history_future
    urban_share.loc[1700] = 0
    # interpolate urban share
    urban_share = urban_share.interpolate()
    rural_share = 1 - urban_share

    urban_pop_total = urban_share*regionalized_total_pop_history_future
    rural_pop_total = rural_share*regionalized_total_pop_history_future
    
    urban_pop_total = urban_pop_total.rename_axis(index = "Time", columns = "Region")
    rural_pop_total = rural_pop_total.rename_axis(index = "Time", columns = "Region")
    
    return urban_pop_total, rural_pop_total
