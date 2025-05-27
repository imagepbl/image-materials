from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from imagematerials.buildings.constants import (
    far_start_year, 
    start_year,
    end_year, 
    global_pop_1700,
    global_pop_1820, 
    known_years,
    full_years_pop)

from imagematerials.util import dataset_to_array
from imagematerials.read_mym import read_mym_df

years_1721_1820 = xr.DataArray(np.arange(far_start_year, start_year), dims=["Time"], coords={"Time": np.arange(far_start_year, start_year)})
years_1820_1970 = xr.DataArray(np.arange(start_year, end_year), dims=["Time"], coords={"Time": np.arange(start_year, end_year)})

def compute_population(image_directory, base_directory):
    # Compute total/rural/urban populations
    tot_population = compute_total_population(image_directory, base_directory)
    rurpop_share, urbpop_share = compute_rurpop_share(image_directory)
    
    # Merge into one xarray DataArray.
    all_population = xr.concat((tot_population, rurpop_share*tot_population, urbpop_share*tot_population),
                               dim="Area")
    all_population = all_population.assign_coords({"Area": ["Total", "Rural", "Urban"]})
    all_population = all_population.transpose("Time", "Region", "Area")

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

    # Extrapolate population to zero from 1720 (population at 1721 > 0)
    population_1721_1820 = (population_1820_future[0]*(1-(start_year - years_1721_1820)/(start_year-far_start_year+1))).transpose()
    population = xr.concat((population_1721_1820, population_1820_future), dim="Time")

    return population, population_1970_future_df

    

def compute_rur_urb_pop(image_directory, population_1970_future_df):
    #rural population total meaning [Million]: the total of people living in rural areas (over time, by region)
    rural_population: pd.DataFrame = read_mym_df(image_directory.joinpath("Socioeconomic", "RURPOPTOT.out"))

    #urban population total meaning [Million]: the total of people living in urban areas (over time, by region)
    urban_population: pd.DataFrame = read_mym_df(image_directory.joinpath("Socioeconomic", "URBPOPTOT.out"))

    # remove emty and global region
    rural_population = rural_population.loc[:, :26]
    urban_population = urban_population.loc[:, :26]

    # extrapolate to history:
    # Calculate each region's share of global population in the base year
    regional_shares = population_1970_future_df.loc[1971]/population_1970_future_df.loc[1971].sum()

    # Calculate total population for 1971 and regional shares
    regional_pop_1971 = population_1970_future_df.loc[1971]
    regional_shares = regional_pop_1971 / regional_pop_1971.sum()

    # Urban share in 1971 per region (percent)
    urban_share_1971 = (urban_population.loc[1971] / regional_pop_1971) * 100

    # Urban share known values for each region
    urban_share_known = pd.DataFrame(index=known_years, columns=regional_shares.index)

    for region in regional_shares.index:
        urban_share_known.loc[1700, region] = 0.0
        urban_share_known.loc[1820, region] = 7.2 #https://www2.census.gov/programs-surveys/decennial/1990/tables/cph-2/table-4.pdf
        urban_share_known.loc[1971, region] = urban_share_1971[region]

    # Total global population known values
    total_global_pop_known = pd.Series(data=[global_pop_1700, global_pop_1820, regional_pop_1971.sum()], index=known_years)

    # Interpolate urban share for all years and regions
    urban_share_interp = pd.DataFrame(index=full_years_pop, columns=regional_shares.index, dtype=float)

    urban_share_known = urban_share_known.astype(float)
    for region in regional_shares.index:
        urban_share_interp[region] = np.interp(full_years_pop, known_years, urban_share_known[region])

    # Calculate rural share = 100 - urban share
    rural_share_interp = 100 - urban_share_interp

    # Interpolate total global population for all years
    total_global_pop_interp = np.interp(full_years_pop, known_years, total_global_pop_known)

    # Calculate total population per region for all years by scaling with constant regional shares

    total_pop_interp = pd.DataFrame(index=full_years_pop, columns=regional_shares.index, dtype=float)
    for year_idx, year in enumerate(full_years_pop):
        total_pop_interp.loc[year] = regional_shares * total_global_pop_interp[year_idx]

    # Calculate urban and rural populations per region and year
    urban_pop_interp = total_pop_interp * (urban_share_interp / 100)
    rural_pop_interp = total_pop_interp * (rural_share_interp / 100)

    return rural_pop_interp, urban_pop_interp
