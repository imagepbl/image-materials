"""Population related functions for computing the building requirements."""

from importlib.resources import files
from pathlib import Path

import pandas as pd
import prism
import xarray as xr

from imagematerials.read_mym import read_mym_df

prism.unit_registry.load_definitions(files("imagematerials") / "units.txt")

def compute_population(image_directory: Path, base_directory: Path) -> xr.DataArray:
    """Compute the population throught time for all regions, total, urban and rural.

    Parameters
    ----------
    image_directory:
        Directory that contains the image data used for preprocessing.
    base_directory:
        Directory specific for the buildings preprocessing.

    Returns
    -------
    all_population:
        Population data for all regions, subdivided over region, area (rural or urban) and time.

    """
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


def compute_total_population(image_directory: Path, base_directory: Path) -> xr.DataArray:
    """Compute the population throught time for all regions, not subdivided over rural/urban.

    Parameters
    ----------
    image_directory:
        Directory that contains the image data used for preprocessing.
    base_directory:
        Directory specific for the buildings preprocessing.

    Returns
    -------
    regionalized_total_pop_history_future_xr:
        Population data for all regions, subdivided over region, area (rural or urban) and time.
    regionalized_total_pop_history_future:
        Same as regionalized_total_pop_history_future_xr, but a pandas dataframe.

    """
    # import total population 1971 - 2100 from IMAGE
    population_1971_future_df: pd.DataFrame = read_mym_df(image_directory.joinpath("Socioeconomic",
                                                                                   "pop.scn"))
    population_1971_future_df = population_1971_future_df.loc[:, :26]
    population_1971_future_df.columns = population_1971_future_df.columns.astype(str)

    # read in historic global population from Maddison Project Database 2020 & 1700 value from
    # https://www.johnstonsarchive.net/other/worldpop.html in 1000 people
    historic_pop = pd.read_csv(base_directory / 'buildings' / 'standard_data'
                               / 'historic_population.csv', index_col=0, header = 0)
    historic_pop = historic_pop.loc[:1971] / 1000 # unit conversion

    # interpolate for missing years
    # with cubic interpolation we see a drop at 1700
    historic_pop = historic_pop.reindex(range(historic_pop.index.min(),
                                              historic_pop.index.max() + 1)
                                        ).interpolate(method="linear")

    # regionalize data to IMAGE regions accoring to 1971 regionalization
    share_regionalization_1971 = (population_1971_future_df.loc[1971]
                                  / population_1971_future_df.loc[1971].sum())

    # create pd dataframe with regionalized total population based on shares in 1971
    regionalized_total_pop = pd.DataFrame(index=historic_pop.index,
                                          columns=share_regionalization_1971.index)

    for region in share_regionalization_1971.index:
        regionalized_total_pop[region] = historic_pop * share_regionalization_1971[region]

    # concat total population data
    regionalized_total_pop_history_future = pd.concat([regionalized_total_pop,
                                                       population_1971_future_df])

    # to xarray
    regionalized_total_pop_history_future = regionalized_total_pop_history_future.rename_axis(
        index = "Time", columns = "Region")

    regionalized_total_pop_history_future_xr = xr.DataArray(
        data=regionalized_total_pop_history_future.values,
        dims=["Time", "Region"],
        coords={"Time": regionalized_total_pop_history_future.index,
                "Region": regionalized_total_pop_history_future.columns}
    )

    return regionalized_total_pop_history_future_xr, regionalized_total_pop_history_future



def compute_rur_urb_pop(image_directory, base_directory):
    """Compute the population throught time for all regions, and rural/urban.

    Parameters
    ----------
    image_directory:
        Directory that contains the image data used for preprocessing.
    base_directory:
        Directory specific for the buildings preprocessing.

    Returns
    -------
    urb_pop_total:
        Urban population data for all regions over time.
    rural_pop_total:
        Rural population data for all regions over time.

    """
    (_, regionalized_total_pop_history_future) = compute_total_population(image_directory,
                                                                          base_directory)

    # rural population total meaning [Million]: the total of people living in rural areas
    # (over time, by region)
    rural_population: pd.DataFrame = read_mym_df(image_directory.joinpath("Socioeconomic",
                                                                          "RURPOPTOT.out"))
    rural_population.columns = rural_population.columns.astype(str)

    # urban population total meaning [Million]: the total of people living in urban areas
    # (over time, by region)
    urban_population: pd.DataFrame = read_mym_df(image_directory.joinpath("Socioeconomic",
                                                                          "URBPOPTOT.out"))
    urban_population.columns = urban_population.columns.astype(str)

    # remove emty and global region
    rural_population = rural_population.loc[:, :"26"]
    urban_population = urban_population.loc[:, :"26"]

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
