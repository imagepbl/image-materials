"""Module for floorspace calculations for the buildings."""
import math
from importlib.resources import files
from pathlib import Path

import numpy as np
import pandas as pd
import prism
import xarray as xr

from imagematerials.buildings.constants import (
    END_YEAR,
    FLAG_ALPHA,
    FLAG_EXPDEC,
    GOMPERTZ_EXPDEC,
    HIST_YEAR,
    INFLATION,
    REGIONS,
    REGIONS_RANGE,
    START_YEAR,
    YEARS,
    urban_q_areas,
    rural_q_areas,
    area_graph,
    commercial_types
)
from imagematerials.buildings.preprocessing.circular_economy_measures import (
    ce_measures_residential_housing,
)
from imagematerials.read_mym import read_mym_df
from imagematerials.util import dataset_to_array, merge_dims

prism.unit_registry.load_definitions(files("imagematerials") / "units.txt")

far_start_year = 1721
start_year = 1820
end_year = 1970
idx = pd.IndexSlice

years_1721_1820 = xr.DataArray(np.arange(far_start_year, start_year), dims=["Time"],
                               coords={"Time": np.arange(far_start_year, start_year)})
years_1820_1971 = xr.DataArray(np.arange(start_year, end_year+1), dims=["Time"],
                               coords={"Time": np.arange(start_year, end_year+1)})
years_1820_1970 = xr.DataArray(np.arange(start_year, end_year), dims=["Time"],
                               coords={"Time": np.arange(start_year, end_year)})


def get_gompertz(base_directory: Path) -> pd.DataFrame:
    """Get the gompertz parameters from a file.

    The filename is determined by the FLAG_ALPHA constant.

    Parameters
    ----------
    base_directory:
        Base directory under which the gompertz parameters should be stored.

    Returns
    -------
    gompertz_parameters:
        The parameters of the gompertz curve.

    """
    # Load fitted regression parameters
    if FLAG_ALPHA == 0:
        gompertz = pd.read_csv(base_directory / "buildings" / "files_commercial" /
                               "Gompertz_parameters.csv", index_col = [0])
    else:
        gompertz = pd.read_csv(base_directory / "buildings" / "files_commercial" /
                               "Gompertz_parameters_alpha.csv", index_col = [0])
    return gompertz

def get_service_value_added(image_directory: Path) -> pd.DataFrame:
    """Load and interpolate service value added data.

    We use the inflation corrected SVA to adjust for the fact
    that IMAGE provides gdp/cap in 2005 US$

    Parameters
    ----------
    image_directory:
        Base directory for the image files.

    """
    # TODO: check with IMAGE team if SVA is still given in 2005 values
    # --> if not correct inflation factor
    service_value_added_2005: pd.DataFrame = read_mym_df(image_directory.joinpath("Socioeconomic",
                                                                                  "sva_pc.scn"))
    service_value_added = service_value_added_2005 * INFLATION

    # # extrapolate to 1970, therefore first add empty 1970 value with nans
    service_value_added.loc[1970] = np.nan
    # TODO cubic does not work, replaced with linear for now, Sebastiaans code had cubic,
    # seems like now just value from 1971 is copied
    service_value_added = service_value_added.sort_index().interpolate(method='linear',
                                                                       limit_direction="both")

    return service_value_added

def get_image_floorspace(image_directory: Path,
                         base_directory: Path) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Get the commercial and residential floorspace computed by image.

    Parameters
    ----------
    image_directory:
        Base directory of the image files.
    base_directory:
        Base directory of all the files.

    Returns
    -------
    floorspace_residential:
        XArray DataArray with dims ``(Time, Region, Area, Quintile)`` for residential buildings.
    floorspace_commercial:
        XArray DataArray with dims ``(Time, Region, Type)`` for commercial buildings.
    minimum_comm:
        Computed minimum floorspace for commercial buildings.

    """
    gompertz = get_gompertz(base_directory)
    service_value_added = get_service_value_added(image_directory)
    commercial_m2_cap_sum = compute_commercial_floor_m2_cap_sum(gompertz, service_value_added)
    commercial_m2_cap, minimum_comm = compute_commercial_floor_m2_cap(
        gompertz, commercial_m2_cap_sum, service_value_added)

    # Residential: (Time, Region, Area, Quintile) — Area=["Urban","Rural"], Quintile=["Q1",...,"Q5"]
    floorspace_residential_xr = get_floorspace_urban_rural(image_directory)

    # Commercial: convert DataFrame (Time×Region) × Type → xarray (Time, Region, Type)
    floorspace_commercial_xr = (
        commercial_m2_cap
        .rename_axis("Type", axis=1)
        .stack()
        .to_xarray()
        .transpose("Time", "Region", "Type")
    )
    floorspace_commercial_xr.coords["Region"] = [
        str(r) for r in floorspace_commercial_xr.coords["Region"].values
    ]

    # Quantify units
    floorspace_residential_xr = prism.Q_(floorspace_residential_xr, "m^2/person")
    floorspace_commercial_xr = prism.Q_(floorspace_commercial_xr, "m^2/person")
    minimum_comm = prism.Q_(minimum_comm, "m^2/person")

    return floorspace_residential_xr, floorspace_commercial_xr, minimum_comm

def extrapolate_floorspace(floorspace_image: xr.DataArray,
                           minimum_values: xr.DataArray | None = None) -> xr.DataArray:
    """Extrapolate the floorspace back to 1721.

    Derive the annual trend (in m2/cap) over the initial 10 years of IMAGE data and
    extrapolate backwards to 1721.

    Parameters
    ----------
    floorspace_image:
        The non-extrapolated floorspace values. Can be either:
        ``(Time, Region, Area, Quintile)`` for residential, or
        ``(Time, Region, Type)`` for commercial.
    minimum_values:
        Optional lower-bound values for non-time/region dimensions.
        For commercial this is typically ``minimum_comm``.

    """
    interp_coor = floorspace_image.sel(Time=range(1971, 1981)).coords
    trend_1971_1981 = xr.DataArray(
        floorspace_image.sel(Time=range(1971, 1981)).to_numpy()/floorspace_image.sel(
            Time=range(1972, 1982)).to_numpy(),
        dims=floorspace_image.dims,
        coords=interp_coor,
    )

    # Average global annual trend across regions and first IMAGE decade
    avg_trend_1971_1981 = trend_1971_1981.mean(["Time", "Region"])

    # Determine lower bounds from IMAGE minima and optional explicit minimum values.
    min_floorspace = floorspace_image.min(["Time", "Region"])
    if (minimum_values is not None
            and set(minimum_values.dims).issubset(set(min_floorspace.dims))):
        # Align labels/order explicitly to avoid join='exact' alignment errors in xarray.where.
        minimum_aligned = minimum_values.reindex(
            {dim: min_floorspace.coords[dim].values for dim in minimum_values.dims}
        )
        min_floorspace = xr.where(
            min_floorspace > minimum_aligned,
            min_floorspace,
            minimum_aligned,
        )

    # Compute the floorspace between 1820 and 1970 with extrapolation
    floor_1820_1970 = floorspace_image.loc[1971]*avg_trend_1971_1981**(end_year+1-years_1820_1971)
    floor_1820_1970 = floor_1820_1970.where(floor_1820_1970 > min_floorspace, min_floorspace)

    # Build far historic segment from the first extrapolated year (1820) while preserving dims.
    early_scale = 1 - (start_year - years_1721_1820) / (start_year - far_start_year + 1)
    floor_1721_1820 = floor_1820_1970.sel(Time=start_year) * early_scale

    # Avoid duplicate 1970 at the historical/IMAGE boundary; keep the IMAGE value.
    if 1970 in floorspace_image.coords["Time"].values:
        floor_1820_1970 = floor_1820_1970.sel(Time=floor_1820_1970.coords["Time"] != 1970)
    # combine historic with IMAGE data here
    floorspace = xr.concat((floor_1721_1820, floor_1820_1970, floorspace_image), dim="Time")
    return floorspace.transpose(*floorspace_image.dims)


def get_floorspace_urban_rural(image_directory: Path) -> xr.DataArray:
    """Load the residential floorspace from a file and restructure into xarray.

    Parameters
    ----------
    image_directory:
        Base image directory to load the file from.

    Returns
    -------
    floorspace:
        DataArray with dims ``(Time, Region, Area, Quintile)`` where
        ``Area=["Urban","Rural"]`` and ``Quintile=["Q1",...,"Q5"]``.
        Units not yet quantified (m^2/person).

    """
    # load IMAGE data-files (MyM file format)
    floorspace: pd.DataFrame = read_mym_df(image_directory.joinpath("EnergyServices",
                                                                    "res_FloorSpace.out"))
    floorspace = floorspace[['time', 'DIM_1', 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]].rename(
        columns={"DIM_1": "Region", "time": "Time",
                 2: 'Urban',    3: 'Rural',
                 4: 'Urban Q1', 5: 'Urban Q2', 6: 'Urban Q3', 7: 'Urban Q4', 8: 'Urban Q5',
                 9: 'Rural Q1', 10: 'Rural Q2', 11: 'Rural Q3', 12: 'Rural Q4', 13: 'Rural Q5'})
    # (column 1 / total average is excluded; we use the urban & rural specific totals)
    floorspace = floorspace[floorspace.Region != REGIONS + 1]  # removing region 27
    floorspace = floorspace[floorspace['Time'].isin(list(range(START_YEAR, END_YEAR + 1)))]
    floorspace = floorspace.set_index(["Time", "Region"])

    # Restructure into (Area, Quintile) dims — same logic as population
    q_labels = ["Q1", "Q2", "Q3", "Q4", "Q5"]
    urban_q_cols = ["Urban Q1", "Urban Q2", "Urban Q3", "Urban Q4", "Urban Q5"]
    rural_q_cols = ["Rural Q1", "Rural Q2", "Rural Q3", "Rural Q4", "Rural Q5"]

    flat_xr = (
        floorspace
        .rename_axis("Area", axis=1)
        .stack()
        .to_xarray()
    )
    flat_xr.coords["Region"] = [str(r) for r in flat_xr.coords["Region"].values]

    urban_q = (
        flat_xr.sel(Area=urban_q_cols)
        .assign_coords(Area=q_labels)
        .rename({"Area": "Quintile"})
        .expand_dims(Area=["Urban"])
    )
    rural_q = (
        flat_xr.sel(Area=rural_q_cols)
        .assign_coords(Area=q_labels)
        .rename({"Area": "Quintile"})
        .expand_dims(Area=["Rural"])
    )

    # assign units (m^2/person) here, before extrapolation
    urban_q = prism.Q_(urban_q, "m^2/person")
    rural_q = prism.Q_(rural_q, "m^2/person")
    return xr.concat([urban_q, rural_q], dim="Area").transpose("Time", "Region", "Area", "Quintile")


def compute_commercial_floor_m2_cap_sum(gompertz: dict,
                                        service_value_added: pd.DataFrame) -> pd.DataFrame:
    """Compute the commercial floorspace for all types summed.

    First the summed averages are computed, and they are split by type in
    :func:`compute_commercial_floor_m2_cap`.

    Parameters
    ----------
    gompertz:
        Dictionary containing the gompertz parameters.
    service_value_added:
        The service value added (SVA) data as computed in :func:`get_service_value_added`.

    Returns
    -------
    commercial_m2_cap:
        Commercial floorspace averages for all types of commercial buildings.

    """
    # Select gompertz curve parameters for the total commercial m2 demand (stock)
    alpha, beta, gamma = (
        (gompertz['All']['a'], gompertz['All']['b'], gompertz['All']['c'])
        if FLAG_EXPDEC == 0 else GOMPERTZ_EXPDEC
    )
    # alpha_low = alpha * LOWCOMM
    alpha_low = alpha

    # find the total commercial m2 stock (in Millions of m2)
    commercial_m2_cap = pd.DataFrame(index=YEARS, columns=REGIONS_RANGE)
    commercial_m2_cap_low = commercial_m2_cap.copy()

    # Compute commercial floorspace using Gompertz curves
    for year in YEARS:
        for region in REGIONS_RANGE:
            exp_factor = math.exp((-gamma/1000) * service_value_added[region][year])
            if FLAG_EXPDEC == 0:
                commercial_m2_cap[region][year] = alpha * math.exp(-beta * exp_factor)
                commercial_m2_cap_low[region][year] = alpha_low * math.exp(-beta * exp_factor)
            else:
                commercial_m2_cap[region][year] = max(0.542, alpha - beta * exp_factor)

    # commercial floorspace is scaled here (in case lowComm is not 1)
    scale_comm = pd.Series([1.0, 0.0], index=[2020, 2060], name='time').reindex(YEARS).interpolate(
        method='linear', limit_direction='both')
    commercial_m2_cap = (commercial_m2_cap.mul(scale_comm, axis=0)
                         + commercial_m2_cap_low.mul((1-scale_comm), axis=0))
    return commercial_m2_cap


def compute_commercial_floor_m2_cap(gompertz: dict,
                                    commercial_m2_cap_sum: pd.DataFrame,
                                    service_value_added: pd.DataFrame):
    """Subdivide the total commercial floorspace across Offices, Retail+, Govt+ & Hotels+.

    Parameters
    ----------
    gompertz:
        Gompertz parameters to use.
    commercial_m2_cap_sum:
        Total commercial floorspace that needs to be subdivided.
        From func:`compute_commercial_floor_m2_cap_sum`.
    service_value_added:
        Service value added (SVA) from func:`get_service_value_added`.

    Returns
    -------
    commercial_m2_cap_all:
        Subdivided commercial floorspace for each subtype (Office, Retail+, Hotels+, Govt+)
    minimum_comm:
        Minimum commercial floorspace.

    """
    types = ["Office", "Retail+", "Hotels+", "Govt+"]
    index = pd.MultiIndex.from_product([types, REGIONS_RANGE, YEARS],
                                       names=["Type", "Region", "Time"])
    commercial_m2_cap_all = pd.DataFrame(index=index, columns=["m2_per_cap"]).fillna(0)
    minimum_comm = xr.DataArray(25.0, dims=["Type"], coords={"Type": types})
    for year in YEARS[1:]:
        for region in REGIONS_RANGE:
            # Calculate floorspace for all types and update the minimum values
            floorspace_commercial_list = {}
            for type_ in types:
                #value = gompertz_value(type_, region, year, service_value_added)
                params = gompertz[type_]
                value = params['a'] * math.exp(-params['b'] * math.exp((-params['c'] / 1000) *
                                                                       service_value_added[region][year]))
                floorspace_commercial_list[type_] = value
                minimum_comm.loc[type_] = min(minimum_comm.loc[type_], value)

            # Sum all floorspace values for normalization
            commercial_sum = sum(floorspace_commercial_list.values())

            # Calculate and assign the floorspace for each type
            for type_ in types:
                commercial_m2_cap_all.loc[(type_, region, year), "m2_per_cap"] = (
                    commercial_m2_cap_sum.loc[year, region] *
                    (floorspace_commercial_list[type_] / commercial_sum)
                )

    commercial_m2_cap_all = commercial_m2_cap_all.unstack("Type")
    commercial_m2_cap_all = commercial_m2_cap_all.droplevel(0, axis = 1)
    return commercial_m2_cap_all, minimum_comm


def compute_housing_type(database_directory: Path) -> xr.DataArray:
    """Compute the housing type shares.

    Parameters
    ----------
    database_directory:
        Directory in which the housing type shares data file can be found.

    Returns
    -------
    housing_type_xr:
        Shares per housing type with dims
        ``(Time, Region, Area, Quintile, Type)``.

    """
    housing_type_data: pd.DataFrame = pd.read_csv(database_directory / 'Housing_type_dynamic.csv',
                                                  index_col = [0,1,2])
    # also interpolate housing type data
    index_ht = pd.MultiIndex.from_product([list(range(HIST_YEAR, END_YEAR + 1)),
                                        list(range(1,REGIONS + 1)),
                                        ['Urban', 'Rural'] ])
    housing_type = pd.DataFrame(np.nan, index=index_ht, columns=housing_type_data.columns)
    housing_type.index.names = ['Time','Region','Area']

    for year in list(housing_type_data.index.levels[0]):
        housing_type.loc[idx[year,:,:],:] = housing_type_data.loc[idx[year,:,:],:]

    for region in list(range(1,REGIONS + 1)):
        for area in ['Urban', 'Rural']:
            housing_types_interpolated = housing_type.loc[idx[:,region,area],:].interpolate(
                method='linear', limit_direction='both')
            housing_type.loc[idx[:,region,area],:] = housing_types_interpolated.values

    housing_type_xr = dataset_to_array(housing_type.to_xarray(), ["Time", "Region", "Area"],
                                       ["Type"])
    housing_type_xr.coords["Region"] = [str(x.values) for x in housing_type_xr.coords["Region"]]

    # Rebroadcast to flat quintile areas first, then split into Area + Quintile coords
    target_areas = list(housing_type_xr.coords["Area"].values) + urban_q_areas + rural_q_areas
    housing_type_xr = area_graph.rebroadcast_xarray(
                        housing_type_xr,
                        output_coords=target_areas,
                        dim="Area")

    q_labels = [f"Q{i}" for i in range(1, len(urban_q_areas) + 1)]
    housing_type_urban = (
        housing_type_xr.sel(Area=urban_q_areas)
        .assign_coords(Area=q_labels)
        .rename({"Area": "Quintile"})
        .expand_dims(Area=["Urban"])
    )
    housing_type_rural = (
        housing_type_xr.sel(Area=rural_q_areas)
        .assign_coords(Area=q_labels)
        .rename({"Area": "Quintile"})
        .expand_dims(Area=["Rural"])
    )
    housing_type_xr = xr.concat([housing_type_urban, housing_type_rural], dim="Area").transpose(
        "Time", "Region", "Area", "Quintile", "Type")

    # Quantify units (share - dimensionless)
    housing_type_xr = prism.Q_(housing_type_xr, "")

    return housing_type_xr


def compute_average_m2_capita(base_directory: Path) -> xr.DataArray:
    """Load the average residential floorspace per capita.

    Parameters
    ----------
    base_directory:
        Base directory in which the Average_m2_per_cap.csv file needs to be located.

    Returns
    -------
    average_m2_capita:
        Average residential floorspace by region, area, quintile and type.

    """
    cap_fp = base_directory.joinpath('buildings', 'files_DB','Average_m2_per_cap.csv')
    average_m2_capita_df: pd.DataFrame = pd.read_csv(cap_fp, index_col = [0,1])
    column_mapping = {'1': 'Detached', '2': 'Semi-detached', '3': 'Appartment', '4': 'High-rise'}
    average_m2_capita_df.rename(columns=column_mapping, inplace=True)
    average_m2_capita = dataset_to_array(average_m2_capita_df.to_xarray(), ["Region", "Area"],
                                         ["Type"])
    average_m2_capita.coords["Region"] = [str(x.values) for x in average_m2_capita.coords["Region"]]

    # Rebroadcast to flat quintile areas first, then split into Area + Quintile coords.
    # TODO: this could be improved with better data availability; quintiles likely have
    # different sqm/cap than the area average, but for now we assume the same.
    target_areas = list(average_m2_capita.coords["Area"].values) + urban_q_areas + rural_q_areas
    average_m2_capita = area_graph.rebroadcast_xarray(
                        average_m2_capita,
                        output_coords=target_areas,
                        dim="Area",)

    q_labels = [f"Q{i}" for i in range(1, len(urban_q_areas) + 1)]
    average_urban = (
        average_m2_capita.sel(Area=urban_q_areas)
        .assign_coords(Area=q_labels)
        .rename({"Area": "Quintile"})
        .expand_dims(Area=["Urban"])
    )
    average_rural = (
        average_m2_capita.sel(Area=rural_q_areas)
        .assign_coords(Area=q_labels)
        .rename({"Area": "Quintile"})
        .expand_dims(Area=["Rural"])
    )
    average_m2_capita = xr.concat([average_urban, average_rural], dim="Area").transpose(
        "Region", "Area", "Quintile", "Type")

    # Quantify units
    average_m2_capita = prism.Q_(average_m2_capita, "m^2/person")

    return average_m2_capita


def compute_housing_residential(population: xr.DataArray,
                                average_m2_capita: xr.DataArray,
                                housing_type: xr.DataArray,
                                floorspace_rururb: xr.DataArray,
                                circular_economy_config: dict) -> xr.DataArray:
    """Compute the total residential floorspace.

    Parameters
    ----------
    population:
        Population for each region at year t, per Area (Rural, Urban) and Total.
    average_m2_capita:
        Average floorspace per capita.
    housing_type:
        Shares over the different housing subtypes.
    floorspace_rururb:
        Residential subdivided over rural and urban regions.
    circular_economy_config:
        Circular economy configuration for base scenario.

    Returns
    -------
    floorspace_residential:
        Residential floorspace over different housing types in urban and rural setting.

    """
    # Calculate the m2 per capita for each housing type
    m2_housing_per_capita = average_m2_capita * housing_type
    # Calculate the share of housing types on a m2 basis
    m2_housing_share = m2_housing_per_capita / m2_housing_per_capita.sum(["Type"])
    total_m2_housing_per_cap = m2_housing_share*floorspace_rururb
    total_m2_housing_per_cap = prism.Q_(total_m2_housing_per_cap, "m^2/person")

    # Implement circular economy measures if configuration is provided
    if 'base' in circular_economy_config.keys():
        total_m2_housing_per_cap = ce_measures_residential_housing(total_m2_housing_per_cap, population,
                                                                   circular_economy_config)

    # Align Area labels/order with the split residential floorspace coordinates.
    population_urbrur = population.sel(Area=total_m2_housing_per_cap.coords["Area"])
    total_m2_housing = total_m2_housing_per_cap * population_urbrur

    # Keep quintiles explicit and merge only Type + Area into the Type axis.
    floorspace_residential = merge_dims(total_m2_housing, "Type", "Area")
    return floorspace_residential.transpose("Time", "Region", "Type", "Quintile")
