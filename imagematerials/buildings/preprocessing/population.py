"""Population related functions for computing the building requirements."""

import numpy as np
import pandas as pd
import prism
import xarray as xr

from importlib.resources import files
from pathlib import Path


from imagematerials.buildings.constants import REGIONS_RANGE, area_labels, urban_q_areas, rural_q_areas
from imagematerials.buildings.util import _as_string_regions, _normalize_shares
from imagematerials.read_mym import read_mym_df

prism.unit_registry.load_definitions(files("imagematerials") / "units.txt")


def _get_pop_q_xr(
    image_directory: Path,
    region_range: range,
    area_labels: dict[int, str] = area_labels) -> xr.DataArray:
    """Load IMAGE population quintile data and convert it to an xarray DataArray.

    Parameters
    ----------
    image_directory : Path
        Directory containing IMAGE scenario files, including
        ``Socioeconomic/pop_q.out``.
    region_range : range
        Region identifiers to keep from the source data.
    area_labels : dict[int, str], optional
        Mapping from numeric area IDs to area names.

    Returns
    -------
    xr.DataArray
        Population data with dimensions ``("Time", "Region", "Area")``.
    """
    pop_q = read_mym_df(image_directory.joinpath("Socioeconomic", "pop_q.out"))
    pop_q = pop_q.rename(columns={"time": "Time", "DIM_1": "Region"})
    value_cols = [c for c in pop_q.columns if c not in ["Time", "Region"]]

    pop_q_xr = (
        pop_q
        .set_index(["Time", "Region"])[value_cols]
        .rename_axis(columns="Area")
        .stack(future_stack=True)
        .to_xarray()
        .transpose("Time", "Region", "Area")
    )
    pop_q_xr = pop_q_xr.assign_coords(Area=[area_labels.get(a, a) for a in pop_q_xr.coords["Area"].values])
    # remove global region (27) by selecting only the specified region range
    pop_q_xr = _as_string_regions(pop_q_xr.sel(Region=region_range))
    return pop_q_xr


def _build_total_population(
        image_directory: Path,
        pop_q_data: xr.DataArray,
        base_population: xr.DataArray,
        regions: list[str],
    ) -> tuple[xr.DataArray, np.ndarray]:
    """Construct a continuous total-population series from historic and IMAGE data.

    We need this to avoid massive build up of buildings stock by artificial start of population growth in 1971, where model is initiated.
    
    Parameters
    ----------
    pop_q_data : xr.DataArray
        IMAGE population data with dimensions including ``Time``, ``Region``, and
        ``Area``. The ``Area='Total'`` slice is used as IMAGE total population.
    base_population : xr.DataArray
        Reference population data used to ensure the final time range covers all
        required years.
    regions : list[str]
        Region codes to include in the output.

    Returns
    -------
    tuple[xr.DataArray, np.ndarray]
        ``(total_population, time_all)`` where:
        - ``total_population`` has dimensions ``("Time", "Region")``
        - ``time_all`` is the full annual timeline used for reindexing.
    """
    base_directory = image_directory.parent.parent
    historic_path = base_directory / "buildings" / "standard_data" / "historic_population.csv"

    historic_population = pd.read_csv(historic_path, index_col=0, header=0)
    # convert to people by dividng by 1000 
    historic_population = historic_population.loc[:1971] / 1000
    historic_population = historic_population.reindex(
        index=range(historic_population.index.min(), historic_population.index.max() + 1)  # Explicitly specify index
    ).interpolate(method="linear")


    historic_population.columns = [str(c) for c in historic_population.columns]
    historic_population.index = historic_population.index.astype(int)
    historic_population.index.name = "Time"

    historic_total = xr.DataArray(
        historic_population.values,
        dims=("Time", "Region"),
        coords={
            "Time": historic_population.index.to_numpy(),
            "Region": historic_population.columns.to_numpy(),
        },
    )

    image_total = _as_string_regions(pop_q_data.sel(Area="Total"))

    time_all = np.arange(
        int(min(historic_total.coords["Time"].min(), image_total.coords["Time"].min(), base_population.coords["Time"].min())),
        int(max(historic_total.coords["Time"].max(), image_total.coords["Time"].max(), base_population.coords["Time"].max())) + 1,
    )

    historic_total = historic_total.reindex(Time=time_all, Region=regions)
    image_total = image_total.reindex(Region=regions).interp(Time=time_all, kwargs={"fill_value": "extrapolate"})

    total_population = historic_total.combine_first(image_total)
    total_population = total_population.interpolate_na(dim="Time", method="linear").clip(min=0)
    return total_population, time_all


def _split_urban_rural(pop_q_data: xr.DataArray, 
                       total_population: xr.DataArray, 
                       regions: list[str], 
                       time_all: np.ndarray) -> xr.DataArray:
    """Split total population into urban and rural components.

    Parameters
    ----------
    pop_q_data : xr.DataArray
        Population data containing at least ``Area`` values ``Urban``, ``Rural``,
        and ``Total``.
    total_population : xr.DataArray
        Total population with dimensions ``("Time", "Region")``.
    regions : list[str]
        Region codes to include.
    time_all : np.ndarray
        Full annual timeline used for interpolation.

    Returns
    -------
    xr.DataArray
        Urban/rural population with dimensions ``("Area", "Time", "Region")``
        (xarray order may vary by operation).
    """
    ur_share = pop_q_data.sel(Area=["Urban", "Rural"]) / pop_q_data.sel(Area="Total")
    ur_share = _as_string_regions(ur_share)
    ur_share = ur_share.reindex(Region=regions).interp(Time=time_all, kwargs={"fill_value": "extrapolate"})
    ur_share = _normalize_shares(ur_share, dim="Area", fallback=0)
    return ur_share * total_population


def _split_quintiles(
    pop_q_data: xr.DataArray,
    urban_rural_population: xr.DataArray,
    regions: list[str],
    time_all: np.ndarray,
    urban_q_areas: list[str],
    rural_q_areas: list[str],
) -> xr.DataArray:
    """Split urban and rural populations into quintiles and combine them with a new `Quintile` dimension.

    Parameters
    ----------
    pop_q_data : xr.DataArray
        Population data containing quintile areas and aggregate urban/rural areas.
    urban_rural_population : xr.DataArray
        Population split into ``Urban`` and ``Rural`` by time and region.
    regions : list[str]
        Region codes to include.
    time_all : np.ndarray
        Full annual timeline used for interpolation.
    urban_q_areas : list[str]
        Area labels for urban quintiles.
    rural_q_areas : list[str]
        Area labels for rural quintiles.

    Returns
    -------
    xr.DataArray
        Combined population array with dimensions ``("Area", "Quintile", "Time", "Region")``.
    """
    urban_q_share = pop_q_data.sel(Area=urban_q_areas) / pop_q_data.sel(Area="Urban")
    urban_q_share = _as_string_regions(urban_q_share)
    urban_q_share = urban_q_share.reindex(Region=regions).interp(Time=time_all, kwargs={"fill_value": "extrapolate"})
    urban_q_share = _normalize_shares(urban_q_share, dim="Area", fallback=1 / len(urban_q_areas))

    rural_q_share = pop_q_data.sel(Area=rural_q_areas) / pop_q_data.sel(Area="Rural")
    rural_q_share = _as_string_regions(rural_q_share)
    rural_q_share = rural_q_share.reindex(Region=regions).interp(Time=time_all, kwargs={"fill_value": "extrapolate"})
    rural_q_share = _normalize_shares(rural_q_share, dim="Area", fallback=1 / len(rural_q_areas))

    urban_q_pop = urban_q_share * urban_rural_population.sel(Area="Urban")
    rural_q_pop = rural_q_share * urban_rural_population.sel(Area="Rural")

    # Align `Area` dimension sizes explicitly before concatenation
    urban_q_pop = urban_q_pop.expand_dims(Quintile=urban_q_areas)
    rural_q_pop = rural_q_pop.expand_dims(Quintile=rural_q_areas)

    # Combine urban and rural quintiles into a single DataArray with a consistent `Area` dimension
    combined_quintiles = xr.concat(
        [urban_q_pop, rural_q_pop],
        dim="Area",
        join="outer",
    )

    return combined_quintiles


def _align_core_population(population_split: xr.DataArray, base_population: xr.DataArray) -> xr.DataArray:
    """Align total, urban, and rural values to a base population dataset.

    Parameters
    ----------
    population_split : xr.DataArray
        Population split data containing at least ``Total``, ``Urban``, and
        ``Rural`` areas.
    base_population : xr.DataArray
        Reference population data for the same core areas.

    Returns
    -------
    xr.DataArray
        Updated ``population_split`` with core areas overwritten on overlapping
        time-region coordinates.
    """
    aligned_base = _as_string_regions(base_population).transpose("Time", "Region", "Area")
    # Find the years and regions that exist in both datasets.
    common_time = np.intersect1d(population_split.coords["Time"].values, aligned_base.coords["Time"].values)
    common_region = np.intersect1d(population_split.coords["Region"].values, aligned_base.coords["Region"].values)

    # For Total/Urban/Rural, replace values with the base dataset on the overlap.
    for area in ["Total", "Urban", "Rural"]:
        population_split.loc[{"Time": common_time, "Region": common_region, "Area": area}] = aligned_base.sel(
            Time=common_time,
            Region=common_region,
            Area=area,
        )
    return population_split


def _equalize_quintiles(population_split: xr.DataArray, quintile_areas: list[str], target_area: str) -> xr.DataArray:
    """Set all quintiles in a group to equal shares of a target area.

    Parameters
    ----------
    population_split : xr.DataArray
        Population split containing quintile areas and a target aggregate area.
    quintile_areas : list[str]
        Area labels corresponding to quintiles that should be equalized.
    target_area : str
        Aggregate area whose values are split equally across quintiles.

    Returns
    -------
    xr.DataArray
        Population split with equalized quintile values.
    """
    equal_vals = population_split.sel(Area=target_area) / len(quintile_areas)
    population_split.loc[{"Area": quintile_areas}] = equal_vals
    return population_split


def compute_population_split(
    image_directory: Path,
    region_range: range = REGIONS_RANGE,
    urban_q_areas: list[str] = urban_q_areas,
    rural_q_areas: list[str] = rural_q_areas) -> xr.DataArray:
    """Build full population split (total, urban/rural, and quintiles).

    Parameters
    ----------
    image_directory : Path
        Directory containing IMAGE scenario output.
    region_range : range, optional
        Region identifiers to keep.
    urban_q_areas : list[str] | None, optional
        Labels for urban quintile areas. Defaults to Urban Q1-Q5.
    rural_q_areas : list[str] | None, optional
        Labels for rural quintile areas. Defaults to Rural Q1-Q5.
        
    Returns
    -------
    xr.DataArray
        Quantified population split in units of people.
    """
    regions_all = [str(r) for r in region_range]

    pop_q_data = _get_pop_q_xr(image_directory, region_range)
    population = pop_q_data.sel(Area=["Total", "Urban", "Rural"]).transpose("Time", "Region", "Area")

    total_population_xr, time_all = _build_total_population(
        image_directory=image_directory,
        pop_q_data=pop_q_data,
        base_population=population,
        regions=regions_all,
    )

    urban_rural_population_xr = _split_urban_rural(
        pop_q_data=pop_q_data,
        total_population=total_population_xr,
        regions=regions_all,
        time_all=time_all,
    )

    combined_quintiles_population_xr = _split_quintiles(
        pop_q_data=pop_q_data,
        urban_rural_population=urban_rural_population_xr,
        regions=regions_all,
        time_all=time_all,
        urban_q_areas=urban_q_areas,
        rural_q_areas=rural_q_areas,
    )

    population_split_xr = xr.concat(
        [
            total_population_xr.expand_dims(Area=["Total"]),
            urban_rural_population_xr,
            combined_quintiles_population_xr,
        ],
        dim="Area",
        join="outer",
    ).transpose("Time", "Region", "Area", "Quintile").clip(min=0)

    population_split_xr = _align_core_population(population_split_xr, population)
    population_split_xr = _equalize_quintiles(population_split_xr, urban_q_areas, "Urban")
    population_split_xr = _equalize_quintiles(population_split_xr, rural_q_areas, "Rural")

    return prism.Q_(population_split_xr, "people")