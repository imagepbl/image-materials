import warnings
from pathlib import Path

import xarray as xr

from imagematerials.buildings.constants import SCENARIO_SELECT
from imagematerials.buildings.preprocessing.floorspace import (
    compute_average_m2_capita,
    compute_housing_residential,
    compute_housing_type,
    extrapolate_floorspace,
    get_image_floorspace,
)
from imagematerials.buildings.preprocessing.lifetimes import compute_lifetimes
from imagematerials.buildings.preprocessing.materials import (
    compute_mat_intensities_commercial,
    compute_mat_intensities_residential,
)
from imagematerials.buildings.preprocessing.population import compute_population
from imagematerials.concepts import knowledge_graph


def buildings_preprocessing(base_directory, climate_policy_config: dict, circular_economy_config: dict):
    base_directory = Path(base_directory)
    database_directory = base_directory / "files_DB" / SCENARIO_SELECT
    image_directory = base_directory / "files_IMAGE" / SCENARIO_SELECT
    assert database_directory.is_dir(), database_directory
    assert image_directory.is_dir()


    # Get floorspace for commercial + urban/rural
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        floorspace_image_commercial_rururb, minimum_comm = get_image_floorspace(image_directory, base_directory)
    floorspace_commercial_rururb = extrapolate_floorspace(floorspace_image_commercial_rururb, minimum_comm)

    # Rural/Urban floorspace [Time, Region, Area]
    floorspace_rururb = floorspace_commercial_rururb.sel({"Type": ["Urban", "Rural"]}).rename({"Type": "Area"})

    # Commercial floorspace [Time, Region, Type]
    floorspace_commercial = floorspace_commercial_rururb.sel(
        {"Type": [x.values for x in floorspace_commercial_rururb.coords["Type"] if x.values not in ["Urban", "Rural"]]})

    population = compute_population(base_directory)
    average_m2_capita = compute_average_m2_capita(base_directory)

    housing_type = compute_housing_type(database_directory)

    floorspace_residential = compute_housing_residential(population, average_m2_capita, housing_type, floorspace_rururb)

    floorspace = xr.concat((floorspace_residential, floorspace_commercial), dim="Type")

    # Lifetime computations, see lifetimes.py

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lifetimes = compute_lifetimes(base_directory, floorspace_commercial.coords["Type"].values)

    mat_intensities_comm = compute_mat_intensities_commercial(database_directory)
    mat_intensities_res = compute_mat_intensities_residential(database_directory)
    mat_intensities = xr.concat((mat_intensities_res, mat_intensities_comm), dim="Type")
    mat_intensities = knowledge_graph.rebroadcast_xarray(
                            mat_intensities, floorspace.coords["Type"].values)

    return {"stocks": floorspace, "lifetimes": lifetimes, "material_intensities": mat_intensities}
