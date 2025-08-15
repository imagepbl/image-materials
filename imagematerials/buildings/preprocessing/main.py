import warnings
from pathlib import Path

import xarray as xr
import numpy as np

from imagematerials.buildings.constants import SCENARIO_SELECT
from imagematerials.buildings.preprocessing.floorspace import (
    compute_average_m2_capita,
    compute_housing_residential,
    compute_housing_type,
    extrapolate_floorspace,
    get_image_floorspace,
    apply_circular_economy_commercial_floorspace
)
from imagematerials.buildings.preprocessing.lifetimes import compute_lifetimes
from imagematerials.buildings.preprocessing.materials import (
    compute_mat_intensities_commercial,
    compute_mat_intensities_residential,
)
from imagematerials.buildings.preprocessing.population import compute_population
from imagematerials.concepts import create_building_graph


def buildings_preprocessing(base_directory, climate_policy_config: dict, circular_economy_config: dict):
    base_directory = Path(base_directory)
    database_directory = base_directory / "buildings" / SCENARIO_SELECT

    image_directory = Path(climate_policy_config["config_file_path"])
    assert database_directory.is_dir(), database_directory
    assert image_directory.is_dir(), image_directory

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
    
    if "base" or "narrow" in circular_economy_config.keys():
        # Implement circular economy for commercial floorspace
        # This is only done for the base and narrow scenarios, as the other scenarios do not have a circular economy component
        floorspace_commercial = apply_circular_economy_commercial_floorspace(floorspace_commercial, circular_economy_config)
        
    # Calculate population ("Total", "Rural", "Urban")
    population = compute_population(image_directory, base_directory)
    
    average_m2_capita = compute_average_m2_capita(base_directory)

    housing_type = compute_housing_type(database_directory)

    floorspace_residential = compute_housing_residential(population, average_m2_capita, housing_type, floorspace_rururb, circular_economy_config)
    
    # Commercial floorspace also needs to be multiplied by population & drop Area dimension
    floorspace_commercial_total = floorspace_commercial * population.sel({"Area": "Total"})
    floorspace_commercial_total = floorspace_commercial_total.drop_vars("Area")
    floorspace = xr.concat((floorspace_residential, floorspace_commercial_total), dim="Type")

    # Lifetime computations, see lifetimes.py

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lifetimes = compute_lifetimes(base_directory, floorspace_commercial.coords["Type"].values, circular_economy_config)

    mat_intensities_comm = compute_mat_intensities_commercial(database_directory, circular_economy_config)
    mat_intensities_res = compute_mat_intensities_residential(database_directory, circular_economy_config)
    mat_intensities = xr.concat((mat_intensities_res, mat_intensities_comm), dim="Type")
    knowledge_graph = create_building_graph()
    mat_intensities = knowledge_graph.rebroadcast_xarray(
                            mat_intensities, floorspace.coords["Type"].values)
    
    #TODO remove this quick fix
    region_coords = np.sort(floorspace.coords["Region"].values.astype(int)).astype(str)
    floorspace = knowledge_graph.rebroadcast_xarray(floorspace, region_coords, dim ="Region")

    return {"stocks": floorspace, "lifetimes": lifetimes, "material_intensities": mat_intensities,
            "knowledge_graph": knowledge_graph, "set_unit_flexible": str(floorspace.pint.units)}
