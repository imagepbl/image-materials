"""Preprocessing of the buildings module with the main method for this."""
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

from imagematerials.buildings.constants import SCENARIO_SELECT
from imagematerials.buildings.preprocessing.circular_economy_measures import (
    apply_circular_economy_commercial_floorspace,
)
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
from imagematerials.concepts import create_building_graph, create_region_graph
from imagematerials.constants import IMAGE_REGIONS


def buildings_preprocessing(base_directory: Path, climate_policy_config: dict,
                            circular_economy_config: dict,
                            image_scenario: str = SCENARIO_SELECT) -> xr.DataArray:
    """Preprocess the buildings data from start to finish.

    Parameters
    ----------
    base_directory:
        Path to the base directory in which the input files are located.
    climate_policy_config:
        Climate policy configuration for which the preprocessing will be done.
    circular_economy_config:
        Climate economy configuration for which the preprocessing will be done.
    image_scenario:
        Image scenario, such as SSP2. The scenario selected should have the input
        data available.

    Returns
    -------
    building_preprocessing:
        Dictionary containing the stocks, lifetimes, material_intensities, knowledge_graph,
        and set_unit_flexible.

    """
    base_directory = Path(base_directory)
    database_directory = base_directory / "buildings" / image_scenario

    image_directory = Path(climate_policy_config["config_file_path"])
    assert database_directory.is_dir(), database_directory
    assert image_directory.is_dir(), image_directory

    # Get floorspace for commercial + urban/rural
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        floorspace_image_commercial_rururb, minimum_comm = get_image_floorspace(image_directory,
                                                                                base_directory)
    floorspace_commercial_rururb = extrapolate_floorspace(floorspace_image_commercial_rururb,
                                                          minimum_comm)

    # Rural/Urban floorspace [Time, Region, Area]
    floorspace_rururb = floorspace_commercial_rururb.sel(
        {"Type": ["Urban", "Rural"]}).rename({"Type": "Area"})

    # Commercial floorspace [Time, Region, Type]
    floorspace_commercial = floorspace_commercial_rururb.sel(
        {"Type": [x.values for x in floorspace_commercial_rururb.coords["Type"] if x.values not in ["Urban", "Rural"]]})
    
    if "base" or "narrow_activity" or "narrow" in circular_economy_config.keys():
        # Implement circular economy for commercial floorspace
        # This is only done for the base and narrow_activity scenarios, as the other scenarios do not have a circular economy component
        floorspace_commercial = apply_circular_economy_commercial_floorspace(floorspace_commercial, circular_economy_config)
        
    # Calculate population ("Total", "Rural", "Urban")
    population = compute_population(image_directory, base_directory)

    average_m2_capita = compute_average_m2_capita(base_directory)

    housing_type = compute_housing_type(database_directory)

    floorspace_residential = compute_housing_residential(population, average_m2_capita,
                                                         housing_type, floorspace_rururb,
                                                         circular_economy_config)

    # Commercial floorspace also needs to be multiplied by population & drop Area dimension
    floorspace_commercial_total = floorspace_commercial * population.sel({"Area": "Total"})
    floorspace_commercial_total = floorspace_commercial_total.drop_vars("Area")
    floorspace = xr.concat((floorspace_residential, floorspace_commercial_total), dim="Type")

    # Lifetime computations, see lifetimes.py

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lifetimes = compute_lifetimes(base_directory, floorspace_commercial.coords["Type"].values,
                                      circular_economy_config)

    mat_intensities_comm = compute_mat_intensities_commercial(database_directory,
                                                              circular_economy_config)
    mat_intensities_res = compute_mat_intensities_residential(database_directory,
                                                              circular_economy_config)
    mat_intensities = xr.concat((mat_intensities_res, mat_intensities_comm), dim="Type")
    knowledge_graph_buildings = create_building_graph()
    mat_intensities = knowledge_graph_buildings.rebroadcast_xarray(
                            mat_intensities, floorspace.coords["Type"].values)

    #TODO remove this quick fix
    region_coords = np.sort(floorspace.coords["Region"].values.astype(int)).astype(str)
    floorspace = knowledge_graph_buildings.rebroadcast_xarray(floorspace, region_coords,
                                                              dim ="Region")


    # convert region names to the standard IMAGE regions
    knowledge_graph_region = create_region_graph()

    floorspace = knowledge_graph_region.rebroadcast_xarray(floorspace, output_coords=IMAGE_REGIONS,
                                                           dim="Region")
    for key, value in lifetimes.items():
        lifetimes[key] = knowledge_graph_region.rebroadcast_xarray(value,
                                                                   output_coords=IMAGE_REGIONS,
                                                                   dim="Region")
    mat_intensities = knowledge_graph_region.rebroadcast_xarray(mat_intensities,
                                                                output_coords=IMAGE_REGIONS,
                                                                dim="Region")

    return {"stocks": floorspace, "lifetimes": lifetimes, "material_intensities": mat_intensities,
            "knowledge_graph": knowledge_graph_buildings,
            "set_unit_flexible": str(floorspace.pint.units)}
