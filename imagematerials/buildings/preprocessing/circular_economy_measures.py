"""Functions for implementing circular economy measures."""
import logging

import numpy as np
import prism
import xarray as xr

from imagematerials.concepts import create_region_graph
from imagematerials.util import apply_change_per_region


def ce_measures_residential_housing(total_m2_housing_per_cap: xr.DataArray,
                                    circular_economy_config: dict):
    """Implement circular economy measures for residential housing.

    Parameters
    ----------
    total_m2_housing_per_cap:
        The total amount of m2 per capita for all residential housing.
    circular_economy_config:
        Configuration of the circular economy.

    Returns
    -------
    total_m2_housing_per_cap:
        Updated values for the total amount of floorspace per capita.

    """
    region_knowledge_graph = create_region_graph()
    regions = total_m2_housing_per_cap.coords["Region"].values

    buildings_config = circular_economy_config["base"]["buildings"]
    base_year = buildings_config["base_year"]
    floor_pc_2020 = buildings_config["residential"]["2020"]["useful_floor_pc"]

    floor_pc_2020_xr = xr.DataArray(
        list(floor_pc_2020.values()),
        coords={"Region": list(floor_pc_2020.keys())},
        dims=["Region"],
        name="floor_pc_2020",
    )

    regions_mapped = list(region_knowledge_graph.find_relations_inverse(regions,
                                                                        floor_pc_2020.keys()))
    floor_pc_2020_mapped = region_knowledge_graph.rebroadcast_xarray(
        floor_pc_2020_xr, output_coords=regions_mapped, dim="Region")
    floor_pc_2020_mapped = prism.Q_(floor_pc_2020_mapped, "m^2/person")
    target_vals = floor_pc_2020_mapped
    current_vals = total_m2_housing_per_cap.sel(Time=base_year)\
                                        .sum(dim="Type")\
                                        .mean(dim="Area")
    scaling_factors = target_vals / current_vals

    total_m2_housing_per_cap.loc[{"Region": regions_mapped}] = total_m2_housing_per_cap.sel(
        Region = regions_mapped) * scaling_factors
    logging.debug("implemented 'base' for Residential Buildings")

    return total_m2_housing_per_cap


# if 'narrow' in circular_economy_config.keys():
#     building_config = circular_economy_config["narrow"]["buildings"]
#     base_year = building_config["base_year"]
#     target_year = building_config["target_year"]

#     residential_scenario_settings = building_config['residential']['m2_change_pc']
#     implementation_rate = building_config['implementation_rate']

#     residential_scenario_settings_xr = xr.DataArray(
#         list(residential_scenario_settings.values()),
#         coords={"Region": list(residential_scenario_settings.keys())},
#         dims=["Region"],
#         name="residential_scenario_settings"
#     )

#     regions_mapped = list(region_knowledge_graph.find_relations_inverse(
#         regions, residential_scenario_settings.keys()))
#     residential_scenario_settings_xr_mapped = region_knowledge_graph.rebroadcast_xarray(
#         residential_scenario_settings_xr, output_coords=regions_mapped, dim="Region")

#     total_m2_housing_per_cap = apply_change_per_region(
#         total_m2_housing_per_cap, base_year, target_year,
#         residential_scenario_settings_xr_mapped, implementation_rate)
#     print("implemented 'narrow' for Residential Buildings")


def apply_circular_economy_commercial_floorspace(floorspace_commercial: xr.DataArray,
                                                 circular_economy_config: dict) -> xr.DataArray:
    """Implement circular economy measures for commercial floorspace.

    Parameters
    ----------
    floorspace_commercial:
        Total commercial floorspace.
    circular_economy_config:
        Configuration file of the circular economy measures.

    Returns
    -------
    floorspace_commercial:
        Updated floorspace for commercial targets with circular economy configuration.

    """
    region_knowledge_graph = create_region_graph()
    regions = floorspace_commercial.coords["Region"].values
    # floorspace_commercial in m^2/cap

    # Base scenario
    if 'base' in circular_economy_config.keys():
        buildings_config = circular_economy_config["base"]["buildings"]

        base_year = buildings_config["base_year"]
        target_year = buildings_config["target_year"]
        floor_pc_2020 = buildings_config["commercial"]["2020"]["useful_floor_pc"]

        floor_pc_2020_xr = xr.DataArray(
            list(floor_pc_2020.values()),
            coords={"Region": list(floor_pc_2020.keys())},
            dims=["Region"],
            name="floor_pc_2020"
        )
        floor_pc_2020_xr = prism.Q_(floor_pc_2020_xr, "m^2/person")

        regions_mapped = list(region_knowledge_graph.find_relations_inverse(
            regions, floor_pc_2020.keys()))
        floor_pc_2020_mapped = region_knowledge_graph.rebroadcast_xarray(
            floor_pc_2020_xr, output_coords=regions_mapped, dim="Region")
        target_vals = floor_pc_2020_mapped
        current_vals = floorspace_commercial.sel(Time=2020).sum(dim="Type")

        scaling_factors = target_vals / current_vals

        floorspace_commercial.loc[{"Region": regions_mapped}] *= scaling_factors
        logging.debug("implemented 'base' for Commercial Buildings")

    ce_scen = None  # INITIALIZE ce_scen
    if "narrow" in circular_economy_config.keys():
        ce_scen = "narrow"
    if "narrow_activity" in circular_economy_config.keys():
        ce_scen = "narrow_activity"
    # narrow_activity scenario
    if ce_scen in circular_economy_config.keys():
        base_year = circular_economy_config[ce_scen]["buildings"]["base_year"]
        target_year = circular_economy_config[ce_scen]["buildings"]["target_year"]

        commercial_scenario_settings = circular_economy_config[ce_scen]["buildings"]['commercial']['m2_change_pc']
        implementation_rate = circular_economy_config[ce_scen]['buildings']['implementation_rate']

        commercial_scenario_settings_xr = xr.DataArray(
            list(commercial_scenario_settings.values()),
            coords={"Region": list(commercial_scenario_settings.keys())},
            dims=["Region"],
            name="commercial_scenario_settings"
        )

        regions_mapped = list(region_knowledge_graph.find_relations_inverse(
            regions, commercial_scenario_settings.keys()))
        commercial_scenario_settings_xr_mapped = region_knowledge_graph.rebroadcast_xarray(
            commercial_scenario_settings_xr, output_coords=regions_mapped, dim="Region")

        region_coords = np.sort(commercial_scenario_settings_xr_mapped.coords["Region"]
                                .values.astype(int)).astype(str)
        commercial_scenario_settings_xr_mapped = region_knowledge_graph.rebroadcast_xarray(
            commercial_scenario_settings_xr_mapped, region_coords, dim ="Region")
        floorspace_commercial = apply_change_per_region(floorspace_commercial, base_year,
                                                        target_year,
                                                        commercial_scenario_settings_xr_mapped,
                                                        implementation_rate)
        logging.debug(f"implemented '{ce_scen}' for Commercial Buildings")
        # fix unit like this for now :TODO to improve
        floorspace_commercial = prism.Q_(floorspace_commercial, "m^2/person")

    return floorspace_commercial.transpose("Time", "Region", "Type")


def circular_economy_measures_material_intensities_residential(
        xr_mat_res_intensities: xr.DataArray, circular_economy_config: dict) -> xr.DataArray:
    """Adjust material intensities for circular economy.

    Parameters
    ----------
    xr_mat_res_intensities:
        Material intensities before circular economy adjustments.
    circular_economy_config:
        Configuration for circular economy parameters.

    Returns
    -------
    xr_mat_res_intensities:
        Adjusted material intensities for circular economy.

    """
    # rename Cohort to Time for compatibility with apply_change_per_region function
    if "Cohort" in xr_mat_res_intensities.dims:
        xr_mat_res_intensities.rename({"Cohort": "Time"})

    ce_scen = None  # INITIALIZE ce_scen
    if "narrow" in circular_economy_config.keys():
        ce_scen = "narrow"
    if "narrow_product" in circular_economy_config.keys():
        ce_scen = "narrow_product"

    # import parameters from config file
    target_year = circular_economy_config[ce_scen]['buildings']['target_year']
    base_year = circular_economy_config[ce_scen]['buildings']['base_year']
    implementation_rate = circular_economy_config[ce_scen]['buildings']['implementation_rate']
    mat_changes = circular_economy_config[ce_scen]['buildings']['material_intensity_change']

    region_knowledge_graph = create_region_graph()
    model_regions = list(xr_mat_res_intensities.coords["Region"].values)
    materials_all = set(xr_mat_res_intensities.coords["material"].values)

    for mat in ("steel", "concrete", "aluminium"):
        if mat not in mat_changes or mat not in materials_all:
            continue

        # 1) TOML -> 1-D DA over region names
        change_dict = mat_changes[mat]
        raw = xr.DataArray(
            list(change_dict.values()),
            coords={"Region": list(change_dict.keys())},
            dims=["Region"],
            name=f"material_intensity_change_{mat}",
        )

        # 2) map region names -> region codes, then align to model order
        regions_mapped = list(region_knowledge_graph.find_relations_inverse(
            model_regions, raw.coords["Region"].values))
        changes_mapped = region_knowledge_graph.rebroadcast_xarray(
            raw, output_coords=regions_mapped, dim="Region")
        changes_mapped = changes_mapped.sel(Region=model_regions).astype(float)

        # 3) apply once per material
        cur = xr_mat_res_intensities.sel(material=mat)
        updated = apply_change_per_region(cur, base_year, target_year, changes_mapped,
                                          implementation_rate)
        updated = updated.reindex(Region=cur.coords["Region"])
        xr_mat_res_intensities.loc[dict(material=mat)] = updated

    # rename back
    if "Time" in xr_mat_res_intensities.dims:
        xr_mat_res_intensities = xr_mat_res_intensities.rename({"Time": "Cohort"})

    logging.debug(f"implemented '{ce_scen}' for Residential Buildings (lightweighting)")
    return xr_mat_res_intensities


def circular_economy_measures_material_intensities_commercial(xr_mat_comm_intensities: xr.DataArray,
                                                              circular_economy_config: dict,
                                                              model_regions: list):
    """Adjust the commercial material intensities for circular economy measures.

    Parameters
    ----------
    xr_mat_comm_intensities:
        Material intensities for the commercial buildings.
    circular_economy_config:
        Configuration for the circular economy.
    model_regions:
        Regions that are in the model.

    Returns
    -------
    xr_mat_comm_intensities:
        Updated commercial material intensities.

    """
    # work array with Time dim
    xr_mat = (xr_mat_comm_intensities.rename({"Cohort": "Time"})
              if "Cohort" in xr_mat_comm_intensities.dims else xr_mat_comm_intensities)
    
    ce_scen = None  # INITIALIZE ce_scen
    if "narrow" in circular_economy_config.keys():
        ce_scen = "narrow"
    if "narrow_product" in circular_economy_config.keys():
        ce_scen = "narrow_product"

    base_year = circular_economy_config[ce_scen]['buildings']['base_year']
    target_year = circular_economy_config[ce_scen]['buildings']['target_year']
    implementation_rate = circular_economy_config[ce_scen]['buildings']['implementation_rate']
    mat_changes = circular_economy_config[ce_scen]['buildings']['material_intensity_change']

    region_graph = create_region_graph()
    materials_order = list(xr_mat.coords["material"].values)

    updated_slices = []
    for mat in materials_order:
        cur = xr_mat.sel(material=mat)

        # only apply for those present in TOML; others pass through unchanged
        if mat in mat_changes:
            change_dict = mat_changes[mat]
            raw = xr.DataArray(
                list(change_dict.values()),
                coords={"Region": list(change_dict.keys())},
                dims=["Region"],
                name=f"mi_change_pc_{mat}",
            )

            # map region names -> region codes; align to model order
            regions_mapped = list(region_graph.find_relations_inverse(model_regions,
                                                                      raw.coords["Region"].values))
            changes_mapped = region_graph.rebroadcast_xarray(raw, output_coords=regions_mapped,
                                                             dim="Region")
            changes_mapped = changes_mapped.sel(Region=model_regions).astype(float)

            # apply once per material
            updated = apply_change_per_region(cur, base_year, target_year, changes_mapped,
                                              implementation_rate)
            # keep Region order & dim order identical to cur
            updated = updated.reindex(Region=cur.coords["Region"]).transpose(*cur.dims)
        else:
            updated = cur

        # attach the material coord and collect
        updated_slices.append(updated.expand_dims(material=[mat]))

    xr_mat_updated = xr.concat(updated_slices, dim="material")

    # rename back to Cohort if needed
    xr_mat_comm_intensities = (xr_mat_updated.rename({"Time": "Cohort"})
                               if "Time" in xr_mat_updated.dims else xr_mat_updated)

    logging.debug("implemented 'narrow_product' for Commercial Buildings (lightweighting)")
    return xr_mat_comm_intensities
