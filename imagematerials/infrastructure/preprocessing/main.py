"""Preprocessing entry point for the infrastructure module.

Reads raw IMAGE driver outputs and infrastructure parameter files and turns
them into the dictionary consumed by ``imagematerials.factory.Sector``
(stocks, lifetimes, material intensities, knowledge graph, etc.).

The pipeline is organised into the same logical stages used in the
buildings and vehicles sectors:

  drivers → network → roads → bridges/tunnels → rail → stocks
                                                     → materials
                                                     → lifetimes
"""
from pathlib import Path

from imagematerials.infrastructure.constants import SCEN
from imagematerials.infrastructure.preprocessing.bridges_tunnels import (
    build_bridge_tunnel_stocks,
    build_parking_stocks,
)
from imagematerials.infrastructure.preprocessing.drivers import (
    load_data_2024,
    load_region_list,
    process_gdp_and_population,
    process_hdi,
    process_vkm,
)
from imagematerials.infrastructure.preprocessing.lifetimes import build_lifetimes
from imagematerials.infrastructure.preprocessing.materials import (
    build_material_intensities,
)
from imagematerials.infrastructure.preprocessing.network import (
    compute_parking_areas,
    compute_road_areas,
    process_urban_area_and_density,
    split_gdp_urban_rural,
    split_vkm_urban_rural,
)
from imagematerials.infrastructure.preprocessing.rail import build_rail_stocks
from imagematerials.infrastructure.preprocessing.roads import build_road_type_stocks
from imagematerials.infrastructure.preprocessing.stocks import combine_all_stocks


def get_preprocessing_data_infrastructure(base_directory, climate_policy_config,
                                            circular_economy_config=None,
                                            image_scenario: str = SCEN):
    """Read raw data and return the dictionary consumed by ``GenericStocks``.

    Parameters
    ----------
    base_directory:
        Root of the raw-data tree (``data/raw``). Infrastructure inputs are
        expected at ``base_directory / "infrastructure" / "standard_data"``
        for files that are common across scenarios and at
        ``base_directory / "infrastructure" / image_scenario`` for files
        that vary by scenario.
    climate_policy_config:
        Dict with ``"config_file_path"``: a path to the IMAGE scenario
        folder (e.g. ``data/raw/image/SSP2_baseline``) that contains
        ``EnergyServices/`` and ``Socioeconomic/`` subfolders. Matches the
        convention used by the buildings, vehicles and electricity sectors.
    circular_economy_config:
        Currently unused for infrastructure; accepted for signature parity
        with the other sectors.
    image_scenario:
        Infrastructure scenario name. Selects the
        ``base_directory / "infrastructure" / <image_scenario>`` folder.
    """
    base_directory = Path(base_directory)
    standard_data_dir = base_directory / "infrastructure" / "standard_data"
    scenario_data_dir = base_directory / "infrastructure" / image_scenario
    image_dir = Path(climate_policy_config["config_file_path"])

    assert standard_data_dir.is_dir(), standard_data_dir
    assert scenario_data_dir.is_dir(), scenario_data_dir
    assert image_dir.is_dir(), image_dir

    # --- Drivers (sections 1-4) ---
    region_list = load_region_list(standard_data_dir)
    data_2024 = load_data_2024(standard_data_dir)
    vkm_pivoted, vkm_rail_pivoted, vkm_rail_HST_pivoted = process_vkm(image_dir, region_list)
    gdp_cap_new, urban_population, rural_population = process_gdp_and_population(
        image_dir, region_list)
    sorted_IMAGE_hdi = process_hdi(standard_data_dir)

    # --- Network (sections 5-8) ---
    (sorted_urban_pop_density, sorted_rural_pop_density, _sorted_pop_density,
     _urban_area_filtered, _rural_area_filtered) = process_urban_area_and_density(
        scenario_data_dir, urban_population, rural_population)

    sorted_gdp_urban_cap, sorted_gdp_rural_cap = split_gdp_urban_rural(
        gdp_cap_new, urban_population, rural_population)

    (sorted_vkm_urban_cap, _sorted_vkm_rural_cap, sorted_vkm_rail_tot,
     sorted_vkm_HST_rail_tot, _sorted_vkm_rail_cap) = split_vkm_urban_rural(
        vkm_pivoted, vkm_rail_pivoted, vkm_rail_HST_pivoted,
        urban_population, rural_population, sorted_gdp_urban_cap, sorted_gdp_rural_cap)

    (adjusted_rural_road_area, adjusted_urban_road_area,
     obsolete_road_rural_area, obsolete_road_urban_area) = compute_road_areas(
        data_2024, sorted_rural_pop_density, sorted_urban_pop_density,
        sorted_gdp_rural_cap, sorted_gdp_urban_cap, sorted_vkm_urban_cap,
        urban_population, rural_population)

    parking_areas = compute_parking_areas(
        data_2024, adjusted_rural_road_area, adjusted_urban_road_area,
        obsolete_road_rural_area, obsolete_road_urban_area,
        sorted_gdp_urban_cap, sorted_gdp_rural_cap)

    # --- Roads (section 9) ---
    roads_result = build_road_type_stocks(
        standard_data_dir, adjusted_rural_road_area, adjusted_urban_road_area,
        obsolete_road_rural_area, obsolete_road_urban_area,
        sorted_IMAGE_hdi, sorted_gdp_urban_cap, sorted_gdp_rural_cap)

    region_coords = roads_result["region_coords"]
    extended_time_coords = roads_result["extended_time_coords"]
    end_year_used = roads_result["end_year"]

    # --- Bridges, tunnels and parking (section 9b) ---
    bt_active_types, bt_active_das, bt_obs_types, bt_obs_das = build_bridge_tunnel_stocks(
        scenario_data_dir, data_2024,
        adjusted_rural_road_area, adjusted_urban_road_area,
        obsolete_road_rural_area, obsolete_road_urban_area,
        extended_time_coords, region_coords)

    parking_types, parking_das, parking_obs_types, parking_obs_das = build_parking_stocks(
        parking_areas, extended_time_coords, region_coords)

    # --- Rail (section 9c) ---
    rail_element_types, rail_element_das, rail_obs_types, rail_obs_das = build_rail_stocks(
        standard_data_dir, data_2024,
        sorted_vkm_rail_tot, sorted_vkm_HST_rail_tot,
        sorted_IMAGE_hdi, sorted_urban_pop_density, urban_population,
        extended_time_coords, region_coords)

    # --- Combine all stocks (section 9d) ---
    combined = combine_all_stocks(
        roads_result, bt_active_types, bt_active_das, bt_obs_types, bt_obs_das,
        parking_types, parking_das, parking_obs_types, parking_obs_das,
        rail_element_types, rail_element_das, rail_obs_types, rail_obs_das)

    # --- Material intensities (section 10) ---
    da_mi, da_perm_agg_mi = build_material_intensities(
        standard_data_dir, scenario_data_dir,
        combined["final_all_types"], roads_result["all_types"], bt_active_types,
        parking_types, rail_element_types,
        combined["all_active_types"], combined["all_obsolete_types"],
        combined["obsolete_to_active_map"],
        region_coords, end_year_used)

    # --- Lifetimes (last section) ---
    lifetimes = build_lifetimes(
        scenario_data_dir, roads_result["combined_types"], bt_active_types, bt_obs_types,
        parking_types, parking_obs_types, rail_element_types, rail_obs_types,
        extended_time_coords)

    # --- Assemble final dictionary in the same shape as the legacy function ---
    return {
        "stocks": combined["stocks"],
        "active_types": combined["active_types"],
        "obsolete_types": combined["obsolete_types"],
        "knowledge_graph": combined["knowledge_graph"],
        "shares": None,
        "obsolete_to_active_map": combined["obsolete_to_active_map"],
        "set_unit_flexible": combined["set_unit_flexible"],
        "material_intensities": da_mi,
        "permanent_aggregate_mi": da_perm_agg_mi,
        "lifetimes": lifetimes,
    }
