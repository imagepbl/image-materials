import warnings
from pathlib import Path

import xarray as xr
import numpy as np

from imagematerials.constants import IMAGE_REGIONS

from imagematerials.vehicles.constants import FOLDER
from imagematerials.concepts import create_building_graph
from imagematerials.concepts import create_region_graph

from imagematerials.vehicles.preprocessing.preprocessing_old import preprocess
from imagematerials.util import summarize_prep_data

def vehicles_preprocessing(base_directory: str, climate_policy_config: dict,
                           circular_economy_config: dict, image_scenario: str = FOLDER):
    # Preparing directory shorthands
    base_directory = Path(base_directory)
    general_data_directory = base_directory.joinpath("vehicles", "standard_data")
    scenario_data_directory = base_directory.joinpath("vehicles", image_scenario)
    climate_policy_data_directory = Path(climate_policy_config["config_file_path"])

    # Sanity check
    assert general_data_directory.is_dir(), general_data_directory
    assert scenario_data_directory.is_dir(), scenario_data_directory
    assert climate_policy_data_directory.is_dir(), climate_policy_data_directory
    
    # Collect results (turn into return statement upon completion of refactor)
    new_results = {
        # "battery_materials": {},
        # "battery_shares": {},
        # "battery_weights": {},
        # "lifetimes": {},
        # "maintenance_material_fractions": {},
        # "material_fractions": {},
        # "stocks": {},
        # "weights": {},
        "set_unit_flexible": "count"
    }

    # Reproduce old situation
    old_preprocessing_results = preprocess(base_directory, climate_policy_config, circular_economy_config)

    # Compare old and new outcomes
    old_summ = summarize_prep_data(old_preprocessing_results)
    new_summ = summarize_prep_data(new_results)
    # Check for each key, uncomment as refactor progresses
    # assert old_summ["battery_materials"] == new_summ["battery_materials"]
    # assert old_summ["battery_shares"] == new_summ["battery_shares"]
    # assert old_summ["battery_weights"] == new_summ["battery_weights"]
    # assert old_summ["lifetimes"] == new_summ["lifetimes"]
    # assert old_summ["maintenance_material_fractions"] == new_summ["maintenance_material_fractions"]
    # assert old_summ["material_fractions"] == new_summ["material_fractions"]
    # assert old_summ["stocks"] == new_summ["stocks"]
    # assert old_summ["weights"] == new_summ["weights"]

    return old_preprocessing_results