import logging
import warnings
from pathlib import Path

import xarray as xr
import numpy as np
import pandas as pd

from imagematerials.constants import IMAGE_REGIONS

from imagematerials.vehicles.constants import (
    FOLDER,
    unit_mapping
)
from imagematerials.concepts import create_region_graph
from imagematerials.util import (
    convert_lifetime,
    pandas_to_xarray
)

from imagematerials.vehicles.modelling_functions import (
    interpolate,
    scenario_change
)

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
        "lifetimes": get_lifetimes(scenario_data_directory, circular_economy_config),
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
    assert old_summ["lifetimes"] == new_summ["lifetimes"]
    # assert old_summ["maintenance_material_fractions"] == new_summ["maintenance_material_fractions"]
    # assert old_summ["material_fractions"] == new_summ["material_fractions"]
    # assert old_summ["stocks"] == new_summ["stocks"]
    # assert old_summ["weights"] == new_summ["weights"]

    return old_preprocessing_results


def get_lifetimes(data_path: str, circular_economy_config: dict):
    """Get vehicle lifetimes from CSV and squeeze it in the right format.

    Parameters
    ----------
    data_path
        Path to the directory containing scenario data
    circular_economy_config
        Dictionary with config for circular economy scenario

    Returns
    -------
        A xarray DataArray with vehicle lifetimes

    Notes
    -----
    TODO: immediately read into xarray, bypass pandas
    """
    lifetimes_vehicles: pd.DataFrame = pd.read_csv(
        data_path.joinpath("lifetimes_years.csv"), index_col=[0, 1])

    # interpolate & complete series
    lifetimes_vehicles = lifetimes_vehicles.rename_axis('mode', axis=1).stack()
    lifetimes_vehicles = lifetimes_vehicles[(lifetimes_vehicles.T != 0)]
    lifetimes_vehicles = lifetimes_vehicles.unstack(['mode', 'data'])
    lifetimes_vehicles = interpolate(pd.DataFrame(lifetimes_vehicles))
    
    # Calculate extended lifetime per mode
    if 'slow' in circular_economy_config.keys():
        target_year = circular_economy_config['slow']['vehicles']['target_year']
        base_year = circular_economy_config['slow']['vehicles']['base_year']
        lifetime_increase = \
            circular_economy_config['slow']['vehicles']['lifetime_increase_percent_slow']
        implementation_rate = circular_economy_config['slow']['vehicles']['implementation_rate']
        # possibilities for implementation rate are: linear, immediate, s-curve

        lifetimes_vehicles = scenario_change(lifetimes_vehicles, base_year, target_year,
            lifetime_increase, implementation_rate, "lifetime")
        
        logging.debug("implemented 'slow' for Vehicles (extend lifetimes)")

    data_xarray = pandas_to_xarray(lifetimes_vehicles, unit_mapping)
    return convert_lifetime(data_xarray)
