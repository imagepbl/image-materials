import logging
import warnings
from pathlib import Path

import xarray as xr
import numpy as np
import pandas as pd

from imagematerials.vehicles.constants import (
    FOLDER
)
from imagematerials.concepts import create_vehicle_graph
from imagematerials.vehicles.preprocessing.battery import (
    get_battery_materials,
    get_battery_shares,
    get_battery_weights
)
from imagematerials.vehicles.preprocessing.materials import (
    get_maintenance_materials,
    get_material_fractions
)
from imagematerials.vehicles.preprocessing.util import get_lifetimes
from imagematerials.vehicles.preprocessing.weights import get_weights
from imagematerials.vehicles.preprocessing.stock_info_prism import get_vehicle_stock_info

def vehicles_preprocessing(base_directory: str, climate_policy_config: dict,
                           circular_economy_config: dict, image_scenario: str = FOLDER):
    # Preparing directory shorthands
    base_directory = Path(base_directory)
    general_data_directory = base_directory.joinpath("vehicles", "standard_data")
    scenario_data_directory = base_directory.joinpath("vehicles", image_scenario)

    # Sanity check
    assert general_data_directory.is_dir(), general_data_directory

    # Create vehicle knowledge graph
    knowledge_graph_vehicle = create_vehicle_graph()

    # Get stock preprocessing info (conversion factors, first year, market share)
    conversion_factor_tkms, first_year_vehicle, market_share = get_vehicle_stock_info(
        scenario_data_directory, general_data_directory, circular_economy_config
    )

    # Collect results
    return {
        "battery_materials": get_battery_materials(scenario_data_directory),
        "battery_shares": get_battery_shares(general_data_directory),
        "battery_weights": get_battery_weights(scenario_data_directory),
        "knowledge_graph": knowledge_graph_vehicle,
        "lifetimes": get_lifetimes(scenario_data_directory, circular_economy_config),
        "maintenance_material_fractions": get_maintenance_materials(general_data_directory),
        "material_fractions": get_material_fractions(scenario_data_directory),
        # Stock-related preprocessing outputs
        "conversion_factor_tkms": conversion_factor_tkms,
        "first_year_vehicle": first_year_vehicle,
        "market_share": market_share,
        "weights": get_weights(scenario_data_directory, general_data_directory,
                               circular_economy_config),
        "set_unit_flexible": "count"
    }