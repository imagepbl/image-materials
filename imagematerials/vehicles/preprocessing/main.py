import warnings
from pathlib import Path

import xarray as xr
import numpy as np

from imagematerials.constants import IMAGE_REGIONS

from imagematerials.vehicles.constants import FOLDER
from imagematerials.concepts import create_building_graph
from imagematerials.concepts import create_region_graph

from imagematerials.vehicles.preprocessing.preprocessing_old import preprocess

def vehicles_preprocessing(base_directory: str, climate_policy_config: dict,
                           circular_economy_config: dict, image_scenario: str = FOLDER):
    # Reproduce old situation
    return preprocess(base_directory, climate_policy_config, circular_economy_config)
