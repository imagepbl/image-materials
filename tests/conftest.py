from pathlib import Path

import pytest

from imagematerials.preprocessing import get_preprocessing_data


@pytest.fixture(scope="session")
def vhc_sector():
    return get_preprocessing_data("vehicles", Path("data", "raw"))


@pytest.fixture(scope="session")
def bld_prep_data():
    climate_policy_scenario_dir = Path("data", "raw") / 'SSP2'
    circular_economy_scenario_dirs = {"slow": Path("data", "raw") / 'circular_economy_scenarios' / 'slow'}
    climate_policy_config = read_climate_policy_config(climate_policy_scenario_dir)
    circular_economy_config = read_circular_economy_config(circular_economy_scenario_dirs)
    prep_data = bld.preprocess(Path("data", "raw"), climate_policy_config, circular_economy_config)
    new_prep_data = {k: v for k, v in prep_data.items()}
    new_prep_data["knowledge_graph"] = create_building_graph()
    new_prep_data["shares"] = None
    return new_prep_data
