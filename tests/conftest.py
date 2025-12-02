from pathlib import Path

import pytest

from imagematerials.preprocessing import get_preprocessing_data


@pytest.fixture(scope="session")
def vhc_sector():
    return get_preprocessing_data("vehicles", Path("data", "raw"))


@pytest.fixture(scope="session")
def vhc_prep_data(vhc_sector):
    return vhc_sector.prep_data

@pytest.fixture(scope="session")
def bld_sector():
    return get_preprocessing_data("buildings", Path("data", "raw"))

@pytest.fixture(scope="session")
def bld_prep_data(bld_sector):
    return bld_sector.prep_data

@pytest.fixture(scope="session")
def elc_sector():
    path_base = Path("data", "raw")
    VARIANT = "VLHO"
    SCEN = "SSP2"
    scen_folder = SCEN + "_" + VARIANT
    climate_policy_scenario_dir = Path(path_base, "image", scen_folder)
    return get_preprocessing_data("electricity", base_dir=path_base,
                                  climate_policy_scenario_dir=climate_policy_scenario_dir)

@pytest.fixture(scope="session")
def elc_prep_data(elc_sector):
    return elc_sector.prep_data
