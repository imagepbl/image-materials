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
