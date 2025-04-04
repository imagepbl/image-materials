import pytest
import warnings
from pathlib import Path

from imagematerials import vehicles as vhc
from imagematerials import buildings as bld

@pytest.fixture(scope="session")
def vhc_prep_data():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return vhc.preprocess(Path("data", "raw"))

@pytest.fixture(scope="session")
def bld_prep_data():
    return bld.preprocess(Path("IMAGE-Mat_old_version", "IMAGE-Mat", "BUMA"))

