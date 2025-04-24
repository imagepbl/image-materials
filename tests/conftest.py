import warnings
from pathlib import Path

import pytest

from imagematerials import buildings as bld
from imagematerials import vehicles as vhc
from imagematerials.concepts import create_building_graph, create_vehicle_graph
from imagematerials.util import rebroadcast_prep_data


@pytest.fixture(scope="session")
def vhc_prep_data():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prep_data = vhc.preprocess(Path("data", "raw"))
    knowledge_graph = create_vehicle_graph()
    new_prep_data = rebroadcast_prep_data(prep_data, knowledge_graph, dim="Type", output_coords=prep_data["shares"].coords["Type"].values)
    new_prep_data = rebroadcast_prep_data(new_prep_data, knowledge_graph, dim="Region", output_coords=prep_data["shares"].coords["Region"].values)
    new_prep_data["knowledge_graph"] = knowledge_graph
    new_prep_data["weights"] = new_prep_data.pop("vehicle_weights")
    return new_prep_data

@pytest.fixture(scope="session")
def bld_prep_data():
    prep_data = bld.preprocess(Path("IMAGE-Mat_old_version", "IMAGE-Mat", "BUMA"))
    new_prep_data = {k: v for k, v in prep_data.items()}
    new_prep_data["knowledge_graph"] = create_building_graph()
    return new_prep_data
