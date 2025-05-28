#!/usr/bin/env python

import json
import warnings
from pathlib import Path

from imagematerials import buildings as bld
from imagematerials import vehicles as vhc
from imagematerials.concepts import create_building_graph, create_vehicle_graph
from imagematerials.util import (
    read_circular_economy_config,
    read_climate_policy_config,
    rebroadcast_prep_data,
    summarize_prep_data,
)

if __name__ == "__main__":
    # Vehicles summary
    climate_policy_scenario_dir = Path("data", "raw") / 'SSP2'
    circular_economy_scenario_dirs = {"slow": Path("data", "raw") / 'circular_economy_scenarios' / 'slow'}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        climate_policy_config = read_climate_policy_config(climate_policy_scenario_dir)
        circular_economy_config = read_circular_economy_config(circular_economy_scenario_dirs)

        prep_data = vhc.preprocess(Path("data", "raw"), climate_policy_config, circular_economy_config)
    knowledge_graph = create_vehicle_graph()
    new_prep_data = rebroadcast_prep_data(prep_data, knowledge_graph, dim="Type", output_coords=prep_data["shares"].coords["Type"].values)
    new_prep_data = rebroadcast_prep_data(new_prep_data, knowledge_graph, dim="Region", output_coords=prep_data["shares"].coords["Region"].values)
    new_prep_data["knowledge_graph"] = knowledge_graph
    new_prep_data["weights"] = new_prep_data.pop("vehicle_weights")

    summary_vhc = summarize_prep_data(new_prep_data)
    with open(Path("tests", "data", "vehicles_summary.json"), "w", encoding="utf8") as handle:
        json.dump(summary_vhc, handle)

    # Buildings summary
    prep_data_bld = bld.preprocess(Path("IMAGE-Mat_old_version", "IMAGE-Mat", "BUMA"))
    new_prep_data = {k: v for k, v in prep_data_bld.items()}
    new_prep_data["knowledge_graph"] = create_building_graph()

    summary_bld = summarize_prep_data(new_prep_data)
    with open(Path("tests", "data", "buildings_summary.json"), "w", encoding="utf8") as handle:
        json.dump(summary_bld, handle)
