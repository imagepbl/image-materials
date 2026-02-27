#!/usr/bin/env python

import json
from pathlib import Path

from imagematerials.preprocessing import get_preprocessing_data
from imagematerials.util import (
    summarize_prep_data
)

base_directory = Path("data", "raw")

climate_policy_scenario_dir = base_directory.joinpath("image", "SSP2_baseline")

if __name__ == "__main__":
    # Vehicles summary
    vhc_sector = get_preprocessing_data("vehicles", base_directory, climate_policy_scenario_dir=climate_policy_scenario_dir)
    summary_vhc = summarize_prep_data(vhc_sector.prep_data)
    with open(Path("tests", "data", "vehicles_summary.json"), "w", encoding="utf8") as handle:
        json.dump(summary_vhc, handle)

    # Buildings summary
    bld_sector = get_preprocessing_data("buildings", base_directory, climate_policy_scenario_dir=climate_policy_scenario_dir)
    summary_bld = summarize_prep_data(bld_sector.prep_data)
    with open(Path("tests", "data", "buildings_summary.json"), "w", encoding="utf8") as handle:
        json.dump(summary_bld, handle)

    # # Electricity summary
    # elc_sector = get_preprocessing_data("electricity", base_dir=base_directory,
    #                                     climate_policy_scenario_dir=climate_policy_scenario_dir)
    # summary_elc = summarize_prep_data(elc_sector.prep_data)
    # with open(Path("tests", "data", "electricity_summary.json"), "w", encoding="utf8") as handle:
    #     json.dump(summary_elc, handle)
