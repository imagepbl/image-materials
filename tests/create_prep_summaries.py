#!/usr/bin/env python

import sys
import json
from pathlib import Path

from imagematerials.preprocessing import get_preprocessing_data
from imagematerials.util import (
    summarize_prep_data
)

base_directory = Path("data", "raw")

climate_policy_scenario_dir = base_directory.joinpath("image", "SSP2_baseline")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        sector = sys.argv[1]
    else:
        sector = "all"

    # Vehicles summary
    if sector in ["all", "vehicles"]:
        vhc_sector = get_preprocessing_data("vehicles", base_directory, climate_policy_scenario_dir=climate_policy_scenario_dir)
        summary_vhc = summarize_prep_data(vhc_sector.prep_data)
        with open(Path("tests", "data", "vehicles_summary.json"), "w", encoding="utf8") as handle:
            json.dump(summary_vhc, handle)

    # Buildings summary
    if sector in ["all", "buildings"]:
        bld_sector = get_preprocessing_data("buildings", base_directory, climate_policy_scenario_dir=climate_policy_scenario_dir)
        summary_bld = summarize_prep_data(bld_sector.prep_data)
        with open(Path("tests", "data", "buildings_summary.json"), "w", encoding="utf8") as handle:
            json.dump(summary_bld, handle)

    # Electricity summary
    if sector in ["all", "electricity"]:
        elc_sector = get_preprocessing_data("electricity", base_dir=base_directory,
                                            climate_policy_scenario_dir=climate_policy_scenario_dir)

        elc_sector_dict = {sector.name: sector.prep_data for sector in elc_sector}
        summary_elc = summarize_prep_data(elc_sector_dict)
        with open(Path("tests", "data", "electricity_summary.json"), "w", encoding="utf8") as handle:
            json.dump(summary_elc, handle)


    # Infrastructure summary
    if sector in ["all", "infrastructure"]:
        infra_sector = get_preprocessing_data("infrastructure", base_dir=base_directory,
                                            climate_policy_scenario_dir=climate_policy_scenario_dir)
        summary_infra = summarize_prep_data(infra_sector.prep_data)

        with open(Path("tests", "data", "infrastructure_summary.json"), "w", encoding="utf8") as handle:
            json.dump(summary_infra, handle)
