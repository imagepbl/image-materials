#!/usr/bin/env python

import json
from pathlib import Path

from imagematerials.preprocessing import get_preprocessing_data
from imagematerials.util import (
    summarize_prep_data
)

test_scen = "SSP2_M_CP"

if __name__ == "__main__":
    # Vehicles summary
    vhc_sector = get_preprocessing_data("vehicles", Path("data", "raw", "image"), test_scen)
    summary_vhc = summarize_prep_data(vhc_sector.prep_data)
    with open(Path("tests", "data", "vehicles_summary.json"), "w", encoding="utf8") as handle:
        json.dump(summary_vhc, handle)

    # Buildings summary
    bld_sector = get_preprocessing_data("buildings", Path("data", "raw", "image"), test_scen)
    summary_bld = summarize_prep_data(bld_sector.prep_data)
    with open(Path("tests", "data", "buildings_summary.json"), "w", encoding="utf8") as handle:
        json.dump(summary_bld, handle)

    # Electricity summary
    VARIANT = "VLHO"
    SCEN = "SSP2"
    scen_folder = SCEN + "_" + VARIANT
    climate_policy_scenario_dir = Path("data", "raw", "image", test_scen)

    elc_sector = get_preprocessing_data("electricity", base_dir=Path("data", "raw"),
                                        climate_policy_scenario_dir=climate_policy_scenario_dir,
                                        standard_scenario=scen_folder)
    summary_elc = summarize_prep_data(elc_sector.prep_data)
    with open(Path("tests", "data", "electricity_summary.json"), "w", encoding="utf8") as handle:
        json.dump(summary_elc, handle)