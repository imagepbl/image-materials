#!/usr/bin/env python

import json
from pathlib import Path

from imagematerials.preprocessing import get_preprocessing_data
from imagematerials.util import (
    summarize_prep_data,
)

if __name__ == "__main__":
    # Vehicles summary
    vhc_sector = get_preprocessing_data("vehicles", Path("data", "raw"))
    summary_vhc = summarize_prep_data(vhc_sector.prep_data)
    with open(Path("tests", "data", "vehicles_summary.json"), "w", encoding="utf8") as handle:
        json.dump(summary_vhc, handle)

    # Buildings summary
    bld_sector = get_preprocessing_data("buildings", Path("data", "raw"))
    summary_bld = summarize_prep_data(bld_sector.prep_data)
    with open(Path("tests", "data", "buildings_summary.json"), "w", encoding="utf8") as handle:
        json.dump(summary_bld, handle)
