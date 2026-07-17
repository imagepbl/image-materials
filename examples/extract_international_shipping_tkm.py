"""Extract international shipping tonne-kilometres (raw and preprocessed).

The source data is the IMAGE/TIMER freight transport output file
`trp_frgt_Tkm.out` (in Mega tonne-km, i.e. 1e6 tkm), which contains 8 freight
modes for 28 "regions" (26 named IMAGE regions + 2 aggregate/unused rows).
Column 6 of that file is "international shipping".

This script writes a tidy CSV with, per year and region:
- the raw value straight out of the .out file
- the "preprocessed" value as produced by
  `imagematerials.vehicles.preprocessing.util.get_tonkms`, which is the
  function used inside `get_preprocessing_data("vehicles", ...)` in
  examples/vehicles.ipynb before international shipping tkm are converted
  into ship stocks.

For this particular variable the preprocessing step only relabels/reindexes
the data (no unit conversion or value changes happen before the ships-stock
step), so the two columns are expected to match for regions 1-26. Region 28
is the IMAGE "World total" (sum over the 26 regions) and region 27 is an
unused/empty row in the source file - both are kept in the output for
completeness and clearly labelled.
"""
from pathlib import Path

import pandas as pd

from imagematerials.constants import IMAGE_REGIONS
from imagematerials.read_mym import read_mym_df
from imagematerials.util import read_climate_policy_config
from imagematerials.vehicles.preprocessing.util import get_tonkms

REPO_ROOT = Path(__file__).resolve().parent.parent
CLIMATE_DATA_PATH = REPO_ROOT / "data" / "raw" / "image" / "SSP2_baseline"
RAW_FILE = CLIMATE_DATA_PATH / "EnergyServices" / "trp_frgt_Tkm.out"
OUT_PATH = Path(__file__).resolve().parent / "output" / "international_shipping_tkm.csv"

SHIPPING_TYPE = "international shipping"

# Region 1-26 are named IMAGE regions, 27 is an unused/empty row in the
# source file, 28 is the IMAGE "World total" (sum over regions 1-26).
REGION_LABELS = {i: region for i, region in enumerate(IMAGE_REGIONS, start=1)}
REGION_LABELS[27] = "Unused"
REGION_LABELS[28] = "World Total"


def get_raw_international_shipping() -> pd.DataFrame:
    """Read international shipping tkm directly from the .out file, unmodified."""
    df = read_mym_df(RAW_FILE)
    df = df.rename(columns={"DIM_1": "region"})
    raw = df[["time", "region", 6]].rename(columns={6: "ton_km_mega"})
    raw["region_name"] = raw["region"].map(REGION_LABELS)
    raw["shipping_type"] = SHIPPING_TYPE
    raw["source"] = "raw"
    return raw


def get_preprocessed_international_shipping() -> pd.DataFrame:
    """Run the project's own preprocessing (get_tonkms) and pull out the same column."""
    climate_policy_config = read_climate_policy_config(CLIMATE_DATA_PATH)
    tonkms_Mtkms = get_tonkms(CLIMATE_DATA_PATH, climate_policy_config)

    prep = tonkms_Mtkms[[SHIPPING_TYPE]].reset_index()
    prep = prep.rename(columns={SHIPPING_TYPE: "ton_km_mega"})
    prep["region_name"] = prep["region"].map(REGION_LABELS)
    prep["shipping_type"] = SHIPPING_TYPE
    prep["source"] = "preprocessed"
    return prep


def main():
    raw = get_raw_international_shipping()
    preprocessed = get_preprocessed_international_shipping()

    combined = pd.concat([raw, preprocessed], ignore_index=True)
    combined = combined.rename(columns={"time": "year"})
    combined = combined[
        ["year", "region", "region_name", "shipping_type", "source", "ton_km_mega"]
    ]
    combined = combined.sort_values(["source", "year", "region"]).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(combined)} rows to {OUT_PATH}")

    # Sanity check: raw and preprocessed values should match exactly, since
    # get_tonkms only relabels/reindexes the raw file for this variable.
    check = raw.merge(
        preprocessed, on=["time", "region"], suffixes=("_raw", "_prep")
    )
    mismatches = check[
        (check["ton_km_mega_raw"] - check["ton_km_mega_prep"]).abs() > 1e-6
    ]
    if len(mismatches):
        print(f"WARNING: {len(mismatches)} rows differ between raw and preprocessed data")
    else:
        print("Raw and preprocessed values match exactly for all years/regions.")


if __name__ == "__main__":
    main()
