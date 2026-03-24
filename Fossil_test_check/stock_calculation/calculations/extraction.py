"""
FUMA Model – global dynamic fossil fuel material model

@Author: Yanan Liang
Affiliation: Institute of Environmental Sciences, Leiden University
Contact: yanan.liang@cml.leidenuniv.nl
Last update date: June 2025
Description:
This script initializes input parameters and loads data for the FUMA model.
"""

import pandas as pd


# calculates the total weight of the in-use infrastructure stock
# expected input data on fuel use should be in the same units as the production weight intensity data (tonne of coal, m3 of gas)
def extraction_stocks(
    fuel, fuel_use, production_intensity_weight, share_per_type
):

    extraction_stocks_one = pd.concat(
        [
            fuel_use.mul(share_per_type[share_per_type.columns[0]], axis=1).mul(
                production_intensity_weight.iloc[0]
            )
        ],
        keys=[production_intensity_weight.index[0]],
        names=["type"],
    )
    extraction_stocks_two = pd.concat(
        [
            fuel_use.mul(share_per_type[share_per_type.columns[1]], axis=1).mul(
                production_intensity_weight.iloc[1]
            )
        ],
        keys=[production_intensity_weight.index[1]],
        names=["type"],
    )
    extraction_stocks = pd.concat(
        [extraction_stocks_one, extraction_stocks_two]
    )
    extraction_stocks.index.names = ["type", "time"]
    extraction_stocks = extraction_stocks.assign(
        fuel=fuel, stage="extraction", unit="kg"
    )
    extraction_stocks = extraction_stocks.set_index(
        ["fuel", "stage", "unit"], append=True
    )

    return extraction_stocks
