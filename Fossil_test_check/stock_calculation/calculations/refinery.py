# -*- coding: utf-8 -*-
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


def refinery_stocks(fuel, fuel_use, refinery_intensity_weight):
    refinery_stocks = pd.concat(
        [fuel_use.mul(refinery_intensity_weight.iloc[0])],
        keys=[refinery_intensity_weight.index[0]],
        names=["type"],
    )

    refinery_stocks = refinery_stocks.assign(
        fuel=fuel, stage="refinery", unit="kg"
    )
    refinery_stocks = refinery_stocks.set_index(
        ["fuel", "stage", "unit"], append=True
    )
    return refinery_stocks
