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

# based on the indexed development of fossil fuel use and known current infrastructure stock (e.g pipeline length or stargae capacity), derive the infrastructure stock development
# Assumption: linear response between fuel consumption and pipeline requirement
# MIND: because of this methodological choice, regions with no infrastructure (i.e. East Africa pipelines) do not develop any infrastructure throughout any scenario


def pipelines_indexed_stock_growth(fuel, fuel_use, pipeline_length, baseyear):
    indexed_growth = fuel_use.div(fuel_use.loc[baseyear], axis=1)

    type_name = pipeline_length.columns[0]
    type_one = pd.concat(
        [indexed_growth.mul(pipeline_length[type_name], axis=1)],
        keys=[type_name],
        names=["type"],
    )
    type_name = pipeline_length.columns[1]
    type_two = pd.concat(
        [indexed_growth.mul(pipeline_length[type_name], axis=1)],
        keys=[type_name],
        names=["type"],
    )

    stock_growth = pd.concat([type_one, type_two])
    stock_growth.index.names = ["type", "time"]
    stock_growth = stock_growth.assign(fuel=fuel, stage="pipelines", unit="km")
    stock_growth = stock_growth.set_index(
        ["fuel", "stage", "unit"], append=True
    )
    return stock_growth
