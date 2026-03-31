"""
FUMA Model – global dynamic fossil fuel material model

@Author: Sebastiaan Deetman & Yanan Liang
Affiliation: Institute of Environmental Sciences, Leiden University
Contact: yanan.liang@cml.leidenuniv.nl
Last update date: June 2025
Description:
This script initializes input parameters and loads data for the FUMA model.
"""

import pandas as pd

idx = pd.IndexSlice


def transport(fuel, fuel_use, transport_demand_per_kg, transport_weight):
    # IN: 1) fuel type (string); 2) fuel use in kg by region & time 3) transport demand in tkm/kg (or tkm/m3 for gas!) 4) transport weight per vehicle type per tkm
    # OUT: 1) tkm required on an annual basis (service provided by the stock on a continuous basis)

    modes = list(transport_demand_per_kg.columns)
    multi_index = pd.MultiIndex.from_product([modes, fuel_use.index])
    tkm = pd.DataFrame(index=multi_index, columns=fuel_use.columns)
    for mode in modes:
        tkm.loc[mode, :] = fuel_use.mul(
            transport_demand_per_kg.loc[fuel, mode]
        ).to_numpy()

    for mode in list(tkm.index.levels[0]):
        tkm.loc[idx[mode, :], :] = (
            tkm.loc[idx[mode, :], :] * transport_weight[mode] / 1000
        )  # now in kg

    tkm.index.names = ["type", "time"]
    transport_stocks = tkm

    transport_stocks = transport_stocks.assign(
        fuel=fuel, stage="transport", unit="kg"
    )
    transport_stocks = transport_stocks.set_index(
        ["fuel", "stage", "unit"], append=True
    )

    return transport_stocks
