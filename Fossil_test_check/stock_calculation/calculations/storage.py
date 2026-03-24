# -*- coding: utf-8 -*-
"""
Created on Tue Oct  4 14:30:47 2022

@author: thije wubs
"""
import pandas as pd


def storage_indexed_stocks_growth(fuel, fuel_use, storage_capacity, baseyear):
    indexed_growth = fuel_use.div(fuel_use.loc[baseyear], axis=1)

    type_name = storage_capacity.columns[0]
    type_one = pd.concat(
        [indexed_growth.mul(storage_capacity[type_name], axis=1)],
        keys=[type_name],
        names=["type"],
    )  # multiplies the storage capacity of the base year with the indexed oil demand
    type_name = storage_capacity.columns[1]
    type_two = pd.concat(
        [indexed_growth.mul(storage_capacity[type_name], axis=1)],
        keys=[type_name],
        names=["type"],
    )

    stock_growth = pd.concat([type_one, type_two])
    stock_growth.index.names = ["type", "time"]
    stock_growth = stock_growth.assign(fuel=fuel, stage="storage", unit="m3")
    stock_growth = stock_growth.set_index(
        ["fuel", "stage", "unit"], append=True
    )
    return stock_growth
