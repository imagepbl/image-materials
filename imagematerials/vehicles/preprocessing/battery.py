"""Battery related preprocessing functions"""
import xarray as xr
import pandas as pd

from imagematerials.vehicles.constants import (
    years_range
)
from imagematerials.vehicles.modelling_functions import interpolate
from imagematerials.vehicles.preprocessing.util import xarray_conversion


def get_battery_shares(general_data_path: str):
    # The share of the battery market (8 battery types used in vehicles)
    #this data is based on a Multi-Nomial-Logit market model & costs in
    # https://doi.org/10.1016/j.resconrec.2020.105200 - since this is
    # scenario dependent it's placed under the "IMAGE" scenario folder
    battery_shares_full: pd.DataFrame = pd.read_csv(
        general_data_path.joinpath("battery_share_inflow.csv"),
        index_col=0
    )
    battery_shares_full = battery_shares_full.loc[years_range]
    battery_shares = interpolate(battery_shares_full)
    return xarray_conversion(battery_shares, (["Cohort"], ["battery"],))


def get_battery_weights(data_path: str):
    # Using the 250 Wh/kg on the kWh of the various batteries a weight
    # (in kg) of the battery per vehicle category is determined
    battery_weights: pd.DataFrame = pd.read_csv(
        data_path.joinpath("battery_weights_kg.csv"),
        index_col=[0, 1]
    )
    battery_weights = interpolate(battery_weights.unstack())
    battery_weights = xarray_conversion(
        battery_weights,
        (["Cohort"], ["Type", "SubType"], {"Type": ["Type", "SubType"]})
    )

    # Set default battery weight to 0
    xr_default_battery = xr.DataArray(0.0, dims=("Cohort", "Type"),
                                      coords={
                                          "Cohort": battery_weights.coords["Cohort"],
                                           "Type": ["Vehicles"]})
    return xr.concat((battery_weights, xr_default_battery), dim="Type")


def get_battery_materials(data_path: str):
    # The material fraction of storage technologies (used to get the
    # vehicle battery composition)
    battery_materials: pd.DataFrame = pd.read_csv(
        data_path.joinpath("battery_materials.csv"),
        index_col=[0, 1]
    )
    battery_materials = interpolate(battery_materials.unstack())
    battery_materials = xarray_conversion(
        battery_materials,
        (["Cohort"], ["material", "battery"],)
    )

    # TODO: Check if this is correct
    bad_coords = battery_materials.coords["battery"]
    new_coords = [x if x != "LMO" else "LMO/LCO" for x in bad_coords.values]
    return battery_materials.assign_coords({"battery": new_coords})
