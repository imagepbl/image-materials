import xarray as xr
import numpy as np
import pandas as pd
import logging

# from imagematerials.constants import IMAGE_REGIONS
from imagematerials.read_mym import read_mym_df
from imagematerials.vehicles.constants import (
    cap_adjustment,
    pkms_label,
    tkms_label,
    unit_mapping,
    years_range
)
from imagematerials.util import (
    dataset_to_array,
    convert_lifetime,
    pandas_to_xarray
)
from imagematerials.vehicles.modelling_functions import (
    interpolate,
    scenario_change
)


def xarray_conversion(df: pd.DataFrame, dimensions):
    """Temporary function. Should be removed once pandas has fully been
    replaced by xarray"""
    data_xar_dataset = pandas_to_xarray(df, unit_mapping)
    return dataset_to_array(data_xar_dataset, *dimensions)


def set_column_names(df: pd.DataFrame, label):
    if df is not None:
        # Filter DataFrame for the output years
        df = df[df["time"].isin(years_range)]

        # Set multi-index based on the first two columns
        df.set_index(["time", "region"], inplace=True)

        # Insert column descriptions if available
        if label is not None:
            df.columns = label
    return df


def get_tonkms(climate_data_path: str, climate_policy_config):
    # The tonne kilometres of freight vehicles of the IMAGE/TIMER SSP2
    # (in Mega Tkm)
    tonkms_Mtkms: pd.DataFrame = read_mym_df(climate_data_path.joinpath(
        climate_policy_config['data_files']['transport']['freight']['Tkm']
    )).rename(columns={"DIM_1": "region"})
    return set_column_names(tonkms_Mtkms, tkms_label)


def get_passengerkms(climate_data_path: str, climate_policy_config):
    # The passenger kilometres from the IMAGE/TIMER SSP2 (in Tera Pkm)
    passengerkms_Tpkms: pd.DataFrame = read_mym_df(climate_data_path.joinpath(
        climate_policy_config['data_files']['transport']['passenger']['kilometers']
    )).rename(columns={"DIM_1": "region"})
    return set_column_names(passengerkms_Tpkms, pkms_label)


def get_ship_capacity(general_data_path: str):
    # boat capacity in tons changing Data is a combination of EQUASIS
    # Gross Tonnage and UNCTAD Dead-Weigh-Tonnage per Gross Tonnage
    cap_of_boats: pd.DataFrame = pd.read_csv(
        general_data_path.joinpath("ships", "capacity_ton_boats.csv"),
        index_col="t"
    ).sort_index(axis=0)
    cap_of_boats_yrs = interpolate(cap_of_boats, change='no')
    return cap_of_boats_yrs.mul(cap_adjustment, axis=1)


def get_lifetimes(data_path: str, circular_economy_config: dict):
    """Get vehicle lifetimes from CSV and squeeze it in the right format.

    Parameters
    ----------
    data_path
        Path to the directory containing scenario data
    circular_economy_config
        Dictionary with config for circular economy scenario

    Returns
    -------
        A xarray DataArray with vehicle lifetimes

    Notes
    -----
    TODO: immediately read into xarray, bypass pandas
    """
    lifetimes_vehicles: pd.DataFrame = pd.read_csv(
        data_path.joinpath("lifetimes_years.csv"), index_col=[0, 1])

    # interpolate & complete series
    lifetimes_vehicles = lifetimes_vehicles.rename_axis('mode', axis=1).stack()
    lifetimes_vehicles = lifetimes_vehicles[(lifetimes_vehicles.T != 0)]
    lifetimes_vehicles = lifetimes_vehicles.unstack(['mode', 'data'])
    lifetimes_vehicles = interpolate(pd.DataFrame(lifetimes_vehicles))
    
    # Calculate extended lifetime per mode
    if 'slow' in circular_economy_config.keys():
        target_year = circular_economy_config['slow']['vehicles']['target_year']
        base_year = circular_economy_config['slow']['vehicles']['base_year']
        lifetime_increase = \
            circular_economy_config['slow']['vehicles']['lifetime_increase_percent_slow']
        implementation_rate = circular_economy_config['slow']['vehicles']['implementation_rate']
        # possibilities for implementation rate are: linear, immediate, s-curve

        lifetimes_vehicles = scenario_change(lifetimes_vehicles, base_year, target_year,
            lifetime_increase, implementation_rate, "lifetime")
        
        logging.debug("implemented 'slow' for Vehicles (extend lifetimes)")

    data_xarray = pandas_to_xarray(lifetimes_vehicles, unit_mapping)
    return convert_lifetime(data_xarray)
