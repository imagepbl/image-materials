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
    flag_enabled,
    pandas_to_xarray
)
from imagematerials.vehicles.modelling_functions import (
    idx,
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


def get_lifetimes(data_path: str, circular_economy_config: dict, resource_efficiency_flags: dict = None):
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
    if flag_enabled(resource_efficiency_flags, "vehicles", "FlagLifetimeExtension"):
        print("Implementing FlagLifetimeExtension for Vehicles (extend lifetimes)")
        flag_config = circular_economy_config['vehicles']['FlagLifetimeExtension']
        target_year = flag_config['target_year']
        base_year = flag_config['base_year']
        lifetime_increase = flag_config['lifetime_increase_percent']
        implementation_rate = flag_config['implementation_rate']
        # possibilities for implementation rate are: linear, immediate, s-curve

        lifetimes_vehicles = scenario_change(lifetimes_vehicles, base_year, target_year,
            lifetime_increase, implementation_rate, "lifetime")

        logging.debug("implemented FlagLifetimeExtension for Vehicles (extend lifetimes)")

    # In lifetimes_years.csv the folded-normal 'stdev' is given as a fraction of
    # the mean lifetime (see the electricity module, which multiplies the mean by
    # STD_LIFETIMES_ELECTR before convert_lifetime). convert_lifetime /
    # FoldedNormalDistribution.get_param, however, expects an absolute standard
    # deviation in years. Without this conversion the scale of the folded normal
    # collapses to ~0.2-0.3 years, turning the lifetime distribution into a near
    # delta spike at the mean. Convert the fraction to absolute years here, after
    # any lifetime extension has been applied to the mean, so the stdev/mean ratio
    # stays fixed at the intended fraction.
    stdev_modes = lifetimes_vehicles.columns.get_level_values('data') == 'stdev'
    means = lifetimes_vehicles.loc[:, idx[:, 'mean']].rename(
        columns={'mean': 'stdev'}, level='data')
    lifetimes_vehicles.loc[:, stdev_modes] = (
        lifetimes_vehicles.loc[:, stdev_modes] * means
    )

    data_xarray = pandas_to_xarray(lifetimes_vehicles, unit_mapping)
    return convert_lifetime(data_xarray)
