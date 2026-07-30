"""Utility functions for calculating vehicle shares used by stocks_prism module.

This module provides functions to compute vehicle technology and fuel type shares
from climate data, which are time-varying and depend on policy configurations.
"""

import xarray as xr
import numpy as np
import pandas as pd

from imagematerials.read_mym import read_mym_df
from imagematerials.vehicles.constants import (
    REGIONS,
    bus_label,
    bus_label_HEV,
    bus_label_ICE,
    car_collists,
    drive_trains,
    truck_label,
    truck_label_BEV,
    truck_label_FCV,
    truck_label_HEV,
    truck_label_ICE,
    truck_label_PHEV,
    typical_modes,
    years_range
)
from imagematerials.vehicles.modelling_functions import interpolate
from imagematerials.vehicles.preprocessing.util import (
    set_column_names,
    xarray_conversion,
    get_passengerkms,
    get_tonnekms,
)


def get_vehicle_shares_prism(timer_vehicle_shares, passenger_kms, tonne_kms, knowledge_graph_vehicle):
    """Calculate vehicle technology and fuel type shares from climate data.

    Parameters
    ----------
    climate_data_path : Path
        Path to climate data directory
    climate_policy_config : dict
        Configuration dictionary containing paths to vehicle share data files
    knowledge_graph_vehicle : KnowledgeGraph
        Vehicle knowledge graph for rebroadcasting shares to regions

    Returns
    -------
    xr.DataArray
        Vehicle shares with dimensions (Time, Type, SubType, Region)
    """
    # The tonne kilometres of freight vehicles of the IMAGE/TIMER SSP2
    # (in Mega Tkm)
    # TODO: implement get_tonnekms correctly
    tonkms_Mtkms = get_tonnekms(tonne_kms)

    # The passenger kilometres from the IMAGE/TIMER SSP2 (in Tera Pkm)
    # TODO: implement get_passengerkms correctly
    passengerkms_Tpkms = get_passengerkms(passenger_kms)

    # TODO: Change to just a single time
    # Vehicle shares of typical vehicles (specified by drivetrain)
    columns: pd.MultiIndex = pd.MultiIndex.from_product([typical_modes, drive_trains,
                                                         list(range(1, REGIONS + 1))],
                                                        names=["type", "fuel", "region"])
    vehicle_shares_typical = pd.DataFrame(
        0, index=years_range, columns=columns)
    vehicle_shares_typical = vehicle_shares_typical.sort_index(axis=1)

    # TODO: Below this line: everything that used to come from a file we
    # imported (buses_vshares, etc.) needs to come from the TimeVariable
    # (timer_vehicle_shares) now.

    # CARS - Aggregate car types into the vehicle_shares DataFram
    for fuel, collist in car_collists.items():
        vehicle_shares_typical[('Cars', fuel)] = \
            car_vshares[collist].sum(axis=1).unstack()

    vehicle_shares_typical[('Cars', 'Trolley')] = 0  # Assuming no trolley cars

    # BUSES and TRUCKS
    # Sum of all buses except Trolleys
    midi_sum = buses_vshares[list(filter(lambda x: x != 'BusElecTrolley',
                                         bus_label))].sum(axis=1)

    # Fill the DataFrame with bus data
    vehicle_shares_typical[('Regular Buses', 'ICE')] = \
        buses_vshares[bus_label_ICE].sum(axis=1).unstack()
    vehicle_shares_typical[('Regular Buses', 'HEV')] = \
        buses_vshares[bus_label_HEV].sum(axis=1).unstack()
    vehicle_shares_typical[('Regular Buses', 'BEV')] = \
        buses_vshares['BusBattElectric'].unstack()
    vehicle_shares_typical[('Regular Buses', 'Trolley')] = \
        buses_vshares['BusElecTrolley'].unstack()

    vehicle_shares_typical[('Midi Buses', 'ICE')] = \
        buses_vshares[bus_label_ICE].sum(axis=1).div(midi_sum).unstack()
    vehicle_shares_typical[('Midi Buses', 'HEV')] = \
        buses_vshares[bus_label_HEV].sum(axis=1).div(midi_sum).unstack()
    vehicle_shares_typical[('Midi Buses', 'BEV')] = \
        buses_vshares['BusBattElectric'].div(midi_sum).unstack()
    vehicle_shares_typical[('Midi Buses', 'Trolley')] = 0

    vehicle_shares_typical.loc[:, pd.IndexSlice[[
        'Regular Buses', 'Midi Buses'], ['FCV', 'PHEV']]] = 0

    # Fill the DataFrame with truck data
    truck_data = {
        'Medium Freight Trucks': medtruck_vshares,
        'Heavy Freight Trucks': hvytruck_vshares,
        # MIND Assumption: used MFT as a market-share for LCVs
        'Light Commercial Vehicles': medtruck_vshares
    }

    for truck_type, truck_df in truck_data.items():
        vehicle_shares_typical[(truck_type, 'ICE')] = \
            truck_df[truck_label_ICE].sum(axis=1).unstack()
        vehicle_shares_typical[(truck_type, 'HEV')] = \
            truck_df[truck_label_HEV].sum(axis=1).unstack()
        vehicle_shares_typical[(truck_type, 'PHEV')] = \
            truck_df[truck_label_PHEV].sum(axis=1).unstack()
        vehicle_shares_typical[(truck_type, 'BEV')] = \
            truck_df[truck_label_BEV].sum(axis=1).unstack()
        vehicle_shares_typical[(truck_type, 'FCV')] = \
            truck_df[truck_label_FCV].sum(axis=1).unstack()
        vehicle_shares_typical[(truck_type, 'Trolley')] = 0

    vehicle_shares_typical = interpolate(vehicle_shares_typical, change='no')
    shares = xarray_conversion(
        vehicle_shares_typical,
        (["Time"], ["Type", "SubType", "Region"], {"Type": ["Type", "SubType"]})
    )

    # Rebroadcast shares to standard regions
    region_coords = np.sort(shares.coords["Region"].values.astype(int)).astype(str)
    return knowledge_graph_vehicle.rebroadcast_xarray(shares, output_coords=region_coords,
                                                      dim="Region")
