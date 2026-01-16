import xarray as xr
import numpy as np
import pandas as pd
import prism


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
    get_passengerkms,
    get_tonkms,
    set_column_names,
    xarray_conversion
)


def get_vehicle_shares(climate_data_path: str, climate_policy_config, knowledge_graph_vehicle):
    # The tonne kilometres of freight vehicles of the IMAGE/TIMER SSP2
    # (in Mega Tkm)
    tonkms_Mtkms = get_tonkms(climate_data_path, climate_policy_config)

    # The passenger kilometres from the IMAGE/TIMER SSP2 (in Tera Pkm)
    passengerkms_Tpkms = get_passengerkms(climate_data_path, climate_policy_config)

    # The vehicle shares of buses of the SSP2                            MIND!
    # FOR the BL this is still the OLD SSP2 file REPLACE LATER
    buses_vshares: pd.DataFrame = read_mym_df(climate_data_path.joinpath(
        climate_policy_config['data_files']['transport']['passenger']['Vshare_bus']
    )).rename(columns={"DIM_1": "region"})
    buses_vshares = set_column_names(buses_vshares, bus_label)

    # The vehicle shares of passenger cars of the SSP2
    car_vshares: pd.DataFrame = read_mym_df(climate_data_path.joinpath(
        climate_policy_config['data_files']['transport']['passenger']['Vshare_car']
    )).rename(columns={"DIM_1": "region"})
    car_vshares = set_column_names(car_vshares, None)

    # The vehicle shares of trucks (medium) of the SSP2
    medtruck_vshares: pd.DataFrame = read_mym_df(climate_data_path.joinpath(
        climate_policy_config['data_files']['transport']['freight']['Vshare_MedTruck']
    )).rename(columns={"DIM_1": "region"})
    medtruck_vshares = set_column_names(medtruck_vshares, truck_label)

    # The vehicle shares of trucks (heavy) of the SSP2
    hvytruck_vshares: pd.DataFrame = read_mym_df(climate_data_path.joinpath(
        climate_policy_config['data_files']['transport']['freight']['Vshare_HvyTruck']
    )).rename(columns={"DIM_1": "region"})
    hvytruck_vshares = set_column_names(hvytruck_vshares, truck_label)

    # %% Vehicle shares of typical vehicles (i.e. specified different drivetrains)
    columns: pd.MultiIndex = pd.MultiIndex.from_product([typical_modes, drive_trains,
                                                         list(range(1, REGIONS + 1))],
                                                        names=["type", "fuel", "region"])
    vehicle_shares_typical = pd.DataFrame(
        0, index=years_range, columns=columns)

    # CARS
    # Aggregate car types into the vehicle_shares DataFrame
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

    # Different from the stocks: cutoff ([:-2]) not needed
    region_coords = np.sort(shares.coords["Region"].values.astype(int)).astype(str)

    # Use the shares to get the stocks for each of the subtypes.
    return knowledge_graph_vehicle.rebroadcast_xarray(shares, output_coords=region_coords,
                                                      dim="Region")
