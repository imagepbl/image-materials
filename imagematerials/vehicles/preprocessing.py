"""_summary_

Returns:
    _type_: _description_
"""
# %%
import argparse
import os
from collections import defaultdict
from pathlib import Path
from typing import Union

import pandas as pd
import pint
import xarray as xr
import numpy as np

from imagematerials.distribution import ALL_DISTRIBUTIONS, NAME_TO_DIST
from imagematerials.read_mym import read_mym_df
from imagematerials.util import dataset_to_array, pandas_to_xarray
from imagematerials.vehicles.constants import (
    END_YEAR,
    FOLDER,
    LIGHT_COMMERCIAL_VEHICLE_SHARE,
    LOAD_FACTOR,
    MEGA_TO_TERA,
    PKMS_TO_VKMS,
    PROJECT,
    REGIONS,
    SCEN,
    SHIPS_YEARS_RANGE,
    START_YEAR,
    TONNES_TO_KGS,
    all_modes,
    bus_label,
    bus_label_HEV,
    bus_label_ICE,
    cap_adjustment,
    car_collists,
    drive_trains,
    mile_adjustment,
    pkms_label,
    tkms_label,
    truck_label,
    truck_label_BEV,
    truck_label_FCV,
    truck_label_HEV,
    truck_label_ICE,
    truck_label_PHEV,
    typical_modes,
    unit_mapping,
    years_range,
)
from imagematerials.vehicles.modelling_functions import interpolate, tkms_to_nr_of_vehicles_fixed


def preprocess(base_dir: str, climate_policy_config: dict, circular_economy_config: dict):
    """Wrapper function for the preprocessing part of the VEMA script.

    Args:
        base_dir (Optional[str], optional): _description_. Defaults to os.getcwd().

    Returns:
        _type_: _description_
    """
    base_path = Path(base_dir)

    # %%
    base_input_data_path = base_path.joinpath("vehicles")
    standard_input_data_path = base_input_data_path.joinpath("standard_data")

    idx = pd.IndexSlice          # needed for slicing multi-index

    target_year = circular_economy_config['config_file_path'] / circular_economy_config['vehicles']['target_year']
    base_year = circular_economy_config['config_file_path'] /circular_economy_config['vehicles']['base_year']
    circular_economy_scenario = circular_economy_config['config_file_path'] /circular_economy_config['vehicles']['circular_economy_scenario']

    # Reading all csv files for vehicles and ships that are external to IMAGE

    # 1) scenario independent data
    load: pd.DataFrame = pd.read_csv(standard_input_data_path. joinpath("load_pass_and_tonnes.csv"))  
    # TODO: add description again here!
    loadfactor: pd.DataFrame = pd.read_csv(standard_input_data_path. joinpath("loadfactor_percentages.csv"))  
    # Percentage of the maximum load that is on average
    market_share: pd.DataFrame = pd.read_csv(standard_input_data_path. joinpath("fraction_tkm_pkm.csv"))  
    # Percentage of tonne-/passengerkilometres
    first_year_vehicle: pd.DataFrame = pd.read_csv(standard_input_data_path. joinpath("first_year_vehicle.csv"))  
    # first year of operation per vehicle-type
    # 1807 was originally in the dataframe
    battery_shares_full: pd.DataFrame = pd.read_csv(
        standard_input_data_path. joinpath("battery_share_inflow.csv"), index_col=0)
    # The share of the battery market (8 battery types used in vehicles), this data is based on a Multi-Nomial-Logit
    # market model & costs in https://doi.org/10.1016/j.resconrec.2020.105200 - since this is scenario dependent it's
    # placed under the "IMAGE" scenario folder

    maintenance_material : pd.DataFrame = pd.read_csv(
        standard_input_data_path. joinpath("maintenance_passenger_cars.csv"), index_col=0)

    # Files related to the international shipping
    nr_of_boats: pd.DataFrame = pd.read_csv(
        standard_input_data_path. joinpath(
            "ships",
            "number_of_boats.csv"),
        index_col="t").sort_index(
            axis=0)
    # number of boats in the global merchant fleet (2005-2018) changing Data
    # by EQUASIS
    cap_of_boats: pd.DataFrame = pd.read_csv(
        standard_input_data_path. joinpath(
            "ships",
            "capacity_ton_boats.csv"),
        index_col="t").sort_index(
            axis=0)
    # boat capacity in tons changing Data is a combination of EQUASIS Gross Tonnage and
    # UNCTAD Dead-Weigh-Tonnage per Gross Tonnage
    loadfactor_boats: pd.DataFrame = pd.read_csv(
        standard_input_data_path. joinpath(
            "ships",
            "loadfactor_boats.csv"),
        index_col="t").sort_index(
            axis=0)
    # loadfactor of boats (fraction) fixed Data is based on Ecoinvent report
    # 14 on Transport (Table 8-19)
    mileage_boats: pd.DataFrame = pd.read_csv(
        standard_input_data_path. joinpath(
            "ships",
            "mileage_kmyr_boats.csv"),
        index_col="t").sort_index(
            axis=0)
    # mileage of boats in km/yr (per ship) fixed Data is based on Ecoinvent
    # report 14 on Transport (Table 8-19)
    weight_boats: pd.DataFrame = pd.read_csv(
        standard_input_data_path. joinpath(
            "ships",
            "weight_percofcap_boats.csv"),
        index_col="t").sort_index(
            axis=0)
    # weight of boats as a percentage of the capacity (%) fixed Data is based on Ecoinvent report 14 on Transport
    # (section 8.4.1)

    # 2) scenario dependent data
    lifetimes_vehicles: pd.DataFrame = pd.read_csv(
        base_input_data_path. joinpath(
            FOLDER, "lifetimes_years.csv"), index_col=[
            0, 1])
    # Average End-of-Life of vehicles in years, this file also contains the setting for the choice of distribution and
    # other lifetime related settings (standard devition, or alternative
    # parameterisation)

    lifetime_increase = circular_economy_config['config_file_path']/ circular_economy_config['vehicles']['lifetime_increase_percent_slow']

    kilometrage: pd.DataFrame = pd.read_csv(base_input_data_path.
                                            joinpath(
                                                FOLDER, "kilometrage.csv"),
                                            index_col="t")
    # kilometrage of passenger cars in kms/yr
    kilometrage_midi_bus: pd.DataFrame = pd.read_csv(
        base_input_data_path. joinpath(
            FOLDER,
            "kilometrage_midi.csv"),
        index_col="t")  # kilometrage of midi-buses in kms/yr
    kilometrage_bus: pd.DataFrame = pd.read_csv(
        base_input_data_path. joinpath(
            FOLDER,
            "kilometrage_bus.csv"),
        index_col="t")  # kilometrage of regular buses in kms/yr
    mileages: pd.DataFrame = pd.read_csv(
        base_input_data_path. joinpath(
            FOLDER,
            "kilometrage_per_year.csv"),
        index_col="t")  # Km/year of all the vehicles (buses & cars have region-specific files)

    # weight and materials related data
    vehicle_weight_kg_simple: pd.DataFrame = pd.read_csv(
        base_input_data_path. joinpath(
            FOLDER,
            "vehicle_weight_kg_simple.csv"),
        index_col=0)  # Weight of a single vehicle of each type in kg
    vehicle_weight_kg_typical: pd.DataFrame = pd.read_csv(
        base_input_data_path. joinpath(
            FOLDER, "vehicle_weight_kg_typical.csv"), index_col=[
            0, 1])  # Weight of a single vehicle of each type in kg
    material_fractions: pd.DataFrame = pd.read_csv(
        base_input_data_path. joinpath(
            FOLDER, "material_fractions_simple.csv"), index_col=[
            0, 1])  # Material fractions in percentages
    material_fractions_type: pd.DataFrame = pd.read_csv(
        base_input_data_path. joinpath(
            FOLDER, "material_fractions_typical.csv"), index_col=[
            0, 1], header=[
                0, 1])
    # Material fractions in percentages, by vehicle sub-type
    battery_weights: pd.DataFrame = pd.read_csv(
        base_input_data_path. joinpath(
            FOLDER,
            "battery_weights_kg.csv"),
        index_col=[
            0,
            1])
    # Using the 250 Wh/kg on the kWh of the various batteries a weight (in kg) of the battery per vehicle category is
    # determined
    battery_materials: pd.DataFrame = pd.read_csv(
        base_input_data_path. joinpath(
            FOLDER, "battery_materials.csv"), index_col=[
            0, 1])
    # The material fraction of storage technologies (used to get the vehicle
    # battery composition)

    #  Reading all out files for vehicles and ships that are internal to IMAGE

    # IMAGE scenario files (total demand in Tkms & Pkms + vehicle shares)
    tonkms_Mtkms: pd.DataFrame = read_mym_df(
        climate_policy_config['config_file_path'] / climate_policy_config['data_files']['transport']['freight']['Tkm']). rename(
        columns={
            "DIM_1": "region"})
    # The tonne kilometres of freight vehicles of the IMAGE/TIMER SSP2 (in
    # Mega Tkm)
    passengerkms_Tpkms: pd.DataFrame = read_mym_df(
        climate_policy_config['config_file_path'] / climate_policy_config['data_files']['passenger']['kilometers']). rename(
        columns={
            "DIM_1": "region"})
    # The passenger kilometres from the IMAGE/TIMER SSP2 (in Tera Pkm)
    buses_vshares: pd.DataFrame = read_mym_df(
        climate_policy_config['config_file_path'] / climate_policy_config['data_files']['passenger']['Vshare_bus']). rename(
        columns={
            "DIM_1": "region"})
    # The vehicle shares of buses of the SSP2                            MIND!
    # FOR the BL this is still the OLD SSP2 file REPLACE LATER
    car_vshares: pd.DataFrame = read_mym_df(
        climate_policy_config['config_file_path'] / climate_policy_config['data_files']['passenger']['Vshare_car']). rename(
        columns={
            "DIM_1": "region"})
    # The vehicle shares of passenger cars of the SSP2
    medtruck_vshares: pd.DataFrame = read_mym_df(
        climate_policy_config['config_file_path'] / climate_policy_config['data_files']['freight']['Vshare_MedTruck']). rename(
        columns={
            "DIM_1": "region"})
    # The vehicle shares of trucks (medium) of the SSP2
    hvytruck_vshares: pd.DataFrame = read_mym_df(
        climate_policy_config['config_file_path'] / climate_policy_config['data_files']['freight']['Vshare_HvyTruck']). rename(
        columns={
            "DIM_1": "region"})
    # The vehicle shares of trucks (heavy) of the SSP2
    loadfactor_car_data: pd.DataFrame = read_mym_df(
        climate_policy_config['config_file_path'] / climate_policy_config['data_files']['passenger']['load']). rename(
        columns={
            "DIM_1": "region"})
    # The loadfactor of passenger vehicles (occupation in nr of
    # people/vehicle) in reference to the base loadfactor (see constants
    # above)

    # preprocessing of the IMAGE files & files with additional assumptions on vehcile materials
    # (renaming, removing 27th region, adding labels etc.)

    # Interpolate kilometrage & mileage data (over the range of original IMAGE data years, so the
    # generic interpolation function is not used here as that would extend the historic tail)
    kilometrage = kilometrage.reindex(
        years_range
    ).interpolate(limit_direction='both')
    kilometrage_bus = kilometrage_bus.reindex(
        years_range
    ).interpolate(limit_direction='both')
    kilometrage_midi_bus = kilometrage_midi_bus.reindex(
        years_range
    ).interpolate(limit_direction='both')
    mileages = mileages.reindex(
        years_range
    ).interpolate(limit_direction='both')

    region_list = list(kilometrage.columns.values)
    # get a list with region names TODO: turn this into a proper mapping based
    # on ...

    # select loadfactor for cars
    loadfactor_car_data.rename(columns={"DIM_1": "region"}, inplace=True)
    car_loadfactor = loadfactor_car_data[["time", "region", 5]].\
        pivot_table(index="time", columns="region").droplevel(level=0, axis=1)\
        * LOAD_FACTOR
    # loadfactor for cars (in persons per vehicle) * LOAD_FACTOR to correct
    # with te TIMER reference
    car_loadfactor = car_loadfactor.apply(lambda x:
                                          [y if y >= 1 else 1 for y in x])
    # To avoid car load (person/vehicle) values ever going below 1, replace
    # all values below 1 with 1
    car_loadfactor = car_loadfactor.loc[list(
        range(START_YEAR, END_YEAR + 1)), :]
    # remove years beyond LAST_YEAR
    car_loadfactor.columns = region_list

    # Define a list of DataFrames and their respective column labels
    df_dict = {
        'tonkms_Mtkms': (tonkms_Mtkms, tkms_label),
        'passengerkms_Tpkms': (passengerkms_Tpkms, pkms_label),
        'buses_vshares': (buses_vshares, bus_label),
        'car_vshares': (car_vshares, None),  # No labels for car_vshares
        'medtruck_vshares': (medtruck_vshares, truck_label),
        'hvytruck_vshares': (hvytruck_vshares, truck_label)
    }

    for df_name, (df, label) in df_dict.items():
        if df is not None:
            # Filter DataFrame for the output years
            df = df[df["time"].isin(years_range)]

            # Set multi-index based on the first two columns
            df.set_index(["time", "region"], inplace=True)

            # Insert column descriptions if available
            if label is not None:
                df.columns = label

            # Update the DataFrame in the dictionary
            df_dict[df_name] = df

    tonkms_Mtkms = df_dict['tonkms_Mtkms']
    passengerkms_Tpkms = df_dict['passengerkms_Tpkms']
    buses_vshares = df_dict['buses_vshares']
    car_vshares = df_dict['car_vshares']
    medtruck_vshares = df_dict['medtruck_vshares']
    hvytruck_vshares = df_dict['hvytruck_vshares']

    battery_shares_full = battery_shares_full.loc[years_range]

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
    vehicle_shares_typical[('Regular Buses', 'ICE')] =  buses_vshares[bus_label_ICE].sum(axis=1).unstack()
    vehicle_shares_typical[('Regular Buses', 'HEV')] =  buses_vshares[bus_label_HEV].sum(axis=1).unstack()
    vehicle_shares_typical[('Regular Buses', 'BEV')] =  buses_vshares['BusBattElectric'].unstack()
    vehicle_shares_typical[('Regular Buses', 'Trolley')] =  buses_vshares['BusElecTrolley'].unstack()

    vehicle_shares_typical[('Midi Buses', 'ICE')] =  buses_vshares[bus_label_ICE].sum(axis=1).div(midi_sum).unstack()
    vehicle_shares_typical[('Midi Buses', 'HEV')] =  buses_vshares[bus_label_HEV].sum(axis=1).div(midi_sum).unstack()
    vehicle_shares_typical[('Midi Buses', 'BEV')] =  buses_vshares['BusBattElectric'].div(midi_sum).unstack()
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
        vehicle_shares_typical[(truck_type, 'ICE')] =  truck_df[truck_label_ICE].sum(axis=1).unstack()
        vehicle_shares_typical[(truck_type, 'HEV')] =  truck_df[truck_label_HEV].sum(axis=1).unstack()
        vehicle_shares_typical[(truck_type, 'PHEV')] =  truck_df[truck_label_PHEV].sum(axis=1).unstack()
        vehicle_shares_typical[(truck_type, 'BEV')] =  truck_df[truck_label_BEV].sum(axis=1).unstack()
        vehicle_shares_typical[(truck_type, 'FCV')] =  truck_df[truck_label_FCV].sum(axis=1).unstack()
        vehicle_shares_typical[(truck_type, 'Trolley')] = 0

    # %% For dynamic variables, apply interpolation and extend over the whole timeframe

    # complete & interpolate the vehicle weight data
    vehicle_weights_simple = interpolate(
        pd.DataFrame(vehicle_weight_kg_simple))

    vehicle_weights_typical = vehicle_weight_kg_typical.rename_axis('mode', axis=1).stack().unstack(['mode', 'type'])
    vehicle_weights_typical = interpolate(
        pd.DataFrame(vehicle_weights_typical))

    # complete & interpolate the vehicle composition data (simple first)
    material_fractions_simple = material_fractions.rename_axis('mode', 
                                                               axis=1).rename_axis(['year', 'material'], 
                                                                                   axis=0).stack().unstack(
            ['mode', 'material'])
    material_fractions_simple = interpolate(
        pd.DataFrame(material_fractions_simple))

    # complete & interpolate the vehicle composition data (by vehicle sub-type
    # second)
    material_fractions_typical = material_fractions_type.rename_axis(['mode', 'type'], 
                                                                      axis=1).rename_axis(['year', 'material'],
                                                          axis=0).stack().stack().unstack(['mode', 'type', 'material'])
    material_fractions_typical = interpolate(pd.DataFrame(
        material_fractions_typical))

    # interpolate & complete series for battery weights, shares & composition
    # too
    battery_weights_typical = interpolate(battery_weights.unstack())
    battery_materials = interpolate(battery_materials.unstack())
    battery_shares = interpolate(battery_shares_full)

    # same for lifetime data
    lifetimes_vehicles = lifetimes_vehicles.rename_axis('mode', axis=1).stack()
    lifetimes_vehicles = lifetimes_vehicles[(lifetimes_vehicles.T != 0)]
    lifetimes_vehicles = lifetimes_vehicles.unstack(['mode', 'data'])
    lifetimes_vehicles = interpolate(pd.DataFrame(lifetimes_vehicles))

    if circular_economy_scenario == "slow":
        lifetimes_vehicles = lifetimes_vehicles[lifetimes_vehicles.index <= base_year].copy()
        lifetimes_vehicles.loc[target_year] = lifetimes_vehicles.loc[base_year]

        for mode, increase in lifetime_increase.items():
            col = (mode, 'mean')
            if col in lifetimes_vehicles.columns:
                base_val = lifetimes_vehicles.loc[base_year, col]
                lifetimes_vehicles.loc[target_year, col] = base_val * (1 + increase / 100)
            else:
                print(f"Missing mode: {col}")
        lifetimes_vehicles = interpolate(pd.DataFrame(lifetimes_vehicles))

    # TODO align dataframe structures below to the now changed dataframe
    # formats

    # Caculating the tonnekilometres for all freight and passenger vehicle types (adjustments are made to: freight air,
    # trucks, and buses)

    # Trucks are calculated differently because the IMAGE model does not account for LCV trucks, which concerns a large
    # portion of the material requirements of road freight
    # the total trucks Tkms remain the same, but a LCV fraction is substracted, and the remainder is re-assigned to
    # medium and heavy trucks according to their original ratio
    trucks_total_tkm = tonkms_Mtkms["Medium Freight Trucks"].unstack() + tonkms_Mtkms["Heavy Freight Trucks"].unstack()
    trucks_LCV_tkm = trucks_total_tkm * LIGHT_COMMERCIAL_VEHICLE_SHARE
    MFT_percshare_tkm = tonkms_Mtkms["Medium Freight Trucks"].unstack() / trucks_total_tkm  # the MFT fraction of the total
    HFT_percshare_tkm = tonkms_Mtkms["Heavy Freight Trucks"].unstack() / trucks_total_tkm   # the HFT fraction of the total
    trucks_min_LCV = trucks_total_tkm - trucks_LCV_tkm
    trucks_MFT_tkm = trucks_min_LCV.mul(MFT_percshare_tkm)             # Used in loop below
    trucks_HFT_tkm = trucks_min_LCV.mul(HFT_percshare_tkm)

    # demand for freight planes is reduced by 50% because about half of the air freight is transported as cargo on
    # passenger planes
    air_freight_tkms = tonkms_Mtkms["Freight Planes"].unstack() * market_share["Passenger Planes"].values[0]

    # Buses are adjusted to account for the higher material intensity of
    # mini-buses
    bus_regl_pkms = passengerkms_Tpkms["bus"].unstack() * market_share["Regular Buses"].values[0]   # in tera pkms
    bus_midi_pkms = passengerkms_Tpkms["bus"].unstack() * market_share["Midi Buses"].values[0]  # in tera pkms

    # Select tkms of passenger cars (which will be adjusted to represent 5
    # types: ICE, HEV, PHEV, BEV & FCV)
    car_pkms = passengerkms_Tpkms["Cars"].unstack()
    # exclude region 27 & 28 (empty & global total), mind
    car_pkms = car_pkms.drop([27, 28], axis=1)
    # that the columns represent generation technologies  # in tera pkms
    car_pkms.columns = region_list

    # %% Calculate the NUMBER OF VEHICLES (stock, on the road) to fulfull the ton-kilometers transport demand
    # TODO: exchange for proper region labels defined elsewhere
    # TODO: look at vehicle type names
    total_nr_vehicles_simple = pd.DataFrame(
        index=years_range,
        columns=pd.MultiIndex.from_product([all_modes,
                                            list(range(1, REGIONS + 3))],
                                           names=["Type", "Region"]
                                           )
    )
    total_nr_vehicles_simple.index.name = "time"

    # Fill the vehicle types that need no conversion    
    for label in ["Passenger Planes", "Trains", "High Speed Trains", "Bikes" ]:
        total_nr_vehicles_simple[label] = tkms_to_nr_of_vehicles_fixed(
            passengerkms_Tpkms[label].unstack(),
            mileages[label],
            load[label].values[0],
            loadfactor[label].values[0]
        )
        

    # %%
    # Fill the vehicle types that need conversion from Mega-ton-kms
    # TODO: loops later, just fill the data for now
    # original ton kilometers are in Mega-ton-kms, div by MEGA_TO_TERA to
    # harmonize with pkms which are in Tera pkms
    # Define the input data for vehicles requiring conversion
    # what does the "M" stand for?
    vehicle_data = {
        "Heavy Freight Trucks":         (trucks_HFT_tkm, "Heavy Freight Trucks", 'M'),
        "Medium Freight Trucks":        (trucks_MFT_tkm, "Medium Freight Trucks", 'M'),
        "Light Commercial Vehicles":    (trucks_LCV_tkm, "Light Commercial Vehicles", 'M'),
        "Freight Planes":               (air_freight_tkms, "Light Commercial Vehicles", 'M'),
        "Freight Trains":               (tonkms_Mtkms['Freight Trains'].unstack(), "Freight Trains", 'M'),
        "Inland Ships":                 (tonkms_Mtkms['Inland Ships'].unstack(), "Inland Ships", 'M')
    }

    # %%

    # Handle the vehicle types that need conversion from Mega km to Tera km
    for out_label, (df, key, unit) in vehicle_data.items():
        total_nr_vehicles_simple[out_label] = tkms_to_nr_of_vehicles_fixed(
            df / MEGA_TO_TERA,
            mileages[key],
            load[key].values[0],
            loadfactor[key].values[0]
        )

    # %%
    # passenger cars and buses are calculated separately (due to regional &
    # changeing mileage & load), first the totals
    car_total_vkms = car_pkms.div(car_loadfactor) * PKMS_TO_VKMS  # now in kms
    car_total_nr = car_total_vkms.div(kilometrage)             # total number of cars
    car_total_nr[["Extra Column", "Extra Column 2"]] = 1
    # remove region labels (for use in functions later on)
    car_total_nr.columns = list(range(1, REGIONS + 3))
    total_nr_vehicles_simple["Cars"] = car_total_nr

    # for buses do the same, but first remove region 27 & 28 (empty & world total) & kilometrage column names
    # bus_regl_pkms  = bus_regl_pkms.drop([27, 28], axis=1)
    # TODO: remove/change this hack!
    kilometrage_bus[["Extra Column", "Extra Column 2"]] = 1
    kilometrage_bus.columns = list(range(1, REGIONS + 3))
    bus_regl_vkms = bus_regl_pkms.div(load["Regular Buses"].values[0]
                                      * loadfactor["Regular Buses"].values[0]) * PKMS_TO_VKMS
    # now in kms
    total_nr_vehicles_simple["Regular Buses"] = bus_regl_vkms.div(
        kilometrage_bus)

    # bus_midi_pkms  = bus_midi_pkms.drop([27, 28], axis=1)
    # TODO: remove/change this hack!
    kilometrage_midi_bus[["Extra Column", "Extra Column 2"]] = 1
    kilometrage_midi_bus.columns = list(range(1, REGIONS + 3))
    bus_midi_vkms = bus_midi_pkms.div(load["Midi Buses"].values[0] * loadfactor["Midi Buses"].values[0]) * PKMS_TO_VKMS
    # now in kms
    total_nr_vehicles_simple["Midi Buses"] = bus_midi_vkms.div( kilometrage_midi_bus)

    # %% for INTERNATIONAL SHIPPING the number of vehicles is calculated differently

    # pre-calculate the shares of the boats based on the number of boats, before adding history/future
    # TODO: boats vs. ships?
    share_of_boats = nr_of_boats.div(nr_of_boats.sum(axis=1), axis=0)

    # does change based on trend in original data
    share_of_boats_yrs = interpolate(share_of_boats, change='yes')
    # could be 'yes' based on better data
    cap_of_boats_yrs = interpolate(cap_of_boats, change='no')
    loadfactor_boats_yrs = interpolate(loadfactor_boats, change='no')
    mileage_boats_yrs = interpolate(mileage_boats, change='no')
    weight_frac_boats_yrs = interpolate(weight_boats, change='no')

    # normalize the share of boats to 1 & adjust the capacity & mileage for
    # smaller ships
    share_of_boats_yrs = share_of_boats_yrs.div(share_of_boats_yrs.sum(axis=1), axis=0)
    cap_of_boats_yrs = cap_of_boats_yrs.mul(cap_adjustment, axis=1)
    mileage_boats_yrs = mileage_boats_yrs.mul(mile_adjustment, axis=1)

    # now derive the number of ships for 4 ship types in four steps:
    # 1) get the share of the ship types in the Tkms shipped (what % of total
    # tkms shipped goes by what ship type?)
    share_of_boats_tkm_all = share_of_boats_yrs * cap_of_boats_yrs  * loadfactor_boats_yrs * mileage_boats_yrs
    share_of_boats_tkm = share_of_boats_tkm_all.div(share_of_boats_tkm_all.sum(axis=1), axis=0)

    # for comparison, we find the difference of the known and the calculated nr of ships (global total)
    # in the period 2005-2018
    # TODO: is this used anywhere?
    diff_ships = pd.DataFrame().reindex_like(nr_of_boats)

    # TODO: it seems this calculation should be able to be simplified
    for size in ["Small", "Medium", "Large", "Very Large"]:
        total_nr_vehicles_simple[f"{size} Ships"] = tonkms_Mtkms["international shipping"].unstack().mul(
            share_of_boats_tkm[size].loc[START_YEAR:], axis=0).mul(MEGA_TO_TERA).div(
                cap_of_boats_yrs[size].loc[START_YEAR:], axis=0).div(
        mileage_boats_yrs[size].loc[START_YEAR:], axis=0)
        diff_ships[size] = total_nr_vehicles_simple[f"{size} Ships"].loc[SHIPS_YEARS_RANGE,   # TODO Make years flexible
            28].div(nr_of_boats[size])

    # TODO what is this for?
    # TODO: remove this, this should be done differently somehow.
    # total_nr_of_ships = total_nr_vehicles_simple["Small Ships"] + \
    #    total_nr_vehicles_simple["Medium Ships"] + \
    #    total_nr_vehicles_simple["Large Ships"] + \
    #    total_nr_vehicles_simple["Very Large Ships"]
    # diff_ships_total = total_nr_of_ships.loc[SHIPS_YEARS_RANGE, 28].div(nr_of_boats.sum(axis=1))

    # capacity of boats is in tonnes, the weight - expressed as a fraction of
    # the capacity - calculated in in kgs here
    weight_boats = weight_frac_boats_yrs * cap_of_boats_yrs * TONNES_TO_KGS

    # Calculate historic tail
    # first_year = first_year_vehicle.drop(columns = ["Cars"])
    # interpolate(total_nr_vehicles_simple, first_year)   #TODO what is this
    # for?

    # %% Interpolate to complete data for the entire model period (including a historic tail to set up the dynamic
    # stock calculations)

    # reformatting lifetime data (because input is not yet region-specific)
    first_year_vehicle_regionalized = pd.DataFrame(
        0, index=first_year_vehicle.index, columns=total_nr_vehicles_simple.columns)
    for vehicle in list(total_nr_vehicles_simple.columns.levels[0]):
        first_year_vehicle_regionalized.loc[:, idx[vehicle, :]] = [first_year_vehicle[vehicle].values[0] for region in
             list(total_nr_vehicles_simple.columns.levels[1])]

    # Doing the interpolation & assigning a starting point for the vehicle
    # stock time series based on first_year_vehicle
    total_nr_vehicles_simple = interpolate(total_nr_vehicles_simple,
                                           first_year_vehicle_regionalized,
                                           change='no')
    vehicle_shares_typical = interpolate(vehicle_shares_typical, change='no')

    results_dict = {
        'total_nr_vehicles_simple': total_nr_vehicles_simple,
        'material_fractions_simple': material_fractions_simple,
        'material_fractions_typical': material_fractions_typical,
        'vehicle_weights_simple': vehicle_weights_simple,
        'vehicle_weights_typical': vehicle_weights_typical,
        'lifetimes': lifetimes_vehicles,
        'battery_weights_typical': battery_weights_typical,
        'battery_materials': battery_materials,
        'battery_shares': battery_shares,
        'weight_boats': weight_boats,
        'vehicle_shares_typical': vehicle_shares_typical
    }
    
    
    # Create a pint UnitRegistry
    ureg = pint.UnitRegistry(force_ndarray_like=True)
    pint.set_application_registry(ureg)
    # preprocessing_results_xarray = preprocessing_results.copy()
    
    
    # Convert the DataFrames to xarray Datasets and apply units
    preprocessing_results_xarray = {}

    # Conversion table for all coordinates, to be removed/adapted after input tables are fixed.
    conversion_table = {
        "total_nr_vehicles_simple": (["Time"], ["Type", "Region"],),
        "material_fractions_simple": (["Cohort"], ["Type", "material"],),
        "material_fractions_typical": (["Cohort"], ["Type", "SubType", "material"], {"Type": ["Type", "SubType"]}),
        "vehicle_weights_simple": (["Cohort"], ["Type"],),
        "vehicle_weights_typical": (["Cohort"], ["Type", "SubType"], {"Type": ["Type", "SubType"]}),
        "battery_weights_typical": (["Cohort"], ["Type", "SubType"], {"Type": ["Type", "SubType"]}),
        "battery_materials": (["Cohort"], ["material", "battery"],),
        "battery_shares": (["Cohort"], ["battery"],),
        "weight_boats": (["Cohort"], ["size"],),
        "vehicle_shares_typical": (["Cohort"], ["Type", "SubType", "Region"], {"Type": ["Type", "SubType"]})
    }
    for df_name, df in results_dict.items():
        if df_name in conversion_table:
            data_xar_dataset = pandas_to_xarray(df, unit_mapping)
            data_xarray = dataset_to_array(data_xar_dataset, *conversion_table[df_name])
        else:
            # lifetimes_vehicles does not need to be converted in the same way.
            data_xarray = pandas_to_xarray(df, unit_mapping)
        preprocessing_results_xarray[df_name] = data_xarray
    # Concatenate or rename typical/simple arrays
    df_name_list = list(preprocessing_results_xarray)
    for df_name in df_name_list:
        if df_name.endswith("simple"):
            df_name_typical = df_name[:-6]+"typical"
            if df_name_typical in df_name_list:
                xar_simple = preprocessing_results_xarray.pop(df_name)
                xar_typical = preprocessing_results_xarray.pop(df_name[:-6]+"typical")
                xar_complete = xr.concat((xar_simple, xar_typical), dim="Type")
                preprocessing_results_xarray[df_name[:-7]] = xar_complete
            else:
                preprocessing_results_xarray[df_name[:-7]] = preprocessing_results_xarray.pop(df_name)
        elif df_name.endswith("typical"):
            df_name_simple = df_name[:-7]+"simple"
            if df_name_simple not in df_name_list:
                preprocessing_results_xarray[df_name[:-8]] = preprocessing_results_xarray.pop(df_name)

    preprocessing_results_xarray["lifetimes"] = convert_life_time_vehicles(preprocessing_results_xarray["lifetimes"])
    preprocessing_results_xarray["stocks"] = preprocessing_results_xarray.pop("total_nr_vehicles")
    preprocessing_results_xarray["shares"] = preprocessing_results_xarray.pop("vehicle_shares")

    # Copy dimensiomns from material_fractions for xr_maintenance_material
    materials = preprocessing_results_xarray['material_fractions'].coords["material"]
    types = preprocessing_results_xarray['material_fractions'].coords["Type"]

    # Initialize xr_maintenance_material with zeros
    xr_maintenance_material = xr.DataArray(
        np.zeros((len(materials), len(types))),  # Shape based on dimensions
        dims=("material", "Type"),
        coords={"material": materials, "Type": types}
    )

    # Assign values from data in xr_maintenance_material where Type contains "Cars"
    cars_mask = np.char.find(types.astype(str), "Cars") >= 0  # Find entries containing "Cars"
    xr_maintenance_material.loc[{"Type": types[cars_mask]}] = maintenance_material["total_material_per_km"].values.reshape(-1, 1)

    preprocessing_results_xarray["maintenance_material_fractions"] = xr_maintenance_material

    # TODO: Check if this is correct
    bad_coords = preprocessing_results_xarray["battery_materials"].coords["battery"]
    new_coords = [x if x != "LMO" else "LMO/LCO" for x in bad_coords.values]
    preprocessing_results_xarray["battery_materials"] = preprocessing_results_xarray["battery_materials"].assign_coords({"battery": new_coords})

    # TODO: vemamodelling.py works with dict of dfs and not only dict of xarrays, therefore now both are returned (for now)
    return preprocessing_results_xarray



def convert_life_time_vehicles(life_time_vehicles: xr.Dataset) -> dict[str, xr.DataArray]:
    """Convert lifetime vehicles dataset to a more appropriate data format.

    This conversion should probably move to the preprocessing stage after we figure out
    the exact details of what the output should look like.

    Parameters
    ----------
    life_time_vehicles
        The input life_time_vehicles xarray dataset. It is supposed to be in a very particular format:
        it contains the parameters for each of the modes. However, the distribution types are not
        the same for each of the modes. Thus, the distribution types need to be inferred from the
        names of the parameters that are given. If multiple parameter sets for multiple distributions
        are given, the Weibull distribution is given preference over the FoldedNormal distribution.

    Returns
    -------
        A dictionary that contains a data array for each of the distributions. Given the setup there is
        an implicit assumption that only one distribution is used for each of the modes. If distribution
        types change over time, this data structure needs to be adjusted.

    """
    # Create a dictionary to find which parameters are available for which mode.
    mode_param = defaultdict(list)
    for mode, par in life_time_vehicles.data_vars:
        mode_param[mode].append(par)

    # Create a dictionary that says which modes are tied to which distribution.
    dist_mode = defaultdict(list)
    modes_done = set()  # temporary
    for dist in ALL_DISTRIBUTIONS:
        for mode, param in mode_param.items():
            if mode not in modes_done and dist.has_param(param):
                dist_mode[dist.name].append(mode)
                modes_done.add(mode)

    # Iterate over all distributions to create a data array for each of them.
    ret_scipy_params = {}
    for dist_name, mode_list in dist_mode.items():
        if len(mode_list) == 0:
            continue
        dist = NAME_TO_DIST[dist_name]

        # param_arrays = {}
        array = xr.DataArray(
            0.0, dims=("Time", "Type", "ScipyParam"),
            coords={
                "Time": life_time_vehicles.coords["year"].to_numpy(),
                "Type": mode_list,
                "ScipyParam": dist.variable_scipy_param})
        for mode in mode_list:
            orig_param_dict = {}
            for param in dist.params:
                orig_param_dict[param] = life_time_vehicles.data_vars[str(mode), param].to_numpy()
            scipy_params = dist.get_param(orig_param_dict)
            for cur_scipy_key, cur_scipy_par in scipy_params.items():
                if cur_scipy_key in dist.variable_scipy_param:
                    array.loc[:, str(mode), cur_scipy_key] = cur_scipy_par
                else:
                    array.attrs[cur_scipy_key] = cur_scipy_par
        ret_scipy_params[dist_name] = array
    return ret_scipy_params


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="IMAGE Materials VEMA preprocessing script."
    )
    parser.add_argument(
        "--path",
        help="Base VEMA folder path.",
        type=Path,
        default=os.getcwd(),
    )
    args = parser.parse_args()

    # Call preprocessing function and make output available in variables
    output_preprocessing = preprocessing(base_dir=args.path)

