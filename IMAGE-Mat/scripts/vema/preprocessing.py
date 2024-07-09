# %%
import argparse
from pathlib import Path
import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib
import time
import os

from read_scripts.read_mym import read_mym_df
from modelling_functions import interpolate, tkms_to_nr_of_vehicles_fixed
# Path fragments and constants
from constants import PROJECT, SCEN, FOLDER, START_YEAR, END_YEAR, LOAD_FACTOR, REGIONS
# Labels
from constants import ( 
                        tkms_label, pkms_label, truck_label, bus_label, 
                        columns_vehicle_output, 
                        bus_label_ICE, bus_label_HEV,
                        truck_label_ICE, truck_label_HEV, truck_label_PHEV, 
                        truck_label_BEV, truck_label_FCV, vshares_label,
                        END_YEAR,
                        REGIONS, LIGHT_COMMERCIAL_VEHICLE_SHARE,
                        MEGA_TO_TERA, PKMS_TO_VKMS, TONNES_TO_KGS,
                        SHIPS_YEARS_RANGE
                       )
base_dir=Path(os.getcwd())

def preprocessing(base_dir=os.getcwd()):
    """Wrapper function for the preprocessing part of the VEMA script.
    """
    
    print(base_dir)
    base_dir = Path(base_dir)
    # %%
    base_dir=Path(os.getcwd())
    base_input_data_path = base_dir.joinpath("..", "..", "input", "vehicles")
    standard_input_data_path = base_input_data_path.joinpath("standard_data")
    image_folder = base_dir.joinpath("..", "..", "image", PROJECT, SCEN)
    standard_output_folder = base_dir.joinpath("..", "..", "output", PROJECT,
                                               FOLDER)
    
    #st = time.time()
    
    idx = pd.IndexSlice          # needed for slicing multi-index
    
    # Reading all csv files for vehicles and ships that are external to IMAGE
    
    # 1) scenario independent data
    load = pd.read_csv(standard_input_data_path.joinpath("load_pass_and_tonnes.csv"))             # TODO: add description again here!
    loadfactor = pd.read_csv(standard_input_data_path.joinpath("loadfactor_percentages.csv"))     # Percentage of the maximum load that is on average 
    market_share = pd.read_csv(standard_input_data_path.joinpath("fraction_tkm_pkm.csv"))         # Percentage of tonne-/passengerkilometres
    first_year_vehicle = pd.read_csv(standard_input_data_path.joinpath("first_year_vehicle.csv")) # first year of operation per vehicle-type - 1807 was originally in the dataframe
    battery_shares_full = pd.read_csv(standard_input_data_path.joinpath("battery_share_inflow.csv"), index_col=0)  # The share of the battery market (8 battery types used in vehicles), this data is based on a Multi-Nomial-Logit market model & costs in https://doi.org/10.1016/j.resconrec.2020.105200 - since this is scenario dependent it"s placed under the "IMAGE" scenario folder
    
    # Files related to the international shipping
    nr_of_boats  = pd.read_csv(standard_input_data_path.joinpath("ships", "number_of_boats.csv"), index_col="t").sort_index(axis=0)           # number of boats in the global merchant fleet (2005-2018)   changing Data by EQUASIS
    cap_of_boats        = pd.read_csv(standard_input_data_path.joinpath("ships", "capacity_ton_boats.csv"), index_col="t").sort_index(axis=0)        # boat capacity in tons                                      changing Data is a combination of EQUASIS Gross Tonnage and UNCTAD Dead-Weigh-Tonnage per Gross Tonnage
    loadfactor_boats    = pd.read_csv(standard_input_data_path.joinpath("ships", "loadfactor_boats.csv"), index_col="t").sort_index(axis=0)          # loadfactor of boats (fraction)                             fixed    Data is based on Ecoinvent report 14 on Transport (Table 8-19)
    mileage_boats       = pd.read_csv(standard_input_data_path.joinpath("ships", "mileage_kmyr_boats.csv"), index_col="t").sort_index(axis=0)        # mileage of boats in km/yr (per ship)                       fixed    Data is based on Ecoinvent report 14 on Transport (Table 8-19)
    weight_boats        = pd.read_csv(standard_input_data_path.joinpath("ships", "weight_percofcap_boats.csv"), index_col="t").sort_index(axis=0)    # weight of boats as a percentage of the capacity (%)        fixed    Data is based on Ecoinvent report 14 on Transport (section 8.4.1)
    
    # 2) scenario dependent data
    lifetimes_vehicles = pd.read_csv(base_input_data_path.joinpath(FOLDER, "lifetimes_years.csv"),  index_col=[0,1])   # Average End-of-Life of vehicles in years, this file also contains the setting for the choice of distribution and other lifetime related settings (standard devition, or alternative parameterisation)
    kilometrage = pd.read_csv(base_input_data_path.joinpath(FOLDER, "kilometrage.csv"),      index_col="t")      # kilometrage of passenger cars in kms/yr
    kilometrage_midi_bus = pd.read_csv(base_input_data_path.joinpath(FOLDER, "kilometrage_midi.csv"), index_col="t")      # kilometrage of midi-buses in kms/yr
    kilometrage_bus = pd.read_csv(base_input_data_path.joinpath(FOLDER, "kilometrage_bus.csv"),  index_col="t")     # kilometrage of regular buses in kms/yr
    mileages = pd.read_csv(base_input_data_path.joinpath(FOLDER, "mileages_km_per_year.csv"), index_col="t") # Km/year of all the vehicles (buses & cars have region-specific files)
    
    # weight and materials related data
    vehicle_weight_kg_simple = pd.read_csv(base_input_data_path.joinpath(FOLDER, "vehicle_weight_kg_simple.csv"),   index_col=0)       # Weight of a single vehicle of each type in kg
    vehicle_weight_kg_typical = pd.read_csv(base_input_data_path.joinpath(FOLDER, "vehicle_weight_kg_typical.csv"),  index_col=[0,1])   # Weight of a single vehicle of each type in kg
    material_fractions = pd.read_csv(base_input_data_path.joinpath(FOLDER, "material_fractions_simple.csv"),  index_col=[0,1])   # Material fractions in percentages
    material_fractions_type = pd.read_csv(base_input_data_path.joinpath(FOLDER, "material_fractions_typical.csv"), index_col=[0,1], header=[0,1])   # Material fractions in percentages, by vehicle sub-type
    battery_weights = pd.read_csv(base_input_data_path.joinpath(FOLDER, "battery_weights_kg.csv"), index_col=[0,1])   # Using the 250 Wh/kg on the kWh of the various batteries a weight (in kg) of the battery per vehicle category is determined
    battery_materials = pd.read_csv(base_input_data_path.joinpath(FOLDER, "battery_materials.csv"), index_col=[0,1])   # The material fraction of storage technologies (used to get the vehicle battery composition)
    
    #  Reading all out files for vehicles and ships that are internal to IMAGE
    
    # IMAGE scenario files (total demand in Tkms & Pkms + vehicle shares)
    tonkms_Mtkms        = read_mym_df(image_folder.joinpath("trp_frgt_Tkm.out"))              # The tonne kilometres of freight vehicles of the IMAGE/TIMER SSP2 (in Mega Tkm)
    passengerkms_Tpkms  = read_mym_df(image_folder.joinpath("trp_trvl_pkm.out"))              # The passenger kilometres from the IMAGE/TIMER SSP2 (in Tera Pkm)
    buses_vshares       = read_mym_df(image_folder.joinpath("trp_trvl_Vshare_bus.out"))       # The vehicle shares of buses of the SSP2                            MIND! FOR the BL this is still the OLD SSP2 file REPLACE LATER
    car_vshares         = read_mym_df(image_folder.joinpath("trp_trvl_Vshare_car.out"))       # The vehicle shares of passenger cars of the SSP2 
    medtruck_vshares    = read_mym_df(image_folder.joinpath("trp_frgt_Vshare_MedTruck.out"))  # The vehicle shares of trucks (medium) of the SSP2 
    hvytruck_vshares    = read_mym_df(image_folder.joinpath("trp_frgt_Vshare_HvyTruck.out"))  # The vehicle shares of trucks (heavy) of the SSP2 
    loadfactor_car_data = read_mym_df(image_folder.joinpath("trp_trvl_Load.out"))             # The loadfactor of passenger vehicles (occupation in nr of people/vehicle) in reference to the base loadfactor (see constants above)
    
    """
    preprocessing of the IMAGE files & files with additional assumptions on vehcile materials (renaming, removing 27th region, adding labels etc.)
    """
    for dataframe in [kilometrage, kilometrage_bus, kilometrage_midi_bus, mileages]:
        dataframe = dataframe.reindex(list(range(START_YEAR, END_YEAR + 1))).interpolate(limit_direction="both")
    
    region_list          = list(kilometrage.columns.values)     # get a list with region names TODO: turn this into a proper mapping based on ...
    
    # select loadfactor for cars
    car_loadfactor = loadfactor_car_data[["time","region", 5]].pivot_table(index="time", columns="region").droplevel(level=0, axis=1)  * LOAD_FACTOR # loadfactor for cars (in persons per vehicle) * LOAD_FACTOR to correct with te TIMER reference
    car_loadfactor = car_loadfactor.apply(lambda x: [y if y >= 1 else 1 for y in x])                      # To avoid car load (person/vehicle) values ever going below 1, replace all values below 1 with 1
    car_loadfactor = car_loadfactor.loc[list(range(START_YEAR, END_YEAR+1)),:]     # remove years beyond LAST_YEAR
    car_loadfactor.columns = region_list
    
    # select data only for requested output years
    tonkms_Mtkms        = tonkms_Mtkms[tonkms_Mtkms["time"].isin(list(range(START_YEAR, END_YEAR+1)))]
    passengerkms_Tpkms  = passengerkms_Tpkms[passengerkms_Tpkms["time"].isin(list(range(START_YEAR, END_YEAR+1)))] 
    buses_vshares       = buses_vshares[buses_vshares["time"].isin(list(range(START_YEAR, END_YEAR+1)))] 
    car_vshares         = car_vshares[car_vshares["time"].isin(list(range(START_YEAR, END_YEAR+1)))] 
    medtruck_vshares    = medtruck_vshares[medtruck_vshares["time"].isin(list(range(START_YEAR, END_YEAR+1)))] 
    hvytruck_vshares    = hvytruck_vshares[hvytruck_vshares["time"].isin(list(range(START_YEAR, END_YEAR+1)))] 
    battery_shares_full = battery_shares_full.loc[list(range(START_YEAR, END_YEAR+1))]
    
    # set multi-index based on the first two columns
    tonkms_Mtkms.set_index(["time", "region"], inplace=True)
    passengerkms_Tpkms.set_index(["time", "region"], inplace=True)
    buses_vshares.set_index(["time", "region"], inplace=True)
    car_vshares.set_index(["time", "region"], inplace=True)
    medtruck_vshares.set_index(["time", "region"], inplace=True)
    hvytruck_vshares.set_index(["time", "region"], inplace=True)
    
    # insert column descriptions
    tonkms_Mtkms.columns       = tkms_label
    passengerkms_Tpkms.columns = pkms_label
    medtruck_vshares.columns   = truck_label
    hvytruck_vshares.columns   = truck_label
    buses_vshares.columns      = bus_label
    
    #%% Vehicle shares of typical vehicles (i.e. specified different drivetrains)
    
    # CARS    
    # aggregate car types into 5 car types
    BEV_collist  = [22, 24]
    PHEV_collist = [23, 21, 20, 19, 18, 17, 16]
    ICE_collist  = [1,2,3,4,5,6,7,25]             # Gas car is considered ICE
    HEV_collist  = [8,9,10,11,12]
    FCV_collist  = [13,14,15]
    car_types = ["ICE","HEV","PHEV","BEV","FCV","Trolley"]
    
    index = pd.MultiIndex.from_product([list(range(START_YEAR, END_YEAR+1)), list(range(1,REGIONS+1))], names=["time", "region"])
    vehicleshare_cars = pd.DataFrame(0, index=index, columns=car_types)
    vehicleshare_cars.loc[idx[:,:],"ICE"]  = car_vshares[ICE_collist].sum(axis=1).to_numpy()
    vehicleshare_cars.loc[idx[:,:],"HEV"]  = car_vshares[HEV_collist].sum(axis=1).to_numpy()
    vehicleshare_cars.loc[idx[:,:],"PHEV"] = car_vshares[PHEV_collist].sum(axis=1).to_numpy()
    vehicleshare_cars.loc[idx[:,:],"BEV"]  = car_vshares[BEV_collist].sum(axis=1).to_numpy()
    vehicleshare_cars.loc[idx[:,:],"FCV"]  = car_vshares[FCV_collist].sum(axis=1).to_numpy()
    
    # BUSES
    # Sum of all buses except Trolleys
    midi_sum = buses_vshares[list(filter(lambda x: x != 'BusElecTrolley', bus_label))].sum(axis=1)
    
    # regular buses are just grouped
    buses_regl_vshares = pd.DataFrame(index=buses_vshares.index, columns=vshares_label)
    buses_regl_vshares['ICE']     = buses_vshares[bus_label_ICE].sum(axis=1)
    buses_regl_vshares['HEV']     = buses_vshares[bus_label_HEV].sum(axis=1)
    buses_regl_vshares['PHEV']    = pd.DataFrame(0, index=buses_vshares.index, columns=['PHEV'])
    buses_regl_vshares['BEV']     = buses_vshares['BusBattElectric']
    buses_regl_vshares['FCV']     = pd.DataFrame(0, index=buses_vshares.index, columns=['FCV'])
    buses_regl_vshares['Trolley'] = buses_vshares['BusElecTrolley']
    
    # midi buses are grouped & divided by the sum of ICE, HEV & BEV, to adjust for the fact that Trolleys (or FCV or PHEV) are not an option for midi buses
    buses_midi_vshares = pd.DataFrame(index=buses_vshares.index, columns=vshares_label)
    buses_midi_vshares['ICE']     = buses_vshares[bus_label_ICE].sum(axis=1).div(midi_sum)
    buses_midi_vshares['HEV']     = buses_vshares[bus_label_HEV].sum(axis=1).div(midi_sum)
    buses_midi_vshares['PHEV']    = pd.DataFrame(0, index=buses_vshares.index, columns=['PHEV'])
    buses_midi_vshares['BEV']     = buses_vshares['BusBattElectric'].div(midi_sum)
    buses_midi_vshares['FCV']     = pd.DataFrame(0, index=buses_vshares.index, columns=['FCV'])
    buses_midi_vshares['Trolley'] = pd.DataFrame(0, index=buses_vshares.index, columns=['Trolley'])
    
    # TRUCKS
    # vehicle shares are grouped as: a) ICE, b) HEV, c) PHEV, d) BEV, e) FCV
    # LCV vehicle shares are determined based on the medium trucks (so not calculated explicitly)
    
    # medium trucks
    trucks_MFT_vshares = pd.DataFrame(index=medtruck_vshares.index, columns=vshares_label)
    trucks_MFT_vshares['ICE']     = medtruck_vshares[truck_label_ICE].sum(axis=1)
    trucks_MFT_vshares['HEV']     = medtruck_vshares[truck_label_HEV].sum(axis=1)
    trucks_MFT_vshares['PHEV']    = medtruck_vshares[truck_label_PHEV].sum(axis=1)
    trucks_MFT_vshares['BEV']     = medtruck_vshares[truck_label_BEV].sum(axis=1)
    trucks_MFT_vshares['FCV']     = medtruck_vshares[truck_label_FCV].sum(axis=1)
    trucks_MFT_vshares['Trolley'] = pd.DataFrame(0, index=index, columns=['Trolley'])                    # No trolley trucks 
    
    # heavy trucks
    trucks_HFT_vshares = pd.DataFrame(index=hvytruck_vshares.index, columns=vshares_label)
    trucks_HFT_vshares['ICE']     = hvytruck_vshares[truck_label_ICE].sum(axis=1)
    trucks_HFT_vshares['HEV']     = hvytruck_vshares[truck_label_HEV].sum(axis=1)
    trucks_HFT_vshares['PHEV']    = hvytruck_vshares[truck_label_PHEV].sum(axis=1)
    trucks_HFT_vshares['BEV']     = hvytruck_vshares[truck_label_BEV].sum(axis=1)
    trucks_HFT_vshares['FCV']     = hvytruck_vshares[truck_label_FCV].sum(axis=1)
    trucks_HFT_vshares['Trolley'] = pd.DataFrame(0, index=hvytruck_vshares.index, columns=['Trolley'])   # No trolley trucks 

    columns = pd.MultiIndex.from_product([['Cars', 'Regular Buses', 'Midi Buses', 
                                           'Heavy Freight Trucks', 'Medium Freight Trucks', 
                                           'Light Commercial Vehicles'], car_types, 
                                           list(range(1,REGIONS+1))], 
                                           names=["vehicle", "type", "region"])
    vehicle_shares_typical         = pd.DataFrame(0, index=list(range(START_YEAR, END_YEAR+1)), columns=columns)
    vehicle_shares_typical['Cars']                  = vehicleshare_cars.unstack()
    vehicle_shares_typical['Regular Buses']         = buses_regl_vshares.unstack()
    vehicle_shares_typical['Midi Buses']            = buses_midi_vshares.unstack()
    vehicle_shares_typical['Heavy Freight Trucks']  = trucks_HFT_vshares.unstack()
    vehicle_shares_typical['Medium Freight Trucks'] = trucks_MFT_vshares.unstack()
    vehicle_shares_typical['Light Commercial Vehicles'] = trucks_MFT_vshares.unstack() # MIND Assumption: used MFT as a market-share for LCVs   
    
    # labels etc.
    # x_graphs        = [i for i in range(START_YEAR, END_YEAR, 1)]                           # this is used as an x-axis for the years in graphs
    # For dynamic variables, apply interpolation and extend over the whole timeframe
    
    # complete & interpolate the vehicle weight data
    vehicle_weights_simple = interpolate(pd.DataFrame(vehicle_weight_kg_simple))
    
    vehicle_weights_typical = vehicle_weight_kg_typical.rename_axis('mode', axis=1).stack().unstack(['mode', 'type'])
    vehicle_weights_typical = interpolate(pd.DataFrame(vehicle_weights_typical))
    
    # complete & interpolate the vehicle composition data (simple first)
    material_fractions_simple = material_fractions.rename_axis('mode', axis=1).rename_axis(['year','material'], axis=0).stack().unstack(['mode', 'material'])
    material_fractions_simple = interpolate(pd.DataFrame(material_fractions_simple))
    
    # complete & interpolate the vehicle composition data (by vehicle sub-type second)
    material_fractions_typical = material_fractions_type.rename_axis(['mode','type'], axis=1).rename_axis(['year','material'], axis=0).stack().stack().unstack(['mode','type', 'material'])
    material_fractions_typical = interpolate(pd.DataFrame(material_fractions_typical))
    
    # interpolate & complete series for battery weights, shares & composition too
    battery_weights_typical    = interpolate(battery_weights.unstack())
    battery_materials  = interpolate(battery_materials.unstack())
    battery_shares     = interpolate(battery_shares_full)
    
    # same for lifetime data
    lifetimes_vehicles = lifetimes_vehicles.rename_axis('mode', axis=1).stack()
    lifetimes_vehicles = lifetimes_vehicles[(lifetimes_vehicles.T != 0)]     
    lifetimes_vehicles = lifetimes_vehicles.unstack(['mode', 'data'])
    lifetimes_vehicles = interpolate(pd.DataFrame(lifetimes_vehicles))
    
    #TODO align dataframe structures below to the now changed dataframe formats 
    
    # Caculating the tonnekilometres for all freight and passenger vehicle types (adjustments are made to: freight air, trucks, and buses)
    
    # Trucks are calculated differently because the IMAGE model does not account for LCV trucks, which concerns a large portion of the material requirements of road freight
    # the total trucks Tkms remain the same, but a LCV fraction is substracted, and the remainder is re-assigned to medium and heavy trucks according to their original ratio
    trucks_total_tkm       = tonkms_Mtkms["medium truck"].unstack() +  tonkms_Mtkms["heavy truck"].unstack()
    trucks_LCV_tkm         = trucks_total_tkm * LIGHT_COMMERCIAL_VEHICLE_SHARE          # 0.04 is the fraction of the tkms driven by light commercial vehicles according to the IEA
    MFT_percshare_tkm      = tonkms_Mtkms["medium truck"].unstack() / trucks_total_tkm  # the MFT fraction of the total 
    HFT_percshare_tkm      = tonkms_Mtkms["heavy truck"].unstack() / trucks_total_tkm   # the HFT fraction of the total 
    trucks_min_LCV         = trucks_total_tkm - trucks_LCV_tkm
    trucks_MFT_tkm         = trucks_min_LCV.mul(MFT_percshare_tkm)                      
    trucks_HFT_tkm         = trucks_min_LCV.mul(HFT_percshare_tkm)
    
    # demand for freight planes is reduced by 50% because about half of the air freight is transported as cargo on passenger planes 
    air_freight_tkms       = tonkms_Mtkms["air cargo"].unstack() * market_share["air_freight"].values[0]
    
    # Buses are adjusted to account for the higher material intensity of mini-buses
    bus_regl_pkms          = passengerkms_Tpkms["bus"].unstack() * market_share["reg_bus"].values[0]   # in tera pkms
    bus_midi_pkms          = passengerkms_Tpkms["bus"].unstack() * market_share["midi_bus"].values[0]  # in tera pkms
    
    # Select tkms of passenger cars (which will be adjusted to represent 5 types: ICE, HEV, PHEV, BEV & FCV)
    car_pkms               = passengerkms_Tpkms["car"].unstack()
    car_pkms               = car_pkms.drop([27, 28], axis=1)    # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies  # in tera pkms                            
    car_pkms.columns       = region_list
    
    # %% 
    # Calculate the NUMBER OF VEHICLES (stock, on the road) to fulfull the ton-kilometers transport demand
    # TODO: exchange for proper region labels defined elsewhere
    # TODO: look at vehicle type names
    total_nr_vehicles_simple = pd.DataFrame(
        index=list(range(START_YEAR, END_YEAR + 1)),
        columns=pd.MultiIndex.from_product(
            [["Regular Buses", "Midi Buses", "Trains", "High Speed Trains",
              "Passenger Planes",
              "Bikes", "Heavy Freight Truck", "Medium Freight Truck", 
              "Light Commercial Vehicle", "Freight Trains", "Small Ships",
              "Medium Ships", "Large Ships", "Very Large Ships",
              "Inland Ships", "Freight Planes"], #TODO should Cars be added?
             list(range(1, REGIONS+3))],
            names=["Type", "Region"]
        )
    )
    total_nr_vehicles_simple.index.name = "time"
    
    # Fill the vehicle types that need no conversion
    for in_label, out_label in [
        ("air_pas", "Passenger Planes"), ("rail_reg", "Trains"),
        ("rail_hst", "High Speed Trains"), ("bicycle", "Bikes")
    ]:
        total_nr_vehicles_simple[out_label] = tkms_to_nr_of_vehicles_fixed(
            passengerkms_Tpkms[in_label].unstack(),
            mileages[in_label].values[0],
            load[in_label].values[0],
            loadfactor[in_label].values[0]
        )
    
    # %%
    # Fill the vehicle types that need conversion from Mega-ton-kms
    # TODO: loops later, just fill the data for now
    # original ton kilometers are in Mega-ton-kms, div by MEGA_TO_TERA to
    # harmonize with pkms which are in Tera pkms
    total_nr_vehicles_simple["Heavy Freight Trucks"] = tkms_to_nr_of_vehicles_fixed(
        trucks_HFT_tkm / MEGA_TO_TERA,
        mileages["HFT"].values[0],
        load["HFT"].values[0],
        loadfactor["HFT"].values[0]
    )
    total_nr_vehicles_simple["Medium Freight Trucks"] = tkms_to_nr_of_vehicles_fixed(
        trucks_MFT_tkm / MEGA_TO_TERA,
        mileages["MFT"].values[0],
        load["MFT"].values[0],
        loadfactor["MFT"].values[0]
    )
    total_nr_vehicles_simple["Light Commercial Vehicles"] = tkms_to_nr_of_vehicles_fixed(
        trucks_LCV_tkm / MEGA_TO_TERA,
        mileages["LCV"].values[0],
        load["LCV"].values[0],
        loadfactor["LCV"].values[0]
    )
    total_nr_vehicles_simple["Freight Planes"] = tkms_to_nr_of_vehicles_fixed(
        air_freight_tkms / MEGA_TO_TERA,
        mileages["air_freight"].values[0],
        load["air_freight"].values[0],
        loadfactor["air_freight"].values[0]
    )
    total_nr_vehicles_simple["Freight Trains"] = tkms_to_nr_of_vehicles_fixed(
        tonkms_Mtkms["freight train"].unstack() / MEGA_TO_TERA,
        mileages["rail_freight"].values[0],
        load["rail_freight"].values[0],
        loadfactor["rail_freight"].values[0]
    )
    total_nr_vehicles_simple["Inland Ships"] = tkms_to_nr_of_vehicles_fixed(
        tonkms_Mtkms["inland shipping"].unstack() / MEGA_TO_TERA,
        mileages["inland_shipping"].values[0],
        load["inland_shipping"].values[0],
        loadfactor["inland_shipping"].values[0]
    )
    
    # %%
    # passenger cars and buses are calculated separately (due to regional & changeing mileage & load), first the totals
    car_total_vkms  = car_pkms.div(car_loadfactor) * PKMS_TO_VKMS    # now in kms
    car_total_nr    = car_total_vkms.div(kilometrage)                 # total number of cars
    car_total_nr[["Extra Column", "Extra Column 2"]] = 1
    car_total_nr.columns = list(range(1,REGIONS+3))                          # remove region labels (for use in functions later on)
    total_nr_vehicles_simple["Cars"] = car_total_nr
    
    # for buses do the same, but first remove region 27 & 28 (empty & world total) & kilometrage column names
    # bus_regl_pkms  = bus_regl_pkms.drop([27, 28], axis=1)
    # TODO: remove/change this hack!
    kilometrage_bus[["Extra Column", "Extra Column 2"]] = 1
    kilometrage_bus.columns = list(range(1,REGIONS+3))
    bus_regl_vkms  = bus_regl_pkms.div(load["reg_bus"].values[0] * loadfactor["reg_bus"].values[0]) * PKMS_TO_VKMS    # now in kms
    total_nr_vehicles_simple["Regular Buses"] = bus_regl_vkms.div(kilometrage_bus)
    
    # bus_midi_pkms  = bus_midi_pkms.drop([27, 28], axis=1)
    # TODO: remove/change this hack!
    kilometrage_midi_bus[["Extra Column", "Extra Column 2"]] = 1
    kilometrage_midi_bus.columns = list(range(1,REGIONS+3))
    bus_midi_vkms  = bus_midi_pkms.div(load["midi_bus"].values[0] * loadfactor["midi_bus"].values[0]) * PKMS_TO_VKMS   # now in kms
    total_nr_vehicles_simple["Midi Buses"] = bus_midi_vkms.div(kilometrage_midi_bus)
    
    # %%
    # for INTERNATIONAL SHIPPING the number of vehicles is calculated differently 
    
    cap_adjustment  = [1, 1, 1, 1]
    mile_adjustment = [1, 1, 1, 1]
    
    #pre-calculate the shares of the boats based on the number of boats, before adding history/future
    # TODO: boats vs. ships?
    share_of_boats       = nr_of_boats.div(nr_of_boats.sum(axis=1), axis=0)
    
    share_of_boats_yrs    =  interpolate(share_of_boats,   change='yes')   # does change based on trend in original data
    cap_of_boats_yrs      =  interpolate(cap_of_boats,     change='no')    # could be 'yes' based on better data
    loadfactor_boats_yrs  =  interpolate(loadfactor_boats, change='no')   
    mileage_boats_yrs     =  interpolate(mileage_boats,    change='no')  
    weight_frac_boats_yrs =  interpolate(weight_boats,     change='no')  
    
    # normalize the share of boats to 1 & adjust the capacity & mileage for smaller ships 
    share_of_boats_yrs   = share_of_boats_yrs.div(share_of_boats_yrs.sum(axis=1), axis=0)
    cap_of_boats_yrs     = cap_of_boats_yrs.mul(cap_adjustment, axis=1)
    mileage_boats_yrs    = mileage_boats_yrs.mul(mile_adjustment, axis=1)
    
    # now derive the number of ships for 4 ship types in four steps: 
    # 1) get the share of the ship types in the Tkms shipped (what % of total tkms shipped goes by what ship type?)
    share_of_boats_tkm_all   = share_of_boats_yrs * cap_of_boats_yrs * loadfactor_boats_yrs * mileage_boats_yrs
    share_of_boats_tkm       = share_of_boats_tkm_all.div(share_of_boats_tkm_all.sum(axis=1), axis=0)
    
    # for comparison, we find the difference of the known and the calculated nr of ships (global total) in the period 2005-2018
    # TODO: is this used anywhere?
    diff_ships = pd.DataFrame().reindex_like(nr_of_boats)
    
    # TODO: it seems this calculation should be able to be simplified
    for size in ["Small", "Medium", "Large", "Very Large"]:
        total_nr_vehicles_simple[f"{size} Ships"] = \
            tonkms_Mtkms["international shipping"].unstack().mul(
                share_of_boats_tkm[size].loc[START_YEAR:], axis=0
            ).mul(MEGA_TO_TERA).div(
                cap_of_boats_yrs[size].loc[START_YEAR:], axis=0
            ).div(
                mileage_boats_yrs[size].loc[START_YEAR:], axis=0
            )
        diff_ships[size] = total_nr_vehicles_simple[f"{size} Ships"].loc[
            SHIPS_YEARS_RANGE,                                       #TODO Mka eyears flexible
            28
        ].div(nr_of_boats[size])
    
    # TODO: remove this, this should be done differently somehow.
    total_nr_of_ships = total_nr_vehicles_simple["Small Ships"] + \
        total_nr_vehicles_simple["Medium Ships"] + \
        total_nr_vehicles_simple["Large Ships"] + \
        total_nr_vehicles_simple["Very Large Ships"]
    diff_ships_total = total_nr_of_ships.loc[SHIPS_YEARS_RANGE, 28].div(nr_of_boats.sum(axis=1))
    
    # capacity of boats is in tonnes, the weight - expressed as a fraction of the capacity - is calculated in in kgs here
    weight_boats  = weight_frac_boats_yrs * cap_of_boats_yrs * TONNES_TO_KGS 
    
    # %% BATTERY WEIGHT SECTION
    # 1) BUSES: original vehcile shares are distributed into two vehicle types (regular and small midi buses)
    # vehicle shares are grouped as: a) ICE, b) HEV, c) trolley, d) BEV, but trolley buses are not relevant for midi busses, so the midi shares are re-calculated based on the sum without trolleys
    
    # Create the DataFrame with MultiIndex columns
    buses_battery_weight = pd.DataFrame(index=buses_vshares.index, columns= \
                                 pd.MultiIndex.from_product([['Regular', 'Midi'], vshares_label], names=['Type', 'Fuel']))
    
    # Regular buses
    buses_battery_weight[('Regular', 'ICE')] = buses_vshares[bus_label_ICE].sum(axis=1)
    buses_battery_weight[('Regular', 'HEV')] = buses_vshares[bus_label_HEV].sum(axis=1)
    buses_battery_weight[('Regular', 'BEV')] = buses_vshares['BusBattElectric']
    buses_battery_weight[('Regular', 'Trolley')] = buses_vshares['BusElecTrolley']
    
    # Midi buses
    buses_battery_weight[('Midi', 'ICE')] = buses_vshares[bus_label_ICE].sum(axis=1).div(midi_sum)
    buses_battery_weight[('Midi', 'HEV')] = buses_vshares[bus_label_HEV].sum(axis=1).div(midi_sum)
    buses_battery_weight[('Midi', 'BEV')] = buses_vshares['BusBattElectric'].div(midi_sum)
    buses_battery_weight[('Midi', 'Trolley')] = 0
    
    buses_battery_weight.loc[:, pd.IndexSlice[:, ['FCV', 'PHEV']]] = 0 # TODO Is this needed?
    
    # 2) TRUCKS
    # vehicle shares are grouped as: a) ICE, b) HEV, c) PHEV, d) BEV, e) FCV
    # LCV vehicle shares are determined based on the medium trucks (so not calculated explicitly)
    
    multi_columns = pd.MultiIndex.from_product([['MFT', 'HFT'], vshares_label], names=['Type', 'Fuel'])
    
    # Create the DataFrame with MultiIndex columns
    trucks_battery_weight = pd.DataFrame(index=medtruck_vshares.index.union(hvytruck_vshares.index), columns=multi_columns)
    
    # Define a dictionary for easy access
    truck_data = {
        'Medium Freight Truck': medtruck_vshares,
        'Heavy Freight Truck': hvytruck_vshares
    }
    
    # Assign values for both MFT and HFT
    for truck_type, truck_df in truck_data.items():
        trucks_battery_weight[(truck_type, 'ICE')]     = truck_df[truck_label_ICE].sum(axis=1)
        trucks_battery_weight[(truck_type, 'HEV')]     = truck_df[truck_label_HEV].sum(axis=1)
        trucks_battery_weight[(truck_type, 'PHEV')]    = truck_df[truck_label_PHEV].sum(axis=1)
        trucks_battery_weight[(truck_type, 'BEV')]     = truck_df[truck_label_BEV].sum(axis=1)
        trucks_battery_weight[(truck_type, 'FCV')]     = truck_df[truck_label_FCV].sum(axis=1)
        trucks_battery_weight[(truck_type, 'Trolley')] = 0  # No trolley trucks

    #%% Calculate historic tail
    first_year = first_year_vehicle.drop(columns = ["Cars"])
    interpolate(total_nr_vehicles_simple, first_year)   #TODO what is this for?
    
    #%% Export intermediate indicators (a.o. files on nr. of vehicles, pkms/tkms)
    # Export total global number of vehicles in the fleet (stock) as csv
    region_list = list(range(1,REGIONS+1))
    index = pd.MultiIndex.from_product([list(total_nr_of_ships.index), region_list], names = ["years","regions"])
    total_nr_vehicles = pd.DataFrame(index=index, columns=columns_vehicle_output)
    total_nr_vehicles["Buses"]        = total_nr_vehicles_simple["Regular Buses"][region_list].stack() + total_nr_vehicles_simple["Midi Buses"][region_list].stack()
    total_nr_vehicles["Trains"]       = total_nr_vehicles_simple["Trains"][region_list].stack()
    total_nr_vehicles["High Speed Trains"]          = total_nr_vehicles_simple["High Speed Trains"][region_list].stack()
    total_nr_vehicles["Cars"]         = car_total_nr[region_list].stack()
    total_nr_vehicles["Planes"]       = total_nr_vehicles_simple["Planes"][region_list].stack()
    total_nr_vehicles["Bikes"]        = total_nr_vehicles_simple["Bikes"][region_list].stack()
    total_nr_vehicles["Trucks"]       = total_nr_vehicles_simple["Heavy Freight Trucks"][region_list].stack() + total_nr_vehicles_simple["Medium Freight Trucks"][region_list].stack() + total_nr_vehicles_simple["Light Commercial Vehicles"][region_list].stack()
    total_nr_vehicles["Freight Trains"] = total_nr_vehicles_simple["Freight Trains"][region_list].stack()
    total_nr_vehicles["Ships"]        = total_nr_of_ships[region_list].stack()
    total_nr_vehicles["Inland Ships"] = total_nr_vehicles_simple["Inland Ships"][region_list].stack()  
    total_nr_vehicles["Freight Planes"] = total_nr_vehicles_simple["Freight Planes"][region_list].stack() 
    
    # Generate csv output file on pkms & tkms (same format as files on number of vehicles (also used later on)), unit: pkm or tkm 
    region_list = list(range(1,REGIONS+1))
    car_pkms.columns = region_list     # remove region labels 
    index = pd.MultiIndex.from_product([list(total_nr_of_ships.index), region_list], names = ["years","regions"])
    total_pkm_tkm = pd.DataFrame(index=index, columns=columns_vehicle_output)
    total_pkm_tkm["Buses"]        = (bus_regl_pkms[region_list].stack() + bus_midi_pkms[region_list].stack()) * PKMS_TO_VKMS
    total_pkm_tkm["Trains"]       = passengerkms_Tpkms["rail_reg"]   * PKMS_TO_VKMS                              
    total_pkm_tkm["High Speed Trains"]          = passengerkms_Tpkms["rail_hst"]     * PKMS_TO_VKMS
    total_pkm_tkm["Cars"]         = car_pkms[region_list].stack() * PKMS_TO_VKMS
    total_pkm_tkm["Planes"]       = passengerkms_Tpkms["air_pas"]     * PKMS_TO_VKMS
    total_pkm_tkm["Bikes"]        = passengerkms_Tpkms["bicycle"]  * PKMS_TO_VKMS
    total_pkm_tkm["Trucks"]       = (trucks_HFT_tkm[region_list].stack() + trucks_MFT_tkm[region_list].stack() + trucks_LCV_tkm[region_list].stack()) * MEGA_TO_TERA
    total_pkm_tkm["Freight Trains"] = tonkms_Mtkms["freight train"] * MEGA_TO_TERA
    total_pkm_tkm["Ships"]        = tonkms_Mtkms["international shipping"] * MEGA_TO_TERA 
    total_pkm_tkm["Inland ships"] = tonkms_Mtkms["inland shipping"] * MEGA_TO_TERA
    total_pkm_tkm["Freight Planes"] = air_freight_tkms[region_list].stack() * MEGA_TO_TERA     # mind that these are the tkms flows with cargo planes, real demand for air cargo is higher due to 50% hitching with passenger flights
    
    # output to IRP
    # output transport drivers to output folder for 450 vs Bl comparisson in overarching figures later on
    """
    tonkms_Mtkms.to_csv(standard_output_folder.joinpath("transport_tkms.csv"), index=True)       # in Mega tkms
    passengerkms_Tpkms.to_csv(standard_output_folder.joinpath("transport_pkms.csv"), index=True) # in Tera pkms
    vehicleshare_cars.to_csv(standard_output_folder.joinpath("car_type_share_regional.csv"), index=True)
    total_nr_vehicles.to_csv(standard_output_folder.joinpath("region_vehicle_nr.csv"), index=True) # regional nr of vehicles 
    total_nr_vehicles.sum(axis=0, level=0).to_csv(standard_output_folder.joinpath("global_vehicle_nr.csv"), index=True) # total global nr of vehicles 
    total_pkm_tkm.sum(axis=0, level=0).to_csv(standard_output_folder.joinpath("global_pkm_tkm.csv"), index=True) # total global pkms & tkms 
    total_pkm_tkm.to_csv(standard_output_folder.joinpath("region_pkm_tkm.csv"), index=True)  # regional pkms & tkms
    
    #%% Interpolate to complete data for the entire model period (including a historic tail to set up the dynamic stock calculations)
    
    # reformatting lifetime data (because input is not yet region-specific)
    first_year_vehicle_regionalized = pd.DataFrame(0, index=first_year_vehicle.index, columns=total_nr_vehicles_simple.columns)
    for vehicle in list(total_nr_vehicles_simple.columns.levels[0]):
        first_year_vehicle_regionalized.loc[:, idx[vehicle,:]] = [first_year_vehicle[vehicle].values[0] for region in list(total_nr_vehicles_simple.columns.levels[1])]
    
    # Doing the interpolation & assigning a starting point for the vehicle stock time series based on first_year_vehicle
    total_nr_vehicles_simple = interpolate(total_nr_vehicles_simple, first_year_vehicle_regionalized, change='no')
    vehicle_shares_typical   = interpolate(vehicle_shares_typical, change='no')  
    
    results_dict = { 
                'total_nr_vehicles_simple': total_nr_vehicles_simple,
                'material_fractions_simple': material_fractions_simple,
                'material_fractions_typical': material_fractions_typical,
                'vehicle_weights_simple': vehicle_weights_simple,
                'vehicle_weights_typical': vehicle_weights_typical,
                'lifetimes_vehicles': lifetimes_vehicles,
                'battery_weights_typical': battery_weights_typical,
                'battery_materials': battery_materials,
                'battery_shares': battery_shares,
                'buses_battery_weight': buses_battery_weight,             # Can be removed, right?
                'trucks_battery_weight': trucks_battery_weight,           # Can be removed, right?
                'weight_boats': weight_boats,
                'vehicle_shares_typical': vehicle_shares_typical
            }

    return results_dict

#%%
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

# %%
