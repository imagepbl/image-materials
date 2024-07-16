# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 18:11:03 2020
@author: Sebastiaan Deetman (deetman@cml.leidenuniv.nl) with contributions from Rombout Huisman 

The purpose of this module is to generate global material demand scenarios for vehicles based on IMAGE model. It can use any set of scenario output files, but initial setup is based on the SSP2 Baseline.

input:  
    1) IMAGE scenario files 
    2) Material & Weight related assumptions
    3) Additional assumptions (translating IMAGE data to the nr of vehicles)

output: 
    1) Graphs on the materials in vehicles
    2) csv database with material use (inflow, stock & outlfow) in kt

"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import time
from pathlib import Path

from dynamic_stock_model_BM import DynamicStockModel as DSM
# from past.builtins import execfile
from read_mym import read_mym_df # in-line execution of a function to read MyM files (function: read_mym_df)
st = time.time()

# settings & constants
START_YEAR = 1971
END_YEAR = 2060
OUT_YEAR = 2060              # year of output generation
REGIONS = 26
idx = pd.IndexSlice          # needed for slicing multi-index
FIRST_YEAR_BOATS = 1900
LOAD_FACTOR = 1.6            # reference loadfactor of cars in TIMER (the trp_trvl_Load.out file is relative to this BASE loadfcator (persons/car))

# scenario settings
# SCEN    = 'SSP2'
# VARIANT = '2D_RE'               # CP or 2D (Add '_RE' for Resource Efficiency)
# FOLDER  = SCEN + '_' + VARIANT

##########################################################################
#### added to read in same input data as IMAGE Materials new vema #########
# Scenario settings
SCEN = "SSP2"
# CP or 2D (Add "_RE" for Resource Efficiency)
VARIANT = "2D_RE"
PROJECT = "mock_project"
FOLDER = SCEN + "_" + VARIANT
# OUTPUT_FOLDER = "../../output/" + PROJECT + "/" + FOLDER

base_dir=os.getcwd()
base_dir = Path(base_dir)
base_input_data_path = base_dir.joinpath("..", "..", "..", "IMAGE-Mat", "input", "vehicles")
standard_input_data_path = base_input_data_path.joinpath("standard_data")
image_folder = base_dir.joinpath("..", "..", "..", "IMAGE-Mat", "image", PROJECT, SCEN)
standard_output_folder = base_dir.joinpath("..", "..", "..", "output", PROJECT, FOLDER)
#######################################################################
#######################################################################

# 'C:/Users/Arp00003/Coding/image-materials/IMAGE-Mat_old_version/IMAGE-Mat/VEMA/../../../IMAGE-Mat/input/vehicles'

#%% 0) Reading & preparing all the CSV & IMAGE files for vehicle categories and material fractions

# reading in all the first csv files containing vehicle data (load, market shares, lifetimes)
load = pd.read_csv(standard_input_data_path.joinpath("load_pass_and_tonnes.csv"))        # Load in terms of maximum number of passengers or kg of good
loadfactor = pd.read_csv(standard_input_data_path.joinpath("loadfactor_percentages.csv"))            # Percentage of the maximum load that is on average 
market_share = pd.read_csv(standard_input_data_path.joinpath("fraction_tkm_pkm.csv"))                   # Percentage of tonne-/passengerkilometres
first_year_vehicle = pd.read_csv(standard_input_data_path.joinpath("first_year_vehicle_old_naming.csv"))                # first year of operation per vehicle-type
lifetimes_vehicles = pd.read_csv(base_input_data_path.joinpath(FOLDER, "lifetimes_years.csv"),  index_col=[0,1])   # Average End-of-Life of vehicles in years, this file also contains the setting for the choice of distribution and other lifetime related settings (standard devition, or alternative parameterisation)
kilometrage = pd.read_csv(base_input_data_path.joinpath(FOLDER, "kilometrage.csv"),      index_col="t")       # kilometrage of passenger cars in kms/yr
kilometrage_midi_bus = pd.read_csv(base_input_data_path.joinpath(FOLDER, "kilometrage_midi.csv"), index_col="t")     # kilometrage of midi-buses in kms/yr
kilometrage_bus = pd.read_csv(base_input_data_path.joinpath(FOLDER, "kilometrage_bus.csv"),  index_col="t")    # kilometrage of regular buses in kms/yr
mileages = pd.read_csv(base_input_data_path.joinpath(FOLDER, "mileages_km_per_year.csv"), index_col="t") # Km/year of all the vehicles (buses & cars have region-specific files)

# weight and materials related data
vehicle_weight_kg_simple = pd.read_csv(base_input_data_path.joinpath(FOLDER, "vehicle_weight_kg_simple.csv"),   index_col=0)       # Weight of a single vehicle of each type in kg
vehicle_weight_kg_typical = pd.read_csv(base_input_data_path.joinpath(FOLDER, "vehicle_weight_kg_typical.csv"),  index_col=[0,1])   # Weight of a single vehicle of each type in kg
material_fractions = pd.read_csv(base_input_data_path.joinpath(FOLDER, "material_fractions_simple.csv"),  index_col=[0,1])   # Material fractions in percentages
material_fractions_type = pd.read_csv(base_input_data_path.joinpath(FOLDER, "material_fractions_typical.csv"), index_col=[0,1], header=[0,1])   # Material fractions in percentages, by vehicle sub-type
battery_weights = pd.read_csv(base_input_data_path.joinpath(FOLDER, "battery_weights_kg.csv"), index_col=[0,1])   # Using the 250 Wh/kg on the kWh of the various batteries a weight (in kg) of the battery per vehicle category is determined
battery_materials = pd.read_csv(base_input_data_path.joinpath(FOLDER, "battery_materials.csv"), index_col=[0,1])   # The material fraction of storage technologies (used to get the vehicle battery composition)

battery_shares_full = pd.read_csv(standard_input_data_path.joinpath("battery_share_inflow.csv"), index_col=0)                 # The share of the battery market (8 battery types used in vehicles), this data is based on a Multi-Nomial-Logit market model & costs in https://doi.org/10.1016/j.resconrec.2020.105200 - since this is scenario dependent it's placed under the 'IMAGE' scenario folder

# Files related to the international shipping
nr_of_boats  = pd.read_csv(standard_input_data_path.joinpath("ships", "number_of_boats.csv"), index_col="t").sort_index(axis=0)           # number of boats in the global merchant fleet (2005-2018)   changing Data by EQUASIS
cap_of_boats        = pd.read_csv(standard_input_data_path.joinpath("ships", "capacity_ton_boats.csv"), index_col="t").sort_index(axis=0)        # boat capacity in tons                                      changing Data is a combination of EQUASIS Gross Tonnage and UNCTAD Dead-Weigh-Tonnage per Gross Tonnage
loadfactor_boats    = pd.read_csv(standard_input_data_path.joinpath("ships", "loadfactor_boats.csv"), index_col="t").sort_index(axis=0)          # loadfactor of boats (fraction)                             fixed    Data is based on Ecoinvent report 14 on Transport (Table 8-19)
mileage_boats       = pd.read_csv(standard_input_data_path.joinpath("ships", "mileage_kmyr_boats.csv"), index_col="t").sort_index(axis=0)        # mileage of boats in km/yr (per ship)                       fixed    Data is based on Ecoinvent report 14 on Transport (Table 8-19)
weight_boats        = pd.read_csv(standard_input_data_path.joinpath("ships", "weight_percofcap_boats.csv"), index_col="t").sort_index(axis=0)    # weight of boats as a percentage of the capacity (%)        fixed    Data is based on Ecoinvent report 14 on Transport (section 8.4.1)

# IMAGE scenario files (total demand in Tkms & Pkms + vehicle shares)
tonkms_Mtkms        = read_mym_df(image_folder.joinpath("trp_frgt_Tkm.out"))              # The tonne kilometres of freight vehicles of the IMAGE/TIMER SSP2 (in Mega Tkm)
passengerkms_Tpkms  = read_mym_df(image_folder.joinpath("trp_trvl_pkm.out"))              # The passenger kilometres from the IMAGE/TIMER SSP2 (in Tera Pkm)
buses_vshares       = read_mym_df(image_folder.joinpath("trp_trvl_Vshare_bus.out"))       # The vehicle shares of buses of the SSP2                            MIND! FOR the BL this is still the OLD SSP2 file REPLACE LATER
car_vshares         = read_mym_df(image_folder.joinpath("trp_trvl_Vshare_car.out"))       # The vehicle shares of passenger cars of the SSP2 
medtruck_vshares    = read_mym_df(image_folder.joinpath("trp_frgt_Vshare_MedTruck.out"))  # The vehicle shares of trucks (medium) of the SSP2 
hvytruck_vshares    = read_mym_df(image_folder.joinpath("trp_frgt_Vshare_HvyTruck.out"))  # The vehicle shares of trucks (heavy) of the SSP2 
loadfactor_car_data = read_mym_df(image_folder.joinpath("trp_trvl_Load.out"))             # The loadfactor of passenger vehicles (occupation in nr of people/vehicle) in reference to the base loadfactor (see constants above)


#%%
"""
preprocessing of the IMAGE files & files with additional assumptions on vehcile materials (renaming, removing 27th region, adding labels etc.)
"""
range_interpol       = list(range(START_YEAR, END_YEAR + 1))
kilometrage          = kilometrage.reindex(range_interpol).interpolate(limit_direction='both')
kilometrage_bus      = kilometrage_bus.reindex(range_interpol).interpolate(limit_direction='both')
kilometrage_midi_bus = kilometrage_midi_bus.reindex(range_interpol).interpolate(limit_direction='both')
mileages             = mileages.reindex(range_interpol).interpolate(limit_direction='both')

region_list          = list(kilometrage.columns.values)     # get a list with region names

# select loadfactor for cars
car_loadfactor = loadfactor_car_data[['time','DIM_1', 5]].pivot_table(index='time', columns='DIM_1').droplevel(level=0, axis=1)  * LOAD_FACTOR # loadfactor for cars (in persons per vehicle) * LOAD_FACTOR to correct with te TIMER reference
car_loadfactor = car_loadfactor.apply(lambda x: [y if y >= 1 else 1 for y in x])                      # To avoid car load (person/vehicle) values ever going below 1, replace all values below 1 with 1
car_loadfactor = car_loadfactor.loc[list(range(START_YEAR, END_YEAR+1)),:]     # remove years beyond LAST_YEAR
car_loadfactor.columns = region_list

# select data only for requested output years
tonkms_Mtkms        = tonkms_Mtkms[tonkms_Mtkms['time'].isin(list(range(START_YEAR, END_YEAR+1)))]
passengerkms_Tpkms  = passengerkms_Tpkms[passengerkms_Tpkms['time'].isin(list(range(START_YEAR, END_YEAR+1)))] 
buses_vshares       = buses_vshares[buses_vshares['time'].isin(list(range(START_YEAR, END_YEAR+1)))] 
car_vshares         = car_vshares[car_vshares['time'].isin(list(range(START_YEAR, END_YEAR+1)))] 
medtruck_vshares    = medtruck_vshares[medtruck_vshares['time'].isin(list(range(START_YEAR, END_YEAR+1)))] 
hvytruck_vshares    = hvytruck_vshares[hvytruck_vshares['time'].isin(list(range(START_YEAR, END_YEAR+1)))] 
battery_shares_full = battery_shares_full.loc[list(range(START_YEAR, END_YEAR+1))]

#set multi-index based on the first two columns
tonkms_Mtkms.set_index(['time', 'DIM_1'], inplace=True)
passengerkms_Tpkms.set_index(['time', 'DIM_1'], inplace=True)
buses_vshares.set_index(['time', 'DIM_1'], inplace=True)
car_vshares.set_index(['time', 'DIM_1'], inplace=True)
medtruck_vshares.set_index(['time', 'DIM_1'], inplace=True)
hvytruck_vshares.set_index(['time', 'DIM_1'], inplace=True)

bus_label   = ['BusOil',	'BusBio',	'BusGas',	'BusElecTrolley',	'Bus Hybrid1',	'Bus Hybrid2',	'BusBattElectric', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']
truck_label = ['Conv. ICE(2000)',	'Conv. ICE(2010)', 'Adv. ICEOil',	'Adv. ICEH2', 'Turbo-petrol IC', 'Diesel ICEOil', 'Diesel ICEBio', 'ICE-HEV-gasoline', 'ICE-HEV-diesel oil', 'ICE-HEV-H2', 'ICE-HEV-CNG Gas', 'ICE-HEV-diesel bio', 'FCV Oil', 'FCV Bio', 'FCV H2', 'PEV-10 OilElec.', 'PEV-30 OilElec.', 'PEV-60 OilElec.', 'PEV-10 BioElec.', 'PEV-30 BioElec.', 'PEV-60 BioElec.', 'BEV Elec.', '', '', '']
car_label   = ['Conv. ICE(2000)',	'Conv. ICE(2010)', 'Adv. ICEOil',	'Adv. ICEH2', 'Turbo-petrol IC', 'Diesel ICEOil', 'Diesel ICEBio', 'ICE-HEV-gasoline', 'ICE-HEV-diesel oil', 'ICE-HEV-H2', 'ICE-HEV-CNG Gas', 'ICE-HEV-diesel bio', 'FCV Oil', 'FCV Bio', 'FCV H2', 'PEV-10 OilElec.', 'PEV-30 OilElec.', 'PEV-60 OilElec.', 'PEV-10 BioElec.', 'PEV-30 BioElec.', 'PEV-60 BioElec.', 'BEV Elec.', 'PHEV_BEV', 'BEV', 'Gas car']
tkms_label  = ['inland shipping', 'freight train', 'medium truck', 'heavy truck', 'air cargo', 'international shipping', 'empty', 'total']
pkms_label  = ['walking', 'biking', 'bus', 'train', 'car', 'hst', 'air', 'total']
columns_vehcile_output = ['Buses','Trains','HST','Cars','Planes','Bikes','Trucks','Cargo Trains','Ships','Inland ships','Cargo Planes']

# insert column descriptions
tonkms_Mtkms.columns       = tkms_label
passengerkms_Tpkms.columns = pkms_label
medtruck_vshares.columns   = truck_label
hvytruck_vshares.columns   = truck_label
buses_vshares.columns      = bus_label

# output transport drivers to output folder for 450 vs Bl comparisson in overarching figures later on
tonkms_Mtkms.to_csv('output\\' + FOLDER + '\\transport_tkms.csv', index=True)       # in Mega tkms
passengerkms_Tpkms.to_csv('output\\' + FOLDER + '\\transport_pkms.csv', index=True) # in Tera pkms

# aggregate car types into 5 car types
BEV_collist  = [22, 24]
PHEV_collist = [23, 21, 20, 19, 18, 17, 16]
ICE_collist  = [1,2,3,4,5,6,7,25]             # Gas car is considered ICE
HEV_collist  = [8,9,10,11,12]
FCV_collist  = [13,14,15]
car_types = ['ICE','HEV','PHEV','BEV','FCV']

index = pd.MultiIndex.from_product([list(kilometrage.index), list(range(1,27))], names=['time', 'DIM_1'])
vehicleshare_cars = pd.DataFrame(index=index, columns=car_types)
vehicleshare_cars.loc[idx[:,:],'ICE']  = car_vshares[ICE_collist].sum(axis=1).to_numpy()
vehicleshare_cars.loc[idx[:,:],'HEV']  = car_vshares[HEV_collist].sum(axis=1).to_numpy()
vehicleshare_cars.loc[idx[:,:],'PHEV'] = car_vshares[PHEV_collist].sum(axis=1).to_numpy()
vehicleshare_cars.loc[idx[:,:],'BEV']  = car_vshares[BEV_collist].sum(axis=1).to_numpy()
vehicleshare_cars.loc[idx[:,:],'FCV']  = car_vshares[FCV_collist].sum(axis=1).to_numpy()

# output to IRP
vehicleshare_cars.to_csv('output\\' + FOLDER + '\\car_type_share_regional.csv', index=True)

# labels etc.
x_graphs        = [i for i in range(START_YEAR, END_YEAR, 1)]                           # this is used as an x-axis for the years in graphs
labels_pas      = ['bicycle', 'rail_reg','rail_hst','midi_bus','reg_bus','air_pas','ICE','HEV','PHEV','BEV', 'FCV']             # Names used to shorten plots
labels_fre      = ['inland_shipping', 'rail_freight','LCV', 'MFT', 'HFT', 'air_freight', 'sea_shipping_small', 'sea_shipping_med', 'sea_shipping_large', 'sea_shipping_vl'] #names used to shorten plots
labels_materials= ['Steel', 'Aluminium', 'Cu', 'Plastics', 'Glass', 'Ti', 'Wood', 'Rubber', 'Li','Co','Ni', 'Mn','Nd','Pb']
labels_ev_batt  = ['NiMH','LMO','NMC','NCA','LFP','Lithium Sulfur','Lithium Ceramic','Lithium-air']


#%% For dynamic variables, apply interpolation and extend over the whole timeframe

# For some files, data is only found for a limited number of years, so we need to infer time-series before & after the data ends
def add_history_and_future(original, first_year, change='no'):  
    """
    This function fills data for the entire scenario period, even when not all years are provided in the input files. 
    Assuming constant values before the first and after the last available year and linear interpolation between available years. Equivalent to the Runge-Kutta interpolation in MyM
    input:  pandas dataframe with at least two timesteps defined as rows
    output: pandas dataframe with all scenario timesteps defined as rows (including the initial stock buildup phase, which differs between vehicles)
    """ 
    #determine the first and last year in the original data (start = first year of data available, first_year (input) = the first year in which vehicles existed & started building a stock) 
    start = original.first_valid_index()
    end = original.last_valid_index()

    #first interpolate between available years
    original_copy = original[:].reindex(list(range(start,end+1))).interpolate()
    
    # add the historic tail (assuming constant values)
    for row in range(first_year,start):
        original_copy.loc[row] = original_copy.loc[start].values
        history = original_copy.sort_index(axis=0)
        
    # add 'future' years after the latest year of the original data (depending on the 'change' settings these remain constant or grow based on an average annual growth rate)
    if change == 'no':
        # if change = 'no', just assume constant values after the last historic year
        for row in range(end, END_YEAR + 1):
            history.loc[row] = history.loc[end].values      # fill new years with the same values as in the last year of the origial data 
            new = history.sort_index(axis=0)
    else:
        # if change = 'yes', first determine the growth rate based on the historic series (start-to-end)
        growthrate = history.pct_change().sum(axis=0)/(end-start)
        # then apply the average annual growth rate to the last historic year
        for row in range(end+1, END_YEAR + 1):
            history.loc[row] = history.loc[row-1].values * (1+growthrate)
            new = history.sort_index(axis=0)
    return new 

# use the add_history_and_future() function to speciyfy dynamic variables over the entire scenario period
# complete & interpolate the vehicle weight data
vehicle_weight_kg_air_pas     = add_history_and_future(pd.DataFrame(vehicle_weight_kg_simple["air_pas"]),         first_year_vehicle['air_pas'].values[0],         change='no')
vehicle_weight_kg_air_frgt    = add_history_and_future(pd.DataFrame(vehicle_weight_kg_simple["air_freight"]),     first_year_vehicle['air_freight'].values[0],     change='no')
vehicle_weight_kg_rail_reg    = add_history_and_future(pd.DataFrame(vehicle_weight_kg_simple["rail_reg"]),        first_year_vehicle['rail_reg'].values[0],        change='no')
vehicle_weight_kg_rail_hst    = add_history_and_future(pd.DataFrame(vehicle_weight_kg_simple["rail_hst"]),        first_year_vehicle['rail_hst'].values[0],        change='no')
vehicle_weight_kg_rail_frgt   = add_history_and_future(pd.DataFrame(vehicle_weight_kg_simple["rail_freight"]),    first_year_vehicle['rail_freight'].values[0],    change='no')
vehicle_weight_kg_inland_ship = add_history_and_future(pd.DataFrame(vehicle_weight_kg_simple["inland_shipping"]), first_year_vehicle['inland_shipping'].values[0], change='no')
vehicle_weight_kg_bicycle     = add_history_and_future(pd.DataFrame(vehicle_weight_kg_simple["bicycle"]),         first_year_vehicle['bicycle'].values[0],         change='no')

vehicle_weight_kg_car    = add_history_and_future(vehicle_weight_kg_typical["car"].unstack(),     first_year_vehicle['car'].values[0],     change='no')
vehicle_weight_kg_LCV    = add_history_and_future(vehicle_weight_kg_typical["LCV"].unstack(),     first_year_vehicle['LCV'].values[0],     change='no')
vehicle_weight_kg_MFT    = add_history_and_future(vehicle_weight_kg_typical["MFT"].unstack(),     first_year_vehicle['MFT'].values[0],     change='no')
vehicle_weight_kg_HFT    = add_history_and_future(vehicle_weight_kg_typical["HFT"].unstack(),     first_year_vehicle['HFT'].values[0],     change='no')
vehicle_weight_kg_bus    = add_history_and_future(vehicle_weight_kg_typical["reg_bus"].unstack(), first_year_vehicle['reg_bus'].values[0], change='no')
vehicle_weight_kg_midi   = add_history_and_future(vehicle_weight_kg_typical["midi_bus"].unstack(),first_year_vehicle['midi_bus'].values[0],change='no')

# complete & interpolate the vehicle composition data (simple first)
material_fractions_air_pas     = add_history_and_future(material_fractions['air_pas'].unstack(),            first_year_vehicle['air_pas'].values[0],         change='no')
material_fractions_air_frgt    = add_history_and_future(material_fractions['air_freight'].unstack(),        first_year_vehicle['air_freight'].values[0],     change='no')
material_fractions_rail_reg    = add_history_and_future(material_fractions['rail_reg'].unstack(),           first_year_vehicle['rail_reg'].values[0],        change='no')
material_fractions_rail_hst    = add_history_and_future(material_fractions['rail_hst'].unstack(),           first_year_vehicle['rail_hst'].values[0],        change='no')
material_fractions_rail_frgt   = add_history_and_future(material_fractions['rail_freight'].unstack(),       first_year_vehicle['rail_freight'].values[0],    change='no')
material_fractions_ship_small  = add_history_and_future(material_fractions['sea_shipping_small'].unstack(), FIRST_YEAR_BOATS,                                change='no')
material_fractions_ship_medium = add_history_and_future(material_fractions['sea_shipping_med'].unstack(),   FIRST_YEAR_BOATS,                                change='no')
material_fractions_ship_large  = add_history_and_future(material_fractions['sea_shipping_large'].unstack(), FIRST_YEAR_BOATS,                                change='no')
material_fractions_ship_vlarge = add_history_and_future(material_fractions['sea_shipping_vl'].unstack(),    FIRST_YEAR_BOATS,                                change='no')
material_fractions_inland_ship = add_history_and_future(material_fractions['inland_shipping'].unstack(),    first_year_vehicle['inland_shipping'].values[0], change='no')
material_fractions_bicycle     = add_history_and_future(material_fractions['bicycle'].unstack(),            first_year_vehicle['bicycle'].values[0],         change='no')

# complete & interpolate the vehicle composition data (by vehicle sub-type second): runtime appr. 11 min.
material_fractions_car        = add_history_and_future(material_fractions_type['car'].unstack(),            first_year_vehicle['car'].values[0],             change='no')
material_fractions_bus_reg    = add_history_and_future(material_fractions_type['reg_bus'].unstack(),        first_year_vehicle['reg_bus'].values[0],         change='no')
material_fractions_bus_midi   = add_history_and_future(material_fractions_type['midi_bus'].unstack(),       first_year_vehicle['midi_bus'].values[0],        change='no')
material_fractions_truck_HFT  = add_history_and_future(material_fractions_type['HFT'].unstack(),            first_year_vehicle['HFT'].values[0],             change='no')
material_fractions_truck_MFT  = add_history_and_future(material_fractions_type['MFT'].unstack(),            first_year_vehicle['MFT'].values[0],             change='no')
material_fractions_truck_LCV  = add_history_and_future(material_fractions_type['LCV'].unstack(),            first_year_vehicle['LCV'].values[0],             change='no')

# interpolate & complete series for battery weights, shares & composition too
battery_weights_full    = add_history_and_future(battery_weights.unstack(),   START_YEAR)
battery_materials_full  = add_history_and_future(battery_materials.unstack(), START_YEAR)

# same for lifetime data
lifetimes_vehicles_mean  = add_history_and_future(lifetimes_vehicles.loc[idx[:,"mean"],:].droplevel(["data"]), START_YEAR)
lifetimes_vehicles_stdev = add_history_and_future(lifetimes_vehicles.loc[idx[:,"stdev"],:].droplevel(["data"]), START_YEAR)
lifetimes_vehicles_shape = add_history_and_future(lifetimes_vehicles.loc[idx[:,"shape"],:].droplevel(["data"]), START_YEAR)
lifetimes_vehicles_scale = add_history_and_future(lifetimes_vehicles.loc[idx[:,"scale"],:].droplevel(["data"]), START_YEAR)

#%% Caculating the tonnekilometres for all freight and passenger vehicle types (adjustments are made to: freight air, trucks, and buses)

# Trucks are calculated differently because the IMAGE model does not account for LCV trucks, which concerns a large portion of the material requirements of road freight
# the total trucks Tkms remain the same, but a LCV fraction is substracted, and the remainder is re-assigned to medium and heavy trucks according to their original ratio
trucks_total_tkm       = tonkms_Mtkms['medium truck'].unstack() +  tonkms_Mtkms['heavy truck'].unstack()
trucks_LCV_tkm         = trucks_total_tkm * 0.04                                    # 0.04 is the fraction of the tkms driven by light commercial vehicles according to the IEA
MFT_percshare_tkm      = tonkms_Mtkms['medium truck'].unstack() / trucks_total_tkm  # the MFT fraction of the total 
HFT_percshare_tkm      = tonkms_Mtkms['heavy truck'].unstack() / trucks_total_tkm   # the HFT fraction of the total 
trucks_min_LCV         = trucks_total_tkm - trucks_LCV_tkm
trucks_MFT_tkm         = trucks_min_LCV.mul(MFT_percshare_tkm)                      
trucks_HFT_tkm         = trucks_min_LCV.mul(HFT_percshare_tkm)

# demand for freight planes is reduced by 50% because about half of the air freight is transported as cargo on passenger planes 
air_freight_tkms       = tonkms_Mtkms['air cargo'].unstack() * market_share['air_freight'].values[0]

# Buses are adjusted to account for the higher material intensity of mini-buses
bus_regl_pkms          = passengerkms_Tpkms['bus'].unstack() * market_share['reg_bus'].values[0]   # in tera pkms
bus_midi_pkms          = passengerkms_Tpkms['bus'].unstack() * market_share['midi_bus'].values[0]  # in tera pkms

# Select tkms of passenger cars (which will be adjusted to represent 5 types: ICE, HEV, PHEV, BEV & FCV)
car_pkms               = passengerkms_Tpkms['car'].unstack()
car_pkms               = car_pkms.drop([27, 28], axis=1)    # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies  # in tera pkms                            
car_pkms.columns       = region_list                                  

#%% Calculate the NUMBER OF VEHICLES (stock, on the road) to fulfull the ton-kilometers transport demand
 
def tkms_to_nr_of_vehicles_fixed(tera_tkms, mileage, load, loadfactor):
   """
   This function translates ton kilometers (by year & by region) to nr of vehicles (same dimms) 
   using fixed indicators on mileage, load capacity and load factor
   """
   # first_translate Tera ton/person- kms into person/ton-kms
   tkms = tera_tkms * 1000000000000  
   # then get the vehicle kilometers required to fulfill the transport demand
   vkms = tkms/(load*loadfactor)
   # then get the number of vehicles by dividing by the mileage
   nr_of_vehicles = vkms.div(mileage, axis=0)
   return nr_of_vehicles

#calculate the number of vehicles on the road (first passenger, then freight)
air_pas_nr      = tkms_to_nr_of_vehicles_fixed(passengerkms_Tpkms['air'].unstack(),    mileages['air_pas'],  load['air_pas'].values[0],    loadfactor['air_pas'].values[0])
rail_reg_nr     = tkms_to_nr_of_vehicles_fixed(passengerkms_Tpkms['train'].unstack(),  mileages['rail_reg'], load['rail_reg'].values[0],   loadfactor['rail_reg'].values[0])
rail_hst_nr     = tkms_to_nr_of_vehicles_fixed(passengerkms_Tpkms['hst'].unstack(),    mileages['rail_hst'], load['rail_hst'].values[0],   loadfactor['rail_hst'].values[0])
bikes_nr        = tkms_to_nr_of_vehicles_fixed(passengerkms_Tpkms['biking'].unstack(), mileages['bicycle'],  load['bicycle'].values[0],    loadfactor['bicycle'].values[0])

# original ton kilometers are in Mega-ton-kms, div by 1000000 to harmonize with pkms which are in Tera pkms
trucks_HFT_nr   = tkms_to_nr_of_vehicles_fixed(trucks_HFT_tkm/1000000,   mileages['HFT'],          load['HFT'].values[0],         loadfactor['HFT'].values[0])
trucks_MFT_nr   = tkms_to_nr_of_vehicles_fixed(trucks_MFT_tkm/1000000,   mileages['MFT'],          load['MFT'].values[0],         loadfactor['MFT'].values[0])
trucks_LCV_nr   = tkms_to_nr_of_vehicles_fixed(trucks_LCV_tkm/1000000,   mileages['LCV'],          load['LCV'].values[0],         loadfactor['LCV'].values[0])
air_freight_nr  = tkms_to_nr_of_vehicles_fixed(air_freight_tkms/1000000,  mileages['air_freight'], load['air_freight'].values[0], loadfactor['air_freight'].values[0])
rail_freight_nr = tkms_to_nr_of_vehicles_fixed(tonkms_Mtkms['freight train'].unstack()/1000000, mileages['rail_freight'], load['rail_freight'].values[0],         loadfactor['rail_freight'].values[0])
inland_ship_nr  = tkms_to_nr_of_vehicles_fixed(tonkms_Mtkms['inland shipping'].unstack()/1000000, mileages['inland_shipping'], load['inland_shipping'].values[0], loadfactor['inland_shipping'].values[0])

# passenger cars and buses are calculated separately (due to regional & changeing mileage & load), first the totals
car_total_vkms  = car_pkms.div(car_loadfactor) * 1000000000000    # now in kms
car_total_nr    = car_total_vkms.div(kilometrage)                 # total number of cars
car_total_nr.columns = list(range(1,27))                          # remove region labels (for use in functions later on)

# for buses do the same, but first remove region 27 & 28 (empty & world total) & kilometrage column names
bus_regl_pkms  = bus_regl_pkms.drop([27, 28], axis=1) 
kilometrage_bus.columns = list(range(1,27))
bus_regl_vkms  = bus_regl_pkms.div(load['reg_bus'].values[0] * loadfactor['reg_bus'].values[0]) * 1000000000000    # now in kms
bus_regl_nr    = bus_regl_vkms.div(kilometrage_bus)                              # total number of regular buses

bus_midi_pkms  = bus_midi_pkms.drop([27, 28], axis=1) 
kilometrage_midi_bus.columns = list(range(1,27))   
bus_midi_vkms  = bus_midi_pkms.div(load['midi_bus'].values[0] * loadfactor['midi_bus'].values[0]) * 1000000000000   # now in kms
bus_midi_nr    = bus_midi_vkms.div(kilometrage_midi_bus)                         # total number of regular buses


#%% for INTERNATIONAL SHIPPING the number of vehicles is calculated differently 

cap_adjustment  = [1, 1, 1, 1]
mile_adjustment = [1, 1, 1, 1]

#pre-calculate the shares of the boats based on the number of boats, before adding history/future
share_of_boats       = nr_of_boats.div(nr_of_boats.sum(axis=1), axis=0)

share_of_boats_yrs    =  add_history_and_future(share_of_boats   , FIRST_YEAR_BOATS,   change='yes')   # could be 'yes' based on better data
cap_of_boats_yrs      =  add_history_and_future(cap_of_boats     , FIRST_YEAR_BOATS,   change='no')    # could be 'yes' based on better data
loadfactor_boats_yrs  =  add_history_and_future(loadfactor_boats , FIRST_YEAR_BOATS,   change='no')   
mileage_boats_yrs     =  add_history_and_future(mileage_boats    , FIRST_YEAR_BOATS,   change='no')  
weight_frac_boats_yrs =  add_history_and_future(weight_boats     , FIRST_YEAR_BOATS,   change='no')  

# normalize the share of boats to 1 & adjust the capacity & mileage for smaller ships 
share_of_boats_yrs   = share_of_boats_yrs.div(share_of_boats_yrs.sum(axis=1), axis=0)
cap_of_boats_yrs     = cap_of_boats_yrs.mul(cap_adjustment, axis=1)
mileage_boats_yrs    = mileage_boats_yrs.mul(mile_adjustment, axis=1)

# now derive the number of ships for 4 ship types in four steps: 
# 1) get the share of the ship types in the Tkms shipped (what % of total tkms shipped goes by what ship type?)
share_of_boats_tkm_all   = share_of_boats_yrs * cap_of_boats_yrs * loadfactor_boats_yrs * mileage_boats_yrs
share_of_boats_tkm       = share_of_boats_tkm_all.div(share_of_boats_tkm_all.sum(axis=1), axis=0)

# 2) get the total tkms shipped by ship-type. (The shares are pre-calculated from 1900 onwards, so a selection from 1971-onwards is applied here)
ship_small_tkm  = tonkms_Mtkms['international shipping'].unstack().mul(share_of_boats_tkm['Small'].loc[START_YEAR:], axis=0)
ship_medium_tkm = tonkms_Mtkms['international shipping'].unstack().mul(share_of_boats_tkm['Medium'].loc[START_YEAR:], axis=0)
ship_large_tkm  = tonkms_Mtkms['international shipping'].unstack().mul(share_of_boats_tkm['Large'].loc[START_YEAR:], axis=0)
ship_vlarge_tkm = tonkms_Mtkms['international shipping'].unstack().mul(share_of_boats_tkm['Very Large'].loc[START_YEAR:], axis=0)

# 3) get the vehicle-kms by ship type (multiply by 1000000 to get from Mega-tkm to tkm)
ship_small_vehkm  = ship_small_tkm.mul(1000000).div(cap_of_boats_yrs['Small'].loc[START_YEAR:], axis=0)
ship_medium_vehkm = ship_medium_tkm.mul(1000000).div(cap_of_boats_yrs['Medium'].loc[START_YEAR:], axis=0)  
ship_large_vehkm  = ship_large_tkm.mul(1000000).div(cap_of_boats_yrs['Large'].loc[START_YEAR:], axis=0)  
ship_vlarge_vehkm = ship_vlarge_tkm.mul(1000000).div(cap_of_boats_yrs['Very Large'].loc[START_YEAR:], axis=0) 

# 4) get the number of ships (stock) by dividing with the mileage
ship_small_nr  = ship_small_vehkm.div(mileage_boats_yrs['Small'].loc[START_YEAR:], axis=0)
ship_medium_nr = ship_medium_vehkm.div(mileage_boats_yrs['Medium'].loc[START_YEAR:], axis=0)  
ship_large_nr  = ship_large_vehkm.div(mileage_boats_yrs['Large'].loc[START_YEAR:], axis=0)  
ship_vlarge_nr = ship_vlarge_vehkm.div(mileage_boats_yrs['Very Large'].loc[START_YEAR:], axis=0) 

# for comparison, we find the difference of the known and the calculated nr of ships (global total) in the period 2005-2018
diff_ships = pd.DataFrame().reindex_like(nr_of_boats)
diff_ships['Small']      =  ship_small_nr.loc[list(range(2005,2018+1)), 28].div(nr_of_boats['Small'])
diff_ships['Medium']     =  ship_medium_nr.loc[list(range(2005,2018+1)), 28].div(nr_of_boats['Medium'])
diff_ships['Large']      =  ship_large_nr.loc[list(range(2005,2018+1)), 28].div(nr_of_boats['Large'])
diff_ships['Very Large'] =  ship_vlarge_nr.loc[list(range(2005,2018+1)), 28].div(nr_of_boats['Very Large'])

total_nr_of_ships = ship_small_nr + ship_medium_nr + ship_large_nr + ship_vlarge_nr
diff_ships_total = total_nr_of_ships.loc[list(range(2005,2018+1)), 28].div(nr_of_boats.sum(axis=1))

#%% Export intermediate indicators (a.o. files on nr. of vehicles, pkms/tkms)

# Export total global number of vehicles in the fleet (stock) as csv
region_list = list(range(1,27))
index = pd.MultiIndex.from_product([list(total_nr_of_ships.index), region_list], names = ['years','regions'])
total_nr_vehicles = pd.DataFrame(index=index, columns=columns_vehcile_output)
total_nr_vehicles['Buses']        = bus_regl_nr[region_list].stack() + bus_midi_nr[region_list].stack()
total_nr_vehicles['Trains']       = rail_reg_nr[region_list].stack()
total_nr_vehicles['HST']          = rail_hst_nr[region_list].stack()
total_nr_vehicles['Cars']         = car_total_nr[region_list].stack()
total_nr_vehicles['Planes']       = air_pas_nr[region_list].stack()
total_nr_vehicles['Bikes']        = bikes_nr[region_list].stack()
total_nr_vehicles['Trucks']       = trucks_HFT_nr[region_list].stack() + trucks_MFT_nr[region_list].stack() + trucks_LCV_nr[region_list].stack()
total_nr_vehicles['Cargo Trains'] = rail_freight_nr[region_list].stack()
total_nr_vehicles['Ships']        = total_nr_of_ships[region_list].stack()
total_nr_vehicles['Inland ships'] = inland_ship_nr[region_list].stack()  
total_nr_vehicles['Cargo Planes'] = air_freight_nr[region_list].stack() 

total_nr_vehicles.sum(axis=0, level=0).to_csv('output\\' + FOLDER + '\\global_vehicle_nr.csv', index=True) # total global nr of vehicles 
total_nr_vehicles.to_csv('output\\' + FOLDER + '\\region_vehicle_nr.csv', index=True) # regional nr of vehicles 

# Generate csv output file on pkms & tkms (same format as files on number of vehicles (also used later on)), unit: pkm or tkm 
region_list = list(range(1,27))
car_pkms.columns = list(range(1,27))      # remove region labels 
index = pd.MultiIndex.from_product([list(total_nr_of_ships.index), region_list], names = ['years','regions'])
total_pkm_tkm = pd.DataFrame(index=index, columns=columns_vehcile_output)
total_pkm_tkm['Buses']        = (bus_regl_pkms[region_list].stack() + bus_midi_pkms[region_list].stack()) * 1000000000000
total_pkm_tkm['Trains']       = passengerkms_Tpkms['train']   * 1000000000000                              
total_pkm_tkm['HST']          = passengerkms_Tpkms['hst']     * 1000000000000
total_pkm_tkm['Cars']         = car_pkms[region_list].stack() * 1000000000000
total_pkm_tkm['Planes']       = passengerkms_Tpkms['air']     * 1000000000000
total_pkm_tkm['Bikes']        = passengerkms_Tpkms['biking']  * 1000000000000
total_pkm_tkm['Trucks']       = (trucks_HFT_tkm[region_list].stack() + trucks_MFT_tkm[region_list].stack() + trucks_LCV_tkm[region_list].stack()) * 1000000
total_pkm_tkm['Cargo Trains'] = tonkms_Mtkms['freight train'] * 1000000
total_pkm_tkm['Ships']        = tonkms_Mtkms['international shipping'] * 1000000 
total_pkm_tkm['Inland ships'] = tonkms_Mtkms['inland shipping'] * 1000000
total_pkm_tkm['Cargo Planes'] = air_freight_tkms[region_list].stack() * 1000000     # mind that these are the tkms flows with cargo planes, real demand for air cargo is higher due to 50% hitching with passenger flights

total_pkm_tkm.sum(axis=0, level=0).to_csv('output\\' + FOLDER + '\\global_pkm_tkm.csv', index=True) # total global pkms & tkms 
total_pkm_tkm.to_csv('output\\' + FOLDER + '\\region_pkm_tkm.csv', index=True)  # regional pkms & tkms 

# some indicators for the model accuracy comparison
inland_ship_nr[[2,11,16,20]].loc[2015].sum()           # inland shipping in 2015 (China, Russia, Europe & US)
car_total_nr[[11,12]].loc[2018].sum()
rail_reg_nr[[1,2,11,12,18,20,23]].loc[2017].sum()      # India, Canada, China, United States, Europe, Japan
rail_freight_nr[[1,2,11,12,18,20,16]].loc[2016].sum()  # India, Canada, China, United States, Europe, Russia

#%% INFLOW-OUTFLOW calculations using the ODYM Dynamic Stock Model (DSM) as a function

# In order to calculate inflow & outflow smoothly (without peaks for the initial years), 
def stock_tail(stock, first_year_veh, choice='stock'):
    """
    In order to avoid an initial inflow shock, this function calculates the size of the initial development of the vehicle stock (in number of vehicles) before the first scenario year of the IMAGE model (1971).
    It does so by adding a so-called historic tail to the stock, by adding a 0 value for first year of operation (e.g. 1926), then linear interpolation of values towards 1971    
    """
    if choice=='stock':
       zero_value = [0 for i in range(0, REGIONS)]
    else:
       zero_value = stock.head(1).values[0]
       
    stock_used = pd.DataFrame(stock).reindex_like(stock)
    stock_used.loc[first_year_veh] = zero_value  # set all regions to 0 in the year of initial operation
    stock_new  = stock_used.reindex(list(range(first_year_veh, END_YEAR + 1))).interpolate()
    return stock_new

# lifetimes also need a tail, but this doesn't start at 0, but remains constant at the first known value (assumed)
def lifetime_tail(lifetime_vehicles_in, vehicle):
    first_year = first_year_vehicle[vehicle].values[0]
    lifetime_vehicles_out = lifetime_vehicles_in[vehicle]
    for year in range(first_year, START_YEAR):
        lifetime_vehicles_out[year] = lifetime_vehicles_in[vehicle][START_YEAR]
    return lifetime_vehicles_out.reindex(list(range(first_year, END_YEAR + 1)))

# apply the dynamic stock model
def inflow_outflow_dynamic_np(stock, fact1, fact2, distribution):
    """
    In this function the stock-driven DSM is applied to return inflow & outflow for all regions
    IN: stock as a dataframe (nr of vehicles by year & region), lifetime (in years), stdev_mult (multiplier to derive the standard deviation from the mean)
    OUT: 3 dataframes of stock, inflow & outflow (nr of vehicles by year^2 & region)
    
    This DYNAMIC variant of the DSM calculates & returns the full unadjusted (cohort specific) matrix (YES, very memory intensive, but neccesary for dynamic assesmment, i.e. changing vehicle weights & compositions)
    """
    inflow         = np.empty((len(stock[0]), len(stock)))
    outflow_cohort = np.empty((len(stock[0]), len(stock), len(stock))) 
    stock_cohort   = np.empty((len(stock[0]), len(stock), len(stock))) 

    # define mean (fact1) & standard deviation (fact2) (applied accross all regions) (For Weibull fact1 = shape & fact2 = scale)
                     
    # fact 1 is a list with the mean lifetime (for Weibull: it's the shape parameter), given for every timestep (e.g. 1900-2100), needed for the DSM as it allows to change lifetime for different cohort
    # fact 2 is a list with the standarddeviation as a fraction of the mean lifetime (for Weibull: it's the scale parameter)
    if distribution == 'FoldedNormal':
       fact2_list = fact1.mul(fact2)   # for the (Folded) normal distribution, the standard_deviation is calculated as the mean times the multiplier (i.e. the standard deviation as a fraction of the mean)
    else:
       fact2_list = fact2

    for region in range(0,len(stock[0])):
        # define and run the DSM   
        if distribution == 'FoldedNormal':
           DSMforward = DSM(t = np.arange(0,len(stock[:,region]),1), s=stock[:,region], lt = {'Type': 'FoldedNormal', 'Mean': np.array(fact1), 'StdDev': np.array(fact2_list)})  # definition of the DSM based on a folded normal distribution
        else:
           DSMforward = DSM(t = np.arange(0,len(stock[:,region]),1), s=stock[:,region], lt = {'Type': 'Weibull', 'Shape': np.array(fact1), 'Scale': np.array(fact2_list)})       # definition of the DSM based on a Weibull distribution
           
        out_sc, out_oc, out_i = DSMforward.compute_stock_driven_model_surplus()                                                                 # run the DSM, to give 3 outputs: stock_by_cohort, outflow_by_cohort & inflow_per_year
        
        # store the regional results in the return dataframe (full range of years (e.g. 1900 onwards), for later use in material calculations  & no moving average applied here)
        outflow_cohort[region,:,:] = out_oc                                                                                                      # sum the outflow by cohort to get the total outflow per year
        stock_cohort[region,:,:]   = out_sc
        inflow[region,:]           = out_i
         
    return inflow.T, outflow_cohort, stock_cohort     


# Pre-calculate the inflow (aka. market-)share corresponding to the (known) share of vehicles in stock (from IMAGE)
def inflow_outflow_typical_np(stock, fact1, fact2, distribution, stock_share):
   """
   In this function the dynamic inflow & outflow calculations are applied to vehicles with sub-types. To return inflow & outflow for all regions & vehicle types
   IN: stock as a dataframe (nr of vehicles by year, region and type), lifetime (in years), stdev_mult (multiplier to derive the standard deviation from the mean)
   OUT: 3 dataframes of stock, inflow & outflow (nr of vehicles by year^2 & region & type)
   
   This TYPICAL variant of the DSM calculates & returns the full unadjusted (cohort specific) matrix (time*time) for every region AND vehicle type (YES, very memory intensive, but neccesary for dynamic assesmment, i.e. changing vehicle weights & compositions)
   """
   initial_year        = stock.first_valid_index()
   initial_year_shares = stock_share.first_valid_index()[0]
   
   inflow              = np.zeros((len(stock_share.iloc[0]), len(stock.iloc[0]), len(stock)))
   outflow_cohort      = np.zeros((len(stock_share.iloc[0]), len(stock.iloc[0]), len(stock), len(stock)))
   stock_cohort        = np.zeros((len(stock_share.iloc[0]), len(stock.iloc[0]), len(stock), len(stock)))

   index = pd.MultiIndex.from_product([stock_share.columns, stock.index], names=['type', 'time'])
   stock_by_vtype      = pd.DataFrame(0, index=index, columns=stock.columns)
   stock_share_used    = stock_share.unstack().stack(level=0).reorder_levels([1,0]).sort_index()
   
   # before running the DSM: 
   # 1) extend the historic coverage of the stock shares (assuming constant pre-1971), and
   for vtype in list(stock_share.columns):
      stock_share_used.loc[idx[vtype, initial_year],:] = stock_share[vtype].unstack().loc[initial_year_shares]
   stock_share_used = stock_share_used.reindex(index).interpolate()
   
   # 2) calculate the stocks of individual (vehicle) types (in nr of vehicles)
   for vtype in list(stock_share.columns):
      stock_by_vtype.loc[idx[vtype,:],:] = stock.mul(stock_share_used.loc[idx[vtype,:],:])

   vtype_list  = list(stock_by_vtype.index.unique('type'))
   vtype_count = list(range(0,len(stock_share.columns))) 
   vtype_dict   = dict(zip(vtype_count, vtype_list))

   # Then run the original DSM for each vehicle type & add it to the inflow, ouflow & stock containers, (for stock: index = vtype, time; columns = region, time)
   for vtype in range(0,len(stock_share.columns)): 
      if stock_share.iloc[:,vtype].sum() > 0.001:
         dsm_inflow, dsm_outflow_coh, dsm_stock_coh = inflow_outflow_dynamic_np(stock_by_vtype.loc[idx[vtype_dict[vtype],:],:].to_numpy(), fact1, fact2, distribution)
         
         inflow[vtype,:,:]           = dsm_inflow.T
         outflow_cohort[vtype,:,:,:] = dsm_outflow_coh
         stock_cohort[vtype,:,:,:]   = dsm_stock_coh
      
      else:
         pass
        
   return inflow, outflow_cohort, stock_cohort

# Calculate the historic tail (& reduce regions to 26)
air_pas_nr     = stock_tail(air_pas_nr[list(range(1,REGIONS+1))],  first_year_vehicle['air_pas'].values[0])
rail_reg_nr    = stock_tail(rail_reg_nr[list(range(1,REGIONS+1))], first_year_vehicle['rail_reg'].values[0])
rail_hst_nr    = stock_tail(rail_hst_nr[list(range(1,REGIONS+1))], first_year_vehicle['rail_hst'].values[0])
bikes_nr       = stock_tail(bikes_nr[list(range(1,REGIONS+1))],    first_year_vehicle['bicycle'].values[0])

air_freight_nr  = stock_tail(air_freight_nr[list(range(1, REGIONS+1))],  first_year_vehicle['air_freight'].values[0])
rail_freight_nr = stock_tail(rail_freight_nr[list(range(1, REGIONS+1))], first_year_vehicle['rail_freight'].values[0])
inland_ship_nr  = stock_tail(inland_ship_nr[list(range(1, REGIONS+1))],  first_year_vehicle['inland_shipping'].values[0])
ship_small_nr   = stock_tail(ship_small_nr[list(range(1, REGIONS+1))],   1900)
ship_medium_nr  = stock_tail(ship_medium_nr[list(range(1, REGIONS+1))],  1900)
ship_large_nr   = stock_tail(ship_large_nr[list(range(1, REGIONS+1))],   1900)
ship_vlarge_nr  = stock_tail(ship_vlarge_nr[list(range(1, REGIONS+1))],  1900)

bus_regl_nr     = stock_tail(bus_regl_nr[list(range(1, REGIONS+1))],  first_year_vehicle['reg_bus'].values[0])
bus_midi_nr     = stock_tail(bus_midi_nr[list(range(1, REGIONS+1))],  first_year_vehicle['midi_bus'].values[0])
car_total_nr    = stock_tail(car_total_nr[list(range(1, REGIONS+1))], first_year_vehicle['car'].values[0])

trucks_HFT_nr   = stock_tail(trucks_HFT_nr[list(range(1, REGIONS+1))], first_year_vehicle['HFT'].values[0])
trucks_MFT_nr   = stock_tail(trucks_HFT_nr[list(range(1, REGIONS+1))], first_year_vehicle['MFT'].values[0])
trucks_LCV_nr   = stock_tail(trucks_LCV_nr[list(range(1, REGIONS+1))], first_year_vehicle['LCV'].values[0])

##################### DYNAMIC MODEL (runtime: ca. 30 sec) ########################################################################
# Calculate the NUMBER of vehicles, total for inflow & by cohort for stock & outflow, first only for simple vehicles

air_pas_in,      air_pas_out_coh,      air_pas_stock_coh      = inflow_outflow_dynamic_np(air_pas_nr.to_numpy(),    lifetime_tail(lifetimes_vehicles_mean,'air_pas'),  lifetime_tail(lifetimes_vehicles_stdev,'air_pas'),  'FoldedNormal')
rail_reg_in,     rail_reg_out_coh,     rail_reg_stock_coh     = inflow_outflow_dynamic_np(rail_reg_nr.to_numpy(),   lifetime_tail(lifetimes_vehicles_mean,'rail_reg'), lifetime_tail(lifetimes_vehicles_stdev,'rail_reg'), 'FoldedNormal')
rail_hst_in,     rail_hst_out_coh,     rail_hst_stock_coh     = inflow_outflow_dynamic_np(rail_hst_nr.to_numpy(),   lifetime_tail(lifetimes_vehicles_mean,'rail_hst'), lifetime_tail(lifetimes_vehicles_stdev,'rail_hst'), 'FoldedNormal')
bikes_in,        bikes_out_coh,        bikes_stock_coh        = inflow_outflow_dynamic_np(bikes_nr.to_numpy(),      lifetime_tail(lifetimes_vehicles_mean,'bicycle'),  lifetime_tail(lifetimes_vehicles_stdev,'bicycle'),  'FoldedNormal')

air_freight_in,  air_freight_out_coh,  air_freight_stock_coh  = inflow_outflow_dynamic_np(air_freight_nr.to_numpy(),  lifetime_tail(lifetimes_vehicles_mean,'air_freight'),        lifetime_tail(lifetimes_vehicles_stdev,'air_freight'),        'FoldedNormal')
rail_freight_in, rail_freight_out_coh, rail_freight_stock_coh = inflow_outflow_dynamic_np(rail_freight_nr.to_numpy(), lifetime_tail(lifetimes_vehicles_mean,'rail_freight'),       lifetime_tail(lifetimes_vehicles_stdev,'rail_freight'),       'FoldedNormal')
inland_ship_in,  inland_ship_out_coh,  inland_ship_stock_coh  = inflow_outflow_dynamic_np(inland_ship_nr.to_numpy(),  lifetime_tail(lifetimes_vehicles_mean,'inland_shipping'),    lifetime_tail(lifetimes_vehicles_stdev,'inland_shipping'),    'FoldedNormal')
ship_small_in,   ship_small_out_coh,   ship_small_stock_coh   = inflow_outflow_dynamic_np(ship_small_nr.to_numpy(),   lifetime_tail(lifetimes_vehicles_mean,'sea_shipping_small'), lifetime_tail(lifetimes_vehicles_stdev,'sea_shipping_small'), 'FoldedNormal')
ship_medium_in,  ship_medium_out_coh,  ship_medium_stock_coh  = inflow_outflow_dynamic_np(ship_medium_nr.to_numpy(),  lifetime_tail(lifetimes_vehicles_mean,'sea_shipping_med'),   lifetime_tail(lifetimes_vehicles_stdev,'sea_shipping_med'),   'FoldedNormal')
ship_large_in,   ship_large_out_coh,   ship_large_stock_coh   = inflow_outflow_dynamic_np(ship_large_nr.to_numpy(),   lifetime_tail(lifetimes_vehicles_mean,'sea_shipping_large'), lifetime_tail(lifetimes_vehicles_stdev,'sea_shipping_large'), 'FoldedNormal')
ship_vlarge_in,  ship_vlarge_out_coh,  ship_vlarge_stock_coh  = inflow_outflow_dynamic_np(ship_vlarge_nr.to_numpy(),  lifetime_tail(lifetimes_vehicles_mean,'sea_shipping_vl'),    lifetime_tail(lifetimes_vehicles_stdev,'sea_shipping_vl'),    'FoldedNormal')

#%% BATTERY VEHICLE CALCULATIONS - Determine the fraction of the fleet that uses batteries, based on vehicle share files
# Batteries are relevant for 1) BUSES 2) TRUCKS
# We use fixed weight & material content assumptions, but we use the development of battery energy density (from the electricity storage calculations) to derive a changing battery capacity (and thus range)
# Battery weight is assumed to be in-addition to the regular vehicle weight

bus_label        = ['BusOil',	'BusBio',	'BusGas',	'BusElecTrolley',	'Bus Hybrid1',	'Bus Hybrid2',	'BusBattElectric', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']
bus_label_ICE    = ['BusOil',	'BusBio',	'BusGas']
bus_label_HEV    = ['BusElecTrolley',	'Bus Hybrid1']
truck_label      = ['Conv. ICE(2000)',	'Conv. ICE(2010)', 'Adv. ICEOil',	'Adv. ICEH2', 'Turbo-petrol IC', 'Diesel ICEOil', 'Diesel ICEBio', 'ICE-HEV-gasoline', 'ICE-HEV-diesel oil', 'ICE-HEV-H2', 'ICE-HEV-CNG Gas', 'ICE-HEV-diesel bio', 'FCV Oil', 'FCV Bio', 'FCV H2', 'PEV-10 OilElec.', 'PEV-30 OilElec.', 'PEV-60 OilElec.', 'PEV-10 BioElec.', 'PEV-30 BioElec.', 'PEV-60 BioElec.', 'BEV Elec.', 'BEV Elec.', 'BEV Elec.', 'BEV Elec.']
truck_label_ICE  = ['Conv. ICE(2000)',	'Conv. ICE(2010)', 'Adv. ICEOil', 'Adv. ICEH2', 'Turbo-petrol IC', 'Diesel ICEOil', 'Diesel ICEBio']
truck_label_HEV  = ['ICE-HEV-gasoline', 'ICE-HEV-diesel oil', 'ICE-HEV-H2', 'ICE-HEV-CNG Gas', 'ICE-HEV-diesel bio']
truck_label_PHEV = ['PEV-10 OilElec.', 'PEV-30 OilElec.', 'PEV-60 OilElec.', 'PEV-10 BioElec.', 'PEV-30 BioElec.', 'PEV-60 BioElec.']
truck_label_BEV  = ['BEV Elec.', 'BEV Elec.', 'BEV Elec.']
truck_label_FCV  = ['FCV Oil', 'FCV Bio', 'FCV H2']
vshares_label    = ['ICE', 'HEV', 'PHEV', 'BEV', 'FCV', 'Trolley']

# 1) BUSES: original vehcile shares are distributed into two vehicle types (regular and small midi buses)
# vehicle shares are grouped as: a) ICE, b) HEV, c) trolley, d) BEV, but trolley buses are not relevant for midi busses, so the midi shares are re-calculated based on the sum without trolleys
midi_sum = buses_vshares[filter(lambda x: x != 'BusElecTrolley',bus_label)].sum(axis=1)     #sum of all except Trolleys

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

# 2) TRUCKS
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

### Then calculate the inflow & outflow for typical vehicles (vehicles with relevant sub types) as well (runtime appr. 1 min.)
bus_regl_in,     bus_regl_out_coh,     bus_regl_stock_coh     = inflow_outflow_typical_np(bus_regl_nr,    lifetime_tail(lifetimes_vehicles_mean, 'reg_bus'),  lifetime_tail(lifetimes_vehicles_stdev, 'reg_bus'),  'FoldedNormal', buses_regl_vshares)
bus_midi_in,     bus_midi_out_coh,     bus_midi_stock_coh     = inflow_outflow_typical_np(bus_midi_nr,    lifetime_tail(lifetimes_vehicles_mean, 'midi_bus'), lifetime_tail(lifetimes_vehicles_stdev, 'midi_bus'), 'FoldedNormal', buses_midi_vshares)
car_in,          car_out_coh,          car_stock_coh          = inflow_outflow_typical_np(car_total_nr,   lifetime_tail(lifetimes_vehicles_shape, 'car'),     lifetime_tail(lifetimes_vehicles_scale, 'car'),      'Weibull',      vehicleshare_cars)

trucks_HFT_in,   trucks_HFT_out_coh,   trucks_HFT_stock_coh   = inflow_outflow_typical_np(trucks_HFT_nr,  lifetime_tail(lifetimes_vehicles_mean, 'HFT'),     lifetime_tail(lifetimes_vehicles_stdev, 'HFT'),      'FoldedNormal',  trucks_HFT_vshares)
trucks_MFT_in,   trucks_MFT_out_coh,   trucks_MFT_stock_coh   = inflow_outflow_typical_np(trucks_MFT_nr,  lifetime_tail(lifetimes_vehicles_mean, 'MFT'),     lifetime_tail(lifetimes_vehicles_stdev, 'MFT'),      'FoldedNormal',  trucks_MFT_vshares)
trucks_LCV_in,   trucks_LCV_out_coh,   trucks_LCV_stock_coh   = inflow_outflow_typical_np(trucks_LCV_nr,  lifetime_tail(lifetimes_vehicles_mean, 'LCV'),     lifetime_tail(lifetimes_vehicles_stdev, 'LCV'),      'FoldedNormal',  trucks_MFT_vshares)  # Assumption: used MFT as a market-share for LCVs

#%% Intermediate export of inflow & outflow of vehicles (for IRP database) ###############

region_list = list(range(1,27))
last_years  = END_YEAR+1 - 1971
index = pd.MultiIndex.from_product([list(total_nr_of_ships.index), region_list], names = ['years','regions'])
total_nr_vehicles_in = pd.DataFrame(index=index, columns=columns_vehcile_output)
total_nr_vehicles_in['Buses']        = bus_regl_in.sum(0)[:,-last_years:].flatten(order='F') + bus_midi_in.sum(0)[:,-last_years:].flatten(order='F')  #flattening a numpy array in the expected order (year columns first)
total_nr_vehicles_in['Trains']       = rail_reg_in[-last_years:,:].flatten(order='C')  # for simple arrays (no vehicle types) the column order is reversed
total_nr_vehicles_in['HST']          = rail_hst_in[-last_years:,:].flatten(order='C')
total_nr_vehicles_in['Cars']         = car_in.sum(0)[:,-last_years:].flatten(order='F')
total_nr_vehicles_in['Planes']       = air_pas_in[-last_years:,:].flatten(order='C')
total_nr_vehicles_in['Bikes']        = bikes_in[-last_years:,:].flatten(order='C')
total_nr_vehicles_in['Trucks']       = trucks_HFT_in.sum(0)[:,-last_years:].flatten(order='F') + trucks_MFT_in.sum(0)[:,-last_years:].flatten(order='F') + trucks_LCV_in.sum(0)[:,-last_years:].flatten(order='F')
total_nr_vehicles_in['Cargo Trains'] = rail_freight_in[-last_years:,:].flatten(order='C')
total_nr_vehicles_in['Ships']        = ship_small_in[-last_years:,:].flatten(order='C') + ship_medium_in[-last_years:,:].flatten(order='C') + ship_large_in[-last_years:,:].flatten(order='C') + ship_vlarge_in[-last_years:,:].flatten(order='C')
total_nr_vehicles_in['Inland ships'] = inland_ship_in[-last_years:,:].flatten(order='C')  
total_nr_vehicles_in['Cargo Planes'] = air_freight_in[-last_years:,:].flatten(order='C')

total_nr_vehicles_in.to_csv('output\\' + FOLDER + '\\region_vehicle_in.csv', index=True) # regional nr of vehicles sold (annually)

total_nr_vehicles_out = pd.DataFrame(index=index, columns=columns_vehcile_output)
total_nr_vehicles_out['Buses']        = bus_regl_out_coh.sum(0).sum(-1)[:,-last_years:].flatten(order='F') + bus_midi_out_coh.sum(0).sum(-1)[:,-last_years:].flatten(order='F')  #flattening a numpy array in the expected order (year columns first)
total_nr_vehicles_out['Trains']       = rail_reg_out_coh.sum(-1)[:,-last_years:].flatten(order='F')  # for simple arrays (no vehicle types) the column order is reversed
total_nr_vehicles_out['HST']          = rail_hst_out_coh.sum(-1)[:,-last_years:].flatten(order='F')
total_nr_vehicles_out['Cars']         = car_out_coh.sum(0).sum(-1)[:,-last_years:].flatten(order='F')
total_nr_vehicles_out['Planes']       = air_pas_out_coh.sum(-1)[:,-last_years:].flatten(order='F')
total_nr_vehicles_out['Bikes']        = bikes_out_coh.sum(-1)[:,-last_years:].flatten(order='F')
total_nr_vehicles_out['Trucks']       = trucks_HFT_out_coh.sum(0).sum(-1)[:,-last_years:].flatten(order='F') + trucks_MFT_out_coh.sum(0).sum(-1)[:,-last_years:].flatten(order='F') + trucks_LCV_out_coh.sum(0).sum(-1)[:,-last_years:].flatten(order='F')
total_nr_vehicles_out['Cargo Trains'] = rail_freight_out_coh.sum(-1)[:,-last_years:].flatten(order='F')
total_nr_vehicles_out['Ships']        = ship_small_out_coh.sum(-1)[:,-last_years:].flatten(order='F') + ship_medium_out_coh.sum(-1)[:,-last_years:].flatten(order='F') + ship_large_out_coh.sum(-1)[:,-last_years:].flatten(order='F') + ship_vlarge_out_coh.sum(-1)[:,-last_years:].flatten(order='F')
total_nr_vehicles_out['Inland ships'] = inland_ship_out_coh.sum(-1)[:,-last_years:].flatten(order='F')
total_nr_vehicles_out['Cargo Planes'] = air_freight_out_coh.sum(-1)[:,-last_years:].flatten(order='F')

total_nr_vehicles_out.to_csv('output\\' + FOLDER + '\\region_vehicle_out.csv', index=True) # regional nr of vehicles sold (annually)


#%% ################### MATERIAL CALCULATIONS ##########################################

# Material calculations for vehicles with only 1 relevant sub-type
def nr_by_cohorts_to_materials_simple_np(inflow, outflow_cohort, stock_cohort, weight, composition):
   """
   for those vehicles with only 1 relevant sub-type, we calculate the material stocks & flows as: 
   Nr * weight (kg) * composition (%)
   INPUT:  -------------- 
   inflow         = numpy array (time, regions)      - number of vehicles by region over time
   outflow_cohort = numpy array (region, time, time) - number of vehicles in outlfow by region over time, by built year
   stock_cohort   = numpy array (region, time, time) - number of vehicles in stock by region over time, by built year
   OUTPUT: --------------
   pd_inflow_mat  = pandas dataframe (index: material, time; column: regions) - inflow of materials in kg of material, by region & time
   pd_outflow_mat = pandas dataframe (index: material, time; column: regions) - inflow of materials in kg of material, by region & time
   pd_stock_mat   = pandas dataframe (index: material, time; column: regions) - inflow of materials in kg of material, by region & time
   """
   inflow_mat  = np.zeros((len(composition.columns), len(inflow[0]), len(inflow)))
   outflow_mat = np.zeros((len(composition.columns), len(inflow[0]), len(inflow)))
   stock_mat   = np.zeros((len(composition.columns), len(inflow[0]), len(inflow)))
   
   for material in range(0, len(composition.columns)): 
      # before running, check if the material is at all relevant in the vehicle (save calculation time)
      if composition.iloc[:,material].sum() > 0.001:          
         for region in range(0,len(inflow[0])):
            composition_used = composition.iloc[:,material].values
            inflow_mat[material,region,:]  = (inflow[:,region] * weight) * composition_used
            outflow_mat[material,region,:] = np.multiply(np.multiply(outflow_cohort[region,:,:].T, weight), composition_used).T.sum(axis=1)
            stock_mat[material,region,:]   = np.multiply(np.multiply(stock_cohort[region,:,:], weight), composition_used).sum(axis=1)
      else:
         pass

   length_materials = len(composition.columns)
   length_time      = END_YEAR + 1 - (END_YEAR + 1 - len(inflow))

   index          = pd.MultiIndex.from_product([composition.columns, range(END_YEAR + 1 - len(inflow), END_YEAR + 1)], names=['time', 'type'])
   pd_inflow_mat  = pd.DataFrame(inflow_mat.transpose(0,2,1).reshape((length_materials * length_time), REGIONS),  index=index, columns=range(1,len(inflow[0]) + 1))
   pd_outflow_mat = pd.DataFrame(outflow_mat.transpose(0,2,1).reshape((length_materials * length_time), REGIONS), index=index, columns=range(1,len(inflow[0]) + 1))
   pd_stock_mat   = pd.DataFrame(stock_mat.transpose(0,2,1).reshape((length_materials * length_time), REGIONS),   index=index, columns=range(1,len(inflow[0]) + 1))
      
   return pd_inflow_mat, pd_outflow_mat, pd_stock_mat


# Material calculations for vehicles with only multiple sub-types
def nr_by_cohorts_to_materials_typical_np(inflow, outflow_cohort, stock_cohort, weight, composition):
   """
   for those vehicles with multiple sub-type, we calculate the material stocks & flows in the same way, so:
   Nr * weight (kg) * composition (%) - but using data on specific vehicle sub-types
   """
   inflow_mat  = np.zeros((len(inflow), len(composition.columns.levels[1]), len(inflow[0]), len(inflow[0][0])))
   outflow_mat = np.zeros((len(inflow), len(composition.columns.levels[1]), len(inflow[0]), len(inflow[0][0])))
   stock_mat   = np.zeros((len(inflow), len(composition.columns.levels[1]), len(inflow[0]), len(inflow[0][0])))
   
   composition.columns.names = ['type', 'material']
   
   # get two dictionaries to keep tarck of the order of materials & vtypes while using numpy
   vtype_list   = list(composition.columns.unique('type'))  # columns.unique() keeps the original order, which is important here (same for materials)
   vtype_count  = list(range(0,len(inflow))) 
   vtype_dict   = dict(zip(vtype_count, vtype_list))
   print(vtype_dict)
   mater_list   = list(composition.columns.unique('material'))
   mater_count  = list(range(0,len(composition.columns.levels[1]))) 
   mater_dict   = dict(zip(mater_count, mater_list))
   print(mater_dict)
   
   for vtype in range(0, len(inflow)): 
      # before running, check if the vehicle type is at all relevant in the vehicle (save calculation time)
      weight_used = weight.loc[:,vtype_dict[vtype]].values
      if stock_cohort[vtype].sum() > 0.001:  
         for material in range(0, len(mater_list)): 
            composition_used = composition.loc[:,idx[vtype_dict[vtype],mater_dict[material]]].values
            # before running, check if the material is at all relevant in the vehicle (save calculation time)
            if composition_used.sum() > 0.001:          
               for region in range(0, len(inflow[0])):
                  inflow_mat[vtype, material, region, :]  = (inflow[vtype, region, :] * weight_used) * composition_used
                  outflow_mat[vtype, material, region, :] = np.multiply(np.multiply(outflow_cohort[vtype, region, :, :].T, weight_used), composition_used).T.sum(axis=1)
                  stock_mat[vtype, material, region, :]   = np.multiply(np.multiply(stock_cohort[vtype, region, :, :], weight_used), composition_used).sum(axis=1)
   
            else:
               pass
      else:
         pass
   
   length_materials = len(composition.columns.levels[1])
   length_time      = END_YEAR + 1 - (END_YEAR + 1 - len(inflow[0][0]))
   
   #return as pandas dataframe, just once
   index          = pd.MultiIndex.from_product([mater_list, list(range((END_YEAR + 1 - len(inflow[0][0])), END_YEAR + 1))], names=['material', 'time'])
   columns        = pd.MultiIndex.from_product([vtype_list, range(1,REGIONS+1)],      names=['type', 'region'])
   pd_inflow_mat  = pd.DataFrame(inflow_mat.transpose(1,3,0,2).reshape((length_materials * length_time), (len(inflow) * REGIONS)), index=index, columns=columns)
   pd_outflow_mat = pd.DataFrame(outflow_mat.transpose(1,3,0,2).reshape((length_materials * length_time),(len(inflow) * REGIONS)), index=index, columns=columns)
   pd_stock_mat   = pd.DataFrame(stock_mat.transpose(1,3,0,2).reshape((length_materials * length_time),  (len(inflow) * REGIONS)), index=index, columns=columns)
         
   return pd_inflow_mat, pd_outflow_mat, pd_stock_mat


# capacity of boats is in tonnes, the weight - expressed as a fraction of the capacity - is calculated in in kgs here
weight_boats  = weight_frac_boats_yrs * cap_of_boats_yrs * 1000 

#%% ############################################# RUNNING THE DYNAMIC STOCK FUNCTIONS  (runtime ca. 10 sec)  ###############################################

# # run the simple material calculations on all vehicles                                                                                             # weight is passed as a series instead of a dataframe (pragmatic choice)
# air_pas_mat_in,      air_pas_mat_out,      air_pas_mat_stock        = nr_by_cohorts_to_materials_simple_np(air_pas_in,      air_pas_out_coh,      air_pas_stock_coh,      vehicle_weight_kg_air_pas["air_pas"].to_numpy(),             material_fractions_air_pas)
# rail_reg_mat_in,     rail_reg_mat_out,     rail_reg_mat_stock       = nr_by_cohorts_to_materials_simple_np(rail_reg_in,     rail_reg_out_coh,     rail_reg_stock_coh,     vehicle_weight_kg_rail_reg["rail_reg"].to_numpy(),           material_fractions_rail_reg)
# rail_hst_mat_in,     rail_hst_mat_out,     rail_hst_mat_stock       = nr_by_cohorts_to_materials_simple_np(rail_hst_in,     rail_hst_out_coh,     rail_hst_stock_coh,     vehicle_weight_kg_rail_hst["rail_hst"].to_numpy(),           material_fractions_rail_hst)
# bikes_mat_in,        bikes_mat_out,        bikes_mat_stock          = nr_by_cohorts_to_materials_simple_np(bikes_in,        bikes_out_coh,        bikes_stock_coh,        vehicle_weight_kg_bicycle["bicycle"].to_numpy(),             material_fractions_bicycle)

# air_freight_mat_in,  air_freight_mat_out,  air_freight_mat_stock    = nr_by_cohorts_to_materials_simple_np(air_freight_in,  air_freight_out_coh,  air_freight_stock_coh,  vehicle_weight_kg_air_frgt["air_freight"].to_numpy(),        material_fractions_air_frgt)
# rail_freight_mat_in, rail_freight_mat_out, rail_freight_mat_stock   = nr_by_cohorts_to_materials_simple_np(rail_freight_in, rail_freight_out_coh, rail_freight_stock_coh, vehicle_weight_kg_rail_frgt["rail_freight"].to_numpy(),      material_fractions_rail_frgt)
# inland_ship_mat_in,  inland_ship_mat_out,  inland_ship_mat_stock    = nr_by_cohorts_to_materials_simple_np(inland_ship_in,  inland_ship_out_coh,  inland_ship_stock_coh,  vehicle_weight_kg_inland_ship["inland_shipping"].to_numpy(), material_fractions_inland_ship)

# ship_small_mat_in,   ship_small_mat_out,   ship_small_mat_stock     = nr_by_cohorts_to_materials_simple_np(ship_small_in,   ship_small_out_coh,   ship_small_stock_coh,   weight_boats['Small'].to_numpy(),                            material_fractions_ship_small)
# ship_medium_mat_in,  ship_medium_mat_out,  ship_medium_mat_stock    = nr_by_cohorts_to_materials_simple_np(ship_medium_in,  ship_medium_out_coh,  ship_medium_stock_coh,  weight_boats['Medium'].to_numpy(),                           material_fractions_ship_medium)
# ship_large_mat_in,   ship_large_mat_out,   ship_large_mat_stock     = nr_by_cohorts_to_materials_simple_np(ship_large_in,   ship_large_out_coh,   ship_large_stock_coh,   weight_boats['Large'].to_numpy(),                            material_fractions_ship_large)
# ship_vlarge_mat_in,  ship_vlarge_mat_out,  ship_vlarge_mat_stock    = nr_by_cohorts_to_materials_simple_np(ship_vlarge_in,  ship_vlarge_out_coh,  ship_vlarge_stock_coh,  weight_boats['Very Large'].to_numpy(),                       material_fractions_ship_vlarge)


# # Calculate the weight of materials in the vehicles with sub-types: stock, inflow & outflow 
# bus_regl_mat_in,     bus_regl_mat_out,     bus_regl_mat_stock       = nr_by_cohorts_to_materials_typical_np(bus_regl_in,   bus_regl_out_coh,   bus_regl_stock_coh,   vehicle_weight_kg_bus,   material_fractions_bus_reg)
# bus_midi_mat_in,     bus_midi_mat_out,     bus_midi_mat_stock       = nr_by_cohorts_to_materials_typical_np(bus_midi_in,   bus_midi_out_coh,   bus_midi_stock_coh,   vehicle_weight_kg_midi,  material_fractions_bus_midi)
# car_total_mat_in,    car_total_mat_out,    car_total_mat_stock      = nr_by_cohorts_to_materials_typical_np(car_in,        car_out_coh,        car_stock_coh,        vehicle_weight_kg_car,   material_fractions_car)    

# trucks_HFT_mat_in,   trucks_HFT_mat_out,   trucks_HFT_mat_stock     = nr_by_cohorts_to_materials_typical_np(trucks_HFT_in, trucks_HFT_out_coh, trucks_HFT_stock_coh, vehicle_weight_kg_HFT,   material_fractions_truck_HFT)
# trucks_MFT_mat_in,   trucks_MFT_mat_out,   trucks_MFT_mat_stock     = nr_by_cohorts_to_materials_typical_np(trucks_MFT_in, trucks_MFT_out_coh, trucks_MFT_stock_coh, vehicle_weight_kg_MFT,   material_fractions_truck_MFT)
# trucks_LCV_mat_in,   trucks_LCV_mat_out,   trucks_LCV_mat_stock     = nr_by_cohorts_to_materials_typical_np(trucks_LCV_in, trucks_LCV_out_coh, trucks_LCV_stock_coh, vehicle_weight_kg_LCV,   material_fractions_truck_LCV)


# # Calculate the materials in batteries (in typical vehicles only) 
# # For batteries this is a 2-step process, first (1) we pre-calculate the average material composition (of the batteries at inflow), based on a globally changing battery share & the changeing battery-specific material composition, both derived from our paper on (a.o.) storage in the electricity system.
# # Then (2) we run the same function to derive the materials in vehicle batteries (based on changeing weight, composition & battery share)
# # In doing so, we are no longer able to know the battery share per vehicle sub-type

# battery_material_composition    = pd.DataFrame(index=pd.MultiIndex.from_product([list(battery_weights_full.columns.levels[0]), battery_materials_full.index]), columns=pd.MultiIndex.from_product([car_types + ['Trolley'], battery_materials_full.columns.levels[0]]))
# battery_weight_total_in         = pd.DataFrame(0, index=pd.MultiIndex.from_product([list(battery_weights_full.columns.levels[0]), battery_materials_full.index]), columns=battery_weights_full.columns.levels[1])
# battery_weight_total_stock      = pd.DataFrame(0, index=pd.MultiIndex.from_product([list(battery_weights_full.columns.levels[0]), battery_materials_full.index]), columns=battery_weights_full.columns.levels[1])

# battery_weight_regional_stock   = pd.DataFrame(0, index=battery_materials_full.index, columns=list(range(1,REGIONS+1)))
# battery_weight_regional_in      = pd.DataFrame(0, index=battery_materials_full.index, columns=list(range(1,REGIONS+1)))
# battery_weight_regional_out     = pd.DataFrame(0, index=battery_materials_full.index, columns=list(range(1,REGIONS+1)))
 
# # for now, 1 global battery market is assumed (so no difference between battery types for different vehicle sub-types), so the composition is duplicated over all vtypes
# for vehicle in list(battery_weights_full.columns.levels[0]):
#    for vtype in list(battery_weights_full.columns.levels[1]):
#       for material in list(battery_materials_full.columns.levels[0]):
#          battery_material_composition.loc[idx[vehicle,:],idx[vtype,material]] = battery_shares_full.mul(battery_materials_full.loc[:,idx[material,:]].droplevel(0,axis=1)).sum(axis=1).values

# last_years = -(END_YEAR +1 - START_YEAR)
# # Battery material calculations (these are made from 1971 onwwards, hence the selection in years) 
# bus_regl_bat_in,     bus_regl_bat_out,     bus_regl_bat_stock       = nr_by_cohorts_to_materials_typical_np(bus_regl_in[:,:,last_years:],  bus_regl_out_coh[:,:,last_years:, last_years:], bus_regl_stock_coh[:,:,last_years:, last_years:], battery_weights_full['reg_bus'],   battery_material_composition.loc[idx['reg_bus',:],:].droplevel(0))
# bus_midi_bat_in,     bus_midi_bat_out,     bus_midi_bat_stock       = nr_by_cohorts_to_materials_typical_np(bus_midi_in[:,:,last_years:],  bus_midi_out_coh[:,:,last_years:, last_years:], bus_midi_stock_coh[:,:,last_years:, last_years:], battery_weights_full['midi_bus'],  battery_material_composition.loc[idx['midi_bus',:],:].droplevel(0))
# car_total_bat_in,    car_total_bat_out,    car_total_bat_stock      = nr_by_cohorts_to_materials_typical_np(car_in[:,:,last_years:],       car_out_coh[:,:,last_years:, last_years:],      car_stock_coh[:,:,last_years:, last_years:],      battery_weights_full['car'],       battery_material_composition.loc[idx['car',:],idx[car_types,:]].droplevel(0))    #mind that cars don't have Trolleys, hence the additional selection

# trucks_HFT_bat_in,   trucks_HFT_bat_out,   trucks_HFT_bat_stock     = nr_by_cohorts_to_materials_typical_np(trucks_HFT_in[:,:,last_years:], trucks_HFT_out_coh[:,:,last_years:, last_years:], trucks_HFT_stock_coh[:,:,last_years:, last_years:], battery_weights_full['HFT'],  battery_material_composition.loc[idx['HFT',:],:].droplevel(0))
# trucks_MFT_bat_in,   trucks_MFT_bat_out,   trucks_MFT_bat_stock     = nr_by_cohorts_to_materials_typical_np(trucks_MFT_in[:,:,last_years:], trucks_MFT_out_coh[:,:,last_years:, last_years:], trucks_MFT_stock_coh[:,:,last_years:, last_years:], battery_weights_full['MFT'],  battery_material_composition.loc[idx['MFT',:],:].droplevel(0))
# trucks_LCV_bat_in,   trucks_LCV_bat_out,   trucks_LCV_bat_stock     = nr_by_cohorts_to_materials_typical_np(trucks_LCV_in[:,:,last_years:], trucks_LCV_out_coh[:,:,last_years:, last_years:], trucks_LCV_stock_coh[:,:,last_years:, last_years:], battery_weights_full['LCV'],  battery_material_composition.loc[idx['LCV',:],:].droplevel(0))

# # Sum the weight of the accounted materials (! so not total weight) in batteries by vehicle & vehicle type, output for figures
# for vtype in list(battery_weights_full.columns.levels[1]):
#       battery_weight_total_in.loc[idx['reg_bus',:],vtype]     = bus_regl_bat_in.loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values
#       battery_weight_total_in.loc[idx['midi_bus',:],vtype]    = bus_midi_bat_in.loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values
#       battery_weight_total_in.loc[idx['LCV',:],vtype]         = trucks_LCV_bat_in.loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values
#       battery_weight_total_in.loc[idx['MFT',:],vtype]         = trucks_MFT_bat_in.loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values
#       battery_weight_total_in.loc[idx['HFT',:],vtype]         = trucks_HFT_bat_in.loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values
      
#       battery_weight_total_stock.loc[idx['reg_bus',:],vtype]  = bus_regl_bat_stock.loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values
#       battery_weight_total_stock.loc[idx['midi_bus',:],vtype] = bus_midi_bat_stock.loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values
#       battery_weight_total_stock.loc[idx['LCV',:],vtype]      = trucks_LCV_bat_stock.loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values
#       battery_weight_total_stock.loc[idx['MFT',:],vtype]      = trucks_MFT_bat_stock.loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values
#       battery_weight_total_stock.loc[idx['HFT',:],vtype]      = trucks_HFT_bat_stock.loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values   
      
#       if vtype == 'Trolley':
#          pass 
#       else:
#          battery_weight_total_in.loc[idx['car',:],vtype]      = car_total_bat_in.loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values
#          battery_weight_total_stock.loc[idx['car',:],vtype]   = car_total_bat_stock.loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values

# battery_weight_total_in.to_csv('output\\' + FOLDER + '\\battery_weight_kg_in.csv', index=True)       # in kg
# battery_weight_total_stock.to_csv('output\\' + FOLDER + '\\battery_weight_kg_stock.csv', index=True) # in kg

# # Regional battery weight (only the accounted materials), used in graph later on
# battery_weight_regional_stock = bus_regl_bat_stock.sum(axis=0,level=1).sum(axis=1,level=1) + bus_midi_bat_stock.sum(axis=0,level=1).sum(axis=1,level=1) + trucks_LCV_bat_stock.sum(axis=0,level=1).sum(axis=1,level=1) + trucks_MFT_bat_stock.sum(axis=0,level=1).sum(axis=1,level=1) + trucks_HFT_bat_stock.sum(axis=0,level=1).sum(axis=1,level=1) + car_total_bat_stock.sum(axis=0,level=1).sum(axis=1,level=1)
# battery_weight_regional_in    = bus_regl_bat_in.sum(axis=0,level=1).sum(axis=1,level=1)    + bus_midi_bat_in.sum(axis=0,level=1).sum(axis=1,level=1)    + trucks_LCV_bat_in.sum(axis=0,level=1).sum(axis=1,level=1)    + trucks_MFT_bat_in.sum(axis=0,level=1).sum(axis=1,level=1)    + trucks_HFT_bat_in.sum(axis=0,level=1).sum(axis=1,level=1)    + car_total_bat_in.sum(axis=0,level=1).sum(axis=1,level=1)     
# battery_weight_regional_out   = bus_regl_bat_out.sum(axis=0,level=1).sum(axis=1,level=1)   + bus_midi_bat_out.sum(axis=0,level=1).sum(axis=1,level=1)   + trucks_LCV_bat_out.sum(axis=0,level=1).sum(axis=1,level=1)   + trucks_MFT_bat_out.sum(axis=0,level=1).sum(axis=1,level=1)   + trucks_HFT_bat_out.sum(axis=0,level=1).sum(axis=1,level=1)   + car_total_bat_out.sum(axis=0,level=1).sum(axis=1,level=1)                                      

# #%% ################################### Organise data for output ###########################################

# year_select = list(range(START_YEAR, END_YEAR + 1))

# # define 6 dataframes on materials in  stock, inflow & outflow X passenger vs. freight vehicles
# index   = pd.MultiIndex.from_product([year_select, list(range(1,REGIONS+1)), ['vehicle','battery'], labels_materials], names=['year', 'region', 'part', 'materials'])
# vehicle_materials_stock_passenger   = pd.DataFrame(index=index, columns=labels_pas)
# vehicle_materials_stock_freight     = pd.DataFrame(index=index, columns=labels_fre)
# vehicle_materials_inflow_passenger  = pd.DataFrame(index=index, columns=labels_pas)
# vehicle_materials_inflow_freight    = pd.DataFrame(index=index, columns=labels_fre)
# vehicle_materials_outflow_passenger = pd.DataFrame(index=index, columns=labels_pas)
# vehicle_materials_outflow_freight   = pd.DataFrame(index=index, columns=labels_fre)

# for material in labels_materials:

#    ############## STARTING WITH SIMPLE VEHICLES ###########################
   
#    # passenger stock, vehicles (in kg)
#    vehicle_materials_stock_passenger.loc[idx[:,:,'vehicle', material],'bicycle']       = bikes_mat_stock.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_stock_passenger.loc[idx[:,:,'vehicle', material],'rail_reg']      = rail_reg_mat_stock.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_stock_passenger.loc[idx[:,:,'vehicle', material],'rail_hst']      = rail_hst_mat_stock.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_stock_passenger.loc[idx[:,:,'vehicle', material],'air_pas']       = air_pas_mat_stock.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values

#    # freight stock (in kg)
#    vehicle_materials_stock_freight.loc[idx[:,:,'vehicle', material],'inland_shipping']     = inland_ship_mat_stock.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_stock_freight.loc[idx[:,:,'vehicle', material],'rail_freight']        = rail_freight_mat_stock.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_stock_freight.loc[idx[:,:,'vehicle', material],'air_freight']         = air_freight_mat_stock.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_stock_freight.loc[idx[:,:,'vehicle', material],'sea_shipping_small']  = ship_small_mat_stock.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_stock_freight.loc[idx[:,:,'vehicle', material],'sea_shipping_med']    = ship_medium_mat_stock.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_stock_freight.loc[idx[:,:,'vehicle', material],'sea_shipping_large']  = ship_large_mat_stock.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_stock_freight.loc[idx[:,:,'vehicle', material],'sea_shipping_vl']     = ship_vlarge_mat_stock.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values

#    # passenger inflow (in kg)
#    vehicle_materials_inflow_passenger.loc[idx[:,:,'vehicle', material],'bicycle']    = bikes_mat_in.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_inflow_passenger.loc[idx[:,:,'vehicle', material],'rail_reg']   = rail_reg_mat_in.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_inflow_passenger.loc[idx[:,:,'vehicle', material],'rail_hst']   = rail_hst_mat_in.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_inflow_passenger.loc[idx[:,:,'vehicle', material],'air_pas']    = air_pas_mat_in.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values

#    # freight inflow (in kg)
#    vehicle_materials_inflow_freight.loc[idx[:,:,'vehicle', material],'inland_shipping']    = inland_ship_mat_in.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_inflow_freight.loc[idx[:,:,'vehicle', material],'rail_freight']       = rail_freight_mat_in.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_inflow_freight.loc[idx[:,:,'vehicle', material],'air_freight']        = air_freight_mat_in.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_inflow_freight.loc[idx[:,:,'vehicle', material],'sea_shipping_small'] = ship_small_mat_in.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_inflow_freight.loc[idx[:,:,'vehicle', material],'sea_shipping_med']   = ship_medium_mat_in.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_inflow_freight.loc[idx[:,:,'vehicle', material],'sea_shipping_large'] = ship_large_mat_in.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_inflow_freight.loc[idx[:,:,'vehicle', material],'sea_shipping_vl']    = ship_vlarge_mat_in.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values

#    # passenger outflow (in kg)
#    vehicle_materials_outflow_passenger.loc[idx[:,:,'vehicle', material],'bicycle']   = bikes_mat_out.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_outflow_passenger.loc[idx[:,:,'vehicle', material],'rail_reg']  = rail_reg_mat_out.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_outflow_passenger.loc[idx[:,:,'vehicle', material],'rail_hst']  = rail_hst_mat_out.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_outflow_passenger.loc[idx[:,:,'vehicle', material],'air_pas']   = air_pas_mat_out.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values

#    # freight outflow (in kg)
#    vehicle_materials_outflow_freight.loc[idx[:,:,'vehicle', material],'inland_shipping']     = inland_ship_mat_out.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_outflow_freight.loc[idx[:,:,'vehicle', material],'rail_freight']        = rail_freight_mat_out.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_outflow_freight.loc[idx[:,:,'vehicle', material],'air_freight']         = air_freight_mat_out.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_outflow_freight.loc[idx[:,:,'vehicle', material],'sea_shipping_small']  = ship_small_mat_out.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_outflow_freight.loc[idx[:,:,'vehicle', material],'sea_shipping_med']    = ship_medium_mat_out.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_outflow_freight.loc[idx[:,:,'vehicle', material],'sea_shipping_large']  = ship_large_mat_out.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_outflow_freight.loc[idx[:,:,'vehicle', material],'sea_shipping_vl']     = ship_vlarge_mat_out.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values

#    ############ CONTINUEING WITH TYPICAL VEHICLES (MATERIALS IN TRUCKS & BUSSES ARE SUMMED, FOR CARS DETAIL BY TYPE IS MAINTAINED) ##########################
   
#    part = 'vehicle'
   
#    # passenger stock, vehicles (in kg)
#    vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'midi_bus']    = bus_midi_mat_stock.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values
#    vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'reg_bus']     = bus_regl_mat_stock.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values
#    vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'ICE']         = car_total_mat_stock.loc[idx[material,year_select],idx['ICE',:]].stack().reorder_levels([1,2,0]).values 
#    vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'HEV']         = car_total_mat_stock.loc[idx[material,year_select],idx['HEV',:]].stack().reorder_levels([1,2,0]).values   
#    vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'PHEV']        = car_total_mat_stock.loc[idx[material,year_select],idx['PHEV',:]].stack().reorder_levels([1,2,0]).values  
#    vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'BEV']         = car_total_mat_stock.loc[idx[material,year_select],idx['BEV',:]].stack().reorder_levels([1,2,0]).values  
#    vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'FCV']         = car_total_mat_stock.loc[idx[material,year_select],idx['FCV',:]].stack().reorder_levels([1,2,0]).values       
   
#    # freight stock (in kg)
#    vehicle_materials_stock_freight.loc[idx[:,:, part, material],'LCV']           = trucks_LCV_mat_stock.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values   
#    vehicle_materials_stock_freight.loc[idx[:,:, part, material],'MFT']           = trucks_MFT_mat_stock.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values  
#    vehicle_materials_stock_freight.loc[idx[:,:, part, material],'HFT']           = trucks_HFT_mat_stock.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values 

#    # passenger inflow (in kg)
#    vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'midi_bus']   = bus_midi_mat_in.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values
#    vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'reg_bus']    = bus_regl_mat_in.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values
#    vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'ICE']        = car_total_mat_in.loc[idx[material,year_select],idx['ICE',:]].stack().reorder_levels([1,2,0]).values 
#    vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'HEV']        = car_total_mat_in.loc[idx[material,year_select],idx['HEV',:]].stack().reorder_levels([1,2,0]).values 
#    vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'PHEV']       = car_total_mat_in.loc[idx[material,year_select],idx['PHEV',:]].stack().reorder_levels([1,2,0]).values 
#    vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'BEV']        = car_total_mat_in.loc[idx[material,year_select],idx['BEV',:]].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'FCV']        = car_total_mat_in.loc[idx[material,year_select],idx['FCV',:]].stack().reorder_levels([1,2,0]).values        
 
#    # freight inflow (in kg)
#    vehicle_materials_inflow_freight.loc[idx[:,:, part, material],'LCV']          = trucks_LCV_mat_in.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values   
#    vehicle_materials_inflow_freight.loc[idx[:,:, part, material],'MFT']          = trucks_MFT_mat_in.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values  
#    vehicle_materials_inflow_freight.loc[idx[:,:, part, material],'HFT']          = trucks_HFT_mat_in.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values    

#    # passenger outflow (in kg)
#    vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'midi_bus']  = bus_midi_mat_out.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values
#    vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'reg_bus']   = bus_regl_mat_out.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values 
#    vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'ICE']       = car_total_mat_out.loc[idx[material,year_select],idx['ICE',:]].stack().reorder_levels([1,2,0]).values  
#    vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'HEV']       = car_total_mat_out.loc[idx[material,year_select],idx['HEV',:]].stack().reorder_levels([1,2,0]).values 
#    vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'PHEV']      = car_total_mat_out.loc[idx[material,year_select],idx['PHEV',:]].stack().reorder_levels([1,2,0]).values 
#    vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'BEV']       = car_total_mat_out.loc[idx[material,year_select],idx['BEV',:]].stack().reorder_levels([1,2,0]).values 
#    vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'FCV']       = car_total_mat_out.loc[idx[material,year_select],idx['FCV',:]].stack().reorder_levels([1,2,0]).values       
   
#    # freight outflow (in kg)
#    vehicle_materials_outflow_freight.loc[idx[:,:, part, material],'LCV'] = trucks_LCV_mat_out.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values  
#    vehicle_materials_outflow_freight.loc[idx[:,:, part, material],'MFT']= trucks_MFT_mat_out.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values 
#    vehicle_materials_outflow_freight.loc[idx[:,:, part, material],'HFT'] = trucks_HFT_mat_out.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values 

#    ############ CONTINUEING WITH BATTERIES (MATERALS IN TRUCKS & BUSSES ARE SUMMED, FOR CARS DETAIL BY TYPE IS MAINTAINED) ##########################

#    part = 'battery'
   
#    # passenger stock, vehicles (in kg)
#    vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'midi_bus']    = bus_midi_bat_stock.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values
#    vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'reg_bus']     = bus_regl_bat_stock.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values
#    vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'ICE']         = car_total_bat_stock.loc[idx[material,year_select],idx['ICE',:]].stack().reorder_levels([1,2,0]).values 
#    vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'HEV']         = car_total_bat_stock.loc[idx[material,year_select],idx['HEV',:]].stack().reorder_levels([1,2,0]).values   
#    vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'PHEV']        = car_total_bat_stock.loc[idx[material,year_select],idx['PHEV',:]].stack().reorder_levels([1,2,0]).values  
#    vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'BEV']         = car_total_bat_stock.loc[idx[material,year_select],idx['BEV',:]].stack().reorder_levels([1,2,0]).values  
#    vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'FCV']         = car_total_bat_stock.loc[idx[material,year_select],idx['FCV',:]].stack().reorder_levels([1,2,0]).values       
   
#    # freight stock (in kg)
#    vehicle_materials_stock_freight.loc[idx[:,:, part, material],'LCV']           = trucks_LCV_bat_stock.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values   
#    vehicle_materials_stock_freight.loc[idx[:,:, part, material],'MFT']           = trucks_MFT_bat_stock.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values  
#    vehicle_materials_stock_freight.loc[idx[:,:, part, material],'HFT']           = trucks_HFT_bat_stock.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values 

#    # passenger inflow (in kg)
#    vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'midi_bus']   = bus_midi_bat_in.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values
#    vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'reg_bus']    = bus_regl_bat_in.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values
#    vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'ICE']        = car_total_bat_in.loc[idx[material,year_select],idx['ICE',:]].stack().reorder_levels([1,2,0]).values 
#    vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'HEV']        = car_total_bat_in.loc[idx[material,year_select],idx['HEV',:]].stack().reorder_levels([1,2,0]).values 
#    vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'PHEV']       = car_total_bat_in.loc[idx[material,year_select],idx['PHEV',:]].stack().reorder_levels([1,2,0]).values 
#    vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'BEV']        = car_total_bat_in.loc[idx[material,year_select],idx['BEV',:]].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'FCV']        = car_total_bat_in.loc[idx[material,year_select],idx['FCV',:]].stack().reorder_levels([1,2,0]).values        
 
#    # freight inflow (in kg)
#    vehicle_materials_inflow_freight.loc[idx[:,:, part, material],'LCV']          = trucks_LCV_bat_in.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values   
#    vehicle_materials_inflow_freight.loc[idx[:,:, part, material],'MFT']          = trucks_MFT_bat_in.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values  
#    vehicle_materials_inflow_freight.loc[idx[:,:, part, material],'HFT']          = trucks_HFT_bat_in.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values    

#    # passenger outflow (in kg)
#    vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'midi_bus']  = bus_midi_bat_out.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values
#    vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'reg_bus']   = bus_regl_bat_out.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values 
#    vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'ICE']       = car_total_bat_out.loc[idx[material,year_select],idx['ICE',:]].stack().reorder_levels([1,2,0]).values  
#    vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'HEV']       = car_total_bat_out.loc[idx[material,year_select],idx['HEV',:]].stack().reorder_levels([1,2,0]).values 
#    vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'PHEV']      = car_total_bat_out.loc[idx[material,year_select],idx['PHEV',:]].stack().reorder_levels([1,2,0]).values 
#    vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'BEV']       = car_total_bat_out.loc[idx[material,year_select],idx['BEV',:]].stack().reorder_levels([1,2,0]).values 
#    vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'FCV']       = car_total_bat_out.loc[idx[material,year_select],idx['FCV',:]].stack().reorder_levels([1,2,0]).values       
   
#    # freight outflow (in kg)
#    vehicle_materials_outflow_freight.loc[idx[:,:, part, material],'LCV']         = trucks_LCV_bat_out.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values  
#    vehicle_materials_outflow_freight.loc[idx[:,:, part, material],'MFT']         = trucks_MFT_bat_out.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values 
#    vehicle_materials_outflow_freight.loc[idx[:,:, part, material],'HFT']         = trucks_HFT_bat_out.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values 


# #%% combine dataframes for output

# # add flow descriptor to the multi-index & fill na values with 0 (the for-loop above didn't cover the battery materials in vhicles without batteries, so these are set to 0 now)
# vehicle_materials_stock_passenger  = pd.concat([vehicle_materials_stock_passenger.fillna(0)],    keys=['stock'],    names=['flow'])   
# vehicle_materials_stock_freight    = pd.concat([vehicle_materials_stock_freight.fillna(0)],      keys=['stock'],    names=['flow'])      
# vehicle_materials_inflow_passenger = pd.concat([vehicle_materials_inflow_passenger.fillna(0)],   keys=['inflow'],   names=['flow'])     
# vehicle_materials_inflow_freight   = pd.concat([vehicle_materials_inflow_freight.fillna(0)],     keys=['inflow'],   names=['flow'])     
# vehicle_materials_outflow_passenger = pd.concat([vehicle_materials_outflow_passenger.fillna(0)], keys=['outflow'],  names=['flow'])      
# vehicle_materials_outflow_freight   = pd.concat([vehicle_materials_outflow_freight.fillna(0)],   keys=['outflow'],  names=['flow']) 

# # concatenate stock, inflow & outflow into 1 dataframe (1 for passenger & 1 for freight)
# vehicle_materials_passenger = pd.concat([vehicle_materials_stock_passenger, vehicle_materials_inflow_passenger,  vehicle_materials_outflow_passenger]) 
# vehicle_materials_freight   = pd.concat([vehicle_materials_stock_freight,   vehicle_materials_inflow_freight,    vehicle_materials_outflow_freight])

# # add category descriptors to the multi-index (pass vs. freight)
# vehicle_materials_passenger = pd.concat([vehicle_materials_passenger], keys=['passenger'], names=['category']) 
# vehicle_materials_freight   = pd.concat([vehicle_materials_freight],    keys=['freight'], names=['category'])  

# vehicle_materials_passenger.columns.name = vehicle_materials_freight.columns.name = 'elements'

# # concatenate into 1 single dataframe & add the 'vehicle' descriptor
# vehicle_materials = pd.concat([vehicle_materials_passenger.stack().unstack(level=2), vehicle_materials_freight.stack().unstack(level=2)])
# vehicle_materials = pd.concat([vehicle_materials], keys=['vehicles'], names=['sector'])

# # re-order multi-index to the desired output & sum to global total
# vehicle_materials_out = vehicle_materials.reorder_levels([3, 2, 0, 1, 6, 5, 4]) / 1000000  # div by 1*10^6 to translate from kg to kt
# vehicle_materials_out.reset_index(inplace=True)                                         # return to columns

# vehicle_materials_out.to_csv('output\\' + FOLDER + '\\vehicle_materials_kt.csv', index=False) # in kt

# time.sleep(1)

# # get the end time
# et = time.time()

# # get the execution time
# elapsed_time = et - st
# print('Execution time:', elapsed_time, 'seconds')

# #%% PLots for Freight & Passenger pkm/tkm demand (2000-2100), global 
# run the simple material calculations on all vehicles                                                                                             # weight is passed as a series instead of a dataframe (pragmatic choice)
# air_pas_mat_in,      air_pas_mat_out,      air_pas_mat_stock        = nr_by_cohorts_to_materials_simple_np(air_pas_in,      air_pas_out_coh,      air_pas_stock_coh,      vehicle_weight_kg_air_pas["air_pas"].to_numpy(),             material_fractions_air_pas)
# rail_reg_mat_in,     rail_reg_mat_out,     rail_reg_mat_stock       = nr_by_cohorts_to_materials_simple_np(rail_reg_in,     rail_reg_out_coh,     rail_reg_stock_coh,     vehicle_weight_kg_rail_reg["rail_reg"].to_numpy(),           material_fractions_rail_reg)
# rail_hst_mat_in,     rail_hst_mat_out,     rail_hst_mat_stock       = nr_by_cohorts_to_materials_simple_np(rail_hst_in,     rail_hst_out_coh,     rail_hst_stock_coh,     vehicle_weight_kg_rail_hst["rail_hst"].to_numpy(),           material_fractions_rail_hst)
# bikes_mat_in,        bikes_mat_out,        bikes_mat_stock          = nr_by_cohorts_to_materials_simple_np(bikes_in,        bikes_out_coh,        bikes_stock_coh,        vehicle_weight_kg_bicycle["bicycle"].to_numpy(),             material_fractions_bicycle)

# air_freight_mat_in,  air_freight_mat_out,  air_freight_mat_stock    = nr_by_cohorts_to_materials_simple_np(air_freight_in,  air_freight_out_coh,  air_freight_stock_coh,  vehicle_weight_kg_air_frgt["air_freight"].to_numpy(),        material_fractions_air_frgt)
# rail_freight_mat_in, rail_freight_mat_out, rail_freight_mat_stock   = nr_by_cohorts_to_materials_simple_np(rail_freight_in, rail_freight_out_coh, rail_freight_stock_coh, vehicle_weight_kg_rail_frgt["rail_freight"].to_numpy(),      material_fractions_rail_frgt)
# inland_ship_mat_in,  inland_ship_mat_out,  inland_ship_mat_stock    = nr_by_cohorts_to_materials_simple_np(inland_ship_in,  inland_ship_out_coh,  inland_ship_stock_coh,  vehicle_weight_kg_inland_ship["inland_shipping"].to_numpy(), material_fractions_inland_ship)

# ship_small_mat_in,   ship_small_mat_out,   ship_small_mat_stock     = nr_by_cohorts_to_materials_simple_np(ship_small_in,   ship_small_out_coh,   ship_small_stock_coh,   weight_boats['Small'].to_numpy(),                            material_fractions_ship_small)
# ship_medium_mat_in,  ship_medium_mat_out,  ship_medium_mat_stock    = nr_by_cohorts_to_materials_simple_np(ship_medium_in,  ship_medium_out_coh,  ship_medium_stock_coh,  weight_boats['Medium'].to_numpy(),                           material_fractions_ship_medium)
# ship_large_mat_in,   ship_large_mat_out,   ship_large_mat_stock     = nr_by_cohorts_to_materials_simple_np(ship_large_in,   ship_large_out_coh,   ship_large_stock_coh,   weight_boats['Large'].to_numpy(),                            material_fractions_ship_large)
# ship_vlarge_mat_in,  ship_vlarge_mat_out,  ship_vlarge_mat_stock    = nr_by_cohorts_to_materials_simple_np(ship_vlarge_in,  ship_vlarge_out_coh,  ship_vlarge_stock_coh,  weight_boats['Very Large'].to_numpy(),                       material_fractions_ship_vlarge)


# # Calculate the weight of materials in the vehicles with sub-types: stock, inflow & outflow 
# bus_regl_mat_in,     bus_regl_mat_out,     bus_regl_mat_stock       = nr_by_cohorts_to_materials_typical_np(bus_regl_in,   bus_regl_out_coh,   bus_regl_stock_coh,   vehicle_weight_kg_bus,   material_fractions_bus_reg)
# bus_midi_mat_in,     bus_midi_mat_out,     bus_midi_mat_stock       = nr_by_cohorts_to_materials_typical_np(bus_midi_in,   bus_midi_out_coh,   bus_midi_stock_coh,   vehicle_weight_kg_midi,  material_fractions_bus_midi)
# car_total_mat_in,    car_total_mat_out,    car_total_mat_stock      = nr_by_cohorts_to_materials_typical_np(car_in,        car_out_coh,        car_stock_coh,        vehicle_weight_kg_car,   material_fractions_car)    

# trucks_HFT_mat_in,   trucks_HFT_mat_out,   trucks_HFT_mat_stock     = nr_by_cohorts_to_materials_typical_np(trucks_HFT_in, trucks_HFT_out_coh, trucks_HFT_stock_coh, vehicle_weight_kg_HFT,   material_fractions_truck_HFT)
# trucks_MFT_mat_in,   trucks_MFT_mat_out,   trucks_MFT_mat_stock     = nr_by_cohorts_to_materials_typical_np(trucks_MFT_in, trucks_MFT_out_coh, trucks_MFT_stock_coh, vehicle_weight_kg_MFT,   material_fractions_truck_MFT)
# trucks_LCV_mat_in,   trucks_LCV_mat_out,   trucks_LCV_mat_stock     = nr_by_cohorts_to_materials_typical_np(trucks_LCV_in, trucks_LCV_out_coh, trucks_LCV_stock_coh, vehicle_weight_kg_LCV,   material_fractions_truck_LCV)


# # Calculate the materials in batteries (in typical vehicles only) 
# # For batteries this is a 2-step process, first (1) we pre-calculate the average material composition (of the batteries at inflow), based on a globally changing battery share & the changeing battery-specific material composition, both derived from our paper on (a.o.) storage in the electricity system.
# # Then (2) we run the same function to derive the materials in vehicle batteries (based on changeing weight, composition & battery share)
# # In doing so, we are no longer able to know the battery share per vehicle sub-type

# battery_material_composition    = pd.DataFrame(index=pd.MultiIndex.from_product([list(battery_weights_full.columns.levels[0]), battery_materials_full.index]), columns=pd.MultiIndex.from_product([car_types + ['Trolley'], battery_materials_full.columns.levels[0]]))
# battery_weight_total_in         = pd.DataFrame(0, index=pd.MultiIndex.from_product([list(battery_weights_full.columns.levels[0]), battery_materials_full.index]), columns=battery_weights_full.columns.levels[1])
# battery_weight_total_stock      = pd.DataFrame(0, index=pd.MultiIndex.from_product([list(battery_weights_full.columns.levels[0]), battery_materials_full.index]), columns=battery_weights_full.columns.levels[1])

# battery_weight_regional_stock   = pd.DataFrame(0, index=battery_materials_full.index, columns=list(range(1,REGIONS+1)))
# battery_weight_regional_in      = pd.DataFrame(0, index=battery_materials_full.index, columns=list(range(1,REGIONS+1)))
# battery_weight_regional_out     = pd.DataFrame(0, index=battery_materials_full.index, columns=list(range(1,REGIONS+1)))
 
# # for now, 1 global battery market is assumed (so no difference between battery types for different vehicle sub-types), so the composition is duplicated over all vtypes
# for vehicle in list(battery_weights_full.columns.levels[0]):
#    for vtype in list(battery_weights_full.columns.levels[1]):
#       for material in list(battery_materials_full.columns.levels[0]):
#          battery_material_composition.loc[idx[vehicle,:],idx[vtype,material]] = battery_shares_full.mul(battery_materials_full.loc[:,idx[material,:]].droplevel(0,axis=1)).sum(axis=1).values

# last_years = -(END_YEAR +1 - START_YEAR)
# # Battery material calculations (these are made from 1971 onwwards, hence the selection in years) 
# bus_regl_bat_in,     bus_regl_bat_out,     bus_regl_bat_stock       = nr_by_cohorts_to_materials_typical_np(bus_regl_in[:,:,last_years:],  bus_regl_out_coh[:,:,last_years:, last_years:], bus_regl_stock_coh[:,:,last_years:, last_years:], battery_weights_full['reg_bus'],   battery_material_composition.loc[idx['reg_bus',:],:].droplevel(0))
# bus_midi_bat_in,     bus_midi_bat_out,     bus_midi_bat_stock       = nr_by_cohorts_to_materials_typical_np(bus_midi_in[:,:,last_years:],  bus_midi_out_coh[:,:,last_years:, last_years:], bus_midi_stock_coh[:,:,last_years:, last_years:], battery_weights_full['midi_bus'],  battery_material_composition.loc[idx['midi_bus',:],:].droplevel(0))
# car_total_bat_in,    car_total_bat_out,    car_total_bat_stock      = nr_by_cohorts_to_materials_typical_np(car_in[:,:,last_years:],       car_out_coh[:,:,last_years:, last_years:],      car_stock_coh[:,:,last_years:, last_years:],      battery_weights_full['car'],       battery_material_composition.loc[idx['car',:],idx[car_types,:]].droplevel(0))    #mind that cars don't have Trolleys, hence the additional selection

# trucks_HFT_bat_in,   trucks_HFT_bat_out,   trucks_HFT_bat_stock     = nr_by_cohorts_to_materials_typical_np(trucks_HFT_in[:,:,last_years:], trucks_HFT_out_coh[:,:,last_years:, last_years:], trucks_HFT_stock_coh[:,:,last_years:, last_years:], battery_weights_full['HFT'],  battery_material_composition.loc[idx['HFT',:],:].droplevel(0))
# trucks_MFT_bat_in,   trucks_MFT_bat_out,   trucks_MFT_bat_stock     = nr_by_cohorts_to_materials_typical_np(trucks_MFT_in[:,:,last_years:], trucks_MFT_out_coh[:,:,last_years:, last_years:], trucks_MFT_stock_coh[:,:,last_years:, last_years:], battery_weights_full['MFT'],  battery_material_composition.loc[idx['MFT',:],:].droplevel(0))
# trucks_LCV_bat_in,   trucks_LCV_bat_out,   trucks_LCV_bat_stock     = nr_by_cohorts_to_materials_typical_np(trucks_LCV_in[:,:,last_years:], trucks_LCV_out_coh[:,:,last_years:, last_years:], trucks_LCV_stock_coh[:,:,last_years:, last_years:], battery_weights_full['LCV'],  battery_material_composition.loc[idx['LCV',:],:].droplevel(0))

# # Sum the weight of the accounted materials (! so not total weight) in batteries by vehicle & vehicle type, output for figures
# for vtype in list(battery_weights_full.columns.levels[1]):
#       battery_weight_total_in.loc[idx['reg_bus',:],vtype]     = bus_regl_bat_in.loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values
#       battery_weight_total_in.loc[idx['midi_bus',:],vtype]    = bus_midi_bat_in.loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values
#       battery_weight_total_in.loc[idx['LCV',:],vtype]         = trucks_LCV_bat_in.loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values
#       battery_weight_total_in.loc[idx['MFT',:],vtype]         = trucks_MFT_bat_in.loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values
#       battery_weight_total_in.loc[idx['HFT',:],vtype]         = trucks_HFT_bat_in.loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values
      
#       battery_weight_total_stock.loc[idx['reg_bus',:],vtype]  = bus_regl_bat_stock.loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values
#       battery_weight_total_stock.loc[idx['midi_bus',:],vtype] = bus_midi_bat_stock.loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values
#       battery_weight_total_stock.loc[idx['LCV',:],vtype]      = trucks_LCV_bat_stock.loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values
#       battery_weight_total_stock.loc[idx['MFT',:],vtype]      = trucks_MFT_bat_stock.loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values
#       battery_weight_total_stock.loc[idx['HFT',:],vtype]      = trucks_HFT_bat_stock.loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values   
      
#       if vtype == 'Trolley':
#          pass 
#       else:
#          battery_weight_total_in.loc[idx['car',:],vtype]      = car_total_bat_in.loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values
#          battery_weight_total_stock.loc[idx['car',:],vtype]   = car_total_bat_stock.loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values

# battery_weight_total_in.to_csv('output\\' + FOLDER + '\\battery_weight_kg_in.csv', index=True)       # in kg
# battery_weight_total_stock.to_csv('output\\' + FOLDER + '\\battery_weight_kg_stock.csv', index=True) # in kg

# # Regional battery weight (only the accounted materials), used in graph later on
# battery_weight_regional_stock = bus_regl_bat_stock.sum(axis=0,level=1).sum(axis=1,level=1) + bus_midi_bat_stock.sum(axis=0,level=1).sum(axis=1,level=1) + trucks_LCV_bat_stock.sum(axis=0,level=1).sum(axis=1,level=1) + trucks_MFT_bat_stock.sum(axis=0,level=1).sum(axis=1,level=1) + trucks_HFT_bat_stock.sum(axis=0,level=1).sum(axis=1,level=1) + car_total_bat_stock.sum(axis=0,level=1).sum(axis=1,level=1)
# battery_weight_regional_in    = bus_regl_bat_in.sum(axis=0,level=1).sum(axis=1,level=1)    + bus_midi_bat_in.sum(axis=0,level=1).sum(axis=1,level=1)    + trucks_LCV_bat_in.sum(axis=0,level=1).sum(axis=1,level=1)    + trucks_MFT_bat_in.sum(axis=0,level=1).sum(axis=1,level=1)    + trucks_HFT_bat_in.sum(axis=0,level=1).sum(axis=1,level=1)    + car_total_bat_in.sum(axis=0,level=1).sum(axis=1,level=1)     
# battery_weight_regional_out   = bus_regl_bat_out.sum(axis=0,level=1).sum(axis=1,level=1)   + bus_midi_bat_out.sum(axis=0,level=1).sum(axis=1,level=1)   + trucks_LCV_bat_out.sum(axis=0,level=1).sum(axis=1,level=1)   + trucks_MFT_bat_out.sum(axis=0,level=1).sum(axis=1,level=1)   + trucks_HFT_bat_out.sum(axis=0,level=1).sum(axis=1,level=1)   + car_total_bat_out.sum(axis=0,level=1).sum(axis=1,level=1)                                      

# #%% ################################### Organise data for output ###########################################

# year_select = list(range(START_YEAR, END_YEAR + 1))

# # define 6 dataframes on materials in  stock, inflow & outflow X passenger vs. freight vehicles
# index   = pd.MultiIndex.from_product([year_select, list(range(1,REGIONS+1)), ['vehicle','battery'], labels_materials], names=['year', 'region', 'part', 'materials'])
# vehicle_materials_stock_passenger   = pd.DataFrame(index=index, columns=labels_pas)
# vehicle_materials_stock_freight     = pd.DataFrame(index=index, columns=labels_fre)
# vehicle_materials_inflow_passenger  = pd.DataFrame(index=index, columns=labels_pas)
# vehicle_materials_inflow_freight    = pd.DataFrame(index=index, columns=labels_fre)
# vehicle_materials_outflow_passenger = pd.DataFrame(index=index, columns=labels_pas)
# vehicle_materials_outflow_freight   = pd.DataFrame(index=index, columns=labels_fre)

# for material in labels_materials:

#    ############## STARTING WITH SIMPLE VEHICLES ###########################
   
#    # passenger stock, vehicles (in kg)
#    vehicle_materials_stock_passenger.loc[idx[:,:,'vehicle', material],'bicycle']       = bikes_mat_stock.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_stock_passenger.loc[idx[:,:,'vehicle', material],'rail_reg']      = rail_reg_mat_stock.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_stock_passenger.loc[idx[:,:,'vehicle', material],'rail_hst']      = rail_hst_mat_stock.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_stock_passenger.loc[idx[:,:,'vehicle', material],'air_pas']       = air_pas_mat_stock.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values

#    # freight stock (in kg)
#    vehicle_materials_stock_freight.loc[idx[:,:,'vehicle', material],'inland_shipping']     = inland_ship_mat_stock.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_stock_freight.loc[idx[:,:,'vehicle', material],'rail_freight']        = rail_freight_mat_stock.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_stock_freight.loc[idx[:,:,'vehicle', material],'air_freight']         = air_freight_mat_stock.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_stock_freight.loc[idx[:,:,'vehicle', material],'sea_shipping_small']  = ship_small_mat_stock.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_stock_freight.loc[idx[:,:,'vehicle', material],'sea_shipping_med']    = ship_medium_mat_stock.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_stock_freight.loc[idx[:,:,'vehicle', material],'sea_shipping_large']  = ship_large_mat_stock.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_stock_freight.loc[idx[:,:,'vehicle', material],'sea_shipping_vl']     = ship_vlarge_mat_stock.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values

#    # passenger inflow (in kg)
#    vehicle_materials_inflow_passenger.loc[idx[:,:,'vehicle', material],'bicycle']    = bikes_mat_in.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_inflow_passenger.loc[idx[:,:,'vehicle', material],'rail_reg']   = rail_reg_mat_in.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_inflow_passenger.loc[idx[:,:,'vehicle', material],'rail_hst']   = rail_hst_mat_in.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_inflow_passenger.loc[idx[:,:,'vehicle', material],'air_pas']    = air_pas_mat_in.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values

#    # freight inflow (in kg)
#    vehicle_materials_inflow_freight.loc[idx[:,:,'vehicle', material],'inland_shipping']    = inland_ship_mat_in.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_inflow_freight.loc[idx[:,:,'vehicle', material],'rail_freight']       = rail_freight_mat_in.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_inflow_freight.loc[idx[:,:,'vehicle', material],'air_freight']        = air_freight_mat_in.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_inflow_freight.loc[idx[:,:,'vehicle', material],'sea_shipping_small'] = ship_small_mat_in.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_inflow_freight.loc[idx[:,:,'vehicle', material],'sea_shipping_med']   = ship_medium_mat_in.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_inflow_freight.loc[idx[:,:,'vehicle', material],'sea_shipping_large'] = ship_large_mat_in.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_inflow_freight.loc[idx[:,:,'vehicle', material],'sea_shipping_vl']    = ship_vlarge_mat_in.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values

#    # passenger outflow (in kg)
#    vehicle_materials_outflow_passenger.loc[idx[:,:,'vehicle', material],'bicycle']   = bikes_mat_out.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_outflow_passenger.loc[idx[:,:,'vehicle', material],'rail_reg']  = rail_reg_mat_out.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_outflow_passenger.loc[idx[:,:,'vehicle', material],'rail_hst']  = rail_hst_mat_out.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_outflow_passenger.loc[idx[:,:,'vehicle', material],'air_pas']   = air_pas_mat_out.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values

#    # freight outflow (in kg)
#    vehicle_materials_outflow_freight.loc[idx[:,:,'vehicle', material],'inland_shipping']     = inland_ship_mat_out.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_outflow_freight.loc[idx[:,:,'vehicle', material],'rail_freight']        = rail_freight_mat_out.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_outflow_freight.loc[idx[:,:,'vehicle', material],'air_freight']         = air_freight_mat_out.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_outflow_freight.loc[idx[:,:,'vehicle', material],'sea_shipping_small']  = ship_small_mat_out.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_outflow_freight.loc[idx[:,:,'vehicle', material],'sea_shipping_med']    = ship_medium_mat_out.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_outflow_freight.loc[idx[:,:,'vehicle', material],'sea_shipping_large']  = ship_large_mat_out.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_outflow_freight.loc[idx[:,:,'vehicle', material],'sea_shipping_vl']     = ship_vlarge_mat_out.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values

#    ############ CONTINUEING WITH TYPICAL VEHICLES (MATERIALS IN TRUCKS & BUSSES ARE SUMMED, FOR CARS DETAIL BY TYPE IS MAINTAINED) ##########################
   
#    part = 'vehicle'
   
#    # passenger stock, vehicles (in kg)
#    vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'midi_bus']    = bus_midi_mat_stock.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values
#    vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'reg_bus']     = bus_regl_mat_stock.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values
#    vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'ICE']         = car_total_mat_stock.loc[idx[material,year_select],idx['ICE',:]].stack().reorder_levels([1,2,0]).values 
#    vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'HEV']         = car_total_mat_stock.loc[idx[material,year_select],idx['HEV',:]].stack().reorder_levels([1,2,0]).values   
#    vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'PHEV']        = car_total_mat_stock.loc[idx[material,year_select],idx['PHEV',:]].stack().reorder_levels([1,2,0]).values  
#    vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'BEV']         = car_total_mat_stock.loc[idx[material,year_select],idx['BEV',:]].stack().reorder_levels([1,2,0]).values  
#    vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'FCV']         = car_total_mat_stock.loc[idx[material,year_select],idx['FCV',:]].stack().reorder_levels([1,2,0]).values       
   
#    # freight stock (in kg)
#    vehicle_materials_stock_freight.loc[idx[:,:, part, material],'LCV']           = trucks_LCV_mat_stock.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values   
#    vehicle_materials_stock_freight.loc[idx[:,:, part, material],'MFT']           = trucks_MFT_mat_stock.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values  
#    vehicle_materials_stock_freight.loc[idx[:,:, part, material],'HFT']           = trucks_HFT_mat_stock.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values 

#    # passenger inflow (in kg)
#    vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'midi_bus']   = bus_midi_mat_in.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values
#    vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'reg_bus']    = bus_regl_mat_in.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values
#    vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'ICE']        = car_total_mat_in.loc[idx[material,year_select],idx['ICE',:]].stack().reorder_levels([1,2,0]).values 
#    vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'HEV']        = car_total_mat_in.loc[idx[material,year_select],idx['HEV',:]].stack().reorder_levels([1,2,0]).values 
#    vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'PHEV']       = car_total_mat_in.loc[idx[material,year_select],idx['PHEV',:]].stack().reorder_levels([1,2,0]).values 
#    vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'BEV']        = car_total_mat_in.loc[idx[material,year_select],idx['BEV',:]].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'FCV']        = car_total_mat_in.loc[idx[material,year_select],idx['FCV',:]].stack().reorder_levels([1,2,0]).values        
 
#    # freight inflow (in kg)
#    vehicle_materials_inflow_freight.loc[idx[:,:, part, material],'LCV']          = trucks_LCV_mat_in.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values   
#    vehicle_materials_inflow_freight.loc[idx[:,:, part, material],'MFT']          = trucks_MFT_mat_in.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values  
#    vehicle_materials_inflow_freight.loc[idx[:,:, part, material],'HFT']          = trucks_HFT_mat_in.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values    

#    # passenger outflow (in kg)
#    vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'midi_bus']  = bus_midi_mat_out.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values
#    vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'reg_bus']   = bus_regl_mat_out.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values 
#    vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'ICE']       = car_total_mat_out.loc[idx[material,year_select],idx['ICE',:]].stack().reorder_levels([1,2,0]).values  
#    vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'HEV']       = car_total_mat_out.loc[idx[material,year_select],idx['HEV',:]].stack().reorder_levels([1,2,0]).values 
#    vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'PHEV']      = car_total_mat_out.loc[idx[material,year_select],idx['PHEV',:]].stack().reorder_levels([1,2,0]).values 
#    vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'BEV']       = car_total_mat_out.loc[idx[material,year_select],idx['BEV',:]].stack().reorder_levels([1,2,0]).values 
#    vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'FCV']       = car_total_mat_out.loc[idx[material,year_select],idx['FCV',:]].stack().reorder_levels([1,2,0]).values       
   
#    # freight outflow (in kg)
#    vehicle_materials_outflow_freight.loc[idx[:,:, part, material],'LCV'] = trucks_LCV_mat_out.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values  
#    vehicle_materials_outflow_freight.loc[idx[:,:, part, material],'MFT']= trucks_MFT_mat_out.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values 
#    vehicle_materials_outflow_freight.loc[idx[:,:, part, material],'HFT'] = trucks_HFT_mat_out.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values 

#    ############ CONTINUEING WITH BATTERIES (MATERALS IN TRUCKS & BUSSES ARE SUMMED, FOR CARS DETAIL BY TYPE IS MAINTAINED) ##########################

#    part = 'battery'
   
#    # passenger stock, vehicles (in kg)
#    vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'midi_bus']    = bus_midi_bat_stock.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values
#    vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'reg_bus']     = bus_regl_bat_stock.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values
#    vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'ICE']         = car_total_bat_stock.loc[idx[material,year_select],idx['ICE',:]].stack().reorder_levels([1,2,0]).values 
#    vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'HEV']         = car_total_bat_stock.loc[idx[material,year_select],idx['HEV',:]].stack().reorder_levels([1,2,0]).values   
#    vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'PHEV']        = car_total_bat_stock.loc[idx[material,year_select],idx['PHEV',:]].stack().reorder_levels([1,2,0]).values  
#    vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'BEV']         = car_total_bat_stock.loc[idx[material,year_select],idx['BEV',:]].stack().reorder_levels([1,2,0]).values  
#    vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'FCV']         = car_total_bat_stock.loc[idx[material,year_select],idx['FCV',:]].stack().reorder_levels([1,2,0]).values       
   
#    # freight stock (in kg)
#    vehicle_materials_stock_freight.loc[idx[:,:, part, material],'LCV']           = trucks_LCV_bat_stock.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values   
#    vehicle_materials_stock_freight.loc[idx[:,:, part, material],'MFT']           = trucks_MFT_bat_stock.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values  
#    vehicle_materials_stock_freight.loc[idx[:,:, part, material],'HFT']           = trucks_HFT_bat_stock.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values 

#    # passenger inflow (in kg)
#    vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'midi_bus']   = bus_midi_bat_in.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values
#    vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'reg_bus']    = bus_regl_bat_in.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values
#    vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'ICE']        = car_total_bat_in.loc[idx[material,year_select],idx['ICE',:]].stack().reorder_levels([1,2,0]).values 
#    vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'HEV']        = car_total_bat_in.loc[idx[material,year_select],idx['HEV',:]].stack().reorder_levels([1,2,0]).values 
#    vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'PHEV']       = car_total_bat_in.loc[idx[material,year_select],idx['PHEV',:]].stack().reorder_levels([1,2,0]).values 
#    vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'BEV']        = car_total_bat_in.loc[idx[material,year_select],idx['BEV',:]].stack().reorder_levels([1,2,0]).values
#    vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'FCV']        = car_total_bat_in.loc[idx[material,year_select],idx['FCV',:]].stack().reorder_levels([1,2,0]).values        
 
#    # freight inflow (in kg)
#    vehicle_materials_inflow_freight.loc[idx[:,:, part, material],'LCV']          = trucks_LCV_bat_in.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values   
#    vehicle_materials_inflow_freight.loc[idx[:,:, part, material],'MFT']          = trucks_MFT_bat_in.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values  
#    vehicle_materials_inflow_freight.loc[idx[:,:, part, material],'HFT']          = trucks_HFT_bat_in.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values    

#    # passenger outflow (in kg)
#    vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'midi_bus']  = bus_midi_bat_out.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values
#    vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'reg_bus']   = bus_regl_bat_out.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values 
#    vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'ICE']       = car_total_bat_out.loc[idx[material,year_select],idx['ICE',:]].stack().reorder_levels([1,2,0]).values  
#    vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'HEV']       = car_total_bat_out.loc[idx[material,year_select],idx['HEV',:]].stack().reorder_levels([1,2,0]).values 
#    vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'PHEV']      = car_total_bat_out.loc[idx[material,year_select],idx['PHEV',:]].stack().reorder_levels([1,2,0]).values 
#    vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'BEV']       = car_total_bat_out.loc[idx[material,year_select],idx['BEV',:]].stack().reorder_levels([1,2,0]).values 
#    vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'FCV']       = car_total_bat_out.loc[idx[material,year_select],idx['FCV',:]].stack().reorder_levels([1,2,0]).values       
   
#    # freight outflow (in kg)
#    vehicle_materials_outflow_freight.loc[idx[:,:, part, material],'LCV']         = trucks_LCV_bat_out.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values  
#    vehicle_materials_outflow_freight.loc[idx[:,:, part, material],'MFT']         = trucks_MFT_bat_out.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values 
#    vehicle_materials_outflow_freight.loc[idx[:,:, part, material],'HFT']         = trucks_HFT_bat_out.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values 


# #%% combine dataframes for output

# # add flow descriptor to the multi-index & fill na values with 0 (the for-loop above didn't cover the battery materials in vhicles without batteries, so these are set to 0 now)
# vehicle_materials_stock_passenger  = pd.concat([vehicle_materials_stock_passenger.fillna(0)],    keys=['stock'],    names=['flow'])   
# vehicle_materials_stock_freight    = pd.concat([vehicle_materials_stock_freight.fillna(0)],      keys=['stock'],    names=['flow'])      
# vehicle_materials_inflow_passenger = pd.concat([vehicle_materials_inflow_passenger.fillna(0)],   keys=['inflow'],   names=['flow'])     
# vehicle_materials_inflow_freight   = pd.concat([vehicle_materials_inflow_freight.fillna(0)],     keys=['inflow'],   names=['flow'])     
# vehicle_materials_outflow_passenger = pd.concat([vehicle_materials_outflow_passenger.fillna(0)], keys=['outflow'],  names=['flow'])      
# vehicle_materials_outflow_freight   = pd.concat([vehicle_materials_outflow_freight.fillna(0)],   keys=['outflow'],  names=['flow']) 

# # concatenate stock, inflow & outflow into 1 dataframe (1 for passenger & 1 for freight)
# vehicle_materials_passenger = pd.concat([vehicle_materials_stock_passenger, vehicle_materials_inflow_passenger,  vehicle_materials_outflow_passenger]) 
# vehicle_materials_freight   = pd.concat([vehicle_materials_stock_freight,   vehicle_materials_inflow_freight,    vehicle_materials_outflow_freight])

# # add category descriptors to the multi-index (pass vs. freight)
# vehicle_materials_passenger = pd.concat([vehicle_materials_passenger], keys=['passenger'], names=['category']) 
# vehicle_materials_freight   = pd.concat([vehicle_materials_freight],    keys=['freight'], names=['category'])  

# vehicle_materials_passenger.columns.name = vehicle_materials_freight.columns.name = 'elements'

# # concatenate into 1 single dataframe & add the 'vehicle' descriptor
# vehicle_materials = pd.concat([vehicle_materials_passenger.stack().unstack(level=2), vehicle_materials_freight.stack().unstack(level=2)])
# vehicle_materials = pd.concat([vehicle_materials], keys=['vehicles'], names=['sector'])

# # re-order multi-index to the desired output & sum to global total
# vehicle_materials_out = vehicle_materials.reorder_levels([3, 2, 0, 1, 6, 5, 4]) / 1000000  # div by 1*10^6 to translate from kg to kt
# vehicle_materials_out.reset_index(inplace=True)                                         # return to columns

# vehicle_materials_out.to_csv('output\\' + FOLDER + '\\vehicle_materials_kt.csv', index=False) # in kt

# time.sleep(1)

# # get the end time
# et = time.time()

# # get the execution time
# elapsed_time = et - st
# print('Execution time:', elapsed_time, 'seconds')

#%% PLots for Freight & Passenger pkm/tkm demand (2000-2100), global 

# # color dictionary for vehicles & materials
# color_vehicle = {
#   "air"                   : "#334fff",
#   "biking"                : "#058b37",
#   "bus"                   : "#f2cc3a",
#   "midi bus"              : "#e89a18",
#   "car"                   : "#09608b",
#   "train"                 : "#76765d",
#   "hst"                   : "#6b6b54",
#   "international shipping": "#199841",
#   "inland shipping"       : "#199881",
#   "freight train"         : "#193198",
#   "medium truck"          : "#811998",
#   "heavy truck"           : "#197098",
#   "air cargo"             : "#578F16"
# }


# #tkms_label  = ['inland shipping', 'freight train', 'medium truck', 'heavy truck', 'air cargo', 'international shipping', 'empty', 'total']
# #pkms_label  = ['walking', 'biking', 'bus', 'train', 'car', 'hst', 'air', 'total']
# font = {'family' : 'sans-serif',
#         'weight' : 'regular',
#         'size'   : 10}

# plt.rc('font', **font)
# index = np.array(list(range(2000,OUT_YEAR + 1)))

# if VARIANT == 'BL':
#     ylim = [70,125]
# else:
#     ylim = [70,125]

# plt.figure()
# fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15,6), frameon=True)
# fig.suptitle('Passenger & Freight Tranport Demand in the SSP2 Baseline (in Pkm & Tkm)', y=1, fontsize=16)
# plt.subplots_adjust(wspace = 0.25, bottom = 0.15)

# ax1.set_title('Passenger Transport Demand (Tera Pkm/yr)', fontsize=12)
# ax1.set_ylim(0,ylim[0])
# ax1.set_ylabel('Tera person-kilometer', rotation='vertical', y=0.8, fontsize=11)
# ax1.yaxis.set_ticks_position('left')
# ax1.xaxis.set_ticks_position('bottom')
# ax1.margins(x=0)
# ax1.plot(index, passengerkms_Tpkms['air'].unstack()[28].loc[2000:OUT_YEAR],    'blue',   label='Airplane',         linewidth=2)
# ax1.plot(index, passengerkms_Tpkms['biking'].unstack()[28].loc[2000:OUT_YEAR], 'green',  label='Bicycle',          linewidth=2)
# ax1.plot(index, passengerkms_Tpkms['bus'].unstack()[28].loc[2000:OUT_YEAR],    'gold',   label='Bus',              linewidth=2)
# ax1.plot(index, passengerkms_Tpkms['car'].unstack()[28].loc[2000:OUT_YEAR],    'red',    label='Car',              linewidth=2)
# ax1.plot(index, passengerkms_Tpkms['train'].unstack()[28].loc[2000:OUT_YEAR],  'purple', label='Train',            linewidth=2)
# ax1.plot(index, passengerkms_Tpkms['hst'].unstack()[28].loc[2000:OUT_YEAR],    'orange', label='High Speed Train', linewidth=2)
# ax1.legend(loc=2, bbox_to_anchor=(0.015, -0.05), ncol=3, frameon=False, fontsize=10)

# ax2.set_title('Freight Transport Demand (Tera Tkm/yr)', fontsize=12)
# ax2.set_ylim(0,ylim[1])
# ax2.set_ylabel('Tera ton-kilometer', rotation='vertical', y=0.8, fontsize=11)
# ax2.yaxis.set_ticks_position('left')
# ax2.xaxis.set_ticks_position('bottom')
# ax2.margins(x=0)
# ax2.plot(index, tonkms_Mtkms['international shipping'].unstack()[28].loc[2000:OUT_YEAR] / 1000000,    'black',    label='Internl. Shipping', linewidth=2)
# ax2.plot(index, tonkms_Mtkms['inland shipping'].unstack()[28].loc[2000:OUT_YEAR]        / 1000000,    'brown',    label='Inland Shipping',   linewidth=2)
# ax2.plot(index, tonkms_Mtkms['freight train'].unstack()[28].loc[2000:OUT_YEAR]          / 1000000,    'violet',   label='Rail Cargo',        linewidth=2)
# ax2.plot(index, (tonkms_Mtkms['medium truck'].unstack()[28].loc[2000:OUT_YEAR] + tonkms_Mtkms['heavy truck'].unstack()[28].loc[2000:OUT_YEAR]) / 1000000, 'grey',     label='Truck',             linewidth=2)
# ax2.plot(index, tonkms_Mtkms['air cargo'].unstack()[28].loc[2000:OUT_YEAR]              / 1000000,    'darkcyan', label='Air Cargo',         linewidth=2)

# ax2.legend(loc=2, bbox_to_anchor=(0.015, -0.05), ncol=3, frameon=False, fontsize=10)
# plt.savefig('output\\'  + FOLDER + '\\graphs\\Pkm-Tkm.jpg', dpi=600, pad_inches=2)
# plt.show()

# #%% Figure on the material composition (wt%)  of all vehicles

# color_material = {
#   "Steel"     : "#16697a",
#   "Aluminium" : "#79a3b1",
#   "Cu"        : "#c7956d",
#   "Plastics"  : "#ffd369",
#   "Glass"     : "#bbbbbb",
#   "Rubber"    : "#0a043c",
#   "Co"        : "#314a73",
#   "Li"        : "#d6d6d6",
#   "Mn"        : "#6d5987",
#   "Nd"        : "#e38846",
#   "Ni"        : "#428a66",
#   "Pb"        : "#5b6b5f",
#   "Ti"        : "#daf2e0",
#   "Wood"      : "#73523b"
# }

# year = 2018
# mul = 100
# material_fractions_truck = (material_fractions_truck_MFT + material_fractions_truck_HFT) / 2    # pre calculate the avreage of Heavy & Medium Trucks

# plotvar_steel      = [material_fractions_car.loc[year, idx['ICE','Steel']]*mul,     material_fractions_bicycle.loc[year,'Steel']*mul,     material_fractions_bus_midi.loc[year, idx['ICE','Steel']]*mul,     material_fractions_bus_reg.loc[year, idx['ICE','Steel']]*mul,     material_fractions_air_pas.loc[year,'Steel']*mul,     material_fractions_rail_reg.loc[year,'Steel']*mul,     material_fractions_rail_hst.loc[year,'Steel']*mul,     0, material_fractions_air_frgt.loc[year,'Steel']*mul,     material_fractions_rail_frgt.loc[year,'Steel']*mul,     material_fractions_inland_ship.loc[year,'Steel']*mul,     material_fractions_truck_LCV.loc[year, idx['ICE','Steel']]*mul,     material_fractions_truck.loc[year, idx['ICE','Steel']]*mul]
# plotvar_aluminium  = [material_fractions_car.loc[year, idx['ICE','Aluminium']]*mul, material_fractions_bicycle.loc[year,'Aluminium']*mul, material_fractions_bus_midi.loc[year, idx['ICE','Aluminium']]*mul, material_fractions_bus_reg.loc[year, idx['ICE','Aluminium']]*mul, material_fractions_air_pas.loc[year,'Aluminium']*mul, material_fractions_rail_reg.loc[year,'Aluminium']*mul, material_fractions_rail_hst.loc[year,'Aluminium']*mul, 0, material_fractions_air_frgt.loc[year,'Aluminium']*mul, material_fractions_rail_frgt.loc[year,'Aluminium']*mul, material_fractions_inland_ship.loc[year,'Aluminium']*mul, material_fractions_truck_LCV.loc[year, idx['ICE','Aluminium']]*mul, material_fractions_truck.loc[year, idx['ICE','Aluminium']]*mul]
# plotvar_cu         = [material_fractions_car.loc[year, idx['ICE','Cu']]*mul,        material_fractions_bicycle.loc[year,'Cu']*mul,        material_fractions_bus_midi.loc[year, idx['ICE','Cu']]*mul,        material_fractions_bus_reg.loc[year, idx['ICE','Cu']]*mul,        material_fractions_air_pas.loc[year,'Cu']*mul,        material_fractions_rail_reg.loc[year,'Cu']*mul,        material_fractions_rail_hst.loc[year,'Cu']*mul,        0, material_fractions_air_frgt.loc[year,'Cu']*mul,        material_fractions_rail_frgt.loc[year,'Cu']*mul,        material_fractions_inland_ship.loc[year,'Cu']*mul,        material_fractions_truck_LCV.loc[year, idx['ICE','Cu']]*mul,        material_fractions_truck.loc[year, idx['ICE','Cu']]*mul]
# plotvar_plastics   = [material_fractions_car.loc[year, idx['ICE','Plastics']]*mul,  material_fractions_bicycle.loc[year,'Plastics']*mul,  material_fractions_bus_midi.loc[year, idx['ICE','Plastics']]*mul,  material_fractions_bus_reg.loc[year, idx['ICE','Plastics']]*mul,  material_fractions_air_pas.loc[year,'Plastics']*mul,  material_fractions_rail_reg.loc[year,'Plastics']*mul,  material_fractions_rail_hst.loc[year,'Plastics']*mul,  0, material_fractions_air_frgt.loc[year,'Plastics']*mul,  material_fractions_rail_frgt.loc[year,'Plastics']*mul,  material_fractions_inland_ship.loc[year,'Plastics']*mul,  material_fractions_truck_LCV.loc[year, idx['ICE','Plastics']]*mul,  material_fractions_truck.loc[year, idx['ICE','Plastics']]*mul]
# plotvar_glass      = [material_fractions_car.loc[year, idx['ICE','Glass']]*mul,     material_fractions_bicycle.loc[year,'Glass']*mul,     material_fractions_bus_midi.loc[year, idx['ICE','Glass']]*mul,     material_fractions_bus_reg.loc[year, idx['ICE','Glass']]*mul,     material_fractions_air_pas.loc[year,'Glass']*mul,     material_fractions_rail_reg.loc[year,'Glass']*mul,     material_fractions_rail_hst.loc[year,'Glass']*mul,     0, material_fractions_air_frgt.loc[year,'Glass']*mul,     material_fractions_rail_frgt.loc[year,'Glass']*mul,     material_fractions_inland_ship.loc[year,'Glass']*mul,     material_fractions_truck_LCV.loc[year, idx['ICE','Glass']]*mul,     material_fractions_truck.loc[year, idx['ICE','Glass']]*mul]
# plotvar_rubber     = [material_fractions_car.loc[year, idx['ICE','Rubber']]*mul,    material_fractions_bicycle.loc[year,'Rubber']*mul,    material_fractions_bus_midi.loc[year, idx['ICE','Rubber']]*mul,    material_fractions_bus_reg.loc[year, idx['ICE','Rubber']]*mul,    material_fractions_air_pas.loc[year,'Rubber']*mul,    material_fractions_rail_reg.loc[year,'Rubber']*mul,    material_fractions_rail_hst.loc[year,'Rubber']*mul,    0, material_fractions_air_frgt.loc[year,'Rubber']*mul,    material_fractions_rail_frgt.loc[year,'Rubber']*mul,    material_fractions_inland_ship.loc[year,'Rubber']*mul,    material_fractions_truck_LCV.loc[year, idx['ICE','Rubber']]*mul,    material_fractions_truck.loc[year, idx['ICE','Rubber']]*mul]

# barWidth = 0.5
# r1 = np.arange(13)

# plt.figure(figsize=(20, 6))
# fig, ax = plt.subplots(figsize=(20,6))
# plt.rc('font', **font)
# plt.yticks(fontsize=16)
# plt.subplots_adjust(wspace = 0.1, bottom = 0.15, right = 0.80)
# ax.set_ylim(ymin=0, ymax=100)
# ax.bar(r1, plotvar_steel,     color=color_material["Steel"],     width=barWidth, edgecolor=None, label='2018')
# bottom = plotvar_steel
# ax.bar(r1, plotvar_aluminium, color=color_material["Aluminium"], width=barWidth, bottom=bottom, edgecolor=None, label='2018')
# bottom = [a+b for a,b in zip(bottom,plotvar_aluminium)]
# ax.bar(r1, plotvar_cu,        color=color_material["Cu"],        width=barWidth, bottom=bottom, edgecolor=None, label='2018')
# bottom = [a+b for a,b in zip(bottom,plotvar_cu)]
# ax.bar(r1, plotvar_plastics,  color=color_material["Plastics"],  width=barWidth, bottom=bottom, edgecolor=None, label='2018')
# bottom = [a+b for a,b in zip(bottom,plotvar_plastics)]
# ax.bar(r1, plotvar_glass,     color=color_material["Glass"],     width=barWidth, bottom=bottom, edgecolor=None, label='2018')
# bottom = [a+b for a,b in zip(bottom,plotvar_glass)]
# ax.bar(r1, plotvar_rubber,    color=color_material["Rubber"],    width=barWidth, bottom=bottom, edgecolor=None, label='2018')

# # Add xticks on the middle of the group bars
# plt.xticks([r + barWidth - 0.5 for r in range(13)], ['Cars', 'Bicycles', 'Midi Bus', 'Bus', 'Airplane', 'Rail', 'High Speed Train', '', 'Air Cargo', 'Rail Cargo', 'Ships', 'Light Truck', 'Other Trucks'], fontsize=11)
# plt.ylabel('vehicle composition (wt%)', fontsize=18, y=0.5)
# plt.title('Default material composition of different vehicles', y=1.08, fontsize=16)

# # Legend & text
# legend_elements = [matplotlib.patches.Patch(facecolor=color_material["Rubber"],    label='Rubber'),
#                    matplotlib.patches.Patch(facecolor=color_material["Glass"],     label='Glass'),
#                    matplotlib.patches.Patch(facecolor=color_material["Plastics"],  label='Plastics'),
#                    matplotlib.patches.Patch(facecolor=color_material["Cu"],        label='Cu'),
#                    matplotlib.patches.Patch(facecolor=color_material["Aluminium"], label='Aluminium'),
#                    matplotlib.patches.Patch(facecolor=color_material["Steel"],     label='Steel')]
# plt.legend(handles=legend_elements, loc=2, bbox_to_anchor=(1,1), ncol=1, frameon=False, fontsize=16)
# fig.savefig('output\\' + FOLDER + '\\graphs\\bars_compositon.png', dpi=600)
# plt.show()

# #%% Figures on results
# # Plot panel (1 * 2) Basic stacked area chart: Global totals of all battery types & materials 
# years_select     = list(range(2000,OUT_YEAR + 1))

# color_battery = {
#   "NiMH"                  : "#334fff",
#   "LMO/LCO"               : "#058b37",
#   "NMC"                   : "#f2cc3a",
#   "NCA"                   : "#e89a18",
#   "LFP"                   : "#db6400",
#   "Li-S"                  : "#fd3a69",
#   "Li-Ceramic"            : "#9d0191",
#   "Li-Air"                : "#120078"
# }

# #prepare data first
# vehicle_list =       ['Bicycles', 'Cars',    'Planes',  'Trains',  'Buses',   'Rail Cargo', 'Air Cargo', 'Trucks',  'Ships']
# colors_vehicles =    ["#343f5c", "#45537a",    "#576b9e",   "#8caaf5", "#d1deff", "#3d4a40",  "#516355", "#809c86", "#aed6b6"]
# veh_colors_dict = {vehicle_list[i]: colors_vehicles[i] for i in range(len(vehicle_list))} 
             
# # pre-process & assign plot data
# graph1_data = battery_shares_full.T[years_select]
# graph1_data = graph1_data.rename(index={'Lithium-air':'Li-Air','Lithium Ceramic ':'Li-Ceramic','Lithium Sulfur':'Li-S'})

# graph2_data = vehicle_materials.loc[idx[:,:,'inflow',:,'battery',:,:],years_select].sum(axis=0, level=5)
# graph2_data_total = graph2_data.sum(axis=0)

# #loop to get fractional contribution
# for material in list(graph2_data.index):
#    if graph2_data.loc[material].sum() > 0.01:
#       graph2_data.loc[material] = graph2_data.loc[material] / graph2_data_total
#    else: 
#       graph2_data = graph2_data.drop(material)

# years_select         = list(range(1990,OUT_YEAR +1))
# window_size          = 10
# years_select_partial = years_select

# graph_data_original = battery_weight_total_in

# # duplicate data for the last available year (to make moving average window work)
# graph_data_extended = graph_data_original
# if (OUT_YEAR == END_YEAR):
#     for year in range(1,window_size):
#         year_used = OUT_YEAR + year
#         add  = graph_data_original.loc[idx[:,OUT_YEAR],:]
#         index = pd.MultiIndex.from_product([list(add.index.levels[0]),[year_used]])
#         add.set_index(index,inplace=True)
#         graph_data_extended = pd.concat([graph_data_extended, add])
                       
# graph3_data = graph_data_extended.sum(axis=1).unstack()[years_select]    # battery weight by vehicle (cars,bus,truck)
# graph4_data = graph_data_extended.sum(axis=0, level=1).T.rolling(window=window_size, axis=1).mean()[years_select]   # battery weight by drivetrain (HEV/PHEV/BEV)

# graph3_data.loc['Trucks',:] =  graph_data_extended.sum(axis=1).unstack()[years_select].loc[['HFT','MFT','LCV'],:].sum(axis=0).rolling(window=window_size).mean()
# graph3_data.loc['Buses',:]  =  graph_data_extended.sum(axis=1).unstack()[years_select].loc[['midi_bus','reg_bus'],:].sum(axis=0).rolling(window=window_size).mean()
# graph3_data.loc['Cars',:]   =  graph_data_extended.sum(axis=1).unstack()[years_select].loc[['car'],:].sum(axis=0).rolling(window=window_size).mean()

# graph3_data_total = graph3_data.loc[['Cars','Buses','Trucks']].sum(axis=0)
# graph4_data_total = graph4_data.sum(axis=0)

# years_select     = list(range(2000,OUT_YEAR + 1))

# graph3_data = graph3_data.div(graph3_data_total, axis=1)[years_select]
# graph4_data = graph4_data.div(graph4_data_total, axis=1)[years_select].drop(['ICE','FCV'])

# material_colorset = [color_material[i] for i in list(graph2_data.index)]
# battery_colorset  = [color_battery[i]  for i in list(graph1_data.index)]
# vehicle_colors_select = [veh_colors_dict['Trucks'],veh_colors_dict['Buses'],veh_colors_dict['Cars']]

# plt.figure()
# fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize=(15,12), frameon=True)
# fig.suptitle('EV Battery Markets, Composition and Demand', y=1, fontsize=16)
# plt.subplots_adjust(wspace = 0.25, hspace = 0.25, bottom = 0.15)
# ax2nd = ax2.twinx()

# ax1.set_title('Market share of battery types (sales/inflow)', fontsize=12)
# ax1.set_ylim(0,100)
# ax1.set_ylabel('%', rotation='vertical', y=0.98, fontsize=13)
# ax1.yaxis.set_ticks_position('left')
# ax1.xaxis.set_ticks_position('bottom')
# ax1.margins(x=0)
# ax1.stackplot(years_select, graph1_data[years_select] * 100, labels=list(graph1_data.index), colors=battery_colorset)
# handles, labels = ax1.get_legend_handles_labels()
# ax1.legend(reversed(handles), reversed(labels), bbox_to_anchor=(0, -0.20), loc='lower left', ncol=4, frameon=False, fontsize=10)

# ax2.set_title('Battery material composition & sales volume', fontsize=12)
# ax2.set_ylim(0,100)
# ax2.set_ylabel('%', rotation='vertical', y=0.98, fontsize=13)
# ax2nd.set_ylabel('battery weight (kg)', rotation='vertical', x=0.1, y=0.78, fontsize=13)
# ax2.yaxis.set_ticks_position('left')
# ax2.xaxis.set_ticks_position('bottom')
# ax2.margins(x=0)
# ax2nd.margins(x=0)
# ax2.stackplot(years_select, graph2_data * 100, labels=list(graph2_data.index), colors=material_colorset)
# ax2nd.plot(years_select, battery_weight_total_in.sum(axis=1).sum(axis=0,level=1)[years_select], '--', color='black', linewidth=3, label="Sales (>)") 
# handles, labels = ax2.get_legend_handles_labels()
# ax2.legend(reversed(handles), reversed(labels), bbox_to_anchor=(0, -0.20), loc='lower left', ncol=5, frameon=False, fontsize=10)
# ax2nd.legend(bbox_to_anchor=(1, -0.20), loc='lower right', ncol=1, frameon=False, fontsize=10)

# ax3.set_title('Market share (inflow) of batteries by vehicle type', fontsize=12)
# ax3.set_ylim(0,100)
# ax3.set_ylabel('%', rotation='vertical', y=0.98, fontsize=13)
# ax3.yaxis.set_ticks_position('left')
# ax3.xaxis.set_ticks_position('bottom')
# ax3.margins(x=0)
# ax3.stackplot(years_select, graph3_data.loc[['Trucks','Buses','Cars'],:] * 100, labels=['Trucks','Buses','Cars'], colors=vehicle_colors_select)
# handles, labels = ax3.get_legend_handles_labels()
# ax3.legend(reversed(handles), reversed(labels), bbox_to_anchor=(0, -0.18), loc='lower left', ncol=4, frameon=False, fontsize=10)

# ax4.set_title('Market share (inflow) of batteries by drivetrain type', fontsize=12)
# ax4.set_ylim(0,100)
# ax4.set_ylabel('%', rotation='vertical', y=0.98, fontsize=13)
# ax4.yaxis.set_ticks_position('left')
# ax4.xaxis.set_ticks_position('bottom')
# ax4.margins(x=0)
# ax4.stackplot(years_select, graph4_data * 100, labels=['BEV','HEV','PHEV','Trolley'], colors=['#ffce89','#fff76a','#adce74','#61b15a'])
# handles, labels = ax4.get_legend_handles_labels()
# ax4.legend(reversed(handles), reversed(labels), bbox_to_anchor=(0, -0.18), loc='lower left', ncol=4, frameon=False, fontsize=10)

# plt.savefig('output\\' + FOLDER + '\\graphs\\battery_market_panel_inflow_new.png', dpi=600)
# plt.show()

# #%% PLot panel (2 * 2) Results 
# # 1) total vehicle weight (including batteries), Stock by vehicle
# # 2) total vehicle weight (including batteries), Stock by material
# # 3) Steel weight, Stock by hevicle + inflow/outflow
# # 4) Aluminium weight, Stock by vehicle + inflow/outflow

# #material = 'Steel'
# #test = vehicle_material_stock_global.loc[idx[material,['sea_shipping_small', 'sea_shipping_med','sea_shipping_large', 'sea_shipping_vl']],:].sum(axis=0)

# bulk_list =          ["Steel" ,  "Aluminium",   "Cu",   "Plastics",   "Glass",   "Rubber"]
                      
# #grouping
# vehicle_material_stock_global = vehicle_materials.loc[idx[:,:,'stock',:,:,:,:],:].sum(axis=0, level=[5,6])
# for material in list(vehicle_material_stock_global.index.levels[0]):
#    vehicle_material_stock_global.loc[(material,'Cars'),:]   = vehicle_material_stock_global.loc[idx[material,['ICE','HEV','PHEV','BEV','FCV']],:].sum(axis=0)
#    vehicle_material_stock_global.loc[(material,'Trains'),:] = vehicle_material_stock_global.loc[idx[material,['rail_reg','rail_hst']],:].sum(axis=0)
#    vehicle_material_stock_global.loc[(material,'Trucks'),:] = vehicle_material_stock_global.loc[idx[material,['LCV','MFT','HFT']],:].sum(axis=0)
#    vehicle_material_stock_global.loc[(material,'Buses'),:]  = vehicle_material_stock_global.loc[idx[material,['reg_bus','midi_bus']],:].sum(axis=0)
#    vehicle_material_stock_global.loc[(material,'Ships'),:]  = vehicle_material_stock_global.loc[idx[material,['inland_shipping', 'sea_shipping_small', 'sea_shipping_med','sea_shipping_large', 'sea_shipping_vl']],:].sum(axis=0)

# vehicle_material_stock_global = vehicle_material_stock_global.rename(index={"bicycle" : "Bicycles", "air_pas": "Planes", "rail_freight": "Rail Cargo", "air_freight" : "Air Cargo"})

# graph1_data = vehicle_material_stock_global.loc[idx[bulk_list,vehicle_list],years_select].sum(axis=0, level=1).reindex(vehicle_list)    # by vehicle

# # main text (estimate of share of other vehicles)
# car_share_in_total_stock = graph1_data.sum(axis=1)['Cars']/graph1_data.sum(axis=1).sum()

# graph2_data = vehicle_material_stock_global.loc[idx[bulk_list,vehicle_list],years_select].sum(axis=0, level=0)                          # by material
# graph3_data = vehicle_material_stock_global.loc[idx['Steel',vehicle_list],years_select].sum(axis=0, level=1).reindex(vehicle_list)      # Steel by vehicle
# graph4_data = vehicle_material_stock_global.loc[idx['Aluminium',vehicle_list],years_select].sum(axis=0, level=1).reindex(vehicle_list)  # Aluminium by vehicle

# graph3_data_in  = vehicle_materials.loc[idx[:,:,'inflow',:,:,'Steel',:],years_select].sum(axis=0, level=5).loc['Steel']
# graph3_data_out = vehicle_materials.loc[idx[:,:,'outflow',:,:,'Steel',:],years_select].sum(axis=0, level=5).loc['Steel']

# graph4_data_in  = vehicle_materials.loc[idx[:,:,'inflow',:,:,'Aluminium',:],years_select].sum(axis=0, level=5).loc['Aluminium']
# graph4_data_out = vehicle_materials.loc[idx[:,:,'outflow',:,:,'Aluminium',:],years_select].sum(axis=0, level=5).loc['Aluminium']

# graph3_data_in[[2006,2007,2008]].mean() # quick comparison to Cullen & Alwood 2012
# graph4_data_in[[2006,2007,2008]].mean() # quick comparison to Cullen & Alwood 2012

# material_colorset = [color_material[i] for i in list(graph2_data.index)]

# plt.figure()
# fig, ((ax1, ax2),(ax3, ax4)) = plt.subplots(2, 2, figsize=(17,15), frameon=True)
# fig.suptitle('Global Material Stocks & Flows in Vehicles', y=0.95, fontsize=16)
# plt.subplots_adjust(wspace = 0.22, bottom = 0.1)
# ax3nd = ax3.twinx()
# ax4nd = ax4.twinx()

# ax1.set_title('Total Vehicle Body Weight (Stock), by Vehicle', fontsize=12)
# ax1.set_ylabel('Total Vehicle Weight (kg)', rotation='vertical', y=0.75, fontsize=13)
# ax1.yaxis.set_ticks_position('left')
# ax1.xaxis.set_ticks_position('bottom')
# ax1.margins(x=0)
# ax1.tick_params(axis='both', which='major', labelsize=12)
# ax1.stackplot(years_select, graph1_data, labels=list(graph1_data.index), colors=colors_vehicles)
# handles, labels = ax1.get_legend_handles_labels()
# ax1.legend(reversed(handles), reversed(labels), bbox_to_anchor=(0, 1), loc=2, ncol=1, frameon=False, fontsize=13)

# ax2.set_title('Total Vehicle Body Weight (Stock), by Material', fontsize=12)
# ax2.set_ylabel('Total Vehicle Weight (kg)', rotation='vertical', y=0.75, fontsize=13)
# ax2.yaxis.set_ticks_position('left')
# ax2.xaxis.set_ticks_position('bottom')
# ax2.margins(x=0)
# ax2.tick_params(axis='both', which='major', labelsize=12)
# ax2.stackplot(years_select, graph2_data, labels=list(graph2_data.index), colors=material_colorset)
# handles, labels = ax2.get_legend_handles_labels()
# ax2.legend(reversed(handles), reversed(labels), bbox_to_anchor=(0, 1), loc=2, ncol=1, frameon=False, fontsize=13)

# ax3.set_title('Global Steel Stocks and Flows in Vehicles', fontsize=12)
# ax3.set_ylabel('Vehicle Steel Stock (kg)', rotation='vertical', y=0.78, fontsize=13)
# ax3nd.set_ylabel('Vehicle Steel Flows (kg/yr)', rotation='vertical', x=0.1, y=0.75, fontsize=13)
# ax3.yaxis.set_ticks_position('left')
# ax3.xaxis.set_ticks_position('bottom')
# ax3.margins(x=0)
# ax3nd.margins(x=0)
# ax3.tick_params(axis='both', which='major', labelsize=12)
# ax3nd.tick_params(axis='both', which='major', labelsize=12)
# ax3.stackplot(years_select, graph3_data, labels=list(graph3_data.index), colors=colors_vehicles)
# ax3nd.plot(years_select, graph3_data_in,  '--', color='black', linewidth=3, label="Total Inflow") 
# ax3nd.plot(years_select, graph3_data_out, '--', color='red',   linewidth=3, label="Total Outflow") 
# handles, labels = ax3.get_legend_handles_labels()
# ax3.legend(reversed(handles), reversed(labels), bbox_to_anchor=(0, 1), loc=2, ncol=1, frameon=False, fontsize=13)
# ax3nd.legend(bbox_to_anchor=(1, 0), loc='lower right', ncol=1, frameon=False, fontsize=13)

# ax4.set_title('Global Aluminium Stocks and Flows in Vehicles', fontsize=12)
# ax4.set_ylabel('Vehicle Aluminium Stock (kg)', rotation='vertical', y=0.75, fontsize=13)
# ax4nd.set_ylabel('Vehicle Aluminium Flows (kg/yr)', rotation='vertical', x=0.1, y=0.72, fontsize=13)
# ax4.yaxis.set_ticks_position('left')
# ax4.xaxis.set_ticks_position('bottom')
# ax4.margins(x=0)
# ax4nd.margins(x=0)
# ax4.tick_params(axis='both', which='major', labelsize=12)
# ax4nd.tick_params(axis='both', which='major', labelsize=12)
# ax4.stackplot(years_select, graph4_data, labels=list(graph4_data.index), colors=colors_vehicles)
# ax4nd.plot(years_select, graph4_data_in,  '--', color='black', linewidth=3, label="Total Inflow") 
# ax4nd.plot(years_select, graph4_data_out, '--', color='red',   linewidth=3, label="Total Outflow") 
# handles, labels = ax4.get_legend_handles_labels()
# ax4.legend(reversed(handles), reversed(labels), bbox_to_anchor=(0, 1), loc=2, ncol=1, frameon=False, fontsize=13)
# ax4nd.legend(bbox_to_anchor=(1, 0), loc='lower right', ncol=1, frameon=False, fontsize=13)

# plt.show()
# plt.savefig('output\\' + FOLDER + '\\graphs\\vehicle_material_results_panel.png', dpi=600)

# #%% Figures on results
# # Plot panel (1 * 2) 3 regional Stock development & consequences i.t.o. inflow/outflow
# import matplotlib.ticker as ticker
# years_average_now   = [2016,2017,2018,2019,2020] 
# years_average_end   = [OUT_YEAR -4,OUT_YEAR -3,OUT_YEAR -2,OUT_YEAR -1,OUT_YEAR] 

# material_select = 'Steel'
# vehcile_list       = ['bicycle', 'rail_reg', 'rail_hst', 'reg_bus', 'midi_bus']           #only those vehicles which go negative by the end of the scenario (may add: 'sea_shipping_small', 'sea_shipping_med', 'sea_shipping_large', 'sea_shipping_vl')
# regions_developed  = [1,2,3,11,12,13,14,15,16,17,24]
# regions_developing = [4,5,6,7,8,9,10,18,19,21,22,25,26]
# regions_chin_jap   = [20,23]
                      
# #apply vehcile grouping first (same as panel above)
# material_flows_grouped = vehicle_materials.loc[idx[:,:,['inflow','outflow'],:,:, material_select,vehcile_list],:].sum(axis=0, level=[2,3,6]) # vehicle_materials in kg
# for flow in list(material_flows_grouped.index.levels[0]):
#    for region in list(material_flows_grouped.index.levels[1]):
#       #material_flows_grouped.loc[(flow, region,'Cars'),:]   = material_flows_grouped.loc[idx[flow, region,['ICE','HEV','PHEV','BEV','FCV']],:].sum(axis=0)
#       material_flows_grouped.loc[(flow, region,'Trains'),:] = material_flows_grouped.loc[idx[flow, region,['rail_reg','rail_hst']],:].sum(axis=0)
#       #material_flows_grouped.loc[(flow, region,'Trucks'),:] = material_flows_grouped.loc[idx[flow, region,['LCV','MFT','HFT']],:].sum(axis=0)
#       material_flows_grouped.loc[(flow, region,'Buses'),:]  = material_flows_grouped.loc[idx[flow, region,['reg_bus','midi_bus']],:].sum(axis=0)
#       #material_flows_grouped.loc[(flow, region,'Ships'),:]  = material_flows_grouped.loc[idx[flow, region,['inland_shipping', 'sea_shipping_small', 'sea_shipping_med','sea_shipping_large', 'sea_shipping_vl']],:].sum(axis=0)

# material_flows_grouped = material_flows_grouped.rename(index={"bicycle" : "Bicycles", "air_pas": "Planes", "rail_freight": "Rail Cargo", "air_freight" : "Air Cargo"})

# # pre-calculate & assign plot data
# in_steel_developed_now   = material_flows_grouped.loc[idx['inflow', regions_developed,  vehicle_list], years_average_now].sum(axis=0, level=2).mean(axis=1)
# in_steel_developing_now  = material_flows_grouped.loc[idx['inflow', regions_developing, vehicle_list], years_average_now].sum(axis=0, level=2).mean(axis=1)
# in_steel_chin_jap_now    = material_flows_grouped.loc[idx['inflow', regions_chin_jap,   vehicle_list], years_average_now].sum(axis=0, level=2).mean(axis=1)

# out_steel_developed_now  = material_flows_grouped.loc[idx['outflow', regions_developed,  vehicle_list], years_average_now].sum(axis=0, level=2).mean(axis=1)
# out_steel_developing_now = material_flows_grouped.loc[idx['outflow', regions_developing, vehicle_list], years_average_now].sum(axis=0, level=2).mean(axis=1)
# out_steel_chin_jap_now   = material_flows_grouped.loc[idx['outflow', regions_chin_jap,   vehicle_list], years_average_now].sum(axis=0, level=2).mean(axis=1)

# in_steel_developed_end   = material_flows_grouped.loc[idx['inflow', regions_developed,  vehicle_list], years_average_end].sum(axis=0, level=2).mean(axis=1)
# in_steel_developing_end  = material_flows_grouped.loc[idx['inflow', regions_developing, vehicle_list], years_average_end].sum(axis=0, level=2).mean(axis=1)
# in_steel_chin_jap_end    = material_flows_grouped.loc[idx['inflow', regions_chin_jap,   vehicle_list], years_average_end].sum(axis=0, level=2).mean(axis=1)

# out_steel_developed_end  = material_flows_grouped.loc[idx['outflow', regions_developed,  vehicle_list], years_average_end].sum(axis=0, level=2).mean(axis=1)
# out_steel_developing_end = material_flows_grouped.loc[idx['outflow', regions_developing, vehicle_list], years_average_end].sum(axis=0, level=2).mean(axis=1)
# out_steel_chin_jap_end   = material_flows_grouped.loc[idx['outflow', regions_chin_jap,   vehicle_list], years_average_end].sum(axis=0, level=2).mean(axis=1)

# diff_steel_developed_now  = in_steel_developed_now - out_steel_developed_now
# diff_steel_developing_now = in_steel_developing_now - out_steel_developing_now
# diff_steel_chin_jap_now   = in_steel_chin_jap_now - out_steel_chin_jap_now

# diff_steel_developed_end  = in_steel_developed_end - out_steel_developed_end
# diff_steel_developing_end = in_steel_developing_end - out_steel_developing_end
# diff_steel_chin_jap_end   = in_steel_chin_jap_end - out_steel_chin_jap_end


# plotvar_bus = [diff_steel_developed_now['Buses'],diff_steel_developing_now['Buses'], diff_steel_chin_jap_now['Buses'],                0, diff_steel_developed_end['Buses'], diff_steel_developing_end['Buses'], diff_steel_chin_jap_end['Buses']]
# plotvar_cyc = [diff_steel_developed_now['Bicycles'],diff_steel_developing_now['Bicycles'], diff_steel_chin_jap_now['Bicycles'],       0, diff_steel_developed_end['Bicycles'], diff_steel_developing_end['Bicycles'], diff_steel_chin_jap_end['Bicycles']]
# #plotvar_car = [diff_steel_developed_now['Cars'],diff_steel_developing_now['Cars'], diff_steel_chin_jap_now['Cars'],                   0, diff_steel_developed_end['Cars'], diff_steel_developing_end['Cars'], diff_steel_chin_jap_end['Cars']]
# #plotvar_trk = [diff_steel_developed_now['Trucks'],diff_steel_developing_now['Trucks'], diff_steel_chin_jap_now['Trucks'],             0, diff_steel_developed_end['Trucks'], diff_steel_developing_end['Trucks'], diff_steel_chin_jap_end['Trucks']]
# plotvar_trn = [diff_steel_developed_now['Trains'],diff_steel_developing_now['Trains'], diff_steel_chin_jap_now['Trains'],             0, diff_steel_developed_end['Trains'], diff_steel_developing_end['Trains'], diff_steel_chin_jap_end['Trains']]
# #plotvar_rlc = [diff_steel_developed_now['Rail Cargo'],diff_steel_developing_now['Rail Cargo'], diff_steel_chin_jap_now['Rail Cargo'], 0, diff_steel_developed_end['Rail Cargo'], diff_steel_developing_end['Rail Cargo'], diff_steel_chin_jap_end['Rail Cargo']]
# #plotvar_shp = [diff_steel_developed_now['Ships'],diff_steel_developing_now['Ships'], diff_steel_chin_jap_now['Ships'],                0, diff_steel_developed_end['Ships'], diff_steel_developing_end['Ships'], diff_steel_chin_jap_end['Ships']]

# #split the data into positive & negative bars
# def pos_neg(data):
#    data_pos = []
#    data_neg = []
#    for item in range(0,len(data)):
#       if data[item] > 0:
#          data_pos.append(data[item])
#          data_neg.append(0)
#       else:
#          data_neg.append(data[item])
#          data_pos.append(0)
#    return data_pos, data_neg

# plotvar_bus_pos, plotvar_bus_neg = pos_neg(plotvar_bus)
# plotvar_cyc_pos, plotvar_cyc_neg = pos_neg(plotvar_cyc)
# #plotvar_car_pos, plotvar_car_neg = pos_neg(plotvar_car)
# #plotvar_trk_pos, plotvar_trk_neg = pos_neg(plotvar_trk)
# plotvar_trn_pos, plotvar_trn_neg = pos_neg(plotvar_trn)
# #plotvar_rlc_pos, plotvar_rlc_neg = pos_neg(plotvar_rlc)
# #plotvar_shp_pos, plotvar_shp_neg = pos_neg(plotvar_shp)

# barWidth = 0.5
# r1 = np.arange(7)

# plt.figure(figsize=(8, 6))
# fig, ax = plt.subplots(figsize=(8,6))
# plt.rc('font', **font)
# plt.yticks(fontsize=10)
# plt.subplots_adjust(wspace = 0.1, bottom = 0.15, right = 0.80)
# ax.set_ylim(ymin=-300000002, ymax=1400000002)
# ax.yaxis.set_major_locator(ticker.MultipleLocator(100000000))
# ax.bar(r1, plotvar_bus_pos, color=veh_colors_dict["Buses"],    width=barWidth, edgecolor=None, label='2018')
# ax.bar(r1, plotvar_bus_neg, color=veh_colors_dict["Buses"],    width=barWidth, edgecolor=None, label='2018')
# bottom = plotvar_bus_pos
# floor = plotvar_bus_neg
# ax.bar(r1, plotvar_cyc_pos, color=veh_colors_dict["Bicycles"], width=barWidth, bottom=bottom, edgecolor=None, label='2018')
# ax.bar(r1, plotvar_cyc_neg, color=veh_colors_dict["Bicycles"], width=barWidth, bottom=floor,  edgecolor=None, label='2018')
# bottom = [a+b for a,b in zip(bottom,plotvar_cyc_pos)]
# floor =  [a+b for a,b in zip(floor, plotvar_cyc_neg)]
# #ax.bar(r1, plotvar_car_pos, color=veh_colors_dict["Cars"],     width=barWidth, bottom=bottom, edgecolor=None, label='2018')
# #ax.bar(r1, plotvar_car_neg, color=veh_colors_dict["Cars"],     width=barWidth, bottom=floor,  edgecolor=None, label='2018')
# #bottom = [a+b for a,b in zip(bottom,plotvar_car_pos)]
# #floor =  [a+b for a,b in zip(floor, plotvar_car_neg)]
# ax.bar(r1, plotvar_trn_pos,  color=veh_colors_dict["Trains"],  width=barWidth, bottom=bottom, edgecolor=None, label='2018')
# ax.bar(r1, plotvar_trn_neg,  color=veh_colors_dict["Trains"],  width=barWidth, bottom=floor, edgecolor=None, label='2018')
# bottom = [a+b for a,b in zip(bottom,plotvar_trn_pos)]
# floor =  [a+b for a,b in zip(floor, plotvar_trn_neg)]
# #ax.bar(r1, plotvar_trk_pos, color=veh_colors_dict["Trucks"],   width=barWidth, bottom=bottom, edgecolor=None, label='2018')
# #ax.bar(r1, plotvar_trk_neg, color=veh_colors_dict["Trucks"],   width=barWidth, bottom=floor, edgecolor=None, label='2018')
# #bottom = [a+b for a,b in zip(bottom,plotvar_trk_pos)]
# #floor  = [a+b for a,b in zip(floor, plotvar_trk_neg)]
# #ax.bar(r1, plotvar_rlc_pos, color=veh_colors_dict["Rail Cargo"], width=barWidth, bottom=bottom, edgecolor=None, label='2018')
# #ax.bar(r1, plotvar_rlc_neg, color=veh_colors_dict["Rail Cargo"], width=barWidth, bottom=floor,  edgecolor=None, label='2018')
# #bottom = [a+b for a,b in zip(bottom,plotvar_rlc_pos)]
# #floor =  [a+b for a,b in zip(floor, plotvar_rlc_neg)]
# #ax.bar(r1, plotvar_shp_pos,    color=veh_colors_dict["Ships"],   width=barWidth, bottom=bottom, edgecolor=None, label='2018')
# #ax.bar(r1, plotvar_shp_neg,    color=veh_colors_dict["Ships"],   width=barWidth, bottom=floor, edgecolor=None, label='2018')
# plt.plot([-0.5, 6.5], [0, 0], 'k-', lw=0.5)

# #Added text
# ax.text(1.1, -500000000, "Current (2020)", ha="center", va="center", rotation=0, size=10)
# ax.text(5, -500000000, str(OUT_YEAR),           ha="center", va="center", rotation=0, size=10)
# bbox_props = dict(boxstyle="rarrow", fc=(0.8, 0.9, 0.9), ec="g", lw=0.5)
# ax.text(3, 55000000,  "Sink",   ha="center", va="center", rotation=90,  size=6, bbox=bbox_props)
# ax.text(3, -95000000, "Source", ha="center", va="center", rotation=-90, size=6, bbox=bbox_props)

# # Add xticks on the middle of the group bars
# plt.xticks([r + barWidth - 0.5 for r in range(7)], ['Steady', 'Developing', 'China+Japan', '', 'Steady', 'Developing', 'China+Japan'], fontsize=8)
# plt.ylabel('Net additions to vehicle stocks (kg steel)', fontsize=11, y=0.5)
# plt.title('Net additions of steel to selected in-use vehicle stocks', y=1.08, fontsize=14)


# # Legend & text
# legend_elements = [#matplotlib.patches.Patch(facecolor=veh_colors_dict["Ships"],       label='Ships'),
#                    #matplotlib.patches.Patch(facecolor=veh_colors_dict["Rail Cargo"],  label='Rail Cargo'),
#                    #matplotlib.patches.Patch(facecolor=veh_colors_dict["Trucks"],      label='Trucks'),
#                    matplotlib.patches.Patch(facecolor=veh_colors_dict["Trains"],      label='Trains'),                
#                    #matplotlib.patches.Patch(facecolor=veh_colors_dict["Cars"],        label='Cars'),
#                    matplotlib.patches.Patch(facecolor=veh_colors_dict["Bicycles"],    label='Bicycles'),
#                    matplotlib.patches.Patch(facecolor=veh_colors_dict["Buses"],       label='Buses')]
# plt.legend(handles=legend_elements, loc=2, bbox_to_anchor=(1,1), ncol=1, frameon=False, fontsize=13)
# fig.savefig('output\\' + FOLDER + '\\graphs\\bars_source_sink_steel.png', dpi=600)
# plt.show()

# #%% Plot on the materials in-use  by vecihle, per tkm or pkm provided (each year)
# period = [2015, 2016, 2017, 2018, 2019, 2020]

# #pre-calculate total weight of vehicles (accounting for different vehcile sub-types)
# cars_weight = 0
# rbus_weight = 0
# mibs_weight = 0
# lcvt_weight = 0
# trck_weight = 0

# for veh_type in ['ICE','HEV','PHEV','BEV','FCV']:
#    cars_weight = cars_weight + car_total_nr[region_list].mul(vehicleshare_cars[veh_type].unstack()).loc[period,:].sum(axis=1).mean()   * vehicle_weight_kg_car[veh_type].loc[period].mean() # MIND for the time being, the non-weighted average weight is used here
#    lcvt_weight = lcvt_weight + trucks_LCV_nr[region_list].mul(trucks_MFT_vshares[veh_type].unstack()).loc[period,:].sum(axis=1).mean() * vehicle_weight_kg_LCV[veh_type].loc[period].mean() # LCV uses MFT vshares
#    trck_weight = trck_weight + trucks_HFT_nr[region_list].mul(trucks_HFT_vshares[veh_type].unstack()).loc[period,:].sum(axis=1).mean() * vehicle_weight_kg_HFT[veh_type].loc[period].mean()
#    trck_weight = trck_weight + trucks_MFT_nr[region_list].mul(trucks_MFT_vshares[veh_type].unstack()).loc[period,:].sum(axis=1).mean() * vehicle_weight_kg_MFT[veh_type].loc[period].mean() # MFT is summed with HFT

# for veh_type in ['ICE','HEV','PHEV','BEV','FCV']:
#    rbus_weight = rbus_weight + bus_regl_nr[region_list].mul(buses_regl_vshares[veh_type].unstack()).loc[period,:].sum(axis=1).mean() * vehicle_weight_kg_bus[veh_type].loc[period].mean() 
#    mibs_weight = mibs_weight + bus_midi_nr[region_list].mul(buses_midi_vshares[veh_type].unstack()).loc[period,:].sum(axis=1).mean() * vehicle_weight_kg_midi[veh_type].loc[period].mean() 

# bike_weight = bikes_nr.loc[period, region_list].sum(axis=1).mean()        * vehicle_weight_kg_bicycle.loc[period, "bicycle"].mean()
# airp_weight = air_pas_nr.loc[period, region_list].sum(axis=1).mean()      * vehicle_weight_kg_air_pas.loc[period, "air_pas"].mean()
# trai_weight = rail_reg_nr.loc[period, region_list].sum(axis=1).mean()     * vehicle_weight_kg_rail_reg.loc[period, "rail_reg"].mean()
# hstr_weight = rail_hst_nr.loc[period, region_list].sum(axis=1).mean()     * vehicle_weight_kg_rail_hst.loc[period, "rail_hst"].mean()
# airc_weight = air_freight_nr.loc[period, region_list].sum(axis=1).mean()  * vehicle_weight_kg_air_frgt.loc[period, "air_freight"].mean()
# rlfr_weight = rail_freight_nr.loc[period, region_list].sum(axis=1).mean() * vehicle_weight_kg_rail_frgt.loc[period, "rail_freight"].mean()
# ship_weight = (inland_ship_nr.loc[period, region_list].sum(axis=1).mean() * vehicle_weight_kg_inland_ship.loc[period, "inland_shipping"].mean()) + (ship_small_nr.loc[period, region_list].sum(axis=1).mean()  * weight_boats.loc[period,'Small'].mean()) +   (ship_medium_nr.loc[period, region_list].sum(axis=1).mean()  * weight_boats.loc[period,'Medium'].mean())  +  (ship_large_nr.loc[period, region_list].sum(axis=1).mean()  * weight_boats.loc[period,'Large'].mean())     + (ship_vlarge_nr.loc[period, region_list].sum(axis=1).mean()  * weight_boats.loc[period,'Very Large'].mean())         

# matint_cars = cars_weight / (car_pkms.loc[period,:].sum(axis=1).mean() * 1000000000000)
# matint_lcvt = lcvt_weight / (trucks_LCV_tkm.loc[period, region_list].sum(axis=1).mean() * 1000000)
# matint_trck = trck_weight / (trucks_MFT_tkm.loc[period, region_list].sum(axis=1).mean() * 1000000 + trucks_HFT_tkm.loc[period, region_list].sum(axis=1).mean() * 1000000)
# matint_rbus = rbus_weight / (bus_regl_pkms.loc[period,:].sum(axis=1).mean() * 1000000000000)
# matint_mibs = mibs_weight / (bus_midi_pkms.loc[period,:].sum(axis=1).mean() * 1000000000000)
# matint_bike = bike_weight / (passengerkms_Tpkms['biking'].unstack().loc[period, region_list].sum(axis=1).mean() * 1000000000000)
# matint_airp = airp_weight / (passengerkms_Tpkms['air'].unstack().loc[period, region_list].sum(axis=1).mean() * 1000000000000)
# matint_trai = trai_weight / (passengerkms_Tpkms['train'].unstack().loc[period, region_list].sum(axis=1).mean() * 1000000000000)
# matint_hstr = hstr_weight / (passengerkms_Tpkms['hst'].unstack().loc[period, region_list].sum(axis=1).mean() * 1000000000000)
# matint_airc = airc_weight / (air_freight_tkms.loc[period, region_list].sum(axis=1).mean() * 1000000)
# matint_rlfr = rlfr_weight / (tonkms_Mtkms['freight train'].unstack().loc[period, region_list].sum(axis=1).mean() * 1000000)
# matint_ship = ship_weight / (tonkms_Mtkms['international shipping'].unstack().loc[period,region_list].sum(axis=1).mean() * 1000000 + tonkms_Mtkms['inland shipping'].unstack().loc[period,region_list].sum(axis=1).mean() * 1000000)

# mul = 1000
# plotvar_matint  = [matint_cars * mul,   matint_bike * mul, matint_mibs * mul, matint_rbus * mul, matint_airp * mul, matint_trai * mul, matint_hstr * mul, 0, matint_airc * mul, matint_rlfr * mul, matint_ship * mul, matint_lcvt * mul, matint_trck * mul]

# barWidth = 0.5
# r1 = np.arange(13)

# plt.figure(figsize=(20, 6))
# fig, ax = plt.subplots(figsize=(20,6))
# plt.rc('font', **font)
# plt.yticks(fontsize=16)
# plt.subplots_adjust(wspace = 0.1, bottom = 0.15, right = 0.80)
# ax.set_ylim(ymin=0, ymax=100)
# ax.bar(r1, plotvar_matint,     color=color_material["Steel"],     width=barWidth, edgecolor=None, label='2018')
# bbox_props = dict(boxstyle="rarrow", fc=(0.8, 0.9, 0.9), ec="b", lw=0.5)
# ax.text(11, 100,  "180",   ha="center", va="center", rotation=90,  size=10, bbox=bbox_props)

# # Add xticks on the middle of the group bars
# plt.xticks([r + barWidth - 0.5 for r in range(13)], ['Cars', 'Bicycles', 'Midi Bus', 'Bus', 'Airplane', 'Rail', 'High Speed Train', '', 'Air Cargo', 'Rail Cargo', 'Ships', 'Light Truck', 'Other Trucks'], fontsize=11)
# plt.ylabel('Vehicle weight intensity (gram/pkm or gram/tkm)', fontsize=14, x=-0.1, y=0.5)
# plt.title('Stock weight intensity of different vehicles (in-use stock weight per unit of annual transport demand)', y=1.08, fontsize=14)

# fig.savefig('output\\' + FOLDER + '\\graphs\\bars_vehicle_material_intensity.png', dpi=600)
# plt.show()

# #%% Figures on results
# # Plot panel (1 * 2) Basic stacked area chart: REgional stock of battery (by weight) and the corresponding regional in/outflow 
# region_colorset = ['#682c0e','#c24914','#fc8621']
# years_select_narrow  = list(range(2000,OUT_YEAR + 1))
# years_select_broad   = list(range(1995,OUT_YEAR + 6))

# # pre-process & assign plot data
# graph1_data     = pd.DataFrame(index=years_select, columns=['Steady','Developing','China+Japan'])
# graph2_data_in  = pd.DataFrame(index=years_select, columns=['Steady','Developing','China+Japan'])
# graph2_data_out = pd.DataFrame(index=years_select, columns=['Steady','Developing','China+Japan'])

# graph1_data['Steady']         = battery_weight_regional_stock.loc[years_select_narrow, regions_developed].sum(axis=1)
# graph1_data['Developing']     = battery_weight_regional_stock.loc[years_select_narrow, regions_developing].sum(axis=1)
# graph1_data['China+Japan']    = battery_weight_regional_stock.loc[years_select_narrow, regions_chin_jap].sum(axis=1)

# graph2_data_in['Steady']      = battery_weight_regional_in.loc[years_select_narrow, regions_developed].sum(axis=1).rolling(window=5).mean().loc[years_select_narrow]
# graph2_data_in['Developing']  = battery_weight_regional_in.loc[years_select_narrow, regions_developing].sum(axis=1).rolling(window=5).mean().loc[years_select_narrow]
# graph2_data_in['China+Japan'] = battery_weight_regional_in.loc[years_select_narrow, regions_chin_jap].sum(axis=1).rolling(window=5).mean().loc[years_select_narrow]

# graph2_data_out['Steady']      = battery_weight_regional_out.loc[years_select_narrow, regions_developed].sum(axis=1).rolling(window=5).mean().loc[years_select_narrow]
# graph2_data_out['Developing']  = battery_weight_regional_out.loc[years_select_narrow, regions_developing].sum(axis=1).rolling(window=5).mean().loc[years_select_narrow]
# graph2_data_out['China+Japan'] = battery_weight_regional_out.loc[years_select_narrow, regions_chin_jap].sum(axis=1).rolling(window=5).mean().loc[years_select_narrow]


# plt.figure()
# fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(15,6), frameon=True)
# fig.suptitle('Regional EV Battery Stock & Flows', y=1, fontsize=16)
# plt.subplots_adjust(wspace = 0.25, bottom = 0.15)

# ax1.set_title('Battery in-use stocks by regional group', fontsize=12)
# ax1.set_ylabel('weight of battery stocks (kg)', rotation='vertical', y=0.65, fontsize=13)
# ax1.yaxis.set_ticks_position('left')
# ax1.xaxis.set_ticks_position('bottom')
# ax1.margins(x=0)
# ax1.stackplot(years_select, graph1_data.T, labels=list(graph1_data.columns), colors=region_colorset)
# handles, labels = ax1.get_legend_handles_labels()
# ax1.legend(reversed(handles), reversed(labels), bbox_to_anchor=(0, -0.15), loc='lower left', ncol=4, frameon=False, fontsize=10)

# ax2.set_title('Battery inflow/outflow by regional group', fontsize=12)
# ax2.set_ylabel('weight of battery flows (kg/yr)', rotation='vertical', y=0.65, fontsize=13)
# ax2.yaxis.set_ticks_position('left')
# ax2.xaxis.set_ticks_position('bottom')
# ax2.margins(x=0)
# ax2.plot(years_select, graph2_data_in['Steady'],       label='Steady (inflow)',    color=region_colorset[0])
# ax2.plot(years_select, graph2_data_out['Steady'],      '--', label='Steady (outflow)',    color=region_colorset[0])
# ax2.plot(years_select, graph2_data_in['Developing'],   label='Developing (inflow)', color=region_colorset[1])
# ax2.plot(years_select, graph2_data_out['Developing'],  '--', label='Developing (outflow)', color=region_colorset[1])
# ax2.plot(years_select, graph2_data_in['China+Japan'],  label='China+Japan (inflow)', color=region_colorset[2])
# ax2.plot(years_select, graph2_data_out['China+Japan'], '--', label='China+Japan (outflow)', color=region_colorset[2])
# handles, labels = ax2.get_legend_handles_labels()
# ax2.legend(reversed(handles), reversed(labels), bbox_to_anchor=(-0.05, -0.20), loc='lower left', ncol=3, frameon=False, fontsize=9)

# plt.savefig('output\\' + FOLDER + '\\graphs\\battery_market_panel_regional_stock_results.jpg', dpi=600)
# plt.show()
