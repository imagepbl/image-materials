# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 18:11:03 2020
@author: Sebastiaan Deetman (deetman@cml.leidenuniv.nl)
with contributions from Rombout Huisman 
"""

import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

os.chdir("C:\\Users\\admin\\Documents\\GitHub\\VEMA")  # SET YOUR PATH HERE
from dynamic_stock_model import DynamicStockModel as DSM
from past.builtins import execfile
execfile('read_mym.py')                                     # in-line execution of a function to read MyM files (function: read_mym_df)

# settings & constants
start_year = 1971
end_year = 2100
regions = 26
idx = pd.IndexSlice             # needed for slicing multi-index
first_year_boats = 1900

# scenario settings
scen    = 'SSP2'
variant = 'BL'
folder  = scen + '_' + variant

#%% reading in all the first csv files containing vehicle data (load, market shares, lifetimes)
mileages                = pd.read_csv('vehicle_data/mileages_km_per_year.csv')              # Km/year of all the vehicles
load                    = pd.read_csv('vehicle_data/load_pass_and_tonnes.csv')              # Load in terms of maximum number of passengers or kg of good
loadfactor              = pd.read_csv('vehicle_data/loadfactor_percentages.csv')            # Percentage of the maximum load that is on average 
market_share            = pd.read_csv('vehicle_data/fraction_tkm_pkm.csv')                  # Percentage of tonne-/passengerkilometres
first_year_vehicle      = pd.read_csv('vehicle_data/first_year_vehicle.csv')                # first year of operation per vehicle-type
lifetimes_vehicles      = pd.read_csv('vehicle_data/lifetimes_years.csv',  index_col=0)     # Average End-of-Life of vehicles in years, this file also contains the setting for the choice of distribution and other lifetime related settings (standard devition, or alternative parameterisation)
kilometrage             = pd.read_csv('vehicle_data/kilometrage.csv',      index_col='t')   # kilometrage of passenger cars in kms/yr
kilometrage_midi_bus    = pd.read_csv('vehicle_data/kilometrage_midi.csv', index_col='t')   # kilometrage of passenger cars in kms/yr
kilometrage_bus         = pd.read_csv('vehicle_data/kilometrage_bus.csv',  index_col='t')   # kilometrage of passenger cars in kms/yr

# kilometrage is defined untill 2018, fill 2018 values untill 2100 
for row in range(2019,2101):
    kilometrage.loc[row]          = kilometrage.loc[2018].values
    kilometrage_bus.loc[row]      = kilometrage_bus.loc[2018].values
    kilometrage_midi_bus.loc[row] = kilometrage_midi_bus.loc[2018].values

for row in range(1971,2010):
    kilometrage_bus.loc[row]      = kilometrage_bus.loc[2018].values
    kilometrage_midi_bus.loc[row] = kilometrage_midi_bus.loc[2018].values

kilometrage_bus      = kilometrage_bus.sort_index()
kilometrage_midi_bus = kilometrage_midi_bus.sort_index()
region_list          = list(kilometrage.columns.values)     # get a list with region names

# weight and materials related data
vehicle_weight_kg_simple       = pd.read_csv('vehicle_data/vehicle_weight_kg_simple.csv',   index_col=0)       # Weight of a single vehicle of each type in kg
vehicle_weight_kg_typical      = pd.read_csv('vehicle_data/vehicle_weight_kg_typical.csv',  index_col=[0,1])   # Weight of a single vehicle of each type in kg
material_fractions             = pd.read_csv('vehicle_data/material_fractions_simple.csv',  index_col=[0,1])   # Material fractions in percentages
material_fractions_type        = pd.read_csv('vehicle_data/material_fractions_typical.csv', index_col=[0,1], header=[0,1])   # Material fractions in percentages, by vehicle sub-type
battery_weights                = pd.read_csv('vehicle_data/battery_weights_kg.csv',         index_col=[0,1])   # Using the 250 Wh/kg on the kWh of the various batteries a weight (in kg) of the battery per vehicle category is determined
battery_materials              = pd.read_csv('vehicle_data/battery_materials.csv',          index_col=[0,1])   # The material fraction of storage technologies (used to get the vehicle battery composition)
battery_shares_full            = pd.read_csv('vehicle_data/battery_share_inflow.csv',       index_col=0)       # The share of the battery market (8 battery types used in vehicles), this data is based on a Multi-Nomial-Logit market model & costs in https://doi.org/10.1016/j.resconrec.2020.105200 - since this is scenario dependent it's placed under the 'IMAGE' scenario folder

#%% Reading & preparing all the CSV & IMAGE files for vehicle categories and material fractions

# IMAGE scenario files (total demand in Tkms & Pkms + vehicle shares)
tonkms_Ttkms        = read_mym_df('IMAGE_files/' + folder + '/mym/trp_frgt_Tkm.out')              # The tonne kilometres of freight vehicles of the IMAGE/TIMER SSP2 (in Tera Tkm)
passengerkms_Tpkms  = read_mym_df('IMAGE_files/' + folder + '/mym/trp_trvl_pkm.out')              # The passenger kilometres from the IMAGE/TIMER SSP2 (in Tera Pkm)
buses_vshares       = read_mym_df('IMAGE_files/' + folder + '/mym/trp_trvl_Vshare_bus.out')       # The vehicle shares of buses of the SSP2                            MIND! FOR the BL this is still the OLD SSP2 file REPLACE LATER
car_vshares         = read_mym_df('IMAGE_files/' + folder + '/mym/trp_trvl_Vshare_car.out')       # The vehicle shares of passenger cars of the SSP2 
medtruck_vshares    = read_mym_df('IMAGE_files/' + folder + '/mym/trp_frgt_Vshare_MedTruck.out')  # The vehicle shares of trucks (medium) of the SSP2 
hvytruck_vshares    = read_mym_df('IMAGE_files/' + folder + '/mym/trp_frgt_Vshare_HvyTruck.out')  # The vehicle shares of trucks (heavy) of the SSP2 
loadfactor_car_data = read_mym_df('IMAGE_files/' + folder + '/mym/loadfactor.out')                # The loadfactor of passenger vehicles (occupation in nr of people/vehicle)

# select loadfactor for cars
car_loadfactor = loadfactor_car_data[['time','DIM_1', 5]].pivot_table(index='time', columns='DIM_1')  # loadfactor for cars (in persons per vehicle)
car_loadfactor = car_loadfactor.apply(lambda x: [y if y >= 1 else 1 for y in x])                      # To avoid car load (person/vehicle) values ever going below 1, replace all values below 1 with 1
car_loadfactor.columns = region_list

# Files related to the international shipping
nr_of_boats         = pd.read_csv('vehicle_data/ships/number_of_boats.csv', index_col='t').sort_index(axis=0)           # number of boats in the global merchant fleet (2005-2018)   changing Data by EQUASIS
cap_of_boats        = pd.read_csv('vehicle_data/ships/capacity_ton_boats.csv', index_col='t').sort_index(axis=0)        # boat capacity in tons                                      changing Data is a combination of EQUASIS Gross Tonnage and UNCTAD Dead-Weigh-Tonnage per Gross Tonnage
loadfactor_boats    = pd.read_csv('vehicle_data/ships/loadfactor_boats.csv', index_col='t').sort_index(axis=0)          # loadfactor of boats (fraction)                             fixed    Data is based on Ecoinvent report 14 on Transport (Table 8-19)
mileage_boats       = pd.read_csv('vehicle_data/ships/mileage_kmyr_boats.csv', index_col='t').sort_index(axis=0)        # mileage of boats in km/yr (per ship)                       fixed    Data is based on Ecoinvent report 14 on Transport (Table 8-19)
weight_boats        = pd.read_csv('vehicle_data/ships/weight_percofcap_boats.csv', index_col='t').sort_index(axis=0)    # weight of boats as a percentage of the capacity (%)        fixed    Data is based on Ecoinvent report 14 on Transport (section 8.4.1)

#set multi-index based on the frist two columns
tonkms_Ttkms.set_index(['time', 'DIM_1'], inplace=True)
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

# insert column descriptions
tonkms_Ttkms.columns = tkms_label
passengerkms_Tpkms.columns = pkms_label
medtruck_vshares.columns = truck_label
hvytruck_vshares.columns = truck_label
buses_vshares.columns = bus_label

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

# labels etc.
x_graphs        = [i for i in range(start_year,end_year,1)]                           # this is used as an x-axis for the years in graphs
labels_pas      = ['bicycle', 'rail_reg','rail_hst','midi_bus','reg_bus','air_pas','ICE','HEV','PHEV','BEV', 'FCV']             # Names used to shorten plots
labels_fre      = ['inland_shipping', 'rail_freight','LCV', 'MFT', 'HFT', 'air_freight', 'sea_shipping_small', 'sea_shipping_med', 'sea_shipping_large', 'sea_shipping_vl'] #names used to shorten plots
labels_materials= ['Steel', 'Aluminium', 'Cu', 'Plastics', 'Glass', 'Ti', 'Wood', 'Rubber', 'Li','Co','Ni', 'Mn','Nd','Pb']
labels_ev_batt  = ['NiMH','LMO','NMC','NCA','LFP','Lithium Sulfur','Lithium Ceramic','Lithium-air']

#%% For dynamic variables, apply interpolation and extend over the whole timeframe

# For some files, data is only found for a limited number of years, so we need to infer time-series before & after the data ends
def add_history_and_future(original, first_year, change='no'):  
   
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
        for row in range(end,end_year+1):
            history.loc[row] = history.loc[end].values      # fill new years with the same values as in the last year of the origial data 
            new = history.sort_index(axis=0)
    else:
        # if change = 'yes', first determine the growth rate based on the historic series (start-to-end)
        growthrate = history.pct_change().sum(axis=0)/(end-start)
        # then apply the average annual growth rate to the last historic year
        for row in range(end+1,end_year+1):
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
material_fractions_ship_small  = add_history_and_future(material_fractions['sea_shipping_small'].unstack(), first_year_boats,                                change='no')
material_fractions_ship_medium = add_history_and_future(material_fractions['sea_shipping_med'].unstack(),   first_year_boats,                                change='no')
material_fractions_ship_large  = add_history_and_future(material_fractions['sea_shipping_large'].unstack(), first_year_boats,                                change='no')
material_fractions_ship_vlarge = add_history_and_future(material_fractions['sea_shipping_vl'].unstack(),    first_year_boats,                                change='no')
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
battery_weights_full    = add_history_and_future(battery_weights.unstack(),     start_year)
battery_materials_full  = add_history_and_future(battery_materials.unstack(),   start_year)

#%% Caculating the tonnekilometres for all freight and passenger vehicle types (adjustments are made to: freight air, trucks, and buses)

# Trucks are calculated differently because the IMAGE model does not account for LCV trucks, which concerns a large portion of the material requirements of road freight
# the total trucks Tkms remain the same, but a LCV fraction is substracted, and the remainder is re-assigned to medium and heavy trucks according to their original ratio
trucks_total_tkm       = tonkms_Ttkms['medium truck'].unstack() +  tonkms_Ttkms['heavy truck'].unstack()
trucks_LCV_tkm         = trucks_total_tkm * 0.04                                    # 0.04 is the fraction of the tkms driven by light commercial vehicles according to the IEA
MFT_percshare_tkm      = tonkms_Ttkms['medium truck'].unstack() / trucks_total_tkm  # the MFT fraction of the total 
HFT_percshare_tkm      = tonkms_Ttkms['heavy truck'].unstack() / trucks_total_tkm   # the HFT fraction of the total 
trucks_min_LCV         = trucks_total_tkm - trucks_LCV_tkm
trucks_MFT_tkm         = trucks_min_LCV.mul(MFT_percshare_tkm)                      
trucks_HFT_tkm         = trucks_min_LCV.mul(HFT_percshare_tkm)

# demand for freight planes is reduced by 50% because about half of the air freight is transported as cargo on passenger planes 
air_freight_tkms       = tonkms_Ttkms['air cargo'].unstack() * market_share['air_freight'].values[0]

# Buses are adjusted to account for the higher material intensity of mini-buses
bus_regl_pkms          = passengerkms_Tpkms['bus'].unstack() * market_share['reg_bus'].values[0]   # in tera pkms
bus_midi_pkms          = passengerkms_Tpkms['bus'].unstack() * market_share['midi_bus'].values[0]  # in tera pkms

# Select tkms of passenger cars (which will be adjusted to represent 5 types: ICE, HEV, PHEV, BEV & FCV)
car_pkms               = passengerkms_Tpkms['car'].unstack()
car_pkms               = car_pkms.drop([27, 28], axis=1)    # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies                                       # in tera pkms
car_pkms.columns       = region_list                                  

#%% Calculate the NUMBER OF VEHICLES (stock, on the road) to fulfull the ton-kilometers transport demand

# this function translates ton kilometers (by year & by region) to nr of vehicles (same dimms), using fixed indicators on mileage, load capacity and load factor  
def tkms_to_nr_of_vehicles_fixed(tera_tkms, mileage, load, loadfactor):
    # first_translate Tera ton/person- kms into person/ton-kms
    tkms = tera_tkms * 1000000000000  
    # then get the vehicle kilometers required to fulfill the transport demand
    vkms = tkms/(load*loadfactor)
    # then get the number of vehicles by dividing by the mileage
    nr_of_vehicles = vkms/mileage
    return nr_of_vehicles

#calculate the number of vehicles on the road (first passenger, then freight)
# bus_regl_nr_old = tkms_to_nr_of_vehicles_fixed(bus_regl_pkms,                          mileages['reg_bus'].values[0],  load['reg_bus'].values[0],    loadfactor['reg_bus'].values[0])
# bus_midi_nr_old = tkms_to_nr_of_vehicles_fixed(bus_midi_pkms,                          mileages['midi_bus'].values[0], load['midi_bus'].values[0],   loadfactor['midi_bus'].values[0])
air_pas_nr      = tkms_to_nr_of_vehicles_fixed(passengerkms_Tpkms['air'].unstack(),    mileages['air_pas'].values[0],  load['air_pas'].values[0],    loadfactor['air_pas'].values[0])
rail_reg_nr     = tkms_to_nr_of_vehicles_fixed(passengerkms_Tpkms['train'].unstack(),  mileages['rail_reg'].values[0], load['rail_reg'].values[0],   loadfactor['rail_reg'].values[0])
rail_hst_nr     = tkms_to_nr_of_vehicles_fixed(passengerkms_Tpkms['hst'].unstack(),    mileages['rail_hst'].values[0], load['rail_hst'].values[0],   loadfactor['rail_hst'].values[0])
bikes_nr        = tkms_to_nr_of_vehicles_fixed(passengerkms_Tpkms['biking'].unstack(), mileages['bicycle'].values[0],  load['bicycle'].values[0],    loadfactor['bicycle'].values[0])

trucks_HFT_nr   = tkms_to_nr_of_vehicles_fixed(trucks_HFT_tkm/1000000,   mileages['HFT'].values[0],          load['HFT'].values[0],         loadfactor['HFT'].values[0])
trucks_MFT_nr   = tkms_to_nr_of_vehicles_fixed(trucks_MFT_tkm/1000000,   mileages['MFT'].values[0],          load['MFT'].values[0],         loadfactor['MFT'].values[0])
trucks_LCV_nr   = tkms_to_nr_of_vehicles_fixed(trucks_LCV_tkm/1000000,   mileages['LCV'].values[0],          load['LCV'].values[0],         loadfactor['LCV'].values[0])
air_freight_nr  = tkms_to_nr_of_vehicles_fixed(air_freight_tkms/1000000,  mileages['air_freight'].values[0], load['air_freight'].values[0], loadfactor['air_freight'].values[0])
rail_freight_nr = tkms_to_nr_of_vehicles_fixed(tonkms_Ttkms['freight train'].unstack()/1000000, mileages['rail_freight'].values[0], load['rail_freight'].values[0],         loadfactor['rail_freight'].values[0])
inland_ship_nr  = tkms_to_nr_of_vehicles_fixed(tonkms_Ttkms['inland shipping'].unstack()/1000000, mileages['inland_shipping'].values[0], load['inland_shipping'].values[0], loadfactor['inland_shipping'].values[0])

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

share_of_boats_yrs    =  add_history_and_future(share_of_boats   , first_year_boats,   change='yes')   # could be 'yes' based on better data
cap_of_boats_yrs      =  add_history_and_future(cap_of_boats     , first_year_boats,   change='no')    # could be 'yes' based on better data
loadfactor_boats_yrs  =  add_history_and_future(loadfactor_boats , first_year_boats,   change='no')   
mileage_boats_yrs     =  add_history_and_future(mileage_boats    , first_year_boats,   change='no')  
weight_frac_boats_yrs =  add_history_and_future(weight_boats     , first_year_boats,   change='no')  

# normalize the share of boats to 1 & adjust the capacity & mileage for smaller ships 
share_of_boats_yrs   = share_of_boats_yrs.div(share_of_boats_yrs.sum(axis=1), axis=0)
cap_of_boats_yrs     = cap_of_boats_yrs.mul(cap_adjustment, axis=1)
mileage_boats_yrs    = mileage_boats_yrs.mul(mile_adjustment, axis=1)

# now derive the number of ships for 4 ship types in four steps: 
# 1) get the share of the ship types in the Tkms shipped (what % of total tkms shipped goes by what ship type?)
share_of_boats_tkm_all   = share_of_boats_yrs * cap_of_boats_yrs * loadfactor_boats_yrs * mileage_boats_yrs
share_of_boats_tkm       = share_of_boats_tkm_all.div(share_of_boats_tkm_all.sum(axis=1), axis=0)

# 2) get the total tkms shipped by ship-type. (The shares are pre-calculated from 1900 onwards, so a selection from 1971-onwards is applied here)
ship_small_tkm  = tonkms_Ttkms['international shipping'].unstack().mul(share_of_boats_tkm['Small'].loc[start_year:], axis=0)
ship_medium_tkm = tonkms_Ttkms['international shipping'].unstack().mul(share_of_boats_tkm['Medium'].loc[start_year:], axis=0)
ship_large_tkm  = tonkms_Ttkms['international shipping'].unstack().mul(share_of_boats_tkm['Large'].loc[start_year:], axis=0)
ship_vlarge_tkm = tonkms_Ttkms['international shipping'].unstack().mul(share_of_boats_tkm['Very Large'].loc[start_year:], axis=0)

# 3) get the vehicle-kms by ship type (multiply by 1000000 to get from Giga-tkm to tkm)
ship_small_vehkm  = ship_small_tkm.mul(1000000).div(cap_of_boats_yrs['Small'].loc[start_year:], axis=0)
ship_medium_vehkm = ship_medium_tkm.mul(1000000).div(cap_of_boats_yrs['Medium'].loc[start_year:], axis=0)  
ship_large_vehkm  = ship_large_tkm.mul(1000000).div(cap_of_boats_yrs['Large'].loc[start_year:], axis=0)  
ship_vlarge_vehkm = ship_vlarge_tkm.mul(1000000).div(cap_of_boats_yrs['Very Large'].loc[start_year:], axis=0) 

# 4) get the number of ships (stock) by dividing with the mileage
ship_small_nr  = ship_small_vehkm.div(mileage_boats_yrs['Small'].loc[start_year:], axis=0)
ship_medium_nr = ship_medium_vehkm.div(mileage_boats_yrs['Medium'].loc[start_year:], axis=0)  
ship_large_nr  = ship_large_vehkm.div(mileage_boats_yrs['Large'].loc[start_year:], axis=0)  
ship_vlarge_nr = ship_vlarge_vehkm.div(mileage_boats_yrs['Very Large'].loc[start_year:], axis=0) 

# for comparison, we find the difference of the known and the calculated nr of ships (global total) in the period 2005-2018
diff_ships = pd.DataFrame().reindex_like(nr_of_boats)
diff_ships['Small']      =  ship_small_nr.loc[list(range(2005,2018+1)), 28].div(nr_of_boats['Small'])
diff_ships['Medium']     =  ship_medium_nr.loc[list(range(2005,2018+1)), 28].div(nr_of_boats['Medium'])
diff_ships['Large']      =  ship_large_nr.loc[list(range(2005,2018+1)), 28].div(nr_of_boats['Large'])
diff_ships['Very Large'] =  ship_vlarge_nr.loc[list(range(2005,2018+1)), 28].div(nr_of_boats['Very Large'])

total_nr_of_ships = ship_small_nr + ship_medium_nr + ship_large_nr + ship_vlarge_nr
diff_ships_total = total_nr_of_ships.loc[list(range(2005,2018+1)), 28].div(nr_of_boats.sum(axis=1))

# Export total global number of vehicles in the fleet (stock) as csv
region_list = list(range(1,27))
total_nr_vehicles = pd.DataFrame(index=total_nr_of_ships.index, columns=['Buses','Trains','HST','Cars','Planes','Bikes','Trucks','Cargo Trains','Ships','Inland ships','Cargo Planes'])
total_nr_vehicles['Buses']        = bus_regl_nr[region_list].sum(axis=1) + bus_midi_nr[region_list].sum(axis=1)
total_nr_vehicles['Trains']       = rail_reg_nr[region_list].sum(axis=1)
total_nr_vehicles['HST']          = rail_hst_nr[region_list].sum(axis=1)
total_nr_vehicles['Cars']         = car_total_nr[region_list].sum(axis=1)
total_nr_vehicles['Planes']       = air_pas_nr[region_list].sum(axis=1)
total_nr_vehicles['Bikes']        = bikes_nr[region_list].sum(axis=1)
total_nr_vehicles['Trucks']       = trucks_HFT_nr[region_list].sum(axis=1) + trucks_MFT_nr[region_list].sum(axis=1) + trucks_LCV_nr[region_list].sum(axis=1)
total_nr_vehicles['Cargo Trains'] = rail_freight_nr[region_list].sum(axis=1)
total_nr_vehicles['Ships']        = total_nr_of_ships[region_list].sum(axis=1) 
total_nr_vehicles['Inland ships'] = inland_ship_nr[region_list].sum(axis=1)
total_nr_vehicles['Cargo Planes'] = air_freight_nr[region_list].sum(axis=1)
total_nr_vehicles.to_csv('output\\' + folder + '\\global_vehicle_nr.csv', index=True) # in nr of vehicles 

# some indicators for the model accuracy comparison
inland_ship_nr[[2,11,16,20]].loc[2015].sum()           # inland shipping in 2015 (China, Russia, Europe & US)
car_total_nr[[11,12]].loc[2018].sum()
rail_reg_nr[[1,2,11,12,18,20,23]].loc[2017].sum()      # India, Canada, China, United States, Europe, Japan
rail_freight_nr[[1,2,11,12,18,20,16]].loc[2016].sum()  # India, Canada, China, United States, Europe, Russia

#%% INFLOW-OUTFLOW claculations using the ODYM Dynamic Stock Model (DSM) as a function

# In order to calculate inflow & outflow smoothly (without peaks for the initial years), we calculate a historic tail to the stock, by adding a 0 value for first year of operation (=1926), then interpolate values towards 1971
def stock_tail(stock, first_year_veh, choice='stock'):
    
    if choice=='stock':
       zero_value = [0 for i in range(0, regions)]
    else:
       zero_value = stock.head(1).values[0]
       
    stock_used = pd.DataFrame(stock).reindex_like(stock)
    stock_used.loc[first_year_veh] = zero_value  # set all regions to 0 in the year of initial operation
    stock_new  = stock_used.reindex(list(range(first_year_veh,end_year+1))).interpolate()
    return stock_new

# first define a Function in which the stock-driven DSM is applied to return (the moving average of the) inflow & outflow for all regions based on a static material intensity assumption
# IN: stock as a dataframe (by year & region), lifetime (in years), stdev_mult (multiplier to derive the standard deviation from the mean); OUT: 2 dataframes of inflow & outflow (by year & region)
# This variant of the DSM calculates & returns the unadjusted (cohort specific) matrix (YES, very memory intensive, but neccesary for dynamic assesmment, i.e. changing vehicle weights & compositions)
def inflow_outflow_dynamic(stock, fact1, fact2, distribution):

    initial_year   = stock.first_valid_index()
    inflow         = pd.DataFrame(index=range(initial_year,end_year+1), columns=stock.columns)
    outflow_cohort = pd.DataFrame(index=range(initial_year,end_year+1), columns=pd.MultiIndex.from_product([list(stock.columns), list(range(initial_year,end_year+1))]))
    stock_cohort   = pd.DataFrame(index=range(initial_year,end_year+1), columns=pd.MultiIndex.from_product([list(stock.columns), list(range(initial_year,end_year+1))]))


    # define mean (fact1) & standard deviation (fact2) (applied accross all regions) (For Weibull fact1 = shape & fact2 = scale)
    fact1_list = [fact1 for i in range(0,len(stock))]                      # list with the fixed (=mean) lifetime, given for every timestep (e.g. 1900-2100), needed for the DSM as it allows to change lifetime for different cohort (even though we keep it constant)
    if distribution == 'FoldedNormal':
       fact2_list = [fact1_list[i] * fact2 for i in range(0,len(stock))]   # for the (Folded) normal distribution, the standard_deviation is calculated as the mean times the multiplier (i.e. the standard deviation as a fraction of the mean)
    else:
       fact2_list = [fact2 for i in range(0,len(stock))]

    for region in list(stock.columns):
        # define and run the DSM   
        if distribution == 'FoldedNormal':
           DSMforward = DSM(t = np.arange(0,len(stock[region]),1), s=np.array(stock[region]), lt = {'Type': 'FoldedNormal', 'Mean': np.array(fact1_list), 'StdDev': np.array(fact2_list)})  # definition of the DSM based on a folded normal distribution
        else:
           DSMforward = DSM(t = np.arange(0,len(stock[region]),1), s=np.array(stock[region]), lt = {'Type': 'Weibull', 'Shape': np.array(fact1_list), 'Scale': np.array(fact2_list)})       # definition of the DSM based on a Weibull distribution
           
        out_sc, out_oc, out_i = DSMforward.compute_stock_driven_model(NegativeInflowCorrect = True)                                                                 # run the DSM, to give 3 outputs: stock_by_cohort, outflow_by_cohort & inflow_per_year
        

        # store the regional results in the return dataframe (full range of years (e.g. 1900 onwards), for later use in material calculations  & no moving average applied here)
        outflow_cohort.loc[:, idx[region,:]] = pd.DataFrame(out_oc, index=list(range(initial_year,end_year+1)), columns=list(range(initial_year,end_year+1))).values                                                                                                      # sum the outflow by cohort to get the total outflow per year
        stock_cohort.loc[:,  idx[region,:]]  = pd.DataFrame(out_sc, index=list(range(initial_year,end_year+1)), columns=list(range(initial_year,end_year+1))).values 
        inflow[region]                       = pd.Series(out_i, index=list(range(initial_year,end_year+1)))
         
    return inflow, outflow_cohort, stock_cohort     


# Pre-calculate the inflow (aka. market-)share corresponding to the (known) share of vehicles in stock (from IMAGE)
def inflow_outflow_typical(stock, fact1, fact2, distribution, stock_share):
   
   initial_year        = stock.first_valid_index()
   initial_year_shares = stock_share.first_valid_index()[0]
   index               = pd.MultiIndex.from_product([stock_share.columns, stock.index], names=['type', 'time'])
   inflow              = pd.DataFrame(0, index=index, columns=stock.columns)
   outflow_cohort      = pd.DataFrame(0, index=index, columns=pd.MultiIndex.from_product([list(stock.columns), list(range(initial_year,end_year+1))]))
   stock_cohort        = pd.DataFrame(0, index=index, columns=pd.MultiIndex.from_product([list(stock.columns), list(range(initial_year,end_year+1))]))

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

   # Then run the original DSM for each vehicle type & add it to the inflow, ouflow & stock containers, (for stock: index = vtype, time; columns = region, time)
   for vtype in list(stock_share.columns): 
      if stock_share[vtype].sum() > 0.001:
         dsm_inflow, dsm_outflow_coh, dsm_stock_coh = inflow_outflow_dynamic(stock_by_vtype.loc[idx[vtype,:],:].droplevel(0), fact1, fact2, distribution)
         
         inflow.loc[idx[vtype,:],:]                 = dsm_inflow.values
         outflow_cohort.loc[idx[vtype,:],idx[:,:]]  = dsm_outflow_coh.values
         stock_cohort.loc[idx[vtype,:],idx[:,:]]    = dsm_stock_coh.values
      
      else:
         pass
        
   return inflow, outflow_cohort, stock_cohort


# Calculate the historic tail (& reduce regions to 26)
air_pas_nr     = stock_tail(air_pas_nr[list(range(1,regions+1))],  first_year_vehicle['air_pas'].values[0])
rail_reg_nr    = stock_tail(rail_reg_nr[list(range(1,regions+1))], first_year_vehicle['rail_reg'].values[0])
rail_hst_nr    = stock_tail(rail_hst_nr[list(range(1,regions+1))], first_year_vehicle['rail_hst'].values[0])
bikes_nr       = stock_tail(bikes_nr[list(range(1,regions+1))],    first_year_vehicle['bicycle'].values[0])

air_freight_nr  = stock_tail(air_freight_nr[list(range(1,regions+1))],  first_year_vehicle['air_freight'].values[0])
rail_freight_nr = stock_tail(rail_freight_nr[list(range(1,regions+1))], first_year_vehicle['rail_freight'].values[0])
inland_ship_nr  = stock_tail(inland_ship_nr[list(range(1,regions+1))],  first_year_vehicle['inland_shipping'].values[0])
ship_small_nr   = stock_tail(ship_small_nr[list(range(1,regions+1))],   1900)
ship_medium_nr  = stock_tail(ship_medium_nr[list(range(1,regions+1))],  1900)
ship_large_nr   = stock_tail(ship_large_nr[list(range(1,regions+1))],   1900)
ship_vlarge_nr  = stock_tail(ship_vlarge_nr[list(range(1,regions+1))],  1900)

bus_regl_nr     = stock_tail(bus_regl_nr[list(range(1,regions+1))],  first_year_vehicle['reg_bus'].values[0])
bus_midi_nr     = stock_tail(bus_midi_nr[list(range(1,regions+1))],  first_year_vehicle['midi_bus'].values[0])
car_total_nr    = stock_tail(car_total_nr[list(range(1,regions+1))],  first_year_vehicle['car'].values[0])

trucks_HFT_nr   = stock_tail(trucks_HFT_nr[list(range(1,regions+1))],  first_year_vehicle['HFT'].values[0])
trucks_MFT_nr   = stock_tail(trucks_HFT_nr[list(range(1,regions+1))],  first_year_vehicle['MFT'].values[0])
trucks_LCV_nr   = stock_tail(trucks_LCV_nr[list(range(1,regions+1))],  first_year_vehicle['LCV'].values[0])

##################### DYNAMIC MODEL (runtime: ca. 14min) ########################################################################
# Calculate the NUMBER of vehicles, total for inflow & by cohort for stock & outflow, first only for simple vehicles

air_pas_in,      air_pas_out_coh,      air_pas_stock_coh      = inflow_outflow_dynamic(air_pas_nr, float(lifetimes_vehicles.loc['mean','air_pas']),  float(lifetimes_vehicles.loc['stdev','air_pas']),  lifetimes_vehicles.loc['type','air_pas'])
rail_reg_in,     rail_reg_out_coh,     rail_reg_stock_coh     = inflow_outflow_dynamic(rail_reg_nr, float(lifetimes_vehicles.loc['mean','rail_reg']), float(lifetimes_vehicles.loc['stdev','rail_reg']), lifetimes_vehicles.loc['type','rail_reg'])
rail_hst_in,     rail_hst_out_coh,     rail_hst_stock_coh     = inflow_outflow_dynamic(rail_hst_nr, float(lifetimes_vehicles.loc['mean','rail_hst']), float(lifetimes_vehicles.loc['stdev','rail_hst']), lifetimes_vehicles.loc['type','rail_hst'])
bikes_in,        bikes_out_coh,        bikes_stock_coh        = inflow_outflow_dynamic(bikes_nr,    float(lifetimes_vehicles.loc['mean','bicycle']),  float(lifetimes_vehicles.loc['stdev','bicycle']),  lifetimes_vehicles.loc['type','bicycle'])

air_freight_in,  air_freight_out_coh,  air_freight_stock_coh  = inflow_outflow_dynamic(air_freight_nr,  float(lifetimes_vehicles.loc['mean','air_freight']),        float(lifetimes_vehicles.loc['stdev','air_freight']),        lifetimes_vehicles.loc['type','air_freight'])
rail_freight_in, rail_freight_out_coh, rail_freight_stock_coh = inflow_outflow_dynamic(rail_freight_nr, float(lifetimes_vehicles.loc['mean','rail_freight']),       float(lifetimes_vehicles.loc['stdev','rail_freight']),       lifetimes_vehicles.loc['type','rail_freight'])
inland_ship_in,  inland_ship_out_coh,  inland_ship_stock_coh  = inflow_outflow_dynamic(inland_ship_nr,  float(lifetimes_vehicles.loc['mean','inland_shipping']),    float(lifetimes_vehicles.loc['stdev','inland_shipping']),    lifetimes_vehicles.loc['type','inland_shipping'])
ship_small_in,   ship_small_out_coh,   ship_small_stock_coh   = inflow_outflow_dynamic(ship_small_nr,   float(lifetimes_vehicles.loc['mean','sea_shipping_small']), float(lifetimes_vehicles.loc['stdev','sea_shipping_small']), lifetimes_vehicles.loc['type','sea_shipping_small'])
ship_medium_in,  ship_medium_out_coh,  ship_medium_stock_coh  = inflow_outflow_dynamic(ship_medium_nr,  float(lifetimes_vehicles.loc['mean','sea_shipping_med']),   float(lifetimes_vehicles.loc['stdev','sea_shipping_med']),   lifetimes_vehicles.loc['type','sea_shipping_med'])
ship_large_in,   ship_large_out_coh,   ship_large_stock_coh   = inflow_outflow_dynamic(ship_large_nr,   float(lifetimes_vehicles.loc['mean','sea_shipping_large']), float(lifetimes_vehicles.loc['stdev','sea_shipping_large']), lifetimes_vehicles.loc['type','sea_shipping_large'])
ship_vlarge_in,  ship_vlarge_out_coh,  ship_vlarge_stock_coh  = inflow_outflow_dynamic(ship_vlarge_nr,  float(lifetimes_vehicles.loc['mean','sea_shipping_vl']),    float(lifetimes_vehicles.loc['stdev','sea_shipping_vl']),   lifetimes_vehicles.loc['type','sea_shipping_vl'])


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


### Then calculate the inflow & outflow for typical vehicles (vehicles with relevant sub types) as well (runtime appr. 40 min.)
bus_regl_in,     bus_regl_out_coh,     bus_regl_stock_coh     = inflow_outflow_typical(bus_regl_nr,    float(lifetimes_vehicles.loc['mean','reg_bus']), float(lifetimes_vehicles.loc['stdev','reg_bus']),  lifetimes_vehicles.loc['type','reg_bus'],  buses_regl_vshares)
bus_midi_in,     bus_midi_out_coh,     bus_midi_stock_coh     = inflow_outflow_typical(bus_midi_nr,    float(lifetimes_vehicles.loc['mean','midi_bus']),float(lifetimes_vehicles.loc['stdev','midi_bus']), lifetimes_vehicles.loc['type','midi_bus'], buses_midi_vshares)
car_in,          car_out_coh,          car_stock_coh          = inflow_outflow_typical(car_total_nr,   float(lifetimes_vehicles.loc['shape','car']),    float(lifetimes_vehicles.loc['scale','car']),      lifetimes_vehicles.loc['type','car'],      vehicleshare_cars)

trucks_HFT_in,   trucks_HFT_out_coh,   trucks_HFT_stock_coh   = inflow_outflow_typical(trucks_HFT_nr,  float(lifetimes_vehicles.loc['mean','HFT']),     float(lifetimes_vehicles.loc['stdev','HFT']),      lifetimes_vehicles.loc['type','HFT'],      trucks_HFT_vshares)
trucks_MFT_in,   trucks_MFT_out_coh,   trucks_MFT_stock_coh   = inflow_outflow_typical(trucks_MFT_nr,  float(lifetimes_vehicles.loc['mean','MFT']),     float(lifetimes_vehicles.loc['stdev','MFT']),      lifetimes_vehicles.loc['type','MFT'],      trucks_MFT_vshares)
trucks_LCV_in,   trucks_LCV_out_coh,   trucks_LCV_stock_coh   = inflow_outflow_typical(trucks_LCV_nr,  float(lifetimes_vehicles.loc['mean','LCV']),     float(lifetimes_vehicles.loc['stdev','LCV']),      lifetimes_vehicles.loc['type','LCV'],      trucks_MFT_vshares)  # Assumption: used MFT as a market-share for LCVs



#%% ################### MATERIAL CALCULATIONS ##########################################

# for those vehicles with only 1 relevant sub-type, we calculate the material stocks & flows as: Nr * weight (kg) * composition (%)
def nr_by_cohorts_to_materials_simple(inflow, outflow_cohort, stock_cohort, weight, composition):
   
   index =   pd.MultiIndex.from_product([composition.columns, inflow.index], names=['time', 'type'])
   inflow_mat  = pd.DataFrame(0, index=index, columns=inflow.columns)
   outflow_mat = pd.DataFrame(0, index=index, columns=inflow.columns)
   stock_mat   = pd.DataFrame(0, index=index, columns=inflow.columns)
   
   for material in list(composition.columns): 
      # before running, check if the material is at all relevant in the vehicle (save calculation time)
      if composition[material].sum() > 0.001:          
         for region in list(inflow.columns):
            inflow_mat.loc[idx[material,:],region]  = inflow[region].mul(weight).mul(composition[material]).values
            outflow_mat.loc[idx[material,:],region] = outflow_cohort.loc[:,idx[region,:]].droplevel(0, axis=1).mul(weight, axis=1).mul(composition[material], axis=1).sum(axis=1).values
            stock_mat.loc[idx[material,:],region]   = stock_cohort.loc[:,idx[region,:]].droplevel(0, axis=1).mul(weight, axis=1).mul(composition[material], axis=1).sum(axis=1).values
      else:
         pass
         
   return inflow_mat, outflow_mat, stock_mat


# for those vehicles with multiple sub-type, we calculate the material stocks & flows in the same way, but using data on specific vehicle sub-types
def nr_by_cohorts_to_materials_typical(inflow, outflow_cohort, stock_cohort, weight, composition):
   
   index       = pd.MultiIndex.from_product([composition.columns.levels[1], list(range(inflow.first_valid_index()[1],end_year+1))], names=['material', 'time'])
   columns     = pd.MultiIndex.from_product([inflow.index.levels[0], inflow.columns],      names=['type', 'region'])
   inflow_mat  = pd.DataFrame(0, index=index, columns=columns)
   outflow_mat = pd.DataFrame(0, index=index, columns=columns)
   stock_mat   = pd.DataFrame(0, index=index, columns=columns)
   
   for vtype in list(inflow.index.levels[0]): 
      # before running, check if the vehicle type is at all relevant in the vehicle (save calculation time)
      if stock_cohort.loc[idx[vtype,:],:].sum().sum() > 0.001:  
         for material in list(composition.columns.levels[1]): 
            # before running, check if the material is at all relevant in the vehicle (save calculation time)
            if composition.loc[:,idx[vtype,material]].sum() > 0.001:          
               for region in list(inflow.columns):
                  
                  inflow_mat.loc[idx[material,:], idx[vtype, region]]  = inflow.loc[idx[vtype,:],region].droplevel(0).mul(weight[vtype]).mul(composition.loc[:,idx[vtype,material]]).values
                  outflow_mat.loc[idx[material,:],idx[vtype, region]]  = outflow_cohort.loc[idx[vtype,:],idx[region,:]].droplevel(0, axis=1).mul(weight[vtype], axis=1).mul(composition.loc[:,idx[vtype,material]], axis=1).sum(axis=1).values
                  stock_mat.loc[idx[material,:],  idx[vtype, region]]  = stock_cohort.loc[idx[vtype,:],idx[region,:]].droplevel(0, axis=1).mul(weight[vtype], axis=1).mul(composition.loc[:,idx[vtype,material]], axis=1).sum(axis=1).values

            else:
               pass
      else:
         pass
         
   return inflow_mat, outflow_mat, stock_mat

# capacity of boats is in tonnes, the weight - expressed as a fraction of the capacity - is calculated in in kgs here
weight_boats  = weight_frac_boats_yrs * cap_of_boats_yrs * 1000 

#%% ############################################# RUNNING THE DYNAMIC STOCK FUNCTIONS  (runtime ca.  55 min)  ###############################################

# run the simple material calculations on all vehicles                                                                                             # weight is passed as a series instead of a dataframe (pragmatic choice)
air_pas_mat_in,      air_pas_mat_out,      air_pas_mat_stock        = nr_by_cohorts_to_materials_simple(air_pas_in,      air_pas_out_coh,      air_pas_stock_coh,      vehicle_weight_kg_air_pas["air_pas"],             material_fractions_air_pas)
rail_reg_mat_in,     rail_reg_mat_out,     rail_reg_mat_stock       = nr_by_cohorts_to_materials_simple(rail_reg_in,     rail_reg_out_coh,     rail_reg_stock_coh,     vehicle_weight_kg_rail_reg["rail_reg"],           material_fractions_rail_reg)
rail_hst_mat_in,     rail_hst_mat_out,     rail_hst_mat_stock       = nr_by_cohorts_to_materials_simple(rail_hst_in,     rail_hst_out_coh,     rail_hst_stock_coh,     vehicle_weight_kg_rail_hst["rail_hst"],           material_fractions_rail_hst)
bikes_mat_in,        bikes_mat_out,        bikes_mat_stock          = nr_by_cohorts_to_materials_simple(bikes_in,        bikes_out_coh,        bikes_stock_coh,        vehicle_weight_kg_bicycle["bicycle"],             material_fractions_bicycle)

air_freight_mat_in,  air_freight_mat_out,  air_freight_mat_stock    = nr_by_cohorts_to_materials_simple(air_freight_in,  air_freight_out_coh,  air_freight_stock_coh,  vehicle_weight_kg_air_frgt["air_freight"],        material_fractions_air_frgt)
rail_freight_mat_in, rail_freight_mat_out, rail_freight_mat_stock   = nr_by_cohorts_to_materials_simple(rail_freight_in, rail_freight_out_coh, rail_freight_stock_coh, vehicle_weight_kg_rail_frgt["rail_freight"],      material_fractions_rail_frgt)
inland_ship_mat_in,  inland_ship_mat_out,  inland_ship_mat_stock    = nr_by_cohorts_to_materials_simple(inland_ship_in,  inland_ship_out_coh,  inland_ship_stock_coh,  vehicle_weight_kg_inland_ship["inland_shipping"], material_fractions_inland_ship)

ship_small_mat_in,   ship_small_mat_out,   ship_small_mat_stock     = nr_by_cohorts_to_materials_simple(ship_small_in,   ship_small_out_coh,   ship_small_stock_coh,   weight_boats['Small'],                            material_fractions_ship_small)
ship_medium_mat_in,  ship_medium_mat_out,  ship_medium_mat_stock    = nr_by_cohorts_to_materials_simple(ship_medium_in,  ship_medium_out_coh,  ship_medium_stock_coh,  weight_boats['Medium'],                           material_fractions_ship_medium)
ship_large_mat_in,   ship_large_mat_out,   ship_large_mat_stock     = nr_by_cohorts_to_materials_simple(ship_large_in,   ship_large_out_coh,   ship_large_stock_coh,   weight_boats['Large'],                            material_fractions_ship_large)
ship_vlarge_mat_in,  ship_vlarge_mat_out,  ship_vlarge_mat_stock    = nr_by_cohorts_to_materials_simple(ship_vlarge_in,  ship_vlarge_out_coh,  ship_vlarge_stock_coh,  weight_boats['Very Large'],                       material_fractions_ship_vlarge)

# Calculate the weight of materials in the vehicles with sub-types: stock, inflow & outflow 
bus_regl_mat_in,     bus_regl_mat_out,     bus_regl_mat_stock       = nr_by_cohorts_to_materials_typical(bus_regl_in,   bus_regl_out_coh,   bus_regl_stock_coh,   vehicle_weight_kg_bus,   material_fractions_bus_reg)
bus_midi_mat_in,     bus_midi_mat_out,     bus_midi_mat_stock       = nr_by_cohorts_to_materials_typical(bus_midi_in,   bus_midi_out_coh,   bus_midi_stock_coh,   vehicle_weight_kg_midi,  material_fractions_bus_midi)
car_total_mat_in,    car_total_mat_out,    car_total_mat_stock      = nr_by_cohorts_to_materials_typical(car_in,        car_out_coh,        car_stock_coh,        vehicle_weight_kg_car,   material_fractions_car)    

trucks_HFT_mat_in,   trucks_HFT_mat_out,   trucks_HFT_mat_stock     = nr_by_cohorts_to_materials_typical(trucks_HFT_in, trucks_HFT_out_coh, trucks_HFT_stock_coh, vehicle_weight_kg_HFT,   material_fractions_truck_HFT)
trucks_MFT_mat_in,   trucks_MFT_mat_out,   trucks_MFT_mat_stock     = nr_by_cohorts_to_materials_typical(trucks_MFT_in, trucks_MFT_out_coh, trucks_MFT_stock_coh, vehicle_weight_kg_MFT,   material_fractions_truck_MFT)
trucks_LCV_mat_in,   trucks_LCV_mat_out,   trucks_LCV_mat_stock     = nr_by_cohorts_to_materials_typical(trucks_LCV_in, trucks_LCV_out_coh, trucks_LCV_stock_coh, vehicle_weight_kg_LCV,   material_fractions_truck_LCV)

# Calculate the materials in batteries (in typical vehicles only) 
# For batteries this is a 2-step process, first (1) we pre-calculate the average material composition (of the batteries at inflow), based on a globally changing battery share & the changeing battery-specific material composition, both derived from our paper on (a.o.) storage in the electricity system.
# Then (2) we run the same function to derive the materials in vehicle batteries (based on changeing weight, composition & battery share)
# In doing so, we are no longer able to know the battery share per vehicle sub-type

battery_material_composition    = pd.DataFrame(index=pd.MultiIndex.from_product([list(battery_weights_full.columns.levels[0]), battery_materials_full.index]), columns=pd.MultiIndex.from_product([list(battery_weights_full.columns.levels[1]), battery_materials_full.columns.levels[0]]))
battery_weight_total_in         = pd.DataFrame(0, index=pd.MultiIndex.from_product([list(battery_weights_full.columns.levels[0]), battery_materials_full.index]), columns=battery_weights_full.columns.levels[1])
battery_weight_total_stock      = pd.DataFrame(0, index=pd.MultiIndex.from_product([list(battery_weights_full.columns.levels[0]), battery_materials_full.index]), columns=battery_weights_full.columns.levels[1])

battery_weight_regional_stock   = pd.DataFrame(0, index=battery_materials_full.index, columns=list(range(1,regions+1)))
battery_weight_regional_in      = pd.DataFrame(0, index=battery_materials_full.index, columns=list(range(1,regions+1)))
battery_weight_regional_out     = pd.DataFrame(0, index=battery_materials_full.index, columns=list(range(1,regions+1)))
 
# for now, 1 global battery market is assumed (so no difference between battery types for different vehicle sub-types), so the composition is duplicated over all vtypes
for vehicle in list(battery_weights_full.columns.levels[0]):
   for vtype in list(battery_weights_full.columns.levels[1]):
      for material in list(battery_materials_full.columns.levels[0]):
         battery_material_composition.loc[idx[vehicle,:],idx[vtype,material]] = battery_shares_full.mul(battery_materials_full.loc[:,idx[material,:]].droplevel(0,axis=1)).sum(axis=1).values


# Battery material calculations (these are made from 1971 onwwards, hence the selection in years) 
bus_regl_bat_in,     bus_regl_bat_out,     bus_regl_bat_stock       = nr_by_cohorts_to_materials_typical(bus_regl_in.loc[idx[:,list(range(1971,2101))],:],   bus_regl_out_coh.loc[idx[:,list(range(1971,2101))],idx[:,list(range(1971,2101))]],   bus_regl_stock_coh.loc[idx[:,list(range(1971,2101))],idx[:,list(range(1971,2101))]],   battery_weights_full['reg_bus'],   battery_material_composition.loc[idx['reg_bus',:],:].droplevel(0))
bus_midi_bat_in,     bus_midi_bat_out,     bus_midi_bat_stock       = nr_by_cohorts_to_materials_typical(bus_midi_in.loc[idx[:,list(range(1971,2101))],:],   bus_midi_out_coh.loc[idx[:,list(range(1971,2101))],idx[:,list(range(1971,2101))]],   bus_midi_stock_coh.loc[idx[:,list(range(1971,2101))],idx[:,list(range(1971,2101))]],   battery_weights_full['midi_bus'],  battery_material_composition.loc[idx['midi_bus',:],:].droplevel(0))
car_total_bat_in,    car_total_bat_out,    car_total_bat_stock      = nr_by_cohorts_to_materials_typical(car_in.loc[idx[:,list(range(1971,2101))],:],        car_out_coh.loc[idx[:,list(range(1971,2101))],idx[:,list(range(1971,2101))]],        car_stock_coh.loc[idx[:,list(range(1971,2101))],idx[:,list(range(1971,2101))]],        battery_weights_full['car'],       battery_material_composition.loc[idx['car',:],:].droplevel(0))    

trucks_HFT_bat_in,   trucks_HFT_bat_out,   trucks_HFT_bat_stock     = nr_by_cohorts_to_materials_typical(trucks_HFT_in.loc[idx[:,list(range(1971,2101))],:], trucks_HFT_out_coh.loc[idx[:,list(range(1971,2101))],idx[:,list(range(1971,2101))]], trucks_HFT_stock_coh.loc[idx[:,list(range(1971,2101))],idx[:,list(range(1971,2101))]], battery_weights_full['HFT'],       battery_material_composition.loc[idx['HFT',:],:].droplevel(0))
trucks_MFT_bat_in,   trucks_MFT_bat_out,   trucks_MFT_bat_stock     = nr_by_cohorts_to_materials_typical(trucks_MFT_in.loc[idx[:,list(range(1971,2101))],:], trucks_MFT_out_coh.loc[idx[:,list(range(1971,2101))],idx[:,list(range(1971,2101))]], trucks_MFT_stock_coh.loc[idx[:,list(range(1971,2101))],idx[:,list(range(1971,2101))]], battery_weights_full['MFT'],       battery_material_composition.loc[idx['MFT',:],:].droplevel(0))
trucks_LCV_bat_in,   trucks_LCV_bat_out,   trucks_LCV_bat_stock     = nr_by_cohorts_to_materials_typical(trucks_LCV_in.loc[idx[:,list(range(1971,2101))],:], trucks_LCV_out_coh.loc[idx[:,list(range(1971,2101))],idx[:,list(range(1971,2101))]], trucks_LCV_stock_coh.loc[idx[:,list(range(1971,2101))],idx[:,list(range(1971,2101))]], battery_weights_full['LCV'],       battery_material_composition.loc[idx['LCV',:],:].droplevel(0))



# Sum the weight of the accounted materials (! so not total weight) in batteries by vehicle & vehicel type, output for figures
for vtype in list(battery_weights_full.columns.levels[1]):
      battery_weight_total_in.loc[idx['reg_bus',:],vtype]     = bus_regl_bat_in.loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values
      battery_weight_total_in.loc[idx['midi_bus',:],vtype]    = bus_midi_bat_in.loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values
      battery_weight_total_in.loc[idx['LCV',:],vtype]         = trucks_LCV_bat_in.loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values
      battery_weight_total_in.loc[idx['MFT',:],vtype]         = trucks_MFT_bat_in.loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values
      battery_weight_total_in.loc[idx['HFT',:],vtype]         = trucks_HFT_bat_in.loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values
      
      battery_weight_total_stock.loc[idx['reg_bus',:],vtype]  = bus_regl_bat_stock.loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values
      battery_weight_total_stock.loc[idx['midi_bus',:],vtype] = bus_midi_bat_stock.loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values
      battery_weight_total_stock.loc[idx['LCV',:],vtype]      = trucks_LCV_bat_stock.loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values
      battery_weight_total_stock.loc[idx['MFT',:],vtype]      = trucks_MFT_bat_stock.loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values
      battery_weight_total_stock.loc[idx['HFT',:],vtype]      = trucks_HFT_bat_stock.loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values   
      
      if vtype == 'Trolley':
         pass 
      else:
         battery_weight_total_in.loc[idx['car',:],vtype]      = car_total_bat_in.loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values
         battery_weight_total_stock.loc[idx['car',:],vtype]   = car_total_bat_stock.loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values

battery_weight_total_in.to_csv('output\\' + folder + '\\battery_weight_kg_in.csv', index=True)       # in kg
battery_weight_total_stock.to_csv('output\\' + folder + '\\battery_weight_kg_stock.csv', index=True) # in kg

# Regional battery weight (only the accounted materials), used in graph later on
battery_weight_regional_stock = bus_regl_bat_stock.sum(axis=0,level=1).sum(axis=1,level=1) + bus_midi_bat_stock.sum(axis=0,level=1).sum(axis=1,level=1) + trucks_LCV_bat_stock.sum(axis=0,level=1).sum(axis=1,level=1) + trucks_MFT_bat_stock.sum(axis=0,level=1).sum(axis=1,level=1) + trucks_HFT_bat_stock.sum(axis=0,level=1).sum(axis=1,level=1) + car_total_bat_stock.sum(axis=0,level=1).sum(axis=1,level=1)
battery_weight_regional_in    = bus_regl_bat_in.sum(axis=0,level=1).sum(axis=1,level=1)    + bus_midi_bat_in.sum(axis=0,level=1).sum(axis=1,level=1)    + trucks_LCV_bat_in.sum(axis=0,level=1).sum(axis=1,level=1)    + trucks_MFT_bat_in.sum(axis=0,level=1).sum(axis=1,level=1)    + trucks_HFT_bat_in.sum(axis=0,level=1).sum(axis=1,level=1)    + car_total_bat_in.sum(axis=0,level=1).sum(axis=1,level=1)     
battery_weight_regional_out   = bus_regl_bat_out.sum(axis=0,level=1).sum(axis=1,level=1)   + bus_midi_bat_out.sum(axis=0,level=1).sum(axis=1,level=1)   + trucks_LCV_bat_out.sum(axis=0,level=1).sum(axis=1,level=1)   + trucks_MFT_bat_out.sum(axis=0,level=1).sum(axis=1,level=1)   + trucks_HFT_bat_out.sum(axis=0,level=1).sum(axis=1,level=1)   + car_total_bat_out.sum(axis=0,level=1).sum(axis=1,level=1)                                      

#%% ################################### Organise data for output ###########################################

year_select = list(range(start_year, 2051))

# define 6 dataframes on materials in  stock, inflow & outflow X passenger vs. freight vehicles
index   = pd.MultiIndex.from_product([year_select, list(range(1,regions+1)), ['vehicle','battery'], labels_materials], names=['year', 'region', 'part', 'materials'])
vehicle_materials_stock_passenger   = pd.DataFrame(index=index, columns=labels_pas)
vehicle_materials_stock_freight     = pd.DataFrame(index=index, columns=labels_fre)
vehicle_materials_inflow_passenger  = pd.DataFrame(index=index, columns=labels_pas)
vehicle_materials_inflow_freight    = pd.DataFrame(index=index, columns=labels_fre)
vehicle_materials_outflow_passenger = pd.DataFrame(index=index, columns=labels_pas)
vehicle_materials_outflow_freight   = pd.DataFrame(index=index, columns=labels_fre)

for material in labels_materials:

   ############## STARTING WITH SIMPLE VEHICLES ###########################
   
   # passenger stock, vehicles (in kg)
   vehicle_materials_stock_passenger.loc[idx[:,:,'vehicle', material],'bicycle']       = bikes_mat_stock.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_stock_passenger.loc[idx[:,:,'vehicle', material],'rail_reg']      = rail_reg_mat_stock.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_stock_passenger.loc[idx[:,:,'vehicle', material],'rail_hst']      = rail_reg_mat_stock.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_stock_passenger.loc[idx[:,:,'vehicle', material],'air_pas']       = air_pas_mat_stock.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values

   # freight stock (in kg)
   vehicle_materials_stock_freight.loc[idx[:,:,'vehicle', material],'inland_shipping']     = inland_ship_mat_stock.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_stock_freight.loc[idx[:,:,'vehicle', material],'rail_freight']        = rail_freight_mat_stock.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_stock_freight.loc[idx[:,:,'vehicle', material],'air_freight']         = air_freight_mat_stock.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_stock_freight.loc[idx[:,:,'vehicle', material],'sea_shipping_small']  = ship_small_mat_stock.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_stock_freight.loc[idx[:,:,'vehicle', material],'sea_shipping_med']    = ship_medium_mat_stock.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_stock_freight.loc[idx[:,:,'vehicle', material],'sea_shipping_large']  = ship_large_mat_stock.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_stock_freight.loc[idx[:,:,'vehicle', material],'sea_shipping_vl']     = ship_vlarge_mat_stock.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values

   # passenger inflow (in kg)
   vehicle_materials_inflow_passenger.loc[idx[:,:,'vehicle', material],'bicycle']    = bikes_mat_in.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_inflow_passenger.loc[idx[:,:,'vehicle', material],'rail_reg']   = rail_reg_mat_in.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_inflow_passenger.loc[idx[:,:,'vehicle', material],'rail_hst']   = rail_hst_mat_in.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_inflow_passenger.loc[idx[:,:,'vehicle', material],'air_pas']    = air_pas_mat_in.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values

   # freight inflow (in kg)
   vehicle_materials_inflow_freight.loc[idx[:,:,'vehicle', material],'inland_shipping']    = inland_ship_mat_in.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_inflow_freight.loc[idx[:,:,'vehicle', material],'rail_freight']       = rail_freight_mat_in.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_inflow_freight.loc[idx[:,:,'vehicle', material],'air_freight']        = air_freight_mat_in.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_inflow_freight.loc[idx[:,:,'vehicle', material],'sea_shipping_small'] = ship_small_mat_in.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_inflow_freight.loc[idx[:,:,'vehicle', material],'sea_shipping_med']   = ship_medium_mat_in.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_inflow_freight.loc[idx[:,:,'vehicle', material],'sea_shipping_large'] = ship_large_mat_in.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_inflow_freight.loc[idx[:,:,'vehicle', material],'sea_shipping_vl']    = ship_vlarge_mat_in.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values

   # passenger outflow (in kg)
   vehicle_materials_outflow_passenger.loc[idx[:,:,'vehicle', material],'bicycle']   = bikes_mat_out.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_outflow_passenger.loc[idx[:,:,'vehicle', material],'rail_reg']  = rail_reg_mat_out.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_outflow_passenger.loc[idx[:,:,'vehicle', material],'rail_hst']  = rail_hst_mat_out.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_outflow_passenger.loc[idx[:,:,'vehicle', material],'air_pas']   = air_pas_mat_out.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values

   # freight outflow (in kg)
   vehicle_materials_outflow_freight.loc[idx[:,:,'vehicle', material],'inland_shipping']     = inland_ship_mat_out.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_outflow_freight.loc[idx[:,:,'vehicle', material],'rail_freight']        = rail_freight_mat_out.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_outflow_freight.loc[idx[:,:,'vehicle', material],'air_freight']         = air_freight_mat_out.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_outflow_freight.loc[idx[:,:,'vehicle', material],'sea_shipping_small']  = ship_small_mat_out.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_outflow_freight.loc[idx[:,:,'vehicle', material],'sea_shipping_med']    = ship_medium_mat_out.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_outflow_freight.loc[idx[:,:,'vehicle', material],'sea_shipping_large']  = ship_large_mat_out.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
   vehicle_materials_outflow_freight.loc[idx[:,:,'vehicle', material],'sea_shipping_vl']     = ship_vlarge_mat_out.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values

   ############ CONTINUEING WITH TYPICAL VEHICLES (MATERIALS IN TRUCKS & BUSSES ARE SUMMED, FOR CARS DETAIL BY TYPE IS MAINTAINED) ##########################
   
   part = 'vehicle'
   
   # passenger stock, vehicles (in kg)
   vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'midi_bus']    = bus_midi_mat_stock.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values
   vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'reg_bus']     = bus_regl_mat_stock.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values
   vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'ICE']         = car_total_mat_stock.loc[idx[material,year_select],idx['ICE',:]].stack().reorder_levels([1,2,0]).values 
   vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'HEV']         = car_total_mat_stock.loc[idx[material,year_select],idx['HEV',:]].stack().reorder_levels([1,2,0]).values   
   vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'PHEV']        = car_total_mat_stock.loc[idx[material,year_select],idx['PHEV',:]].stack().reorder_levels([1,2,0]).values  
   vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'BEV']         = car_total_mat_stock.loc[idx[material,year_select],idx['BEV',:]].stack().reorder_levels([1,2,0]).values  
   vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'FCV']         = car_total_mat_stock.loc[idx[material,year_select],idx['FCV',:]].stack().reorder_levels([1,2,0]).values       
   
   # freight stock (in kg)
   vehicle_materials_stock_freight.loc[idx[:,:, part, material],'LCV']           = trucks_LCV_mat_stock.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values   
   vehicle_materials_stock_freight.loc[idx[:,:, part, material],'MFT']           = trucks_MFT_mat_stock.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values  
   vehicle_materials_stock_freight.loc[idx[:,:, part, material],'HFT']           = trucks_HFT_mat_stock.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values 

   # passenger inflow (in kg)
   vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'midi_bus']   = bus_midi_mat_in.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values
   vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'reg_bus']    = bus_regl_mat_in.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values
   vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'ICE']        = car_total_mat_in.loc[idx[material,year_select],idx['ICE',:]].stack().reorder_levels([1,2,0]).values 
   vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'HEV']        = car_total_mat_in.loc[idx[material,year_select],idx['HEV',:]].stack().reorder_levels([1,2,0]).values 
   vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'PHEV']       = car_total_mat_in.loc[idx[material,year_select],idx['PHEV',:]].stack().reorder_levels([1,2,0]).values 
   vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'BEV']        = car_total_mat_in.loc[idx[material,year_select],idx['BEV',:]].stack().reorder_levels([1,2,0]).values
   vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'FCV']        = car_total_mat_in.loc[idx[material,year_select],idx['FCV',:]].stack().reorder_levels([1,2,0]).values        
 
   # freight inflow (in kg)
   vehicle_materials_inflow_freight.loc[idx[:,:, part, material],'LCV']          = trucks_LCV_mat_in.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values   
   vehicle_materials_inflow_freight.loc[idx[:,:, part, material],'MFT']          = trucks_MFT_mat_in.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values  
   vehicle_materials_inflow_freight.loc[idx[:,:, part, material],'HFT']          = trucks_HFT_mat_in.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values    

   # passenger outflow (in kg)
   vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'midi_bus']  = bus_midi_mat_out.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values
   vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'reg_bus']   = bus_regl_mat_out.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values 
   vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'ICE']       = car_total_mat_out.loc[idx[material,year_select],idx['ICE',:]].stack().reorder_levels([1,2,0]).values  
   vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'HEV']       = car_total_mat_out.loc[idx[material,year_select],idx['HEV',:]].stack().reorder_levels([1,2,0]).values 
   vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'PHEV']      = car_total_mat_out.loc[idx[material,year_select],idx['PHEV',:]].stack().reorder_levels([1,2,0]).values 
   vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'BEV']       = car_total_mat_out.loc[idx[material,year_select],idx['BEV',:]].stack().reorder_levels([1,2,0]).values 
   vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'FCV']       = car_total_mat_out.loc[idx[material,year_select],idx['FCV',:]].stack().reorder_levels([1,2,0]).values       
   
   # freight outflow (in kg)
   vehicle_materials_outflow_freight.loc[idx[:,:, part, material],'LCV'] = trucks_LCV_mat_out.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values  
   vehicle_materials_outflow_freight.loc[idx[:,:, part, material],'MFT']= trucks_MFT_mat_out.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values 
   vehicle_materials_outflow_freight.loc[idx[:,:, part, material],'HFT'] = trucks_HFT_mat_out.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values 

   ############ CONTINUEING WITH BATTERIES (MATERALS IN TRUCKS & BUSSES ARE SUMMED, FOR CARS DETAIL BY TYPE IS MAINTAINED) ##########################

   part = 'battery'
   
   # passenger stock, vehicles (in kg)
   vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'midi_bus']    = bus_midi_bat_stock.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values
   vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'reg_bus']     = bus_regl_bat_stock.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values
   vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'ICE']         = car_total_bat_stock.loc[idx[material,year_select],idx['ICE',:]].stack().reorder_levels([1,2,0]).values 
   vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'HEV']         = car_total_bat_stock.loc[idx[material,year_select],idx['HEV',:]].stack().reorder_levels([1,2,0]).values   
   vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'PHEV']        = car_total_bat_stock.loc[idx[material,year_select],idx['PHEV',:]].stack().reorder_levels([1,2,0]).values  
   vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'BEV']         = car_total_bat_stock.loc[idx[material,year_select],idx['BEV',:]].stack().reorder_levels([1,2,0]).values  
   vehicle_materials_stock_passenger.loc[idx[:,:, part, material],'FCV']         = car_total_bat_stock.loc[idx[material,year_select],idx['FCV',:]].stack().reorder_levels([1,2,0]).values       
   
   # freight stock (in kg)
   vehicle_materials_stock_freight.loc[idx[:,:, part, material],'LCV']           = trucks_LCV_bat_stock.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values   
   vehicle_materials_stock_freight.loc[idx[:,:, part, material],'MFT']           = trucks_MFT_bat_stock.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values  
   vehicle_materials_stock_freight.loc[idx[:,:, part, material],'HFT']           = trucks_HFT_bat_stock.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values 

   # passenger inflow (in kg)
   vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'midi_bus']   = bus_midi_bat_in.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values
   vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'reg_bus']    = bus_regl_bat_in.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values
   vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'ICE']        = car_total_bat_in.loc[idx[material,year_select],idx['ICE',:]].stack().reorder_levels([1,2,0]).values 
   vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'HEV']        = car_total_bat_in.loc[idx[material,year_select],idx['HEV',:]].stack().reorder_levels([1,2,0]).values 
   vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'PHEV']       = car_total_bat_in.loc[idx[material,year_select],idx['PHEV',:]].stack().reorder_levels([1,2,0]).values 
   vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'BEV']        = car_total_bat_in.loc[idx[material,year_select],idx['BEV',:]].stack().reorder_levels([1,2,0]).values
   vehicle_materials_inflow_passenger.loc[idx[:,:, part, material],'FCV']        = car_total_bat_in.loc[idx[material,year_select],idx['FCV',:]].stack().reorder_levels([1,2,0]).values        
 
   # freight inflow (in kg)
   vehicle_materials_inflow_freight.loc[idx[:,:, part, material],'LCV']          = trucks_LCV_bat_in.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values   
   vehicle_materials_inflow_freight.loc[idx[:,:, part, material],'MFT']          = trucks_MFT_bat_in.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values  
   vehicle_materials_inflow_freight.loc[idx[:,:, part, material],'HFT']          = trucks_HFT_bat_in.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values    

   # passenger outflow (in kg)
   vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'midi_bus']  = bus_midi_bat_out.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values
   vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'reg_bus']   = bus_regl_bat_out.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values 
   vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'ICE']       = car_total_bat_out.loc[idx[material,year_select],idx['ICE',:]].stack().reorder_levels([1,2,0]).values  
   vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'HEV']       = car_total_bat_out.loc[idx[material,year_select],idx['HEV',:]].stack().reorder_levels([1,2,0]).values 
   vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'PHEV']      = car_total_bat_out.loc[idx[material,year_select],idx['PHEV',:]].stack().reorder_levels([1,2,0]).values 
   vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'BEV']       = car_total_bat_out.loc[idx[material,year_select],idx['BEV',:]].stack().reorder_levels([1,2,0]).values 
   vehicle_materials_outflow_passenger.loc[idx[:,:, part, material],'FCV']       = car_total_bat_out.loc[idx[material,year_select],idx['FCV',:]].stack().reorder_levels([1,2,0]).values       
   
   # freight outflow (in kg)
   vehicle_materials_outflow_freight.loc[idx[:,:, part, material],'LCV']         = trucks_LCV_bat_out.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values  
   vehicle_materials_outflow_freight.loc[idx[:,:, part, material],'MFT']         = trucks_MFT_bat_out.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values 
   vehicle_materials_outflow_freight.loc[idx[:,:, part, material],'HFT']         = trucks_HFT_bat_out.loc[idx[material,year_select],:].sum(axis=1, level=1).stack().reorder_levels([1,2,0]).values 


#%% combine dataframes for output

# add flow descriptor to the multi-index & fill na values with 0 (the for-loop above didn't cover the battery materials in vhicles without batteries, so these are set to 0 now)
vehicle_materials_stock_passenger  = pd.concat([vehicle_materials_stock_passenger.fillna(0)],    keys=['stock'],    names=['flow'])   
vehicle_materials_stock_freight    = pd.concat([vehicle_materials_stock_freight.fillna(0)],      keys=['stock'],    names=['flow'])      
vehicle_materials_inflow_passenger = pd.concat([vehicle_materials_inflow_passenger.fillna(0)],   keys=['inflow'],   names=['flow'])     
vehicle_materials_inflow_freight   = pd.concat([vehicle_materials_inflow_freight.fillna(0)],     keys=['inflow'],   names=['flow'])     
vehicle_materials_outflow_passenger = pd.concat([vehicle_materials_outflow_passenger.fillna(0)], keys=['outflow'],  names=['flow'])      
vehicle_materials_outflow_freight   = pd.concat([vehicle_materials_outflow_freight.fillna(0)],   keys=['outflow'],  names=['flow']) 

# concatenate stock, inflow & outflow into 1 dataframe (1 for passenger & 1 for freight)
vehicle_materials_passenger = pd.concat([vehicle_materials_stock_passenger, vehicle_materials_inflow_passenger,  vehicle_materials_outflow_passenger]) 
vehicle_materials_freight   = pd.concat([vehicle_materials_stock_freight,   vehicle_materials_inflow_freight,    vehicle_materials_outflow_freight])

# add category descriptors to the multi-index (pass vs. freight)
vehicle_materials_passenger = pd.concat([vehicle_materials_passenger], keys=['passenger'], names=['category']) 
vehicle_materials_freight   = pd.concat([vehicle_materials_freight],    keys=['freight'], names=['category'])  

vehicle_materials_passenger.columns.name = vehicle_materials_freight.columns.name = 'elements'

# concatenate into 1 single dataframe & add the 'vehicle' descriptor
vehicle_materials = pd.concat([vehicle_materials_passenger.stack().unstack(level=2), vehicle_materials_freight.stack().unstack(level=2)])
vehicle_materials = pd.concat([vehicle_materials], keys=['vehicles'], names=['sector'])

# re-order multi-index to the desired output & sum to global total
vehicle_materials_out = vehicle_materials.reorder_levels([3, 2, 0, 1, 6, 5, 4]) / 1000000  # div by 1*10^6 to translate from kg to kt
vehicle_materials_out.reset_index(inplace=True)                                         # return to columns

vehicle_materials_out.to_csv('output\\' + folder + '\\vehicle_materials_kt.csv', index=False) # in kt

