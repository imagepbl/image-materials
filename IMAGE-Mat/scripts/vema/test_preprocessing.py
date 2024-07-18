# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 13:39:52 2024

@author: Arp00003
"""

# Overview:
    
#TODO:'total_nr_vehicles_simple': assertion error! --> data completely different all years, alll types

# DONE:
# - 'battery_materials' done no assertion error 10-07-24
# - 'battery_shares'  done no assertion error 10-07-24
# - 'battery_weights_typical  done no assertion error 10-07-24
# - 'lifetimes_vehicles'   done no assertion error 11-07-2024
# - 'material_fractions_simple   done no assertion error 11-07-2024
# - 'material_fractions_typical':  done no assertion error 11-07-2024
# - vehicles_shares_typical  done no assertion errors 12-07-2024
# - 'vehicle_weights_simple':  done no assertion error 11-07-2024
# - 'vehicle_weights_typical'  done no assertion error 11-07-2024
# - 'weight boats'  done no assertion error 11-07-2024


import os
import numpy as np
import pandas as pd
from preprocessing import preprocessing

# get new values form preprocessing script
output_preprocessing = preprocessing()

# Get the current directory
current_directory = os.path.dirname(os.path.abspath(__file__))

# Define the relative path to the target directory
relative_path = os.path.join(current_directory, '..', '..', '..', 'IMAGE-Mat_old_version', 'IMAGE-Mat', 'VEMA')

# Change the directory
os.chdir(relative_path)

# Verify the change
print("Current Directory:", os.getcwd())

# load all old variables via globals - note: stock modelling and plotting is commented out in this VEMA version to save time while re-running!
from Vehicle_Material_Model import *

#%%  compare dfs automized (only works if dimensions of ald df are still exactlely the same as new)
dict_new_to_old_keys = {
    'battery_materials': battery_materials_full,
    'battery_shares': battery_shares_full,
    'battery_weights_typical': battery_weights_full,
    'weight_boats': weight_boats
    }
    
for key, old_value in dict_new_to_old_keys.items():
    print(key)
    
    # adjust years, as new values have somtimes longer stocktail than old values and therefore cant be compared
    if key == 'weight_boats':
        new_value = output_preprocessing.get(key).loc[1900:]
    
    else: 
        new_value = output_preprocessing.get(key).loc['1971':]
        # print(new_value)
    assert (new_value == old_value).all().all()
    
    
#%% Lifetime vehicles
# compare to lifetime_vehicles_mean, lifetime_vehicles_scale,lifetime_vehicles_shape, lifetime_vehicles_stdev

# get new value
new_lifetimes_vehicles = output_preprocessing.get('lifetimes_vehicles')

for key in ['mean', 'stdev', 'scale', 'shape']: 
    if key in {'mean', 'stdev'}:
        print(key)
        new_lifetimes_vehicles_compare = new_lifetimes_vehicles.loc[1971:, pd.IndexSlice[:, [key]]] # slice second level column index
        new_lifetimes_vehicles_compare = new_lifetimes_vehicles_compare.droplevel(1, axis=1) # drop level to be able to compare
        
        # get old value
        old_key = ('lifetimes_vehicles_'+ key)
        old_value = globals()[old_key]
        # print(old_value)
        assert (new_lifetimes_vehicles_compare == old_value).all().all()
    
    # shape and scale is only compared for cars
    if key in {'scale', 'shape'}:
        print(key)
        new_lifetimes_vehicles_compare = new_lifetimes_vehicles.loc[1971:, pd.IndexSlice[:, [key]]] # slice second level column index
        new_lifetimes_vehicles_compare = new_lifetimes_vehicles_compare.droplevel(1, axis=1) # drop level to be able to compare
        new_lifetimes_vehicles_compare = new_lifetimes_vehicles_compare.squeeze()
        # get old value
        old_key = ('lifetimes_vehicles_'+ key)
        old_value = globals()[old_key]['car']
        # print(old_value)
        assert (new_lifetimes_vehicles_compare == old_value).all().all()
    

#%% material fractions simple

for key in output_preprocessing['material_fractions_simple'].columns.get_level_values(0).unique():
    print(key)
    # replace keys as they were changed in the rewriting process
    old_key = ("material_fractions_" + key).replace("freight", "frgt")
    old_key = old_key.replace("sea_shipping", "ship")
    old_key = old_key.replace("inland_shipping", "inland_ship")
    old_key = old_key.replace("med", "medium")
    old_key = old_key.replace("_vl", "_vlarge")
    old_value = globals()[old_key].sort_index(axis = 1)
    
    new_value = output_preprocessing['material_fractions_simple'][key].sort_index(axis = 1)
    highest_val = max(new_value.index[0], old_value.index[0])
    
    # print(highest_val, new_value.index[0], old_value.index[0])
    assert (old_value.loc[highest_val:].sort_index(axis = 1) == new_value.loc[highest_val:].sort_index(axis = 1)).all().all()


#%% material fractions typical

for key in output_preprocessing['material_fractions_typical'].columns.get_level_values(0).unique():
    print(key)
    # replace old keys 
    old_key = ("material_fractions_" + key).replace("freight", "frgt")
    old_key = old_key.replace("sea_shipping", "ship")
    old_key = old_key.replace("inland_shipping", "inland_ship")
    old_key = old_key.replace("med", "medium")
    old_key = old_key.replace("_vl", "_vlarge")
    old_key = old_key.replace("_midi_bus", "_bus_midi")
    old_key = old_key.replace("_reg_bus", "_bus_reg")
    
    for try_sub in ["HFT", "LCV", "MFT"]:
        old_key = old_key.replace(try_sub, f"truck_{try_sub}")

    old_value = globals()[old_key].sort_index(axis = 1)
    
    new_value = output_preprocessing['material_fractions_typical'][key].sort_index(axis = 1)
    # highest value is start year that both old and new have in common, to enable comparison
    highest_val = max(new_value.index[0], old_value.index[0])
    
    # print(highest_val, new_value.index[0], old_value.index[0])
    assert (old_value.loc[highest_val:].sort_index(axis = 1) == new_value.loc[highest_val:].sort_index(axis = 1)).all().all()


#%% vehicle shares typical

for vehicle in output_preprocessing['vehicle_shares_typical'].columns.get_level_values(0).unique():
    print(vehicle)
  
    # get new vshare variable per vehicle
    new_value = output_preprocessing['vehicle_shares_typical'][vehicle]
    
    # get old vshare variable per vehicle
    old_key = vehicle
    old_key = old_key.replace("Cars", "vehicleshare_cars")
    old_key = old_key.replace("Regular Buses", "buses_regl_vshares")
    old_key = old_key.replace("Midi Buses", "buses_midi_vshares")
    old_key = old_key.replace("Heavy Freight Trucks", "trucks_HFT_vshares")
    old_key = old_key.replace("Medium Freight Trucks", "trucks_MFT_vshares")
    # comment from old VEMA: LCV vehicle shares are determined based on the medium trucks (so not calculated explicitly)
    old_key = old_key.replace("Light Commercial Vehicles", "trucks_MFT_vshares")
 
    old_value = globals()[old_key]
    
    # ensure that only data is compared that is available for both (1971 and later)
    new_value = new_value.loc[old_value.index.get_level_values(0).min():, ]
    
    # me;t and pivot old variable to bring it to same format as new variable 
    old_value_pivot = old_value.melt(ignore_index = False).reset_index(names=['time', 'DIM_1'])
    old_value_pivot = old_value_pivot.pivot(index = 'time', columns = ['variable', 'DIM_1'])
    old_value_pivot = old_value_pivot.droplevel(0, axis = 1)
    
    #sort columns to compare
    old_value_pivot = old_value_pivot.sort_index(axis = 1)
    new_value = new_value.sort_index(axis = 1)
    
    # exlude Trolley for cars from comparison
    if vehicle == 'Cars':
        new_value = new_value.loc[:, :'PHEV']

    # print(old_value_pivot)
    # print(new_value)    
    assert np.isclose(old_value_pivot, new_value).all().all()
    

#%% vehicle_weights_simple

for key in output_preprocessing['vehicle_weights_simple'].columns.get_level_values(0).unique():
    print(key)
    # replace old keys 
    old_key = ("vehicle_weight_kg_" + key)
    old_key = old_key.replace("air_freight", "air_frgt")
    old_key = old_key.replace("rail_freight", "rail_frgt")
    old_key = old_key.replace('inland_shipping', "inland_ship")
    
    # get old value from globals
    old_value = globals()[old_key]
    old_value = old_value.squeeze() # make Series to compare
    # print(old_value)
    # get new value
    new_value = output_preprocessing['vehicle_weights_simple'][key]
    # print(new_value)

    # get earliest year to enable comparison
    highest_val = max(new_value.index[0], old_value.index[0])
    
    assert (old_value.loc[highest_val:] == new_value.loc[highest_val:]).all().all()
    

#%% vehicle_weights_typical

for key in output_preprocessing['vehicle_weights_typical'].columns.get_level_values(0).unique():
    print(key)
    old_key = ("vehicle_weight_kg_" + key)
    old_key = old_key.replace("midi_bus", "midi")
    old_key = old_key.replace("reg_bus", "bus")
    
    # get old value from globals
    old_value = globals()[old_key]
    old_value = old_value.squeeze() # make Series to compare
    # print(old_value)
    # get new value
    new_value = output_preprocessing['vehicle_weights_typical'][key]
    # print(new_value)
    
    # get earliest year to enable comparison
    highest_val = max(new_value.index[0], old_value.index[0])
    
    assert (old_value.loc[highest_val:].sort_index(axis = 1) == new_value.loc[highest_val:].sort_index(axis = 1)).all().all()


#%% total nr of vehicles simple

# TODO: this gives major errors. Values are equal until & inlcusive 2018 and deviate from 2019 on

total_nr_vehicles.columns.name = 'type'

total_nr_vehicles = total_nr_vehicles.rename(columns={'Cargo Planes': 'Freight Planes',
                                                      'Cargo Trains' : 'Freight Trains',
                                                      'HST': 'High Speed Trains',
                                                      'Inland ships': 'Inland Ships'})

# bring total_nr_vehicles_simple from output preprocessing in same format as old data format
total_vehicles_new = output_preprocessing['total_nr_vehicles_simple']
total_vehicles_new = total_vehicles_new.melt(var_name = ['Type', 'DIM_1'], ignore_index = False).reset_index(names=['time']) # melt to be able to pivot
total_vehicles_new = total_vehicles_new.pivot(index=['time', 'DIM_1'], columns='Type').loc[1971:]  # pivot to same structure as total_nr_vehicles
total_vehicles_new = total_vehicles_new.droplevel(0, axis = 1) # drop uneseccary level
total_vehicles_new = total_vehicles_new.drop([27, 28], level=1) # drop empty and global column

for vehicle_type in total_vehicles_new.columns:
    print(vehicle_type)
    # these are not in total nr simple vehicles (are also not outputted to main, so they are skipped)
    if vehicle_type in {'Light Commercial Vehicles', 'Medium Freight Trucks', 
                        'Medium Ships', 'Heavy Freight Trucks', 
                        'Light Commercial Vehicles', 'Large Ships',
                        'Midi Buses', 'Regular Buses', 'Small Ships',
                        'Very Large Ships'}:
        continue
    
    
    print(total_vehicles_new[vehicle_type])
    print(total_nr_vehicles[vehicle_type])
    
    assert (total_vehicles_new[vehicle_type] == total_nr_vehicles[vehicle_type]).all().all()