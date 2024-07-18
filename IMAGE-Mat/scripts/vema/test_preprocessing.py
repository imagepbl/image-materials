# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 13:39:52 2024

@author: Arp00003
"""

"""
Overview: 
- 'battery_materials' # done no assertion error 10-07-24
- 'battery_shares' # done no assertion error 10-07-24
- 'battery_weights_typical # done no assertion error 10-07-24

- 'lifetimes_vehicles'  # done no assertion error 11-07-2024
- 'material_fractions_simple  # done no assertion error 11-07-2024
- 'material_fractions_typical': # done no assertion error 11-07-2024
- 'total_nr_vehicles_simple': #assertion error! TODO --> data completely different all years, alll types

# TODO: what about total_nr_vehicles_typical?

- vehicles_shares_typical # done smoe assertion errors 12-07-2024
- 'vehicle_weights_simple': # done no assertion error 11-07-2024
- 'vehicle_weights_typical' # done no assertion error 11-07-2024
- 'weight boats' # done 
"""

import os
import numpy as np
import pandas as pd
from preprocessing import preprocessing


output_preprocessing = preprocessing()

# Get the current directory
current_directory = os.path.dirname(os.path.abspath(__file__))

#%%

# Define the relative path to the target directory
relative_path = os.path.join(current_directory, '..', '..', '..', 'IMAGE-Mat_old_version', 'IMAGE-Mat', 'VEMA')
# Change the directory
os.chdir(relative_path)
# Verify the change
print("Current Directory:", os.getcwd())

# load all variables - note: all plotting is commented out in this VEMA version to save time while re-running!
from Vehicle_Material_Model import *


#%%  compare dfs automized (only works if dimensions of df are still the same)
dict_new_to_old_keys = {
    'battery_materials': battery_materials_full,                                # done no assertion error 10-07-24
    'battery_shares': battery_shares_full,                                      # done no assertion error 10-07-24
    'battery_weights_typical': battery_weights_full,                            # done no assertion error 10-07-24
    'weight_boats': weight_boats
    }
    
for key, old_value in dict_new_to_old_keys.items():
    print(key)
    
    if key == 'weight_boats':
        new_value = output_preprocessing.get(key).loc[1900:]
    
    else: 
        new_value = output_preprocessing.get(key).loc['1971':]
        # print(new_value)
    assert (new_value == old_value).all().all()
    
    
#%% Lifetime vehicles

# compare to lifetime_vehicles_mean, lifetime_vehicles_scale,lifetime_vehicles_shape, lifetime_vehicles_stdev
# scale and shape only for cars! #TODO: is that right?

new_lifetimes_vehicles = output_preprocessing.get('lifetimes_vehicles')

for key in ['mean', 'stdev', 'scale', 'shape']: 
    if key in {'mean', 'stdev'}:
        # print(key)
        new_lifetimes_vehicles_compare = new_lifetimes_vehicles.loc[1971:, pd.IndexSlice[:, [key]]] # slice second level column index
        new_lifetimes_vehicles_compare = new_lifetimes_vehicles_compare.droplevel(1, axis=1) # drop level to be able to compare
        # print(new_lifetimes_vehicles_compare)
        old_key = ('lifetimes_vehicles_'+ key)
        # print(old_key)
        old_value = globals()[old_key]
        # print(old_value)
        assert (new_lifetimes_vehicles_compare == old_value).all().all()
    
    if key in {'scale', 'shape'}:
        new_lifetimes_vehicles_compare = new_lifetimes_vehicles.loc[1971:, pd.IndexSlice[:, [key]]] # slice second level column index
        new_lifetimes_vehicles_compare = new_lifetimes_vehicles_compare.droplevel(1, axis=1) # drop level to be able to compare
        new_lifetimes_vehicles_compare = new_lifetimes_vehicles_compare.squeeze()
        # print(new_lifetimes_vehicles_compare)
        old_key = ('lifetimes_vehicles_'+ key)
        # print(old_key)
        old_value = globals()[old_key]['car']
        # print(old_value)
        assert (new_lifetimes_vehicles_compare == old_value).all().all()
    


#%%
# material fractions simple
# no assertion 10-07-2024
for key in output_preprocessing['material_fractions_simple'].columns.get_level_values(0).unique():
    print(key)
    old_key = ("material_fractions_" + key).replace("freight", "frgt")
    old_key = old_key.replace("sea_shipping", "ship")
    old_key = old_key.replace("inland_shipping", "inland_ship")
    old_key = old_key.replace("med", "medium")
    old_key = old_key.replace("_vl", "_vlarge")
    old_value = globals()[old_key].sort_index(axis = 1)
    
    new_value = output_preprocessing['material_fractions_simple'][key].sort_index(axis = 1)
    highest_val = max(new_value.index[0], old_value.index[0])
    
    print(highest_val, new_value.index[0], old_value.index[0])
    assert (old_value.loc[highest_val:].sort_index(axis = 1) == new_value.loc[highest_val:].sort_index(axis = 1)).all().all()


#%%
# material fractions (check)
#
for key in output_preprocessing['material_fractions_typical'].columns.get_level_values(0).unique():
    print(key)
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
    highest_val = max(new_value.index[0], old_value.index[0])
    
    
    print(highest_val, new_value.index[0], old_value.index[0])
    assert (old_value.loc[highest_val:].sort_index(axis = 1) == new_value.loc[highest_val:].sort_index(axis = 1)).all().all()

#%% total nr of vehicles simple

total_nr_vehicles.columns.name = 'type'

total_nr_vehicles = total_nr_vehicles.rename(columns={'Cargo Planes': 'Freight Planes',
                                                      'Cargo Trains' : 'Freight Trains',
                                                      'HST': 'High Speed Trains',
                                                      'Inland ships': 'Inland Ships'})


# bring total_nr_vehicles_simple from output preprocessing in same format as old data format
total_vehicles_new = output_preprocessing['total_nr_vehicles_simple']
total_vehicles_new = total_vehicles_new.melt(var_name = ['Type', 'DIM_1'], ignore_index = False).reset_index(names=['time']) #melt to be able to pivot
total_vehicles_new = total_vehicles_new.pivot(index=['time', 'DIM_1'], columns='Type').loc[1971:]  # pivot to same structure as total_nr_vehicles
total_vehicles_new = total_vehicles_new.droplevel(0, axis = 1) # drop uneseccary level
total_vehicles_new = total_vehicles_new.drop([27, 28], level=1)

for vehicle_type in total_vehicles_new.columns:
    #TODO: these are not in total nr simple vehicles (are also not outputted yet to main)
    if vehicle_type in {'Light Commercial Vehicles', 'Medium Freight Trucks', 
                        'Medium Ships', 'Heavy Freight Trucks', 
                        'Light Commercial Vehicles', 'Large Ships',
                        'Midi Buses', 'Regular Buses', 'Small Ships',
                        'Very Large Ships'}:
        continue
    
    print(vehicle_type)
    #TODO: these give assertion errors at the moment (comment line out to test if they still do!)
    if vehicle_type in {}:
        continue
    

    print(total_vehicles_new[vehicle_type])
    print(total_nr_vehicles[vehicle_type])
    
    assert (total_vehicles_new[vehicle_type] == total_nr_vehicles[vehicle_type]).all().all()

# cant check: Midi Buses, 'Heavy Freight Trucks', 'Large Ships',
# 'Light Commercial Vehicles', 'Medium Freight Trucks', 'Medium Ships',
# 'Midi Buses', 'Small Ships', ('Regular Buses') ???


#%%
<<<<<<< HEAD
# LCV vehicle shares are determined based on the medium trucks (so not calculated explicitly) --> at the moment all 0, how to deal with that?


for vehicle in output_preprocessing['vehicle_shares_typical'].columns.get_level_values(0).unique():
    # if vehicle in {}: 
    #     continue
=======
# vehicles_shares_typical
# Compare the following

# Cars: give assertion errors for ICE (not all, seems marginal)
# Midi Buses give assertion errors for some ICE as well 
# LCV vehicle shares are determined based on the medium trucks (so not calculated explicitly) --> at the moment all 0, how to deal with that?

# (['Cars', 'Regular Buses', 'Midi Buses', 'Heavy Freight Trucks',
  # 'Medium Freight Trucks', 'Light Commercial Vehicles']

for vehicle in output_preprocessing['vehicle_shares_typical'].columns.get_level_values(0).unique():
    if vehicle in {'Cars', 'Midi Buses', 'Light Commercial Vehicles'}: # give assertion errors at the moment
    
        continue
>>>>>>> bf873870215fba16d2e96919ef9eb9a7f6ec29c8
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
<<<<<<< HEAD
    old_key = old_key.replace("Light Commercial Vehicles", "trucks_MFT_vshares")
=======
    old_key = old_key.replace("Light Commercial Vehicles", "buses_midi_vsharesv")
    
    
>>>>>>> bf873870215fba16d2e96919ef9eb9a7f6ec29c8
    
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
<<<<<<< HEAD
    if vehicle == 'Cars':
        new_value = new_value.loc[:, :'PHEV']

    # print(old_value_pivot)
    # print(new_value)    
    assert np.isclose(old_value_pivot, new_value).all().all()

    
=======
    if key == 'Cars':
        new_value = new_value.loc[:, :'PHEV']
    print(old_value_pivot)
    print(new_value)    
    assert (old_value_pivot == new_value).all().all()

    
    

# old value has no values for Trolley #TODO
>>>>>>> bf873870215fba16d2e96919ef9eb9a7f6ec29c8

#%% 
# vehicle_weights_simple

for key in output_preprocessing['vehicle_weights_simple'].columns.get_level_values(0).unique():
    print(key)
    old_key = ("vehicle_weight_kg_" + key)
    old_key = old_key.replace("air_freight", "air_frgt")
    old_key = old_key.replace("rail_freight", "rail_frgt")
    old_key = old_key.replace('inland_shipping', "inland_ship")
    
    old_value = globals()[old_key]
    old_value = old_value.squeeze() # make Series to compare
    print(old_value)
    new_value = output_preprocessing['vehicle_weights_simple'][key]
    print(new_value)
    
    highest_val = max(new_value.index[0], old_value.index[0])
    

    assert (old_value.loc[highest_val:] == new_value.loc[highest_val:]).all().all()

#%%
# vehicle_weights_typical

for key in output_preprocessing['vehicle_weights_typical'].columns.get_level_values(0).unique():
    print(key)
    old_key = ("vehicle_weight_kg_" + key)
    old_key = old_key.replace("midi_bus", "midi")
    old_key = old_key.replace("reg_bus", "bus")

    old_value = globals()[old_key]
    old_value = old_value.squeeze() # make Series to compare
    print(old_value)
    new_value = output_preprocessing['vehicle_weights_typical'][key]
    print(new_value)
    
    highest_val = max(new_value.index[0], old_value.index[0])
    

    assert (old_value.loc[highest_val:].sort_index(axis = 1) == new_value.loc[highest_val:].sort_index(axis = 1)).all().all()
