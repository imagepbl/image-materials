# %%
import pandas as pd
from preprocessing import preprocessing
from modelling_functions import (
                                inflow_outflow_dynamic_np,
                                inflow_outflow_typical_np,
                                nr_by_cohorts_to_materials_simple_np,
                                nr_by_cohorts_to_materials_typical_np
                                )
from constants import ( 
                        START_YEAR, END_YEAR,
                        columns_vehicle_output,
                        REGIONS,
                        OUTPUT_FOLDER
                       )

# Core modelling of stock dynamics & material use, assumes input as pandas dataFrames
# 1. Add assumptions on historic development
# 2. Run stock calculations, inflow and outflow and stocks by cohort
#    Input:  stock and lifetimes
#    Output: inflow and outflow and stocks by cohort
# 3. Intermediate export of results
# 4. Material calculations
# 5. Preparing output

# %%
idx = pd.IndexSlice
preprocessing_results = preprocessing()

all_keys = list(preprocessing_results['total_nr_vehicles_simple'].columns.levels[0].unique())

key_map_simple = {}
key_map_simple['Passenger Planes'] = 'air_pas'
key_map_simple['Bikes'] = 'bicycle'
key_map_simple['Freight Planes'] = 'air_freight'
key_map_simple['Freight Trains'] = 'rail_freight'
key_map_simple['High Speed Trains'] = 'rail_hst'
key_map_simple['Inland Ships'] = 'inland_shipping'
key_map_simple['Large Ships'] = 'sea_shipping_large'
key_map_simple['Medium Ships'] = 'sea_shipping_med'
key_map_simple['Small Ships'] = 'sea_shipping_small'
key_map_simple['Trains'] = 'rail_reg'
key_map_simple['Very Large Ships'] = 'sea_shipping_vl'

key_map_typical = {}
key_map_typical['Cars'] = 'car'
key_map_typical['Light Commercial Vehicles'] = 'LCV'
key_map_typical['Medium Freight Trucks'] = 'MFT'
key_map_typical['Heavy Freight Trucks'] = 'HFT'
key_map_typical['Midi Buses'] = 'midi_bus'
key_map_typical['Regular Buses'] = 'reg_bus'

#%% INFLOW-OUTFLOW calculations using the ODYM Dynamic Stock Model (DSM) as a function

##################### DYNAMIC MODEL (runtime: ca. 30 sec) #######################################
# Calculate the NUMBER of vehicles, total for inflow & by cohort for stock & outflow
# first only for simple vehicles

region_selection = list(range(1,27)) #TODO: Change to 26 or 28 regions, decide later (set to 26 for now)

vehicle_stocks_and_flows_simple = {}
for key in key_map_simple:
    vehicle_stocks_and_flows_simple[key] = inflow_outflow_dynamic_np(
            preprocessing_results['total_nr_vehicles_simple'].loc[:, idx[key, region_selection]].to_numpy(),
            preprocessing_results['lifetimes_vehicles'].loc[:, idx[key_map_simple[key], 'mean']],
            preprocessing_results['lifetimes_vehicles'].loc[:, idx[key_map_simple[key], 'stdev']],
            'FoldedNormal')

vehicle_stocks_and_flows_typical = {}
for key in key_map_typical:
    vehicle_stocks_and_flows_typical[key] = inflow_outflow_typical_np(
            preprocessing_results['total_nr_vehicles_simple'].loc[:, idx[key, region_selection]].droplevel(0,axis=1),
            preprocessing_results['lifetimes_vehicles'].loc[:, idx[key_map_typical[key], 'mean']],
            preprocessing_results['lifetimes_vehicles'].loc[:, idx[key_map_typical[key], 'stdev']],
            'FoldedNormal', preprocessing_results['vehicle_shares_typical'][key])

#%% Intermediate export of inflow & outflow of vehicles (for IRP database) ###############
""" TODO move to postprocessing?

region_list = list(range(1,27))
last_years  = END_YEAR+1 - START_YEAR
index = pd.MultiIndex.from_product([list(range(START_YEAR, END_YEAR+1)), region_list], names = ['years','regions'])
total_nr_vehicles_in = pd.DataFrame(index=index, columns=columns_vehicle_output)
total_nr_vehicles_in['Buses']        = vehicle_stocks_and_flows_typical['Regular Buses'][0].sum(0)[:,-last_years:].flatten(order='F') + vehicle_stocks_and_flows_typical['Midi Buses'][0].sum(0)[:,-last_years:].flatten(order='F')  #flattening a numpy array in the expected order (year columns first)
total_nr_vehicles_in['Trains']       = vehicle_stocks_and_flows_simple['Trains'][0][-last_years:,:].flatten(order='C')  # for simple arrays (no vehicle types) the column order is reversed
total_nr_vehicles_in['High Speed Trains']          = vehicle_stocks_and_flows_simple['High Speed Trains'][0][-last_years:,:].flatten(order='C')
total_nr_vehicles_in['Cars']         = vehicle_stocks_and_flows_typical['Cars'][0].sum(0)[:,-last_years:].flatten(order='F')
total_nr_vehicles_in['Passenger Planes']       = vehicle_stocks_and_flows_simple['Planes'][0][-last_years:,:].flatten(order='C')
total_nr_vehicles_in['Bikes']        = vehicle_stocks_and_flows_simple['Bikes'][0][-last_years:,:].flatten(order='C')
total_nr_vehicles_in['Trucks']       = vehicle_stocks_and_flows_typical['Heavy Freight Trucks'][0].sum(0)[:,-last_years:].flatten(order='F') + vehicle_stocks_and_flows_typical['Medium Freight Trucks'][0].sum(0)[:,-last_years:].flatten(order='F') + vehicle_stocks_and_flows_typical['Light Commercial Vehicles'][0].sum(0)[:,-last_years:].flatten(order='F')
total_nr_vehicles_in['Freight Trains'] = vehicle_stocks_and_flows_simple['Freight Trains'][0][-last_years:,:].flatten(order='C')
total_nr_vehicles_in['Small Ships']        = vehicle_stocks_and_flows_simple['Small Ships'][0][-last_years:,:].flatten(order='C') + vehicle_stocks_and_flows_simple['Medium Ships'][0][-last_years:,:].flatten(order='C') + vehicle_stocks_and_flows_simple['Large Ships'][0][-last_years:,:].flatten(order='C') + vehicle_stocks_and_flows_simple['Very Large Ships'][0][-last_years:,:].flatten(order='C')
total_nr_vehicles_in['Inland ships'] = vehicle_stocks_and_flows_simple['Inland Ships'][0][-last_years:,:].flatten(order='C')  
total_nr_vehicles_in['Freight Planes'] = vehicle_stocks_and_flows_simple['Freight Planes'][0][-last_years:,:].flatten(order='C')

total_nr_vehicles_in.to_csv(OUTPUT_FOLDER.joinpath('region_vehicle_in.csv'), index=True) # regional nr of vehicles sold (annually)

total_nr_vehicles_out = pd.DataFrame(index=index, columns=columns_vehicle_output)
total_nr_vehicles_out['Buses']        = vehicle_stocks_and_flows_typical['Regular Buses'][1].sum(0).sum(-1)[:,-last_years:].flatten(order='F') + vehicle_stocks_and_flows_typical['Midi Buses'][1].sum(0).sum(-1)[:,-last_years:].flatten(order='F')  #flattening a numpy array in the expected order (year columns first)
total_nr_vehicles_out['Trains']       = vehicle_stocks_and_flows_simple['Trains'][1].sum(-1)[:,-last_years:].flatten(order='F')  # for simple arrays (no vehicle types) the column order is reversed
total_nr_vehicles_out['High Speed Trains']          = vehicle_stocks_and_flows_simple['High Speed Trains'][1].sum(-1)[:,-last_years:].flatten(order='F')
total_nr_vehicles_out['Cars']         = vehicle_stocks_and_flows_typical['Cars'][1].sum(0).sum(-1)[:,-last_years:].flatten(order='F')
total_nr_vehicles_out['Passenger Planes']       = vehicle_stocks_and_flows_simple['Planes'][1].sum(-1)[:,-last_years:].flatten(order='F')
total_nr_vehicles_out['Bikes']        = vehicle_stocks_and_flows_simple['Bikes'][1].sum(-1)[:,-last_years:].flatten(order='F')
total_nr_vehicles_out['Trucks']       = vehicle_stocks_and_flows_typical['Heavy Freight Trucks'][1].sum(0).sum(-1)[:,-last_years:].flatten(order='F') + vehicle_stocks_and_flows_typical['Medium Freight Trucks'][1].sum(0).sum(-1)[:,-last_years:].flatten(order='F') + vehicle_stocks_and_flows_typical['Light Commercial Vehicles'][1].sum(0).sum(-1)[:,-last_years:].flatten(order='F')
total_nr_vehicles_out['Cargo Trains'] = vehicle_stocks_and_flows_simple['Freight Trains'][1].sum(-1)[:,-last_years:].flatten(order='F')
total_nr_vehicles_out['Ships']        = vehicle_stocks_and_flows_simple['Small Ships'][1].sum(-1)[:,-last_years:].flatten(order='F') + vehicle_stocks_and_flows_simple['Medium Ships'][1].sum(-1)[:,-last_years:].flatten(order='F') + vehicle_stocks_and_flows_simple['Large Ships'][1].sum(-1)[:,-last_years:].flatten(order='F') + vehicle_stocks_and_flows_simple['Very Large Ships'][1].sum(-1)[:,-last_years:].flatten(order='F')
total_nr_vehicles_out['Inland ships'] = vehicle_stocks_and_flows_simple['Inland Ships'][1].sum(-1)[:,-last_years:].flatten(order='F')
total_nr_vehicles_out['Cargo Planes'] = vehicle_stocks_and_flows_simple['Freight Planes'][1].sum(-1)[:,-last_years:].flatten(order='F')

total_nr_vehicles_out.to_csv(OUTPUT_FOLDER.joinpath('region_vehicle_out.csv'), index=True) # regional nr of vehicles sold (annually)

"""
#%% ################### MATERIAL CALCULATIONS ##########################################

#%% ############################################# RUNNING THE DYNAMIC STOCK FUNCTIONS  (runtime ca. 10 sec)  ###############################################

# run the material calculations on simple vehicles 
vehicle_materials_simple = {}
for key in key_map_simple:
    if key_map_simple[key] in preprocessing_results['vehicle_weights_simple'] and \
       key_map_simple[key] in preprocessing_results['material_fractions_simple']:
           data_in, data_out, stock_cohort = vehicle_stocks_and_flows_simple[key] 
           vehicle_materials_simple[key] =  nr_by_cohorts_to_materials_simple_np(
                                            data_in,  data_out, stock_cohort, 
                                            preprocessing_results['vehicle_weights_simple'][key_map_simple[key]].to_numpy(),
                                            preprocessing_results['material_fractions_simple'][key_map_simple[key]])

# run the material calculations on typical vehicles 
vehicle_materials_typical = {}
for key in key_map_typical:
    if key_map_typical[key] in preprocessing_results['vehicle_weights_typical'] and \
       key_map_typical[key] in preprocessing_results['material_fractions_typical']:
           data_in, data_out, stock_cohort = vehicle_stocks_and_flows_typical[key] 
           vehicle_materials_typical[key] = nr_by_cohorts_to_materials_typical_np(
                                            data_in,  data_out, stock_cohort, 
                                            preprocessing_results['vehicle_weights_typical'][key_map_typical[key]],
                                            preprocessing_results['material_fractions_typical'][key_map_typical[key]])

# Buses and trucks are not yet in the vehicle weight & material fraction data yet
'''
# run the simple material calculations on all vehicles                                                                                             # weight is passed as a series instead of a dataframe (pragmatic choice)
air_pas_mat_in,      air_pas_mat_out,      air_pas_mat_stock        = nr_by_cohorts_to_materials_simple_np(air_pas_in,      air_pas_out_coh,      air_pas_stock_coh,      vehicle_weight_kg_air_pas["air_pas"].to_numpy(),             material_fractions_air_pas)
rail_reg_mat_in,     rail_reg_mat_out,     rail_reg_mat_stock       = nr_by_cohorts_to_materials_simple_np(rail_reg_in,     rail_reg_out_coh,     rail_reg_stock_coh,     vehicle_weight_kg_rail_reg["rail_reg"].to_numpy(),           material_fractions_rail_reg)
rail_hst_mat_in,     rail_hst_mat_out,     rail_hst_mat_stock       = nr_by_cohorts_to_materials_simple_np(rail_hst_in,     rail_hst_out_coh,     rail_hst_stock_coh,     vehicle_weight_kg_rail_hst["rail_hst"].to_numpy(),           material_fractions_rail_hst)
bikes_mat_in,        bikes_mat_out,        bikes_mat_stock          = nr_by_cohorts_to_materials_simple_np(bikes_in,        bikes_out_coh,        bikes_stock_coh,        vehicle_weight_kg_bicycle["bicycle"].to_numpy(),             material_fractions_bicycle)

air_freight_mat_in,  air_freight_mat_out,  air_freight_mat_stock    = nr_by_cohorts_to_materials_simple_np(air_freight_in,  air_freight_out_coh,  air_freight_stock_coh,  vehicle_weight_kg_air_frgt["air_freight"].to_numpy(),        material_fractions_air_frgt)
rail_freight_mat_in, rail_freight_mat_out, rail_freight_mat_stock   = nr_by_cohorts_to_materials_simple_np(rail_freight_in, rail_freight_out_coh, rail_freight_stock_coh, vehicle_weight_kg_rail_frgt["rail_freight"].to_numpy(),      material_fractions_rail_frgt)
inland_ship_mat_in,  inland_ship_mat_out,  inland_ship_mat_stock    = nr_by_cohorts_to_materials_simple_np(inland_ship_in,  inland_ship_out_coh,  inland_ship_stock_coh,  vehicle_weight_kg_inland_ship["inland_shipping"].to_numpy(), material_fractions_inland_ship)

ship_small_mat_in,   ship_small_mat_out,   ship_small_mat_stock     = nr_by_cohorts_to_materials_simple_np(ship_small_in,   ship_small_out_coh,   ship_small_stock_coh,   weight_boats['Small'].to_numpy(),                            material_fractions_ship_small)
ship_medium_mat_in,  ship_medium_mat_out,  ship_medium_mat_stock    = nr_by_cohorts_to_materials_simple_np(ship_medium_in,  ship_medium_out_coh,  ship_medium_stock_coh,  weight_boats['Medium'].to_numpy(),                           material_fractions_ship_medium)
ship_large_mat_in,   ship_large_mat_out,   ship_large_mat_stock     = nr_by_cohorts_to_materials_simple_np(ship_large_in,   ship_large_out_coh,   ship_large_stock_coh,   weight_boats['Large'].to_numpy(),                            material_fractions_ship_large)
ship_vlarge_mat_in,  ship_vlarge_mat_out,  ship_vlarge_mat_stock    = nr_by_cohorts_to_materials_simple_np(ship_vlarge_in,  ship_vlarge_out_coh,  ship_vlarge_stock_coh,  weight_boats['Very Large'].to_numpy(),                       material_fractions_ship_vlarge)

# Calculate the weight of materials in the vehicles with sub-types: stock, inflow & outflow 
bus_regl_mat_in,     bus_regl_mat_out,     bus_regl_mat_stock       = nr_by_cohorts_to_materials_typical_np(bus_regl_in,   bus_regl_out_coh,   bus_regl_stock_coh,   vehicle_weight_kg_bus,   material_fractions_bus_reg)
bus_midi_mat_in,     bus_midi_mat_out,     bus_midi_mat_stock       = nr_by_cohorts_to_materials_typical_np(bus_midi_in,   bus_midi_out_coh,   bus_midi_stock_coh,   vehicle_weight_kg_midi,  material_fractions_bus_midi)
car_total_mat_in,    car_total_mat_out,    car_total_mat_stock      = nr_by_cohorts_to_materials_typical_np(car_in,        car_out_coh,        car_stock_coh,        vehicle_weight_kg_car,   material_fractions_car)    

trucks_HFT_mat_in,   trucks_HFT_mat_out,   trucks_HFT_mat_stock     = nr_by_cohorts_to_materials_typical_np(trucks_HFT_in, trucks_HFT_out_coh, trucks_HFT_stock_coh, vehicle_weight_kg_HFT,   material_fractions_truck_HFT)
trucks_MFT_mat_in,   trucks_MFT_mat_out,   trucks_MFT_mat_stock     = nr_by_cohorts_to_materials_typical_np(trucks_MFT_in, trucks_MFT_out_coh, trucks_MFT_stock_coh, vehicle_weight_kg_MFT,   material_fractions_truck_MFT)
trucks_LCV_mat_in,   trucks_LCV_mat_out,   trucks_LCV_mat_stock     = nr_by_cohorts_to_materials_typical_np(trucks_LCV_in, trucks_LCV_out_coh, trucks_LCV_stock_coh, vehicle_weight_kg_LCV,   material_fractions_truck_LCV)
'''

# Calculate the materials in batteries (in typical vehicles only) 
# For batteries this is a 2-step process, first (1) we pre-calculate the average material composition (of the batteries at inflow), based on a globally changing battery share & the changeing battery-specific material composition, both derived from our paper on (a.o.) storage in the electricity system.
# Then (2) we run the same function to derive the materials in vehicle batteries (based on changeing weight, composition & battery share)
# In doing so, we are no longer able to know the battery share per vehicle sub-type

battery_material_composition    = pd.DataFrame(index=pd.MultiIndex.from_product([list(preprocessing_results['battery_weights_typical'].columns.levels[0]), preprocessing_results['battery_materials'].index]), columns=pd.MultiIndex.from_product([preprocessing_results['battery_weights_typical'].columns.levels[1], preprocessing_results['battery_materials'].columns.levels[0]], names=['type','material']))
battery_weight_total_in         = pd.DataFrame(0, index=pd.MultiIndex.from_product([list(preprocessing_results['battery_weights_typical'].columns.levels[0]), preprocessing_results['battery_materials'].index]), columns=preprocessing_results['battery_weights_typical'].columns.levels[1])
battery_weight_total_stock      = pd.DataFrame(0, index=pd.MultiIndex.from_product([list(preprocessing_results['battery_weights_typical'].columns.levels[0]), preprocessing_results['battery_materials'].index]), columns=preprocessing_results['battery_weights_typical'].columns.levels[1])
battery_weight_regional_in      = pd.DataFrame(0, index=preprocessing_results['battery_materials'].index, columns=list(range(1,REGIONS+1)))
battery_weight_regional_out     = pd.DataFrame(0, index=preprocessing_results['battery_materials'].index, columns=list(range(1,REGIONS+1)))
battery_weight_regional_stock   = pd.DataFrame(0, index=preprocessing_results['battery_materials'].index, columns=list(range(1,REGIONS+1)))
 
# for now, 1 global battery market is assumed (so no difference between battery types for different vehicle sub-types), so the composition is duplicated over all vtypes
for vehicle in list(preprocessing_results['battery_weights_typical'].columns.levels[0]):
   for vtype in list(preprocessing_results['battery_weights_typical'].columns.levels[1]):
      for material in list(preprocessing_results['battery_materials'].columns.levels[0]):
         battery_material_composition.loc[idx[vehicle,:],idx[vtype,material]] = preprocessing_results['battery_shares'].mul(preprocessing_results['battery_materials'].loc[:,idx[material,:]].droplevel(0,axis=1)).sum(axis=1).values


# run the material calculations on typical vehicles 
battery_materials_typical = {}
for key in key_map_typical:
    if key_map_typical[key] in preprocessing_results['battery_weights_typical']:
           data_in, data_out, stock_cohort = vehicle_stocks_and_flows_typical[key] 
           battery_materials_typical[key] = nr_by_cohorts_to_materials_typical_np(
                                            data_in,  data_out, stock_cohort, 
                                            preprocessing_results['battery_weights_typical'][key_map_typical[key]],
                                            battery_material_composition.loc[idx[key_map_typical[key],:],:].droplevel(0))

'''
# Battery material calculations 
bus_regl_bat_in,     bus_regl_bat_out,     bus_regl_bat_stock       = nr_by_cohorts_to_materials_typical_np(bus_regl_in,  bus_regl_out_coh, bus_regl_stock_coh, battery_weights_full['reg_bus'],   battery_material_composition.loc[idx['reg_bus',:],:].droplevel(0))
bus_midi_bat_in,     bus_midi_bat_out,     bus_midi_bat_stock       = nr_by_cohorts_to_materials_typical_np(bus_midi_in,  bus_midi_out_coh, bus_midi_stock_coh, battery_weights_full['midi_bus'],  battery_material_composition.loc[idx['midi_bus',:],:].droplevel(0))
car_total_bat_in,    car_total_bat_out,    car_total_bat_stock      = nr_by_cohorts_to_materials_typical_np(car_in,       car_out_coh,      car_stock_coh,      battery_weights_full['car'],       battery_material_composition.loc[idx['car',:],idx[car_types,:]].droplevel(0))    #mind that cars don't have Trolleys, hence the additional selection

trucks_HFT_bat_in,   trucks_HFT_bat_out,   trucks_HFT_bat_stock     = nr_by_cohorts_to_materials_typical_np(trucks_HFT_in, trucks_HFT_out_coh, trucks_HFT_stock_coh, battery_weights_full['HFT'],  battery_material_composition.loc[idx['HFT',:],:].droplevel(0))
trucks_MFT_bat_in,   trucks_MFT_bat_out,   trucks_MFT_bat_stock     = nr_by_cohorts_to_materials_typical_np(trucks_MFT_in, trucks_MFT_out_coh, trucks_MFT_stock_coh, battery_weights_full['MFT'],  battery_material_composition.loc[idx['MFT',:],:].droplevel(0))
trucks_LCV_bat_in,   trucks_LCV_bat_out,   trucks_LCV_bat_stock     = nr_by_cohorts_to_materials_typical_np(trucks_LCV_in, trucks_LCV_out_coh, trucks_LCV_stock_coh, battery_weights_full['LCV'],  battery_material_composition.loc[idx['LCV',:],:].droplevel(0))
'''

# Sum the weight of the accounted materials (! so not total weight) in batteries by vehicle & vehicle type, output for figures
for vtype in list(preprocessing_results['battery_weights_typical'].columns.levels[1]):
    for key in key_map_typical: 
        battery_weight_total_in.loc[idx[key_map_typical[key],:],vtype] = battery_materials_typical[key][0].loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values
        battery_weight_total_stock.loc[idx[key_map_typical[key],:],vtype] = battery_materials_typical[key][2].loc[idx[:,:],idx[vtype,:]].sum(level=1).sum(axis=1, level=0).values
        
battery_weight_total_in.to_csv(OUTPUT_FOLDER / 'battery_weight_kg_in.csv', index=True)       # in kg
battery_weight_total_stock.to_csv(OUTPUT_FOLDER / 'battery_weight_kg_stock.csv', index=True) # in kg

# Regional battery weight (only the accounted materials), used in graph later on

for key in key_map_typical:   
    battery_weight_regional_in    += battery_materials_typical[key][0].sum(axis=0,level=1).sum(axis=1,level=1) # inflow = 1st output    
    battery_weight_regional_out   += battery_materials_typical[key][1].sum(axis=0,level=1).sum(axis=1,level=1) # outflow = 2nd output                                    
    battery_weight_regional_stock += battery_materials_typical[key][2].sum(axis=0,level=1).sum(axis=1,level=1) # stock = 3rd output
    


# %%
