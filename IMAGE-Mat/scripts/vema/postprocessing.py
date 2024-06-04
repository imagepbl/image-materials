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
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


def postprocess(OUTPUT_FOLDER
                ):

    # Sum the weight of the accounted materials (! so not total weight) in batteries by vehicle & vehicle type, output for figures
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

    battery_weight_total_in.to_csv(OUTPUT_FOLDER + '\\battery_weight_kg_in.csv', index=True)       # in kg
    battery_weight_total_stock.to_csv(OUTPUT_FOLDER + '\\battery_weight_kg_stock.csv', index=True) # in kg

    # Regional battery weight (only the accounted materials), used in graph later on
    battery_weight_regional_stock = bus_regl_bat_stock.sum(axis=0,level=1).sum(axis=1,level=1) + bus_midi_bat_stock.sum(axis=0,level=1).sum(axis=1,level=1) + trucks_LCV_bat_stock.sum(axis=0,level=1).sum(axis=1,level=1) + trucks_MFT_bat_stock.sum(axis=0,level=1).sum(axis=1,level=1) + trucks_HFT_bat_stock.sum(axis=0,level=1).sum(axis=1,level=1) + car_total_bat_stock.sum(axis=0,level=1).sum(axis=1,level=1)
    battery_weight_regional_in    = bus_regl_bat_in.sum(axis=0,level=1).sum(axis=1,level=1)    + bus_midi_bat_in.sum(axis=0,level=1).sum(axis=1,level=1)    + trucks_LCV_bat_in.sum(axis=0,level=1).sum(axis=1,level=1)    + trucks_MFT_bat_in.sum(axis=0,level=1).sum(axis=1,level=1)    + trucks_HFT_bat_in.sum(axis=0,level=1).sum(axis=1,level=1)    + car_total_bat_in.sum(axis=0,level=1).sum(axis=1,level=1)     
    battery_weight_regional_out   = bus_regl_bat_out.sum(axis=0,level=1).sum(axis=1,level=1)   + bus_midi_bat_out.sum(axis=0,level=1).sum(axis=1,level=1)   + trucks_LCV_bat_out.sum(axis=0,level=1).sum(axis=1,level=1)   + trucks_MFT_bat_out.sum(axis=0,level=1).sum(axis=1,level=1)   + trucks_HFT_bat_out.sum(axis=0,level=1).sum(axis=1,level=1)   + car_total_bat_out.sum(axis=0,level=1).sum(axis=1,level=1)                                      

    # ################################### Organise data for output ###########################################

    year_select = list(range(START_YEAR, END_YEAR + 1))

    # define 6 dataframes on materials in  stock, inflow & outflow X passenger vs. freight vehicles
    index   = pd.MultiIndex.from_product([year_select, list(range(1,REGIONS+1)), ['vehicle','battery'], labels_materials], names=['year', 'region', 'part', 'materials'])
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
        vehicle_materials_stock_passenger.loc[idx[:,:,'vehicle', material],'rail_hst']      = rail_hst_mat_stock.loc[idx[material,year_select],:].stack().reorder_levels([1,2,0]).values
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




    # combine dataframes for output

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

    vehicle_materials_out.to_csv(OUTPUT_FOLDER + '\\vehicle_materials_kt.csv', index=False) # in kt

    # ========================================== Visualization


