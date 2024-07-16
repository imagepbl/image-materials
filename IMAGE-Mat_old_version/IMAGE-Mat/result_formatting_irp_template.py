# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 11:31:59 2020
@author: S. Deetman - deetman@cml.leidenuniv.nl
"""
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter, AutoMinorLocator)
import os
import seaborn as sns     # color palettes
import json
import copy

#choices & settings
material_select = "Cu"
end_year = 2060
year_select = list(range(1971,end_year+1))
year_select = [str(i) for i in year_select]
idx = pd.IndexSlice                     # needed for slicing multi-index
region_dict = {'Canada':1, 'US':2, 'Mexico':3, 'Rest C.Am.':4, 'Brazil':5, 'Rest S.Am.':6, 'N.Africa':7, 'W.Africa':8, 'E.Africa':9, 'South Africa':10, 'W.Europe':11, 'C.Europe':12, 'Turkey':13, 'Ukraine':14, 'Stan':15, 'Russia':16, 'M.East':17, 'India':18, 'Korea':19, 'China':20, 'SE.Asia':21, 'Indonesia':22, 'Japan':23, 'Oceania':24, 'Rest S.Asia':25, 'Rest S.Africa':26 }
region_colorset = ['#59250c','#d45b26','#f0c22b']
regions_steady   = [1,2,3,11,12,13,14,15,16,17,24]
regions_fastdv   = [4,5,6,7,8,9,10,18,19,21,22,25,26]
regions_chinjp   = [20,23]

#scenario selection
ssp = "SSP2"
variant = "2D"
scenario_variant = ssp + "_" + variant + "\\"

os.chdir("C:\\Users\\Admin\\surfdrive\\\Projects\\IRP\\GRO23\\Modelling\\2060")

if variant == "CP_RE" or variant == "2D_RE":
    folder_buildings    = "BUMA\\output\\SSP2_CP_RE\\"                                   # building folder is same for Baseline & Climate variants because there is no climate policy effect on floorspace, but it is different for the RE scenario                         
else:
    folder_buildings    = "BUMA\\output\\SSP2_CP\\"                                      # building folder is same for CP (Baseline) & 2D because there is no climate policy effect on floorspace
folder_electricity  = "ELMA\\output\\" + scenario_variant + "default\\"
folder_vehicles     = "VEMA\\output\\" + scenario_variant
folder_recycling    = "Recycling_scenarios\\output\\" + scenario_variant

# Read IRP output template (csv)
template           = pd.read_csv('template//in//IRP_IIASA_template_proposal_materials.csv', index_col=[0,1,2,3,4])

# Read output files on drivers (nr of vehicles, MW of generation & m2 of buildings), global results
vehicle_nr      = pd.read_csv(folder_vehicles +    'global_vehicle_nr.csv', index_col=0)
vehicle_nr_reg  = pd.read_csv(folder_vehicles +    'region_vehicle_nr.csv', index_col=[0,1])         # regional number of cars (in-use)
vehicle_in_reg  = pd.read_csv(folder_vehicles +    'region_vehicle_in.csv', index_col=[0,1])         # regional car sales (annual)
vehicle_out_reg = pd.read_csv(folder_vehicles +    'region_vehicle_out.csv', index_col=[0,1])        # regional car outflow (nr of vehicles scrapped annually)

vehicle_share_reg = pd.read_csv(folder_vehicles +    'car_type_share_regional.csv', index_col=[0,1]) # regional car-type share (e.g. % electric)
electricity_km    = pd.read_csv(folder_electricity + 'grid_length_HV_km.csv', index_col=[0,1])       # since the HV grid length growth is based on the growth of the generation capacity, we take it as the growth indicator here 
buildings_m2      = pd.read_csv(folder_buildings +   'sqmeters_output.csv',   index_col=[0,1,2,3])   # in MILLIONS of square meters

transport_pkm_tkm   = pd.read_csv(folder_vehicles +  'region_pkm_tkm.csv', index_col=[0,1])          # Travel & freight transport demand in pkms & tkms
people_per_building = pd.read_csv(folder_buildings + 'people_per_building_type.csv', index_col=[0,1]) # Million of people per building type

# Read output files on material use (Baseline)
vehicle_mat            = pd.read_csv(folder_vehicles +    'vehicle_materials_kt.csv'    , index_col=[0,1,2,3,4,5,6]) # in kt
electricity_stor_mat   = pd.read_csv(folder_electricity + 'stor_materials_output_kt.csv', index_col=[0,1,2,3,4,5])   # in kt
electricity_gcap_mat   = pd.read_csv(folder_electricity + 'gcap_materials_output_kt.csv', index_col=[0,1,2,3,4,5])   # in kt
electricity_grid_mat   = pd.read_csv(folder_electricity + 'grid_materials_output_kt.csv', index_col=[0,1,2,3,4,5])   # in kt
buildings_mat          = pd.read_csv(folder_buildings +   'material_output.csv'         , index_col=[0,1,2,3,4])     # in kt

# Read output files on recycling flows
inflow_mat_recycled    = pd.read_csv(folder_recycling +    'regional_demand.csv'        , index_col=[0,1,2])         # in kt
outflow_mat_recycled   = pd.read_csv(folder_recycling +    'regional_waste.csv'         , index_col=[0,1,2])         # in kt

#%% Direct output grouped & aggregated material results for IRPspint (GTEM/IMAGE common region definition)

materials_select_buildings = ['Aluminium', 'Cu', 'Concrete', 'Glass', 'Steel', 'Wood']
materials_select_vehicles  = ['Aluminium', 'Cu', 'Concrete', 'Glass', 'Steel', 'Plastics']
materials_select           = ['Aluminium', 'Cu', 'Concrete', 'Glass', 'Steel']
material_dict = {'aluminium': 'Aluminium', 'copper': 'Cu', 'concrete': 'Concrete', 'glass': 'Glass', 'steel': 'Steel', 'wood': 'Wood'}

# Material indicators
# First duplicate 1990 values for the historic time-period (given that they are only generated from 1990-onwards in ELMA)
for year in range(int(year_select[0]),1990+1):
    electricity_stor_mat[str(year)] = electricity_stor_mat['1990']
    
electricity_stor_mat = electricity_stor_mat.sort_index(axis=1, ascending=True)

electricity_irp_output   = pd.concat([electricity_stor_mat.loc[:,year_select], electricity_gcap_mat.loc[:,year_select], electricity_grid_mat.loc[:,year_select]]).rename(region_dict, axis='index', level=0).sum(level=[0,1,3,5], axis=0).loc[idx[:,:,:,materials_select],:].stack().unstack(level=0)
vehicle_irp_output       = vehicle_mat.sum(level=[0,1,3,4,5], axis=0).loc[idx[:,:,:,:,materials_select_vehicles],:].stack().unstack(level=0)
buildings_irp_output     = buildings_mat.sum(level=[0,1,3,4], axis=0).rename(material_dict, axis='index', level=3).loc[idx[:,:,:,materials_select_buildings],:].stack().unstack(level=0)

# Other indicators
buildings_m2_irp_output  = buildings_m2[year_select].sum(level=[0,1,2,3], axis=0).stack().unstack(level=0) #year_select is applied here, because buildings have a long tail

vehicle_share_irp_output = vehicle_share_reg.stack().unstack(level=1) 
vehicle_km_irp_output    = transport_pkm_tkm.stack().unstack(level=1)     # pkms & tkms (service)
vehicle_nr_irp_output    = vehicle_nr_reg.stack().unstack(level=1)        # stock
vehicle_in_irp_output    = vehicle_in_reg.stack().unstack(level=1)        # inflow
vehicle_out_irp_output   = vehicle_out_reg.stack().unstack(level=1)       # outflow

electricity_km_irp_output  = electricity_km.sum(axis=0, level=1).T.rename(region_dict, axis='index', level=0).loc[:,list(range(1990,end_year+1))]
people_per_building_irp_output = people_per_building[year_select]

#%% Output in NATIVE IMAGE region format
#group regions, return regions to the first level of the multi-index & years as columns. Finally: select years
# material files
electricity_irp_native = electricity_irp_output.stack().unstack(level=3).reorder_levels([3,0,1,2])[year_select]
vehicle_irp_native     = vehicle_irp_output.stack().unstack(level=4).reorder_levels([4,0,1,2,3])[year_select]
buildings_irp_native   = buildings_irp_output.stack().unstack(level=3).reorder_levels([3,0,1,2])[year_select]

# other indicators
buildings_m2_irp_native = buildings_m2_irp_output.stack().unstack(level=3)
vehicle_km_irp_native   = vehicle_km_irp_output.transpose().stack()[list(map(int, year_select))]
vehicle_nr_irp_native   = vehicle_nr_irp_output.transpose().stack()[list(map(int, year_select))]
vehicle_in_irp_native   = vehicle_in_irp_output.transpose().stack()[list(map(int, year_select))]
vehicle_out_irp_native  = vehicle_out_irp_output.transpose().stack()[list(map(int, year_select))]
car_share_irp_native    = vehicle_share_irp_output.unstack().transpose()[list(map(int, year_select))]

# Generate CSVs for output (MATERIALS)
electricity_irp_native.to_csv('output\\' + scenario_variant + 'Native\\electricity_materials_kt_SSP2_BL.csv')   # in kt
vehicle_irp_native.to_csv('output\\' + scenario_variant + 'Native\\vehicle_materials_kt_SSP2_BL.csv')           # in kt
buildings_irp_native.to_csv('output\\' + scenario_variant + 'Native\\building_materials_kt_SSP2_BL.csv')        # in kt

# Generate CSVs for output (OTHER INDICATORS)
car_share_irp_native.to_csv('output\\' + scenario_variant + 'Native\\car_type_share_SSP2_BL.csv')    # as a fraction, common regional formatting
vehicle_nr_irp_native.to_csv('output\\' + scenario_variant + 'Native\\vehicle_nr_SSP2_BL.csv')       # nr of vehicles, common regional formatting
buildings_m2_irp_native.to_csv('output\\' + scenario_variant + 'Native\\buildings_million_m2_SSP2_BL.csv')

# Compare total inflow per material with disaggregated inflow (inc. reuse/recycling)
select_mat = 'Steel'
total_inflow_irp_native = electricity_irp_native.loc[idx[:,'inflow',:,select_mat],:].sum(level=[0,3]) + vehicle_irp_native.loc[idx[:,'inflow',:,:,select_mat],:].sum(level=[0,4]) + buildings_irp_native.loc[idx[:,'inflow',:,select_mat],:].sum(level=[0,3])
total_inflow_recycling  = inflow_mat_recycled.sum(level=[0,2]).loc[idx[:,select_mat],:]
test_total_inflow       = total_inflow_irp_native - total_inflow_recycling[year_select]


#%% Output for IRP Reporting (single file)
pas_car   = ['BEV', 'FCV', 'HEV', 'ICE', 'PHEV']
road_veh  = ['HFT', 'LCV', 'MFT', 'bicycle', 'midi_bus', 'reg_bus', 'BEV', 'FCV', 'HEV', 'ICE', 'PHEV'] # total (including cars)
ships_veh = ['inland_shipping', 'sea_shipping_large', 'sea_shipping_med', 'sea_shipping_small', 'sea_shipping_vl']
rail_veh  = ['rail_hst', 'rail_reg', 'rail_freight']
air_veh   = ['air_freight', 'air_pas']
multi_fam   = ['appartments', 'semi-detached']
residential = ['rural', 'urban']

flows = ['inflow', 'outflow', 'stock']
flow_dict = {'stock':'Material Stock', 'inflow':'Material Demand', 'outflow':'Total Scrap'}     # dictionary for 3 flows (of MATERIALS)
prod_dict = {'stock':'Product Stock', 'inflow':'Stock Addition',  'outflow':'Stock Retirement'} # dictionary for 3 flows (of Product indicators)

mat_dict =  {'Steel':'Steel', 'Cu':'Non-Ferrous Metals|Copper', 'Aluminium':'Non-Ferrous Metals|Aluminium', 'Glass':'Glass', 'Plastics':'Chemicals|Plastics', 'Concrete':'Concrete', 'Wood':'Construction Wood'}

vehicle_km_irp_used   = vehicle_km_irp_native    # pkms & tkms (service)
vehicle_nr_irp_used   = vehicle_nr_irp_native    # stock   (nr of vehicles)
vehicle_in_irp_used   = vehicle_in_irp_native    # inflow  (nr of vehicles)
vehicle_out_irp_used  = vehicle_out_irp_native   # outflow (nr of vehicles)
buildings_m2_irp_used = buildings_m2_irp_native
electricity_irp_used  = electricity_irp_native
vehicle_irp_used      = vehicle_irp_native
buildings_irp_used    = buildings_irp_native

for region in range(1,27):
    template.loc[idx[:,:,region,'Product Stock|Transportation|Vehicles|Road|Passenger Cars',:],:] = vehicle_nr_irp_used.loc[idx[region,'Cars'],:].values          / 1000 # thousands of vehicles
    template.loc[idx[:,:,region,'Product Stock|Transportation|Vehicles|Road|Buses',:],:]          = vehicle_nr_irp_used.loc[idx[region,'Buses'],:].values         / 1000 # thousands of vehicles
    template.loc[idx[:,:,region,'Product Stock|Transportation|Vehicles|Road|Bicycles',:],:]       = vehicle_nr_irp_used.loc[idx[region,'Bikes'],:].values         / 1000 # thousands of vehicles
    template.loc[idx[:,:,region,'Product Stock|Transportation|Vehicles|Road|Trucks',:],:]         = vehicle_nr_irp_used.loc[idx[region,'Trucks'],:].values        / 1000 # thousands of vehicles
    template.loc[idx[:,:,region,'Product Stock|Transportation|Vehicles|Rail|Trains',:],:]         = vehicle_nr_irp_used.loc[idx[region,'Trains'],:].values        / 1000 # thousands of vehicles
    template.loc[idx[:,:,region,'Product Stock|Transportation|Vehicles|Rail|Cargo',:],:]          = vehicle_nr_irp_used.loc[idx[region,'Cargo Trains'],:].values  / 1000 # thousands of vehicles
    template.loc[idx[:,:,region,'Product Stock|Transportation|Air|Planes',:],:]                   = vehicle_nr_irp_used.loc[idx[region,'Planes'],:].values        / 1000 # thousands of vehicles
    template.loc[idx[:,:,region,'Product Stock|Transportation|Air|Cargo',:],:]                    = vehicle_nr_irp_used.loc[idx[region,'Cargo Planes'],:].values  / 1000 # thousands of vehicles
    template.loc[idx[:,:,region,'Product Stock|Transportation|Ships',:],:]                        = (vehicle_nr_irp_used.loc[idx[region,'Ships'],:].values + vehicle_nr_irp_used.loc[idx[region,'Inland ships'],:].values) / 1000 # thousands of vehicles

    template.loc[idx[:,:,region,'Stock Addition|Transportation|Vehicles|Road|Passenger Cars',:],:] = vehicle_in_irp_used.loc[idx[region,'Cars'],:].values          / 1000 # thousands of vehicles
    template.loc[idx[:,:,region,'Stock Addition|Transportation|Vehicles|Road|Buses',:],:]          = vehicle_in_irp_used.loc[idx[region,'Buses'],:].values         / 1000 # thousands of vehicles
    template.loc[idx[:,:,region,'Stock Addition|Transportation|Vehicles|Road|Bicycles',:],:]       = vehicle_in_irp_used.loc[idx[region,'Bikes'],:].values         / 1000 # thousands of vehicles
    template.loc[idx[:,:,region,'Stock Addition|Transportation|Vehicles|Road|Trucks',:],:]         = vehicle_in_irp_used.loc[idx[region,'Trucks'],:].values        / 1000 # thousands of vehicles
    template.loc[idx[:,:,region,'Stock Addition|Transportation|Vehicles|Rail|Trains',:],:]         = vehicle_in_irp_used.loc[idx[region,'Trains'],:].values        / 1000 # thousands of vehicles
    template.loc[idx[:,:,region,'Stock Addition|Transportation|Vehicles|Rail|Cargo',:],:]          = vehicle_in_irp_used.loc[idx[region,'Cargo Trains'],:].values  / 1000 # thousands of vehicles
    template.loc[idx[:,:,region,'Stock Addition|Transportation|Air|Planes',:],:]                   = vehicle_in_irp_used.loc[idx[region,'Planes'],:].values        / 1000 # thousands of vehicles
    template.loc[idx[:,:,region,'Stock Addition|Transportation|Air|Cargo',:],:]                    = vehicle_in_irp_used.loc[idx[region,'Cargo Planes'],:].values  / 1000 # thousands of vehicles
    template.loc[idx[:,:,region,'Stock Addition|Transportation|Ships',:],:]                        = (vehicle_in_irp_used.loc[idx[region,'Ships'],:].values + vehicle_in_irp_used.loc[idx[region,'Inland ships'],:].values) / 1000 # thousands of vehicles       / 1000 # thousands of vehicles

    template.loc[idx[:,:,region,'Stock Retirement|Transportation|Vehicles|Road|Passenger Cars',:],:] = vehicle_out_irp_used.loc[idx[region,'Cars'],:].values          / 1000 # thousands of vehicles
    template.loc[idx[:,:,region,'Stock Retirement|Transportation|Vehicles|Road|Buses',:],:]          = vehicle_out_irp_used.loc[idx[region,'Buses'],:].values         / 1000 # thousands of vehicles
    template.loc[idx[:,:,region,'Stock Retirement|Transportation|Vehicles|Road|Bicycles',:],:]       = vehicle_out_irp_used.loc[idx[region,'Bikes'],:].values         / 1000 # thousands of vehicles
    template.loc[idx[:,:,region,'Stock Retirement|Transportation|Vehicles|Road|Trucks',:],:]         = vehicle_out_irp_used.loc[idx[region,'Trucks'],:].values        / 1000 # thousands of vehicles
    template.loc[idx[:,:,region,'Stock Retirement|Transportation|Vehicles|Rail|Trains',:],:]         = vehicle_out_irp_used.loc[idx[region,'Trains'],:].values        / 1000 # thousands of vehicles
    template.loc[idx[:,:,region,'Stock Retirement|Transportation|Vehicles|Rail|Cargo',:],:]          = vehicle_out_irp_used.loc[idx[region,'Cargo Trains'],:].values  / 1000 # thousands of vehicles
    template.loc[idx[:,:,region,'Stock Retirement|Transportation|Air|Planes',:],:]                   = vehicle_out_irp_used.loc[idx[region,'Planes'],:].values        / 1000 # thousands of vehicles
    template.loc[idx[:,:,region,'Stock Retirement|Transportation|Air|Cargo',:],:]                    = vehicle_out_irp_used.loc[idx[region,'Cargo Planes'],:].values  / 1000 # thousands of vehicles
    template.loc[idx[:,:,region,'Stock Retirement|Transportation|Ships',:],:]                        = (vehicle_out_irp_used.loc[idx[region,'Ships'],:].values + vehicle_out_irp_used.loc[idx[region,'Inland ships'],:].values) / 1000 # thousands of vehicles

    template.loc[idx[:,:,region,'Services|Transportation|Road|Passenger Cars',:],:] = vehicle_km_irp_used.loc[idx[region,'Cars'],:].values          / 1000000000 # from pkm to Gpkm (annual)
    template.loc[idx[:,:,region,'Services|Transportation|Road|Buses',:],:]          = vehicle_km_irp_used.loc[idx[region,'Buses'],:].values         / 1000000000 # from pkm to Gpkm (annual)
    template.loc[idx[:,:,region,'Services|Transportation|Road|Bicycles',:],:]       = vehicle_km_irp_used.loc[idx[region,'Bikes'],:].values         / 1000000000 # from pkm to Gpkm (annual)
    template.loc[idx[:,:,region,'Services|Transportation|Road|Trucks',:],:]         = vehicle_km_irp_used.loc[idx[region,'Trucks'],:].values        / 1000000000 # from tkm to Gtkm (annual)
    template.loc[idx[:,:,region,'Services|Transportation|Rail|Passenger',:],:]      = vehicle_km_irp_used.loc[idx[region,'Trains'],:].values        / 1000000000 # from pkm to Gpkm (annual)
    template.loc[idx[:,:,region,'Services|Transportation|Rail|Freight',:],:]        = vehicle_km_irp_used.loc[idx[region,'Cargo Trains'],:].values  / 1000000000 # from tkm to Gtkm (annual)
    template.loc[idx[:,:,region,'Services|Transportation|Air|Passenger',:],:]       = vehicle_km_irp_used.loc[idx[region,'Planes'],:].values        / 1000000000 # from pkm to Gpkm (annual)
    template.loc[idx[:,:,region,'Services|Transportation|Air|Freight',:],:]         = vehicle_km_irp_used.loc[idx[region,'Cargo Planes'],:].values  / 1000000000 # from tkm to Gtkm (annual)
    template.loc[idx[:,:,region,'Services|Transportation|Ships',:],:]               = (vehicle_km_irp_used.loc[idx[region,'Ships'],:].values + vehicle_km_irp_used.loc[idx[region,'Inland ships'],:].values) / 1000000000 # from tkm to Gtkm (annual)
    
    template.loc[idx[:,:,region,'Services|People Housed|Single-Family Houses',:],:] = people_per_building_irp_output.loc[idx['det',region],:].values             # million of people per housing type
    template.loc[idx[:,:,region,'Services|People Housed|Multi-Family Houses',:],:]  = people_per_building_irp_output.loc[idx['sem',region],:].values + people_per_building_irp_output.loc[idx['app',region],:].values # million of people per housing type
    template.loc[idx[:,:,region,'Services|People Housed|Residential Towers',:],:]   = people_per_building_irp_output.loc[idx['hig',region],:].values             # million of people per housing type

    # regional total material flows (disaggregated to: recycled, reused, virgin input & losses) units: original from kt/yr to Mt/yr in template
    for material in ['Steel', 'Aluminium', 'Glass', 'Wood', 'Concrete', 'Cu']: 
        template.loc[idx[:,:,region,'Total flows|Virgin input|' + mat_dict[material],:],:] = inflow_mat_recycled.loc[idx[region,'virgin input', material],year_select].values / 1000 # from kt/yr to Mt/yr
        template.loc[idx[:,:,region,'Total flows|Recycled|' + mat_dict[material],:],:]     = (inflow_mat_recycled.loc[idx[region,'regional recycling', material],year_select].values / 1000) + (inflow_mat_recycled.loc[idx[region,'global recycling', material],year_select].values / 1000) 
        template.loc[idx[:,:,region,'Total flows|Reused|' + mat_dict[material],:],:]       = inflow_mat_recycled.loc[idx[region,'reused', material],year_select].values / 1000 # from kt/yr to Mt/yr
        template.loc[idx[:,:,region,'Total flows|Losses|' + mat_dict[material],:],:]       = (outflow_mat_recycled.loc[idx[region,'losses', material],year_select].values / 1000) + (outflow_mat_recycled.loc[idx[region,'surplus losses', material],year_select].values / 1000) 
          
    # loop over flows (stock, in-/outflow) to fill the related indictors
    for flow in flows:
        
        # First, for PRODUCT related indicators 
        template.loc[idx[:,:,region, prod_dict[flow] + '|Buildings|Residential|Single-Family Houses',:],:] = buildings_m2_irp_used.loc[idx[ flow, 'detached',  :, region],:].sum(axis=0).values * 1000000     # millions m2 to square meters  
        template.loc[idx[:,:,region, prod_dict[flow] + '|Buildings|Residential|Multi-Family Houses',:],:]  = buildings_m2_irp_used.loc[idx[ flow,  multi_fam,  :, region],:].sum(axis=0).values * 1000000     # millions m2 to square meters  
        template.loc[idx[:,:,region, prod_dict[flow] + '|Buildings|Residential|Residential Towers',:],:]   = buildings_m2_irp_used.loc[idx[ flow, 'high-rise', :, region],:].sum(axis=0).values * 1000000     # millions m2 to square meters  
        template.loc[idx[:,:,region, prod_dict[flow] + '|Buildings|Commercial|Retail',:],:]                = buildings_m2_irp_used.loc[idx[ flow, 'retail',    :, region],:].values * 1000000                 # millions m2 to square meters  
        template.loc[idx[:,:,region, prod_dict[flow] + '|Buildings|Commercial|Hotels',:],:]                = buildings_m2_irp_used.loc[idx[ flow, 'hotels',    :, region],:].values * 1000000                 # millions m2 to square meters  
        template.loc[idx[:,:,region, prod_dict[flow] + '|Buildings|Commercial|Offices',:],:]               = buildings_m2_irp_used.loc[idx[ flow, 'office',    :, region],:].values * 1000000                 # millions m2 to square meters  
        template.loc[idx[:,:,region, prod_dict[flow] + '|Buildings|Commercial|Public Buildings',:],:]      = buildings_m2_irp_used.loc[idx[ flow, 'govern',    :, region],:].values * 1000000                 # millions m2 to square meters  
        
        # then for MATERIAL related indicators
        # in vehicles
        for material in ['Steel', 'Aluminium', 'Glass', 'Plastics']:
            template.loc[idx[:,:,region, flow_dict[flow] + '|Transportation|Vehicles|Road|Passenger Cars|'  + mat_dict[material],:],:] = vehicle_irp_used.loc[idx[region, flow, :, pas_car, material],:].sum(axis=0).values   /1000    # original unit: kt(/yr), required: Mt(/yr)
            template.loc[idx[:,:,region, flow_dict[flow] + '|Transportation|Vehicles|Road|'  + mat_dict[material],:],:]           = vehicle_irp_used.loc[idx[region, flow, :, road_veh, material],:].sum(axis=0).values  /1000    # original unit: kt(/yr), required: Mt(/yr)
            template.loc[idx[:,:,region, flow_dict[flow] + '|Transportation|Vehicles|Rail|'  + mat_dict[material],:],:]           = vehicle_irp_used.loc[idx[region, flow, :, rail_veh, material],:].sum(axis=0).values  /1000    # original unit: kt(/yr), required: Mt(/yr)
            template.loc[idx[:,:,region, flow_dict[flow] + '|Transportation|Air|'   + mat_dict[material],:],:]                    = vehicle_irp_used.loc[idx[region, flow, :, air_veh,  material],:].sum(axis=0).values  /1000    # original unit: kt(/yr), required: Mt(/yr)
            template.loc[idx[:,:,region, flow_dict[flow] + '|Transportation|Ships|' + mat_dict[material],:],:]                    = vehicle_irp_used.loc[idx[region, flow, :, ships_veh, material],:].sum(axis=0).values /1000    # original unit: kt(/yr), required: Mt(/yr)
        
        #copper is refered to as 'Cu' in the original data, so not included in the material loop (due to use of 'Copper' in the IRP variable name)
        template.loc[idx[:,:,region, flow_dict[flow] + '|Transportation|Vehicles|Road|Passenger Cars|Non-Ferrous Metals|Copper',:],:] = vehicle_irp_used.loc[idx[region, flow, :, pas_car, 'Cu'],:].sum(axis=0).values   /1000    # original unit: kt(/yr), required: Mt(/yr)
        template.loc[idx[:,:,region, flow_dict[flow] + '|Transportation|Vehicles|Road|Non-Ferrous Metals|Copper',:],:]           = vehicle_irp_used.loc[idx[region, flow, :, road_veh, 'Cu'],:].sum(axis=0).values  /1000    # original unit: kt(/yr), required: Mt(/yr)
        template.loc[idx[:,:,region, flow_dict[flow] + '|Transportation|Vehicles|Rail|Non-Ferrous Metals|Copper',:],:]           = vehicle_irp_used.loc[idx[region, flow, :, rail_veh, 'Cu'],:].sum(axis=0).values  /1000    # original unit: kt(/yr), required: Mt(/yr)
        template.loc[idx[:,:,region, flow_dict[flow] + '|Transportation|Air|Non-Ferrous Metals|Copper',:],:]                     = vehicle_irp_used.loc[idx[region, flow, :, air_veh,  'Cu'],:].sum(axis=0).values  /1000    # original unit: kt(/yr), required: Mt(/yr)
        template.loc[idx[:,:,region, flow_dict[flow] + '|Transportation|Ships|Non-Ferrous Metals|Copper',:],:]                   = vehicle_irp_used.loc[idx[region, flow, :, ships_veh, 'Cu'],:].sum(axis=0).values /1000    # original unit: kt(/yr), required: Mt(/yr)
        
        # in buildings 
        for material in ['Steel', 'Aluminium', 'Glass', 'Wood', 'Concrete']: 
            template.loc[idx[:,:,region, flow_dict[flow] + '|Buildings|Residential|' + mat_dict[material],:],:] = buildings_irp_used.loc[idx[region, flow, residential,  material],:].sum(axis=0).values /1000    # original unit: kt(/yr), required: Mt(/yr)
            template.loc[idx[:,:,region, flow_dict[flow] + '|Buildings|Commercial|' + mat_dict[material],:],:]  = buildings_irp_used.loc[idx[region, flow, 'commercial', material],:].values             /1000    # original unit: kt(/yr), required: Mt(/yr)
        
        template.loc[idx[:,:,region, flow_dict[flow] + '|Buildings|Residential|Non-Ferrous Metals|Copper',:],:] = buildings_irp_used.loc[idx[region, flow, residential,  'Cu'],:].sum(axis=0).values /1000   # original unit: kt(/yr), required: Mt(/yr)
        template.loc[idx[:,:,region, flow_dict[flow] + '|Buildings|Commercial|Non-Ferrous Metals|Copper',:],:]  = buildings_irp_used.loc[idx[region, flow, 'commercial', 'Cu'],:].values             /1000   # original unit: kt(/yr), required: Mt(/yr)
        
        # in electricity infrastructure
        for material in ['Steel', 'Aluminium', 'Glass', 'Concrete']:
            template.loc[idx[:,:,region, flow_dict[flow] + '|Electricity|Generation|'  + mat_dict[material],:],:] = electricity_irp_used.loc[idx[region, flow, 'generation', material],:].values   /1000 # original unit: kt(/yr), required: Mt(/yr)
            template.loc[idx[:,:,region, flow_dict[flow] + '|Electricity|Grid|'  + mat_dict[material],:],:]       = electricity_irp_used.loc[idx[region, flow, 'grid', material],:].values  /1000        # original unit: kt(/yr), required: Mt(/yr)
            template.loc[idx[:,:,region, flow_dict[flow] + '|Electricity|Storage|'  + mat_dict[material],:],:]    = electricity_irp_used.loc[idx[region, flow, 'storage', material],:].values  /1000     # original unit: kt(/yr), required: Mt(/yr)

        #copper is refered to as 'Cu' in the original data, so not included in the material loop (due to use of 'Copper' in the IRP variable name)
        template.loc[idx[:,:,region, flow_dict[flow] + '|Electricity|Generation|Non-Ferrous Metals|Copper',:],:]     = electricity_irp_used.loc[idx[region, flow, 'generation', 'Cu'],:].values   /1000 # original unit: kt(/yr), required: Mt(/yr)
        template.loc[idx[:,:,region, flow_dict[flow] + '|Electricity|Grid|Non-Ferrous Metals|Copper',:],:]     = electricity_irp_used.loc[idx[region, flow, 'grid', 'Cu'],:].values  /1000              # original unit: kt(/yr), required: Mt(/yr)
        template.loc[idx[:,:,region, flow_dict[flow] + '|Electricity|Storage|Non-Ferrous Metals|Copper',:],:]     = electricity_irp_used.loc[idx[region, flow, 'storage', 'Cu'],:].values  /1000        # original unit: kt(/yr), required: Mt(/yr)
    

# rename regions from integers (1-26) to required output pased on dictionary
region_dict = {1:'CAN',2:'USA',3:'MEX',4:'RCAM',5:'BRA',6:'RSAM',7:'NAF',8:'WAF',9:'EAF',10:'SAF',11:'WEU',12:'CEU',13:'TUR',14:'UKR',15:'STAN',16:'RUS',17:'ME',18:'INDIA',19:'KOR',20:'CHN',21:'SEAS',22:'INDO',23:'JAP',24:'OCE',25:'RSAS',26:'RSAF'}
template = template.rename(region_dict, axis='index', level=2)

# apply 5-yr moving average (without lag, by duplicating last year twice)
# & output as csv for years  2000-2060
select_years_output = ['2005','2010','2015','2020','2025','2030','2035','2040','2045','2050','2055','2060']
select_years_output = [str(i) for i in select_years_output]

template['2061'] = template['2062'] = template['2060']
template['1969'] = template['1970'] = template['1971']
template = template.sort_index(axis=1, ascending=True)
template_moving_average = template.rolling(window=5, axis=1, center=True).mean()

# add world-totals
add = template_moving_average.loc[idx[:,:,'RSAF',:,:],:]
add = add.rename({'RSAF':'World'}, axis='index', level=2)
for item in list(add.index.levels[3]):
    add.loc[idx[:,:,:,item,:],:] = template_moving_average.loc[idx[:,:,:,item,:],:].sum(axis=0, level=3).values

template_moving_average_world = template_moving_average.append(add)

template_clean = template_moving_average_world.dropna(axis=0, how='all')
template_clean[select_years_output].to_csv('template\\out\\' + scenario_variant + 'template_IRP_newest_' + variant + '.csv')

#%% Output for IRP reporting (requested indicators) 1) Regional aggregation

# Regional aggregation: IMAGE regions to 7 IRP regions
# Assumed input: index: [model, scenario, 26 regions, variable, unit], columns: [time]

inv_region_dict = {v: k for k, v in region_dict.items()}
inv_region_tupl = list(inv_region_dict .items())
region_list_irp = ['North America','Latin America + Caribbean', 'Europe', 'EECCA', 'Africa', 'West Asia', 'Asia + Pacific']
region_dict_irp = dict(zip(list(range(1,7+1)), region_list_irp))

def agg_image_regions(df):
    
    #df = template_clean[select_years_output]
    irp_list_nr = list(range(1,7+1))
    df2 = template_clean[select_years_output].reset_index()
    df_new = df2.replace(inv_region_dict)
    df_new = df_new.set_index(['model', 'scenario','region','variable','unit']).loc[idx[:,:,irp_list_nr,:],:]
    for col in df_new.columns:
        df_new[col].values[:] = 0
    
    df_new.loc[idx[:,:,1,:],:] = df.loc[idx[:,:,[region_dict[1],region_dict[2]],:,:],:].sum(level=[0,1,3,4]).values                                                 # North-America
    df_new.loc[idx[:,:,2,:],:] = df.loc[idx[:,:,[region_dict[3],region_dict[4],region_dict[5],region_dict[6]],:,:],:].sum(level=[0,1,3,4]).values                   # Latin America & Caribean
    df_new.loc[idx[:,:,3,:],:] = df.loc[idx[:,:,[region_dict[11],region_dict[12],region_dict[13]],:,:],:].sum(level=[0,1,3,4]).values                               # Europe
    df_new.loc[idx[:,:,4,:],:] = df.loc[idx[:,:,[region_dict[14],region_dict[15],region_dict[16]],:,:],:].sum(level=[0,1,3,4]).values                               # EECCA
    df_new.loc[idx[:,:,5,:],:] = df.loc[idx[:,:,[region_dict[7],region_dict[8],region_dict[9],region_dict[10],region_dict[26]],:,:],:].sum(level=[0,1,3,4]).values  # Africa
    df_new.loc[idx[:,:,6,:],:] = df.loc[idx[:,:,region_dict[17],:,:],:].values                                                                                      # West-Asia
    df_new.loc[idx[:,:,7,:],:] = df.loc[idx[:,:,[region_dict[18],region_dict[19],region_dict[20],region_dict[21],region_dict[22],region_dict[23],region_dict[24],region_dict[25]],:,:],:].sum(level=[0,1,3,4]).values # Asia & Pacific

    return df_new

template_clean_7reg = agg_image_regions(template_clean[select_years_output])
template_clean_7reg = template_clean_7reg.reset_index().replace(region_dict_irp)
template_clean_7reg = template_clean_7reg.set_index(['model', 'scenario','region','variable','unit'])

# add world-totals
add = template_clean_7reg.loc[idx[:,:,'Asia + Pacific',:,:],:]
add = add.rename({'Asia + Pacific':'World'}, axis='index', level=2)
for item in list(add.index.levels[3]):
    add.loc[idx[:,:,:,item,:],:] = template_clean_7reg.loc[idx[:,:,:,item,:],:].sum(axis=0, level=3).values
template_clean_7reg = template_clean_7reg.append(add)

# Before generating cvs output on a 7 region level, check if the sum equals the sum of the IMAGE regions
test1 = template_clean_7reg.loc[idx[:,:,:,'Product Stock|Transportation|Vehicles|Road|Passenger Cars',:],'2005'].sum(axis=0) 
test2 = template_clean.loc[idx[:,:,:,'Product Stock|Transportation|Vehicles|Road|Passenger Cars',:],'2005'].sum()
test  = test1 - test2 # should be 0

template_clean_7reg.to_csv('template\\out\\' + scenario_variant + 'template_IRP_newest_' + variant + '_7reg.csv')

#%% Output for IRP reporting (requested indicators) 2) Income group aggregation

# output in income group format
# income_dict = {1:'CAN',2:'USA',3:'MEX',4:'RCAM',5:'BRA',6:'RSAM',7:'NAF',8:'WAF',9:'EAF',10:'SAF',11:'WEU',12:'CEU',13:'TUR',14:'UKR',15:'STAN',16:'RUS',17:'ME',18:'INDIA',19:'KOR',20:'CHN',21:'SEAS',22:'INDO',23:'JAP',24:'OCE',25:'RSAS',26:'RSAF'}

# def agg_image_income_groups(df):
#    return df_new

#%% Output to IMAGE 

# reverse dictionary to return IMAGE regions as integer
region_dict = {1:'CAN',2:'USA',3:'MEX',4:'RCAM',5:'BRA',6:'RSAM',7:'NAF',8:'WAF',9:'EAF',10:'SAF',11:'WEU',12:'CEU',13:'TUR',14:'UKR',15:'STAN',16:'RUS',17:'ME',18:'INDIA',19:'KOR',20:'CHN',21:'SEAS',22:'INDO',23:'JAP',24:'OCE',25:'RSAS',26:'RSAF'}
template = template.rename(region_dict, axis='index', level=2)

# select categories for total demand (inflow Mt/yr)
dem_categories =                                  \
['Material Demand|Transportation|Vehicles|Road|', \
'Material Demand|Transportation|Vehicles|Rail|',  \
'Material Demand|Transportation|Air|',            \
'Material Demand|Transportation|Ships|',          \
'Material Demand|Buildings|Residential|',         \
'Material Demand|Buildings|Commercial|',          \
'Material Demand|Electricity|Generation|',        \
'Material Demand|Electricity|Grid|',              \
'Material Demand|Electricity|Storage|'            ]

# select categories for virgin raw material input (i.e the fraction not reused/recycled)
vir_categories = ['Total flows|Virgin input|']
       
material = 'Steel'
variables_vir = vir_categories[0] + material
variables_dem = []
for catergory in dem_categories:
    variables_dem.append(catergory + material)

# Materials in Mt/yr
total_demand_steel = template_clean.loc[idx[:,:,:,variables_dem,:],year_select].sum(level=[2,4])
virgin_input_steel = template_clean.loc[idx[:,:,:,variables_vir,:],year_select].sum(level=[2,4])

total_demand_steel.to_csv('template\\out\\' + scenario_variant + 'IMAGE\\' + ssp + '_' + variant + '_demand_steel.csv')
virgin_input_steel.to_csv('template\\out\\' + scenario_variant + 'IMAGE\\' + ssp + '_' + variant + '_virgin_steel.csv')

material = 'Concrete'
variables_vir = vir_categories[0] + material
variables_dem = []
for catergory in dem_categories:
    variables_dem.append(catergory + material)

total_demand_concr = template_clean.loc[idx[:,:,:,variables_dem,:],year_select].sum(level=[2,4])
virgin_input_concr = template_clean.loc[idx[:,:,:,variables_vir,:],year_select].sum(level=[2,4])

total_demand_concr.to_csv('template\\out\\' + scenario_variant + 'IMAGE\\' + ssp + '_' + variant + '_demand_concrete.csv')
virgin_input_concr.to_csv('template\\out\\' + scenario_variant + 'IMAGE\\' + ssp + '_' + variant + '_virgin_concrete.csv')
