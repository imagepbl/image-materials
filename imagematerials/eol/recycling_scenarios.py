# -*- coding: utf-8 -*-
"""
Created on Wed 2 Nov 2022
@author: S. Deetman - deetman@cml.leidenuniv.nl
"""
import pandas as pd
import numpy as np
import seaborn as sns
import os

# general settings
end_year    = 2060
start_year  = 1971
year_select = [str(i) for i in list(range(start_year,end_year+1))]
stor_start  = 1990
year_stor   = [str(i) for i in list(range(start_year,stor_start+1))]
idx = pd.IndexSlice                     # needed for slicing multi-index
region_dict = {'Canada':1, 'US':2, 'Mexico':3, 'Rest C.Am.':4, 'Brazil':5, 'Rest S.Am.':6, 'N.Africa':7, 'W.Africa':8, 'E.Africa':9, 'South Africa':10, 'W.Europe':11, 'C.Europe':12, 'Turkey':13, 'Ukraine':14, 'Stan':15, 'Russia':16, 'M.East':17, 'India':18, 'Korea':19, 'China':20, 'SE.Asia':21, 'Indonesia':22, 'Japan':23, 'Oceania':24, 'Rest S.Asia':25, 'Rest S.Africa':26 }

#scenario selection
ssp = "SSP2"
variant = "2D_RE"
scenario_variant = ssp + "_" + variant + "\\"

os.chdir("C:\\Users\\Admin\\surfdrive\\\Projects\\IRP\\GRO23\\Modelling\\2060")

if variant == "CP_RE" or variant == "2D_RE":
    folder_buildings    = "BUMA\\output\\SSP2_CP_RE\\"                                   # building folder is same for BL & 2D because there is no climate policy effect on floorspace, but it is different for the RE scenario                         
else:
    folder_buildings    = "BUMA\\output\\SSP2_CP\\"                                      # building folder is same for BL & 2D because there is no climate policy effect on floorspace

folder_electricity  = "ELMA\\output\\" + scenario_variant + "\\default\\"
folder_vehicles     = "VEMA\\output\\" + scenario_variant
folder_assumptions  = "Recycling_scenarios\\assumptions\\" + scenario_variant + '\\'

# Read output files on materials 
vehicle_mat            = pd.read_csv(folder_vehicles +    'vehicle_materials_kt.csv'    , index_col=[0,1,2,3,4,5,6]) # in kt
electricity_stor_mat   = pd.read_csv(folder_electricity + 'stor_materials_output_kt.csv', index_col=[0,1,2,3,4,5])   # in kt
electricity_gcap_mat   = pd.read_csv(folder_electricity + 'gcap_materials_output_kt.csv', index_col=[0,1,2,3,4,5])   # in kt
electricity_grid_mat   = pd.read_csv(folder_electricity + 'grid_materials_output_kt.csv', index_col=[0,1,2,3,4,5])   # in kt
buildings_mat          = pd.read_csv(folder_buildings +   'material_output.csv'         , index_col=[0,1,2,3,4])     # in kt

collection_in = pd.read_csv(folder_assumptions + 'collection.csv', index_col=[0,1,2,3])                              # EoL collection rate (%)
reuse_in      = pd.read_csv(folder_assumptions + 'reuse.csv', index_col=[0,1,2,3])                                   # EoL reuse rate (%), this is reuse of material components (not te full product) reused or repurposed within the sector without additional processing)
recycling_in  = pd.read_csv(folder_assumptions + 'recycling.csv', index_col=[0,1,2,3])                               # EoL recycling rate (%), this is the fraction of materials that is recycled through processing and ends up at a global market. i.e. the ratio between the outflow (minus collection losses & reuse) and the amount that eventually replaces virgin use of the same material after recyling.

#%% Data pre-processing

# Set storage material use to 0 before stor_start
electricity_stor_mat[year_stor] = 0
electricity_stor_mat = electricity_stor_mat.reindex(sorted(electricity_stor_mat.columns), axis=1)

# define materials for output
materials_select_buildings = ['Aluminium', 'Cu', 'Concrete', 'Glass', 'Steel', 'Wood']
materials_select_vehicles  = ['Aluminium', 'Cu', 'Glass', 'Steel', 'Wood']
materials_select           = ['Aluminium', 'Cu', 'Concrete', 'Glass', 'Steel']
material_dict = {'aluminium': 'Aluminium', 'copper': 'Cu', 'concrete': 'Concrete', 'glass': 'Glass', 'steel': 'Steel', 'wood': 'Wood'}

# Material indicators grouped & aggregated material results for IRP
electricity_irp   = pd.concat([electricity_stor_mat.loc[:,year_select], electricity_gcap_mat.loc[:,year_select], electricity_grid_mat.loc[:,year_select]]).rename(region_dict, axis='index', level=0).sum(level=[0,1,3,5], axis=0).loc[idx[:,:,:,materials_select],:].sort_index()
vehicle_irp       = vehicle_mat.sum(level=[0,1,3,5], axis=0).loc[idx[:,:,:,materials_select_vehicles],:].sort_index()
buildings_irp     = buildings_mat.sum(level=[0,1,3,4], axis=0).rename(material_dict, axis='index', level=3).loc[idx[:,:,:,materials_select_buildings],year_select].sort_index()

# interpolate and return with time as columns (mind that 'Sector' Sub-level is dropped, as it is redundant)
def interpolate_waste_data(original):

    original_reduced = original.droplevel(2) # then remove the sector-level (=redundant)
    
    index = pd.MultiIndex.from_product([list(range(start_year,end_year+1)), original.index.levels[1], original.index.levels[3]])
    updated1 = original_reduced.reindex(index).unstack(level=0).stack(level=0)  # stack time as columns
    updated2  = updated1.interpolate(limit_direction='both', axis=1)   # interpolate both ways
    
    return updated2

collection = interpolate_waste_data(collection_in)
reuse      = interpolate_waste_data(reuse_in)
recycling  = interpolate_waste_data(recycling_in)

collection.index = collection.index.rename('materials', level=2)
reuse.index      = reuse.index.rename('materials', level=2)
recycling.index  = recycling.index.rename('materials', level=2)

#%% Take outflow indicators and apply E-o-L collection, reuse & recycling rates (functional re-use or recycling at the material level)

elec_cat = ['generation', 'storage','grid']
vehi_cat = ['passenger', 'freight']
buil_cat = ['commercial', 'urban', 'rural']
wast_cat = ['losses','reusable','recyclable','surplus losses']

"""
index = pd.MultiIndex.from_product([electricity_irp.index.levels[0], wast_cat, electricity_irp.index.levels[3]])
electricity_waste = pd.DataFrame(index=index, columns=list(range(start_year,end_year+1)))
vehicles_waste    = pd.DataFrame(index=index, columns=list(range(start_year,end_year+1)))
buildings_waste   = pd.DataFrame(index=index, columns=list(range(start_year,end_year+1)))

# Core
rate      = collection.loc[idx[:,elec_cat,materials_select],:].sort_index() 
volume    = electricity_irp.loc[idx[:,'outflow',:,:],:].droplevel(1).sort_index()
output    = volume.mul(1-rate.values)
"""

def waste_flows(waste_rate, waste_volume, cat, mat):
    #waste_rate   = reuse
    #waste_volume = remaining
        
    rate      = waste_rate.loc[idx[:, cat , mat],:].sort_index() 
    volume    = waste_volume.sort_index()
    output    = volume.mul(rate.values)
    return output                        # output is the volume of the waste flow after reduction

def assign_waste_flows(volume, collection, reuse, recycling, cat, mat):
    '''
    # DEBUG DEFINITIONS
    volume = vehicle_irp
    cat    = vehi_cat
    mat    = materials_select_vehicles 
    '''
    
    index = pd.MultiIndex.from_product([volume.index.levels[0], wast_cat, mat], names=['regions', 'waste', 'materials'])
    waste = pd.DataFrame(index=index, columns=list(range(start_year,end_year+1))).sort_index()    
    
    # assign waste flows (losses, reuse & recyling; based on sector & application specific rates)
    initial   = volume.loc[idx[:,'outflow',:,:],:].droplevel(1).sort_index()    # determine the initial size of the toal outflow
    losses    = initial - waste_flows(collection, initial, cat, mat)            # what is not collected becomes a loss (e.g. landfill or waste-combustion)
    remaining = initial - losses                                                # of the remainder (E-O-L outflow - what is not collected)
    reused    = waste_flows(reuse, remaining, cat, mat)                         # a part is available for reuse
    remaining = remaining - reused                                              # of what remains (after collection & reuse)
    recycled  = waste_flows(recycling, remaining, cat, mat)                     # another fraction is available for recycling
    losses    += (remaining - recycled)                                         # what is not recycled is added to the losses, so losses represents the sum of what is not collected and what is not recycled (reuse is subtracted before recycling, so technically losses also represents what is not reused)
    
    waste.loc[idx[:,'losses',:],:]   = losses.sum(level=[0,2]).values 
    waste.loc[idx[:,'reusable',:],:]   = reused.sum(level=[0,2]).values 
    waste.loc[idx[:,'recyclable',:],:] = recycled.sum(level=[0,2]).values
    
    #test1 = waste.sum(level=[0,2]).loc[idx[:,'Steel'],:]
    #test2 = vehicle_irp.loc[idx[:,'outflow',:,'Steel'],:].sum(level=[0,3])

    return waste

electricity_waste = assign_waste_flows(electricity_irp, collection, reuse, recycling, elec_cat, materials_select)
vehicle_waste     = assign_waste_flows(vehicle_irp,     collection, reuse, recycling, vehi_cat, materials_select_vehicles)
buildings_waste   = assign_waste_flows(buildings_irp,   collection, reuse, recycling, buil_cat, materials_select_buildings)

# Test if sum of outflow is still the sum of the accounted waste flows
test_elec = electricity_waste.sum(axis=0, level=[0,2]) - electricity_irp.loc[idx[:,'outflow',:,:],:].sum(axis=0, level=[0,3])
test_vehi = vehicle_waste.sum(axis=0, level=[0,2]) - vehicle_irp.loc[idx[:,'outflow',:,:],:].sum(axis=0, level=[0,3])
test_buil = buildings_waste.sum(axis=0, level=[0,2]) - buildings_irp.loc[idx[:,'outflow',:,:],:].sum(axis=0, level=[0,3])

mat_select = 'Steel'
test_steel_irp = electricity_irp.loc[idx[:,'outflow',:,mat_select],:].sum(axis=0, level=[0,3]) + vehicle_irp.loc[idx[:,'outflow',:,mat_select],:].sum(axis=0, level=[0,3]) + buildings_irp.loc[idx[:,'outflow',:,mat_select],:].sum(axis=0, level=[0,3])
test_steel_rec = electricity_waste.sum(axis=0, level=[0,2]).loc[idx[:,mat_select],:] + vehicle_waste.sum(axis=0, level=[0,2]).loc[idx[:,mat_select],:] + buildings_waste.sum(axis=0, level=[0,2]).loc[idx[:,mat_select],:]
test_steel     = test_steel_irp - test_steel_rec.values

if (-0.0001 < (test_elec.sum().sum() + test_vehi.sum().sum() + test_buil.sum().sum() + test_steel.sum().sum()) > 0.0001): 
    print("WARNING, waste flows are not equal to outlfow") 

#%% Effects on the primary raw material input (INFLOW disaggregated into reused, recycled or virgin raw material origins) 

# starting point is the total inflow per region & per sector of all materials (over time = columns), in kt/yr
electricity_inflow = electricity_irp.loc[idx[:,'inflow',:,:],:].sum(axis=0, level=[0,3]) 
buildings_inflow   = buildings_irp.loc[idx[:,'inflow',:,:],:].sum(axis=0, level=[0,3]) 
vehicle_inflow     = vehicle_irp.loc[idx[:,'inflow',:,:],:].sum(axis=0, level=[0,3]) 

# plot waste & inflow graphs
import matplotlib.pyplot as plt
import Recycling_scenarios.graphs_recycling

Recycling_scenarios.graphs_recycling.graph_sectors(electricity_waste, buildings_waste, vehicle_waste, scenario_variant)
Recycling_scenarios.graphs_recycling.graph_sectors_inflow(electricity_inflow.sum(level=1), buildings_inflow.sum(level=1), vehicle_inflow.sum(level=1), scenario_variant)

#%% 1) First assign sectoral re-use (re-deployed within the sector and within the region, to replace the same material)
# Mind that if reuse is larger than demand, the surplus is not reused and added to losses (*** could later be adjusted to account for regional long-term storage of surplus reusable materials)
col_dict = dict(zip(electricity_inflow.columns, electricity_waste.columns))  # ensure same names for columns

# determine surplus (i.e. when sectoral reuse > total required sectoral inflow)
electricity_surplus = (- electricity_inflow.rename(columns=col_dict) + electricity_waste.loc[idx[:,'reusable',:],:].droplevel(1))
buildings_surplus = (- buildings_inflow.rename(columns=col_dict) + buildings_waste.loc[idx[:,'reusable',:],:].droplevel(1))
vehicle_surplus = (- vehicle_inflow.rename(columns=col_dict) + vehicle_waste.loc[idx[:,'reusable',:],:].droplevel(1))

electricity_surplus[electricity_surplus < 0] = 0
buildings_surplus[buildings_surplus < 0] = 0
vehicle_surplus[vehicle_surplus < 0] = 0

# determine the amount actually reused (i.e. amount reusable - surplus)
electricity_reuse = electricity_waste.loc[idx[:,'reusable',:],:].droplevel(1) - electricity_surplus
buildings_reuse   = buildings_waste.loc[idx[:,'reusable',:],:].droplevel(1) - buildings_surplus
vehicle_reuse     = vehicle_waste.loc[idx[:,'reusable',:],:].droplevel(1) - vehicle_surplus

# assign surplus as additional losses
electricity_waste.loc[idx[:,'surplus losses',:],:] = electricity_surplus.values
buildings_waste.loc[idx[:,'surplus losses',:],:]   = buildings_surplus.values
vehicle_waste.loc[idx[:,'surplus losses',:],:]     = vehicle_surplus.values

# determine the remaining sectoral demand
electricity_remaining = electricity_inflow - electricity_reuse.values
buildings_remaining   = buildings_inflow - buildings_reuse.values
vehicle_remaining     = vehicle_inflow - vehicle_reuse.values

# *** Possible intermediate step here is a minimum virgin raw-material input per material (to ensure recycled material quality)


#%% 2) Then determine availability of recycled material at the regional market

# insert missing index items as series of 0
index = pd.MultiIndex.from_product([buildings_waste.index.levels[0], buildings_waste.index.levels[2]], names=['regions', 'materials'])
regional_recycled = pd.DataFrame(0, index=index, columns=list(range(start_year,end_year+1)))  # regional available recycled materials (kt/yr)
regional_inflow   = pd.DataFrame(0, index=index, columns=list(range(start_year,end_year+1)))  # regional required material inflow, after reuse (kt/yr)
total_inflow      = pd.DataFrame(0, index=index, columns=list(range(start_year,end_year+1)))  # regional required material inflow, after reuse (kt/yr)

demand_cat = ['reused', 'regional recycling', 'global recycling', 'virgin input']
index = pd.MultiIndex.from_product([buildings_waste.index.levels[0], demand_cat, buildings_waste.index.levels[2]], names=['regions', 'demand', 'materials'])
regional_demand  = pd.DataFrame(0, index=index, columns=list(range(start_year,end_year+1)))

# waste categories (combining sectoral waste flows to a regional total)
index = pd.MultiIndex.from_product([buildings_waste.index.levels[0], wast_cat, buildings_waste.index.levels[2]], names=['regions', 'waste', 'materials'])
regional_waste   = pd.DataFrame(0, index=index, columns=list(range(start_year,end_year+1)))  # regional required material inflow, after reuse (kt/yr)

# not all materials are present in all applications, hence complicated loop needed. *** improve later
for material in materials_select_buildings:
    regional_recycled.loc[idx[:,material],:] += buildings_waste.loc[idx[:,'recyclable',material],:].droplevel(1)
    regional_inflow.loc[idx[:,material],:]   += buildings_remaining.loc[idx[:,material],:].values
    total_inflow.loc[idx[:,material],:]      += buildings_inflow.loc[idx[:,material],:].values
    regional_demand.loc[idx[:,'reused',material],:]   += buildings_reuse.loc[idx[:,material],:].values
    regional_waste.loc[idx[:,:,material],:]  += buildings_waste.loc[idx[:,:,material],:]
    try:
        regional_recycled.loc[idx[:,material],:]          += electricity_waste.loc[idx[:,'recyclable',material],:].droplevel(1)
        regional_inflow.loc[idx[:,material],:]            += electricity_remaining.loc[idx[:,material],:].values
        total_inflow.loc[idx[:,material],:]               += electricity_inflow.loc[idx[:,material],:].values
        regional_demand.loc[idx[:,'reused',material],:]   += electricity_reuse.loc[idx[:,material],:].values
        regional_waste.loc[idx[:,:,material],:]           += electricity_waste.loc[idx[:,:,material],:]
    except:
        pass
    try:
        regional_recycled.loc[idx[:,material],:]         += vehicle_waste.loc[idx[:,'recyclable',material],:].droplevel(1)
        regional_inflow.loc[idx[:,material],:]           += vehicle_remaining.loc[idx[:,material],:].values
        total_inflow.loc[idx[:,material],:]              += vehicle_inflow.loc[idx[:,material],:].values
        regional_demand.loc[idx[:,'reused',material],:]  += vehicle_reuse.loc[idx[:,material],:].values
        regional_waste.loc[idx[:,:,material],:]          += vehicle_waste.loc[idx[:,:,material],:]
    except:
        pass

'''
# Test if regional inflow is indeed equal to the sum of the remaining demand after reuse
material = 'Steel'
test1 = regional_inflow.loc[idx[:,material],:]
test2 = electricity_remaining.loc[idx[:,material],:] + buildings_remaining.loc[idx[:,material],:] + vehicle_remaining.loc[idx[:,material],:]
test = test1 - test2.values
test.sum().sum() # should be 0
'''

# determine surplus (i.e. when available recycled flows > remaining required inflow (after reuse))
regional_surplus = (- regional_inflow + regional_recycled)
regional_surplus[regional_surplus < 0] = 0

# assign amounts to regionally recycled
regional_demand.loc[idx[:,'regional recycling',:],:] = regional_recycled.values - regional_surplus.values

# determine remaining demand (after reuse & regional recycling)
regional_remaining = regional_inflow - regional_demand.loc[idx[:,'regional recycling',:],:].droplevel(1)

'''
test1 = regional_remaining
test2 = total_inflow - regional_demand.loc[idx[:,'regional recycling',:],:].droplevel(1) - regional_demand.loc[idx[:,'reused',:],:].droplevel(1)
test = test1 - test2
test.sum().sum()  # should be zero 
'''

# *** Possible intermediate step is to add transport & supply chain losses when shipping to/from global market

# 3) Then, if total regional demand is fulfilled, the extra recycled material flow (if available) ends up at the global market and is distributed to regions whose demand is not yet fulfilled
global_recycled_available = regional_surplus.sum(level=1)       # global recycled material availability (surplus from regional supply or recyled material) (kt/yr)
global_remaining_demand   = regional_remaining.sum(level=1)     # sum of remaining worldwide demand for material (inflow) after reuse & regional recycling (kt/yr)      

global_share_recyclable   = global_recycled_available / global_remaining_demand   # share of the remaining demand that will be recycled (anything above 1 is considered a surplus loss)
global_share_recycled     = global_share_recyclable[global_share_recyclable < 1].fillna(1)    # surplus as a fraction of the remaining demand, is above 0 only in those years that there is a global surplus  

# alt. global_surplus_2   = global_remaining_demand.mul(global_share_recyclable[global_share_recyclable > 1].fillna(1) - 1)
global_surplus            = global_recycled_available - global_remaining_demand  # Global surplus (kt)
global_surplus[global_surplus < 0] = 0

# assign amounts that are recycled via a global market (surplus from regional recycling, after reuse) & virgin raw material input as the remainder 
regional_demand.loc[idx[:,'global recycling',:],:] = regional_remaining.mul(global_share_recycled).values
regional_remaining_virgin                          = regional_remaining - regional_demand.loc[idx[:,'global recycling',:],:].droplevel(1)
regional_demand.loc[idx[:,'virgin input',:],:]     = regional_remaining_virgin.values

'''
# Test if total outflow - global recycling + regional recycling + reuse + losses EQUALS surplus losses (before assigning global losses due to surplus)
test_global_recycling   = regional_demand.loc[idx[:,'global recycling','Steel'],:].droplevel(1).values
test_regional_recycling = regional_demand.loc[idx[:,'regional recycling','Steel'],:].droplevel(1).values
test_reuse              = regional_demand.loc[idx[:,'reused','Steel'],:].droplevel(1).values
test_losses             = regional_waste.loc[idx[:,'losses','Steel'],:].droplevel(1).values
test_surplus_losses     = regional_waste.loc[idx[:,'surplus losses','Steel'],:].droplevel(1).values
test1 = test_steel_irp - test_global_recycling - test_regional_recycling - test_reuse - test_losses - test_surplus_losses
test2 = global_surplus.loc['Steel']
test = test1 - test2
test.sum().sum()        # should be 0
'''

# redistribute the global surplus losses to the regions, based on their share of the regional surplus in the global surplus
# So, the rationale is that regions with a regional surplus will only be able to 'sell' their surplus at the global market to the point where the global market is saturated. After that, it becomes a local waste surplus (added to 'surplus losses' category, which includes surplus losses from reuse as well) 
regional_surplus_share   = regional_surplus / regional_surplus.sum(level=1).replace({ 0 : np.nan })
regional_surplus_losses  = regional_surplus_share.mul(global_surplus).fillna(0)
regional_waste.loc[idx[:,'surplus losses',:],:] += regional_surplus_losses.values                     # mind that the Global surplus is added to the pre-existing reuse surplus here (hence +=)

# Final tests
# The disaggregation of demand into: reused, recyled and virgin inputs is now know, so time for a check to see if the original inflow matches the sum of the disaggregated flows
# Also, the sum of losses, reusable & recyclable flows should add up to the total outflow GLOBALLY (not regionally though, because of minor global recycling flows)
# Also, the sum of losses, surplus losses, reused & recycled flows should add up to the total outflow GLOBALLY (not regionally though, because of minor global recycling flows)
test_inflow  = total_inflow - regional_demand.sum(level=[0,2])
test_outflow1 = regional_waste.sum(level=[0,2]).loc[idx[:,'Steel'],:] - regional_waste.loc[idx[:,'surplus losses','Steel'],:].droplevel(1) - test_steel_irp.values
test_outflow2 = regional_waste.loc[idx[:,'surplus losses','Steel'],:].droplevel([1,2]) + regional_waste.loc[idx[:,'losses','Steel'],:].droplevel([1,2]) + regional_demand.loc[idx[:,'global recycling','Steel'],:].droplevel([1,2]) + regional_demand.loc[idx[:,'regional recycling','Steel'],:].droplevel([1,2]) + regional_demand.loc[idx[:,'reused','Steel'],:].droplevel([1,2]) - test_steel_irp.values
if (-0.0001 < (test_inflow.sum().sum() + test_outflow1.sum().sum() + test_outflow2.sum().sum()) > 0.0001): 
    print("WARNING, waste flows are not equal to outlfow") 

#%% Figures on Global waste & inflow dynamics

Recycling_scenarios.graphs_recycling.graph_inflow(regional_demand.sum(level=[1,2]), scenario_variant, demand_cat)
Recycling_scenarios.graphs_recycling.graph_waste(regional_waste.sum(level=[1,2]), regional_demand.sum(level=[1,2]), scenario_variant, demand_cat)

#%% VIRGIN input fraction


#%% Output files

# Generate CSVs for output (MATERIAL flows, disaggregated E-o-L & demand)
regional_waste.to_csv('Recycling_scenarios\\output\\' + scenario_variant + '\\regional_waste.csv')   # in kt
regional_demand.to_csv('Recycling_scenarios\\output\\' + scenario_variant + '\\regional_demand.csv') # in kt


