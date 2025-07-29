# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 08:59:30 2020
@author: Sebasiaan Deetman (deetman@cml.leidenuniv.nl)

This module is used to calculate the materials involved in the electricity generation capacity and electricity storage capacity
Input:  1) scenario files from the IMAGE Integrated Assessment Model
        2) data files on material intensities, costs, lifetimes and weights
Output: Total global material use (stocks, inflow & outflow) in the electricity sector for the period 2000-2050 based on:
        1) the second Shared socio-economic pathway (SSP2) Baseline
        2) the second Shared socio-economic pathway (SSP2) 2-degree Climate Policy scenario
		
4 senitivity settings are defined:
1) 'default'    is the default model setting, used for the outcomes as described in the main text
2) 'high_stor'  defines a pessimistic setting with regard to storage demand (high) and availability (low)
3) 'high_grid'  defines alternative assumptions with respect to the growth of the grid (not relevant here, see grid_materials.py)


Sebastiaans Electricity_sector.py should be splitted into preprocessing.py and electricity.py.
This is still work in progress. The preprocessing.py file is already available and some parts are already transferred.

"""
###########################################################################################################
#%% define imports, counters & settings
###########################################################################################################

import pandas as pd
import numpy as np
import xarray as xr
import os
import math
import scipy
import warnings
from pathlib import Path
import sys 


from imagematerials.electricity.preprocessing import (
    get_preprocessing_data_gen
    # get_preprocessing_data_grid,
    # get_preprocessing_data_stor
)
# from imagematerials.util import import_from_netcdf, export_to_netcdf
from imagematerials.model import GenericMainModel, GenericMaterials, GenericStocks, Maintenance, MaterialIntensities
from imagematerials.factory import ModelFactory, Sector
import prism


VARIANT = "VLHO"
SCEN = "SSP2"
scen_folder = SCEN + "_" + VARIANT
# path_base = Path().resolve() # TODO absolute path of file "preprocessing.py" ? current solution can differ depending on IDE used (?) 
path_current = Path().resolve()
path_base = path_current.parent.parent # base path of the project -> image-materials



####################################################################################################################
#%% Generation
####################################################################################################################

prep_data = get_preprocessing_data_gen(path_base, scen_folder)

# # Define the complete timeline, including historic tail
time_start = prep_data["stocks"].coords["Time"].min().values
time_end = 2060
complete_timeline = prism.Timeline(time_start, time_end, 1)
simulation_timeline = prism.Timeline(1970, time_end, 1)


sec_electr_gen = Sector("electr_gen", prep_data)



main_model_factory = ModelFactory(
    sec_electr_gen, complete_timeline
    ).add(GenericStocks
    ).add(MaterialIntensities
    ).finish()

main_model_factory.simulate(simulation_timeline)

list(main_model_factory.electr_gen)






















###########################################################################################################
   
# OLD CODE:

###########################################################################################################
#%% 0) Before we start the calculations we define the general functions used in multiple parts of the code
###########################################################################################################
# TODO: delete this section


# 1st stock modelling version: -----------------------------------------------------------------------------------------------

# using the DSM model
from dynamic_stock_model import DynamicStockModel as DSM

# first define a Function in which the stock-driven DSM is applied to return (the moving average of the) inflow & outflow for all regions
def inflow_outflow(stock, lifetime, material_intensity, key):

    initial_year = stock.first_valid_index()
    outflow_mat  = pd.DataFrame(index=pd.MultiIndex.from_product([range(startyear,outyear+1), material_intensity.columns]), columns=stock.columns)
    inflow_mat   = pd.DataFrame(index=pd.MultiIndex.from_product([range(startyear,outyear+1), material_intensity.columns]), columns=stock.columns)   
    stock_mat    = pd.DataFrame(index=pd.MultiIndex.from_product([range(startyear,outyear+1), material_intensity.columns]), columns=stock.columns)
    out_oc_mat   = pd.DataFrame(index=pd.MultiIndex.from_product([range(first_year_grid,outyear+1), material_intensity.columns]), columns=stock.columns)
    out_sc_mat   = pd.DataFrame(index=pd.MultiIndex.from_product([range(first_year_grid,outyear+1), material_intensity.columns]), columns=stock.columns)
    out_in_mat   = pd.DataFrame(index=pd.MultiIndex.from_product([range(first_year_grid,outyear+1), material_intensity.columns]), columns=stock.columns)

    # define mean & standard deviation
    mean_list = list(lifetime)
    stdev_list = [mean_list[i] * stdev_mult for i in range(0,len(stock))]  

    for region in list(stock.columns):
        # define and run the DSM                                                                                            # list with the fixed (=mean) lifetime of grid elements, given for every timestep (1926-2100), needed for the DSM as it allows to change lifetime for different cohort (even though we keep it constant)
        DSMforward = DSM(t = np.arange(0,len(stock[region]),1), s=np.array(stock[region]), lt = {'Type': 'FoldedNormal', 'Mean': np.array(mean_list), 'StdDev': np.array(stdev_list)})  # definition of the DSM based on a folded normal distribution
        out_sc, out_oc, out_i = DSMforward.compute_stock_driven_model(NegativeInflowCorrect = True)                                                                 # run the DSM, to give 3 outputs: stock_by_cohort, outflow_by_cohort & inflow_per_year

        #convert to pandas df before multiplication with material intensity
        index=list(range(first_year_grid, outyear+1))
        out_sc_pd = pd.DataFrame(out_sc, index=index,  columns=index)
        out_oc_pd = pd.DataFrame(out_oc, index=index,  columns=index)
        out_in_pd = pd.DataFrame(out_i,  index=index)

        # sum the outflow & stock by cohort (using cohort specific material intensities)
        for material in list(material_intensity.columns):    
           out_oc_mat.loc[idx[:,material],region] = out_oc_pd.mul(material_intensity.loc[:,material], axis=1).sum(axis=1).to_numpy()
           out_sc_mat.loc[idx[:,material],region] = out_sc_pd.mul(material_intensity.loc[:,material], axis=1).sum(axis=1).to_numpy() 
           out_in_mat.loc[idx[:,material],region] = out_in_pd.mul(material_intensity.loc[:,material], axis=0).to_numpy()                
    
           # apply moving average to inflow & outflow & return only 1971-2050 values
           outflow_mat.loc[idx[:,material],region] = pd.Series(out_oc_mat.loc[idx[2:,material],region].astype('float64').values, index=list(range(initial_year, outyear + 1))).rolling(window=5).mean().loc[list(range(1971,outyear + 1))].to_numpy()    # Apply moving average                                                                                                      # sum the outflow by cohort to get the total outflow per year
           inflow_mat.loc[idx[:,material],region]  = pd.Series(out_in_mat.loc[idx[2:,material],region].astype('float64').values, index=list(range(initial_year, outyear + 1))).rolling(window=5).mean().loc[list(range(1971,outyear + 1))].to_numpy()
           stock_mat.loc[idx[:,material],region]   = out_sc_mat.loc[idx[:,material],region].loc[list(range(1971,outyear + 1))].to_numpy()                                                                                                        # sum the stock by cohort to get the total stock per year
        
    return pd.concat([inflow_mat.stack().unstack(level=1)], keys=[key], axis=1), pd.concat([outflow_mat.stack().unstack(level=1)], keys=[key], axis=1), pd.concat([stock_mat.stack().unstack(level=1)], keys=[key], axis=1)


# 2nd stock modelling version: -----------------------------------------------------------------------------------------------

# self-implemented stock model, which include market share calculations

# Here, we define the market share of the stock based on a pre-calculation with several steps: 
# 1) use Global total stock development, the market shares of the inflow and technology specific lifetimes to derive 
# the shares in the stock, assuming that pre-1990 100% of the stock was determined by Lead-acid batteries. 
# 2) Then, apply the global stock-related market shares to disaggregate the stock to technologies in all regions 
# (assuming that battery markets are global markets)
# As the development of the total stock of dedicated electricity storage is known, but we don't know the inflow, 
# and only the market share related to the inflow we need to calculate the inflow first. 

# first we define the survival of the 1990 stock (assumed 100% Lead-acid, for each cohort in 1990)
pre_time = 20                           # the years required for the pre-calculation of the Lead-acid stock
cohorts = outyear-switchtime            # nr. of cohorts in the stock calculations (after switchtime)
timeframe = np.arange(0,cohorts+1)      # timeframe for the pre-calcluation
pre_year_list = list(range(switchtime-pre_time,switchtime+1))   # list of years to pre-calculate the Lead-acid stock for

def stock_share_calc(stock, market_share, init_tech, techlist):
    
    #define new dataframes
    stock_cohorts = pd.DataFrame(index=pd.MultiIndex.from_product([stock.columns,range(switchtime,outyear+1)]), columns=pd.MultiIndex.from_product([techlist, range(switchtime-pre_time,endyear+1)])) # year, by year, by tech
    inflow_by_tech = pd.DataFrame(index=pd.MultiIndex.from_product([stock.columns,range(switchtime,outyear+1)]), columns=techlist)
    outflow_cohorts = pd.DataFrame(index=pd.MultiIndex.from_product([stock.columns,range(switchtime,outyear+1)]), columns=pd.MultiIndex.from_product([techlist, range(switchtime-pre_time,endyear+1)])) # year by year by tech
    
    #specify lifetime & other settings
    mean = storage_lifetime_interpol.loc[switchtime,init_tech] # select the mean lifetime for Lead-acid batteries in 1990
    stdev = mean * stdev_mult               # we use thesame standard-deviation as for generation technologies, given that these apply to 'energy systems' more generally
    survival_init = scipy.stats.foldnorm.sf(timeframe, mean/stdev, 0, scale=stdev)
    techlist_new = techlist
    techlist_new.remove(init_tech)          # techlist without Lead-acid (or other init_tech)
    
    # actual inflow & outflow calculations, this bit takes long!
    # loop over regions, technologies and years to calculate the inflow, stock & outflow of storage technologies, given their share of the inflow.
    for region in stock.columns:
        
        # pre-calculate the stock by cohort of the initial stock of Lead-acid
        multiplier_pre = stock.loc[switchtime,region]/survival_init.sum()                                    # the stock is subdivided by the previous cohorts according to the survival function (only allowed when assuming steady stock inflow) 
        
        #pre-calculate the stock as lists (for efficiency)
        initial_stock_years = [np.flip(survival_init[0:pre_time+1]) * multiplier_pre]
            
        for year in range(1, (outyear-switchtime)+1):                                                                   # then fill the columns with the remaining fractions
            initial_stock_years.append(initial_stock_years[0] * survival_init[year])
    
        stock_cohorts.loc[idx[region,:],idx[init_tech, list(range(switchtime-pre_time,switchtime+1))]] = initial_stock_years       # fill the stock dataframe according to the pre-calculated stock 
        outflow_cohorts.loc[idx[region,:],idx[init_tech, list(range(switchtime-pre_time,switchtime+1))]] = stock_cohorts.loc[idx[region,:],idx[init_tech, list(range(switchtime-pre_time,switchtime+1))]].shift(1, axis=0) - stock_cohorts.loc[idx[region,:],idx[init_tech, list(range(switchtime-pre_time,switchtime+1))]]
    
        # set the other stock cohorts to zero
        stock_cohorts.loc[idx[region,:],idx[techlist_new, pre_year_list]] = 0
        outflow_cohorts.loc[idx[region,:],idx[techlist_new, pre_year_list]] = 0
        inflow_by_tech.loc[idx[region, switchtime], techlist_new] = 0                                                   # inflow of other technologies in 1990 = 0
        
        # except for outflow and inflow in 1990 (switchtime), which can be pre-calculated for Deep-cycle Lead Acid (@ steady state inflow, inflow = outflow = stock/lifetime)
        outflow_cohorts.loc[idx[region, switchtime], idx[init_tech,:]] = outflow_cohorts.loc[idx[region, switchtime+1], idx[init_tech,:]]     # given the assumption of steady state inflow (pre switchtime), we can determine that the outflow is the same in switchyear as in switchyear+1                                        
        inflow_by_tech.loc[idx[region, switchtime], init_tech] =   stock.loc[switchtime,region]/mean                                          # given the assumption of steady state inflow (pre switchtime), we can determine the inflow to be the same as the outflow at a value of stock/avg. lifetime                                                          
        
        # From switchtime onwards, define a stock-driven model with a known market share (by tech) of the new inflow 
        for year in range(switchtime+1,outyear+1):
            
            # calculate the remaining stock as the sum of all cohorts in a year, for each technology
            remaining_stock = 0 # reset remaining stock
            for tech in inflow_by_tech.columns:
                remaining_stock += stock_cohorts.loc[idx[region,year],idx[tech,:]].sum()
               
            # total inflow required (= required stock - remaining stock);    
            inflow = max(0, stock.loc[year, region] - remaining_stock)   # max 0 avoids negative inflow, but allows for idle stock surplus in case the size of the required stock is declining more rapidly than it's natural decay
            
            stock_cohorts_list = []
                   
            # enter the new inflow & apply the survival rate, which is different for each technology, so calculate the surviving fraction in stock for each technology  
            for tech in inflow_by_tech.columns:
                # apply the known market share to the inflow
                inflow_by_tech.loc[idx[region,year],tech] = inflow * market_share.loc[year,tech]
                # first calculate the  (based on lifetimes specific to the year of inflow)
                survival = scipy.stats.foldnorm.sf(np.arange(0,(outyear+1)-year), storage_lifetime_interpol.loc[year,tech]/(storage_lifetime_interpol.loc[year,tech]*0.2), 0, scale=storage_lifetime_interpol.loc[year,tech]*0.2)           
                # then apply the survival to the inflow in current cohort, both the inflow & the survival are entered into the stock_cohort dataframe in 1 step
                stock_cohorts_list.append(inflow_by_tech.loc[idx[region,year],tech]  *  survival)
                
            stock_cohorts.loc[idx[region,list(range(year,outyear+1))],idx[:,year]] = list(map(list, zip(*stock_cohorts_list)))        
        
    # separate the outflow (by cohort) calculation (separate shift calculation for each region & tech is MUCH more efficient than including it in additional loop over years)
    # calculate the outflow by cohort based on the stock by cohort that was just calculated
    for region in stock.columns:
        for tech in inflow_by_tech.columns:
            outflow_cohorts.loc[idx[region,:],idx[tech,:]] = stock_cohorts.loc[idx[region,:],idx[tech,:]].shift(1,axis=0) - stock_cohorts.loc[idx[region,:],idx[tech,:]]

    return inflow_by_tech, stock_cohorts, outflow_cohorts


###########################################################################################################
###########################################################################################################
#%% 1) Generation 
###########################################################################################################
###########################################################################################################



###########################################################################################################
#%%% 1.1) Stock Modelling
###########################################################################################################

#start with materials in generation capacity (this is the easiest, as the stock AND the new capacity is pre-calculated in TIMER, 
# based on fixed lifetime assumptions, we only add the outflow, based on a fixed lifetime DSM and the same lifetimes as in TIMER)


# Apply the DSM to find inflow & outflow of Generation capacity

# then apply the Dynamic Stock Model to find inflow & outflow (5yr moving average)
index = pd.MultiIndex.from_product([list(range(startyear,outyear+1)), gcap.index.levels[1]])
gcap_inflow  = pd.DataFrame(index=index, columns=pd.MultiIndex.from_product([gcap.columns, gcap_material_list], names=['technologies','materials']))
gcap_outflow = pd.DataFrame(index=index, columns=pd.MultiIndex.from_product([gcap.columns, gcap_material_list], names=['technologies','materials']))
gcap_stock   = pd.DataFrame(index=index, columns=pd.MultiIndex.from_product([gcap.columns, gcap_material_list], names=['technologies','materials']))

# Materials in Gcap (in: Gcap: MW, lifetime: yrs, materials intensity: gram/MW)
for tech in gcap_tech_list: 
    gcap_inflow.loc[idx[:,:],idx[tech,:]], gcap_outflow.loc[idx[:,:],idx[tech,:]], gcap_stock.loc[idx[:,:],idx[tech,:]] = inflow_outflow(gcap_new.loc[idx[:,:],tech].unstack(level=1), gcap_lifetime.loc[:,tech], gcap_materials_interpol.loc[idx[:,:],tech].unstack(), tech)


###########################################################################################################
#%%% 1.2) Save Output
###########################################################################################################

#prepare variables on materials in generation capacity (in gram) for output in csv
gcap_stock = gcap_stock.stack().stack().unstack(level=0)               # use years as column names
gcap_stock = pd.concat([gcap_stock], keys=['stock'], names=['flow'])

gcap_inflow = gcap_inflow.stack().stack().unstack(level=0)   # use years as column names
gcap_inflow = pd.concat([gcap_inflow], keys=['inflow'], names=['flow'])

gcap_outflow = gcap_outflow.stack().stack().unstack(level=0)           # use years as column names
gcap_outflow = pd.concat([gcap_outflow], keys=['outflow'], names=['flow'])

gcap_materials_all = pd.concat([gcap_stock, gcap_inflow, gcap_outflow])
gcap_materials_all = pd.concat([gcap_materials_all], keys=['electricity'], names=['sector'])
gcap_materials_all = pd.concat([gcap_materials_all], keys=['generation'], names=['category'])
gcap_materials_all = gcap_materials_all.reorder_levels([3, 2, 1, 0, 5, 4]) / 1_000_000_000   # gram to kt

gcap_materials_all.to_csv(path_elma / 'output' / scen_folder / sa_settings / 'gcap_materials_output_kt.csv') # in kt


# Average generation material intensity calculations (originally weight is in grams)

# total_global_gcap = gcap.sum(axis=0, level=0).loc[:outyear].sum(axis=1) # level argument in sum() has been deprecated in pandas 2.0 
# First reset index to work more easily
gcap_reset = gcap.reset_index()
# Group by years, sum across regions
gcap_grouped = gcap_reset.groupby("years").sum(numeric_only=True)
# Select only years up to 'outyear' (assumes outyear is an int)
gcap_filtered = gcap_grouped.loc[:outyear]
# Final total across all columns (technologies)
total_global_gcap = gcap_filtered.sum(axis=1) # Gcap in MW
total_global_wght = gcap_stock.groupby(level=[2]).sum() / 1000000      # Weight in tons 
intensity_gcap = total_global_wght.div(total_global_gcap, axis=1)
intensity_gcap.to_csv(path_elma / 'output' / scen_folder / sa_settings / 'material_intensity_gcap_ton_per_MW.csv') # ton/MW





###########################################################################################################
###########################################################################################################
#%% 2) STORAGE
###########################################################################################################
###########################################################################################################


#----------------------------------------------------------------------------------------------------------
###########################################################################################################
#%% 2.1) Vehicles
###########################################################################################################
#----------------------------------------------------------------------------------------------------------


###########################################################################################################
#%%% 2.1.1) Stock Modelling
###########################################################################################################


# then we use that market share in combination with the stock developments to derive the stock share 
# Here we use the vehcile stock (number of cars) as a proxy for the development of the battery stock (given that we're calculating the actual battery stock still, and just need to account for the dynamics of purchases to derive te stock share here) 
EV_inflow_by_tech, EV_stock_cohorts, EV_outflow_cohorts = stock_share_calc(vehicles_EV, market_share_EVs, 'NiMH', ['NiMH', 'LMO', 'NMC', 'NCA', 'LFP', 'Lithium Sulfur', 'Lithium Ceramic ', 'Lithium-air'])


###########################################################################################################
#%%% 2.1.2) Save Output
###########################################################################################################


# EV_stock =  EV_stock_cohorts.loc[idx[:,:],idx[:,:]].sum(axis=1, level=0) #original
# EV_storage_stock_abs  = EV_stock.sum(axis=0, level=1)                           # sum over all regions to get the global share of the stock
# EV_storage_inflow_abs = EV_inflow_by_tech.sum(axis=0, level=1)                  # sum over all regions to get the global share of the inflow
# EV_storage_stock_share  = pd.DataFrame(index=EV_storage_stock_abs.index,  columns=EV_storage_stock_abs.columns)
# EV_storage_inflow_share = pd.DataFrame(index=EV_storage_inflow_abs.index, columns=EV_storage_inflow_abs.columns)

EV_stock =  EV_stock_cohorts.T.groupby(level=0).sum().T # sum(level) is and groupby(axis) will be deprecated -> transose first with .T (instead of specifying axis), then groupby level, then sum. To get intial shape back, transpose again with .T
EV_storage_stock_abs  = EV_stock.groupby(level=1).sum()                           # sum over all regions to get the global share of the stock
EV_storage_inflow_abs = EV_inflow_by_tech.groupby(level=1).sum()                   # sum over all regions to get the global share of the inflow

# Calc. global share of different battery technologies (stock & inflow)
EV_storage_stock_share  = pd.DataFrame(index=EV_storage_stock_abs.index,  columns=EV_storage_stock_abs.columns)
EV_storage_inflow_share = pd.DataFrame(index=EV_storage_inflow_abs.index, columns=EV_storage_inflow_abs.columns)
for tech in EV_storage_stock_abs.columns:
    EV_storage_stock_share.loc[:,tech]  = EV_storage_stock_abs.loc[:,tech].div(EV_storage_stock_abs.sum(axis=1))
    EV_storage_inflow_share.loc[:,tech] = EV_storage_inflow_abs.loc[:,tech].div(EV_storage_inflow_abs.sum(axis=1))

EV_storage_stock_share.to_csv(path_elma / 'output' / scen_folder / sa_settings / 'battery_share_stock.csv') # Average global car battery share (in stock) is exported to be used in paper on vehicles
EV_storage_inflow_share.to_csv(path_elma / 'output' / scen_folder / sa_settings / 'battery_share_inflow.csv') # Average global car battery share (in inflow) is exported to be used in paper on vehicles
  
#The global share of the battery technologies in stock is then used to derive the (weihgted) average density (kg/kWh)
weighted_average_density_stock  = EV_storage_stock_share.mul(storage_density_interpol[EV_battery_list]).sum(axis=1)
weighted_average_density_inflow = EV_storage_inflow_share.mul(storage_density_interpol[EV_battery_list]).sum(axis=1)

weighted_average_density_stock.loc[:outyear].to_csv(path_elma / 'output' / scen_folder / sa_settings / 'ev_battery_density_stock.csv')        # Average car battery density (in stock) is exported to be used in paper on vehicles
weighted_average_density_inflow.loc[:outyear].to_csv(path_elma / 'output' / scen_folder / sa_settings / 'ev_battery_density_inflow.csv')      # Average car battery density (in inflow) is exported to be used in paper on vehicles

# assumed fixed energy densities before 1990 (=NiMH)
add = pd.Series(weighted_average_density_stock[weighted_average_density_stock.first_valid_index()], index=list(range(startyear,1990)))
weighted_average_density = pd.concat([weighted_average_density_stock, add]).sort_index(axis=0)

# With a pre-determined battery capacity in 2018, we assume an increasing capacity (as an effect of an increased density) based on a fixed weight assumption
BEV_dynamic_capacity = (weighted_average_density[2018] * BEV_capacity) / weighted_average_density
PHEV_dynamic_capacity = (weighted_average_density[2018] * PHEV_capacity) / weighted_average_density

# Define the V2G settings over time (assuming slow adoption to the maximum usable capacity)
year_v2g_start = 2025-1 # -1 leads to usable capacity in 2025
year_full_capacity = 2040
v2g_usable = pd.Series(index=list(range(startyear,outyear+1)), name='years')    #fraction of the cars ready and willing to use v2g
for year in list(range(startyear,outyear+1)):
    if (year <= year_v2g_start):
        v2g_usable[year] = 0
    elif (year >= year_full_capacity):
        v2g_usable[year] = 1
v2g_usable = v2g_usable.interpolate()

#total capacity per car connected to v2g (in kWh)
max_capacity_BEV = v2g_usable * capacity_usable_BEV  
max_capacity_PHEV = v2g_usable * capacity_usable_PHEV   

#usable capacity per car connected to v2g (in kWh)
usable_capacity_BEV = max_capacity_BEV.mul(BEV_dynamic_capacity[:outyear])     
usable_capacity_PHEV = max_capacity_PHEV.mul(PHEV_dynamic_capacity[:outyear])  

# Vehicle storage in MWh
storage_BEV = storage_PHEV = pd.DataFrame().reindex_like(vehicles_BEV)
for region in region_list:
    storage_PHEV[region] = vehicles_PHEV[region].mul(usable_capacity_PHEV) /1000
    storage_BEV[region] = vehicles_BEV[region].mul(usable_capacity_BEV) /1000

storage_vehicles = storage_BEV + storage_PHEV




#----------------------------------------------------------------------------------------------------------
###########################################################################################################
#%% 2.2) Hydro Power
###########################################################################################################
#----------------------------------------------------------------------------------------------------------



###########################################################################################################
#%%% 2.2.1) Stock Modelling
###########################################################################################################


phs_storage_inflow, phs_storage_outflow, phs_storage_stock  = inflow_outflow(phs_storage_stock_tail, storage_lifetime_PHS, stor_materials_interpol.loc[idx[:,:],'PHS'].unstack() * PHS_kg_perkWh * 1000, 'PHS')    # PHS lifetime is fixed at 60 yrs anyway so, we simply select 1 value

# then, calculate the market share of technologies in the stock (for region & by year), also a global total market share of the stock is calculated to compare to the inflow market share 
inflow_by_tech, stock_cohorts, outflow_cohorts = stock_share_calc(oth_storage, storage_market_share, 'Deep-cycle Lead-Acid', list(storage_lifetime_interpol.columns)) # run the function that calculates stock shares from total stock & inflow shares



###########################################################################################################
#%%% 2.2.2) Further Processing
###########################################################################################################

# apply the dedicated electricity storage market shares to the demand for dedicated (other) storage to find storage by type and by region
stock = stock_cohorts.loc[idx[:,:],idx[:,:]].sum(axis=1, level=0)
storage_stock_abs = stock.sum(axis=0, level=1)                              # sum over all regions to get the global share of the stock
stock_total = stock.sum(axis=1).unstack(level=0)                            # total stock by region
storage_stock_share = pd.DataFrame(index=storage_stock_abs.index, columns=storage_stock_abs.columns)

for tech in storage_stock_abs.columns:
    storage_stock_share.loc[:,tech] = storage_stock_abs.loc[:,tech].div(storage_stock_abs.sum(axis=1))

# then, the sum of the global inflow & outflow is calculated for comparison in the figures below
outflow = outflow_cohorts.loc[idx[:,:],idx[:,:]].sum(axis=1, level=0)
outflow_total = outflow.sum(axis=1).unstack(level=0)                        # total by regions

inflow_total = inflow_by_tech.sum(axis=1).unstack(level=0)



###########################################################################################################
#%%% 2.2.3) Graphs output
###########################################################################################################


import graphs_elec

graphs_elec.graph_global_ev_capacity(BEV_dynamic_capacity, PHEV_dynamic_capacity,  max_capacity_BEV, max_capacity_PHEV, scen_folder, sa_settings) # make a graph on the average capacity of EVs and their availability to V2G
graphs_elec.graph_regional_storage(region_list, storage_PHEV, storage_BEV, storage, scen_folder, scenario, sa_settings)                           # make a graph on regional storage demand vs V2G supply
graphs_elec.graph_storage_demand_vs_availability(region_list, phs_storage_theoretical, storage_vehicles, storage, scen_folder, sa_settings)       # make a graph on storage demand vs the availability in 3 different storage types (phs, evs and dedicated) 
graphs_elec.graph_regional_dedicated_storage(oth_storage, scen_folder, scenario, sa_settings)                                                     # make a graph on regional dedicated storage demand
graphs_elec.graph_market_share(storage_market_share, scen_folder, sa_settings)                                                                    # make a graph on the market shares of storage technologies
graphs_elec.graph_market_share_pie(storage_market_share, scen_folder, sa_settings)                                                                # make a pie chart on the market share of dedicated storage technologies




###########################################################################################################
###########################################################################################################
#%% 3) Material calculations
###########################################################################################################
###########################################################################################################


# define empty dataframe for the weight of storage technolgies (in kg)
i_storage_weight = pd.DataFrame().reindex_like(stock)            
o_storage_weight = pd.DataFrame().reindex_like(stock)           
s_storage_weight = pd.DataFrame().reindex_like(stock)      

index  = i_storage_weight.index
column = pd.MultiIndex.from_product([i_storage_weight.columns, storage_materials.index], names=['technologies', 'materials'])
i_storage_materials = pd.DataFrame(index=index.set_names(['regions', 'years']), columns=column)            
o_storage_materials= []
s_storage_materials= []

# calculate the total weight of the inflow of storage technologies in kg based on the (CHANGING!) energy density (kg/kWh) and storage demand (MWh)
for region in oth_storage.columns:
   for material in storage_materials.index:
      inflow_multiplier = inflow_by_tech.loc[idx[region,:],:]
      inflow_multiplier.index = inflow_multiplier.index.droplevel(0)
      i_storage_weight.loc[idx[region,:],:]    = storage_density_interpol.mul(inflow_multiplier).values * 1000       # * 1000 because density is in kg/kWh and storage is in MWh
      for tech in storage_density_interpol.columns:
         i_storage_materials.loc[idx[region,:],idx[tech,material]] = i_storage_weight.loc[idx[region,:],tech].mul(stor_materials_interpol.loc[idx[list(range(1990,endyear+1)),material],tech].to_numpy()) / 1000000 # kg to kt

# calculate the weight of the stock and the outflow (in kg)
# this one's tricky, the outflow and stock are calculated by year AND cohort, because the total weight is represented by the sum of the weight of each cohort. As the density (kg/kWh) is different for every cohort (battery weight changes over time, older batteries are heavier), the total outflow FOR EACH YEAR is the sumproduct of the density and the outflow by cohort.
# 'intermediate' variables are use to speed up computing (by avoiding accessing dataframes unnecesarily), still, this loop takes about 25 minutes
for region in oth_storage.columns: #['India']:
    for tech in storage_density_interpol.columns: #['Flywheel']: 
        for year in storage_density_interpol.index:
            
            # the first series of the sumproduct (the storage capacity in MWh) has a multi-level index, so the second level needs to be removed before it can be multiplied
            series_mwh_o = outflow_cohorts.loc[idx[region,year],idx[tech,:]]
            series_mwh_o.index = series_mwh_o.index.droplevel(0)                
            o_storage_weight_intermediate               = series_mwh_o.loc[list(range(1990, outyear + 1))].mul(storage_density_interpol[tech] * 1000)       # * 1000 because density is in kg/kWh and storage is in MWh
            o_storage_weight.loc[idx[region,year],tech] = o_storage_weight_intermediate.sum()
            o_storage_materials_intermediate = stor_materials_interpol.loc[idx[list(range(1990, outyear + 1)),:],tech].unstack().mul(o_storage_weight_intermediate, axis=0).sum(axis=0)
            
            # same for stock
            series_mwh_s = stock_cohorts.loc[idx[region,year],idx[tech,:]]
            series_mwh_s.index = series_mwh_s.index.droplevel(0)                
            s_storage_weight_intermediate               = series_mwh_s.mul(storage_density_interpol[tech] * 1000)       # * 1000 because density is in kg/kWh and storage is in MWh
            s_storage_weight.loc[idx[region,year],tech] = s_storage_weight_intermediate.sum()  
            s_storage_materials_intermediate = stor_materials_interpol.loc[idx[list(range(1990, outyear + 1)),:],tech].unstack().mul(s_storage_weight_intermediate, axis=0).sum(axis=0)
            
            o_storage_materials.append([region, tech, year, o_storage_materials_intermediate[0], o_storage_materials_intermediate[1], o_storage_materials_intermediate[2], o_storage_materials_intermediate[3], o_storage_materials_intermediate[4], o_storage_materials_intermediate[5], o_storage_materials_intermediate[6], o_storage_materials_intermediate[7], o_storage_materials_intermediate[8]])
            s_storage_materials.append([region, tech, year, s_storage_materials_intermediate[0], s_storage_materials_intermediate[1], s_storage_materials_intermediate[2], s_storage_materials_intermediate[3], s_storage_materials_intermediate[4], s_storage_materials_intermediate[5], s_storage_materials_intermediate[6], s_storage_materials_intermediate[7], s_storage_materials_intermediate[8]]) 
            
order = list(s_storage_materials_intermediate.index)

# restore storage materials to pandas dataframe (unit: kg)
o_storage_materials = pd.DataFrame(o_storage_materials).set_index([0,1,2]).rename(columns=dict(zip(list(range(3,12)), order))).stack().unstack(level=[1,3]).reindex_like(i_storage_materials) / 1000000
s_storage_materials = pd.DataFrame(s_storage_materials).set_index([0,1,2]).rename(columns=dict(zip(list(range(3,12)), order))).stack().unstack(level=[1,3]).reindex_like(i_storage_materials) / 1000000

# Insert the materials (in kg) of the PHS storage component, which was calculated before 
for material in storage_materials.index:
   i_storage_materials.loc[idx[:,:], idx['PHS', material]] = phs_storage_inflow.reorder_levels([1,0]).loc[:,idx['PHS',material]]  / 1000000
   s_storage_materials.loc[idx[:,:], idx['PHS', material]] = phs_storage_stock.reorder_levels([1,0]).loc[:,idx['PHS',material]]   / 1000000
   o_storage_materials.loc[idx[:,:], idx['PHS', material]] = phs_storage_outflow.reorder_levels([1,0]).loc[:,idx['PHS',material]] / 1000000 

#----------------------------------------------------------------------------------------------------------------------
#%% avoided material calculations due to V2G
#----------------------------------------------------------------------------------------------------------------------
v2g_vs_dedicated = (evs_storage.loc[[2045,2046,2047,2048,2049,2050],:].sum(axis=1).sum()/6) / (oth_storage.loc[[2045,2046,2047,2048,2049,2050],:].sum(axis=1).sum()/6) # evs_storage & oth_storage are in MWh
v2g_vs_dedicated.tofile('output\\' + scen_folder + '\\' + sa_settings + '\\v2g_vs_dedicated.csv', sep=',')  # how much bigger is v2g than dedicated? 
average_storage_intensity = storage_stock_share.mul(storage_density_interpol).sum(axis=1)  # kg /kWh
material_avoided = evs_storage.loc[[2045,2046,2047,2048,2049,2050],:].sum(axis=1).sum() * 1000 * (average_storage_intensity[[2045,2046,2047,2048,2049,2050]].sum()/6) / 1000000 # in kt
material_avoided.tofile('output\\' + scen_folder + '\\' + sa_settings + '\\v2g_material_avoided_kt.csv', sep=',')  # how much weight of dedicated storage is avoided when V2G is assumed?, in kt 

#----------------------------------------------------------------------------------------------------------------------
#%% Export data to excel (in kt)
#----------------------------------------------------------------------------------------------------------------------

# full storage dataset (by technology)
i_materials_by_tech_out = i_storage_materials.stack(level=0).stack().unstack(level=1).reset_index(inplace=False)  # kg to kt
i_materials_by_tech_out.insert(1, 'flow', 'inflow')     # add a 'flow' column, 
s_materials_by_tech_out = s_storage_materials.stack(level=0).stack().unstack(level=1).reset_index(inplace=False)  # kg to kt
s_materials_by_tech_out.insert(1, 'flow', 'stock')      # add a 'flow' column
o_materials_by_tech_out = o_storage_materials.stack(level=0).stack().unstack(level=1).reset_index(inplace=False)  # kg to kt
o_materials_by_tech_out.insert(1, 'flow', 'outflow')    # add a 'flow' column

#combine stock, inflow and outflow for output & export to excel (in kt)
output_by_tech = pd.concat([s_materials_by_tech_out, i_materials_by_tech_out, o_materials_by_tech_out]) 
output_by_tech.insert(2, 'category', 'storage')      # add a 'category' column
output_by_tech.insert(2, 'sector', 'electricity')    # add a 'sector' column

output_by_tech.to_csv('output\\' + scen_folder + '\\' + sa_settings + '\\stor_materials_output_kt.csv', index=False) # in kt

# total materials in storage (summed over all technologies) 
output_by_tech.set_index(['regions', 'flow', 'sector', 'category', 'technologies', 'materials'], inplace=True)
output_sum = output_by_tech.unstack(level=4).sum(axis=1, level=0)
output_sum.to_csv('output\\' + scen_folder + '\\' + sa_settings + '\\export_storage_sum_kt.csv') 
