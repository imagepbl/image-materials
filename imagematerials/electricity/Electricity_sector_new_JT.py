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

"""
###########################################################################################################
#%% define imports, counters & settings
###########################################################################################################

import pandas as pd
import numpy as np
import os
import math
import scipy
import warnings
from pathlib import Path
import sys 
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from imagematerials.read_mym import read_mym_df

scenario = "SSP2"
variant  = "2D_RE"
sa_settings = "default"         # settings for the sensitivity analysis (default, high_stor, high_grid)
scen_folder = scenario + "_" + variant

# Define paths ----------------------------------------------------------------------
#YOUR_DIR = "C:\\Users\\Admin\\surfdrive\\Projects\\IRP\\GRO23\\Modelling\\2060\\ELMA"   # Change the running directory here
# os.chdir(YOUR_DIR)
# path_current = Path.cwd() # 
path_current = Path(__file__).resolve().parent # absolute path of file
path_base = path_current.parent.parent # base path of the project -> image-materials
# NEW
path_image_output = Path(path_base, "data", "raw", scenario, "EnergyServices")
path_external_data_standard = Path(path_base, "data", "raw", "electricity", "standard_data")
path_external_data_scenario = Path(path_base, "data", "raw", "electricity", scen_folder)
path_elma_out = Path(path_base, 'imagematerials', 'electricity', 'out_test', 'Old_ELMA_output')
# OLD
# path_elma = Path(path_base, "IMAGE-Mat_old_version", "IMAGE-Mat", "ELMA")
# path_scenario_output = Path(path_base, "IMAGE-Mat_old_version", "IMAGE-Mat", "ELMA", scenario, scen_folder)

assert path_image_output.is_dir()
assert path_external_data_standard.is_dir()
assert path_external_data_scenario.is_dir()



sys.path.append(str(path_current))
from dynamic_stock_model import DynamicStockModel as DSM






# Define constants ---------------------------------------------------------------
cohorts = 50
startyear = 1971
endyear = 2060
outyear = 2060                  # latest year of output data
first_year_grid = 1926          # UK Electricity supply act - https://www.bbc.com/news/uk-politics-11619751   
years = endyear - startyear  + 1
switchtime = 1990
vehicles = 25
regions = 26
epg_techs = 34          # number of electricity generation technologies

weibull_shape = 1.89
weibull_scale = 10.3
stdev_mult = 0.214      # multiplier that defines the standard deviation (standard deviation = mean * multiplier)
LOAD_FACTOR = 1.6       # reference loadfactor of cars in TIMER (the trp_trvl_Load.out file is relative to this BASE loadfcator (persons/car))

# from past.builtins import execfile
# execfile('read_mym.py')
idx = pd.IndexSlice             # needed for slicing multi-index

###########################################################################################################
#%% Read in files. FIRST: IMAGE/TIMER files
###########################################################################################################

# # The vehicle shares of trucks (heavy) of the SSP2
#     loadfactor_car_data: pd.DataFrame = read_mym_df(
#         image_folder.joinpath("trp_trvl_Load.out")). rename(
#         columns={
#             "DIM_1": "region"})



# read TIMER installed storage capacity (MWh, reservoir)
# storage = read_mym_df(path_scenario_output.joinpath("StorResTot.out"))    #storage capacity in MWh (reservoir, so energy capacity, not power capacity, the latter is used later on in the pumped hydro storage calculations)
storage = read_mym_df(path_image_output.joinpath("StorResTot.out")) # new
storage.drop(storage.iloc[:, -2:], inplace = True, axis = 1)    # drop global total column and empty (27) column

if sa_settings == 'high_stor':
   storage_multiplier = storage
   for year in range(2021,2051):
        storage_multiplier.loc[year] = storage.loc[year] * (1 + (1/30*(year-2020)))
   for year in range(2051,endyear+1):
        storage_multiplier.loc[year] = storage.loc[year] * 2

# kilometrage = pd.read_csv(path_scenario_output / 'kilometrage.csv', index_col='t')        #kilometrage in kms/yr
kilometrage = pd.read_csv(path_external_data_scenario / 'kilometrage.csv', index_col='t')
# kilometrage is defined untill 2008, fill 2008 values untill 2100 
kilometrage = kilometrage.reindex(list(range(startyear,endyear))).interpolate(limit_direction='both')
region_list = list(kilometrage.columns.values)                          # get a list with region names
    
# loadfactor_data = read_mym_df(path_scenario_output / 'trp_trvl_Load.out') 
loadfactor_data = read_mym_df(path_image_output / 'trp_trvl_Load.out') 
loadfactor = loadfactor_data[['time','DIM_1', 5]].pivot_table(index='time', columns='DIM_1') # loadfactor for cars (in persons per vehicle)
loadfactor = loadfactor.loc[list(range(startyear,endyear+1))] * LOAD_FACTOR  # car loadfactor is expressed compared to an average global loadfactor from the IMAGE-TIMER-TRAVEL model (original: Girod, 2012; further elaborations by Edelenbosch et al.) 
    
# passengerkms_data = read_mym_df(path_scenario_output / 'trp_trvl_pkm.out')                # passenger kilometers in Tera pkm
passengerkms_data = read_mym_df(path_image_output / 'trp_trvl_pkm.out')
indexNames = passengerkms_data[ passengerkms_data['DIM_1'] >= 27 ].index
passengerkms_data.drop(indexNames , inplace=True)
passengerkms = passengerkms_data[['time','DIM_1', 5]].pivot_table(index='time', columns='DIM_1').loc[list(range(startyear,endyear+1))]
  
BEV_collist = [22, 23, 24, 25] # battery electric vehicles
PHEV_collist= [21, 20, 19, 18, 17, 16] # plug-in hybrid electric vehicles
# vehicleshare_data = read_mym_df(path_scenario_output / 'trp_trvl_Vshare_car.out')
vehicleshare_data = read_mym_df(path_image_output / 'trp_trvl_Vshare_car.out')
vehicleshare_data['battery'] = vehicleshare_data[BEV_collist].sum(axis=1)
vehicleshare_data['PHEV'] = vehicleshare_data[PHEV_collist].sum(axis=1)

# get the regional sum of the BEV & PHEV fraction of the fleet, and replace the column names with the region names
PHEV_share = vehicleshare_data[['time','DIM_1', 'PHEV']].pivot_table(index='time',columns='DIM_1').loc[list(range(startyear,endyear+1))]
BEV_share = vehicleshare_data[['time','DIM_1', 'battery']].pivot_table(index='time',columns='DIM_1').loc[list(range(startyear,endyear+1))]

# before we run any calculations, replace the column names to region names
PHEV_share.columns = region_list
BEV_share.columns = region_list
passengerkms.columns = region_list
loadfactor.columns = region_list
storage.columns = region_list

# material compositions (storage & generation capacity)
# storage_materials = pd.read_csv(path_elma / 'data' / 'storage_materials_dynamic.csv',index_col=[0,1]).transpose()            # wt% of total battery weight for various materials, total battery weight is given by the density file above
storage_materials = pd.read_csv(path_external_data_standard / 'storage_materials_dynamic.csv',index_col=[0,1]).transpose()
# composition_generation = pd.read_csv(path_elma / 'data' / scen_folder / 'composition_generation.csv',index_col=[0,1]).transpose()  # in gram/MW
composition_generation = pd.read_csv(path_external_data_scenario / 'composition_generation.csv',index_col=[0,1]).transpose()

gcap_tech_list = list(composition_generation.loc[:,idx[2020,:]].droplevel(axis=1, level=0).columns)    #list of names of the generation technologies (workaround to retain original order)
gcap_material_list = list(composition_generation.index.values)  #list of materials the generation technologies

# Generation capacity (stock & inflow/new) in MW peak capacity, FILES from TIMER
# gcap_data = read_mym_df(path_scenario_output / 'Gcap.out')
gcap_data = read_mym_df(path_image_output / 'Gcap.out')
gcap_data = gcap_data.loc[~gcap_data['DIM_1'].isin([27,28])]    # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
gcap = pd.pivot_table(gcap_data[gcap_data['time'].isin(list(range(startyear,endyear+1)))], index=['time','DIM_1'], values=list(range(1,epg_techs+1)))  #gcap as multi-index (index = years & regions (26); columns = technologies (34));  the last column in gcap_data (= totals) is now removed

# renaming multi-index dataframe: generation capacity, based on the regions in grid_length_Hv & technologies as given
gcap.index = pd.MultiIndex.from_product([list(range(startyear,endyear+1)), region_list], names=['years', 'regions'])
gcap.columns = gcap_tech_list

# lifetimes of Gcap tech (original data according to van Vuuren 2006, PhD Thesis)
# gcap_lifetime = pd.read_csv(path_scenario_output / 'LTTechnical_dynamic.csv', index_col=['Year','DIM_1'])  
gcap_lifetime = pd.read_csv(path_external_data_scenario / 'LTTechnical_dynamic.csv', index_col=['Year','DIM_1'])        
gcap_lifetime.index = gcap_lifetime.index.set_levels(gcap_tech_list, level=1)
gcap_lifetime = gcap_lifetime.unstack().droplevel(axis=1, level=0)

# # read in the storage share in 2016 according to IEA (Technology perspectives 2017)
# storage_IEA = pd.read_csv(path_elma / 'data' / 'storage_IEA2016.csv', index_col=0)

# # read in the storage costs according to IRENA storage report & other sources in the SI
# storage_costs = pd.read_csv(path_elma / 'data' / 'storage_cost.csv', index_col=0).transpose()

# # read in the assumed malus & bonus of storage costs (malus for advanced technologies, still under development; bonus for batteries currently used in EVs, we assume that a large volume of used EV batteries will be available and used for dedicated electricity storage, thus lowering costs), only the bonus remains by 2030
# storage_malus = pd.read_csv(path_elma / 'data' / 'storage_malus.csv', index_col=0).transpose()

# #read in the assumptions on the long-term price decline after 2050. Prices are in $ct / kWh electricity cycled (the fraction of the annual growth rate (determined based on 2018-2030) that will be applied after 2030, ranging from 0.25 to 1 - 0.25 means the price decline is not expected to continue strongly, while 1 means that the same (2018-2030) annual price decline is also applied between 2030 and 2050)
# storage_ltdecline = pd.Series(pd.read_csv(path_elma / 'data' / 'storage_ltdecline.csv',index_col=0,  header=None).transpose().iloc[0])

# #read in the energy density assumptions (kg/kWh storage capacity - mass required to store one unit of energy — more mass per energy = worse performance)
# storage_density = pd.read_csv(path_elma / 'data' / 'storage_density_kg_per_kwh.csv',index_col=0).transpose()

# #read in the lifetime of storage technologies (in yrs). The lifetime is assumed to be 1.5* the number of cycles divided by the number of days in a year (assuming diurnal use, and 50% extra cycles before replacement, representing continued use below 80% remaining capacity) OR the maximum lifetime in years, which-ever comes first 
# storage_lifetime = pd.read_csv(path_elma / 'data' / 'storage_lifetime.csv',index_col=0).transpose()


# read in the storage share in 2016 according to IEA (Technology perspectives 2017)
storage_IEA = pd.read_csv(path_external_data_standard / 'storage_IEA2016.csv', index_col=0)

# read in the storage costs according to IRENA storage report & other sources in the SI
storage_costs = pd.read_csv(path_external_data_standard / 'storage_cost.csv', index_col=0).transpose()

# read in the assumed malus & bonus of storage costs (malus for advanced technologies, still under development; bonus for batteries currently used in EVs, we assume that a large volume of used EV batteries will be available and used for dedicated electricity storage, thus lowering costs), only the bonus remains by 2030
storage_malus = pd.read_csv(path_external_data_standard / 'storage_malus.csv', index_col=0).transpose()

#read in the assumptions on the long-term price decline after 2050. Prices are in $ct / kWh electricity cycled (the fraction of the annual growth rate (determined based on 2018-2030) that will be applied after 2030, ranging from 0.25 to 1 - 0.25 means the price decline is not expected to continue strongly, while 1 means that the same (2018-2030) annual price decline is also applied between 2030 and 2050)
storage_ltdecline = pd.Series(pd.read_csv(path_external_data_standard / 'storage_ltdecline.csv',index_col=0,  header=None).transpose().iloc[0])

#read in the energy density assumptions (kg/kWh storage capacity - mass required to store one unit of energy — more mass per energy = worse performance)
storage_density = pd.read_csv(path_external_data_standard / 'storage_density_kg_per_kwh.csv',index_col=0).transpose()

#read in the lifetime of storage technologies (in yrs). The lifetime is assumed to be 1.5* the number of cycles divided by the number of days in a year (assuming diurnal use, and 50% extra cycles before replacement, representing continued use below 80% remaining capacity) OR the maximum lifetime in years, which-ever comes first 
storage_lifetime = pd.read_csv(path_external_data_standard / 'storage_lifetime.csv',index_col=0).transpose()

###########################################################################################################
#%% Prepare model specific variables by interpolation
###########################################################################################################


# turn index to integer for sorting during the next step
storage_costs.index = storage_costs.index.astype('int64')
storage_malus.index = storage_malus.index.astype('int64')
storage_density.index = storage_density.index.astype('int64')
storage_lifetime.index = storage_lifetime.index.astype('int64')

# to interpolate between 2018 and 2030, first create empty rows (NaN values) 
storage_start = storage_costs.first_valid_index()
storage_end =   storage_costs.last_valid_index()
for i in range(storage_start+1,storage_end):
    storage_costs = pd.concat([storage_costs, pd.DataFrame(index=[i])]) #, ignore_index=True
    storage_malus = pd.concat([storage_malus, pd.DataFrame(index=[i])])         # mind: the malus needs to be defined for the same years as the cost indications
    storage_density = pd.concat([storage_density, pd.DataFrame(index=[i])])     # mind: the density needs to be defined for the same years as the cost indications
    storage_lifetime = pd.concat([storage_lifetime, pd.DataFrame(index=[i])])   # mind: the lifetime needs to be defined for the same years as the cost indications
    
# then, do the actual interpolation on the sorted dataframes                                                    
storage_costs_interpol = storage_costs.sort_index(axis=0).interpolate(axis=0)#.index.astype('int64')
storage_malus_interpol = storage_malus.sort_index(axis=0).interpolate(axis=0)
storage_density_interpol = storage_density.sort_index(axis=0).interpolate(axis=0)  # density calculation continue with the material calculations
storage_lifetime_interpol = storage_lifetime.sort_index(axis=0).interpolate(axis=0)  # lifetime calculation continue with the material calculations

# fix the energy density (kg/kwh) of storage technologies after 2030
for year in range(2030+1,outyear+1):
    # storage_density_interpol = storage_density_interpol.append(pd.Series(storage_density_interpol.loc[storage_density_interpol.last_valid_index()], name=year))
    row = storage_density_interpol.loc[[storage_density_interpol.last_valid_index()]]
    row.index = [year]
    storage_density_interpol = pd.concat([storage_density_interpol, row])
    

# assumed fixed energy densities before 2018
for year in reversed(range(switchtime,storage_start)):
    # storage_density_interpol = storage_density_interpol.append(pd.Series(storage_density_interpol.loc[storage_density_interpol.first_valid_index()], name=year)).sort_index(axis=0)
    row = storage_density_interpol.loc[[storage_density_interpol.first_valid_index()]]
    row.index = [year]
    storage_density_interpol = pd.concat([storage_density_interpol, row]).sort_index(axis=0)

# Interpolate material intensities (dynamic content for gcap & storage technologies between 1926 to 2100, based on data files)
index = pd.MultiIndex.from_product([list(range(first_year_grid, outyear+1)), list(storage_materials.index)])
stor_materials_interpol = pd.DataFrame(index=index, columns=storage_materials.columns.levels[1])
index = pd.MultiIndex.from_product([list(range(first_year_grid, outyear+1)), list(composition_generation.index)])
gcap_materials_interpol = pd.DataFrame(index=index, columns=composition_generation.columns.levels[1])

# material intensities for storage
for cat in list(storage_materials.columns.levels[1]):
   stor_materials_1st   = storage_materials.loc[:,idx[storage_materials.columns[0][0],cat]]
   stor_materials_interpol.loc[idx[first_year_grid ,:],cat] = stor_materials_1st.to_numpy()                # set the first year (1926) values to the first available values in the dataset (for the year 2000) 
   stor_materials_interpol.loc[idx[storage_materials.columns.levels[0].min(),:],cat] = storage_materials.loc[:, idx[storage_materials.columns.levels[0].min(),cat]].to_numpy()                # set the middle year (2000) values to the first available values in the dataset (for the year 2000) 
   stor_materials_interpol.loc[idx[storage_materials.columns.levels[0].max(),:],cat] = storage_materials.loc[:, idx[storage_materials.columns.levels[0].max(),cat]].to_numpy()                # set the last year (2100) values to the last available values in the dataset (for the year 2050) 
   stor_materials_interpol.loc[idx[:,:],cat] = stor_materials_interpol.loc[idx[:,:],cat].unstack().astype('float64').interpolate().stack()

# material intensities for gcap
for cat in list(composition_generation.columns.levels[1]):
   gcap_materials_1st   = composition_generation.loc[:,idx[composition_generation.columns[0][0],cat]]
   gcap_materials_interpol.loc[idx[first_year_grid ,:],cat] = gcap_materials_1st.to_numpy()                # set the first year (1926) values to the first available values in the dataset (for the year 2000) 
   gcap_materials_interpol.loc[idx[composition_generation.columns.levels[0].min(),:],cat] = composition_generation.loc[:, idx[composition_generation.columns.levels[0].min(),cat]].to_numpy()                # set the middle year (2000) values to the first available values in the dataset (for the year 2000) 
   gcap_materials_interpol.loc[idx[composition_generation.columns.levels[0].max(),:],cat] = composition_generation.loc[:, idx[composition_generation.columns.levels[0].max(),cat]].to_numpy()                # set the last year (2100) values to the last available values in the dataset (for the year 2050) 
   gcap_materials_interpol.loc[idx[:,:],cat] = gcap_materials_interpol.loc[idx[:,:],cat].unstack().astype('float64').interpolate().stack()

# interpolate Gcap (technical) lifetime data
gcap_lifetime = gcap_lifetime.reindex(list(range(first_year_grid,outyear+1)), axis=0).interpolate(limit_direction='both')
   

###########################################################################################################
#%% 0) Before we start the calculations we define the general functions used in multiple parts of the code
###########################################################################################################

storage_lifetime_interpol = storage_lifetime.sort_index(axis=0).interpolate(axis=0)  # lifetime calculation continue with the material calculations


# ----------------- |||| Loop to derive stock share from total stock and market (inflow) share \\\\\ ----------------------------------------------------------

# First the lifetime of storage technologies needs to be defined over time, before running the dynamic stock function
# before 2018
for year in reversed(range(startyear,storage_start)):
    # storage_lifetime_interpol = pd.concat([storage_lifetime_interpol, pd.Series(storage_lifetime_interpol.loc[storage_lifetime_interpol.first_valid_index()], name=year)])
    row = pd.DataFrame([storage_lifetime_interpol.loc[storage_lifetime_interpol.first_valid_index()]])
    storage_lifetime_interpol.loc[year] = row.iloc[0]
# after 2030
for year in range(2030+1,outyear+1):
    # storage_lifetime_interpol = pd.concat([storage_lifetime_interpol, pd.Series(storage_lifetime_interpol.loc[storage_lifetime_interpol.last_valid_index()], name=year)])
    row = pd.DataFrame([storage_lifetime_interpol.loc[storage_lifetime_interpol.last_valid_index()]])
    storage_lifetime_interpol.loc[year] = row.iloc[0]

storage_lifetime_interpol = storage_lifetime_interpol.sort_index(axis=0)
# drop the PHS from the interpolated lifetime frame, as the PHS is calculated separately
storage_lifetime_interpol = storage_lifetime_interpol.drop(columns=['PHS'])

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
        multiplier_pre = stock.loc[switchtime,region]/survival_init.sum()  # the stock is subdivided by the previous cohorts according to the survival function (only allowed when assuming steady stock inflow) 
        
        #pre-calculate the stock as lists (for efficiency)
        initial_stock_years = [np.flip(survival_init[0:pre_time+1]) * multiplier_pre]
            
        for year in range(1, (outyear-switchtime)+1):    # then fill the columns with the remaining fractions
            initial_stock_years.append(initial_stock_years[0] * survival_init[year])
    
        stock_cohorts.loc[idx[region,:],idx[init_tech, list(range(switchtime-pre_time,switchtime+1))]] = initial_stock_years       # fill the stock dataframe according to the pre-calculated stock 
        outflow_cohorts.loc[idx[region,:],idx[init_tech, list(range(switchtime-pre_time,switchtime+1))]] = stock_cohorts.loc[idx[region,:],idx[init_tech, list(range(switchtime-pre_time,switchtime+1))]].shift(1, axis=0) - stock_cohorts.loc[idx[region,:],idx[init_tech, list(range(switchtime-pre_time,switchtime+1))]]
    
        # set the other stock cohorts to zero
        stock_cohorts.loc[idx[region,:],idx[techlist_new, pre_year_list]] = 0
        outflow_cohorts.loc[idx[region,:],idx[techlist_new, pre_year_list]] = 0
        inflow_by_tech.loc[idx[region, switchtime], techlist_new] = 0    # inflow of other technologies in 1990 = 0
        
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
#%% 1) start with materials in generation capacity (this is the easiest, as the stock AND the new capacity is pre-calculated in TIMER, 
# based on fixed lifetime assumptions, we only add the outflow, based on a fixed lifetime DSM and the same lifetimes as in TIMER)
###########################################################################################################
###########################################################################################################


###########################################################################################################
#%% 1.1) Apply the DSM to find inflow & outflow of Generation capacity
###########################################################################################################


from dynamic_stock_model import DynamicStockModel as DSM
   
# In order to calculate inflow & outflow smoothly (without peaks for the initial years), we calculate a historic tail to the stock, 
# by adding a 0 value for first year of operation (=1926), then interpolate values towards 1971
def stock_tail(stock):
    zero_value = [0 for i in range(0,regions)]
    stock_used = pd.DataFrame(stock).reindex_like(stock)
    stock_used.loc[first_year_grid] = zero_value  # set all regions to 0 in the year of initial operation
    stock_new  = stock_used.reindex(list(range(first_year_grid,outyear+1))).interpolate()
    return stock_new

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

# Calculate the historic tail to the Gcap (stock) 
gcap_new = pd.DataFrame(index=pd.MultiIndex.from_product([range(first_year_grid,outyear+1), region_list], names=['years', 'regions']), columns=gcap.columns)

for tech in gcap_tech_list:
    gcap_new.loc[idx[:,:],tech] = stock_tail(gcap.loc[idx[:,:],tech].unstack(level=1)).stack()

# then apply the Dynamic Stock Model to find inflow & outflow (5yr moving average)
index = pd.MultiIndex.from_product([list(range(startyear,outyear+1)), gcap.index.levels[1]])
gcap_inflow  = pd.DataFrame(index=index, columns=pd.MultiIndex.from_product([gcap.columns, gcap_material_list], names=['technologies','materials']))
gcap_outflow = pd.DataFrame(index=index, columns=pd.MultiIndex.from_product([gcap.columns, gcap_material_list], names=['technologies','materials']))
gcap_stock   = pd.DataFrame(index=index, columns=pd.MultiIndex.from_product([gcap.columns, gcap_material_list], names=['technologies','materials']))

# Materials in Gcap (in: Gcap: MW, lifetime: yrs, materials intensity: gram/MW)
for tech in gcap_tech_list: 
    gcap_inflow.loc[idx[:,:],idx[tech,:]], gcap_outflow.loc[idx[:,:],idx[tech,:]], gcap_stock.loc[idx[:,:],idx[tech,:]] = inflow_outflow(gcap_new.loc[idx[:,:],tech].unstack(level=1), gcap_lifetime.loc[:,tech], gcap_materials_interpol.loc[idx[:,:],tech].unstack(), tech)

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
gcap_materials_all = gcap_materials_all.reorder_levels([3, 2, 1, 0, 5, 4]) / 1000000000   # gram to kt

# gcap_materials_all.to_csv(path_elma / 'output' / scen_folder / sa_settings / 'gcap_materials_output_kt.csv') # in kt
# gcap_materials_all.to_csv(path_elma_out / 'gcap_materials_output_kt.csv') # in kt



###########################################################################################################
#%%% Visualize STOCKS Materials
###########################################################################################################

#================================================================================
#%%%% SUM over TECHs - per region

dict_materials_colors = {
    'Steel':     '#FF9B85',
    'Aluminium': '#B9FAF8',
    'Concrete':  '#AAF683',
    'Plastics':  '#60D394',
    'Glass':     '#EE6055',
    'Cu':        '#FB6376',
    'Nd':        '#B8D0EB',
    'Ta':        '#B298DC',
    'Co':        "#F669C0",
    'Pb':        '#6F2DBD',
    'Mn':        "#31E7E7",
    'Ni':        '#FCB1A6',
    'Other':     '#FFD97D'
}

# Sum over technologies dimension
gcap_stock_reg_mat = gcap_stock.groupby(['flow', 'regions', 'materials']).sum()
gcap_stock_reg_mat = gcap_stock_reg_mat.droplevel('flow').T

data_all = gcap_stock_reg_mat.copy()

# Selected regions ------------------------------------------------------

# Pick desired regions by name
regions = ["Brazil", "C.Europe", "China"] 

types_level1 = [m for m in data_all.columns.get_level_values(1).unique() if m in ["Steel", "Concrete"]]
types_level2 = [m for m in data_all.columns.get_level_values(1).unique() if m in ["Aluminium", "Cu"]]
types_level3 = [m for m in data_all.columns.get_level_values(1).unique() if m not in (types_level1 + types_level2)]

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 12))
linewidth = 2
s_legend = 12
s_label = 14

for i, region in enumerate(regions):
    # Top row: Level 1 types
    for t in types_level1:
        if (region, t) in data_all.columns:
            data_plot = data_all[(region, t)]
            axes[0, i].plot(data_plot.index, data_plot.values, label=t, color=dict_materials_colors[t], linewidth=linewidth)
    axes[0, i].grid(alpha=0.2)
    axes[0, i].set_title(region)
    axes[0, i].legend(loc='upper left', fontsize=s_legend)

    # Middle row: Level 2 types
    for t in types_level2:
        if (region, t) in data_all.columns:
            data_plot = data_all[(region, t)]
            axes[1, i].plot(data_plot.index, data_plot.values, label=t, color=dict_materials_colors[t], linewidth=linewidth)
    axes[1, i].grid(alpha=0.2)
    axes[1, i].legend(loc='upper left', fontsize=s_legend)

    # Bottom row: Level 3 types
    for t in types_level3:
        if (region, t) in data_all.columns:
            data_plot = data_all[(region, t)]
            axes[2, i].plot(data_plot.index, data_plot.values, label=t, color=dict_materials_colors[t], linewidth=linewidth)
    axes[2, i].grid(alpha=0.2)
    axes[2, i].set_xlabel("Time")
    axes[2, i].legend(loc='upper left', fontsize=s_legend)

# Y-axis labels on the left
axes[0, 0].set_ylabel("Value")
axes[1, 0].set_ylabel("Value")
axes[2, 0].set_ylabel("Value")


plt.suptitle("Generation - Stock Materials", fontsize=16)
plt.tight_layout()
region_str = "_".join(regions)
fig.savefig(path_elma_out / f"ELMA_Gen_stock-materials_{region_str}_1971.png", dpi=300)
plt.show()




#================================================================================
#%%%% SUM over TECHs - world

dict_materials_colors = {
    'Steel':     '#FF9B85',
    'Aluminium': '#B9FAF8',
    'Concrete':  '#AAF683',
    'Plastics':  '#60D394',
    'Glass':     '#EE6055',
    'Cu':        '#FB6376',
    'Nd':        '#B8D0EB',
    'Ta':        '#B298DC',
    'Co':        '#F669C0',
    'Pb':        '#6F2DBD',
    'Mn':        "#31E7E7",
    'Ni':        '#FCB1A6',
    'Other':     '#FFD97D'
}

# data IEA ---------------------------------------------------------------------
# values for APS  scenario (NZE scenario)

# Cu -------------
# years = np.array([2024, 2030, 2035, 2040, 2045, 2050])
# values_solar = np.array([1657, 2803, 2726, 2626, 2448, 2758])*1000 # values in kt -> to kg
# values_wind = np.array([534, 1171, 1038, 919, 968, 1314])*1000 # values in kt -> to kg
# values = values_solar + values_wind
# df_iea_gen_cu = pd.DataFrame({ # lines & transformers, copper
#     'Year': years,
#     'Cu': values
# })
# df_iea_gen_cu.set_index('Year', inplace=True)
#------- ---------------------------------------------------------------------

gcap_stock_reg_mat = gcap_stock.groupby(['flow', 'regions', 'materials']).sum()
gcap_stock_reg_mat = gcap_stock_reg_mat.droplevel('flow').T
gcap_stock_mat = gcap_stock_reg_mat.groupby(level='materials', axis=1).sum()

data_all = gcap_stock_mat.copy()/1_000_000 # convert grams to tonnes

types_level1 = ["Steel", "Concrete"]
types_level2 = ["Aluminium", "Cu"]
types_level3 = [m for m in data_all.columns if m not in (types_level1 + types_level2)]



fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 10))
linewidth = 2
s_legend = 12
s_label = 14

# Top row: types_level1
for t in types_level1:
    if t in data_all.columns:
        data_plot = data_all[t]
        axes[0].plot(data_plot.index, data_plot.values, label=t, color=dict_materials_colors[t], linewidth=linewidth)

axes[0].grid(alpha=0.2)
axes[0].legend(loc='upper left', fontsize=s_legend)
axes[0].set_ylabel('Material stock (t)', fontsize=s_label)
axes[0].tick_params(axis='both', which='major', labelsize=s_legend) # set font size of axis ticks

# Middle row: types_level2
for t in types_level2:
    if t in data_all.columns:
        data_plot = data_all[t]
        axes[1].plot(data_plot.index, data_plot.values, label=t, color=dict_materials_colors[t], linewidth=linewidth)

axes[1].grid(alpha=0.2)
axes[1].legend(loc='upper left', fontsize=s_legend)
axes[1].set_ylabel('Material stock (t)', fontsize=s_label)
axes[1].tick_params(axis='both', which='major', labelsize=s_legend)

# Bottom row: types_level3
for t in types_level3:
    if t in data_all.columns:
        data_plot = data_all[t]
        axes[2].plot(data_plot.index, data_plot.values, label=t, color=dict_materials_colors[t], linewidth=linewidth)

axes[2].grid(alpha=0.2)
axes[2].legend(loc='upper left', fontsize=s_legend)
axes[2].set_xlabel('Time', fontsize=s_label)
axes[2].set_ylabel('Material stock (t)', fontsize=s_label)
axes[2].tick_params(axis='both', which='major', labelsize=s_legend)

plt.suptitle('ELMA: Generation - Stocks Materials - World', fontsize=16)
plt.tight_layout()
fig.savefig(path_elma_out / 'ELMA_Gen_stock-materials_world_1971.png', dpi=300)
# fig.savefig(path_elma_out / 'ELMA_Gen_stock-materials_world_1971.pdf', dpi=300)
fig.savefig(path_elma_out / 'ELMA_Gen_stock-materials_world_1971.svg', dpi=300)
plt.show()




#================================================================================
#%%%% Per TECH category - world

# Define mapping: technology -> category
gen_tech_to_category = {
    "Solar PV": 'Solar', 
    "Solar PV residential": 'Solar',
    "CSP": 'Solar', 
    "Wind onshore": 'Wind',
    "Wind offshore": 'Wind', 
    "Wave": 'Other Renewables',
    "Hydro": 'Other Renewables',
    "Other Renewables": 'Other Renewables',
    "Geothermal": 'Other Renewables',
    'Hydrogen power': 'Hydrogen',
    "Nuclear": 'Nuclear',
    "Conv. Coal": 'Fossil',
    "Conv. Oil": 'Fossil',
    "Conv. Natural Gas": 'Fossil',
    "Waste": 'Fossil',
    "IGCC": 'Fossil',
    "OGCC": 'Fossil',
    "NG CC": 'Fossil',
    "Biomass CC": 'Biomass',
    "Coal + CCS": 'Fossil + CCS',
    "Oil/Coal + CCS": 'Fossil + CCS',
    "Natural Gas + CCS": 'Fossil + CCS',
    "Biomass + CCS": 'Biomass',
    "CHP Coal": 'Fossil',
    "CHP Oil": 'Fossil',
    "CHP Natural Gas": 'Fossil',
    "CHP Biomass": 'Biomass',
    "CHP Geothermal": 'Other Renewables',
    "CHP Hydrogen": 'Hydrogen',
    "CHP Coal + CCS": 'Fossil + CCS',
    "CHP Oil + CCS": 'Fossil + CCS',
    "CHP Natural Gas + CCS": 'Fossil + CCS',
    "CHP Biomass + CCS": 'Biomass'
}

dict_gentechcat_colors = {
    'Solar':             "#FBBF09",
    'Wind':              "#4BABFF",
    'Biomass':           "#42DD88",
    'Other Renewables':  "#B6F795",
    'Hydrogen':          '#B9FAF8',
    'Nuclear':           "#B06106",
    'Fossil':            "#575354",
    'Fossil + CCS':      "#BBB8B9"
}

# data IEA ---------------------------------------------------------------------
# values for APS  scenario (NZE scenario)

# Cu -------------
# years = np.array([2024, 2030, 2035, 2040, 2045, 2050])
# values_solar = np.array([1657, 2803, 2726, 2626, 2448, 2758])*1000 # values in kt -> to kg
# values_wind = np.array([534, 1171, 1038, 919, 968, 1314])*1000 # values in kt -> to kg
# values = values_solar + values_wind
# df_iea_gen_cu = pd.DataFrame({ # lines & transformers, copper
#     'Year': years,
#     'Cu': values
# })
# df_iea_gen_cu.set_index('Year', inplace=True)
#------- ---------------------------------------------------------------------


# Step 1: Get technology level from index
tech_level = gcap_stock.index.get_level_values(3)
# Step 2: Map technologies to categories
category = tech_level.map(gen_tech_to_category)
# Step 2: Build a new MultiIndex with the category as level 3
new_index = pd.MultiIndex.from_arrays([
    gcap_stock.index.get_level_values(0),  # 'stock'
    gcap_stock.index.get_level_values(1),  # Region
    gcap_stock.index.get_level_values(2),  # Material
    category                                # Mapped Category
], names=gcap_stock.index.names)
# Step 3: Assign new index
gcap_stock_techcat = gcap_stock.copy()
gcap_stock_techcat.index = new_index
# Step 5: Group by (region, material, category), sum
gcap_stock_techcat = gcap_stock_techcat.groupby(
    [gcap_stock_techcat.index.get_level_values(1),  # Region
     gcap_stock_techcat.index.get_level_values(2),  # Material
     gcap_stock_techcat.index.get_level_values(3)]  # Category
).sum()
gcap_stock_techcat.index.names = ['regions', 'materials', 'technology_category']
gcap_stock_techcat = gcap_stock_techcat.T # index = years

gcap_stock_techcat_mat = gcap_stock_techcat.groupby(level=['materials','technology_category'], axis=1).sum()
data_all = gcap_stock_techcat_mat.copy()
# rearrange column order for the stacked plot
desired_order = ['Fossil', 'Fossil + CCS', 'Nuclear', 'Hydrogen', 'Biomass', 'Wind', 'Solar', 'Other Renewables']
new_columns = []
for material in data_all.columns.get_level_values(0).unique():
    for cat in desired_order:
        if (material, cat) in data_all.columns:
            new_columns.append((material, cat))
data_all = data_all.loc[:, new_columns]
data_all = data_all/ 1_000_000  # Convert from gram to tonnes (t)



materials = ['Steel', 'Aluminium', 'Nd', 'Co']
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
s_legend = 12
s_label = 14
for i, material in enumerate(materials):
    if material in data_all.columns.get_level_values(0):
        row = i // 2
        col = i % 2
        # Select data for this material (columns under this material)
        data_plot = data_all.loc[:, material]
        colors = [dict_gentechcat_colors[cat] for cat in data_plot.columns] # select colors based on technology category
        data_plot.plot.area(ax=axes[row, col], stacked=True, color = colors)

        axes[row, col].set_title(material, fontsize=15)
        axes[row, col].set_ylabel('Material stock (t)', fontsize=s_label)
        handles, labels = axes[row, col].get_legend_handles_labels() # reverse the order of legend to match the stacked plot
        axes[row, col].legend(handles[::-1], labels[::-1], loc='upper left', fontsize=s_legend)

        # Scientific notation for y-axis
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits((-2, 2))  # Force scientific for large/small numbers
        axes[row, col].yaxis.set_major_formatter(formatter)
        axes[row, col].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
        axes[row, col].tick_params(axis='both', which='major', labelsize=s_legend) # set font size of axis ticks

for col in range(2): # Set x-labels only for bottom row
    axes[1, col].set_xlabel('Year', fontsize=s_label)

plt.suptitle("ELMA: Generation - Stocks Materials per Tech. Cat. - World", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to make room for the suptitle
# fig.savefig(path_elma_out / "ELMA_Gen_stock-materials-techcat_st-al-nd-co_world_1971.png", dpi=300)
# fig.savefig(path_elma_out / "ELMA_Gen_stock-materials-techcat_st-al-nd-co_world_1971.pdf", dpi=300)
# fig.savefig(path_elma_out / "ELMA_Gen_stock-materials-techcat_st-al-nd-co_world_1971.svg", dpi=300)
plt.show()




###########################################################################################################
#%%% Visualize Inflow Materials
###########################################################################################################


#================================================================================
#%%%% SUM over TECHs - per region

# Sum over technologies dimension
gcap_inflow_reg_mat = gcap_inflow.groupby(['flow', 'regions', 'materials']).sum()
gcap_inflow_reg_mat = gcap_inflow_reg_mat.droplevel('flow').T
regions = gcap_inflow_reg_mat.columns.get_level_values(0).unique()[:2]  # First 2 regions

types_level1 = [m for m in gcap_inflow_reg_mat.columns.get_level_values(1).unique() if m in ["Steel", "Concrete"]]
types_level2 = [m for m in gcap_inflow_reg_mat.columns.get_level_values(1).unique() if m in ["Aluminium", "Cu"]]
types_level3 = [m for m in gcap_inflow_reg_mat.columns.get_level_values(1).unique() if m not in (types_level1 + types_level2)]

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 12), sharex=True)
axes[0, 1].sharey(axes[0, 0])
axes[1, 1].sharey(axes[1, 0])
axes[2, 1].sharey(axes[2, 0])

data_all = gcap_inflow_reg_mat.copy()

for i, region in enumerate(regions):
    # Top row: 
    for t in types_level1:
        # Select the column using (region, material) tuple
        if (region, t) in data_all.columns:
            data_plot = data_all[(region, t)]  # This gives you a Series with years as index
            axes[0, i].plot(data_plot.index, data_plot.values, label=t)
    
    axes[0, i].set_title(f"{region}")
    axes[0, i].set_xlabel("Time")
    axes[0, i].legend()

    # Middle row: 
    for t in types_level2:
        if (region, t) in data_all.columns:
            data_plot = data_all[(region, t)]
            axes[1, i].plot(data_plot.index, data_plot.values, label=t)
    
    axes[1, i].set_title(f"{region}")
    axes[1, i].set_xlabel("Time")
    axes[1, i].legend(loc='upper left')

    # Bottom row: 
    for t in types_level3:
        if (region, t) in data_all.columns:
            data_plot = data_all[(region, t)]
            axes[2, i].plot(data_plot.index, data_plot.values, label=t)
    
    axes[2, i].set_title(f"{region}")
    axes[2, i].set_xlabel("Time")
    axes[2, i].legend(loc='upper left')

# Label only left side with y-axis label
axes[0, 0].set_ylabel("Value")
axes[1, 0].set_ylabel("Value")
axes[2, 0].set_ylabel("Value")

plt.suptitle("Generation - Inflow Materials", fontsize=16)
plt.tight_layout()
# fig.savefig(path_elma_out / "ELMA_Gen_inflow-materials_1971_Brazil-CEurope.png", dpi=300)
plt.show()




#================================================================================
#%%%% SUM over TECHs - world

dict_materials_colors = {
    'Steel':     '#FF9B85',
    'Aluminium': '#B9FAF8',
    'Concrete':  '#AAF683',
    'Plastics':  '#60D394',
    'Glass':     '#EE6055',
    'Cu':        '#FB6376',
    'Nd':        '#B8D0EB',
    'Ta':        '#B298DC',
    'Co':        '#A663CC',
    'Pb':        '#6F2DBD',
    'Mn':        "#31E7E7",
    'Ni':        '#FCB1A6',
    'Other':     '#FFD97D'
}

# data IEA ---------------------------------------------------------------------
# values for APS  scenario (NZE scenario)
# Cu -------------
years = np.array([2024, 2030, 2035, 2040, 2045, 2050])
values_solar = np.array([1657, 2803, 2726, 2626, 2448, 2758])*1000 # values in kt -> to kg
values_wind = np.array([534, 1171, 1038, 919, 968, 1314])*1000 # values in kt -> to kg
values = values_solar + values_wind
df_iea_gen_cu = pd.DataFrame({ # lines & transformers, copper
    'Year': years,
    'Cu': values
})
df_iea_gen_cu.set_index('Year', inplace=True)
#------- ---------------------------------------------------------------------

# Sum over technologies dimension
gcap_inflow_reg_mat = gcap_inflow.groupby(['flow', 'regions', 'materials']).sum()
gcap_inflow_reg_mat = gcap_inflow_reg_mat.droplevel('flow').T
gcap_inflow_mat = gcap_inflow_reg_mat.groupby(level='materials', axis=1).sum()

types_level1 = ["Steel", "Concrete"]
types_level2 = ["Aluminium", "Cu"]
types_level3 = [m for m in gcap_inflow_mat.columns if m not in (types_level1 + types_level2)]



fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 10), sharex=True)
data_all = gcap_inflow_mat.copy()
# Top row: types_level1
for t in types_level1:
    if t in data_all.columns:
        data_plot = data_all[t]
        axes[0].plot(data_plot.index, data_plot.values, label=t, color=dict_materials_colors[t])

axes[0].set_xlabel("Time")
axes[0].set_ylabel("Value")
axes[0].legend()

# Middle row: types_level2
for t in types_level2:
    if t in data_all.columns:
        data_plot = data_all[t]
        axes[1].plot(data_plot.index, data_plot.values, label=t, color=dict_materials_colors[t])
    if t == "Cu":
        # Add IEA data for copper
        axes[1].scatter(df_iea_gen_cu.index, df_iea_gen_cu['Cu'], label="IEA Cu solar+wind", color='#00a5cf', s=4)

axes[1].set_xlabel("Time")
axes[1].set_ylabel("Value")
axes[1].legend(loc='upper left')

# Bottom row: types_level3
for t in types_level3:
    if t in data_all.columns:
        data_plot = data_all[t]
        axes[2].plot(data_plot.index, data_plot.values, label=t, color=dict_materials_colors[t])

axes[2].set_xlabel("Time")
axes[2].set_ylabel("Value")
axes[2].legend(loc='upper left')

plt.suptitle("ELMA: Generation - Inflow Materials - World", fontsize=16)
plt.tight_layout()
# fig.savefig(path_elma_out / "ELMA_Gen_inflow-materials_world_1971.png", dpi=300)
# fig.savefig(path_elma_out / "ELMA_Gen_inflow-materials_world_1971.pdf", dpi=300)
# fig.savefig(path_elma_out / "ELMA_Gen_inflow-materials_world_1971.svg", dpi=300)
plt.show()








###########################################################################################################
#%% Average generation material intensity calculations (originally weight is in grams)
###########################################################################################################

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
# intensity_gcap.to_csv(path_elma_out / 'material_intensity_gcap_ton_per_MW.csv') # ton/MW

###########################################################################################################
#%% 2) Determine MARKET SHARE of the storage capacity using a multi-nomial logit function
###########################################################################################################


# determine the annual % decline of the costs based on the 2018-2030 data (original, before applying the malus)
decline = ((storage_costs_interpol.loc[storage_start,:]-storage_costs_interpol.loc[storage_end,:])/(storage_end-storage_start))/storage_costs_interpol.loc[storage_start,:]
decline_used = decline*storage_ltdecline

# storage_costs_interpol.index = storage_malus_interpol.index # ADDED, to avoid index mismatch in the next step
storage_costs_new = storage_costs_interpol * storage_malus_interpol

# calculate the development from 2030 to 2050 (using annual price decline)
for year in range(storage_end+1,2050+1):
    # print(year)
    # storage_costs_new = storage_costs_new.append(pd.Series(storage_costs_new.loc[storage_costs_new.last_valid_index()]*(1-decline_used), name=year))
    # storage_costs_new = pd.concat([storage_costs_new, pd.Series(storage_costs_new.loc[storage_costs_new.last_valid_index()] * (1 - decline_used), name=year)])
    row = pd.DataFrame([storage_costs_new.loc[storage_costs_new.last_valid_index()] * (1 - decline_used)])
    storage_costs_new.loc[year] = row.iloc[0]

# for historic price development, assume 2x AVERAGE annual price decline on all technologies, except lead-acid (so that lead-acid gets a relative price advantage from 1970-2018)
for year in reversed(range(startyear,storage_start)):
    # storage_costs_new = storage_costs_new.append(pd.Series(storage_costs_new.loc[storage_costs_new.first_valid_index()]*(1+(2*decline_used.mean())), name=year)).sort_index(axis=0)
    row = pd.DataFrame([storage_costs_new.loc[storage_costs_new.first_valid_index()]*(1+(2*decline_used.mean()))])
    storage_costs_new.loc[year] = row.iloc[0]

storage_costs_new.sort_index(axis=0, inplace=True) 
storage_costs_new.loc[1971:2017,'Deep-cycle Lead-Acid'] = storage_costs_new.loc[2018,'Deep-cycle Lead-Acid']  # restore the exception (set to constant 2018 values)

# Multinomial Logit function, assumes input of an ordered dataframe with rows as years and columns as technologies, values as prices. Logitpar is the calibrated Logit parameter (usually a nagetive number between 0 and 1)
def MNLogit(df, logitpar):
    new_dataframe = pd.DataFrame(index=df.index, columns=df.columns)
    for year in range(df.index[0],df.index[-1]+1): #from first to last year
        yearsum = 0
        for column in df.columns:
            yearsum += math.exp(logitpar * df.loc[year,column]) # calculate the sum of the prices
        for column in df.columns:
            new_dataframe.loc[year,column] = math.exp(logitpar * df.loc[year,column])/yearsum
    return new_dataframe      # the retuned dataframe contains the market shares

# use the storage price development in the logit model to get market shares
storage_market_share = MNLogit(storage_costs_new, -0.2)

# fix the market share of storage technologies after 2050
for year in range(2050+1,outyear+1):
    # storage_market_share = storage_market_share.append(pd.Series(storage_market_share.loc[storage_market_share.last_valid_index()], name=year))
    row = pd.DataFrame([storage_market_share.loc[storage_market_share.last_valid_index()]])
    storage_market_share.loc[year] = row.iloc[0]


###########################################################################################################
###########################################################################################################
#%% 2) Second Section: Availability of storage in vehicles & pumped hydro. 
###########################################################################################################
###########################################################################################################


###########################################################################################################
#%% 2.1) Vehicles (total nr. of)
#2.1##########################################################################################################


# BEV & PHEV vehicle stats
BEV_capacity  = 59.6    #kWh current battery capacity of full electric vehicles, see current_specs.xlsx
PHEV_capacity = 11.2    #kWh current battery capacity of plugin electric vehicles, see current_specs.xlsx

if sa_settings == 'high_stor':
   capacity_usable_PHEV = 0.025   # 2.5% of capacity of PHEV is usable as storage (in the pessimistic sensitivity variant)
   capacity_usable_BEV  = 0.05    # 5  % of capacity of BEVs is usable as storage (in the pessimistic sensitivity variant)
else: 
   capacity_usable_PHEV = 0.05    # 5% of capacity of PHEV is usable as storage
   capacity_usable_BEV  = 0.10    # 10% of capacity of BEVs is usable as storage

vehicle_kms = passengerkms.loc[:outyear] * 1000000000000 / loadfactor.loc[:outyear]        # conversion from tera-Tkms  
vehicles_all = vehicle_kms / kilometrage.loc[:outyear]

vehicles_PHEV = vehicles_all * PHEV_share.loc[:outyear]
vehicles_BEV = vehicles_all * BEV_share.loc[:outyear]
vehicles_EV = vehicles_PHEV + vehicles_BEV

# For the availability of vehicle store capacity we apply the assumption of fixed weight,
# So we first need to know the average density of the stock of EV batteries
# to get there we need to know the share of the technologies in the stock (based on lifetime & share in the inflow/purchases)
# to get there, we first need to know the market share of new stock additions (=inflow/purchases)

#First we calculate the share of the inflow using only a few of the technologies in the storage market share
#The selection represents only the batteries that are suitable for EV & mobile applications
EV_battery_list = ['NiMH', 'LMO', 'NMC', 'NCA', 'LFP', 'Lithium Sulfur', 'Lithium Ceramic ', 'Lithium-air']
#normalize the selection of market shares, so that total market share is 1 again (taking the relative share in the selected battery techs)
market_share_EVs = pd.DataFrame().reindex_like(storage_market_share[EV_battery_list])

for year in list(range(startyear,outyear+1)):
    for tech in EV_battery_list:
        market_share_EVs.loc[year, tech] = storage_market_share[EV_battery_list].loc[year,tech] /storage_market_share[EV_battery_list].loc[year].sum()

# then we use that market share in combination with the stock developments to derive the stock share 
# Here we use the vehcile stock (number of cars) as a proxy for the development of the battery stock (given that we're calculating the actual battery stock still, and just need to account for the dynamics of purchases to derive te stock share here) 
EV_inflow_by_tech, EV_stock_cohorts, EV_outflow_cohorts = stock_share_calc(vehicles_EV, market_share_EVs, 'NiMH', ['NiMH', 'LMO', 'NMC', 'NCA', 'LFP', 'Lithium Sulfur', 'Lithium Ceramic ', 'Lithium-air'])

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

# EV_storage_stock_share.to_csv(path_elma_out / 'battery_share_stock.csv') # Average global car battery share (in stock) is exported to be used in paper on vehicles
# EV_storage_inflow_share.to_csv(path_elma_out / 'battery_share_inflow.csv') # Average global car battery share (in inflow) is exported to be used in paper on vehicles
  
#The global share of the battery technologies in stock is then used to derive the (weihgted) average density (kg/kWh)
weighted_average_density_stock  = EV_storage_stock_share.mul(storage_density_interpol[EV_battery_list]).sum(axis=1)
weighted_average_density_inflow = EV_storage_inflow_share.mul(storage_density_interpol[EV_battery_list]).sum(axis=1)

# weighted_average_density_stock.loc[:outyear].to_csv(path_elma_out / 'ev_battery_density_stock.csv')        # Average car battery density (in stock) is exported to be used in paper on vehicles
# weighted_average_density_inflow.loc[:outyear].to_csv(path_elma_out / 'ev_battery_density_inflow.csv')      # Average car battery density (in inflow) is exported to be used in paper on vehicles

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

###########################################################################################################
#%% 2.2) Take the TIMER Hydro-dam capacity (MW) & compare it to Pumped hydro capacity (MW) projections from the International Hydropower Association
#2.2##########################################################################################################

Gcap_hydro = gcap_data[['time','DIM_1', 7]].pivot_table(index='time', columns='DIM_1')   # IMAGE-TIMER Hydro dam capacity (power, in MW)
Gcap_hydro.columns = region_list
Gcap_hydro = Gcap_hydro.loc[:outyear]

#storage capacity in MW (power capacity), to compare it to Pumped hydro storage projections (also given in MW, power capacity)
# storage_power = read_mym_df(path_scenario_output / 'StorCapTot.out')   
storage_power = read_mym_df(path_image_output / 'StorCapTot.out')               
storage_power.drop(storage_power.iloc[:, -2:], inplace = True, axis = 1)    
storage_power.columns = region_list
storage_power = storage_power.loc[:outyear]

#Disaggregate the Pumped hydro-storgae projections to 26 IMAGE regions according to the relative Hydro-dam power capacity (also MW) within 5 regions reported by the IHS (international Hydropwer Association)
# phs_projections = pd.read_csv(path_scenario_output / 'PHS.csv', index_col='t')                                  # pumped hydro storage capacity (MW)
phs_projections = pd.read_csv(path_external_data_standard / 'PHS.csv', index_col='t')
phs_regions = [[10,11],[19],[1],[22],[0,2,3,4,5,6,7,8,9,12,13,14,15,16,17,18,20,21,23,24,25]]   # subregions in IHS data for Europe, China, US, Japan, RoW, MIND: region refers to IMAGE region MINUS 1
phs_projections_IMAGE = pd.DataFrame(index=Gcap_hydro.index, columns=Gcap_hydro.columns)        # empty dataframe

for column in range(0,len(phs_regions)):
    sum_data = Gcap_hydro.iloc[:,phs_regions[column]].sum(axis=1)                               # first, get the sum of all hydropower in the IHS regions (to divide over in second step)
    for region in range(0,regions):
        if region in phs_regions[column]:
            phs_projections_IMAGE.iloc[:,region] = phs_projections.iloc[:,column] * (Gcap_hydro.iloc[:,region]/sum_data)

# Then fill the years after 2030 (end of IHS projections) according to the Gcap annual growth rate (assuming a fixed percentage of Hydro dams will be built with Pumped hydro capabilities after )
if sa_settings == 'high_stor':
   phs_projections_IMAGE.loc[2030:outyear] =  phs_projections_IMAGE.loc[2030] * (Gcap_hydro.loc[2030:outyear]/Gcap_hydro.loc[2030:outyear])  # no growth after 2030 in the high_stor sensitivity variant
else:
   phs_projections_IMAGE.loc[2030:outyear] =  phs_projections_IMAGE.loc[2030] * (Gcap_hydro.loc[2030:outyear]/Gcap_hydro.loc[2030])

# Calculate the fractions of the storage capacity that is provided through pumped hydro-storage, electric vehicles or other storage (larger than 1 means the capacity superseeds the demand for energy storage, in terms of power in MW or enery in MWh) 
phs_storage_fraction = phs_projections_IMAGE.divide(storage_power.loc[:outyear]).clip(upper=1)      # the phs storage fraction deployed to fulfill storage demand, both phs & storage_power here are expressed in MW
storage_remaining = storage.loc[:outyear] * (1 - phs_storage_fraction)

if sa_settings == 'high_stor':
   oth_storage_fraction = 0.5 * storage_remaining 
   oth_storage_fraction += ((storage_remaining * 0.5) - storage_vehicles).clip(lower=0)    
   oth_storage_fraction = oth_storage_fraction.divide(storage).where(oth_storage_fraction > 0, 0).clip(lower=0) 
   evs_storage_fraction = 1 - (phs_storage_fraction + oth_storage_fraction)     # electric vehicle storage (BEV + PHEV) capacity and total storage demand are expressed as MWh
else: 
   oth_storage_fraction = (storage_remaining - storage_vehicles).clip(lower=0)    
   oth_storage_fraction = oth_storage_fraction.divide(storage.loc[:outyear]).where(oth_storage_fraction > 0, 0).clip(lower=0)      
   evs_storage_fraction = 1 - (phs_storage_fraction + oth_storage_fraction)     # electric vehicle storage (BEV + PHEV) capacity and total storage demand are expressed as MWh
   
checksum = phs_storage_fraction + evs_storage_fraction + oth_storage_fraction   # should be 1 for all fields

# absolute storage capacity (MWh)
phs_storage_theoretical = phs_projections_IMAGE.divide(storage_power) * storage.loc[:outyear]       # theoretically available PHS storage (MWh; fraction * total) only used in the graphs that show surplus capacity
phs_storage = phs_storage_fraction * storage.loc[:outyear]
evs_storage = evs_storage_fraction * storage.loc[:outyear]
oth_storage = oth_storage_fraction * storage.loc[:outyear]

#output for Main text figure 2 (storage reservoir, in MWh for 3 storage types)
storage_out_phs = pd.concat([phs_storage], keys=['phs'], names=['type']) 
storage_out_evs = pd.concat([evs_storage], keys=['evs'], names=['type']) 
storage_out_oth = pd.concat([oth_storage], keys=['oth'], names=['type']) 
storage_out = pd.concat([storage_out_phs, storage_out_evs, storage_out_oth])
# storage_out.to_csv(path_elma_out / 'storage_by_type_MWh.csv')        # in MWh

# derive inflow & outflow (in MWh) for PHS, for later use in the material calculations 
PHS_kg_perkWh = 26.8    # kg per kWh storage capacity (as weight addition to existing hydro plants to make them pumped) 
phs_storage_stock_tail   = stock_tail(phs_storage.astype(float))
storage_lifetime_PHS = storage_lifetime['PHS'].reindex(list(range(first_year_grid,outyear+1)), axis=0).interpolate(limit_direction='both')
phs_storage_inflow, phs_storage_outflow, phs_storage_stock  = inflow_outflow(phs_storage_stock_tail, storage_lifetime_PHS, stor_materials_interpol.loc[idx[:,:],'PHS'].unstack() * PHS_kg_perkWh * 1000, 'PHS')    # PHS lifetime is fixed at 60 yrs anyway so, we simply select 1 value

###########################################################################################################
#%% Graphs output
###########################################################################################################


import graphs_elec

# graphs_elec.graph_global_ev_capacity(BEV_dynamic_capacity, PHEV_dynamic_capacity,  max_capacity_BEV, max_capacity_PHEV, scen_folder, sa_settings) # make a graph on the average capacity of EVs and their availability to V2G
# graphs_elec.graph_regional_storage(region_list, storage_PHEV, storage_BEV, storage, scen_folder, scenario, sa_settings)                           # make a graph on regional storage demand vs V2G supply
# graphs_elec.graph_storage_demand_vs_availability(region_list, phs_storage_theoretical, storage_vehicles, storage, scen_folder, sa_settings)       # make a graph on storage demand vs the availability in 3 different storage types (phs, evs and dedicated) 
# graphs_elec.graph_regional_dedicated_storage(oth_storage, scen_folder, scenario, sa_settings)                                                     # make a graph on regional dedicated storage demand
# graphs_elec.graph_market_share(storage_market_share, scen_folder, sa_settings)                                                                    # make a graph on the market shares of storage technologies
# graphs_elec.graph_market_share_pie(storage_market_share, scen_folder, sa_settings)                                                                # make a pie chart on the market share of dedicated storage technologies

###########################################################################################################
###########################################################################################################
#%% 3) Calculate the materials in dedicated electricity storage capacity
###########################################################################################################
###########################################################################################################

###########################################################################################################
#%% 3.2) apply the dedicated electricity storage market shares to the demand for dedicated (other) storage to find storage by type and by region
###########################################################################################################



###########################################################################################################
#%% -------- INflow & OUTflow calculations ---------
###########################################################################################################


# then, calculate the market share of technologies in the stock (for region & by year), also a global total market share of the stock is calculated to compare to the inflow market share 
inflow_by_tech, stock_cohorts, outflow_cohorts = stock_share_calc(oth_storage, storage_market_share, 'Deep-cycle Lead-Acid', list(storage_lifetime_interpol.columns)) # run the function that calculates stock shares from total stock & inflow shares
stock = stock_cohorts.loc[idx[:,:],idx[:,:]].sum(axis=1).groupby(level=0).sum() #.sum(axis=1, level=0)
storage_stock_abs = stock.sum(axis=0).groupby(level=1).sum()     #.sum(axis=0, level=1)                              # sum over all regions to get the global share of the stock
stock_total = stock.sum(axis=1).unstack(level=0)                            # total stock by region
storage_stock_share = pd.DataFrame(index=storage_stock_abs.index, columns=storage_stock_abs.columns)

for tech in storage_stock_abs.columns:
    storage_stock_share.loc[:,tech] = storage_stock_abs.loc[:,tech].div(storage_stock_abs.sum(axis=1))

# then, the sum of the global inflow & outflow is calculated for comparison in the figures below
outflow = outflow_cohorts.loc[idx[:,:],idx[:,:]].sum(axis=1).groupby(level=0).sum() #.sum(axis=1, level=0)
outflow_total = outflow.sum(axis=1).unstack(level=0)                        # total by regions

inflow_total = inflow_by_tech.sum(axis=1).unstack(level=0)


###########################################################################################################
###########################################################################################################
#%% 4) ---------- Material calculations ---------------
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
# v2g_vs_dedicated.tofile('output\\' + scen_folder + '\\' + sa_settings + '\\v2g_vs_dedicated.csv', sep=',')  # how much bigger is v2g than dedicated? 
# v2g_vs_dedicated.tofile(path_elma_out, 'v2g_vs_dedicated.csv', sep=',')  # how much bigger is v2g than dedicated? 

average_storage_intensity = storage_stock_share.mul(storage_density_interpol).sum(axis=1)  # kg /kWh
material_avoided = evs_storage.loc[[2045,2046,2047,2048,2049,2050],:].sum(axis=1).sum() * 1000 * (average_storage_intensity[[2045,2046,2047,2048,2049,2050]].sum()/6) / 1000000 # in kt
# material_avoided.tofile('output\\' + scen_folder + '\\' + sa_settings + '\\v2g_material_avoided_kt.csv', sep=',')  # how much weight of dedicated storage is avoided when V2G is assumed?, in kt 
# material_avoided.tofile(path_elma_out, 'v2g_material_avoided_kt.csv', sep=',')  # how much weight of dedicated storage is avoided when V2G is assumed?, in kt 

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

# output_by_tech.to_csv('output\\' + scen_folder + '\\' + sa_settings + '\\stor_materials_output_kt.csv', index=False) # in kt
# output_by_tech.to_csv(path_elma_out, 'stor_materials_output_kt.csv', index=False) # in kt

# total materials in storage (summed over all technologies) 
output_by_tech.set_index(['regions', 'flow', 'sector', 'category', 'technologies', 'materials'], inplace=True)
output_sum = output_by_tech.unstack(level=4).sum(axis=1, level=0)
# output_sum.to_csv('output\\' + scen_folder + '\\' + sa_settings + '\\export_storage_sum_kt.csv') 
# output_sum.to_csv(path_elma_out, 'export_storage_sum_kt.csv') 
