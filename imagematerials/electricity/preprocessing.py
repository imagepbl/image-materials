import pandas as pd
import numpy as np
import os
import math
import scipy
import warnings
from pathlib import Path

from imagematerials.distribution import ALL_DISTRIBUTIONS, NAME_TO_DIST
from imagematerials.read_mym import read_mym_df
from imagematerials.util import dataset_to_array, pandas_to_xarray
# from imagematerials.electricity.modelling_functions import
# from imagematerials.electricity.constants import ( # TODO: import not working at the moment
#     START_YEAR,
#     FIRST_YEAR,
#     END_YEAR,
#     OUT_YEAR,
#     SCEN,
#     REGIONS,
#     MEGA_TO_TERA,
#     PKMS_TO_VKMS,
#     TONNES_TO_KGS,
#     LOAD_FACTOR
# )
SCEN = 'SSP2' 
VARIANT = '2D_RE'
# import inspect
# path_current = Path(inspect.getfile(inspect.currentframe())).resolve()#.parent

# base_dir = "../data/raw"
# test = Path(base_dir)

# Define paths ----------------------------------------------------------------------
#YOUR_DIR = "C:\\Users\\Admin\\surfdrive\\Projects\\IRP\\GRO23\\Modelling\\2060\\ELMA"   # Change the running directory here
# os.chdir(YOUR_DIR)
scen_folder = SCEN + "_" + VARIANT
path_base = Path().resolve() # TODO absolute path of file "preprocessing.py" ? current solution can differ depending on IDE used (?) 
# path_base = path_current.parent.parent # base path of the project -> image-materials
path_image_output = Path(path_base, "data", "raw", SCEN)
path_external_data_standard = Path(path_base, "data", "raw", "electricity", "standard_data")
path_external_data_scenario = Path(path_base, "data", "raw", "electricity", scen_folder)

assert path_image_output.is_dir()
assert path_external_data_standard.is_dir()
assert path_external_data_scenario.is_dir()

sa_settings = "default"         # settings for the sensitivity analysis (default, high_stor, high_grid)

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
#%% Read in files
###########################################################################################################


# 1. External Data ======================================================================================== 

# STORAGE ------------------------------------------------------

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

kilometrage = pd.read_csv(path_external_data_scenario / 'kilometrage.csv', index_col='t')   #annual car mileage in kms/yr, based  mostly  on  Pauliuk  et  al.  (2012a)

# material compositions (storage)
storage_materials = pd.read_csv(path_external_data_standard / 'storage_materials_dynamic.csv',index_col=[0,1]).transpose()  # wt% of total battery weight for various materials, total battery weight is given by the density file above


# GENERATION ------------------------------------------------------

# lifetimes of Gcap tech (original data according to van Vuuren 2006, PhD Thesis)
gcap_lifetime = pd.read_csv(path_external_data_scenario / 'LTTechnical_dynamic.csv', index_col=['Year','DIM_1'])        

# material compositions (generation capacity)
composition_generation = pd.read_csv(path_external_data_scenario / 'composition_generation.csv',index_col=[0,1]).transpose()  # in gram/MW



# 2. IMAGE/TIMER files ====================================================================================

# STORAGE ------------------------------------------------------

# # The vehicle shares of trucks (heavy) of the SSP2
#     loadfactor_car_data: pd.DataFrame = read_mym_df(
#         image_folder.joinpath("trp_trvl_Load.out")). rename(
#         columns={
#             "DIM_1": "region"})

# read TIMER installed storage capacity (MWh, reservoir)
storage = read_mym_df(path_image_output.joinpath("StorResTot.out"))   #storage capacity in MWh (reservoir, so energy capacity, not power capacity, the latter is used later on in the pumped hydro storage calculations)
    
loadfactor_data = read_mym_df(path_image_output / 'trp_trvl_Load.out') 
  
passengerkms_data = read_mym_df(path_image_output / 'trp_trvl_pkm.out')   # passenger kilometers in Tera pkm

vehicleshare_data = read_mym_df(path_image_output / 'trp_trvl_Vshare_car.out')



# GENERATION ------------------------------------------------------

# Generation capacity (stock & inflow/new) in MW peak capacity, FILES from TIMER
gcap_data = read_mym_df(path_image_output / 'Gcap.out')






###########################################################################################################
#%% Prepare model specific variables
###########################################################################################################

# STORAGE ----------------------------------------------------------------

storage.drop(storage.iloc[:, -2:], inplace = True, axis = 1)    # drop global total column and empty (27) column

if sa_settings == 'high_stor':
   storage_multiplier = storage
   for year in range(2021,2051):
        storage_multiplier.loc[year] = storage.loc[year] * (1 + (1/30*(year-2020)))
   for year in range(2051,endyear+1):
        storage_multiplier.loc[year] = storage.loc[year] * 2


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

# material intensities for storage
for cat in list(storage_materials.columns.levels[1]):
   stor_materials_1st   = storage_materials.loc[:,idx[storage_materials.columns[0][0],cat]]
   stor_materials_interpol.loc[idx[first_year_grid ,:],cat] = stor_materials_1st.to_numpy()                # set the first year (1926) values to the first available values in the dataset (for the year 2000) 
   stor_materials_interpol.loc[idx[storage_materials.columns.levels[0].min(),:],cat] = storage_materials.loc[:, idx[storage_materials.columns.levels[0].min(),cat]].to_numpy()                # set the middle year (2000) values to the first available values in the dataset (for the year 2000) 
   stor_materials_interpol.loc[idx[storage_materials.columns.levels[0].max(),:],cat] = storage_materials.loc[:, idx[storage_materials.columns.levels[0].max(),cat]].to_numpy()                # set the last year (2100) values to the last available values in the dataset (for the year 2050) 
   stor_materials_interpol.loc[idx[:,:],cat] = stor_materials_interpol.loc[idx[:,:],cat].unstack().astype('float64').interpolate().stack()

# Car Batteries ----------

# kilometrage is defined untill 2008, fill 2008 values untill 2100 
kilometrage = kilometrage.reindex(list(range(startyear,endyear))).interpolate(limit_direction='both')
region_list = list(kilometrage.columns.values)                          # get a list with region names

loadfactor = loadfactor_data[['time','DIM_1', 5]].pivot_table(index='time', columns='DIM_1') # loadfactor for cars (in persons per vehicle)
loadfactor = loadfactor.loc[list(range(startyear,endyear+1))] * LOAD_FACTOR  # car loadfactor is expressed compared to an average global loadfactor from the IMAGE-TIMER-TRAVEL model (original: Girod, 2012; further elaborations by Edelenbosch et al.) 
  
indexNames = passengerkms_data[ passengerkms_data['DIM_1'] >= 27 ].index
passengerkms_data.drop(indexNames , inplace=True)
passengerkms = passengerkms_data[['time','DIM_1', 5]].pivot_table(index='time', columns='DIM_1').loc[list(range(startyear,endyear+1))]
  
BEV_collist = [22, 23, 24, 25] # battery electric vehicles
PHEV_collist= [21, 20, 19, 18, 17, 16] # plug-in hybrid electric vehicles

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


# GENERATION ----------------------------------------------------------------

gcap_tech_list = list(composition_generation.loc[:,idx[2020,:]].droplevel(axis=1, level=0).columns)    #list of names of the generation technologies (workaround to retain original order)
gcap_material_list = list(composition_generation.index.values)  #list of materials the generation technologies

gcap_data = gcap_data.loc[~gcap_data['DIM_1'].isin([27,28])]    # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
gcap = pd.pivot_table(gcap_data[gcap_data['time'].isin(list(range(startyear,endyear+1)))], index=['time','DIM_1'], values=list(range(1,epg_techs+1)))  #gcap as multi-index (index = years & regions (26); columns = technologies (34));  the last column in gcap_data (= totals) is now removed

# renaming multi-index dataframe: generation capacity, based on the regions in grid_length_Hv & technologies as given
gcap.index = pd.MultiIndex.from_product([list(range(startyear,endyear+1)), region_list], names=['years', 'regions'])
gcap.columns = gcap_tech_list

# Interpolate material intensities (dynamic content for gcap & storage technologies between 1926 to 2100, based on data files)
index = pd.MultiIndex.from_product([list(range(first_year_grid, outyear+1)), list(composition_generation.index)])
gcap_materials_interpol = pd.DataFrame(index=index, columns=composition_generation.columns.levels[1])

# material intensities for gcap
for cat in list(composition_generation.columns.levels[1]):
   gcap_materials_1st   = composition_generation.loc[:,idx[composition_generation.columns[0][0],cat]]
   gcap_materials_interpol.loc[idx[first_year_grid ,:],cat] = gcap_materials_1st.to_numpy()                # set the first year (1926) values to the first available values in the dataset (for the year 2000) 
   gcap_materials_interpol.loc[idx[composition_generation.columns.levels[0].min(),:],cat] = composition_generation.loc[:, idx[composition_generation.columns.levels[0].min(),cat]].to_numpy()                # set the middle year (2000) values to the first available values in the dataset (for the year 2000) 
   gcap_materials_interpol.loc[idx[composition_generation.columns.levels[0].max(),:],cat] = composition_generation.loc[:, idx[composition_generation.columns.levels[0].max(),cat]].to_numpy()                # set the last year (2100) values to the last available values in the dataset (for the year 2050) 
   gcap_materials_interpol.loc[idx[:,:],cat] = gcap_materials_interpol.loc[idx[:,:],cat].unstack().astype('float64').interpolate().stack()

# interpolate Gcap (technical) lifetime data
gcap_lifetime.index = gcap_lifetime.index.set_levels(gcap_tech_list, level=1)
gcap_lifetime = gcap_lifetime.unstack().droplevel(axis=1, level=0)
gcap_lifetime = gcap_lifetime.reindex(list(range(first_year_grid,outyear+1)), axis=0).interpolate(limit_direction='both')


   

















