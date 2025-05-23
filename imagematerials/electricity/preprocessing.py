#%%
import pandas as pd
import numpy as np
import os
import math
import scipy
import warnings
from pathlib import Path
import pint
import xarray as xr
from collections import defaultdict

# path_current = Path(__file__).resolve().parent # absolute path of file
# path_base = path_current.parent.parent # base path of the project -> image-materials
# sys.path.append(str(path_base))


from imagematerials.distribution import ALL_DISTRIBUTIONS, NAME_TO_DIST
from imagematerials.read_mym import read_mym_df
from imagematerials.util import dataset_to_array, pandas_to_xarray, convert_life_time_vehicles
# from imagematerials.electricity.modelling_functions import MNLogit, stock_tail #TODO: not working, why??
def stock_tail(stock):
    zero_value = [0 for i in range(0,regions)]
    stock_used = pd.DataFrame(stock).reindex_like(stock)
    stock_used.loc[first_year_grid] = zero_value  # set all regions to 0 in the year of initial operation
    stock_new  = stock_used.reindex(list(range(first_year_grid,outyear+1))).interpolate()
    return stock_new
# Multinomial Logit function, assumes input of an ordered dataframe with rows as years and columns as technologies, values as prices. Logitpar is the calibrated Logit parameter (usually a nagetive number between 0 and 1)
def MNLogit(df, logitpar):
    new_dataframe = pd.DataFrame(index=df.index, columns=df.columns)
    for year in range(df.index[0],df.index[-1]+1): #from first to last year
        yearsum = 0
        for column in df.columns:
            yearsum += math.exp(logitpar * df.loc[year,column]) # calculate the sum of the prices
        for column in df.columns:
            new_dataframe.loc[year,column] = math.exp(logitpar * df.loc[year,column])/yearsum
    return new_dataframe    # the retuned dataframe contains the market shares
# ensure consistent spelling of the time index
def standardize_index_spelling(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize the index name of a DataFrame related to time/year information."""
    index_name = df.index.name
    if index_name is None:
        return df  # Nothing to standardize
    standard_map = {
        "Time": "time",
        "Year": "year",
        "yr": "year",
        "years": "year",
    }
    # Convert exact matches
    if index_name in standard_map:
        df.index.name = standard_map[index_name]
    return df
#-----------------------------------

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
# path_base = Path().resolve() # TODO absolute path of file "preprocessing.py" ? current solution can differ depending on IDE used (?) 
path_current = Path().resolve()
path_base = path_current.parent.parent # base path of the project -> image-materials
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
epg_techs = 34          # number of electricity generation technologies -> 33 technologies + 1 empty row
# epg_techs = 33

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

# Hydro-dam power capacity (also MW) within 5 regions reported by the IHS (international Hydropwer Association)
phs_projections = pd.read_csv(path_external_data_standard / 'PHS.csv', index_col='t')   # pumped hydro storage capacity (MW)


# GENERATION ------------------------------------------------------

# lifetimes of Gcap tech (original data according to van Vuuren 2006, PhD Thesis)
gcap_lifetime = pd.read_csv(path_external_data_scenario / 'LTTechnical_dynamic.csv', index_col=['Year','DIM_1'])        
# gcap_lifetime = standardize_index_spelling(gcap_lifetime) # standardize the index name
# material compositions (generation capacity)
composition_generation = pd.read_csv(path_external_data_scenario / 'composition_generation.csv',index_col=[0,1]).transpose()  # in gram/MW
# composition_generation = composition_generation.drop(columns=[col for col in composition_generation.columns if col[1] == '<EMPTY>'])



# 2. IMAGE/TIMER files ====================================================================================

# STORAGE ------------------------------------------------------

# # The vehicle shares of trucks (heavy) of the SSP2
#     loadfactor_car_data: pd.DataFrame = read_mym_df(
#         image_folder.joinpath("trp_trvl_Load.out")). rename(
#         columns={
#             "DIM_1": "region"})

# read TIMER installed storage capacity (MWh, reservoir)
storage = read_mym_df(path_image_output.joinpath("StorResTot.out"))   #storage capacity in MWh (reservoir, so energy capacity, not power capacity, the latter is used later on in the pumped hydro storage calculations)
    
#storage capacity in MW (power capacity), to compare it to Pumped hydro storage projections (also given in MW, power capacity)
storage_power = read_mym_df(path_image_output / 'StorCapTot.out')  

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

# Lifetimes #

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

# Market Shares #

# Determine MARKET SHARE of the storage capacity using a multi-nomial logit function
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
storage_costs_new.loc[1971:2017,'Deep-cycle Lead-Acid'] = storage_costs_new.loc[2018,'Deep-cycle Lead-Acid']        # restore the exception (set to constant 2018 values)

# use the storage price development in the logit model to get market shares
storage_market_share = MNLogit(storage_costs_new, -0.2)

# fix the market share of storage technologies after 2050
for year in range(2050+1,outyear+1):
    # storage_market_share = storage_market_share.append(pd.Series(storage_market_share.loc[storage_market_share.last_valid_index()], name=year))
    row = pd.DataFrame([storage_market_share.loc[storage_market_share.last_valid_index()]])
    storage_market_share.loc[year] = row.iloc[0]



# Car Batteries ----------

# kilometrage is defined until 2008, fill 2008 values until 2100 
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


#%% GENERATION ----------------------------------------------------------------

region_list = list(kilometrage.columns.values)  # to get region list without running storage - TODO 

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

# Calculate the historic tail to the Gcap (stock) 
gcap_new = pd.DataFrame(index=pd.MultiIndex.from_product([range(first_year_grid,outyear+1), region_list], names=['years', 'regions']), columns=gcap.columns)
for tech in gcap_tech_list:
    gcap_new.loc[idx[:,:],tech] = stock_tail(gcap.loc[idx[:,:],tech].unstack(level=1)).stack()


# Bring dataframes into correct shape for the results_dict
# I need:
#   A. with only Types: stocks Types (GW), lifetimes Types, material intensities Types
#   B. with also SubTypes: stocks Types (GW), lifetimes SubTypes, material intensities SubTypes, market shares SubTypes

# A.
# stocks: (years, regions) index and technologies as columns -> years as index and (technology, region) as columns
gcap_stock = gcap_new.unstack(level='regions')

# lifetimes
df_mean = gcap_lifetime.copy()
df_stdev = df_mean * stdev_mult
df_mean.columns = [(col, 'mean') for col in df_mean.columns] # Rename columns to multi-level tuples
df_stdev.columns = [(col, 'stdev') for col in df_stdev.columns]
gcap_lifetime_distr = pd.concat([df_mean, df_stdev], axis=1) # Concatenate along columns

# MIs: (years, material) index and technologies as columns -> years as index and (technology, Material) as columns
gcap_materials_interpol.index.names = ["Year", "Material"]
# gcap_materials_interpol = gcap_materials_interpol.loc[:, ~(gcap_materials_interpol == 0.0).all()] # delete empty columns
gcap_types_materials = gcap_materials_interpol.unstack(level='Material')

###########################################################################################################
#%% Prep_data File
###########################################################################################################


ureg = pint.UnitRegistry(force_ndarray_like=True)
# Define the units for each dimension
unit_mapping = {
    'time': ureg.year,
    'year': ureg.year,
    'Year': ureg.year,
    'kg': ureg.kilogram,
    'yr': ureg.year,
    '%': ureg.percent,
    't': ureg.tonne,
    'MW': ureg.megawatt, #added
    'GW': ureg.gigawatt, #added
}

# Conversion table for all coordinates, to be removed/adapted after input tables are fixed.
conversion_table = {
    "gcap_stock": (["Time"], ["Type", "Region"],),
    "gcap_types_materials": (["Cohort"], ["Type", "material"],)
    # "gcap_materials_interpol": (["Cohort"], ["Type", "SubType", "material"], {"Type": ["Type", "SubType"]})
}


# results_dict = {
#         'total_nr_vehicles_simple': total_nr_vehicles_simple,
#         'material_fractions_simple': material_fractions_simple,
#         'material_fractions_typical': material_fractions_typical,
#         'vehicle_weights_simple': vehicle_weights_simple,
#         'vehicle_weights_typical': vehicle_weights_typical,
#         'lifetimes': lifetimes_vehicles,
#         'battery_weights_typical': battery_weights_typical,
#         'battery_materials': battery_materials,
#         'battery_shares': battery_shares,
#         'weight_boats': weight_boats,
#         'vehicle_shares_typical': vehicle_shares_typical
#     }
results_dict = {
        'gcap_stock': gcap_stock,
        'gcap_types_materials': gcap_types_materials,
        'gcap_lifetime_distr': gcap_lifetime_distr,
}

# df = gcap_materials_interpol.copy()


# Convert the DataFrames to xarray Datasets and apply units
preprocessing_results_xarray = {}


for df_name, df in results_dict.items():
    if df_name in conversion_table:
        data_xar_dataset = pandas_to_xarray(df, unit_mapping)
        data_xarray = dataset_to_array(data_xar_dataset, *conversion_table[df_name])
    else:
        # lifetimes_vehicles does not need to be converted in the same way.
        data_xarray = pandas_to_xarray(df, unit_mapping)
    preprocessing_results_xarray[df_name] = data_xarray





preprocessing_results_xarray["lifetimes"] = convert_life_time_vehicles(preprocessing_results_xarray["gcap_lifetime_distr"])
preprocessing_results_xarray["stocks"] = preprocessing_results_xarray.pop("gcap_stock")
preprocessing_results_xarray["material_intensities"] = preprocessing_results_xarray.pop("gcap_types_materials")
# preprocessing_results_xarray["shares"] = preprocessing_results_xarray.pop("vehicle_shares")



###########################################################################################################
#%% Run Stock Model New
###########################################################################################################
import prism
from imagematerials.model import GenericMainModel, GenericMaterials, GenericStocks, Maintenance, MaterialIntensities
from imagematerials.factory import ModelFactory
from imagematerials.concepts import create_electricity_graph

prep_data = preprocessing_results_xarray.copy()

# # Define the complete timeline, including historic tail
time_start = prep_data["stocks"].coords["Time"].min().values
# time_start = 1960
complete_timeline = prism.Timeline(time_start, 2060, 1)
simulation_timeline = prism.Timeline(1970, 2060, 1)

# Define the coordinates of all dimensions.
Region = list(prep_data["stocks"].coords["Region"].values)
Time = [t for t in complete_timeline]
Cohort = Time
Type = list(prep_data["stocks"].coords["Type"].values)
material = list(prep_data["material_intensities"].coords["material"].values)

# Create
# main_model_normal = GenericMainModel(
#     complete_timeline, Region=Region, Time=Time, Cohort=Cohort, Type=Type, prep_data=prep_data,
#     compute_materials=True, compute_battery_materials=False, compute_maintenance_materials=False, 
#     material=material)

new_prep_data = prep_data.copy()
new_prep_data["knowledge_graph"] = create_electricity_graph()


main_model_factory = ModelFactory(
    new_prep_data, complete_timeline
    ).add(GenericStocks
    ).add(MaterialIntensities
    ).finish()

main_model_factory.simulate(simulation_timeline)



###########################################################################################################
#%% Tests: Check results

list(main_model_factory.default)

main_model_factory.inflow

tv = main_model_factory.inflow

# Extract years
years = np.arange(tv.timeline.start, tv.timeline.end + 1, tv.timeline.stepsize)

# Extract dimension labels and coordinates
dim_labels = [d.label for d in tv.dims]
dim_coords = {d.label: d.coords for d in tv.dims}

# Example: extract regions and types
regions = [str(r) for r in dim_coords["Region"]]
types = [str(t) for t in dim_coords["Type"]]




###########################################################################################################
#%% Visualize Stocks
import matplotlib.pyplot as plt


da_stocks = main_model_factory.stocks.copy()

regions = da_stocks.Region.values[:2]  # First 2 regions
types_top = da_stocks.Type.values[1:15]   # Types 1–10
types_bottom = da_stocks.Type.values[15:30]  # Types 11–20

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 8), sharex=True, sharey=True)

for i, region in enumerate(regions):
    # Top row: Types 1–10
    for t in types_top:
        da_stocks.sel(Type=t, Region=region).plot(ax=axes[0, i], label=t)
    axes[0, i].set_title(f"{region} (Types 1–10)")
    axes[0, i].legend()

    # Bottom row: Types 11–20
    for t in types_bottom:
        da_stocks.sel(Type=t, Region=region).plot(ax=axes[1, i], label=t)
    axes[1, i].set_title(f"{region} (Types 11–20)")
    axes[1, i].set_xlabel("Time")
    axes[1, i].legend(loc ='upper left')

# Label only left side with y-axis label
axes[0, 0].set_ylabel("Value")
axes[1, 0].set_ylabel("Value")

plt.tight_layout()
plt.show()



###########################################################################################################
#%% Visualize Inflow Materials



# da_x = main_model_factory.inflow.to_array()
da_x = main_model_factory.inflow_materials.to_array()
da_x = main_model_factory.inflow_materials.to_array().sum('Type')

regions = da_x.Region.values[:2]  # First 2 regions
# types_top = da_x.material.values[1:6]   # Types 1–10
# types_bottom = da_x.material.values[6:12]  # Types 11–20
types_level1 = [m for m in da_x.material.values if m in ["Steel", "Concrete"]]
types_level2 = [m for m in da_x.material.values if m in ["Aluminium", "Cu"]]
types_level3 = [m for m in da_x.material.values if m not in (types_level1 + types_level2)]

fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 12), sharex=True)

axes[0, 1].sharey(axes[0, 0])
axes[1, 1].sharey(axes[1, 0])
axes[2, 1].sharey(axes[2, 0])

for i, region in enumerate(regions):
    # Top row: 
    for t in types_level1:
        da_x.sel(material=t, Region=region).plot(ax=axes[0, i], label=t)
    axes[0, i].set_title(f"{region}")
    axes[0, i].set_xlabel("Time")
    axes[0, i].legend()

    # Middle row: 
    for t in types_level2:
        da_x.sel(material=t, Region=region).plot(ax=axes[1, i], label=t)
    axes[1, i].set_title(f"{region}")
    axes[1, i].set_xlabel("Time")
    axes[1, i].legend(loc ='upper left')

    # Bottom row: 
    for t in types_level3:
        da_x.sel(material=t, Region=region).plot(ax=axes[2, i], label=t)
    axes[2, i].set_title(f"{region}")
    axes[2, i].set_xlabel("Time")
    axes[2, i].legend(loc ='upper left')

# Label only left side with y-axis label
axes[0, 0].set_ylabel("Value")
axes[1, 0].set_ylabel("Value")
axes[2, 0].set_ylabel("Value")

plt.tight_layout()
plt.show()






# %%
