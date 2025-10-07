#%%
import pandas as pd
import numpy as np
import os
import scipy
import warnings
from pathlib import Path
import pint
import xarray as xr
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from matplotlib.ticker import ScalarFormatter

# path_current = Path(__file__).resolve().parent # absolute path of file
# path_base = path_current.parent.parent # base path of the project -> image-materials
# sys.path.append(str(path_base))

import prism
from imagematerials.distribution import ALL_DISTRIBUTIONS, NAME_TO_DIST
from imagematerials.read_mym import read_mym_df
from imagematerials.util import dataset_to_array, pandas_to_xarray, convert_life_time_vehicles
from imagematerials.model import GenericMainModel, GenericStocks, SharesInflowStocks, Maintenance, GenericMaterials, MaterialIntensities
from imagematerials.factory import ModelFactory, Sector
from imagematerials.concepts import create_electricity_graph, create_region_graph
from imagematerials.electricity.utils import MNLogit, stock_tail, create_prep_data, stock_share_calc


from imagematerials.electricity.constants import (
    YEAR_FIRST,
    YEAR_FIRST_GRID,
    YEAR_SWITCH,
    STANDARD_SCEN_EXTERNAL_DATA,
    SENS_ANALYSIS,
    REGIONS,
    TECH_GEN,
    STD_LIFETIMES_ELECTR,
    MEGA_TO_TERA,
    PKMS_TO_VKMS,
    TONNES_TO_KGS,
    LOAD_FACTOR,
    BEV_CAPACITY_CURRENT,
    PHEV_CAPACITY_CURRENT,
    unit_mapping,
    DICT_GENTECH_TO_CATEGORY,
    DICT_GENTECH_STYLES,
    DICT_STOR_STYLES,
    DICT_GEN_CATEGORY_COLORS,
    DICT_MATERIALS_COLORS,
    DICT_GRID_COLORS,
    DICT_GRID_STYLES_1,
    DICT_GRID_STYLES_2
)

from imagematerials.electricity.electr_external_data import (
    df_iea_cu_aps,
    df_iea_cu_nzs,
    df_iea_co_aps,
    df_iea_mn_aps,
    df_iea_ni_aps
)
SCEN = "SSP2"
# VARIANT = "VLHO"
VARIANT = "M_CP"
# VARIANT = "BL"
# Define paths ----------------------------------------------------------------------
#YOUR_DIR = "C:\\Users\\Admin\\surfdrive\\Projects\\IRP\\GRO23\\Modelling\\2060\\ELMA"   # Change the running directory here
# os.chdir(YOUR_DIR)
scen_folder = SCEN + "_" + VARIANT
# path_base = Path().resolve() # TODO absolute path of file "preprocessing.py" ? current solution can differ depending on IDE used (?) 
path_current = Path().resolve()
path_base = path_current.parent.parent # base path of the project -> image-materials

path_image_output = Path(path_base, "data", "raw", "image", scen_folder, "EnergyServices")
# path_image_output = Path(path_base, "data", "raw", "image", scen_folder)

# TEST---
path_image_output_SSP2_BL = Path(path_base, "data", "raw", "image", "SSP2_BL")
path_image_output_SSP2_450 = Path(path_base, "data", "raw", "image", "SSP2_450")
path_image_output_SSP2_M_CP = Path(path_base, "data", "raw", "image", "SSP2_M_CP", "EnergyServices")
path_image_output_SSP2_VLHO = Path(path_base, "data", "raw", "image", "SSP2_VLHO", "EnergyServices")
#-----------
path_external_data_standard = Path(path_base, "data", "raw", "electricity", "standard_data")
path_external_data_scenario = Path(path_base, "data", "raw", "electricity", scen_folder)
# test if path_external_data_scenario exists and if not set to standard scenario
if not path_external_data_scenario.exists():
    path_external_data_scenario = Path(path_base, "data", "raw", "electricity", STANDARD_SCEN_EXTERNAL_DATA)
print(f"Path to image output: {path_image_output}")

assert path_image_output.is_dir()
assert path_external_data_standard.is_dir()
assert path_external_data_scenario.is_dir()

# create the folder out_test if it does not exist
if not (path_base / 'imagematerials' / 'electricity' / 'out_test').is_dir():
    (path_base / 'imagematerials' / 'electricity' / 'out_test').mkdir(parents=True)


# for dependencies (this is how Sebastiaan had it):
# YEAR_START = 1971 # start year of historic IMAGE data = start of model period (including stock-development from scratch, which needs to be the oldest year of any vehicle, all stock
# calculations are initiated in this year, so this has an effect on runtime)
# YEAR_FIRST_GRID = 1926   # UK Electricity supply act - https://www.bbc.com/news/uk-politics-11619751   
# YEAR_FIRST = 1807  # first_year_vehicle.values.min()
# YEAR_END = 2060    # end year of the calculations
# YEAR_OUT = 2060    # year of output generation = last year of reporting (in the output files) 
# YEAR_LAST = 2060   # last year available in the IMAGE data-files (which are input to ELMA)
# YEAR_SWITCH = 1990 # year after which other batteries than lead-acid are allowed


# from past.builtins import execfile
# execfile('read_mym.py')
idx = pd.IndexSlice             # needed for slicing multi-index


# TODO: decide on variable naming convention
# V1: sector_variableinquestion_additionalinformation (gcap_lifetime_interpolated)
# V2: variableinquestion_sector_additionalinformation (lifetime_gcap_interpolated)


###########################################################################################################
#%% 0) Compare Sebastiaans scenarios to the new 
###########################################################################################################

#%%% Generation

gcap_data_CP = read_mym_df(path_image_output_SSP2_M_CP / 'Gcap.out')
gcap_data_BL = read_mym_df(path_image_output_SSP2_BL / 'Gcap.out')
kilometrage = pd.read_csv(path_external_data_scenario / 'kilometrage.csv', index_col='t')   #annual car mileage in kms/yr, based  mostly  on  Pauliuk  et  al.  (2012a)
composition_generation = pd.read_csv(path_external_data_scenario / 'composition_generation.csv',index_col=[0,1]).transpose()  # in gram/MW

# knowledge_graph_region = create_region_graph()
# knowledge_graph_electr = create_electricity_graph()
# mat_intensities = knowledge_graph.rebroadcast_xarray(mat_intensities, floorspace.coords["Type"].values)

region_list = list(kilometrage.columns.values)   
gcap_tech_list = list(composition_generation.loc[:,idx[2020,:]].droplevel(axis=1, level=0).columns)    #list of names of the generation technologies (workaround to retain original order)

gcap_CP = gcap_data_CP.loc[~gcap_data_CP['DIM_1'].isin([27,28])]    # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
gcap_CP = pd.pivot_table(gcap_CP[gcap_CP['time'].isin(list(range(YEAR_START,YEAR_END+1)))], index=['time','DIM_1'], values=list(range(1,TECH_GEN+1)))  #gcap as multi-index (index = years & regions (26); columns = technologies (34));  the last column in gcap_data (= totals) is now removed
# renaming multi-index dataframe: generation capacity, based on the regions in grid_length_Hv & technologies as given
gcap_CP.index = pd.MultiIndex.from_product([list(range(YEAR_START,YEAR_END+1)), region_list], names=['years', 'regions'])
gcap_CP.columns = gcap_tech_list

gcap_BL = gcap_data_BL.loc[~gcap_data_BL['DIM_1'].isin([27,28])]    # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
gcap_BL = pd.pivot_table(gcap_BL[gcap_BL['time'].isin(list(range(YEAR_START,YEAR_END+1)))], index=['time','DIM_1'], values=list(range(1,TECH_GEN-3)))  #gcap as multi-index (index = years & regions (26); columns = technologies (34));  the last column in gcap_data (= totals) is now removed
# renaming multi-index dataframe: generation capacity, based on the regions in grid_length_Hv & technologies as given
gcap_BL.index = pd.MultiIndex.from_product([list(range(YEAR_START,YEAR_END+1)), region_list], names=['years', 'regions'])
gcap_BL.columns = gcap_tech_list

# 1. Technology comparison (total over years & regions)
tech_CP = gcap_CP.sum()
tech_BL = gcap_BL.sum()

plt.figure(figsize=(12,6))
plt.bar(tech_CP.index, tech_CP.values, alpha=0.6, label="CP")
plt.bar(tech_BL.index, tech_BL.values, alpha=0.6, label="BL")
plt.xticks(rotation=90)
plt.ylabel("Total Capacity")
plt.title("Total Capacity by Technology")
plt.legend()
plt.tight_layout()
plt.show()


# 2. Regional comparison (total over years & technologies)
region_CP = gcap_CP.groupby(level=1).sum().sum(axis=1)
region_BL = gcap_BL.groupby(level=1).sum().sum(axis=1)

plt.figure(figsize=(12,6))
x = range(len(region_CP))
plt.bar(x, region_CP.values, width=0.4, label="CP", align="center")
plt.bar([i+0.4 for i in x], region_BL.values, width=0.4, label="BL", align="center")
plt.xticks([i+0.2 for i in x], region_CP.index, rotation=90)
plt.ylabel("Total Capacity")
plt.title("Total Capacity by Region")
plt.legend()
plt.tight_layout()
plt.show()


# 3. Global time series comparison (sum over all regions & techs)
ts_CP = gcap_CP.groupby(level=0).sum().sum(axis=1)
ts_BL = gcap_BL.groupby(level=0).sum().sum(axis=1)

plt.figure(figsize=(12,6))
plt.plot(ts_CP.index, ts_CP.values, label="CP", lw=2)
plt.plot(ts_BL.index, ts_BL.values, label="BL", lw=2, linestyle="--")
plt.ylabel("Total Capacity")
plt.xlabel("Year")
plt.title("Global Capacity over Time: CP vs BL")
plt.legend()
plt.tight_layout()
plt.show()



##############################################################
#%%% Storage

path_test_plots = Path(path_base, "imagematerials", "electricity", "out_test")

storage_VLHO = read_mym_df(path_image_output_SSP2_VLHO.joinpath("StorResTot.out"))
storage_CP = read_mym_df(path_image_output_SSP2_M_CP.joinpath("StorResTot.out"))
storage_450 = read_mym_df(path_image_output_SSP2_450.joinpath("StorResTot.out"))
storage_BL = read_mym_df(path_image_output_SSP2_BL.joinpath("StorResTot.out"))
kilometrage = pd.read_csv(path_external_data_scenario / 'kilometrage.csv', index_col='t')   #annual car mileage in kms/yr, based  mostly  on  Pauliuk  et  al.  (2012a)


storage_CP = storage_CP.iloc[:, :26]    # drop global total column and empty (27) column
storage_VLHO = storage_VLHO.iloc[:, :26]
storage_BL = storage_BL.iloc[:, :26]    # drop global total column and empty (27) column
storage_450 = storage_450.iloc[:, :26]

region_list = list(kilometrage.columns.values)   
storage_CP.columns = region_list
storage_VLHO.columns = region_list
storage_BL.columns = region_list
storage_450.columns = region_list


# 2. Global time series (sum over all regions)
ts_CP = storage_CP.sum(axis=1)
ts_VLHO = storage_VLHO.sum(axis=1)
ts_BL = storage_BL.sum(axis=1)
ts_450 = storage_450.sum(axis=1)

fig, ax1 = plt.subplots(figsize=(12, 6))
# Primary axis (absolute values)
ax1.plot(ts_CP.index, ts_CP.values, label="SSP2_M_CP", lw=2, color="#800f2f")
ax1.plot(ts_VLHO.index, ts_VLHO.values, label="SSP2_VLHO", lw=2, color="#023e8a")
ax1.plot(ts_BL.index, ts_BL.values, label="SSP2_BL", lw=2, color="#a4133c", linestyle="--")
ax1.plot(ts_450.index, ts_450.values, label="SSP2_450", lw=2, color="#0077b6", linestyle="--")
ax1.set_ylabel("Storage (MWh)")
ax1.set_xlabel("Year")
ax1.set_title("Global Storage over Time (TIMER StorResTot.out in MWh)")
# Secondary axis (ratios)
ax2 = ax1.twinx()
ratio_CP_BL = ts_CP / ts_BL
ratio_VLHO_450 = ts_VLHO / ts_450
ax2.plot(ratio_CP_BL.index, ratio_CP_BL.values, label="SSP2_M_CP / SSP2_BL", color="#ffb3c1", lw=2, linestyle=":")
ax2.plot(ratio_VLHO_450.index, ratio_VLHO_450.values, label="SSP2_VLHO / SSP2_450", color="#caf0f8", lw=2, linestyle=":")
ax2.set_ylabel("Ratio")
# Combine legends
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc="best")

fig.savefig(path_test_plots / f"TIMER_storage_scenario_comparison.png", dpi=300)
plt.tight_layout()
plt.show()




# 3. Selected regions time series (e.g. US, China, India, Europe)
regions = ["US", "China", "India", "W.Europe"]

plt.figure(figsize=(12,6))
for r in regions:
    plt.plot(storage_CP.index, storage_CP[r], label=f"CP {r}", lw=2)
    plt.plot(storage_BL.index, storage_BL[r], label=f"BL {r}", lw=2, linestyle="--")

plt.ylabel("Storage")
plt.xlabel("Year")
plt.title("Regional Storage over Time (Selected)")
plt.legend()
plt.tight_layout()
plt.show()


###########################################################################################################
###########################################################################################################
#%% 1) Generation 
###########################################################################################################
###########################################################################################################



def get_preprocessing_data_gen(base_dir: str, SCEN, VARIANT, YEAR_START, YEAR_END, YEAR_OUT): #, climate_policy_config: dict, circular_economy_config: dict

    scen_folder = SCEN + "_" + VARIANT
    path_image_output = Path(path_base, "data", "raw", "image", scen_folder, "EnergyServices")
    path_external_data_standard = Path(path_base, "data", "raw", "electricity", "standard_data")
    path_external_data_scenario = Path(path_base, "data", "raw", "electricity", scen_folder)
    # test if path_external_data_scenario exists and if not set to standard scenario
    if not path_external_data_scenario.exists():
        path_external_data_scenario = Path(path_base, "data", "raw", "electricity", STANDARD_SCEN_EXTERNAL_DATA)
    print(f"Path to image output: {path_image_output}")

    assert path_image_output.is_dir()
    assert path_external_data_standard.is_dir()
    assert path_external_data_scenario.is_dir()

    years = YEAR_END - YEAR_START  + 1

    idx = pd.IndexSlice   

    ###########################################################################################################
    # Read in files #

    # 1. External Data --------------------------------------------- 

    # lifetimes of Gcap tech (original data according to van Vuuren 2006, PhD Thesis)
    gcap_lifetime = pd.read_csv(path_external_data_scenario / 'LTTechnical_dynamic.csv', index_col=['Year','DIM_1'])        
    # material compositions (generation capacity)
    composition_generation = pd.read_csv(path_external_data_scenario / 'composition_generation.csv',index_col=[0,1]).transpose()  # in gram/MW
    kilometrage = pd.read_csv(path_external_data_scenario / 'kilometrage.csv', index_col='t')  # to get region list without running storage - TODO: get regions from different source

    # 2. IMAGE/TIMER files -----------------------------------------

    # Generation capacity (stock & inflow/new) in MW peak capacity, FILES from TIMER
    gcap_data = read_mym_df(path_image_output / 'Gcap.out')

    ###########################################################################################################
    # Prepare model specific variables #
    
    region_list = list(kilometrage.columns.values)   

    gcap_tech_list = list(composition_generation.loc[:,idx[2020,:]].droplevel(axis=1, level=0).columns)    #list of names of the generation technologies (workaround to retain original order)
    gcap_material_list = list(composition_generation.index.values)  #list of materials the generation technologies

    gcap_data = gcap_data.loc[~gcap_data['DIM_1'].isin([27,28])]    # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
    gcap = pd.pivot_table(gcap_data[gcap_data['time'].isin(list(range(YEAR_START,YEAR_END+1)))], index=['time','DIM_1'], values=list(range(1,TECH_GEN+1)))  #gcap as multi-index (index = years & regions (26); columns = technologies (34));  the last column in gcap_data (= totals) is now removed

    # renaming multi-index dataframe: generation capacity, based on the regions in grid_length_Hv & technologies as given
    gcap.index = pd.MultiIndex.from_product([list(range(YEAR_START,YEAR_END+1)), region_list], names=['years', 'regions'])
    gcap.columns = gcap_tech_list

    # Interpolate material intensities (dynamic content for gcap & storage technologies between 1926 to 2100, based on data files)
    index = pd.MultiIndex.from_product([list(range(YEAR_FIRST_GRID, YEAR_OUT+1)), list(composition_generation.index)])
    gcap_materials_interpol = pd.DataFrame(index=index, columns=composition_generation.columns.levels[1])

    # material intensities for gcap
    for cat in list(composition_generation.columns.levels[1]):
        gcap_materials_1st   = composition_generation.loc[:,idx[composition_generation.columns[0][0],cat]]
        gcap_materials_interpol.loc[idx[YEAR_FIRST_GRID ,:],cat] = gcap_materials_1st.to_numpy()                # set the first year (1926) values to the first available values in the dataset (for the year 2000) 
        gcap_materials_interpol.loc[idx[composition_generation.columns.levels[0].min(),:],cat] = composition_generation.loc[:, idx[composition_generation.columns.levels[0].min(),cat]].to_numpy()                # set the middle year (2000) values to the first available values in the dataset (for the year 2000) 
        gcap_materials_interpol.loc[idx[composition_generation.columns.levels[0].max(),:],cat] = composition_generation.loc[:, idx[composition_generation.columns.levels[0].max(),cat]].to_numpy()                # set the last year (2100) values to the last available values in the dataset (for the year 2050) 
        gcap_materials_interpol.loc[idx[:,:],cat] = gcap_materials_interpol.loc[idx[:,:],cat].unstack().astype('float64').interpolate().stack()

    # interpolate Gcap (technical) lifetime data
    gcap_lifetime.index = gcap_lifetime.index.set_levels(gcap_tech_list, level=1)
    gcap_lifetime = gcap_lifetime.unstack().droplevel(axis=1, level=0)
    gcap_lifetime = gcap_lifetime.reindex(list(range(YEAR_FIRST_GRID,YEAR_OUT+1)), axis=0).interpolate(limit_direction='both')

    # Calculate the historic tail to the Gcap (stock) 
    gcap_new = pd.DataFrame(index=pd.MultiIndex.from_product([range(YEAR_FIRST_GRID,YEAR_OUT+1), region_list], names=['years', 'regions']), columns=gcap.columns)
    for tech in gcap_tech_list:
        gcap_new.loc[idx[:,:],tech] = stock_tail(gcap.loc[idx[:,:],tech].unstack(level=1), YEAR_OUT).stack()


    # Bring dataframes into correct shape for the results_dict
    # I need:
    #   A. with only Types: stocks Types (GW), lifetimes Types, material intensities Types
    #   B. with also SubTypes: stocks Types (GW), lifetimes SubTypes, material intensities SubTypes, market shares SubTypes

    # A.
    # stocks: (years, regions) index and technologies as columns -> years as index and (technology, region) as columns
    gcap_stock = gcap_new.unstack(level='regions')

    # lifetimes
    df_mean = gcap_lifetime.copy()
    df_stdev = df_mean * STD_LIFETIMES_ELECTR
    df_mean.columns = [(col, 'mean') for col in df_mean.columns] # Rename columns to multi-level tuples
    df_stdev.columns = [(col, 'stdev') for col in df_stdev.columns]
    gcap_lifetime_distr = pd.concat([df_mean, df_stdev], axis=1) # Concatenate along columns

    # MIs: (years, material) index and technologies as columns -> years as index and (technology, Material) as columns
    gcap_materials_interpol.index.names = ["Year", "Material"]
    # gcap_materials_interpol = gcap_materials_interpol.loc[:, ~(gcap_materials_interpol == 0.0).all()] # delete empty columns
    gcap_types_materials = gcap_materials_interpol.unstack(level='Material')


    ###########################################################################################################
    # Prep_data File #
    
    # Conversion table for all coordinates, to be removed/adapted after input tables are fixed.
    conversion_table = {
        "gcap_stock": (["Time"], ["Type", "Region"],),
        "gcap_types_materials": (["Cohort"], ["Type", "material"],)
        # "gcap_materials_interpol": (["Cohort"], ["Type", "SubType", "material"], {"Type": ["Type", "SubType"]})
    }

    results_dict = {
            'gcap_stock': gcap_stock,
            'gcap_types_materials': gcap_types_materials,
            'gcap_lifetime_distr': gcap_lifetime_distr,
    }

    prep_data = create_prep_data(results_dict, conversion_table, unit_mapping)

    return prep_data




# sector_electr_gen = get_preprocessing_data("electr_gen", Path("..", "data", "raw"), cache="prep_electr_gen.nc")

# vhc_sector = get_preprocessing_data("vehicles", Path("..", "data", "raw"), cache="prep_vema.nc")



#----------------------------------------------------------------------------------------------------------
###########################################################################################################
#%%% 1.1) Read in files
###########################################################################################################
#----------------------------------------------------------------------------------------------------------

YEAR_START = 1971   # start year of the simulation period
YEAR_END = 2100     # end year of the calculations
YEAR_OUT = 2100     # year of output generation = last year of reporting

# 1. External Data ======================================================================================== 

# lifetimes of Gcap tech (original data according to van Vuuren 2006, PhD Thesis)
gcap_lifetime = pd.read_csv(path_external_data_scenario / 'LTTechnical_dynamic.csv', index_col=['Year','DIM_1'])        
# gcap_lifetime = standardize_index_spelling(gcap_lifetime) # standardize the index name
# material compositions (generation capacity)
composition_generation = pd.read_csv(path_external_data_scenario / 'composition_generation.csv',index_col=[0,1]).transpose()  # in gram/MW
# composition_generation = composition_generation.drop(columns=[col for col in composition_generation.columns if col[1] == '<EMPTY>'])
kilometrage = pd.read_csv(path_external_data_scenario / 'kilometrage.csv', index_col='t')  # to get region list without running storage - TODO: get regions from different source


# 2. IMAGE/TIMER files ====================================================================================

# Generation capacity (stock & inflow/new) in MW peak capacity, FILES from TIMER
gcap_data = read_mym_df(path_image_output / 'Gcap.out')


#----------------------------------------------------------------------------------------------------------
###########################################################################################################
#%%% 1.2) Prepare model specific variables
###########################################################################################################
#----------------------------------------------------------------------------------------------------------

region_list = list(kilometrage.columns.values)   

gcap_tech_list = list(composition_generation.loc[:,idx[2020,:]].droplevel(axis=1, level=0).columns)    #list of names of the generation technologies (workaround to retain original order)
gcap_material_list = list(composition_generation.index.values)  #list of materials the generation technologies

gcap_data = gcap_data.loc[~gcap_data['DIM_1'].isin([27,28])]    # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
gcap = pd.pivot_table(gcap_data[gcap_data['time'].isin(list(range(YEAR_START,YEAR_END+1)))], index=['time','DIM_1'], values=list(range(1,TECH_GEN+1)))  #gcap as multi-index (index = years & regions (26); columns = technologies (34));  the last column in gcap_data (= totals) is now removed

# renaming multi-index dataframe: generation capacity, based on the regions in grid_length_Hv & technologies as given
gcap.index = pd.MultiIndex.from_product([list(range(YEAR_START,YEAR_END+1)), region_list], names=['years', 'regions'])
gcap.columns = gcap_tech_list

# Interpolate material intensities (dynamic content for gcap & storage technologies between 1926 to 2100, based on data files)
index = pd.MultiIndex.from_product([list(range(YEAR_FIRST_GRID, YEAR_OUT+1)), list(composition_generation.index)])
gcap_materials_interpol = pd.DataFrame(index=index, columns=composition_generation.columns.levels[1])

# material intensities for gcap
for cat in list(composition_generation.columns.levels[1]):
   gcap_materials_1st   = composition_generation.loc[:,idx[composition_generation.columns[0][0],cat]]
   gcap_materials_interpol.loc[idx[YEAR_FIRST_GRID ,:],cat] = gcap_materials_1st.to_numpy()                # set the first year (1926) values to the first available values in the dataset (for the year 2000) 
   gcap_materials_interpol.loc[idx[composition_generation.columns.levels[0].min(),:],cat] = composition_generation.loc[:, idx[composition_generation.columns.levels[0].min(),cat]].to_numpy()                # set the middle year (2000) values to the first available values in the dataset (for the year 2000) 
   gcap_materials_interpol.loc[idx[composition_generation.columns.levels[0].max(),:],cat] = composition_generation.loc[:, idx[composition_generation.columns.levels[0].max(),cat]].to_numpy()                # set the last year (2100) values to the last available values in the dataset (for the year 2050) 
   gcap_materials_interpol.loc[idx[:,:],cat] = gcap_materials_interpol.loc[idx[:,:],cat].unstack().astype('float64').interpolate().stack()

# interpolate Gcap (technical) lifetime data
gcap_lifetime.index = gcap_lifetime.index.set_levels(gcap_tech_list, level=1)
gcap_lifetime = gcap_lifetime.unstack().droplevel(axis=1, level=0)
gcap_lifetime = gcap_lifetime.reindex(list(range(YEAR_FIRST_GRID,YEAR_OUT+1)), axis=0).interpolate(limit_direction='both')

# Calculate the historic tail to the Gcap (stock) 
gcap_new = pd.DataFrame(index=pd.MultiIndex.from_product([range(YEAR_FIRST_GRID,YEAR_OUT+1), region_list], names=['years', 'regions']), columns=gcap.columns)
for tech in gcap_tech_list:
    gcap_new.loc[idx[:,:],tech] = stock_tail(gcap.loc[idx[:,:],tech].unstack(level=1), YEAR_OUT).stack()


# Bring dataframes into correct shape for the results_dict
# I need:
#   A. with only Types: stocks Types (GW), lifetimes Types, material intensities Types
#   B. with also SubTypes: stocks Types (GW), lifetimes SubTypes, material intensities SubTypes, market shares SubTypes

# A.
# stocks: (years, regions) index and technologies as columns -> years as index and (technology, region) as columns
gcap_stock = gcap_new.unstack(level='regions')

# lifetimes
df_mean = gcap_lifetime.copy()
df_stdev = df_mean * STD_LIFETIMES_ELECTR
df_mean.columns = [(col, 'mean') for col in df_mean.columns] # Rename columns to multi-level tuples
df_stdev.columns = [(col, 'stdev') for col in df_stdev.columns]
gcap_lifetime_distr = pd.concat([df_mean, df_stdev], axis=1) # Concatenate along columns

# MIs: (years, material) index and technologies as columns -> years as index and (technology, Material) as columns
gcap_materials_interpol.index.names = ["Year", "Material"]
# gcap_materials_interpol = gcap_materials_interpol.loc[:, ~(gcap_materials_interpol == 0.0).all()] # delete empty columns
gcap_types_materials = gcap_materials_interpol.unstack(level='Material')


#----------------------------------------------------------------------------------------------------------
###########################################################################################################
#%%% 1.3) Prep_data File
###########################################################################################################
#----------------------------------------------------------------------------------------------------------


# Conversion table for all coordinates, to be removed/adapted after input tables are fixed.
conversion_table = {
    "gcap_stock": (["Time"], ["Type", "Region"],),
    "gcap_types_materials": (["Cohort"], ["Type", "material"],)
    # "gcap_materials_interpol": (["Cohort"], ["Type", "SubType", "material"], {"Type": ["Type", "SubType"]})
}

results_dict = {
        'gcap_stock': gcap_stock,
        'gcap_types_materials': gcap_types_materials,
        'gcap_lifetime_distr': gcap_lifetime_distr,
}


prep_data = create_prep_data(results_dict, conversion_table, unit_mapping)
prep_data["stocks"] = prism.Q_(prep_data["stocks"], "MW")
prep_data["material_intensities"] = prism.Q_(prep_data["material_intensities"], "g/MW")
prep_data["set_unit_flexible"] = prism.U_(prep_data["stocks"]) # prism.U_ gives the unit back
# set_unit_flexible is needed by the model to deal with the fact the in the beginning of the model it doesn't know th data yet and needs to work with a placeholder/flexible unit (see model.py) 


#----------------------------------------------------------------------------------------------------------
###########################################################################################################
#%%% 1.4) Run Stock Model New
###########################################################################################################
# TODO: move this to electricity.py
#----------------------------------------------------------------------------------------------------------

prep_data = get_preprocessing_data_gen(path_base, scen_folder, YEAR_START, YEAR_END, YEAR_OUT)

# # Define the complete timeline, including historic tail
time_start = prep_data["stocks"].coords["Time"].min().values
complete_timeline = prism.Timeline(time_start, YEAR_END, 1)
simulation_timeline = prism.Timeline(YEAR_START, YEAR_END, 1) #1970

sec_electr_gen = Sector("electr_gen", prep_data)

main_model_factory = ModelFactory(
    sec_electr_gen, complete_timeline
    ).add(GenericStocks
    ).add(MaterialIntensities
    ).finish()

main_model_factory.simulate(simulation_timeline)




#Tests: Check results
# list(main_model_factory.elctr_gen)

path_test_plots = Path(path_base, "imagematerials", "electricity", "out_test", scen_folder, "Figures")

###########################################################################################################
#%%% Visualize STOCKS Gen. Cap.
###########################################################################################################


#================================================================================
#%%%% Per TECH - per Region - main_model_factory

regions = ['Brazil', 'C.Europe', 'China'] 

da_stocks = main_model_factory.stocks

data_all = da_stocks
data_all = data_all.sel(Type=data_all.Type != '<EMPTY>')

data_all = data_all.sel(Time=slice(1971, None), Region=regions)

threshold = 100_000
techs_upper = [coord_name.item() for coord_name in data_all.coords['Type']  
               if data_all.sel(Type = coord_name).values.max() > threshold]
techs_lower = [coord_name.item() for coord_name in data_all.coords['Type']  
               if data_all.sel(Type = coord_name).values.max() <= threshold]


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 8))  # Now 3 columns for 3 regions
linewidth = 2
s_legend = 12
s_label = 14

for i, region in enumerate(regions):  # regions now has length 3
    col = i  # Column index: 0, 1, 2

    # Top row: Types 1–15
    for t in techs_upper:
        data_plot = data_all.sel(Type=t, Region=region)
        color, ls = DICT_GENTECH_STYLES.get(t, ('black', '-'))
        axes[0, col].plot(data_plot.Time, data_plot.values, label=t, color=color, linestyle=ls, linewidth=linewidth)
    axes[0, col].set_title(f"{region}", fontsize=15)
    axes[0, col].grid(alpha=0.3, linestyle='--')
    axes[0, col].tick_params(axis='both', which='major', labelsize=s_legend)
    axes[0, 0].set_ylabel("Peak Capacity (MW)", fontsize=s_label)

    # Bottom row: Types 16–30
    for t in techs_lower:
        data_plot = data_all.sel(Type=t, Region=region)
        color, ls = DICT_GENTECH_STYLES.get(t, ('black', '-'))
        axes[1, col].plot(data_plot.Time, data_plot.values, label=t, color=color, linestyle=ls, linewidth=linewidth)
    axes[1, col].grid(alpha=0.3, linestyle='--')
    axes[1, col].tick_params(axis='both', which='major', labelsize=s_legend)
    axes[1, 0].set_ylabel("Peak Capacity (MW)", fontsize=s_label)
    axes[1, col].set_xlabel("Time", fontsize=s_label)

# Y-axis number formatting and hiding right y-axis ticks
# for ax in axes.flat:
#     ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
# for row in range(2):
#     for col in [1, 2]:  # Hide y-tick labels for middle and right columns
#         axes[row, col].tick_params(labelleft=False)

axes[0, 2].legend(fontsize=s_legend, ncol=2, loc='upper center', bbox_to_anchor=(-1.7, -1.41))
axes[1, 2].legend(fontsize=s_legend, ncol=3, loc='upper center', bbox_to_anchor=(-0.2, -0.21))

plt.suptitle(f"{scen_folder}: Generation - Stocks: Peak Capacity (MW)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to make room for the suptitle
region_str = "_".join(regions)
# fig.savefig(path_test_plots / f"Gen_stocks_{region_str}.png", dpi=300, bbox_inches='tight')
# fig.savefig(path_test_plots / f"Gen_stocks_{region_str}_1971.png", dpi=300, bbox_inches='tight')
plt.show()


#================================================================================
#%%%% Sum & Per TECH - World

# da_x = main_model_factory.inflow.to_array().sum('Type')
da_stocks = main_model_factory.stocks#.to_array()

data_all = da_stocks.sel(Type=da_stocks.Type != '<EMPTY>')
data_all_1 = data_all.sum('Region')
data_all_2 = data_all.sum('Type').sum('Region')

data_all_1 = data_all_1.sel(Time=slice(1971, None))
data_all_2 = data_all_2.sel(Time=slice(1971, None))

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 11))
linewidth = 2
s_legend = 12
s_label = 14

# Second subplot: data_all_2 (summed over Region and Type, likely only time and material left)
data_all_2.plot(ax=axes[0], color='black', linewidth=linewidth)
axes[0].grid(alpha=0.3, linestyle='--')
axes[0].tick_params(axis='both', which='major', labelsize=s_legend)
axes[0].set_xlabel(" ", fontsize=s_label)
axes[0].set_ylabel("Peak Capacity (MW)", fontsize=s_label)
axes[0].set_title('Sum over Type and Region')

# First subplot: data_all_1 (summed over Region, still over Type and time likely)
for t in data_all_1.Type.values:
    data_plot = data_all_1.sel(Type=t)
    color, ls = DICT_GENTECH_STYLES.get(t, ('black', '-'))
    axes[1].plot(data_plot.Time, data_plot.values, label=t, color=color, linestyle=ls, linewidth=linewidth)
    axes[1].grid(alpha=0.3, linestyle='--')
    axes[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
    axes[1].tick_params(axis='both', which='major', labelsize=s_legend)
    axes[1].set_xlabel("Time", fontsize=s_label)
    axes[1].set_ylabel("Peak Capacity (MW)", fontsize=s_label)
    axes[1].legend(fontsize='small', ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.2))
    axes[1].set_title('Sum over Region')

plt.suptitle(f"{scen_folder}: Generation - Stocks: Peak Capacity (MW)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to make room for the suptitle
# fig.savefig(path_test_plots / f"Gen_inflow_world.png", dpi=300, bbox_inches='tight')
# fig.savefig(path_test_plots / f"Gen_stocks_world_1971.png", dpi=300, bbox_inches='tight')
plt.show()


#================================================================================
#%%%% TIMER Gen. Cap. stocks

# 2 COUNTRIES
countries = ['Brazil','C.Europe']
threshold = 100_000
techs_upper = [col for col in gcap.columns if gcap[col].max() > threshold] #gcap.columns[15:30]
techs_lower = [col for col in gcap.columns if gcap[col].max() <= threshold] #gcap.columns[:15]

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 8), sharex=True)
axes[0, 1].sharey(axes[0, 0])
axes[1, 1].sharey(axes[1, 0])

for i, country in enumerate(countries):
    df = gcap[gcap.index.get_level_values(1) == country].copy()
    df.index = df.index.droplevel(1)

    # Top row: techs 15–30
    for tech in techs_upper:
        color, ls = DICT_GENTECH_STYLES[tech]
        axes[0, i].plot(df.index, df[tech], label=tech, color=color, linestyle=ls)
    # axes[0, i].set_title(f"{country} (Technologies 16–30)")
    axes[0, i].grid(alpha=0.3, linestyle='--')

    # Bottom row: techs 0–14
    for tech in techs_lower:
        color, ls = DICT_GENTECH_STYLES[tech]
        axes[1, i].plot(df.index, df[tech], label=tech, color=color, linestyle=ls)
    axes[1, i].set_xlabel('Year')
    axes[1, i].grid(alpha=0.3, linestyle='--')
    

# Add Y-axis labels only on left side
axes[0, 0].set_ylabel('Value')
axes[1, 0].set_ylabel('Value')
for ax in axes.flat: # change y axis ticks from 100000 to 100,000
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
axes[0, 0].legend(fontsize='small', ncol=2, loc='upper left')
axes[1, 0].legend(fontsize='small', ncol=2, loc='upper left')

plt.suptitle(f"Generation - Stocks in {countries[0]}_{countries[1]} (TIMER)", fontsize=16)
plt.tight_layout()
# fig.savefig(path_test_plots / f"TIMER_Gen_stocks_{countries[0]}-{countries[1]}.png", dpi=300)
plt.show()




# 1 COUNTRY
country = 'C.Europe'
df = gcap[gcap.index.get_level_values(1) == country]
df.index = df.index.droplevel(1)

# Split columns into two groups
techs_upper = df.columns[15:30]
techs_lower = df.columns[:15]

# Create subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

for tech in techs_upper:
    color, ls = DICT_GENTECH_STYLES[tech]
    df[tech].plot(ax=ax1, label=tech, color=color, linestyle=ls)
ax1.set_ylabel('Value')
ax1.legend(ncol=2, fontsize='small', loc='upper left')

# Lower panel: tech 16–30
for tech in techs_lower:
    color, ls = DICT_GENTECH_STYLES[tech]
    df[tech].plot(ax=ax2, label=tech, color=color, linestyle=ls)
ax2.set_xlabel('Year')
ax2.set_ylabel('Value')
ax2.legend(ncol=2, fontsize='small', loc='upper left')

plt.tight_layout()
plt.suptitle(f"Generation - Stocks in {country} (TIMER)", fontsize=16)
# fig.savefig(path_test_plots / f"TIMER_Gen_stocks_{country}.png", dpi=300)
plt.show()




###########################################################################################################
#%%% Visualize STOCKS Materials
###########################################################################################################

#================================================================================
#%%%% SUM over TECHs - per region

# Sum over technologies dimension

da_stocks_mat = main_model_factory.stock_by_cohort_materials.copy() #stock_by_cohort_materials

data_all = da_stocks_mat
data_all = data_all.sel(Type=data_all.Type != '<EMPTY>').sum('Type')
# data_all = data_all.sel(Type=data_all.Type != '<EMPTY>').sum('Region')

# Pick desired regions by name
regions = ["Brazil", "C.Europe", "China"] 

types_level1 = [m for m in data_all.material.values if m in ["Steel", "Concrete"]]
types_level2 = [m for m in data_all.material.values if m in ["Aluminium", "Cu"]]
types_level3 = [m for m in data_all.material.values if m not in (types_level1 + types_level2)]

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 12), sharex=True)
linewidth = 2
s_legend = 12
s_label = 14

data_plot = data_all.sel(Time=slice(1971, None)).pint.to("t") # convert grams to tonnes

for i, region in enumerate(regions):
    # Top row: Level 1 materials
    for mat in types_level1:
        if (region in data_plot.Region.values) and (mat in data_plot.material.values):
            data_plot.sel(material=mat, Region=region).plot(ax=axes[0, i], label=mat, color=DICT_MATERIALS_COLORS[mat], linewidth=linewidth)
    axes[0, i].set_title(f"{region}")
    axes[0, i].set_xlabel(" ")
    axes[0, i].set_ylabel(" ")
    axes[0, i].grid(alpha=0.3, linestyle='--')
    axes[0, i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
    axes[0, i].tick_params(axis='both', which='major', labelsize=s_legend) # set font size of axis ticks
    axes[0, 2].legend(loc='upper left', fontsize=s_legend)

    # Middle row: Level 2 materials
    for mat in types_level2:
        if (region in data_plot.Region.values) and (mat in data_plot.material.values):
            data_plot.sel(material=mat, Region=region).plot(ax=axes[1, i], label=mat, color=DICT_MATERIALS_COLORS[mat], linewidth=linewidth)
    axes[1, i].set_title(f" ")
    axes[1, i].set_xlabel(" ")
    axes[1, i].set_ylabel(" ")
    axes[1, i].grid(alpha=0.3, linestyle='--')
    axes[1, i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
    axes[1, i].tick_params(axis='both', which='major', labelsize=s_legend) # set font size of axis ticks
    axes[1, 2].legend(loc='upper left', fontsize=s_legend)

    # Bottom row: Level 3 materials
    for mat in types_level3:
        if (region in data_plot.Region.values) and (mat in data_plot.material.values):
            data_plot.sel(material=mat, Region=region).plot(ax=axes[2, i], label=mat, color=DICT_MATERIALS_COLORS[mat], linewidth=linewidth)
    axes[2, i].set_title(f" ")
    axes[2, i].set_xlabel("Time", fontsize=s_label)
    axes[2, i].set_ylabel(" ")
    axes[2, i].grid(alpha=0.3, linestyle='--')
    axes[2, i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
    axes[2, i].tick_params(axis='both', which='major', labelsize=s_legend) # set font size of axis ticks
    axes[2, 2].legend(loc='upper left', fontsize=s_legend)
    

axes[0, 0].set_ylabel("Material stocks (t)", fontsize=s_label)
axes[1, 0].set_ylabel("Material stocks (t)", fontsize=s_label)
axes[2, 0].set_ylabel("Material stocks (t)", fontsize=s_label)

plt.suptitle(f"{scen_folder}: Generation - Stocks Materials", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to make room for the suptitle
region_str = "_".join(regions)
# fig.savefig(path_test_plots / f"Gen_stocks-materials_{region_str}_1971.png", dpi=300)
# fig.savefig(path_test_plots / f"Gen_stocks-materials_{region_str}_1971.svg", dpi=300)
plt.show()


#================================================================================
#%%%% SUM over TECHs - world

da_stocks_mat = main_model_factory.stock_by_cohort_materials.copy() #stock_by_cohort_materials
data_all = da_stocks_mat
data_all = data_all.sel(Type=data_all.Type != '<EMPTY>').sum('Type').sum('Region')
# data_plot = data_all.pint.to("t") # convert grams to tonnes
data_plot = data_all.sel(Time=slice(1971, None)).pint.to("t") # only from 1971 onwards, convert grams to tonnes

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 10))
linewidth = 2
s_legend = 12
s_label = 14

# Top row: Level 1 materials
for mat in types_level1:
    if mat in data_plot.material.values:
        data_plot.sel(material=mat).plot(ax=axes[0], label=mat, color=DICT_MATERIALS_COLORS[mat], linewidth=linewidth)
axes[0].set_title(f" ")
axes[0].set_xlabel(" ")
axes[0].set_ylabel("Material stocks (t)", fontsize=s_label)
axes[0].grid(alpha=0.3, linestyle='--')
axes[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
axes[0].tick_params(axis='both', which='major', labelsize=s_legend) # set font size of axis ticks
axes[0].legend(loc='upper left', fontsize=s_legend)

# Middle row: Level 2 materials
for mat in types_level2:
    if mat in data_plot.material.values:
        data_plot.sel(material=mat).plot(ax=axes[1], label=mat, color=DICT_MATERIALS_COLORS[mat], linewidth=linewidth)

axes[1].set_title(" ")
axes[1].set_xlabel(" ")
axes[1].set_ylabel("Material stocks (t)", fontsize=s_label)
axes[1].grid(alpha=0.3, linestyle='--')
axes[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
axes[1].tick_params(axis='both', which='major', labelsize=s_legend) # set font size of axis ticks
axes[1].legend(loc='upper left', fontsize=s_legend)

# Bottom row: Level 3 materials
for mat in types_level3:
    if mat in data_plot.material.values:
        data_plot.sel(material=mat).plot(ax=axes[2], label=mat, color=DICT_MATERIALS_COLORS[mat], linewidth=linewidth)
axes[2].set_title(" ")
axes[2].set_xlabel("Time", fontsize=s_label)
axes[2].set_ylabel("Material stocks (t)", fontsize=s_label)
axes[2].grid(alpha=0.3, linestyle='--')
axes[2].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
axes[2].tick_params(axis='both', which='major', labelsize=s_legend) # set font size of axis ticks
axes[2].legend(loc='upper left', fontsize=s_legend)

plt.suptitle(f"{scen_folder}: Generation - Stocks Materials - World", fontsize=16)
plt.tight_layout()
# fig.savefig(path_test_plots / "Gen_stocks-materials_world.png", dpi=300)
# fig.savefig(path_test_plots / "Gen_stocks-materials_world_1971.png", dpi=300)
# fig.savefig(path_test_plots / "Gen_stocks-materials_world_1971.pdf", dpi=300)
# fig.savefig(path_test_plots / "Gen_stocks-materials_world_1971.svg", dpi=300)
plt.show()



#================================================================================
#%%%% Per TECH category - world - STACKED

# Define mapping: technology -> category
# DICT_GENTECH_TO_CATEGORY = {
#     "Solar PV": 'Solar', 
#     "Solar PV residential": 'Solar',
#     "CSP": 'Solar', 
#     "Wind onshore": 'Wind',
#     "Wind offshore": 'Wind', 
#     "Wave": 'Other Renewables',
#     "Hydro": 'Other Renewables',
#     "Other Renewables": 'Other Renewables',
#     "Geothermal": 'Other Renewables',
#     'Hydrogen power': 'Hydrogen',
#     "Nuclear": 'Nuclear',
#     "Conv. Coal": 'Fossil',
#     "Conv. Oil": 'Fossil',
#     "Conv. Natural Gas": 'Fossil',
#     "Waste": 'Fossil',
#     "IGCC": 'Fossil',
#     "OGCC": 'Fossil',
#     "NG CC": 'Fossil',
#     "Biomass CC": 'Biomass',
#     "Coal + CCS": 'Fossil + CCS',
#     "Oil/Coal + CCS": 'Fossil + CCS',
#     "Natural Gas + CCS": 'Fossil + CCS',
#     "Biomass + CCS": 'Biomass',
#     "CHP Coal": 'Fossil',
#     "CHP Oil": 'Fossil',
#     "CHP Natural Gas": 'Fossil',
#     "CHP Biomass": 'Biomass',
#     "CHP Geothermal": 'Other Renewables',
#     "CHP Hydrogen": 'Hydrogen',
#     "CHP Coal + CCS": 'Fossil + CCS',
#     "CHP Oil + CCS": 'Fossil + CCS',
#     "CHP Natural Gas + CCS": 'Fossil + CCS',
#     "CHP Biomass + CCS": 'Biomass'
# }

# DICT_GEN_CATEGORY_COLORS = {
#     'Solar':             "#FBBF09",
#     'Wind':              "#4BABFF",
#     'Biomass':           "#42DD88",
#     'Other Renewables':  "#B6F795",
#     'Hydrogen':          '#B9FAF8',
#     'Nuclear':           "#B06106",
#     'Fossil':            "#575354",
#     'Fossil + CCS':      "#BBB8B9"
# }


data_all = main_model_factory.stock_by_cohort_materials.copy() #stock_by_cohort_materials
data_all = data_all.sel(Type=data_all.Type != '<EMPTY>', Time=slice(1971, None)).pint.to("t") # only from 1971 onwards, convert grams to tonnes
data_all = data_all.sum('Region')

# Step 1: Get technology level from index
tech_level = data_all.Type.values
# Step 2: Map technologies to categories
category = pd.Series(tech_level).map(DICT_GENTECH_TO_CATEGORY)

# Step 3: Add this mapping as a new coordinate to the DataArray
data_all.coords['tech_category'] = ('Type', category)

# Step 4: Group by this new coordinate and sum over Type
data_all = data_all.groupby('tech_category').sum(dim='Type')

# # Step 2: Build a new MultiIndex with the category as level 3
# new_index = pd.MultiIndex.from_arrays([
#     gcap_stock.index.get_level_values(0),  # 'stock'
#     gcap_stock.index.get_level_values(1),  # Region
#     gcap_stock.index.get_level_values(2),  # Material
#     category                                # Mapped Category
# ], names=gcap_stock.index.names)
# # Step 3: Assign new index
# gcap_stock_techcat = gcap_stock.copy()
# gcap_stock_techcat.index = new_index
# # Step 5: Group by (region, material, category), sum
# gcap_stock_techcat = gcap_stock_techcat.groupby(
#     [gcap_stock_techcat.index.get_level_values(1),  # Region
#      gcap_stock_techcat.index.get_level_values(2),  # Material
#      gcap_stock_techcat.index.get_level_values(3)]  # Category
# ).sum()
# gcap_stock_techcat.index.names = ['regions', 'materials', 'technology_category']
# gcap_stock_techcat = gcap_stock_techcat.T # index = years

# gcap_stock_techcat_mat = gcap_stock_techcat.groupby(level=['materials','technology_category'], axis=1).sum()
# data_all = gcap_stock_techcat_mat.copy()
# rearrange column order for the stacked plot
desired_order = ['Fossil', 'Fossil + CCS', 'Nuclear', 'Hydrogen', 'Biomass', 'Wind', 'Solar', 'Other Renewables']
# new_columns = []
# for material in data_all.material.values:
#     for cat in desired_order:
#         if (material, cat) in data_all.columns:
#             new_columns.append((material, cat))
# data_all = data_all.loc[:, new_columns]



materials = ['Steel', 'Aluminium', 'Nd', 'Co']
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
s_legend = 12
s_label = 14
for i, material in enumerate(materials):
    row = i // 2
    col = i % 2
    # Select data for this material (columns under this material)
    data_plot = data_all.sel(material = material)
    data_plot = data_plot.drop_vars('material')
    data_plot = data_plot.to_pandas()
    data_plot = data_plot[desired_order] # Reorder columns
    colors = [DICT_GEN_CATEGORY_COLORS[cat] for cat in data_plot.columns] # select colors based on technology category
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

plt.suptitle(f"{scen_folder}: Generation - Stocks Materials per Tech. Cat. - World", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to make room for the suptitle
# fig.savefig(path_test_plots / "Gen_stock-materials-techcat_st-al-nd-co_world_1971.png", dpi=300)
plt.show()


###########################################################################################################
#%%% Visualize INFLOW Gen. Cap.
###########################################################################################################


#================================================================================
#%%%% Per TECH - per Region

# da_x = main_model_factory.inflow.to_array().sum('Type')
da_inflow = main_model_factory.inflow.to_array()

data_all = da_inflow
data_all = data_all.sel(Type=data_all.Type != '<EMPTY>')

data_all = data_all.sel(time=slice(1971, None), Region=regions)

regions = ['Brazil', 'C.Europe', 'China'] 
threshold = 10_000
techs_upper = [coord_name.item() for coord_name in data_all.coords['Type']  
               if data_all.sel(Type = coord_name).values.max() > threshold]
techs_lower = [coord_name.item() for coord_name in data_all.coords['Type']  
               if data_all.sel(Type = coord_name).values.max() <= threshold]


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 8))  # Now 3 columns for 3 regions
linewidth = 2
s_legend = 12
s_label = 14

for i, region in enumerate(regions):  # regions now has length 3
    col = i  # Column index: 0, 1, 2

    # Top row: Types 1–15
    for t in techs_upper:
        data_plot = data_all.sel(Type=t, Region=region)
        color, ls = DICT_GENTECH_STYLES.get(t, ('black', '-'))
        axes[0, col].plot(data_plot.time, data_plot.values, label=t, color=color, linestyle=ls, linewidth=linewidth)
    axes[0, col].set_title(f"{region}", fontsize=15)
    axes[0, col].grid(alpha=0.3, linestyle='--')
    axes[0, col].tick_params(axis='both', which='major', labelsize=s_legend)
    axes[0, 0].set_ylabel("Peak Capacity (MW)", fontsize=s_label)

    # Bottom row: Types 16–30
    for t in techs_lower:
        data_plot = data_all.sel(Type=t, Region=region)
        color, ls = DICT_GENTECH_STYLES.get(t, ('black', '-'))
        axes[1, col].plot(data_plot.time, data_plot.values, label=t, color=color, linestyle=ls, linewidth=linewidth)
    axes[1, col].grid(alpha=0.3, linestyle='--')
    axes[1, col].tick_params(axis='both', which='major', labelsize=s_legend)
    axes[1, 0].set_ylabel("Peak Capacity (MW)", fontsize=s_label)
    axes[1, col].set_xlabel("Time", fontsize=s_label)

# Y-axis number formatting and hiding right y-axis ticks
for ax in axes.flat:
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

for row in range(2):
    for col in [1, 2]:  # Hide y-tick labels for middle and right columns
        axes[row, col].tick_params(labelleft=False)
axes[0, 2].legend(fontsize=s_legend, ncol=2, loc='upper center', bbox_to_anchor=(-1.7, -1.41))
axes[1, 2].legend(fontsize=s_legend, ncol=3, loc='upper center', bbox_to_anchor=(-0.2, -0.21))

plt.suptitle(f"{scen_folder}: Generation - Inflow: Peak Capacity (MW)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to make room for the suptitle
region_str = "_".join(regions)
# fig.savefig(path_test_plots / f"Gen_inflow_{region_str}.png", dpi=300, bbox_inches='tight')
# fig.savefig(path_test_plots / f"Gen_inflow_{region_str}_1971.png", dpi=300, bbox_inches='tight')
plt.show()



#================================================================================
#%%%% Sum & Per TECH - World

# da_x = main_model_factory.inflow.to_array().sum('Type')
da_inflow = main_model_factory.inflow.to_array()

data_all = da_inflow.sel(Type=da_inflow.Type != '<EMPTY>')
data_all_1 = data_all.sum('Region')
data_all_2 = data_all.sum('Type').sum('Region')

data_all_1 = data_all_1.sel(time=slice(1971, None))
data_all_2 = data_all_2.sel(time=slice(1971, None))

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 9))
linewidth = 2
s_legend = 12
s_label = 14

# Second subplot: data_all_2 (summed over Region and Type, likely only time and material left)
data_all_2.plot(ax=axes[0], color='black', linewidth=linewidth)
axes[0].grid(alpha=0.3, linestyle='--')
axes[0].tick_params(axis='both', which='major', labelsize=s_legend)
axes[0].set_xlabel(" ", fontsize=s_label)
axes[0].set_ylabel("Peak Capacity (MW)", fontsize=s_label)
axes[0].set_title('Sum over Type and Region')

# First subplot: data_all_1 (summed over Region, still over Type and time likely)
for t in data_all_1.Type.values:
    data_plot = data_all_1.sel(Type=t)
    color, ls = DICT_GENTECH_STYLES.get(t, ('black', '-'))
    axes[1].plot(data_plot.time, data_plot.values, label=t, color=color, linestyle=ls, linewidth=linewidth)
    axes[1].grid(alpha=0.3, linestyle='--')
    axes[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
    axes[1].tick_params(axis='both', which='major', labelsize=s_legend)
    axes[1].set_xlabel("Time", fontsize=s_label)
    axes[1].set_ylabel("Peak Capacity (MW)", fontsize=s_label)
    axes[1].legend(fontsize='small', ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.2))
    axes[1].set_title('Sum over Region')

plt.suptitle(f"{scen_folder}: Generation - Inflow: Peak Capacity (MW)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to make room for the suptitle
# fig.savefig(path_test_plots / f"Gen_inflow_world.png", dpi=300, bbox_inches='tight')
# fig.savefig(path_test_plots / f"Gen_inflow_world_1971.png", dpi=300, bbox_inches='tight')
plt.show()





###########################################################################################################
#%%% Visualize INFLOW Materials
###########################################################################################################


#================================================================================
#%%%% Sum over Tech. - per region

data_all = main_model_factory.inflow_materials.to_array().sum('Type')

regions = ['Brazil', 'C.Europe', 'China']  # Brazil and C.Europe
# regions = da_x.Region.values[:2]  # First 2 regions
# types_top = da_x.material.values[1:6]   # Types 1–10
# types_bottom = da_x.material.values[6:12]  # Types 11–20
types_level1 = [m for m in data_all.material.values if m in ["Steel", "Concrete"]]
types_level2 = [m for m in data_all.material.values if m in ["Aluminium", "Cu"]]
types_level3 = [m for m in data_all.material.values if m not in (types_level1 + types_level2)]

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 12))
linewidth = 2
s_legend = 12
s_label = 14

data_plot = data_all.sel(time=slice(1971, None)).pint.to("t") # convert grams to tonnes

for i, region in enumerate(regions):
    # Top row: Level 1 materials
    for mat in types_level1:
        if (region in data_plot.Region.values) and (mat in data_plot.material.values):
            data_plot.sel(material=mat, Region=region).plot(ax=axes[0, i], label=mat, color=DICT_MATERIALS_COLORS[mat], linewidth=linewidth)
    axes[0, i].set_title(f"{region}")
    axes[0, i].set_xlabel(" ")
    axes[0, i].set_ylabel(" ")
    axes[0, i].grid(alpha=0.3, linestyle='--')
    axes[0, i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
    axes[0, i].tick_params(axis='both', which='major', labelsize=s_legend) # set font size of axis ticks
    axes[0, 2].legend(loc='upper left', fontsize=s_legend)

    # Middle row: Level 2 materials
    for mat in types_level2:
        if (region in data_plot.Region.values) and (mat in data_plot.material.values):
            data_plot.sel(material=mat, Region=region).plot(ax=axes[1, i], label=mat, color=DICT_MATERIALS_COLORS[mat], linewidth=linewidth)
    axes[1, i].set_title(f" ")
    axes[1, i].set_xlabel(" ")
    axes[1, i].set_ylabel(" ")
    axes[1, i].grid(alpha=0.3, linestyle='--')
    axes[1, i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
    axes[1, i].tick_params(axis='both', which='major', labelsize=s_legend) # set font size of axis ticks
    axes[1, 2].legend(loc='upper left', fontsize=s_legend)

    # Bottom row: Level 3 materials
    for mat in types_level3:
        if (region in data_plot.Region.values) and (mat in data_plot.material.values):
            data_plot.sel(material=mat, Region=region).plot(ax=axes[2, i], label=mat, color=DICT_MATERIALS_COLORS[mat], linewidth=linewidth)
    axes[2, i].set_title(f" ")
    axes[2, i].set_ylabel(" ")
    axes[2, i].set_xlabel("Time", fontsize=s_label)
    axes[2, i].grid(alpha=0.3, linestyle='--')
    axes[2, i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
    axes[2, i].tick_params(axis='both', which='major', labelsize=s_legend) # set font size of axis ticks
    axes[2, 2].legend(loc='upper left', fontsize=s_legend)
    

axes[0, 0].set_ylabel("Material inflow (t)", fontsize=s_label)
axes[1, 0].set_ylabel("Material inflow (t)", fontsize=s_label)
axes[2, 0].set_ylabel("Material inflow (t)", fontsize=s_label)

plt.suptitle(f"{scen_folder}: Generation - Inflow Materials", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to make room for the suptitle
region_str = "_".join(regions)
# fig.savefig(path_test_plots / f"Gen_inflow-materials_{region_str}_1971.png", dpi=300, bbox_inches='tight')
# fig.savefig(path_test_plots / f"Gen_inflow-materials_{region_str}_1971.svg", dpi=300, bbox_inches='tight')
plt.show()


# 2 regions ---------------
# fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(14, 12), sharex=True)

# axes[0, 1].sharey(axes[0, 0])
# axes[1, 1].sharey(axes[1, 0])
# axes[2, 1].sharey(axes[2, 0])

# for i, region in enumerate(regions):
#     # Top row: 
#     for mat in types_level1:
#         da_x.sel(material=mat, Region=region).plot(ax=axes[0, i], label=mat, color=DICT_MATERIALS_COLORS[mat])
#     axes[0, i].set_title(f"{region}")
#     axes[0, i].set_xlabel("Time")
#     axes[0, i].legend()

#     # Middle row: 
#     for mat in types_level2:
#         da_x.sel(material=mat, Region=region).plot(ax=axes[1, i], label=mat, color=DICT_MATERIALS_COLORS[mat])
#     axes[1, i].set_title(f"{region}")
#     axes[1, i].set_xlabel("Time")
#     axes[1, i].legend(loc ='upper left')

#     # Bottom row: 
#     for mat in types_level3:
#         da_x.sel(material=mat, Region=region).plot(ax=axes[2, i], label=mat, color=DICT_MATERIALS_COLORS[mat])
#     axes[2, i].set_title(f"{region}")
#     axes[2, i].set_xlabel("Time")
#     axes[2, i].legend(loc ='upper left')

# # Label only left side with y-axis label
# axes[0, 0].set_ylabel("Value")
# axes[1, 0].set_ylabel("Value")
# axes[2, 0].set_ylabel("Value")

# plt.suptitle("Generation - Inflow Materials", fontsize=16)
# plt.tight_layout()
# region_str = "_".join(regions)
# # fig.savefig(path_test_plots / f"Gen_inflow-materials_{region_str}_1971.png", dpi=300)
# # fig.savefig(path_test_plots / f"Gen_inflow-materials_{region_str}_1971.svg", dpi=300)
# plt.show()



#================================================================================
#%%%% Sum over Tech. - World



data_all = main_model_factory.inflow_materials.to_array().sum('Type').sum('Region')
types_level1 = [m for m in data_all.material.values if m in ["Steel", "Concrete"]]
types_level2 = [m for m in data_all.material.values if m in ["Aluminium", "Cu"]]
types_level3 = [m for m in data_all.material.values if m not in (types_level1 + types_level2)]


fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 10))

# data_plot = data_all.pint.to("t")  # convert grams to tonnes
data_plot = data_all.sel(time=slice(1971, None)).pint.to("t") # only from 1971 onwards

# Top row: Level 1 materials
for mat in types_level1:
    if mat in data_plot.material.values:
        data_plot.sel(material=mat).plot(ax=axes[0], label=mat, color=DICT_MATERIALS_COLORS[mat])
axes[0].grid(alpha=0.3, linestyle='--')
axes[0].tick_params(axis='both', which='major', labelsize=s_legend)
axes[0].set_xlabel(" ", fontsize=s_label)
axes[0].set_ylabel("Material Inflow (t)", fontsize=s_label)
axes[0].legend(fontsize=s_legend, ncol=1, loc='upper center', bbox_to_anchor=(1.08, 1))
axes[0].set_title(" ")

# Middle row: Level 2 materials
for mat in types_level2:
    if mat in data_plot.material.values:
        data_plot.sel(material=mat).plot(ax=axes[1], label=mat, color=DICT_MATERIALS_COLORS[mat])

axes[1].scatter(df_iea_cu_aps.index, df_iea_cu_aps['Gen_solar']+df_iea_cu_aps['Gen_wind']+df_iea_cu_aps['Gen_other'], 
                label=f"{df_iea_cu_aps.name}"+"\nclean en.gen.", s=10, edgecolors=DICT_MATERIALS_COLORS['Cu'], facecolors='none')

axes[1].grid(alpha=0.3, linestyle='--')
axes[1].tick_params(axis='both', which='major', labelsize=s_legend)
axes[1].set_xlabel(" ", fontsize=s_label)
axes[1].set_ylabel("Material Inflow (t)", fontsize=s_label)
axes[1].legend(fontsize=s_legend, ncol=1, loc='upper center', bbox_to_anchor=(1.1, 1))
axes[1].set_title(" ")

# Bottom row: Level 3 materials
for mat in types_level3:
    if mat in data_plot.material.values:
        data_plot.sel(material=mat).plot(ax=axes[2], label=mat, color=DICT_MATERIALS_COLORS[mat])

axes[2].scatter(df_iea_ni_aps.index, df_iea_ni_aps['Gen_solar']+df_iea_ni_aps['Gen_wind']+df_iea_ni_aps['Gen_other'], 
                label=f"{df_iea_ni_aps.name}"+"\nclean en.gen.", s=10, edgecolors=DICT_MATERIALS_COLORS['Ni'], facecolors='none')

axes[2].grid(alpha=0.3, linestyle='--')
axes[2].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
axes[2].tick_params(axis='both', which='major', labelsize=s_legend)
axes[2].set_xlabel("Time", fontsize=s_label)
axes[2].set_ylabel("Material Inflow (t)", fontsize=s_label)
axes[2].legend(fontsize=s_legend, ncol=1, loc='upper center', bbox_to_anchor=(1.1, 1))
axes[2].set_title(" ")

plt.suptitle(f"{scen_folder}: Generation - Inflow Materials - World", fontsize=16)
plt.tight_layout()
# fig.savefig(path_test_plots / "Gen_inflow-materials_world.png", dpi=300, bbox_inches='tight')
# fig.savefig(path_test_plots / "Gen_inflow-materials_world_1971.png", dpi=300, bbox_inches='tight')
# fig.savefig(path_test_plots / "Gen_inflow-materials_world_1971.pdf", dpi=300, bbox_inches='tight')
# fig.savefig(path_test_plots / "Gen_inflow-materials_world_1971.svg", dpi=300, bbox_inches='tight')
plt.show()



###########################################################################################################
#%%% Visualize Outflow Gen. Cap.
###########################################################################################################


#================================================================================
#%%%% Per TECH - per Region

da_outflow = main_model_factory.outflow_by_cohort.to_array()

data_all = da_outflow
data_all = data_all.sel(Type=data_all.Type != '<EMPTY>').sum('Cohort')

data_all = data_all.sel(time=slice(1971, None))

regions = ['Brazil', 'C.Europe', 'China'] 
threshold = 10_000

techs_upper = [coord_name.item() for coord_name in data_all.coords['Type']  
               if data_all.sel(Type = coord_name).values.max() > threshold]
techs_lower = [coord_name.item() for coord_name in data_all.coords['Type']  
               if data_all.sel(Type = coord_name).values.max() <= threshold]


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 8))  # Now 3 columns for 3 regions
linewidth = 2
s_legend = 12
s_label = 14

for i, region in enumerate(regions):  # regions now has length 3
    col = i  # Column index: 0, 1, 2

    # Top row: Types 1–15
    for t in techs_upper:
        data_plot = data_all.sel(Type=t, Region=region)
        color, ls = DICT_GENTECH_STYLES.get(t, ('black', '-'))
        axes[0, col].plot(data_plot.time, data_plot.values, label=t, color=color, linestyle=ls, linewidth=linewidth)
    axes[0, col].set_title(f"{region}", fontsize=15)
    axes[0, col].grid(alpha=0.3, linestyle='--')
    axes[0, col].tick_params(axis='both', which='major', labelsize=s_legend)
    axes[0, 0].set_ylabel("Peak Capacity (MW)", fontsize=s_label)

    # Bottom row: Types 16–30
    for t in techs_lower:
        data_plot = data_all.sel(Type=t, Region=region)
        color, ls = DICT_GENTECH_STYLES.get(t, ('black', '-'))
        axes[1, col].plot(data_plot.time, data_plot.values, label=t, color=color, linestyle=ls, linewidth=linewidth)
    axes[1, col].grid(alpha=0.3, linestyle='--')
    axes[1, col].tick_params(axis='both', which='major', labelsize=s_legend)
    axes[1, 0].set_ylabel("Peak Capacity (MW)", fontsize=s_label)
    axes[1, col].set_xlabel("Time", fontsize=s_label)

# Y-axis number formatting and hiding right y-axis ticks
for ax in axes.flat:
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

for row in range(2):
    for col in [1, 2]:  # Hide y-tick labels for middle and right columns
        axes[row, col].tick_params(labelleft=False)
axes[0, 2].legend(fontsize=s_legend, ncol=1, loc='upper center', bbox_to_anchor=(-2, -1.5))
axes[1, 2].legend(fontsize=s_legend, ncol=4, loc='upper center', bbox_to_anchor=(-0.5, -0.3))

plt.suptitle(f"{scen_folder}: Generation - Outflow: Peak Capacity (MW)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to make room for the suptitle
region_str = "_".join(regions)
# fig.savefig(path_test_plots / f"Gen_outflow_{region_str}_1971.png", dpi=300, bbox_inches='tight')
plt.show()



#================================================================================
#%%%% Sum & Per TECH - World

da_outflow = main_model_factory.outflow_by_cohort.to_array()

data_all   = da_outflow.sel(Type=da_inflow.Type != '<EMPTY>')
data_all_1 = data_all.sum('Cohort').sum('Region')
data_all_2 = data_all.sum('Cohort').sum('Type').sum('Region')

data_all_1 = data_all_1.sel(time=slice(1971, None))
data_all_2 = data_all_2.sel(time=slice(1971, None))

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 9))
linewidth = 2
s_legend = 12
s_label = 14

# Second subplot: data_all_2 (summed over Region and Type, likely only time and material left)
data_all_2.plot(ax=axes[0], color='black', linewidth=linewidth)
axes[0].grid(alpha=0.3, linestyle='--')
axes[0].tick_params(axis='both', which='major', labelsize=s_legend)
axes[0].set_xlabel(" ", fontsize=s_label)
axes[0].set_ylabel("Peak Capacity (MW)", fontsize=s_label)
axes[0].set_title('Sum over Type and Region')

# First subplot: data_all_1 (summed over Region, still over Type and time likely)
for t in data_all_1.Type.values:
    data_plot = data_all_1.sel(Type=t)
    color, ls = DICT_GENTECH_STYLES.get(t, ('black', '-'))
    axes[1].plot(data_plot.time, data_plot.values, label=t, color=color, linestyle=ls, linewidth=linewidth)
    axes[1].grid(alpha=0.3, linestyle='--')
    axes[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
    axes[1].tick_params(axis='both', which='major', labelsize=s_legend)
    axes[1].set_xlabel("Time", fontsize=s_label)
    axes[1].set_ylabel("Peak Capacity (MW)", fontsize=s_label)
    axes[1].legend(fontsize='small', ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.2))
    axes[1].set_title('Sum over Region')

plt.suptitle(f"{scen_folder}: Generation - Outflow: Peak Capacity (MW)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to make room for the suptitle
# fig.savefig(path_test_plots / f"Gen_inflow_world.png", dpi=300)
# fig.savefig(path_test_plots / f"Gen_outflow_world_1971.png", dpi=300)
plt.show()



###########################################################################################################
#%%% Visualize OUTFLOW Materials
###########################################################################################################


#================================================================================
#%%%% Sum over Tech. - per region

data_all = main_model_factory.outflow_by_cohort_materials.to_array().sum('Type')

regions = ['Brazil', 'C.Europe', 'China']  # Brazil and C.Europe

types_level1 = [m for m in data_all.material.values if m in ["Steel", "Concrete"]]
types_level2 = [m for m in data_all.material.values if m in ["Aluminium", "Cu"]]
types_level3 = [m for m in data_all.material.values if m not in (types_level1 + types_level2)]

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 12))
linewidth = 2
s_legend = 12
s_label = 14

# data_plot = data_all.pint.to("t") # convert grams to tonnes
data_plot = data_all.sel(time=slice(1971, None)).pint.to("t") 

for i, region in enumerate(regions):
    # Top row: Level 1 materials
    for mat in types_level1:
        if (region in data_plot.Region.values) and (mat in data_plot.material.values):
            data_plot.sel(material=mat, Region=region).plot(ax=axes[0, i], label=mat, color=DICT_MATERIALS_COLORS[mat], linewidth=linewidth)
    axes[0, i].set_title(f"{region}")
    axes[0, i].set_xlabel(" ")
    axes[0, i].set_ylabel(" ")
    axes[0, i].grid(alpha=0.3, linestyle='--')
    axes[0, i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
    axes[0, i].tick_params(axis='both', which='major', labelsize=s_legend) # set font size of axis ticks
    axes[0, 2].legend(fontsize=s_legend, ncol=1, loc='upper center', bbox_to_anchor=(1.189, 1))

    # Middle row: Level 2 materials
    for mat in types_level2:
        if (region in data_plot.Region.values) and (mat in data_plot.material.values):
            data_plot.sel(material=mat, Region=region).plot(ax=axes[1, i], label=mat, color=DICT_MATERIALS_COLORS[mat], linewidth=linewidth)
    axes[1, i].set_title(f" ")
    axes[1, i].set_xlabel(" ")
    axes[1, i].set_ylabel(" ")
    axes[1, i].grid(alpha=0.3, linestyle='--')
    axes[1, i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
    axes[1, i].tick_params(axis='both', which='major', labelsize=s_legend) # set font size of axis ticks
    axes[1, 2].legend(fontsize=s_legend, ncol=1, loc='upper center', bbox_to_anchor=(1.19, 1))

    # Bottom row: Level 3 materials
    for mat in types_level3:
        if (region in data_plot.Region.values) and (mat in data_plot.material.values):
            data_plot.sel(material=mat, Region=region).plot(ax=axes[2, i], label=mat, color=DICT_MATERIALS_COLORS[mat], linewidth=linewidth)
    axes[2, i].set_title(f" ")
    axes[2, i].set_ylabel(" ")
    axes[2, i].set_xlabel("Time", fontsize=s_label)
    axes[2, i].grid(alpha=0.3, linestyle='--')
    axes[2, i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
    axes[2, i].tick_params(axis='both', which='major', labelsize=s_legend) # set font size of axis ticks
    axes[2, 2].legend(fontsize=s_legend, ncol=1, loc='upper center', bbox_to_anchor=(1.18, 1))
    

axes[0, 0].set_ylabel("Material outflow (t)", fontsize=s_label)
axes[1, 0].set_ylabel("Material outflow (t)", fontsize=s_label)
axes[2, 0].set_ylabel("Material outflow (t)", fontsize=s_label)

plt.suptitle(f"{scen_folder}: Generation - Outflow Materials", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to make room for the suptitle
region_str = "_".join(regions)
# fig.savefig(path_test_plots / f"Gen_outflow-materials_{region_str}.png", dpi=300)
# fig.savefig(path_test_plots / f"Gen_outflow-materials_{region_str}_1971.png", dpi=300)
# fig.savefig(path_test_plots / f"Gen_outflow-materials_{region_str}_1971.svg", dpi=300)
plt.show()



#================================================================================
#%%%% Sum over Tech. - World


data_all = main_model_factory.outflow_by_cohort_materials.to_array().sum('Type').sum('Region')
types_level1 = [m for m in data_all.material.values if m in ["Steel", "Concrete"]]
types_level2 = [m for m in data_all.material.values if m in ["Aluminium", "Cu"]]
types_level3 = [m for m in data_all.material.values if m not in (types_level1 + types_level2)]


fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 10))

# data_plot = data_all.pint.to("t")  # convert grams to tonnes
data_plot = data_all.sel(time=slice(1971, None)).pint.to("t") # only from 1971 onwards

# Top row: Level 1 materials
for mat in types_level1:
    if mat in data_plot.material.values:
        data_plot.sel(material=mat).plot(ax=axes[0], label=mat, color=DICT_MATERIALS_COLORS[mat])
axes[0].grid(alpha=0.3, linestyle='--')
axes[0].tick_params(axis='both', which='major', labelsize=s_legend)
axes[0].set_xlabel(" ", fontsize=s_label)
axes[0].set_ylabel("Material Outflow (t)", fontsize=s_label)
axes[0].legend(fontsize=s_legend, ncol=1, loc='upper center', bbox_to_anchor=(1.08, 1))
axes[0].set_title(" ")

# Middle row: Level 2 materials
for mat in types_level2:
    if mat in data_plot.material.values:
        data_plot.sel(material=mat).plot(ax=axes[1], label=mat, color=DICT_MATERIALS_COLORS[mat])
axes[1].grid(alpha=0.3, linestyle='--')
axes[1].tick_params(axis='both', which='major', labelsize=s_legend)
axes[1].set_xlabel(" ", fontsize=s_label)
axes[1].set_ylabel("Material Outflow (t)", fontsize=s_label)
axes[1].legend(fontsize=s_legend, ncol=1, loc='upper center', bbox_to_anchor=(1.1, 1))
axes[1].set_title(" ")

# Bottom row: Level 3 materials
for mat in types_level3:
    if mat in data_plot.material.values:
        data_plot.sel(material=mat).plot(ax=axes[2], label=mat, color=DICT_MATERIALS_COLORS[mat])
axes[2].grid(alpha=0.3, linestyle='--')
axes[2].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
axes[2].tick_params(axis='both', which='major', labelsize=s_legend)
axes[2].set_xlabel("Time", fontsize=s_label)
axes[2].set_ylabel("Material Outflow (t)", fontsize=s_label)
axes[2].legend(fontsize=s_legend, ncol=1, loc='upper center', bbox_to_anchor=(1.1, 1))
axes[2].set_title(" ")

plt.suptitle(f"{scen_folder}: Generation - Outflow Materials - World", fontsize=16)
plt.tight_layout()
# fig.savefig(path_test_plots / "Gen_outflow-materials_world.png", dpi=300)
fig.savefig(path_test_plots / "Gen_outflow-materials_world_1971.png", dpi=300)
# fig.savefig(path_test_plots / "Gen_outflow-materials_world_1971.pdf", dpi=300)
# fig.savefig(path_test_plots / "Gen_outflow-materials_world_1971.svg", dpi=300)
plt.show()


###########################################################################################################
#%%% ALL in one Fig.
###########################################################################################################


#================================================================================
#%%%% Tech. X - Brazil
#================================================================================


# type_tech, tech_str = 'Solar PV residential', 'PVres' # name of technology in stock model, abbreviation for saved plot
type_tech, tech_str = 'Hydro', 'Hydro'
regions = ["Brazil", "C.Europe", "China"] 
region = regions[0]

da_stocks = main_model_factory.stocks.sel(Type=type_tech)
da_inflow = main_model_factory.inflow.to_array().sel(Type=type_tech)
da_outflow = main_model_factory.outflow_by_cohort.to_array().sel(Type=type_tech).sum('Cohort')

# da_stocks_mat = main_model_factory.stock_by_cohort_materials.sel(Type=type_tech) #stock_by_cohort_materials
da_inflow_mat = main_model_factory.inflow_materials.to_array().sel(Type=type_tech) 
da_outflow_mat = main_model_factory.outflow_by_cohort_materials.to_array().sel(Type=type_tech) 

da_stocks = da_stocks.sel(Time=slice(1971, None))
da_inflow = da_inflow.sel(time=slice(1971, None))
da_outflow = da_outflow.sel(time=slice(1971, None))
da_inflow_mat = da_inflow_mat.sel(time=slice(1971, None)).pint.to("t") # convert grams to tonnes
da_outflow_mat = da_outflow_mat.sel(time=slice(1971, None)).pint.to("t")


types_level1 = ["Concrete"] # PV: "Aluminium", Hydro: Concrete
types_level2 = ["Steel"]
types_level3 = ["Cu"]
materials = types_level1 + types_level2 + types_level3


fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 12))
linewidth = 2
s_legend = 12
s_label = 14

color_stocks  = "#4F70F6"
color_inflow  = "#67B2F4"
color_outflow = "#4A7BA5"
color_inflow_mat = ["#B9FAF8", '#FF9B85', '#FB6376']
color_outflow_mat = ["#0ECAC4", "#C8300E", "#B40E21"]

for i, mat in enumerate(materials):
    # I. Top row: Level 1 materials --------------------
    # 1. Y-axis (Left): Stocks
    da_stocks.sel(Region=region).plot(ax=axes[i], label='stocks', color=color_stocks, linewidth=linewidth) 
    axes[i].set_title(f"{region}")
    axes[i].set_xlabel(" ")
    axes[i].set_ylabel("Stocks (MW)", fontsize=s_label, color=color_stocks)
    axes[i].grid(alpha=0.3, linestyle='--')
    axes[i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
    axes[i].tick_params(axis='both', which='major', labelsize=s_legend, color=color_stocks) # set font size of axis ticks
    axes[i].legend(loc='upper left', fontsize=s_legend)
    axes[i].set_title(' ')

    # 2. Y-axis (Left): Inflow and Outflow
    ax2 = axes[i].twinx()
    ax2.spines["left"].set_position(("axes", -0.1))  # Shift second left y-axis
    ax2.spines["left"].set_visible(True)
    ax2.yaxis.set_label_position('left')
    ax2.yaxis.set_ticks_position('left')

    da_inflow.sel(Region=region).plot(ax=ax2, color=color_inflow, linestyle = ":", linewidth=linewidth+2, label='Inflow')
    da_outflow.sel(Region=region).plot(ax=ax2, color=color_outflow, linestyle = "--", linewidth=linewidth, label='Outflow')
    ax2.set_ylabel("In/Outflow (MW)", fontsize=s_label, color=color_outflow)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
    ax2.yaxis.offsetText.set_x(-0.11)
    ax2.tick_params(axis='y', labelcolor=color_outflow, labelsize=s_legend)
    axes[i].legend(loc='upper left', fontsize=s_legend)
    ax2.set_title(' ')

    # 3. Y-axis (Right): Material inflow/outflow
    # mat = types_level1[0]
    ax3 = axes[i].twinx()
    da_inflow_mat.sel(material=mat, Region=region).plot(ax=ax3, color=color_inflow_mat[i], linestyle = ":", linewidth=linewidth+1, label='Material Inflow')
    da_outflow_mat.sel(material=mat, Region=region).plot(ax=ax3, color=color_outflow_mat[i], linestyle = "--", linewidth=linewidth, label='Material Outflow')
    ax3.set_ylabel("Material In/Outflow (t)", fontsize=s_label, color=color_outflow_mat[i])
    ax3.ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
    ax3.tick_params(axis='y', labelcolor=color_outflow_mat[i], labelsize=s_legend)
    ax3.set_title(mat, fontsize=15)

    lines1, labels1 = axes[i].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()

    axes[i].legend(
        handles=lines1 + lines2 + lines3,
        labels=labels1 + labels2 + labels3,
        loc='upper left',
        fontsize=s_legend
    )

plt.suptitle(f"{region} - {type_tech}", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to make room for the suptitle
# fig.savefig(path_test_plots / f"Gen_{tech_str}_{region}_1971.png", dpi=300)
# fig.savefig(path_test_plots / f"Gen_{tech_str}_{region}_1971.svg", dpi=300)
plt.show()


# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
# da_inflow.sel(Region=region).plot(ax=axes, color=color_inflow, linestyle = ":", linewidth=linewidth+1, label='Inflow')
# da_outflow.sel(Region=region).plot(ax=axes, color=color_outflow, linestyle = "--", linewidth=linewidth, label='Outflow')
# axes.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

#================================================================================
#%%%% Solar PV - C.Europe
#================================================================================


type_tech, tech_str = 'Solar PV residential', 'PVres' # name of technology in stock model, abbreviation for saved plot
# type_tech, tech_str = 'Hydro', 'Hydro'
regions = ["Brazil", "C.Europe", "China"]
region = regions[1]

da_stocks = main_model_factory.stocks.sel(Type=type_tech)
da_inflow = main_model_factory.inflow.to_array().sel(Type=type_tech)
da_outflow = main_model_factory.outflow_by_cohort.to_array().sel(Type=type_tech).sum('Cohort')

# da_stocks_mat = main_model_factory.stock_by_cohort_materials.sel(Type=type_tech) #stock_by_cohort_materials
da_inflow_mat = main_model_factory.inflow_materials.to_array().sel(Type=type_tech) 
da_outflow_mat = main_model_factory.outflow_by_cohort_materials.to_array().sel(Type=type_tech) 

da_stocks = da_stocks.sel(Time=slice(1971, None))
da_inflow = da_inflow.sel(time=slice(1971, None))
da_outflow = da_outflow.sel(time=slice(1971, None))
da_inflow_mat = da_inflow_mat.sel(time=slice(1971, None)).pint.to("t") # convert grams to tonnes
da_outflow_mat = da_outflow_mat.sel(time=slice(1971, None)).pint.to("t")


types_level1 = ["Concrete"] # PV: "Aluminium", Hydro: Concrete
types_level2 = ["Steel"]
types_level3 = ["Cu"]
materials = types_level1 + types_level2 + types_level3


fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 12))
linewidth = 2
s_legend = 12
s_label = 14

color_stocks  = "#4F70F6"
color_inflow  = "#67B2F4"
color_outflow = "#4A7BA5"
color_inflow_mat = ["#B9FAF8", '#FF9B85', '#FB6376']
color_outflow_mat = ["#0ECAC4", "#C8300E", "#B40E21"]

for i, mat in enumerate(materials):
    # I. Top row: Level 1 materials --------------------
    # 1. Y-axis (Left): Stocks
    da_stocks.sel(Region=region).plot(ax=axes[i], label='stocks', color=color_stocks, linewidth=linewidth) 
    axes[i].set_title(f"{region}")
    axes[i].set_xlabel(" ")
    axes[i].set_ylabel("Stocks (MW)", fontsize=s_label, color=color_stocks)
    axes[i].grid(alpha=0.3, linestyle='--')
    axes[i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
    axes[i].tick_params(axis='both', which='major', labelsize=s_legend, color=color_stocks) # set font size of axis ticks
    axes[i].legend(loc='upper left', fontsize=s_legend)
    axes[i].set_title(' ')

    # 2. Y-axis (Left): Inflow and Outflow
    ax2 = axes[i].twinx()
    ax2.spines["left"].set_position(("axes", -0.1))  # Shift second left y-axis
    ax2.spines["left"].set_visible(True)
    ax2.yaxis.set_label_position('left')
    ax2.yaxis.set_ticks_position('left')

    da_inflow.sel(Region=region).plot(ax=ax2, color=color_inflow, linestyle = ":", linewidth=linewidth+1.5, label='Inflow')
    da_outflow.sel(Region=region).plot(ax=ax2, color=color_outflow, linestyle = "--", linewidth=linewidth+0.5, label='Outflow')
    ax2.set_ylabel("In/Outflow (MW)", fontsize=s_label, color=color_outflow)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
    ax2.yaxis.offsetText.set_x(-0.11)
    ax2.tick_params(axis='y', labelcolor=color_outflow, labelsize=s_legend)
    axes[i].legend(loc='upper left', fontsize=s_legend)
    ax2.set_title(' ')

    # 3. Y-axis (Right): Material inflow/outflow
    # mat = types_level1[0]
    ax3 = axes[i].twinx()
    da_inflow_mat.sel(material=mat, Region=region).plot(ax=ax3, color=color_inflow_mat[i], linestyle = ":", linewidth=linewidth+1, label='Material Inflow')
    da_outflow_mat.sel(material=mat, Region=region).plot(ax=ax3, color=color_outflow_mat[i], linestyle = "--", linewidth=linewidth, label='Material Outflow')
    ax3.set_ylabel("Material In/Outflow (t)", fontsize=s_label, color=color_outflow_mat[i])
    ax3.ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
    ax3.tick_params(axis='y', labelcolor=color_outflow_mat[i], labelsize=s_legend)
    ax3.set_title(mat, fontsize=15)

    lines1, labels1 = axes[i].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()

    axes[i].legend(
        handles=lines1 + lines2 + lines3,
        labels=labels1 + labels2 + labels3,
        loc='upper left',
        fontsize=s_legend
    )

plt.suptitle(f"{scen_folder}: {region} - {type_tech}", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to make room for the suptitle
# fig.savefig(path_test_plots / f"Gen_{tech_str}_{region}_1971.png", dpi=300)
# fig.savefig(path_test_plots / f"Gen_{tech_str}_{region}_1971.svg", dpi=300)
plt.show()


# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 12))
# da_outflow.sel(Region="C.Europe").plot()


#================================================================================
#%%%% Solar PV - China
#================================================================================


# type_tech, tech_str = 'Solar PV residential', 'PVres' # name of technology in stock model, abbreviation for saved plot
type_tech, tech_str = 'Hydro', 'Hydro'
regions = ["Brazil", "C.Europe", "China"] 
region = regions[2]

da_stocks = main_model_factory.stocks.sel(Type=type_tech)
da_inflow = main_model_factory.inflow.to_array().sel(Type=type_tech)
da_outflow = main_model_factory.outflow_by_cohort.to_array().sel(Type=type_tech).sum('Cohort')

# da_stocks_mat = main_model_factory.stock_by_cohort_materials.sel(Type=type_tech) #stock_by_cohort_materials
da_inflow_mat = main_model_factory.inflow_materials.to_array().sel(Type=type_tech) 
da_outflow_mat = main_model_factory.outflow_by_cohort_materials.to_array().sel(Type=type_tech) 

da_stocks = da_stocks.sel(Time=slice(1971, None))
da_inflow = da_inflow.sel(time=slice(1971, None))
da_outflow = da_outflow.sel(time=slice(1971, None))
da_inflow_mat = da_inflow_mat.sel(time=slice(1971, None)).pint.to("t") # convert grams to tonnes
da_outflow_mat = da_outflow_mat.sel(time=slice(1971, None)).pint.to("t")


types_level1 = ["Concrete"] # PV: "Aluminium", Hydro: Concrete
types_level2 = ["Steel"]
types_level3 = ["Cu"]
materials = types_level1 + types_level2 + types_level3



fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(12, 12))
linewidth = 2
s_legend = 12
s_label = 14

color_stocks  = "#4F70F6"
color_inflow  = "#67B2F4"
color_outflow = "#4A7BA5"
color_inflow_mat = ["#B9FAF8", '#FF9B85', '#FB6376']
color_outflow_mat = ["#0ECAC4", "#C8300E", "#B40E21"]

for i, mat in enumerate(materials):
    # I. Top row: Level 1 materials --------------------
    # 1. Y-axis (Left): Stocks
    da_stocks.sel(Region=region).plot(ax=axes[i], label='stocks', color=color_stocks, linewidth=linewidth) 
    axes[i].set_title(f"{region}")
    axes[i].set_xlabel(" ")
    axes[i].set_ylabel("Stocks (MW)", fontsize=s_label, color=color_stocks)
    axes[i].grid(alpha=0.3, linestyle='--')
    axes[i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
    axes[i].tick_params(axis='both', which='major', labelsize=s_legend, color=color_stocks) # set font size of axis ticks
    axes[i].legend(loc='upper left', fontsize=s_legend)
    axes[i].set_title(' ')

    # 2. Y-axis (Left): Inflow and Outflow
    ax2 = axes[i].twinx()
    ax2.spines["left"].set_position(("axes", -0.1))  # Shift second left y-axis
    ax2.spines["left"].set_visible(True)
    ax2.yaxis.set_label_position('left')
    ax2.yaxis.set_ticks_position('left')

    da_inflow.sel(Region=region).plot(ax=ax2, color=color_inflow, linestyle = ":", linewidth=linewidth+1.5, label='Inflow')
    da_outflow.sel(Region=region).plot(ax=ax2, color=color_outflow, linestyle = "--", linewidth=linewidth+0.5, label='Outflow')
    ax2.set_ylabel("In/Outflow (MW)", fontsize=s_label, color=color_outflow)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
    ax2.yaxis.offsetText.set_x(-0.11)
    ax2.tick_params(axis='y', labelcolor=color_outflow, labelsize=s_legend)
    axes[i].legend(loc='upper left', fontsize=s_legend)
    ax2.set_title(' ')

    # 3. Y-axis (Right): Material inflow/outflow
    # mat = types_level1[0]
    ax3 = axes[i].twinx()
    da_inflow_mat.sel(material=mat, Region=region).plot(ax=ax3, color=color_inflow_mat[i], linestyle = ":", linewidth=linewidth+1, label='Material Inflow')
    da_outflow_mat.sel(material=mat, Region=region).plot(ax=ax3, color=color_outflow_mat[i], linestyle = "--", linewidth=linewidth, label='Material Outflow')
    ax3.set_ylabel("Material In/Outflow (t)", fontsize=s_label, color=color_outflow_mat[i])
    ax3.ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
    ax3.tick_params(axis='y', labelcolor=color_outflow_mat[i], labelsize=s_legend)
    ax3.set_title(mat, fontsize=15)

    lines1, labels1 = axes[i].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()

    axes[i].legend(
        handles=lines1 + lines2 + lines3,
        labels=labels1 + labels2 + labels3,
        loc='upper left',
        fontsize=s_legend
    )

plt.suptitle(f"{region} - {type_tech}", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to make room for the suptitle
# fig.savefig(path_test_plots / f"Gen_{tech_str}_{region}_1971.png", dpi=300)
# fig.savefig(path_test_plots / f"Gen_{tech_str}_{region}_1971.svg", dpi=300)
plt.show()



###########################################################################################################
#%%% Other
###########################################################################################################


#%%%% Material Intenisties ------------------------------------------------------------

# Load data
df = pd.read_csv(path_external_data_scenario / 'composition_generation.csv',index_col=[0,1])
df = df.reset_index()
# Clean up empty rows
df = df[df['g/MW'] != '<EMPTY>']

# Ensure year is numeric
df['year'] = df['year'].astype(int)
n_rows = 4
n_cols = 3

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 12), sharex=True)
axes = axes.flatten()
# Plot each material
for i, material in enumerate(df.columns[2:]):
    ax = axes[i]
    for tech in df['g/MW'].unique():
        sub = df[df['g/MW'] == tech]
        if len(sub) == 2:
            ax.plot(sub['year'], sub[material], marker='o', label=tech)
    ax.set_title(material)
    # ax.grid(True)
    if i % n_cols == 0:
        ax.set_ylabel('[g/MW]')
    if i >= (n_rows - 1) * n_cols:
        ax.set_xlabel('Year')

# One shared legend outside the plot
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, bbox_to_anchor=(0.9, 0.5), loc='center left')
plt.tight_layout(rect=[0, 0, 0.85, 1])
fig.savefig(path_test_plots / f"Gen_MIs.png", dpi=300)
plt.show()



# Step 1: Pivot the data for easy comparison
df_pivot = df.pivot(index='g/MW', columns='year')

# Step 2: Compare values between 2020 and 2050
change_df = (df_pivot.xs(2050, level=1, axis=1) != df_pivot.xs(2020, level=1, axis=1)).astype(int)

# Step 3: Format the change rows back to original style
change_df['year'] = 'change'
change_df.reset_index(inplace=True)

# Step 4: Append change rows to original DataFrame
df_combined = pd.concat([df, change_df], ignore_index=True)

a= df_combined[df_combined['year'] == 'change'].sum(numeric_only=True)


# path_test_plots = Path(path_base, "imagematerials", "electricity", "out_test", scen_folder, "Figures")




















###########################################################################################################
###########################################################################################################
#%% 2) STORAGE
###########################################################################################################
###########################################################################################################

from imagematerials.electricity.preprocessing import (
    get_preprocessing_data_gen,
    get_preprocessing_data_grid,
    get_preprocessing_data_stor
)
# from imagematerials.util import import_from_netcdf, export_to_netcdf
from imagematerials.model import GenericMainModel, GenericMaterials, GenericStocks, Maintenance, MaterialIntensities, SharesInflowStocks

YEAR_FIRST_STOR = 1907 # first use of pumped storage was in 1907 at the Engeweiher pumped storage facility near Schaffhausen, Switzerland (Mitali et al. 2022)
YEAR_START = 1971  # start year of the simulation period
YEAR_END = 2100    # end year of the calculations
YEAR_OUT = 2100    # year of output generation = last year of reporting

prep_data_phs, prep_data_oth_storage = get_preprocessing_data_stor(path_base, SCEN, VARIANT, YEAR_START, YEAR_END, YEAR_OUT)

# PHS =======================================================================================
time_start = prep_data_phs["stocks"].coords["Time"].min().values
complete_timeline = prism.Timeline(time_start, YEAR_END, 1)
simulation_timeline = prism.Timeline(YEAR_START, YEAR_END, 1) #1970

sec_electr_stor_phs = Sector("electr_stor_phs", prep_data_phs)

main_model_factory_phs = ModelFactory(
    sec_electr_stor_phs, complete_timeline
    ).add(GenericStocks
    ).add(MaterialIntensities
    ).finish()

main_model_factory_phs.simulate(simulation_timeline)
list(main_model_factory_phs.electr_stor_phs)

# Other Storage ==============================================================================
time_start = prep_data_oth_storage["stocks"].coords["Time"].min().values
complete_timeline = prism.Timeline(time_start, YEAR_END, 1)
simulation_timeline = prism.Timeline(YEAR_START, YEAR_END, 1) #1970

sec_electr_stor_oth = Sector("electr_stor_oth", prep_data_oth_storage, check_coordinates=False)

main_model_factory_oth = ModelFactory(
    sec_electr_stor_oth, complete_timeline
    ).add(SharesInflowStocks
    ).add(MaterialIntensities
    ).finish()

main_model_factory_oth.simulate(simulation_timeline)
list(main_model_factory_oth.electr_stor_oth)

path_test_plots = Path(path_base, "imagematerials", "electricity", "out_test", scen_folder, "Figures")

#----------------------------------------------------------------------------------------------------------
###########################################################################################################
#%%% 2.1) Read in files
###########################################################################################################
#----------------------------------------------------------------------------------------------------------


# 1. External Data ======================================================================================== 


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

# material compositions (storage) in wt%
storage_materials = pd.read_csv(path_external_data_standard / 'storage_materials_dynamic.csv',index_col=[0,1]).transpose()  # wt% of total battery weight for various materials, total battery weight is given by the density file above

# Hydro-dam power capacity (also MW) within 5 regions reported by the IHA (international Hydropwer Association)
phs_projections = pd.read_csv(path_external_data_standard / 'PHS.csv', index_col='t')   # pumped hydro storage capacity (MW)


# 2. IMAGE/TIMER files ====================================================================================


# # The vehicle shares of trucks (heavy) of the SSP2
#     loadfactor_car_data: pd.DataFrame = read_mym_df(
#         image_folder.joinpath("trp_trvl_Load.out")). rename(
#         columns={
#             "DIM_1": "region"})

# read TIMER installed storage capacity (MWh, reservoir)
storage = read_mym_df(path_image_output.joinpath("StorResTot.out"))   #storage capacity in MWh (reservoir, so energy capacity, not power capacity, the latter is used later on in the pumped hydro storage calculations)
    
#storage capacity in MW (power capacity), to compare it to Pumped hydro storage projections (also given in MW, power capacity)
storage_power = read_mym_df(path_image_output / 'StorCapTot.out')  

# loadfactor_data = read_mym_df(path_image_output / 'trp_trvl_Load.out') 
  
# passengerkms_data = read_mym_df(path_image_output / 'trp_trvl_pkm.out')   # passenger kilometers in Tera pkm

# vehicleshare_data = read_mym_df(path_image_output / 'trp_trvl_Vshare_car.out')

# Generation capacity (stock & inflow/new) in MW peak capacity, FILES from TIMER
gcap_data = read_mym_df(path_image_output / 'Gcap.out') # needed to get hydro power for storage
gcap_data = gcap_data.iloc[:, :26]



# ----------------------------------------------------------------------------------------------------------
# ##########################################################################################################
# %% 2.2) Prepare general variables
# ##########################################################################################################
# ----------------------------------------------------------------------------------------------------------

# Calculations used in both 'vehicle battery storage' and 'other storage'
YEAR_FIRST = 1907 # first use of pumped storage was in 1907 at the Engeweiher pumped storage facility near Schaffhausen, Switzerland (Mitali et al. 2022)
YEAR_START = 1971
YEAR_END = 2100
YEAR_OUT = 2100

##################
# Interpolations #
##################

storage = storage.iloc[:, :26]    # drop global total column and empty (27) column

# J: in high storage scenario the storage demand linearly increases between 2021 and 2050 compared to its original value until it is double by 2050, and then remains constant
if SENS_ANALYSIS == 'high_stor':
   storage_multiplier = storage
   for year in range(2021,2051):
        storage_multiplier.loc[year] = storage.loc[year] * (1 + (1/30*(year-2020)))
   for year in range(2051,YEAR_END+1):
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

# energy density ---
# fix the energy density (kg/kwh) of storage technologies after 2030
for year in range(2030+1,YEAR_OUT+1):
    # storage_density_interpol = storage_density_interpol.append(pd.Series(storage_density_interpol.loc[storage_density_interpol.last_valid_index()], name=year))
    row = storage_density_interpol.loc[[storage_density_interpol.last_valid_index()]]
    row.index = [year]
    storage_density_interpol = pd.concat([storage_density_interpol, row])
# assumed fixed energy densities before 2018
for year in reversed(range(YEAR_FIRST_GRID,storage_start)): # was YEAR_SWITCH, storage_start
    # storage_density_interpol = storage_density_interpol.append(pd.Series(storage_density_interpol.loc[storage_density_interpol.first_valid_index()], name=year)).sort_index(axis=0)
    row = storage_density_interpol.loc[[storage_density_interpol.first_valid_index()]]
    row.index = [year]
    storage_density_interpol = pd.concat([storage_density_interpol, row]).sort_index(axis=0)

# storage material intensity ---
# Interpolate material intensities (dynamic content for gcap & storage technologies between 1926 to 2100, based on data files)
index = pd.MultiIndex.from_product([list(range(YEAR_FIRST_GRID, YEAR_OUT+1)), list(storage_materials.index)])
stor_materials_interpol = pd.DataFrame(index=index, columns=storage_materials.columns.levels[1])
# material intensities for storage
for cat in list(storage_materials.columns.levels[1]):
   stor_materials_1st   = storage_materials.loc[:,idx[storage_materials.columns[0][0],cat]]
   stor_materials_interpol.loc[idx[YEAR_FIRST_GRID ,:],cat] = stor_materials_1st.to_numpy()  # set the first year (1926) values to the first available values in the dataset (for the year 2000) 
   stor_materials_interpol.loc[idx[storage_materials.columns.levels[0].min(),:],cat] = storage_materials.loc[:, idx[storage_materials.columns.levels[0].min(),cat]].to_numpy() # set the middle year (2000) values to the first available values in the dataset (for the year 2000) 
   stor_materials_interpol.loc[idx[storage_materials.columns.levels[0].max(),:],cat] = storage_materials.loc[:, idx[storage_materials.columns.levels[0].max(),cat]].to_numpy() # set the last year (2100) values to the last available values in the dataset (for the year 2050) 
   stor_materials_interpol.loc[idx[:,:],cat] = stor_materials_interpol.loc[idx[:,:],cat].unstack().astype('float64').interpolate().stack()

#############
# Lifetimes #
#############

# First the lifetime of storage technologies needs to be defined over time, before running the dynamic stock function
# before 2018
for year in reversed(range(YEAR_START,storage_start)):
    # storage_lifetime_interpol = pd.concat([storage_lifetime_interpol, pd.Series(storage_lifetime_interpol.loc[storage_lifetime_interpol.first_valid_index()], name=year)])
    row = pd.DataFrame([storage_lifetime_interpol.loc[storage_lifetime_interpol.first_valid_index()]])
    storage_lifetime_interpol.loc[year] = row.iloc[0]
# after 2030
for year in range(2030+1,YEAR_OUT+1):
    # storage_lifetime_interpol = pd.concat([storage_lifetime_interpol, pd.Series(storage_lifetime_interpol.loc[storage_lifetime_interpol.last_valid_index()], name=year)])
    row = pd.DataFrame([storage_lifetime_interpol.loc[storage_lifetime_interpol.last_valid_index()]])
    storage_lifetime_interpol.loc[year] = row.iloc[0]

storage_lifetime_interpol = storage_lifetime_interpol.sort_index(axis=0)
# drop the PHS from the interpolated lifetime frame, as the PHS is calculated separately
storage_lifetime_interpol = storage_lifetime_interpol.drop(columns=['PHS'])

#################
# Market Shares #
#################

# Determine MARKET SHARE of the storage capacity using a multi-nomial logit function

# storage costs ---
# determine the annual % decline of the costs based on the 2018-2030 data (original, before applying the malus)
decline = ((storage_costs_interpol.loc[storage_start,:]-storage_costs_interpol.loc[storage_end,:])/(storage_end-storage_start))/storage_costs_interpol.loc[storage_start,:]
decline_used = decline*storage_ltdecline #TODO: what is happening here? Why?
# storage_ltdecline is a single number and should describe the long-term decline after 2030 relative to the 2018-2030 decline

storage_costs_new = storage_costs_interpol * storage_malus_interpol
# calculate the development from 2030 to 2050 (using annual price decline)
for year in range(storage_end+1,2050+1):
    # print(year)
    # storage_costs_new = storage_costs_new.append(pd.Series(storage_costs_new.loc[storage_costs_new.last_valid_index()]*(1-decline_used), name=year))
    # storage_costs_new = pd.concat([storage_costs_new, pd.Series(storage_costs_new.loc[storage_costs_new.last_valid_index()] * (1 - decline_used), name=year)])
    row = pd.DataFrame([storage_costs_new.loc[storage_costs_new.last_valid_index()] * (1 - decline_used)])
    storage_costs_new.loc[year] = row.iloc[0]
# for historic price development, assume 2x AVERAGE annual price decline on all technologies, except lead-acid (so that lead-acid gets a relative price advantage from 1970-2018)
for year in reversed(range(YEAR_START,storage_start)):
    # storage_costs_new = storage_costs_new.append(pd.Series(storage_costs_new.loc[storage_costs_new.first_valid_index()]*(1+(2*decline_used.mean())), name=year)).sort_index(axis=0)
    row = pd.DataFrame([storage_costs_new.loc[storage_costs_new.first_valid_index()]*(1+(2*decline_used.mean()))])
    storage_costs_new.loc[year] = row.iloc[0]

storage_costs_new.sort_index(axis=0, inplace=True) 
storage_costs_new.loc[1971:2017,'Deep-cycle Lead-Acid'] = storage_costs_new.loc[2018,'Deep-cycle Lead-Acid'] # restore the exception (set to constant 2018 values)


# market shares ---
# use the storage price development in the logit model to get market shares
storage_market_share = MNLogit(storage_costs_new, -0.2) #assumes input of an ordered dataframe with rows as years and columns as technologies, values as prices. Logitpar is the calibrated Logit parameter (usually a nagetive number between 0 and 1)

# fix the market share of storage technologies after 2050
for year in range(2050+1,YEAR_OUT+1):
    # storage_market_share = storage_market_share.append(pd.Series(storage_market_share.loc[storage_market_share.last_valid_index()], name=year))
    row = pd.DataFrame([storage_market_share.loc[storage_market_share.last_valid_index()]])
    storage_market_share.loc[year] = row.iloc[0]

# total = storage_market_share.sum(axis=1)
region_list = list(kilometrage.columns.values)   
storage.columns = region_list

#----------------------------------------------------------------------------------------------------------
###########################################################################################################
#%% 2.3) Vehicles
###########################################################################################################
#----------------------------------------------------------------------------------------------------------

# Vehicle specific variables

# kilometrage is defined until 2008, fill 2008 values until 2100 
kilometrage = kilometrage.reindex(list(range(YEAR_START,YEAR_END))).interpolate(limit_direction='both')
region_list = list(kilometrage.columns.values)                          # get a list with region names

loadfactor = loadfactor_data[['time','DIM_1', 5]].pivot_table(index='time', columns='DIM_1') # loadfactor for cars (in persons per vehicle)
loadfactor = loadfactor.loc[list(range(YEAR_START,YEAR_END+1))] * LOAD_FACTOR  # car loadfactor is expressed compared to an average global loadfactor from the IMAGE-TIMER-TRAVEL model (original: Girod, 2012; further elaborations by Edelenbosch et al.) 
  
indexNames = passengerkms_data[ passengerkms_data['DIM_1'] >= 27 ].index
passengerkms_data.drop(indexNames , inplace=True)
passengerkms = passengerkms_data[['time','DIM_1', 5]].pivot_table(index='time', columns='DIM_1').loc[list(range(YEAR_START,YEAR_END+1))]
  
BEV_collist = [22, 23, 24, 25] # battery electric vehicles
PHEV_collist= [21, 20, 19, 18, 17, 16] # plug-in hybrid electric vehicles

vehicleshare_data['battery'] = vehicleshare_data[BEV_collist].sum(axis=1)
vehicleshare_data['PHEV'] = vehicleshare_data[PHEV_collist].sum(axis=1)

# get the regional sum of the BEV & PHEV fraction of the fleet, and replace the column names with the region names
PHEV_share = vehicleshare_data[['time','DIM_1', 'PHEV']].pivot_table(index='time',columns='DIM_1').loc[list(range(YEAR_START,YEAR_END+1))]
BEV_share = vehicleshare_data[['time','DIM_1', 'battery']].pivot_table(index='time',columns='DIM_1').loc[list(range(YEAR_START,YEAR_END+1))]

# storage.drop(storage.iloc[:, -2:], inplace = True, axis = 1)    # drop global total column and empty (27) column

# before we run any calculations, replace the column names to region names
PHEV_share.columns = region_list
BEV_share.columns = region_list
passengerkms.columns = region_list
loadfactor.columns = region_list



# BEV & PHEV vehicle stats

if SENS_ANALYSIS == 'high_stor':
   capacity_usable_PHEV = 0.025   # 2.5% of capacity of PHEV is usable as storage (in the pessimistic sensitivity variant)
   capacity_usable_BEV  = 0.05    # 5  % of capacity of BEVs is usable as storage (in the pessimistic sensitivity variant)
else: 
   capacity_usable_PHEV = 0.05    # 5% of capacity of PHEV is usable as storage
   capacity_usable_BEV  = 0.10    # 10% of capacity of BEVs is usable as storage

vehicle_kms = passengerkms.loc[:YEAR_OUT] * PKMS_TO_VKMS / loadfactor.loc[:YEAR_OUT]        # conversion from tera-Tkms  
vehicles_all = vehicle_kms / kilometrage.loc[:YEAR_OUT]

vehicles_PHEV = vehicles_all * PHEV_share.loc[:YEAR_OUT]
vehicles_BEV = vehicles_all * BEV_share.loc[:YEAR_OUT]
vehicles_EV = vehicles_PHEV + vehicles_BEV

# For the availability of vehicle store capacity we apply the assumption of fixed weight,
# So we first need to know the average density of the stock of EV batteries
# to get there we need to know the share of the technologies in the stock (based on lifetime & share in the inflow/purchases)
# to get there, we first need to know the market share of new stock additions (=inflow/purchases)

#First we calculate the share of the inflow using only a few of the technologies in the storage market share
#The selection represents only the batteries that are suitable for EV & mobile applications
EV_battery_list = ['NiMH', 'LMO', 'NMC', 'NCA', 'LFP', 'Lithium Sulfur', 'Lithium Ceramic', 'Lithium-air']
#normalize the selection of market shares, so that total market share is 1 again (taking the relative share in the selected battery techs)
market_share_EVs = pd.DataFrame().reindex_like(storage_market_share[EV_battery_list])

for year in list(range(YEAR_START,YEAR_OUT+1)):
    for tech in EV_battery_list:
        market_share_EVs.loc[year, tech] = storage_market_share[EV_battery_list].loc[year,tech] /storage_market_share[EV_battery_list].loc[year].sum()


test = market_share_EVs.sum(axis=1) # -> market share sums to 1 for each year

###########################################################################################################
#%%% 2.3.1) Prep_data File
###########################################################################################################


# ureg = pint.UnitRegistry(force_ndarray_like=True)
# # Define the units for each dimension
# unit_mapping = {
#     'time': ureg.year,
#     'year': ureg.year,
#     'Year': ureg.year,
#     'kg': ureg.kilogram,
#     'yr': ureg.year,
#     '%': ureg.percent,
#     't': ureg.tonne,
#     'MW': ureg.megawatt, #added
#     'GW': ureg.gigawatt, #added
# }

# # Conversion table for all coordinates, to be removed/adapted after input tables are fixed.
# conversion_table = {
#     "gcap_stock": (["Time"], ["Type", "Region"],),
#     "gcap_types_materials": (["Cohort"], ["Type", "material"],)
#     # "gcap_materials_interpol": (["Cohort"], ["Type", "SubType", "material"], {"Type": ["Type", "SubType"]})
# }

# # results_dict = {
# #         'total_nr_vehicles_simple': total_nr_vehicles_simple,
# #         'material_fractions_simple': material_fractions_simple,
# #         'material_fractions_typical': material_fractions_typical,
# #         'vehicle_weights_simple': vehicle_weights_simple,
# #         'vehicle_weights_typical': vehicle_weights_typical,
# #         'lifetimes': lifetimes_vehicles,
# #         'battery_weights_typical': battery_weights_typical,
# #         'battery_materials': battery_materials,
# #         'battery_shares': battery_shares,
# #         'weight_boats': weight_boats,
# #         'vehicle_shares_typical': vehicle_shares_typical
# #     }

# results_dict = {
#         'gcap_stock': gcap_stock,
#         'gcap_types_materials': gcap_types_materials,
#         'gcap_lifetime_distr': gcap_lifetime_distr,
# }

# # df = gcap_materials_interpol.copy()


# # Convert the DataFrames to xarray Datasets and apply units
# preprocessing_results_xarray = {}


# for df_name, df in results_dict.items():
#     if df_name in conversion_table:
#         data_xar_dataset = pandas_to_xarray(df, unit_mapping)
#         data_xarray = dataset_to_array(data_xar_dataset, *conversion_table[df_name])
#     else:
#         # lifetimes_vehicles does not need to be converted in the same way.
#         data_xarray = pandas_to_xarray(df, unit_mapping)
#     preprocessing_results_xarray[df_name] = data_xarray



# preprocessing_results_xarray["lifetimes"] = convert_life_time_vehicles(preprocessing_results_xarray["gcap_lifetime_distr"])
# preprocessing_results_xarray["stocks"] = preprocessing_results_xarray.pop("gcap_stock")
# preprocessing_results_xarray["material_intensities"] = preprocessing_results_xarray.pop("gcap_types_materials")
# # preprocessing_results_xarray["shares"] = preprocessing_results_xarray.pop("vehicle_shares")




###########################################################################################################
#%%% 2.3.2) Stock Modelling
###########################################################################################################
# TODO: move this to electricity.py
import scipy

def stock_share_calc(stock, market_share, init_tech, techlist, storage_lifetime_interpol):

    # Here, we define the market share of the stock based on a pre-calculation with several steps: 
    # 1) use Global total stock development, the market shares of the inflow and technology specific lifetimes to derive 
    # the shares in the stock, assuming that pre-1990 100% of the stock was determined by Lead-acid batteries. 
    # 2) Then, apply the global stock-related market shares to disaggregate the stock to technologies in all regions 
    # (assuming that battery markets are global markets)
    # As the development of the total stock of dedicated electricity storage is known, but we don't know the inflow, 
    # and only the market share related to the inflow we need to calculate the inflow first. 

    # first we define the survival of the 1990 stock (assumed 100% Lead-acid, for each cohort in 1990)
    pre_time = 20                           # the years required for the pre-calculation of the Lead-acid stock
    cohorts = YEAR_OUT-YEAR_SWITCH            # nr. of cohorts in the stock calculations (after YEAR_SWITCH)
    timeframe = np.arange(0,cohorts+1)      # timeframe for the pre-calcluation
    pre_year_list = list(range(YEAR_SWITCH-pre_time,YEAR_SWITCH+1))   # list of years to pre-calculate the Lead-acid stock for

    #define new dataframes
    stock_cohorts = pd.DataFrame(index=pd.MultiIndex.from_product([stock.columns,range(YEAR_SWITCH,YEAR_OUT+1)]), columns=pd.MultiIndex.from_product([techlist, range(YEAR_SWITCH-pre_time,YEAR_END+1)])) # year, by year, by tech
    inflow_by_tech = pd.DataFrame(index=pd.MultiIndex.from_product([stock.columns,range(YEAR_SWITCH,YEAR_OUT+1)]), columns=techlist)
    outflow_cohorts = pd.DataFrame(index=pd.MultiIndex.from_product([stock.columns,range(YEAR_SWITCH,YEAR_OUT+1)]), columns=pd.MultiIndex.from_product([techlist, range(YEAR_SWITCH-pre_time,YEAR_END+1)])) # year by year by tech
    
    #specify lifetime & other settings
    stdev_mult = 0.214
    mean = storage_lifetime_interpol.loc[YEAR_SWITCH,init_tech] # select the mean lifetime for Lead-acid batteries in 1990
    stdev = mean * stdev_mult               # we use thesame standard-deviation as for generation technologies, given that these apply to 'energy systems' more generally
    survival_init = scipy.stats.foldnorm.sf(timeframe, mean/stdev, 0, scale=stdev)
    techlist_new = techlist
    techlist_new.remove(init_tech)          # techlist without Lead-acid (or other init_tech)
    
    # actual inflow & outflow calculations, this bit takes long!
    # loop over regions, technologies and years to calculate the inflow, stock & outflow of storage technologies, given their share of the inflow.
    for region in stock.columns:
        
        # pre-calculate the stock by cohort of the initial stock of Lead-acid
        multiplier_pre = stock.loc[YEAR_SWITCH,region]/survival_init.sum()   # the stock is subdivided by the previous cohorts according to the survival function (only allowed when assuming steady stock inflow) 
        
        #pre-calculate the stock as lists (for efficiency)
        initial_stock_years = [np.flip(survival_init[0:pre_time+1]) * multiplier_pre]
            
        for year in range(1, (YEAR_OUT-YEAR_SWITCH)+1):      # then fill the columns with the remaining fractions
            initial_stock_years.append(initial_stock_years[0] * survival_init[year])
    
        stock_cohorts.loc[idx[region,:],idx[init_tech, list(range(YEAR_SWITCH-pre_time,YEAR_SWITCH+1))]] = initial_stock_years       # fill the stock dataframe according to the pre-calculated stock 
        outflow_cohorts.loc[idx[region,:],idx[init_tech, list(range(YEAR_SWITCH-pre_time,YEAR_SWITCH+1))]] = stock_cohorts.loc[idx[region,:],idx[init_tech, list(range(YEAR_SWITCH-pre_time,YEAR_SWITCH+1))]].shift(1, axis=0) - stock_cohorts.loc[idx[region,:],idx[init_tech, list(range(YEAR_SWITCH-pre_time,YEAR_SWITCH+1))]]
    
        # set the other stock cohorts to zero
        stock_cohorts.loc[idx[region,:],idx[techlist_new, pre_year_list]] = 0
        outflow_cohorts.loc[idx[region,:],idx[techlist_new, pre_year_list]] = 0
        inflow_by_tech.loc[idx[region, YEAR_SWITCH], techlist_new] = 0                                                   # inflow of other technologies in 1990 = 0
        
        # except for outflow and inflow in 1990 (YEAR_SWITCH), which can be pre-calculated for Deep-cycle Lead Acid (@ steady state inflow, inflow = outflow = stock/lifetime)
        outflow_cohorts.loc[idx[region, YEAR_SWITCH], idx[init_tech,:]] = outflow_cohorts.loc[idx[region, YEAR_SWITCH+1], idx[init_tech,:]]     # given the assumption of steady state inflow (pre YEAR_SWITCH), we can determine that the outflow is the same in switchyear as in switchyear+1                                        
        inflow_by_tech.loc[idx[region, YEAR_SWITCH], init_tech] =   stock.loc[YEAR_SWITCH,region]/mean                                          # given the assumption of steady state inflow (pre YEAR_SWITCH), we can determine the inflow to be the same as the outflow at a value of stock/avg. lifetime                                                          
        
        # From YEAR_SWITCH onwards, define a stock-driven model with a known market share (by tech) of the new inflow 
        for year in range(YEAR_SWITCH+1,YEAR_OUT+1):
            
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
                survival = scipy.stats.foldnorm.sf(np.arange(0,(YEAR_OUT+1)-year), storage_lifetime_interpol.loc[year,tech]/(storage_lifetime_interpol.loc[year,tech]*0.2), 0, scale=storage_lifetime_interpol.loc[year,tech]*0.2)           
                # then apply the survival to the inflow in current cohort, both the inflow & the survival are entered into the stock_cohort dataframe in 1 step
                stock_cohorts_list.append(inflow_by_tech.loc[idx[region,year],tech]  *  survival)
                
            stock_cohorts.loc[idx[region,list(range(year,YEAR_OUT+1))],idx[:,year]] = list(map(list, zip(*stock_cohorts_list)))        
        
    # separate the outflow (by cohort) calculation (separate shift calculation for each region & tech is MUCH more efficient than including it in additional loop over years)
    # calculate the outflow by cohort based on the stock by cohort that was just calculated
    for region in stock.columns:
        for tech in inflow_by_tech.columns:
            outflow_cohorts.loc[idx[region,:],idx[tech,:]] = stock_cohorts.loc[idx[region,:],idx[tech,:]].shift(1,axis=0) - stock_cohorts.loc[idx[region,:],idx[tech,:]]

    return inflow_by_tech, stock_cohorts, outflow_cohorts

import time

start_time = time.time()

# then we use that market share in combination with the stock developments to derive the stock share 
# Here we use the vehcile stock (number of cars) as a proxy for the development of the battery stock (given that we're calculating the actual battery stock still, and just need to account for the dynamics of purchases to derive te stock share here) 
EV_inflow_by_tech, EV_stock_cohorts, EV_outflow_cohorts = stock_share_calc(vehicles_EV, market_share_EVs, 'NiMH', ['NiMH', 'LMO', 'NMC', 'NCA', 'LFP', 'Lithium Sulfur', 'Lithium Ceramic ', 'Lithium-air'], storage_lifetime_interpol)
# takes ~ 1-2 min to run
end_time = time.time()
print(f"Execution time: {end_time - start_time:.4f} seconds")


###########################################################################################################
#%%% 2.3.3) Postprocess EV
###########################################################################################################
#

EV_stock =  EV_stock_cohorts.T.groupby(level=0).sum().T   # sum(level) is and groupby(axis) will be deprecated -> transose first with .T (instead of specifying axis), then groupby level, then sum. To get intial shape back, transpose again with .T
EV_storage_stock_abs  = EV_stock.groupby(level=1).sum()   # sum over all regions to get the global share of the stock
EV_storage_inflow_abs = EV_inflow_by_tech.groupby(level=1).sum()  # sum over all regions to get the global share of the inflow

# Calc. global share of different battery technologies (stock & inflow)
EV_storage_stock_share  = pd.DataFrame(index=EV_storage_stock_abs.index,  columns=EV_storage_stock_abs.columns)
EV_storage_inflow_share = pd.DataFrame(index=EV_storage_inflow_abs.index, columns=EV_storage_inflow_abs.columns)
for tech in EV_storage_stock_abs.columns:
    EV_storage_stock_share.loc[:,tech]  = EV_storage_stock_abs.loc[:,tech].div(EV_storage_stock_abs.sum(axis=1))
    EV_storage_inflow_share.loc[:,tech] = EV_storage_inflow_abs.loc[:,tech].div(EV_storage_inflow_abs.sum(axis=1))

# EV_storage_stock_share.to_csv(path_base / 'imagematerials' / 'electricity' / 'out_test'  / 'battery_share_stock.csv') # Average global car battery share (in stock) is exported to be used in paper on vehicles
# EV_storage_inflow_share.to_csv(path_base / 'imagematerials' / 'electricity' / 'out_test'  / 'battery_share_inflow.csv') # Average global car battery share (in inflow) is exported to be used in paper on vehicles
  
#The global share of the battery technologies in stock is then used to derive the (weihgted) average density (kg/kWh)
weighted_average_density_stock  = EV_storage_stock_share.mul(storage_density_interpol[EV_battery_list]).sum(axis=1)
weighted_average_density_inflow = EV_storage_inflow_share.mul(storage_density_interpol[EV_battery_list]).sum(axis=1)

# weighted_average_density_stock.loc[:YEAR_OUT].to_csv(path_base / 'imagematerials' / 'electricity' / 'out_test'  / 'ev_battery_density_stock.csv')        # Average car battery density (in stock) is exported to be used in paper on vehicles
# weighted_average_density_inflow.loc[:YEAR_OUT].to_csv(path_base / 'imagematerials' / 'electricity' / 'out_test'  / 'ev_battery_density_inflow.csv')      # Average car battery density (in inflow) is exported to be used in paper on vehicles

# assumed fixed energy densities before 1990 (=NiMH)
add = pd.Series(weighted_average_density_stock[weighted_average_density_stock.first_valid_index()], index=list(range(YEAR_START,1990)))
weighted_average_density = pd.concat([weighted_average_density_stock, add]).sort_index(axis=0)

# With a pre-determined battery capacity in 2018, we assume an increasing capacity (as an effect of an increased density) based on a fixed weight assumption
BEV_dynamic_capacity = (weighted_average_density[2018] * BEV_CAPACITY_CURRENT) / weighted_average_density
PHEV_dynamic_capacity = (weighted_average_density[2018] * PHEV_CAPACITY_CURRENT) / weighted_average_density

# Define the V2G settings over time (assuming slow adoption to the maximum usable capacity)
year_v2g_start = 2025-1 # -1 leads to usable capacity in 2025
year_full_capacity = 2040
v2g_usable = pd.Series(index=list(range(YEAR_START,YEAR_OUT+1)), name='years')    #fraction of the cars ready and willing to use v2g
for year in list(range(YEAR_START,YEAR_OUT+1)):
    if (year <= year_v2g_start):
        v2g_usable[year] = 0
    elif (year >= year_full_capacity):
        v2g_usable[year] = 1
v2g_usable = v2g_usable.interpolate()

#total capacity per car connected to v2g (in kWh)
max_capacity_BEV = v2g_usable * capacity_usable_BEV  
max_capacity_PHEV = v2g_usable * capacity_usable_PHEV   

#usable capacity per car connected to v2g (in kWh)
usable_capacity_BEV = max_capacity_BEV.mul(BEV_dynamic_capacity[:YEAR_OUT])     
usable_capacity_PHEV = max_capacity_PHEV.mul(PHEV_dynamic_capacity[:YEAR_OUT])  

# Vehicle storage in MWh
storage_BEV = storage_PHEV = pd.DataFrame().reindex_like(vehicles_BEV)
for region in region_list:
    storage_PHEV[region] = vehicles_PHEV[region].mul(usable_capacity_PHEV) /1000
    storage_BEV[region] = vehicles_BEV[region].mul(usable_capacity_BEV) /1000

storage_vehicles = storage_BEV + storage_PHEV


###########################################################################################################
#%% 2.4) Hydro Power & Other Storage
###########################################################################################################

# OPTION: NO V2G ---------------------------------------------------------------
storage_vehicles = pd.DataFrame(0, index=storage.index, columns=storage.columns)  # set vehicle storage to zero when not using V2G
storage_vehicles = storage_vehicles.loc[:YEAR_OUT]
#-------------------------------------------------------------------------------

# Take the TIMER Hydro-dam capacity (MW) & compare it to Pumped hydro capacity (MW) projections from the International Hydropower Association

Gcap_hydro = gcap_data[['time','DIM_1', 7]].pivot_table(index='time', columns='DIM_1')   # IMAGE-TIMER Hydro dam capacity (power, in MW)
Gcap_hydro = Gcap_hydro.iloc[:, :26]
region_list = list(kilometrage.columns.values)  # get a list with region names
Gcap_hydro.columns = region_list
Gcap_hydro = Gcap_hydro.loc[:YEAR_OUT]

#storage capacity in MW (power capacity), to compare it to Pumped hydro storage projections (also given in MW, power capacity)              
# storage_power.drop(storage_power.iloc[:, -2:], inplace = True, axis = 1) # error prone
storage_power = storage_power.iloc[:, :26]  
storage_power.columns = region_list
storage_power = storage_power.loc[:YEAR_OUT]

#Disaggregate the Pumped hydro-storgae projections to 26 IMAGE regions according to the relative Hydro-dam power capacity (also MW) within 5 regions reported by the IHA (international Hydropwer Association)
phs_regions = [[10,11],[19],[1],[22],[0,2,3,4,5,6,7,8,9,12,13,14,15,16,17,18,20,21,23,24,25]]   # subregions in IHA data for Europe, China, US, Japan, RoW, MIND: region refers to IMAGE region MINUS 1
phs_projections_IMAGE = pd.DataFrame(index=Gcap_hydro.index, columns=Gcap_hydro.columns)        # empty dataframe

for column in range(0,len(phs_regions)):
    sum_data = Gcap_hydro.iloc[:,phs_regions[column]].sum(axis=1) # first, get the sum of all hydropower in the IHA regions (to divide over in second step)
    for region in range(0,REGIONS):
        if region in phs_regions[column]:
            phs_projections_IMAGE.iloc[:,region] = phs_projections.iloc[:,column] * (Gcap_hydro.iloc[:,region]/sum_data) 
            # J: allocate share of the phs_projections to each IMAGE region based on the share of that region on the generation capacity of the IHA region it is part of
            # J: (Gcap_hydro.iloc[:,region]/sum_data) is between 0 and 1, so the phs_projections are disaggregated to IMAGE regions

# Then fill the years after 2030 (end of IHA projections) according to the Gcap annual growth rate (assuming a fixed percentage of Hydro dams will be built with Pumped hydro capabilities after )
if SENS_ANALYSIS == 'high_stor':
   phs_projections_IMAGE.loc[2030:YEAR_OUT] =  phs_projections_IMAGE.loc[2030] * (Gcap_hydro.loc[2030:YEAR_OUT]/Gcap_hydro.loc[2030:YEAR_OUT])  # no growth after 2030 in the high_stor sensitivity variant
else:
   phs_projections_IMAGE.loc[2030:YEAR_OUT] =  phs_projections_IMAGE.loc[2030] * (Gcap_hydro.loc[2030:YEAR_OUT]/Gcap_hydro.loc[2030])

# Calculate the fractions of the storage capacity that is provided through pumped hydro-storage, electric vehicles or other storage (larger than 1 means the capacity superseeds the demand for energy storage, in terms of power in MW or enery in MWh) 
phs_storage_fraction = phs_projections_IMAGE.divide(storage_power.loc[:YEAR_OUT]).clip(upper=1) # the phs storage fraction deployed to fulfill storage demand, both phs & storage_power here are expressed in MW
storage_remaining = storage.loc[:YEAR_OUT] * (1 - phs_storage_fraction)

if SENS_ANALYSIS == 'high_stor':
   oth_storage_fraction = 0.5 * storage_remaining 
   oth_storage_fraction += ((storage_remaining * 0.5) - storage_vehicles).clip(lower=0)    
   oth_storage_fraction = oth_storage_fraction.divide(storage).where(oth_storage_fraction > 0, 0).clip(lower=0) 
   evs_storage_fraction = 1 - (phs_storage_fraction + oth_storage_fraction)     # electric vehicle storage (BEV + PHEV) capacity and total storage demand are expressed as MWh
else: 
   oth_storage_fraction = (storage_remaining - storage_vehicles).clip(lower=0)    
   oth_storage_fraction = oth_storage_fraction.divide(storage.loc[:YEAR_OUT]).where(oth_storage_fraction > 0, 0).clip(lower=0)      
   evs_storage_fraction = 1 - (phs_storage_fraction + oth_storage_fraction)     # electric vehicle storage (BEV + PHEV) capacity and total storage demand are expressed as MWh
   
checksum = phs_storage_fraction + evs_storage_fraction + oth_storage_fraction   # should be 1 for all fields

# absolute storage capacity (MWh)
phs_storage_theoretical = phs_projections_IMAGE.divide(storage_power) * storage.loc[:YEAR_OUT] # ??? theoretically available PHS storage (MWh; fraction * total) only used in the graphs that show surplus capacity
phs_storage = phs_storage_fraction * storage.loc[:YEAR_OUT]
evs_storage = evs_storage_fraction * storage.loc[:YEAR_OUT]
oth_storage = oth_storage_fraction * storage.loc[:YEAR_OUT]

#output for Main text figure 2 (storage reservoir, in MWh for 3 storage types)
storage_out_phs = pd.concat([phs_storage], keys=['phs'], names=['type']) 
storage_out_evs = pd.concat([evs_storage], keys=['evs'], names=['type']) 
storage_out_oth = pd.concat([oth_storage], keys=['oth'], names=['type']) 
storage_out = pd.concat([storage_out_phs, storage_out_evs, storage_out_oth])
# storage_out.to_csv(path_base / 'imagematerials' / 'electricity' / 'out_test'  / 'storage_by_type_MWh.csv')        # in MWh

# derive inflow & outflow (in MWh) for PHS, for later use in the material calculations 
PHS_kg_perkWh = 26.8   # kg per kWh storage capacity (as weight addition to existing hydro plants to make them pumped) 
phs_storage_stock_tail = stock_tail(phs_storage.astype(float), YEAR_OUT)
storage_lifetime_PHS = storage_lifetime['PHS'].reindex(list(range(YEAR_FIRST_GRID,YEAR_OUT+1)), axis=0).interpolate(limit_direction='both')

#%%%% TESTS TESTS --------------------------------------------------------

# Test PHS interpolation pre-1971
regions_to_plot = ["China", "Brazil", "US"]  # adjust to your available columns
df = phs_storage_stock_tail.loc[:1980,:]  # replace with your actual variable
fig, ax = plt.subplots(figsize=(10, 6))
for region in regions_to_plot:
    if region in df.columns:
        ax.plot(df.index[:-1], df[region].iloc[:-1], label=region)  # up to before last timestep
        # ax.plot(phs_storage.index[:-1], phs_storage[region].iloc[:-1], label=region)
ax.set_xlabel("Year")
ax.set_ylabel("Value")
ax.set_title("Comparison of Regions")
ax.legend()
ax.grid(alpha=0.3, linestyle="--")
plt.show()


# Storage tier comparisons
region_colors = {
    "China": "blue",
    "Brazil": "green",
    "Mexico": "orange"
}
scenario_linestyles = {
    "PHS": "-",
    "V2G": "--",
    "other storage": ":"
}

# Plot 1 absolute----------------------------
dfs = [phs_storage, evs_storage, oth_storage]
names = ["PHS", "V2G", "other storage"]  # labels for legend

regions_to_plot = ["China", "Brazil", "Mexico"]

fig, ax = plt.subplots(figsize=(10, 6))
for df, name in zip(dfs, names):
    # Ensure numeric (your data is object type)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.iloc[:-1]  # drop last timestep
    
    for region in regions_to_plot:
        if region in df.columns:
            ax.plot(df.index, df[region],
                    label=f"{region} - {name}",
                    color=region_colors[region],
                    linestyle=scenario_linestyles[name])

ax.set_xlabel("Year")
ax.set_ylabel("Value")
ax.set_title("Comparison of Regions Across Scenarios")
ax.legend()
ax.grid(alpha=0.3, linestyle="--")
plt.legend()
plt.show()

# Plot 1 relative----------------------------
dfs = [phs_storage_fraction, evs_storage_fraction, oth_storage_fraction]
names = ["PHS", "V2G", "other storage"]  # labels for legend

regions_to_plot = ["China", "Brazil", "Mexico"]

fig, ax = plt.subplots(figsize=(10, 6))
for df, name in zip(dfs, names):
    # Ensure numeric (your data is object type)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.iloc[:-1]  # drop last timestep
    
    for region in regions_to_plot:
        if region in df.columns:
            ax.plot(df.index, df[region],
                    label=f"{region} - {name}",
                    color=region_colors[region],
                    linestyle=scenario_linestyles[name])

ax.set_xlabel("Year")
ax.set_ylabel("Value")
ax.set_title("Comparison of Regions Across Scenarios")
ax.legend()
ax.grid(alpha=0.3, linestyle="--")
plt.legend()
plt.show()

###########################################################################################################
#%%% 2.4.1) Prep_data File
###########################################################################################################


# PHS -----------------------------------------------------------------------------------------------------

phs_stock = phs_storage_stock_tail.copy() #it was oth_storage.copy() Hä?

# Bring dataframes into correct shape for the results_dict

# stocks: years as index and regions as columns -> years as index and (technology, region) as columns
# Current columns are regions
regions = phs_stock.columns.tolist()

# Create a MultiIndex with technology "PHS" for all columns
multi_cols = pd.MultiIndex.from_tuples([("PHS", r) for r in regions], names=["technology", "region"])
# Assign to the DataFrame
phs_stock.columns = multi_cols


# lifetimes
df_mean = storage_lifetime_PHS.copy().to_frame()
df_stdev = df_mean * STD_LIFETIMES_ELECTR
df_mean.columns = [(col, 'mean') for col in df_mean.columns] # Rename columns to multi-level tuples
df_stdev.columns = [(col, 'stdev') for col in df_stdev.columns]
phs_lifetime_distr = pd.concat([df_mean, df_stdev], axis=1) # Concatenate along columns
phs_lifetime_distr.index.name = 'Year'

# MIs: (years, material) index and technologies as columns -> years as index and (technology, Material) as columns
phs_materials = stor_materials_interpol.loc[idx[:,:],'PHS'].unstack() * PHS_kg_perkWh * 1000 # wt% * kg/kWh * 1000 kWh/MWh = kg/MWh
# Current columns are materials
materials = phs_materials.columns.tolist()
# Create a MultiIndex with technology "PHS" for all columns
multi_cols = pd.MultiIndex.from_tuples([("PHS", m) for m in materials], names=["technology", "material"])
# Assign to the DataFrame
phs_materials.columns = multi_cols


# Conversion table for all coordinates, to be removed/adapted after input tables are fixed.
conversion_table = {
    "phs_stock": (["Time"], ["Type", "Region"],),
    "phs_materials": (["Cohort"], ["Type", "material"],)
    # "gcap_materials_interpol": (["Cohort"], ["Type", "SubType", "material"], {"Type": ["Type", "SubType"]})
}

results_dict = {
        'phs_stock': phs_stock,
        'phs_materials': phs_materials,
        'phs_lifetime_distr': phs_lifetime_distr,
}


prep_data_phs = create_prep_data(results_dict, conversion_table, unit_mapping)
prep_data_phs["stocks"] = prism.Q_(prep_data_phs["stocks"], "MWh")
prep_data_phs["material_intensities"] = prism.Q_(prep_data_phs["material_intensities"], "kg/MWh")
prep_data_phs["set_unit_flexible"] = prism.U_(prep_data_phs["stocks"]) # prism.U_ gives the unit back


# Other storage--------------------------------------------------------------------------------------------

oth_storage_stock = oth_storage.copy()

# Bring dataframes into correct shape for the results_dict

# stocks: (years, regions) index and technologies as columns -> years as index and (technology, region) as columns
# Current columns are regions
regions = oth_storage.columns.tolist()
# stor_tech = list(storage_lifetime_interpol.columns)
# Create a MultiIndex with technology "Other Storage" for all columns
multi_cols = pd.MultiIndex.from_tuples([("Other Storage", r) for r in regions], names=["technology", "region"])
# multi_cols = pd.MultiIndex.from_product([stor_tech, regions], names=["technology", "region"])
# Assign to the DataFrame
oth_storage_stock.columns = multi_cols


# lifetimes
df_mean = storage_lifetime_interpol.copy() #.to_frame()
df_stdev = df_mean * STD_LIFETIMES_ELECTR
df_mean.columns = [(col, 'mean') for col in df_mean.columns] # Rename columns to multi-level tuples
df_stdev.columns = [(col, 'stdev') for col in df_stdev.columns]
oth_storage_lifetime_distr = pd.concat([df_mean, df_stdev], axis=1) # Concatenate along columns
oth_storage_lifetime_distr.index.name = 'Year'

# MIs: (years, material) index and technologies as columns -> years as index and (technology, Material) as columns
stor_tech = list(storage_lifetime_interpol.columns)
# 2nd level of the MultiIndex are materials (1971,'Aluminium')
oth_storage_materials = stor_materials_interpol.copy()
oth_storage_materials.index.names = ["year", "material"]  # assign names
oth_storage_materials = oth_storage_materials.reset_index(level=["year", "material"])
# Pivot so we get (technology, material) as columns, years as index
oth_storage_materials = oth_storage_materials.melt(id_vars=["year", "material"], var_name="technology", value_name="value")
oth_storage_materials = oth_storage_materials.pivot_table(index="year", columns=["technology", "material"], values="value")
# Ensure proper MultiIndex column names
oth_storage_materials.columns = pd.MultiIndex.from_tuples(oth_storage_materials.columns, names=["technology", "material"])
xr_oth_storage_materials = dataset_to_array(pandas_to_xarray(oth_storage_materials, unit_mapping), *(["Cohort"], ["Type", "material"],))
xr_storage_density_interpol = dataset_to_array(pandas_to_xarray(storage_density_interpol, unit_mapping), *(["Cohort"], ["Type"],))
oth_storage_materialintens = xr_oth_storage_materials * xr_storage_density_interpol



# Conversion table for all coordinates, to be removed/adapted after input tables are fixed.
# conversion_table = {
#     "oth_storage_stock": (["Time"], ["Type", "Region"],),
#     "oth_storage_materials": (["Cohort"], ["Type", "material"],), #SubType
#     "oth_storage_shares": (["Cohort"], ["Type",]) #SubType
# }
conversion_table = {
    "oth_storage_stock": (["Time"], ["SuperType", "Region"],),
    "oth_storage_materials": (["Cohort"], ["Type", "material"],), #SubType
    "oth_storage_shares": (["Cohort"], ["Type",]) #SubType
}
## "gcap_materials_interpol": (["Cohort"], ["Type", "SubType", "material"], {"Type": ["Type", "SubType"]})

results_dict = {
        'oth_storage_stock': oth_storage_stock,
        'oth_storage_materials': oth_storage_materialintens.sel(Cohort=slice(1971, None)),
        'oth_storage_lifetime_distr': oth_storage_lifetime_distr,
        'oth_storage_shares': storage_market_share
}


prep_data_oth_storage = create_prep_data(results_dict, conversion_table, unit_mapping)
prep_data_oth_storage["stocks"] = prism.Q_(prep_data_oth_storage["stocks"], "MWh")
prep_data_oth_storage["material_intensities"] = prism.Q_(prep_data_oth_storage["material_intensities"], "kg/kWh")
prep_data_oth_storage["shares"] = prism.Q_(prep_data_oth_storage["shares"], "share")
prep_data_oth_storage["set_unit_flexible"] = prism.U_(prep_data_oth_storage["stocks"]) # prism.U_ gives the unit back


# TESTS TESTS --------------------------------------------------------


total = storage_market_share.sum(axis=1)
df = storage_market_share
plt.figure(figsize=(12, 8))
for column in df.columns:
    plt.plot(df.index, df[column], label=column, linewidth=2)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Values', fontsize=12)
plt.title('Technology Trends Over Time', fontsize=14, fontweight='bold')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


###########################################################################################################
#%%% 2.4.2) Stock Modelling
###########################################################################################################
# TODO: move this to electricity.py

# OLD OLD OLD ==========================================================================
# Next step: stock modelling
phs_storage_inflow, phs_storage_outflow, phs_storage_stock  = inflow_outflow(phs_storage_stock_tail, storage_lifetime_PHS, stor_materials_interpol.loc[idx[:,:],'PHS'].unstack() * PHS_kg_perkWh * 1000, 'PHS')    # PHS lifetime is fixed at 60 yrs anyway so, we simply select 1 value

inflow_by_tech, stock_cohorts, outflow_cohorts = stock_share_calc(oth_storage, storage_market_share, 'Deep-cycle Lead-Acid', list(storage_lifetime_interpol.columns),storage_lifetime_interpol) # run the function that calculates stock shares from total stock & inflow shares
# stock, market_share, init_tech, techlist, storage_lifetime_interpol


# TESTS TESTS --------------------------------------------------------

stock_cohorts.index.names = ["Region", "Time"]
stock_cohorts.columns.names = ["Type", "Cohort"]
df_stacked = stock_cohorts.stack([0, 1])  # stack the columns levels
da = xr.DataArray(df_stacked) #convert to xarray
da = da.unstack()  
da = da.transpose("Time", "Region", "Type", "Cohort")

c = 1992
t=2000
a_stocks = prep_data_oth_storage["stocks"].loc[t]
a_shares = prep_data_oth_storage["shares"]
knowledge_graph = create_electricity_graph()
a_stock_diff = knowledge_graph.aggregate_sum(da.loc[t].sum("Cohort"), a_stocks.coords["Type"].values, dim="Type")
a_inflow_tech = knowledge_graph.rebroadcast_xarray(a_stock_diff, da.coords["Type"].values, dim="Type", shares=a_shares.sel(Cohort=t))
# this is working -> theoretically the SharesInflowStocks model should work as well?


from imagematerials.survival import ScipySurvival, SurvivalMatrix
a_lifetimes = prep_data_oth_storage["lifetimes"]
survival = ScipySurvival(a_lifetimes, a_shares.coords["Type"], knowledge_graph=knowledge_graph) #self.stocks.coords["Type"],
survival_matrix = SurvivalMatrix(survival)
b_inflow_tech = xr.where(a_inflow_tech>0, a_inflow_tech/survival_matrix[t, c].drop("Cohort"), 0)
a = a_inflow_tech/survival_matrix[t, c].drop("Cohort")

a_MI = prep_data_oth_storage["material_intensities"]

# REGION = prism.Dimension("Region")
# aa = prism.Coords[REGION]

# =======================================================================================



# PHS =======================================================================================

# # Define the complete timeline, including historic tail
time_start = prep_data_phs["stocks"].coords["Time"].min().values
complete_timeline = prism.Timeline(time_start, YEAR_END, 1)
simulation_timeline = prism.Timeline(YEAR_START, YEAR_END, 1) #1970

sec_electr_stor_phs = Sector("electr_stor_phs", prep_data_phs)

main_model_factory_phs = ModelFactory(
    sec_electr_stor_phs, complete_timeline
    ).add(GenericStocks
    ).add(MaterialIntensities
    ).finish()

main_model_factory_phs.simulate(simulation_timeline)
list(main_model_factory_phs.electr_stor_phs)



# Other Storage ==============================================================================
from imagematerials.model import GenericMainModel, GenericStocks, SharesInflowStocks, Maintenance, GenericMaterials, MaterialIntensities #, MaterialIntensitiesTEST

time_start = prep_data_oth_storage["stocks"].coords["Time"].min().values
complete_timeline = prism.Timeline(time_start, YEAR_END, 1)
simulation_timeline = prism.Timeline(YEAR_START, YEAR_END, 1) #1970

sec_electr_stor_oth = Sector("electr_stor_oth", prep_data_oth_storage, check_coordinates=False)

main_model_factory_oth = ModelFactory(
    sec_electr_stor_oth, complete_timeline
    ).add(SharesInflowStocks
    ).add(MaterialIntensities
    ).finish()

main_model_factory_oth.simulate(simulation_timeline)
list(main_model_factory_oth.electr_stor_oth)



path_test_plots = Path(path_base, "imagematerials", "electricity", "out_test", scen_folder, "Figures")

###########################################################################################################
#%%% 2.4.3) Material calculations
###########################################################################################################


# apply the dedicated electricity storage market shares to the demand for dedicated (other) storage to find storage by type and by region
# stock = stock_cohorts.loc[idx[:,:],idx[:,:]].sum(axis=1, level=0)
# storage_stock_abs = stock.sum(axis=0, level=1) 
stock =  stock_cohorts.T.groupby(level=0).sum().T  # sum(level) is and groupby(axis) will be deprecated -> transose first with .T (instead of specifying axis), then groupby level, then sum. To get intial shape back, transpose again with .T
storage_stock_abs  = stock.groupby(level=1).sum()  # sum over all regions to get the global share of the stock
stock_total = stock.sum(axis=1).unstack(level=0)   # total stock by region
storage_stock_share = pd.DataFrame(index=storage_stock_abs.index, columns=storage_stock_abs.columns)

for tech in storage_stock_abs.columns:
    storage_stock_share.loc[:,tech] = storage_stock_abs.loc[:,tech].div(storage_stock_abs.sum(axis=1))

# then, the sum of the global inflow & outflow is calculated for comparison in the figures below
# outflow = outflow_cohorts.loc[idx[:,:],idx[:,:]].sum(axis=1, level=0)
outflow =  outflow_cohorts.T.groupby(level=0).sum().T # sum(level) is and groupby(axis) will be deprecated -> transose first with .T (instead of specifying axis), then groupby level, then sum. To get intial shape back, transpose again with .T
outflow_total = outflow.sum(axis=1).unstack(level=0)  # total by regions

inflow_total = inflow_by_tech.sum(axis=1).unstack(level=0)


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
      i_storage_weight.loc[idx[region,:],:] = storage_density_interpol.mul(inflow_multiplier).values * 1000  # * 1000 because density is in kg/kWh and storage is in MWh
      for tech in storage_density_interpol.columns:
         i_storage_materials.loc[idx[region,:],idx[tech,material]] = i_storage_weight.loc[idx[region,:],tech].mul(stor_materials_interpol.loc[idx[list(range(1990,endyear+1)),material],tech].to_numpy()) / 1000000 # kg to kt

# calculate the weight of the stock and the outflow (in kg)
# this one's tricky, the outflow and stock are calculated by year AND cohort, because the total weight is represented by the sum of the weight of each cohort. 
# As the density (kg/kWh) is different for every cohort (battery weight changes over time, older batteries are heavier), the total outflow FOR EACH YEAR is the 
# sumproduct of the density and the outflow by cohort.
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




###########################################################################################################
#%%% 2.4.3) Visulalization PHS
###########################################################################################################

#%%%% Sum & Per TECH - World

# da_x = main_model_factory.inflow.to_array().sum('Type')
da_stocks = main_model_factory_phs.stocks#.to_array()

data_all = da_stocks.sel(Type=da_stocks.Type != '<EMPTY>')
data_all_1 = data_all.sum('Region')

data_all_1 = data_all_1.sel(Time=slice(1971, None))

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
linewidth = 2
s_legend = 12
s_label = 14

# Second subplot: data_all_2 (summed over Region and Type, likely only time and material left)
data_all_1.plot(ax=axes, color='black', linewidth=linewidth)
axes.grid(alpha=0.3, linestyle='--')
axes.tick_params(axis='both', which='major', labelsize=s_legend)
axes.set_xlabel(" ", fontsize=s_label)
axes.set_ylabel("Pumped Hydro Storage (MWh)", fontsize=s_label)
axes.set_title('Sum over Region')

# # First subplot: data_all_1 (summed over Region, still over Type and time likely)
# for t in data_all_1.Type.values:
#     data_plot = data_all_1.sel(Type=t)
#     color, ls = DICT_GENTECH_STYLES.get(t, ('black', '-'))
#     axes[1].plot(data_plot.Time, data_plot.values, label=t, color=color, linestyle=ls, linewidth=linewidth)
#     axes[1].grid(alpha=0.3, linestyle='--')
#     axes[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
#     axes[1].tick_params(axis='both', which='major', labelsize=s_legend)
#     axes[1].set_xlabel("Time", fontsize=s_label)
#     axes[1].set_ylabel("Peak Capacity (MW)", fontsize=s_label)
#     axes[1].legend(fontsize='small', ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.2))
#     axes[1].set_title('Sum over Region')

plt.suptitle(f"{scen_folder}: Generation - Stocks: Peak Capacity (MW)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to make room for the suptitle
# fig.savefig(path_test_plots / f"Gen_inflow_world.png", dpi=300, bbox_inches='tight')
# fig.savefig(path_test_plots / f"Gen_stocks_world_1971.png", dpi=300, bbox_inches='tight')
plt.show()


#================================================================================
#%%%% SUM over TECHs - world


da_stocks_mat = main_model_factory_phs.stock_by_cohort_materials.copy() #stock_by_cohort_materials

data_all = da_stocks_mat
# data_all = data_all.sel(Type=data_all.Type != '<EMPTY>').sum('Type')
# data_all = data_all.sel(Type=data_all.Type != '<EMPTY>').sum('Region')

# Pick desired regions by name
regions = ["Brazil", "C.Europe", "China"] 

types_level1 = [m for m in data_all.material.values if m in ["Steel", "Concrete"]]
types_level2 = [m for m in data_all.material.values if m in ["Aluminium", "Cu"]]
types_level3 = [m for m in data_all.material.values if m not in (types_level1 + types_level2)]

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 12), sharex=True)
linewidth = 2
s_legend = 12
s_label = 14

data_plot = data_all.sel(Time=slice(1971, None)).pint.to("t") # convert grams to tonnes

for i, region in enumerate(regions):
    # Top row: Level 1 materials
    for mat in types_level1:
        if (region in data_plot.Region.values) and (mat in data_plot.material.values):
            data_plot.sel(material=mat, Region=region).plot(ax=axes[0, i], label=mat, color=DICT_MATERIALS_COLORS[mat], linewidth=linewidth)
    axes[0, i].set_title(f"{region}")
    axes[0, i].set_xlabel(" ")
    axes[0, i].set_ylabel(" ")
    axes[0, i].grid(alpha=0.3, linestyle='--')
    axes[0, i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
    axes[0, i].tick_params(axis='both', which='major', labelsize=s_legend) # set font size of axis ticks
    axes[0, 2].legend(loc='upper left', fontsize=s_legend)

    # Middle row: Level 2 materials
    for mat in types_level2:
        if (region in data_plot.Region.values) and (mat in data_plot.material.values):
            data_plot.sel(material=mat, Region=region).plot(ax=axes[1, i], label=mat, color=DICT_MATERIALS_COLORS[mat], linewidth=linewidth)
    axes[1, i].set_title(f" ")
    axes[1, i].set_xlabel(" ")
    axes[1, i].set_ylabel(" ")
    axes[1, i].grid(alpha=0.3, linestyle='--')
    axes[1, i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
    axes[1, i].tick_params(axis='both', which='major', labelsize=s_legend) # set font size of axis ticks
    axes[1, 2].legend(loc='upper left', fontsize=s_legend)

    # Bottom row: Level 3 materials
    for mat in types_level3:
        if (region in data_plot.Region.values) and (mat in data_plot.material.values):
            data_plot.sel(material=mat, Region=region).plot(ax=axes[2, i], label=mat, color=DICT_MATERIALS_COLORS[mat], linewidth=linewidth)
    axes[2, i].set_title(f" ")
    axes[2, i].set_xlabel("Time", fontsize=s_label)
    axes[2, i].set_ylabel(" ")
    axes[2, i].grid(alpha=0.3, linestyle='--')
    axes[2, i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
    axes[2, i].tick_params(axis='both', which='major', labelsize=s_legend) # set font size of axis ticks
    axes[2, 2].legend(loc='upper left', fontsize=s_legend)
    

axes[0, 0].set_ylabel("Material inflow (t)", fontsize=s_label)
axes[1, 0].set_ylabel("Material inflow (t)", fontsize=s_label)
axes[2, 0].set_ylabel("Material inflow (t)", fontsize=s_label)

plt.suptitle(f"{scen_folder}: Generation - Stocks Materials", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to make room for the suptitle
region_str = "_".join(regions)
# fig.savefig(path_test_plots / f"Gen_stocks-materials_{region_str}_1971.png", dpi=300)
# fig.savefig(path_test_plots / f"Gen_stocks-materials_{region_str}_1971.svg", dpi=300)
plt.show()


#================================================================================
#%%%% SUM over TECHs - world

da_stocks_mat = main_model_factory_phs.stock_by_cohort_materials.copy()
data_all = da_stocks_mat.sel(Type=da_stocks_mat.Type != '<EMPTY>').sum('Type').sum('Region')
data_plot = data_all.sel(Time=slice(1971, None)) / 1_000_000  # grams → tonnes

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 8))
linewidth = 2
s_legend = 12
s_label = 14

# Upper row: Copper
for mat in ["Cu"]:
    if mat in data_plot.material.values:
        data_plot.sel(material=mat).plot(ax=axes[0], label=mat, color=DICT_MATERIALS_COLORS.get(mat, None), linewidth=linewidth)
# axes[0].set_title("Copper stocks", fontsize=s_label)
axes[0].set_ylabel("Material stocks (t)", fontsize=s_label)
axes[0].grid(alpha=0.3, linestyle='--')
axes[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
axes[0].tick_params(axis='both', which='major', labelsize=s_legend)
axes[0].legend(loc='upper left', fontsize=s_legend)

# Lower row: Aluminium and Steel
for mat in ["Steel"]:
    if mat in data_plot.material.values:
        data_plot.sel(material=mat).plot(ax=axes[1], label=mat, color=DICT_MATERIALS_COLORS.get(mat, None), linewidth=linewidth)
# axes[1].set_title("Aluminium & Steel stocks", fontsize=s_label)
axes[1].set_xlabel("Time", fontsize=s_label)
axes[1].set_ylabel("Material stocks (t)", fontsize=s_label)
axes[1].grid(alpha=0.3, linestyle='--')
axes[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
axes[1].tick_params(axis='both', which='major', labelsize=s_legend)
axes[1].legend(loc='upper left', fontsize=s_legend)

for mat in ["Aluminium"]:
    if mat in data_plot.material.values:
        data_plot.sel(material=mat).plot(ax=axes[2], label=mat, color=DICT_MATERIALS_COLORS.get(mat, None), linewidth=linewidth)
# axes[2].set_title("Aluminium & Steel stocks", fontsize=s_label)
axes[2].set_xlabel("Time", fontsize=s_label)
axes[2].set_ylabel("Material stocks (t)", fontsize=s_label)
axes[2].grid(alpha=0.3, linestyle='--')
axes[2].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
axes[2].tick_params(axis='both', which='major', labelsize=s_legend)
axes[2].legend(loc='upper left', fontsize=s_legend)

plt.tight_layout()
plt.show()



###########################################################################################################
#%%% 2.4.3) Visulalization Other STorage
###########################################################################################################

import itertools
# Grid Storage technologies
technologies = [
    "Flywheel", "Compressed Air", "Hydrogen FC", "NiMH", "Deep-cycle Lead-Acid", "LMO",
    "NMC", "NCA", "LFP", "LTO", "Zinc-Bromide", "Vanadium Redox", "Sodium-Sulfur", "ZEBRA",
    "Lithium Sulfur", "Lithium Ceramic", "Lithium-air"
]
# Define color and linestyle pools
colors = plt.get_cmap('tab20').colors  # 20 distinct colors
linestyles = ['-', '--', ':'] #'-.'
# Create a cycle of (color, linestyle) combinations
style_combinations = list(itertools.product(colors, linestyles))
# Map technologies to (color, linestyle)
DICT_STOR_STYLES = {tech: style_combinations[i] for i, tech in enumerate(technologies)}


#================================================================================
#%%%% STOCKS - Per TECH - per Region

da_inflow = main_model_factory_oth.stock_by_cohort.sum('Cohort')
data_all = da_inflow.sel(Type=da_inflow.Type != '<EMPTY>').sel(Time=slice(2000, None), Region=regions)

da_inflow_phs = main_model_factory_phs.stock_by_cohort.sum('Cohort').sel(Type='PHS', Time=slice(2000, None), Region=regions)

regions = ['Brazil', 'C.Europe', 'China'] 
types = data_all.coords['Type'].values
mid = len(types)//2
techs_upper = list(types[:mid])
techs_lower = list(types[mid:])

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 10))
linewidth = 2
s_legend = 12
s_label = 14

for i, region in enumerate(regions):
    col = i

    # Top row: Types 1–mid
    for t in techs_upper:
        data_plot = data_all.sel(Type=t, Region=region)
        color, ls = DICT_STOR_STYLES.get(t, ('black', '-'))
        axes[0, col].plot(data_plot.Time, data_plot.values, label=t, color=color, linestyle=ls, linewidth=linewidth)
    axes[0, col].set_title(f"{region}", fontsize=15)
    axes[0, col].grid(alpha=0.3, linestyle='--')
    axes[0, col].tick_params(axis='both', which='major', labelsize=s_legend)
    axes[0, 0].set_ylabel("Stocks (MWh)", fontsize=s_label)

    # Middle row: Types mid–end
    for t in techs_lower:
        data_plot = data_all.sel(Type=t, Region=region)
        color, ls = DICT_STOR_STYLES.get(t, ('black', '-'))
        axes[1, col].plot(data_plot.Time, data_plot.values, label=t, color=color, linestyle=ls, linewidth=linewidth)
    axes[1, col].grid(alpha=0.3, linestyle='--')
    axes[1, col].tick_params(axis='both', which='major', labelsize=s_legend)
    axes[1, 0].set_ylabel("Stocks (MWh)", fontsize=s_label)
    # axes[1, col].set_xlabel("Time", fontsize=s_label)

    # Bottom row: PHS inflow
    data_plot = da_inflow_phs.sel(Region=region)
    axes[2, col].plot(data_plot.Time, data_plot.values, color='black', linewidth=linewidth, label='PHS')
    axes[2, col].grid(alpha=0.3, linestyle='--')
    axes[2, col].tick_params(axis='both', which='major', labelsize=s_legend)
    axes[2, 0].set_ylabel("Stocks (MWh)", fontsize=s_label)
    axes[2, col].set_xlabel("Time", fontsize=s_label)

# Format y-axis with commas
for ax in axes.flat:
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

# Legends
axes[0, 2].legend(fontsize=s_legend, ncol=2, loc='upper center', bbox_to_anchor=(-1.8, -2.6))
axes[1, 2].legend(fontsize=s_legend, ncol=3, loc='upper center', bbox_to_anchor=(-0.35, -1.4))
axes[2, 2].legend(fontsize=s_legend, loc='upper center', bbox_to_anchor=(0.6, -0.21))

plt.suptitle(f"{scen_folder}: Storage - Stocks: Capacity (MWh)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.98])
region_str = "_".join(regions)
fig.savefig(path_test_plots / f"Stor_stocks_{region_str}.png", dpi=300, bbox_inches='tight')
plt.show()




# # da_x = main_model_factory.inflow.to_array().sum('Type')
# da_inflow = main_model_factory_oth.stock_by_cohort.sum('Cohort') #.to_array()

# data_all = da_inflow
# data_all = data_all.sel(Type=data_all.Type != '<EMPTY>')

# data_all = data_all.sel(Time=slice(2000, None), Region=regions)

# regions = ['Brazil', 'C.Europe', 'China'] 
# types = data_all.coords['Type'].values
# mid = len(types)// 2 
# techs_upper = list(types[:mid])
# techs_lower = list(types[mid:])


# fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 8))  # Now 3 columns for 3 regions
# linewidth = 2
# s_legend = 12
# s_label = 14

# for i, region in enumerate(regions):  # regions now has length 3
#     col = i  # Column index: 0, 1, 2

#     # Top row: Types 1–15
#     for t in techs_upper:
#         data_plot = data_all.sel(Type=t, Region=region)
#         color, ls = DICT_STOR_STYLES.get(t, ('black', '-'))
#         axes[0, col].plot(data_plot.Time, data_plot.values, label=t, color=color, linestyle=ls, linewidth=linewidth)
#     axes[0, col].set_title(f"{region}", fontsize=15)
#     axes[0, col].grid(alpha=0.3, linestyle='--')
#     axes[0, col].tick_params(axis='both', which='major', labelsize=s_legend)
#     axes[0, 0].set_ylabel("Stocks (MWh)", fontsize=s_label)

#     # Bottom row: Types 16–30
#     for t in techs_lower:
#         data_plot = data_all.sel(Type=t, Region=region)
#         color, ls = DICT_STOR_STYLES.get(t, ('black', '-'))
#         axes[1, col].plot(data_plot.Time, data_plot.values, label=t, color=color, linestyle=ls, linewidth=linewidth)
#     axes[1, col].grid(alpha=0.3, linestyle='--')
#     axes[1, col].tick_params(axis='both', which='major', labelsize=s_legend)
#     axes[1, 0].set_ylabel("Stocks (MWh)", fontsize=s_label)
#     axes[1, col].set_xlabel("Time", fontsize=s_label)

# # Y-axis number formatting and hiding right y-axis ticks
# for ax in axes.flat:
#     ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

# axes[0, 2].legend(fontsize=s_legend, ncol=2, loc='upper center', bbox_to_anchor=(-1.7, -1.41))
# axes[1, 2].legend(fontsize=s_legend, ncol=3, loc='upper center', bbox_to_anchor=(-0.2, -0.21))

# plt.suptitle(f"{scen_folder}: Storage - Stocks: Capacity (MWh)", fontsize=16)
# plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to make room for the suptitle
# region_str = "_".join(regions)
# # fig.savefig(path_test_plots / f"Grid_othstor_stocks_{region_str}.png", dpi=300, bbox_inches='tight')
# # fig.savefig(path_test_plots / f"Gen_inflow_{region_str}_1971.png", dpi=300, bbox_inches='tight')
# plt.show()



#================================================================================
#%%%% STOCKS - Bar plot: storage capacity per storage type 

data_oth = main_model_factory_oth.stock_by_cohort.copy()
data_phs = main_model_factory_phs.stock_by_cohort.copy()

storage_subtypes_categories = ["mechanical storage", "lithium batteries", "molten salt and flow batteries", "other"]
knowledge_graph = create_electricity_graph()
data_oth_cat = knowledge_graph.aggregate_sum(data_oth, storage_subtypes_categories, dim="Type")

data_all = xr.concat([data_oth_cat, data_phs], dim='Type')

data_all = data_all.sel(Type=data_all.Type != '<EMPTY>', Time=slice(1990, None)).pint.to("TWh") # only from 1971 onwards, convert grams to tonnes
data_all = data_all.sum('Region')

years = [2015, 2050] #, 2100
data_plot = data_all.sum(dim="Cohort")
data_plot = data_plot.to_pandas().T
data_plot = data_plot.loc[years, :].copy()
data_plot["Other Storage"] = data_plot.drop(columns=["PHS"]).sum(axis=1)
data_plot = data_plot[["PHS", "Other Storage"]]


fig, ax = plt.subplots(figsize=(8, 6))
s_legend = 14
s_label = 16
data_plot.plot(kind="bar", stacked=True, ax=ax, color=["#6F4126", "#F69B58"])
# Labels and formatting
ax.set_xlabel("Year", fontsize=s_label)
ax.set_ylabel("Storage stock (TWh)", fontsize=s_label)
ax.set_title(f"{scen_folder}: Storage by type", fontsize=s_label)
ax.legend(fontsize=s_legend, loc="upper left")
# ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
ax.tick_params(axis='both', which='major', labelsize=s_label)

# fig.savefig(path_test_plots / "Stor_stock-category_barplot1.png", dpi=300)
# fig.savefig(path_test_plots / "Stor_stock-category_barplot2.png", dpi=300)
plt.tight_layout()
plt.show()


#%%%% STOCKS Sum & Per TECH - World ------------------------------------------------------

# da_x = main_model_factory.inflow.to_array().sum('Type')
da_stocks = main_model_factory_oth.stocks#.to_array()

data_all = da_stocks.sel(Type=da_stocks.Type != '<EMPTY>')
data_all_1 = data_all.sum('Region')

data_all_1 = data_all_1.sel(Time=slice(1971, None))

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
linewidth = 2
s_legend = 12
s_label = 14

# Second subplot: data_all_2 (summed over Region and Type, likely only time and material left)
data_all_1.plot(ax=axes, color='black', linewidth=linewidth)
axes.grid(alpha=0.3, linestyle='--')
axes.tick_params(axis='both', which='major', labelsize=s_legend)
axes.set_xlabel(" ", fontsize=s_label)
axes.set_ylabel("Other Storage (MWh)", fontsize=s_label)
axes.set_title('Sum over Region')

# # First subplot: data_all_1 (summed over Region, still over Type and time likely)
# for t in data_all_1.Type.values:
#     data_plot = data_all_1.sel(Type=t)
#     color, ls = DICT_GENTECH_STYLES.get(t, ('black', '-'))
#     axes[1].plot(data_plot.Time, data_plot.values, label=t, color=color, linestyle=ls, linewidth=linewidth)
#     axes[1].grid(alpha=0.3, linestyle='--')
#     axes[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
#     axes[1].tick_params(axis='both', which='major', labelsize=s_legend)
#     axes[1].set_xlabel("Time", fontsize=s_label)
#     axes[1].set_ylabel("Peak Capacity (MW)", fontsize=s_label)
#     axes[1].legend(fontsize='small', ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.2))
#     axes[1].set_title('Sum over Region')

plt.suptitle(f"{scen_folder}: Generation - Stocks: Peak Capacity (MW)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to make room for the suptitle
# fig.savefig(path_test_plots / f"Gen_inflow_world.png", dpi=300, bbox_inches='tight')
# fig.savefig(path_test_plots / f"Gen_stocks_world_1971.png", dpi=300, bbox_inches='tight')
plt.show()




#================================================================================
#%%%% INFLOW - Per TECH - per Region

# da_x = main_model_factory.inflow.to_array().sum('Type')
da_inflow = main_model_factory_oth.inflow.to_array()
da_inflow_phs = main_model_factory_phs.inflow.to_array()

regions = ['Brazil', 'C.Europe', 'China'] 
# data_all = xr.concat([da_inflow, da_inflow_phs], dim='Type')
data_all = da_inflow.sel(Type=da_inflow.Type != '<EMPTY>', time=slice(2000, None), Region=regions)
da_inflow_phs = da_inflow_phs.sel(Type='PHS', time=slice(2000, None), Region=regions)

types = data_all.coords['Type'].values
mid = len(types)// 2 
techs_upper = list(types[:mid])
techs_lower = list(types[mid:])


fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 8))  # Now 3 columns for 3 regions
linewidth = 2
s_legend = 12
s_label = 14

for i, region in enumerate(regions):  # regions now has length 3
    col = i  # Column index: 0, 1, 2

    # Top row: Types 1–15
    for t in techs_upper:
        data_plot = data_all.sel(Type=t, Region=region)
        color, ls = DICT_STOR_STYLES.get(t, ('black', '-'))
        axes[0, col].plot(data_plot.time, data_plot.values, label=t, color=color, linestyle=ls, linewidth=linewidth)
    axes[0, col].set_title(f"{region}", fontsize=15)
    axes[0, col].grid(alpha=0.3, linestyle='--')
    axes[0, col].tick_params(axis='both', which='major', labelsize=s_legend)
    axes[0, 0].set_ylabel("Inflow (MWh)", fontsize=s_label)

    # Bottom row: Types 16–30
    for t in techs_lower:
        data_plot = data_all.sel(Type=t, Region=region)
        color, ls = DICT_STOR_STYLES.get(t, ('black', '-'))
        axes[1, col].plot(data_plot.time, data_plot.values, label=t, color=color, linestyle=ls, linewidth=linewidth)
    axes[1, col].grid(alpha=0.3, linestyle='--')
    axes[1, col].tick_params(axis='both', which='major', labelsize=s_legend)
    axes[1, 0].set_ylabel("Inflow (MWh)", fontsize=s_label)
    # axes[1, col].set_xlabel("Time", fontsize=s_label)

    #3rd row: PHS inflow
    data_plot = da_inflow_phs.sel(Region=region)
    axes[2, col].plot(data_plot.time, data_plot.values, color='black', linewidth=linewidth,label="PHS")
    axes[2, col].grid(alpha=0.3, linestyle='--')
    axes[2, col].tick_params(axis='both', which='major', labelsize=s_legend)
    axes[2, 0].set_ylabel("Inflow (MWh)", fontsize=s_label)
    axes[2, col].set_xlabel("Time", fontsize=s_label)

# Y-axis number formatting and hiding right y-axis ticks
for ax in axes.flat:
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

# 2 rows
# axes[0, 2].legend(fontsize=s_legend, ncol=2, loc='upper center', bbox_to_anchor=(-1.7, -1.41))
# axes[1, 2].legend(fontsize=s_legend, ncol=3, loc='upper center', bbox_to_anchor=(-0.2, -0.21))
# 3 rows
axes[0, 2].legend(fontsize=s_legend, ncol=2, loc='upper center', bbox_to_anchor=(-1.9, -2.8))
axes[1, 2].legend(fontsize=s_legend, ncol=3, loc='upper center', bbox_to_anchor=(-0.4, -1.7))
axes[2, 2].legend(fontsize=s_legend, ncol=3, loc='upper center', bbox_to_anchor=(0.6, -0.5))

plt.suptitle(f"{scen_folder}: Storage - Inflow: Capacity (MWh)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to make room for the suptitle
region_str = "_".join(regions)
fig.savefig(path_test_plots / f"Stor_inflow_{region_str}.png", dpi=300, bbox_inches='tight')
plt.show()


#================================================================================
#%%%% INFLOW - Per TECH - World

# da_x = main_model_factory.inflow.to_array().sum('Type')
da_inflow = main_model_factory_oth.inflow.to_array()
da_inflow_phs = main_model_factory_phs.inflow.to_array()

data_all = da_inflow.sel(Type=da_inflow.Type != '<EMPTY>', time=slice(2000, None)).sum('Region')
da_inflow_phs = da_inflow_phs.sel(Type='PHS', time=slice(2000, None)).sum('Region')

types = data_all.coords['Type'].values


fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 10))
linewidth = 2
s_legend = 12
s_label = 14

# Top row: All technologies
for t in types:
    data_plot = data_all.sel(Type=t)
    color, ls = DICT_STOR_STYLES.get(t, ('black', '-'))
    axes[0].plot(data_plot.time, data_plot.values, label=t, color=color, linestyle=ls, linewidth=linewidth)
axes[0].set_title("World", fontsize=15)
axes[0].grid(alpha=0.3, linestyle='--')
axes[0].tick_params(axis='both', which='major', labelsize=s_legend)
axes[0].set_ylabel("Inflow (MWh)", fontsize=s_label)

# Bottom row: PHS inflow
data_plot = da_inflow_phs
axes[1].plot(data_plot.time, data_plot.values, color='black', linewidth=linewidth, label='PHS')
axes[1].grid(alpha=0.3, linestyle='--')
axes[1].tick_params(axis='both', which='major', labelsize=s_legend)
axes[1].set_ylabel("Inflow (MWh)", fontsize=s_label)
axes[1].set_xlabel("Time", fontsize=s_label)

# Format and legends
for ax in axes.flat:
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

axes[0].legend(fontsize=s_legend, ncol=3, loc='upper center', bbox_to_anchor=(0.4, -1.8))
axes[1].legend(fontsize=s_legend, loc='upper center', bbox_to_anchor=(0.9, -0.7))

plt.suptitle(f"{scen_folder}: Storage - Inflow: Capacity (MWh)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.98])
fig.savefig(path_test_plots / f"Stor_inflow_world.png", dpi=300, bbox_inches='tight')
plt.show()



#================================================================================
#%%%  MATERIALS
#================================================================================

#================================================================================
#%%%% MATERIAL STOCKS - Per TECH category - world - STACKED

DICT_STOR_CATEGORY_COLORS = {
    'mechanical storage':              "#FBBF09",
    'PHS':                             "#4BABFF",
    'lithium batteries':               "#42DD88",
    'molten salt and flow batteries':  "#ffb3c1",
    'other':                           '#BBB8B9'
}

DICT_STOR_CATEGORY_COLORS_SEBASTIAAN = {
    'mechanical storage':              "#AC501A",
    'PHS':                             "#6F4126",
    'lithium batteries':               "#E95E0D",
    'molten salt and flow batteries':  "#F07C32",
    'other':                           '#F69B58'
}

data_oth = main_model_factory_oth.stock_by_cohort_materials.copy()
data_phs = main_model_factory_phs.stock_by_cohort_materials.copy()

storage_subtypes_categories = ["mechanical storage", "lithium batteries", "molten salt and flow batteries", "other"]
knowledge_graph = create_electricity_graph()
data_oth_cat = knowledge_graph.aggregate_sum(data_oth, storage_subtypes_categories, dim="Type")

data_all = xr.concat([data_oth_cat, data_phs], dim='Type')

t_end = 2100
data_all = data_all.sel(Type=data_all.Type != '<EMPTY>', Time=slice(1990, t_end)).pint.to("kilotonne") # only from 1971 onwards, convert grams to tonnes
data_all = data_all.sum('Region')


desired_order = ['PHS', 'mechanical storage', 'lithium batteries', 'molten salt and flow batteries', 'other']

# 4 Subplots: 4 materials
materials = ['Steel', 'Aluminium', 'Nd', 'Co']
fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(14, 10))
# unit = prism.U_(data_all)
s_legend = 12
s_label = 14
for i, material in enumerate(materials):
    row = i // 2
    col = i % 2
    # Select data for this material (columns under this material)
    data_plot = data_all.sel(material = material)
    data_plot = data_plot.drop_vars('material')
    data_plot = data_plot.to_pandas().T
    data_plot = data_plot[desired_order] # Reorder columns
    colors = [DICT_STOR_CATEGORY_COLORS_SEBASTIAAN[cat] for cat in data_plot.columns] # select colors based on technology category
    data_plot.plot.area(ax=axes[row, col], stacked=True, color = colors)

    axes[row, col].set_title(material, fontsize=15)
    axes[row, col].set_ylabel(f'Material stock (kt)', fontsize=s_label) #{unit}
    handles, labels = axes[row, col].get_legend_handles_labels() # reverse the order of legend to match the stacked plot
    axes[row, col].legend(handles[::-1], labels[::-1], loc='upper left', fontsize=s_legend)

    axes[row, col].ticklabel_format(style='plain', axis='y')
    axes[row, col].get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
    axes[row, col].tick_params(axis='both', which='major', labelsize=s_legend)


for col in range(2): # Set x-labels only for bottom row
    axes[1, col].set_xlabel('Year', fontsize=s_label)

plt.suptitle(f"{scen_folder}: Storage - Stocks Materials per Tech. Cat. - World", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to make room for the suptitle
# fig.savefig(path_test_plots / f"Stor_stock-materials-techcategory_st-al-nd-co_world_{t_end}.png", dpi=300)
plt.show()

# Sebastiaans numbers taken from figures in his thesis:
# total = data_all.sum("Type")
# ratio_steel_BL = total.sel(material="Steel", Time = 2100)/7000
# ratio_aluminium_BL = total.sel(material="Aluminium", Time = 2100)/300
# ratio_neodymium_BL = total.sel(material="Nd", Time = 2100)/9
# ratio_cobalt_BL = total.sel(material="Co", Time = 2100)/1.5
# # ->steel: 109; aluminium: 287, neodymium: 273, cobalt: 271 for SSP_M_CP



# 1 Subplots: 1 material -----------------------------------------------------------------
# material = 'Steel'

# # Select data for this material
# data_plot = data_all.sel(material=material)
# data_plot = data_plot.drop_vars('material')
# data_plot = data_plot.to_pandas().T
# data_plot = data_plot[desired_order]  # Reorder columns


# fig, ax = plt.subplots(figsize=(8, 6))
# colors = [DICT_STOR_CATEGORY_COLORS_SEBASTIAAN[cat] for cat in data_plot.columns]
# data_plot.plot.area(ax=ax, stacked=True, color=colors)
# # Formatting
# ax.set_title(material, fontsize=15)
# ax.set_ylabel('Material stock (kt)', fontsize=14)
# ax.set_xlabel('Year', fontsize=14)
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles[::-1], labels[::-1], loc='upper left', fontsize=12)
# ax.ticklabel_format(style='plain', axis='y')
# ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

# plt.tight_layout()
# # fig.savefig(path_test_plots / f"Stor_stock-materials-techcategory_{material.lower()}_world_1990.png", dpi=300)
# plt.show()





#================================================================================
#%%%% MATERIAL STOCKS - SUM over TECHs - per region

# Sum over technologies dimension

da_stocks_mat = main_model_factory_oth.stock_by_cohort_materials.copy() #stock_by_cohort_materials

data_all = da_stocks_mat
data_all = data_all.sel(Type=data_all.Type != '<EMPTY>').sum('Type')
# data_all = data_all.sel(Type=data_all.Type != '<EMPTY>').sum('Region')

# Pick desired regions by name
regions = ["Brazil", "C.Europe", "China"] 

types_level1 = [m for m in data_all.material.values if m in ["Steel", "Concrete"]]
types_level2 = [m for m in data_all.material.values if m in ["Aluminium", "Cu"]]
types_level3 = [m for m in data_all.material.values if m not in (types_level1 + types_level2)]

fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(18, 12), sharex=True)
linewidth = 2
s_legend = 12
s_label = 14

data_plot = data_all.sel(Time=slice(2020, None)).pint.to("t") # convert to tonnes

for i, region in enumerate(regions):
    # Top row: Level 1 materials
    for mat in types_level1:
        if (region in data_plot.Region.values) and (mat in data_plot.material.values):
            data_plot.sel(material=mat, Region=region).plot(ax=axes[0, i], label=mat, color=DICT_MATERIALS_COLORS[mat], linewidth=linewidth)
    axes[0, i].set_title(f"{region}")
    axes[0, i].set_xlabel(" ")
    axes[0, i].set_ylabel(" ")
    axes[0, i].grid(alpha=0.3, linestyle='--')
    axes[0, i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
    axes[0, i].tick_params(axis='both', which='major', labelsize=s_legend) # set font size of axis ticks
    axes[0, 2].legend(loc='upper left', fontsize=s_legend)

    # Middle row: Level 2 materials
    for mat in types_level2:
        if (region in data_plot.Region.values) and (mat in data_plot.material.values):
            data_plot.sel(material=mat, Region=region).plot(ax=axes[1, i], label=mat, color=DICT_MATERIALS_COLORS[mat], linewidth=linewidth)
    axes[1, i].set_title(f" ")
    axes[1, i].set_xlabel(" ")
    axes[1, i].set_ylabel(" ")
    axes[1, i].grid(alpha=0.3, linestyle='--')
    axes[1, i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
    axes[1, i].tick_params(axis='both', which='major', labelsize=s_legend) # set font size of axis ticks
    axes[1, 2].legend(loc='upper left', fontsize=s_legend)

    # Bottom row: Level 3 materials
    for mat in types_level3:
        if (region in data_plot.Region.values) and (mat in data_plot.material.values):
            data_plot.sel(material=mat, Region=region).plot(ax=axes[2, i], label=mat, color=DICT_MATERIALS_COLORS[mat], linewidth=linewidth)
    axes[2, i].set_title(f" ")
    axes[2, i].set_xlabel("Time", fontsize=s_label)
    axes[2, i].set_ylabel(" ")
    axes[2, i].grid(alpha=0.3, linestyle='--')
    axes[2, i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
    axes[2, i].tick_params(axis='both', which='major', labelsize=s_legend) # set font size of axis ticks
    axes[2, 2].legend(loc='upper left', fontsize=s_legend)
    

axes[0, 0].set_ylabel("Material stocks (t)", fontsize=s_label)
axes[1, 0].set_ylabel("Material stocks (t)", fontsize=s_label)
axes[2, 0].set_ylabel("Material stocks (t)", fontsize=s_label)

plt.suptitle(f"{scen_folder}: Grid Storage - Stock Materials", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to make room for the suptitle
region_str = "_".join(regions)
# fig.savefig(path_test_plots / f"Grid_othstor_stocks-materials_{region_str}.png", dpi=300)
# fig.savefig(path_test_plots / f"Gen_stocks-materials_{region_str}_1971.svg", dpi=300)
plt.show()


#================================================================================
#%%%% MATERIAL STOCKS - SUM over TECHs - world

da_stocks_mat = main_model_factory_oth.stock_by_cohort_materials.copy() #stock_by_cohort_materials
data_all = da_stocks_mat
data_all = data_all.sel(Type=data_all.Type != '<EMPTY>').sum('Type').sum('Region')
# data_plot = data_all.pint.to("t") # convert grams to tonnes
data_plot = data_all.sel(Time=slice(1971, None)).pint.to("t") # only from 1971 onwards, convert grams to tonnes

fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 10))
linewidth = 2
s_legend = 12
s_label = 14

# Top row: Level 1 materials
for mat in types_level1:
    if mat in data_plot.material.values:
        data_plot.sel(material=mat).plot(ax=axes[0], label=mat, color=DICT_MATERIALS_COLORS[mat], linewidth=linewidth)
axes[0].set_title(f" ")
axes[0].set_xlabel(" ")
axes[0].set_ylabel("Material stocks (t)", fontsize=s_label)
axes[0].grid(alpha=0.3, linestyle='--')
axes[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
axes[0].tick_params(axis='both', which='major', labelsize=s_legend) # set font size of axis ticks
axes[0].legend(loc='upper left', fontsize=s_legend)

# Middle row: Level 2 materials
for mat in types_level2:
    if mat in data_plot.material.values:
        data_plot.sel(material=mat).plot(ax=axes[1], label=mat, color=DICT_MATERIALS_COLORS[mat], linewidth=linewidth)

axes[1].set_title(" ")
axes[1].set_xlabel(" ")
axes[1].set_ylabel("Material stocks (t)", fontsize=s_label)
axes[1].grid(alpha=0.3, linestyle='--')
axes[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
axes[1].tick_params(axis='both', which='major', labelsize=s_legend) # set font size of axis ticks
axes[1].legend(loc='upper left', fontsize=s_legend)

# Bottom row: Level 3 materials
for mat in types_level3:
    if mat in data_plot.material.values:
        data_plot.sel(material=mat).plot(ax=axes[2], label=mat, color=DICT_MATERIALS_COLORS[mat], linewidth=linewidth)
axes[2].set_title(" ")
axes[2].set_xlabel("Time", fontsize=s_label)
axes[2].set_ylabel("Material stocks (t)", fontsize=s_label)
axes[2].grid(alpha=0.3, linestyle='--')
axes[2].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
axes[2].tick_params(axis='both', which='major', labelsize=s_legend) # set font size of axis ticks
axes[2].legend(loc='upper left', fontsize=s_legend)

plt.suptitle(f"{scen_folder}: Generation - Stocks Materials - World", fontsize=16)
plt.tight_layout()
# fig.savefig(path_test_plots / "Gen_stocks-materials_world.png", dpi=300)
# fig.savefig(path_test_plots / "Gen_stocks-materials_world_1971.png", dpi=300)
# fig.savefig(path_test_plots / "Gen_stocks-materials_world_1971.pdf", dpi=300)
# fig.savefig(path_test_plots / "Gen_stocks-materials_world_1971.svg", dpi=300)
plt.show()




###########################################################################################################
###########################################################################################################
#%% 3) TRANSMISSION GRID 
###########################################################################################################
###########################################################################################################



def get_preprocessing_data_grid(base_dir: str, SCEN, VARIANT, YEAR_START, YEAR_END, YEAR_OUT): #, climate_policy_config: dict, circular_economy_config: dict

    scen_folder = SCEN + "_" + VARIANT
    path_image_output = Path(path_base, "data", "raw", "image", scen_folder, "EnergyServices")
    path_external_data_standard = Path(path_base, "data", "raw", "electricity", "standard_data")
    path_external_data_scenario = Path(path_base, "data", "raw", "electricity", scen_folder)
    
    # test if path_external_data_scenario exists and if not set to standard scenario
    if not path_external_data_scenario.exists():
        path_external_data_scenario = Path(path_base, "data", "raw", "electricity", STANDARD_SCEN_EXTERNAL_DATA)

    scen_BL_folder = SCEN + "_M_CP"  # baseline scenario
    path_image_output_BL = Path(path_base, "data", "raw", "image", scen_BL_folder, "EnergyServices")
    # TODO: check if this is necessary (shouldn't historical periode anyway be the same for all scenarios?)
    # + if it is, should the baseline scenario be given as a parameter or can it be inferred from the scenario name?

    assert path_image_output.is_dir()
    assert path_external_data_standard.is_dir()
    assert path_external_data_scenario.is_dir()

    years = YEAR_END - YEAR_START  + 1

    idx = pd.IndexSlice   

    ###########################################################################################################
    # Read in files #

    # 1. External Data --------------------------------------------- 

    grid_length_Hv = pd.read_csv(path_external_data_standard /'grid_length_Hv.csv', index_col=0, names=None).transpose()    # lenght of the High-voltage (Hv) lines in the grid, based on Open Street Map (OSM) analysis (km)
    ratio_Hv = pd.read_csv(path_external_data_standard / 'Hv_ratio.csv', index_col=0)                                        # Ratio between the length of Medium-voltage (Mv) and Low-voltage (Lv) lines in relation to Hv lines (km Lv /km Hv) & (km Mv/ km Hv)
    underground_ratio = pd.read_csv(path_external_data_standard / 'underground_ratio.csv', index_col=[0,1])           # these contain the definition of the constants in the linear function used to determine the relation between income & the percentage of underground power-lines (% underground = mult * gdp/cap + add)
    grid_additions = pd.read_csv(path_external_data_standard / 'grid_additions.csv', index_col=0)                            # Transformers & substations per km of grid (Hv, Mv & Lv, in units/km)


    # dynamic or scenario-dependent data (lifetimes & material intensity)

    lifetime_grid_elements = pd.read_csv(path_external_data_scenario  / 'operational_lifetime_grid.csv', index_col=0)         # Average lifetime in years of grid elements

    # dynamic material intensity files (kg/km or kg/unit)
    materials_grid = pd.read_csv(path_external_data_scenario / 'Materials_grid_dynamic.csv', index_col=[0,1])                # Material intensity of grid lines specific material content for Hv, Mv & Lv lines, & specific for underground vs. aboveground lines. (kg/km)
    materials_grid_additions = pd.read_csv(path_external_data_scenario / 'Materials_grid_additions.csv', index_col=[0,1])    # (not part of the SA yet) Additional infrastructure required for grid connections, such as transformers & substations (material compositin in kg/unit)

    # IMAGE file: GDP per capita (US-dollar 2005, ppp), used to derive underground-aboveground ratio based on income levels
    gdp_pc = pd.read_csv(path_external_data_scenario / 'gdp_pc.csv', index_col=0)  # TODO: check why it says this is an IMAGE file (why is it .csv?)


    # 2. IMAGE/TIMER files ---------------------------------------------

    # Generation capacity (stock & inflow/new) in MW peak capacity, FILES from TIMER
    gcap_data = read_mym_df(path_image_output / 'Gcap.out')
    # gcap_BL_data = read_mym_df('SSP2\\SSP2_BL\\Gcap.out') # baseline scenario? TODO: what is the purpose of reading in the scneario + the baseline?
    gcap_BL_data = read_mym_df(path_image_output_BL / 'Gcap.out')

    ###########################################################################################################
    # Prepare model specific variables #

    gcap_data = gcap_data.loc[~gcap_data['DIM_1'].isin([27,28])]    # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
    gcap = pd.pivot_table(gcap_data[gcap_data['time'].isin(list(range(YEAR_START,YEAR_END+1)))], index=['time','DIM_1'], values=list(range(1,TECH_GEN+1)))  #gcap as multi-index (index = years & regions (26); columns = technologies (28));  the last column in gcap_data (= totals) is now removed

    gcap_BL_data = gcap_BL_data.loc[~gcap_BL_data['DIM_1'].isin([27,28])]    # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
    gcap_BL = pd.pivot_table(gcap_BL_data[gcap_BL_data['time'].isin(list(range(YEAR_START,YEAR_END+1)))], index=['time','DIM_1'], values=list(range(1,TECH_GEN+1)))  #gcap as multi-index (index = years & regions (26); columns = technologies (28));  the last column in gcap_data (= totals) is now removed

    region_list = list(grid_length_Hv.columns.values)
    material_list = list(materials_grid.columns.values)

    # renaming multi-index dataframe: generation capacity, based on the regions in grid_length_Hv & technologies as given
    gcap_techlist = ['Solar PV', 'Solar Decentral', 'CSP', 'Wind onshore', 'Wind offshore', 'Wave', 'Hydro', 'Other Renewables', 'Geothermal', 'Hydrogen', 'Nuclear', '<EMPTY>', 'Conv. Coal', 'Conv. Oil', 'Conv. Natural Gas', 'Waste', 'IGCC', 'OGCC', 'NG CC', 'Biomass CC', 'Coal + CCS', 'Oil/Coal + CCS', 'Natural Gas + CCS', 'Biomass + CCS', 'CHP Coal', 'CHP Oil', 'CHP Natural Gas', 'CHP Biomass', 'CHP Coal + CCS', 'CHP Oil + CCS', 'CHP Natural Gas + CCS', 'CHP Biomass + CCS', 'CHP Geothermal', 'CHP Hydrogen']
    gcap.index = pd.MultiIndex.from_product([list(range(YEAR_START,YEAR_END+1)), region_list], names=['years', 'regions'])
    gcap_BL.index = pd.MultiIndex.from_product([list(range(YEAR_START,YEAR_END+1)), region_list], names=['years', 'regions'])
    gcap.columns = gcap_techlist
    gcap_BL.columns = gcap_techlist

    gdp_pc.columns = region_list
    # gdp_pc = gdp_pc.drop([1970]).drop(list(range(YEAR_END+1,YEAR_LAST+1)))


    # length calculations ----------------------------------------------------------------------------

    # only the regional total (peak) generation capacity is used as a proxy for the grid growth (BL to 2016, then BL or 450)
    gcap_BL_total = gcap_BL.sum(axis=1).unstack()
    gcap_BL_total = gcap_BL_total[region_list]               # re-order columns to the original TIMER order
    gcap_growth = gcap_BL_total / gcap_BL_total.loc[2016]    # define growth according to 2016 as base year
    gcap_total = gcap.sum(axis=1).unstack()
    gcap_total = gcap_total[region_list]                     # re-order columns to the original TIMER order
    gcap_growth.loc[2016:YEAR_END] = gcap_total.loc[2016:YEAR_END] / gcap_total.loc[2016]        # define growth according to 2016 as base year

    # in the sensitivity variant, additional growth is presumed after 2020 based on the fraction of variable renewable energy (vre) generation capacity (solar & wind)
    vre_fraction = gcap[['Solar PV', 'CSP', 'Wind onshore', 'Wind offshore']].sum(axis=1).unstack().divide(gcap.sum(axis=1).unstack())
    add_growth = vre_fraction * 1                  # 0.2 = 20% additional HV lines per doubling of vre gcap
    red_growth = (1-vre_fraction) * 0.7            # 0.2 = 20% less HV lines per doubling of baseline gcap
    add_growth.loc[list(range(1971,2020+1)),:] = 0  # pre 2020, additional HV grid growth is 0, afterwards the additional line length is gradually introduced (towards 2050)
    red_growth.loc[list(range(1971,2020+1)),:] = 0  # pre 2020, reduction of HV grid growth is 0, afterwards the line length reduction is gradually introduced (towards 2050)
    for year in range(2020,2050+1):
        add_growth.loc[year] = add_growth.loc[year] * (1/30*(year-2020)) 
        red_growth.loc[year] = red_growth.loc[year] * (1/30*(year-2020)) 

    # Hv length (in kms) is region-specific. However, we use a single ratio between the length of Hv and Mv networks, the same applies to Lv networks 
    grid_length_Mv = grid_length_Hv.mul(ratio_Hv['HV to MV'])
    grid_length_Lv = grid_length_Hv.mul(ratio_Hv['HV to LV'])

    # define grid length over time (fixed in 2016, growth according to gcap)
    grid_length_Hv_time = pd.DataFrame().reindex_like(gcap_total)
    grid_length_Mv_time = pd.DataFrame().reindex_like(gcap_total)
    grid_length_Lv_time = pd.DataFrame().reindex_like(gcap_total)

    #implement growth correction (sensitivity variant)
    if SENS_ANALYSIS == 'high_grid':
        gcap_growth_HV = gcap_growth.add(add_growth.reindex_like(gcap_growth)).subtract(red_growth.reindex_like(gcap_growth))
    else: 
        gcap_growth_HV = gcap_growth

    for year in range(YEAR_START, YEAR_END+1):
        grid_length_Hv_time.loc[year] = gcap_growth_HV.loc[year].mul(grid_length_Hv.loc['2016'])
        grid_length_Mv_time.loc[year] = gcap_growth.loc[year].mul(grid_length_Mv.loc['2016'])
        grid_length_Lv_time.loc[year] = gcap_growth.loc[year].mul(grid_length_Lv.loc['2016'])

    # define underground vs. aboveground fraction (%) based on static ratios (the Hv length is the aboveground fraction according to Open Street Maps, we add the underground fractions for 3 voltage networks)
    # Based on new insights from Kalt et al. 2021, we adjust the underground ratios downwards for non-European regions
    function_Hv_under = pd.DataFrame(index=gdp_pc.index, columns=gdp_pc.columns)
    function_Mv_under = pd.DataFrame(index=gdp_pc.index, columns=gdp_pc.columns)
    function_Lv_under = pd.DataFrame(index=gdp_pc.index, columns=gdp_pc.columns)

    for region in region_list:
        if region in ['W.Europe','C.Europe']:
            select_proxy = 'Europe'
        else:
            select_proxy = 'Other'
        #print(str(region) + ': ' + select_proxy)
        function_Hv_under[region] = gdp_pc[region] * underground_ratio.loc[idx[select_proxy,'mult'],'HV'] + underground_ratio.loc[idx[select_proxy,'add'],'HV']
        function_Mv_under[region] = gdp_pc[region] * underground_ratio.loc[idx[select_proxy,'mult'],'MV'] + underground_ratio.loc[idx[select_proxy,'add'],'MV']
        function_Lv_under[region] = gdp_pc[region] * underground_ratio.loc[idx[select_proxy,'mult'],'LV'] + underground_ratio.loc[idx[select_proxy,'add'],'LV']

    # maximize linear function at 100 & minimize at 0 (%)
    function_Hv_under = function_Hv_under.apply(lambda x: [y if y >= 0 else 0 for y in x])                     
    function_Hv_under = function_Hv_under.apply(lambda x: [y if y <= 100 else 100 for y in x])  
    function_Mv_under = function_Mv_under.apply(lambda x: [y if y >= 0 else 0 for y in x])  
    function_Mv_under = function_Mv_under.apply(lambda x: [y if y <= 100 else 100 for y in x])  
    function_Lv_under = function_Lv_under.apply(lambda x: [y if y >= 0 else 0 for y in x])  
    function_Lv_under = function_Lv_under.apply(lambda x: [y if y <= 100 else 100 for y in x])

    # MIND! the HV lines found in OSM (+national sources) are considered as the total of the aboveground line length + the underground line length
    grid_length_Hv_total = grid_length_Hv_time                                      # assuming the length from OSM IS the abovegrond fraction
    grid_length_Hv_above = grid_length_Hv_total * (1 - (function_Hv_under/100)) 
    grid_length_Hv_under = grid_length_Hv_total * function_Hv_under/100

    # out for main text figure 2
    grid_length_HV_out_a = pd.concat([grid_length_Hv_above], keys=['aboveground'], names=['type']) 
    grid_length_HV_out_u = pd.concat([grid_length_Hv_under], keys=['underground'], names=['type']) 
    grid_length_HV_out   = pd.concat([grid_length_HV_out_a, grid_length_HV_out_u])
    # grid_length_HV_out.to_csv(path_base / 'imagematerials' / 'electricity' / 'out_test' / 'grid_length_HV_km.csv') # in km

    grid_length_Mv_above = grid_length_Mv_time * (1 - function_Mv_under/100)
    grid_length_Mv_under = grid_length_Mv_time * function_Mv_under/100
    grid_length_Mv_total = grid_length_Mv_above + grid_length_Mv_under

    grid_length_Lv_above = grid_length_Lv_time * (1 - function_Lv_under/100)
    grid_length_Lv_under = grid_length_Lv_time * function_Lv_under/100
    grid_length_Lv_total = grid_length_Lv_above + grid_length_Lv_under

    grid_subst_Hv = grid_length_Hv_total.mul(grid_additions.loc['Substations','HV'])        # number of substations on HV network
    grid_subst_Mv = grid_length_Mv_total.mul(grid_additions.loc['Substations','MV'])        # # of substations
    grid_subst_Lv = grid_length_Lv_total.mul(grid_additions.loc['Substations','LV'])        # # of substations
    grid_trans_Hv = grid_length_Hv_total.mul(grid_additions.loc['Transformers','HV'])       # number of transformers on the HV network
    grid_trans_Mv = grid_length_Mv_total.mul(grid_additions.loc['Transformers','MV'])       # # of transformers
    grid_trans_Lv = grid_length_Lv_total.mul(grid_additions.loc['Transformers','LV'])       # # of transformers


    ##################
    # Interpolations #
    ##################

    # Interpolate material intensities (dynamic content from 1926 to 2100, based on data files)
    index                             = pd.MultiIndex.from_product([list(range(YEAR_FIRST_GRID,YEAR_END+1)), list(materials_grid.index.levels[1])])
    materials_grid_interpol           = pd.DataFrame(index=index, columns=materials_grid.columns)
    materials_grid_additions_interpol = pd.DataFrame(index=pd.MultiIndex.from_product([list(range(YEAR_FIRST_GRID,YEAR_END+1)), list(materials_grid_additions.index.levels[1])]), columns=materials_grid_additions.columns)

    for cat in list(materials_grid.index.levels[1]):
        materials_grid_1st   = materials_grid.loc[idx[materials_grid.index[0][0], cat],:]
        materials_grid_interpol.loc[idx[YEAR_FIRST_GRID ,cat],:] = materials_grid_1st                # set the first year (1926) values to the first available values in the dataset (for the year 2000) 
        materials_grid_interpol.loc[idx[materials_grid.index.levels[0].min(),cat],:] = materials_grid.loc[idx[materials_grid.index.levels[0].min(),cat],:]                # set the middle year (2000) values to the first available values in the dataset (for the year 2000) 
        materials_grid_interpol.loc[idx[materials_grid.index.levels[0].max(),cat],:] = materials_grid.loc[idx[materials_grid.index.levels[0].max(),cat],:]                # set the last year (2100) values to the last available values in the dataset (for the year 2050) 
        materials_grid_interpol.loc[idx[:,cat],:] = materials_grid_interpol.loc[idx[:,cat],:].astype('float32').reindex(list(range(YEAR_FIRST_GRID,YEAR_END+1)), level=0).interpolate()

    for cat in list(materials_grid_additions.index.levels[1]):
        materials_grid_additions_1st   = materials_grid_additions.loc[idx[materials_grid_additions.index[0][0], cat],:]
        materials_grid_additions_interpol.loc[idx[YEAR_FIRST_GRID ,cat],:] = materials_grid_additions_1st          # set the first year (1926) values to the first available values in the dataset (for the year 2000) 
        materials_grid_additions_interpol.loc[idx[materials_grid_additions.index.levels[0].min(),cat],:] = materials_grid_additions.loc[idx[materials_grid_additions.index.levels[0].min(),cat],:]                # set the middle year (2000) values to the first available values in the dataset (for the year 2000) 
        materials_grid_additions_interpol.loc[idx[materials_grid_additions.index.levels[0].max(),cat],:] = materials_grid_additions.loc[idx[materials_grid_additions.index.levels[0].max(),cat],:]                # set the last year (2100) values to the last available values in the dataset (for the year 2050) 
        materials_grid_additions_interpol.loc[idx[:,cat],:] = materials_grid_additions_interpol.loc[idx[:,cat],:].astype('float32').reindex(list(range(YEAR_FIRST_GRID,YEAR_END+1)), level=0).interpolate()

    # call the stock_tail function on all lines, substations & transformers, to add historic stock tail between 1926 & 1971
    grid_length_Hv_above_new = stock_tail(grid_length_Hv_above, YEAR_OUT) # km
    grid_length_Mv_above_new = stock_tail(grid_length_Mv_above, YEAR_OUT) # km
    grid_length_Lv_above_new = stock_tail(grid_length_Lv_above, YEAR_OUT) # km
    grid_length_Hv_under_new = stock_tail(grid_length_Hv_under, YEAR_OUT) # km 
    grid_length_Mv_under_new = stock_tail(grid_length_Mv_under, YEAR_OUT) # km
    grid_length_Lv_under_new = stock_tail(grid_length_Lv_under, YEAR_OUT) # km
    grid_subst_Hv_new = stock_tail(grid_subst_Hv, YEAR_OUT)               # units
    grid_subst_Mv_new = stock_tail(grid_subst_Mv, YEAR_OUT)               # units
    grid_subst_Lv_new = stock_tail(grid_subst_Lv, YEAR_OUT)               # units
    grid_trans_Hv_new = stock_tail(grid_trans_Hv, YEAR_OUT)               # units
    grid_trans_Mv_new = stock_tail(grid_trans_Mv, YEAR_OUT)               # units
    grid_trans_Lv_new = stock_tail(grid_trans_Lv, YEAR_OUT)               # units


    #############
    # Lifetimes #
    #############

    # data only for lines, substations and transformer -> bring in knowledge_graph format: HV - Lines, MV - Lines, LV - Lines, HV - Transformers, etc.
    expanded_data = {}
    for typ in ["Lines", "Transformers", "Substations"]:
        for level in ["HV", "MV", "LV"]:
            new_col = f"{level} - {typ}"
            expanded_data[new_col] = lifetime_grid_elements[typ]
    lifetime_grid_elements = pd.DataFrame(expanded_data, index=lifetime_grid_elements.index)
    lifetime_grid_elements.rename_axis('Year', inplace=True)

    # no differentiation between HV, MV & LV lines as well as between aboveground and belowground
    # Types: lines, transformers, substations
    lifetime_grid_elements.loc[YEAR_FIRST_GRID,:]  = lifetime_grid_elements.loc[lifetime_grid_elements.first_valid_index(),:]
    lifetime_grid_elements.loc[YEAR_OUT,:]         = lifetime_grid_elements.loc[lifetime_grid_elements.last_valid_index(),:]
    lifetime_grid_elements                         = lifetime_grid_elements.reindex(list(range(YEAR_FIRST_GRID, YEAR_OUT+1))).interpolate()
    # TODO: check why lifetime for lines is interpoltaed from 2020 - 40yrs to 2050 - 48 yrs and then back to 2060 - 40 yrs -> should stay at 48 yrs?
    df_mean = lifetime_grid_elements.copy()
    df_stdev = df_mean * STD_LIFETIMES_ELECTR
    df_mean.columns = [(col, 'mean') for col in df_mean.columns] # Rename columns to multi-level tuples
    df_stdev.columns = [(col, 'stdev') for col in df_stdev.columns]
    lifetime_grid_distr = pd.concat([df_mean, df_stdev], axis=1) # Concatenate along columns


    # Materials

    # harmonize units of MI -> then materials_grid and materials_grid_additions can be merged in one dataframe
    # NOT POSSIBLE - separate stock modelling necessary

    # Grid Additions MIs ---
    # # for substations and transformer we multiply the MI (kg/unit of substation or transformer) with the number of substations or transformers per kilometer of grid length
    # materials_grid_additions_kgperkm             = materials_grid_additions_to_kgperkm(materials_grid_additions_interpol, grid_additions)
    # # material intensities: (years, tech. type) index and materials as columns -> years as index and (tech. type, materials) as columns
    # materials_grid_additions_kgperkm.index.names = ["Year", "Type"]
    # materials_grid_additions_kgperkm             = materials_grid_additions_kgperkm.unstack(level='Type')   # bring tech. type from row index to column header
    # materials_grid_additions_kgperkm.columns     = materials_grid_additions_kgperkm.columns.swaplevel(0, 1) # Swap the levels of the MultiIndex columns
    # materials_grid_additions_kgperkm             = materials_grid_additions_kgperkm.sort_index(axis=1)
    # # rename columns to match knowledge graph
    # new_level_0 = [col.replace(' ', ' - ', 1) for col in materials_grid_additions_kgperkm.columns.get_level_values(0)]
    # new_columns = pd.MultiIndex.from_arrays([
    #     new_level_0,
    #     materials_grid_additions_kgperkm.columns.get_level_values(1)
    # ], names=materials_grid_additions_kgperkm.columns.names)
    # materials_grid_additions_kgperkm.columns = new_columns

    materials_grid_additions_kgperunit            = materials_grid_additions_interpol.copy()
    # material intensities: (years, tech. type) index and materials as columns -> years as index and (tech. type, materials) as columns
    materials_grid_additions_kgperunit.index.names = ["Year", "Type"]
    materials_grid_additions_kgperunit             = materials_grid_additions_kgperunit.unstack(level='Type')   # bring tech. type from row index to column header
    materials_grid_additions_kgperunit.columns     = materials_grid_additions_kgperunit.columns.swaplevel(0, 1) # Swap the levels of the MultiIndex columns
    materials_grid_additions_kgperunit             = materials_grid_additions_kgperunit.sort_index(axis=1)
    # rename columns to match knowledge graph
    new_level_0 = [col.replace(' ', ' - ', 1) for col in materials_grid_additions_kgperunit.columns.get_level_values(0)]
    new_columns = pd.MultiIndex.from_arrays([
        new_level_0,
        materials_grid_additions_kgperunit.columns.get_level_values(1)
    ], names=materials_grid_additions_kgperunit.columns.names)
    materials_grid_additions_kgperunit.columns = new_columns

    # Grid MIs ---
    materials_grid_kgperkm              = materials_grid_interpol.copy() # copy the interpolated material intensities
    materials_grid_kgperkm.index.names  = ["Year", "Type"]
    materials_grid_kgperkm              = materials_grid_kgperkm.unstack(level='Type') # bring tech. type from row index to column header
    materials_grid_kgperkm.columns      = materials_grid_kgperkm.columns.swaplevel(0, 1) # Swap the levels of the MultiIndex columns
    materials_grid_kgperkm              = materials_grid_kgperkm.sort_index(axis=1)
    # rename columns to match knowledge graph
    new_level_0 = [col.replace(' ', ' - Lines - ', 1) for col in materials_grid_kgperkm.columns.get_level_values(0)]
    new_columns = pd.MultiIndex.from_arrays([
        new_level_0,
        materials_grid_kgperkm.columns.get_level_values(1)
    ], names=materials_grid_kgperkm.columns.names)
    materials_grid_kgperkm.columns = new_columns

    grid_dict_add = dict({
        'HV - Substations':         grid_subst_Hv_new,
        'HV - Transformers':        grid_trans_Hv_new,
        'MV - Substations':         grid_subst_Mv_new,
        'MV - Transformers':        grid_trans_Mv_new,
        'LV - Substations':         grid_subst_Lv_new,
        'LV - Transformers':        grid_trans_Lv_new
    })

    grid_dict_lines = dict({
            'HV - Lines - Overhead':    grid_length_Hv_above_new,
            'HV - Lines - Underground': grid_length_Hv_under_new,
            'MV - Lines - Overhead':    grid_length_Mv_above_new,
            'MV - Lines - Underground': grid_length_Mv_under_new,
            'LV - Lines - Overhead':    grid_length_Lv_above_new,
            'LV - Lines - Underground': grid_length_Lv_under_new,
        })
        
    grid_stock_lines = pd.concat(grid_dict_lines, axis=1) # Concatenate with keys to create MultiIndex ('Name', 'Region')
    grid_stock_lines = grid_stock_lines.sort_index(axis=1)

    grid_stock_add = pd.concat(grid_dict_add, axis=1) # Concatenate with keys to create MultiIndex ('Name', 'Region')
    grid_stock_add = grid_stock_add.sort_index(axis=1)


    ###########################################################################################################
    # Prep_data File #

    conversion_table = {
    "grid_stock_lines": (["Time"], ["Type", "Region"],),
    "materials_grid_kgperkm": (["Cohort"], ["Type", "material"],),
    "grid_stock_add": (["Time"], ["Type", "Region"],),
    "materials_grid_add_kgperunit": (["Cohort"], ["Type", "material"],),
    "grid_stock": (["Time"], ["Type", "Region"],), # TODO: delete
    "materials_grid_combined_kgperkm": (["Cohort"], ["Type", "material"],) # TODO: delete
    }

    results_dict_lines = {
            'grid_stock_lines': grid_stock_lines,
            'materials_grid_kgperkm': materials_grid_kgperkm,
            'lifetime_grid_distr': lifetime_grid_distr,
    }
    results_dict_add = {
            'grid_stock_add': grid_stock_add,
            'materials_grid_add_kgperunit': materials_grid_additions_kgperunit,
            'lifetime_grid_distr': lifetime_grid_distr,
    }

    prep_data_lines = create_prep_data(results_dict_lines, conversion_table, unit_mapping)
    prep_data_add = create_prep_data(results_dict_add, conversion_table, unit_mapping)

    return prep_data_lines, prep_data_add



#----------------------------------------------------------------------------------------------------------
###########################################################################################################
#%%% 3.1) Read in files
###########################################################################################################
#----------------------------------------------------------------------------------------------------------

YEAR_START = 1971   # start year of the simulation period
YEAR_END = 2100     # end year of the calculations
YEAR_OUT = 2100     # year of output generation = last year of reporting


# 1. External Data ======================================================================================== 

grid_length_Hv = pd.read_csv(path_external_data_standard /'grid_length_Hv.csv', index_col=0, names=None).transpose()    # lenght of the High-voltage (Hv) lines in the grid, based on Open Street Map (OSM) analysis (km)
ratio_Hv = pd.read_csv(path_external_data_standard / 'Hv_ratio.csv', index_col=0)                                        # Ratio between the length of Medium-voltage (Mv) and Low-voltage (Lv) lines in relation to Hv lines (km Lv /km Hv) & (km Mv/ km Hv)
underground_ratio = pd.read_csv(path_external_data_standard / 'underground_ratio.csv', index_col=[0,1])           # these contain the definition of the constants in the linear function used to determine the relation between income & the percentage of underground power-lines (% underground = mult * gdp/cap + add)
grid_additions = pd.read_csv(path_external_data_standard / 'grid_additions.csv', index_col=0)                            # Transformers & substations per km of grid (Hv, Mv & Lv, in units/km)


# dynamic or scenario-dependent data (lifetimes & material intensity)

lifetime_grid_elements = pd.read_csv(path_external_data_scenario  / 'operational_lifetime_grid.csv', index_col=0)         # Average lifetime in years of grid elements

# dynamic material intensity files (kg/km or kg/unit)
materials_grid = pd.read_csv(path_external_data_scenario / 'Materials_grid_dynamic.csv', index_col=[0,1])                # Material intensity of grid lines specific material content for Hv, Mv & Lv lines, & specific for underground vs. aboveground lines. (kg/km)
materials_grid_additions = pd.read_csv(path_external_data_scenario / 'Materials_grid_additions.csv', index_col=[0,1])    # (not part of the SA yet) Additional infrastructure required for grid connections, such as transformers & substations (material compositin in kg/unit)

# IMAGE file: GDP per capita (US-dollar 2005, ppp), used to derive underground-aboveground ratio based on income levels
gdp_pc = pd.read_csv(path_external_data_scenario / 'gdp_pc.csv', index_col=0)  # TODO: check why it says this is an IMAGE file (why is it .csv?)


# 2. IMAGE/TIMER files ====================================================================================

# Generation capacity (stock & inflow/new) in MW peak capacity, FILES from TIMER
gcap_data = read_mym_df(path_image_output / 'Gcap.out')
# gcap_BL_data = read_mym_df('SSP2\\SSP2_BL\\Gcap.out') # baseline scenario? TODO: what is the purpose of reading in the scneario + the baseline?
gcap_BL_data = read_mym_df(path_image_output / 'Gcap.out')


#----------------------------------------------------------------------------------------------------------
###########################################################################################################
#%%% 3.2) Prepare model specific variables
###########################################################################################################
#----------------------------------------------------------------------------------------------------------

gcap_data = gcap_data.loc[~gcap_data['DIM_1'].isin([27,28])]    # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
gcap = pd.pivot_table(gcap_data[gcap_data['time'].isin(list(range(YEAR_START,YEAR_END+1)))], index=['time','DIM_1'], values=list(range(1,TECH_GEN+1)))  #gcap as multi-index (index = years & regions (26); columns = technologies (28));  the last column in gcap_data (= totals) is now removed

gcap_BL_data = gcap_BL_data.loc[~gcap_BL_data['DIM_1'].isin([27,28])]    # exclude region 27 & 28 (empty & global total), mind that the columns represent generation technologies
gcap_BL = pd.pivot_table(gcap_BL_data[gcap_BL_data['time'].isin(list(range(YEAR_START,YEAR_END+1)))], index=['time','DIM_1'], values=list(range(1,TECH_GEN+1)))  #gcap as multi-index (index = years & regions (26); columns = technologies (28));  the last column in gcap_data (= totals) is now removed

region_list = list(grid_length_Hv.columns.values)
material_list = list(materials_grid.columns.values)

# renaming multi-index dataframe: generation capacity, based on the regions in grid_length_Hv & technologies as given
gcap_techlist = ['Solar PV', 'Solar Decentral', 'CSP', 'Wind onshore', 'Wind offshore', 'Wave', 'Hydro', 'Other Renewables', 'Geothermal', 'Hydrogen', 'Nuclear', '<EMPTY>', 'Conv. Coal', 'Conv. Oil', 'Conv. Natural Gas', 'Waste', 'IGCC', 'OGCC', 'NG CC', 'Biomass CC', 'Coal + CCS', 'Oil/Coal + CCS', 'Natural Gas + CCS', 'Biomass + CCS', 'CHP Coal', 'CHP Oil', 'CHP Natural Gas', 'CHP Biomass', 'CHP Coal + CCS', 'CHP Oil + CCS', 'CHP Natural Gas + CCS', 'CHP Biomass + CCS', 'CHP Geothermal', 'CHP Hydrogen']
gcap.index = pd.MultiIndex.from_product([list(range(YEAR_START,YEAR_END+1)), region_list], names=['years', 'regions'])
gcap_BL.index = pd.MultiIndex.from_product([list(range(YEAR_START,YEAR_END+1)), region_list], names=['years', 'regions'])
gcap.columns = gcap_techlist
gcap_BL.columns = gcap_techlist

gdp_pc.columns = region_list
# gdp_pc = gdp_pc.drop([1970]).drop(list(range(YEAR_END+1,YEAR_LAST+1)))


# length calculations ----------------------------------------------------------------------------

# only the regional total (peak) generation capacity is used as a proxy for the grid growth (BL to 2016, then BL or 450)
gcap_BL_total = gcap_BL.sum(axis=1).unstack()
gcap_BL_total = gcap_BL_total[region_list]               # re-order columns to the original TIMER order
gcap_growth = gcap_BL_total / gcap_BL_total.loc[2016]    # define growth according to 2016 as base year
gcap_total = gcap.sum(axis=1).unstack()
gcap_total = gcap_total[region_list]                     # re-order columns to the original TIMER order
gcap_growth.loc[2016:YEAR_END] = gcap_total.loc[2016:YEAR_END] / gcap_total.loc[2016]        # define growth according to 2016 as base year

# in the sensitivity variant, additional growth is presumed after 2020 based on the fraction of variable renewable energy (vre) generation capacity (solar & wind)
vre_fraction = gcap[['Solar PV', 'CSP', 'Wind onshore', 'Wind offshore']].sum(axis=1).unstack().divide(gcap.sum(axis=1).unstack())
add_growth = vre_fraction * 1                  # 0.2 = 20% additional HV lines per doubling of vre gcap
red_growth = (1-vre_fraction) * 0.7            # 0.2 = 20% less HV lines per doubling of baseline gcap
add_growth.loc[list(range(1971,2020+1)),:] = 0  # pre 2020, additional HV grid growth is 0, afterwards the additional line length is gradually introduced (towards 2050)
red_growth.loc[list(range(1971,2020+1)),:] = 0  # pre 2020, reduction of HV grid growth is 0, afterwards the line length reduction is gradually introduced (towards 2050)
for year in range(2020,2050+1):
   add_growth.loc[year] = add_growth.loc[year] * (1/30*(year-2020)) 
   red_growth.loc[year] = red_growth.loc[year] * (1/30*(year-2020)) 

# Hv length (in kms) is region-specific. However, we use a single ratio between the length of Hv and Mv networks, the same applies to Lv networks 
grid_length_Mv = grid_length_Hv.mul(ratio_Hv['HV to MV'])
grid_length_Lv = grid_length_Hv.mul(ratio_Hv['HV to LV'])

# define grid length over time (fixed in 2016, growth according to gcap)
grid_length_Hv_time = pd.DataFrame().reindex_like(gcap_total)
grid_length_Mv_time = pd.DataFrame().reindex_like(gcap_total)
grid_length_Lv_time = pd.DataFrame().reindex_like(gcap_total)

#implement growth correction (sensitivity variant)
if SENS_ANALYSIS == 'high_grid':
   gcap_growth_HV = gcap_growth.add(add_growth.reindex_like(gcap_growth)).subtract(red_growth.reindex_like(gcap_growth))
else: 
   gcap_growth_HV = gcap_growth

for year in range(YEAR_START, YEAR_END+1):
   grid_length_Hv_time.loc[year] = gcap_growth_HV.loc[year].mul(grid_length_Hv.loc['2016'])
   grid_length_Mv_time.loc[year] = gcap_growth.loc[year].mul(grid_length_Mv.loc['2016'])
   grid_length_Lv_time.loc[year] = gcap_growth.loc[year].mul(grid_length_Lv.loc['2016'])

# define underground vs. aboveground fraction (%) based on static ratios (the Hv length is the aboveground fraction according to Open Street Maps, we add the underground fractions for 3 voltage networks)
# Based on new insights from Kalt et al. 2021, we adjust the underground ratios downwards for non-European regions
function_Hv_under = pd.DataFrame(index=gdp_pc.index, columns=gdp_pc.columns)
function_Mv_under = pd.DataFrame(index=gdp_pc.index, columns=gdp_pc.columns)
function_Lv_under = pd.DataFrame(index=gdp_pc.index, columns=gdp_pc.columns)

for region in region_list:
    if region in ['W.Europe','C.Europe']:
        select_proxy = 'Europe'
    else:
        select_proxy = 'Other'
    #print(str(region) + ': ' + select_proxy)
    function_Hv_under[region] = gdp_pc[region] * underground_ratio.loc[idx[select_proxy,'mult'],'HV'] + underground_ratio.loc[idx[select_proxy,'add'],'HV']
    function_Mv_under[region] = gdp_pc[region] * underground_ratio.loc[idx[select_proxy,'mult'],'MV'] + underground_ratio.loc[idx[select_proxy,'add'],'MV']
    function_Lv_under[region] = gdp_pc[region] * underground_ratio.loc[idx[select_proxy,'mult'],'LV'] + underground_ratio.loc[idx[select_proxy,'add'],'LV']

# maximize linear function at 100 & minimize at 0 (%)
function_Hv_under = function_Hv_under.apply(lambda x: [y if y >= 0 else 0 for y in x])                     
function_Hv_under = function_Hv_under.apply(lambda x: [y if y <= 100 else 100 for y in x])  
function_Mv_under = function_Mv_under.apply(lambda x: [y if y >= 0 else 0 for y in x])  
function_Mv_under = function_Mv_under.apply(lambda x: [y if y <= 100 else 100 for y in x])  
function_Lv_under = function_Lv_under.apply(lambda x: [y if y >= 0 else 0 for y in x])  
function_Lv_under = function_Lv_under.apply(lambda x: [y if y <= 100 else 100 for y in x])

# MIND! the HV lines found in OSM (+national sources) are considered as the total of the aboveground line length + the underground line length
grid_length_Hv_total = grid_length_Hv_time                                      # assuming the length from OSM IS the abovegrond fraction
grid_length_Hv_above = grid_length_Hv_total * (1 - (function_Hv_under/100)) 
grid_length_Hv_under = grid_length_Hv_total * function_Hv_under/100

# out for main text figure 2
grid_length_HV_out_a = pd.concat([grid_length_Hv_above], keys=['aboveground'], names=['type']) 
grid_length_HV_out_u = pd.concat([grid_length_Hv_under], keys=['underground'], names=['type']) 
grid_length_HV_out   = pd.concat([grid_length_HV_out_a, grid_length_HV_out_u])
# grid_length_HV_out.to_csv(path_base / 'imagematerials' / 'electricity' / 'out_test' / 'grid_length_HV_km.csv') # in km

grid_length_Mv_above = grid_length_Mv_time * (1 - function_Mv_under/100)
grid_length_Mv_under = grid_length_Mv_time * function_Mv_under/100
grid_length_Mv_total = grid_length_Mv_above + grid_length_Mv_under

grid_length_Lv_above = grid_length_Lv_time * (1 - function_Lv_under/100)
grid_length_Lv_under = grid_length_Lv_time * function_Lv_under/100
grid_length_Lv_total = grid_length_Lv_above + grid_length_Lv_under

grid_subst_Hv = grid_length_Hv_total.mul(grid_additions.loc['Substations','HV'])        # number of substations on HV network
grid_subst_Mv = grid_length_Mv_total.mul(grid_additions.loc['Substations','MV'])        # # of substations
grid_subst_Lv = grid_length_Lv_total.mul(grid_additions.loc['Substations','LV'])        # # of substations
grid_trans_Hv = grid_length_Hv_total.mul(grid_additions.loc['Transformers','HV'])       # number of transformers on the HV network
grid_trans_Mv = grid_length_Mv_total.mul(grid_additions.loc['Transformers','MV'])       # # of transformers
grid_trans_Lv = grid_length_Lv_total.mul(grid_additions.loc['Transformers','LV'])       # # of transformers




##################
# Interpolations #
##################

# Interpolate material intensities (dynamic content from 1926 to 2100, based on data files)
index                             = pd.MultiIndex.from_product([list(range(YEAR_FIRST_GRID,YEAR_END+1)), list(materials_grid.index.levels[1])])
materials_grid_interpol           = pd.DataFrame(index=index, columns=materials_grid.columns)
materials_grid_additions_interpol = pd.DataFrame(index=pd.MultiIndex.from_product([list(range(YEAR_FIRST_GRID,YEAR_END+1)), list(materials_grid_additions.index.levels[1])]), columns=materials_grid_additions.columns)

for cat in list(materials_grid.index.levels[1]):
   materials_grid_1st   = materials_grid.loc[idx[materials_grid.index[0][0], cat],:]
   materials_grid_interpol.loc[idx[YEAR_FIRST_GRID ,cat],:] = materials_grid_1st                # set the first year (1926) values to the first available values in the dataset (for the year 2000) 
   materials_grid_interpol.loc[idx[materials_grid.index.levels[0].min(),cat],:] = materials_grid.loc[idx[materials_grid.index.levels[0].min(),cat],:]                # set the middle year (2000) values to the first available values in the dataset (for the year 2000) 
   materials_grid_interpol.loc[idx[materials_grid.index.levels[0].max(),cat],:] = materials_grid.loc[idx[materials_grid.index.levels[0].max(),cat],:]                # set the last year (2100) values to the last available values in the dataset (for the year 2050) 
   materials_grid_interpol.loc[idx[:,cat],:] = materials_grid_interpol.loc[idx[:,cat],:].astype('float32').reindex(list(range(YEAR_FIRST_GRID,YEAR_END+1)), level=0).interpolate()

for cat in list(materials_grid_additions.index.levels[1]):
   materials_grid_additions_1st   = materials_grid_additions.loc[idx[materials_grid_additions.index[0][0], cat],:]
   materials_grid_additions_interpol.loc[idx[YEAR_FIRST_GRID ,cat],:] = materials_grid_additions_1st          # set the first year (1926) values to the first available values in the dataset (for the year 2000) 
   materials_grid_additions_interpol.loc[idx[materials_grid_additions.index.levels[0].min(),cat],:] = materials_grid_additions.loc[idx[materials_grid_additions.index.levels[0].min(),cat],:]                # set the middle year (2000) values to the first available values in the dataset (for the year 2000) 
   materials_grid_additions_interpol.loc[idx[materials_grid_additions.index.levels[0].max(),cat],:] = materials_grid_additions.loc[idx[materials_grid_additions.index.levels[0].max(),cat],:]                # set the last year (2100) values to the last available values in the dataset (for the year 2050) 
   materials_grid_additions_interpol.loc[idx[:,cat],:] = materials_grid_additions_interpol.loc[idx[:,cat],:].astype('float32').reindex(list(range(YEAR_FIRST_GRID,YEAR_END+1)), level=0).interpolate()


# call the stock_tail function on all lines, substations & transformers, to add historic stock tail between 1926 & 1971
grid_length_Hv_above_new = stock_tail(grid_length_Hv_above, YEAR_OUT) # km
grid_length_Mv_above_new = stock_tail(grid_length_Mv_above, YEAR_OUT) # km
grid_length_Lv_above_new = stock_tail(grid_length_Lv_above, YEAR_OUT) # km
grid_length_Hv_under_new = stock_tail(grid_length_Hv_under, YEAR_OUT) # km 
grid_length_Mv_under_new = stock_tail(grid_length_Mv_under, YEAR_OUT) # km
grid_length_Lv_under_new = stock_tail(grid_length_Lv_under, YEAR_OUT) # km
grid_subst_Hv_new = stock_tail(grid_subst_Hv, YEAR_OUT)               # units
grid_subst_Mv_new = stock_tail(grid_subst_Mv, YEAR_OUT)               # units
grid_subst_Lv_new = stock_tail(grid_subst_Lv, YEAR_OUT)               # units
grid_trans_Hv_new = stock_tail(grid_trans_Hv, YEAR_OUT)               # units
grid_trans_Mv_new = stock_tail(grid_trans_Mv, YEAR_OUT)               # units
grid_trans_Lv_new = stock_tail(grid_trans_Lv, YEAR_OUT)               # units


#############
# Lifetimes #
#############

# data only for lines, substations and transformer -> bring in knowledge_graph format: HV - Lines, MV - Lines, LV - Lines, HV - Transformers, etc.
expanded_data = {}
for typ in ["Lines", "Transformers", "Substations"]:
    for level in ["HV", "MV", "LV"]:
        new_col = f"{level} - {typ}"
        expanded_data[new_col] = lifetime_grid_elements[typ]
lifetime_grid_elements = pd.DataFrame(expanded_data, index=lifetime_grid_elements.index)
lifetime_grid_elements.rename_axis('Year', inplace=True)

# no differentiation between HV, MV & LV lines as well as between aboveground and belowground
# Types: lines, transformers, substations
lifetime_grid_elements.loc[YEAR_FIRST_GRID,:]  = lifetime_grid_elements.loc[lifetime_grid_elements.first_valid_index(),:]
lifetime_grid_elements.loc[YEAR_OUT,:]         = lifetime_grid_elements.loc[lifetime_grid_elements.last_valid_index(),:]
lifetime_grid_elements                         = lifetime_grid_elements.reindex(list(range(YEAR_FIRST_GRID, YEAR_OUT+1))).interpolate()
# TODO: check why lifetime for lines is interpoltaed from 2020 - 40yrs to 2050 - 48 yrs and then back to 2060 - 40 yrs -> should stay at 48 yrs?


#######################################################################################################
#%%%% NEW

# lifetimes
df_mean = lifetime_grid_elements.copy()
df_stdev = df_mean * STD_LIFETIMES_ELECTR
df_mean.columns = [(col, 'mean') for col in df_mean.columns] # Rename columns to multi-level tuples
df_stdev.columns = [(col, 'stdev') for col in df_stdev.columns]
lifetime_grid_distr = pd.concat([df_mean, df_stdev], axis=1) # Concatenate along columns


# Materials

# harmonize units of MI -> then materials_grid and materials_grid_additions can be merged in one dataframe
# NOT POSSIBLE - separate stock modelling necessary

# Grid Additions MIs ---
# # for substations and transformer we multiply the MI (kg/unit of substation or transformer) with the number of substations or transformers per kilometer of grid length
# materials_grid_additions_kgperkm             = materials_grid_additions_to_kgperkm(materials_grid_additions_interpol, grid_additions)
# # material intensities: (years, tech. type) index and materials as columns -> years as index and (tech. type, materials) as columns
# materials_grid_additions_kgperkm.index.names = ["Year", "Type"]
# materials_grid_additions_kgperkm             = materials_grid_additions_kgperkm.unstack(level='Type')   # bring tech. type from row index to column header
# materials_grid_additions_kgperkm.columns     = materials_grid_additions_kgperkm.columns.swaplevel(0, 1) # Swap the levels of the MultiIndex columns
# materials_grid_additions_kgperkm             = materials_grid_additions_kgperkm.sort_index(axis=1)
# # rename columns to match knowledge graph
# new_level_0 = [col.replace(' ', ' - ', 1) for col in materials_grid_additions_kgperkm.columns.get_level_values(0)]
# new_columns = pd.MultiIndex.from_arrays([
#     new_level_0,
#     materials_grid_additions_kgperkm.columns.get_level_values(1)
# ], names=materials_grid_additions_kgperkm.columns.names)
# materials_grid_additions_kgperkm.columns = new_columns

materials_grid_additions_kgperunit            = materials_grid_additions_interpol.copy()
# material intensities: (years, tech. type) index and materials as columns -> years as index and (tech. type, materials) as columns
materials_grid_additions_kgperunit.index.names = ["Year", "Type"]
materials_grid_additions_kgperunit             = materials_grid_additions_kgperunit.unstack(level='Type')   # bring tech. type from row index to column header
materials_grid_additions_kgperunit.columns     = materials_grid_additions_kgperunit.columns.swaplevel(0, 1) # Swap the levels of the MultiIndex columns
materials_grid_additions_kgperunit             = materials_grid_additions_kgperunit.sort_index(axis=1)
# rename columns to match knowledge graph
new_level_0 = [col.replace(' ', ' - ', 1) for col in materials_grid_additions_kgperunit.columns.get_level_values(0)]
new_columns = pd.MultiIndex.from_arrays([
    new_level_0,
    materials_grid_additions_kgperunit.columns.get_level_values(1)
], names=materials_grid_additions_kgperunit.columns.names)
materials_grid_additions_kgperunit.columns = new_columns

# Grid MIs ---
materials_grid_kgperkm              = materials_grid_interpol.copy() # copy the interpolated material intensities
materials_grid_kgperkm.index.names  = ["Year", "Type"]
materials_grid_kgperkm              = materials_grid_kgperkm.unstack(level='Type') # bring tech. type from row index to column header
materials_grid_kgperkm.columns      = materials_grid_kgperkm.columns.swaplevel(0, 1) # Swap the levels of the MultiIndex columns
materials_grid_kgperkm              = materials_grid_kgperkm.sort_index(axis=1)
# rename columns to match knowledge graph
new_level_0 = [col.replace(' ', ' - Lines - ', 1) for col in materials_grid_kgperkm.columns.get_level_values(0)]
new_columns = pd.MultiIndex.from_arrays([
    new_level_0,
    materials_grid_kgperkm.columns.get_level_values(1)
], names=materials_grid_kgperkm.columns.names)
materials_grid_kgperkm.columns = new_columns

# # print_df_info(materials_grid_kgperkm, "Materials Grid kg/km")
# # print_df_info(materials_grid_additions_kgperkm, "Materials Grid Additions kg/km")

# materials_grid_combined_kgperkm = pd.concat([materials_grid_kgperkm, materials_grid_additions_kgperkm], axis=1) # 


# # Stocks
# grid_dict = dict({
#         'HV - Lines - Overhead':    grid_length_Hv_above_new,
#         'HV - Lines - Underground': grid_length_Hv_under_new,
#         'HV - Substations':         grid_subst_Hv_new,
#         'HV - Transformers':        grid_trans_Hv_new,
#         'MV - Lines - Overhead':    grid_length_Mv_above_new,
#         'MV - Lines - Underground': grid_length_Mv_under_new,
#         'MV - Substations':         grid_subst_Mv_new,
#         'MV - Transformers':        grid_trans_Mv_new,
#         'LV - Lines - Overhead':    grid_length_Lv_above_new,
#         'LV - Lines - Underground': grid_length_Lv_under_new,
#         'LV - Substations':         grid_subst_Lv_new,
#         'LV - Transformers':        grid_trans_Lv_new
#     })
# grid_stock = pd.concat(grid_dict, axis=1) # Concatenate with keys to create MultiIndex ('Name', 'Region')
# grid_stock = grid_stock.sort_index(axis=1)

grid_dict_add = dict({
        'HV - Substations':         grid_subst_Hv_new,
        'HV - Transformers':        grid_trans_Hv_new,
        'MV - Substations':         grid_subst_Mv_new,
        'MV - Transformers':        grid_trans_Mv_new,
        'LV - Substations':         grid_subst_Lv_new,
        'LV - Transformers':        grid_trans_Lv_new
    })

grid_dict_lines = dict({
        'HV - Lines - Overhead':    grid_length_Hv_above_new,
        'HV - Lines - Underground': grid_length_Hv_under_new,
        'MV - Lines - Overhead':    grid_length_Mv_above_new,
        'MV - Lines - Underground': grid_length_Mv_under_new,
        'LV - Lines - Overhead':    grid_length_Lv_above_new,
        'LV - Lines - Underground': grid_length_Lv_under_new,
    })
    
grid_stock_lines = pd.concat(grid_dict_lines, axis=1) # Concatenate with keys to create MultiIndex ('Name', 'Region')
grid_stock_lines = grid_stock_lines.sort_index(axis=1)

grid_stock_add = pd.concat(grid_dict_add, axis=1) # Concatenate with keys to create MultiIndex ('Name', 'Region')
grid_stock_add = grid_stock_add.sort_index(axis=1)

#----------------------------------------------------------------------------------------------------------
###########################################################################################################
#%%% 3.3) Prep_data File
###########################################################################################################
#----------------------------------------------------------------------------------------------------------


ureg = pint.UnitRegistry(force_ndarray_like=True)
# Define the units for each dimension
unit_mapping = { # TODO: move to constants.py
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
# conversion_table = {
#     "grid_stock": (["Time"], ["Type", "Region"],),
#     "materials_grid_combined_kgperkm": (["Cohort"], ["Type", "material"],)
#     # "gcap_materials_interpol": (["Cohort"], ["Type", "SubType", "material"], {"Type": ["Type", "SubType"]})
# }

conversion_table = {
    "grid_stock_lines": (["Time"], ["Type", "Region"],),
    "materials_grid_kgperkm": (["Cohort"], ["Type", "material"],),
    "grid_stock_add": (["Time"], ["Type", "Region"],),
    "materials_grid_add_kgperunit": (["Cohort"], ["Type", "material"],),
    "grid_stock": (["Time"], ["Type", "Region"],), # TODO: delete
    "materials_grid_combined_kgperkm": (["Cohort"], ["Type", "material"],) # TODO: delete
}



# results_dict = {
#         'grid_stock': grid_stock,
#         'materials_grid_combined_kgperkm': materials_grid_combined_kgperkm,
#         'lifetime_grid_distr': lifetime_grid_distr,
# }
results_dict_lines = {
        'grid_stock_lines': grid_stock_lines,
        'materials_grid_kgperkm': materials_grid_kgperkm,
        'lifetime_grid_distr': lifetime_grid_distr,
}
results_dict_add = {
        'grid_stock_add': grid_stock_add,
        'materials_grid_add_kgperunit': materials_grid_additions_kgperunit,
        'lifetime_grid_distr': lifetime_grid_distr,
}


prep_data_lines = create_prep_data(results_dict_lines, conversion_table, unit_mapping)
prep_data_add = create_prep_data(results_dict_add, conversion_table, unit_mapping)

prep_data_lines["stocks"] = prism.Q_(prep_data_lines["stocks"], "km")
prep_data_lines["material_intensities"] = prism.Q_(prep_data_lines["material_intensities"], "kg/km")
prep_data_lines["set_unit_flexible"] = prism.U_(prep_data_lines["stocks"]) # prism.U_ gives the unit back
# set_unit_flexible is needed by the model to deal with the fact the in the beginning of the model it doesn't know th data yet and needs to work with a placeholder/flexible unit (see model.py) 

prep_data_add["stocks"] = prism.Q_(prep_data_add["stocks"], "count")
prep_data_add["material_intensities"] = prism.Q_(prep_data_add["material_intensities"], "kg/count")
prep_data_add["set_unit_flexible"] = prism.U_(prep_data_add["stocks"]) # prism.U_ gives the unit back



#----------------------------------------------------------------------------------------------------------
###########################################################################################################
#%%% 3.4) Run Stock Model New
###########################################################################################################
# TODO: move this to electricity.py
#----------------------------------------------------------------------------------------------------------


prep_data_lines, prep_data_add = get_preprocessing_data_grid(path_base, SCEN, VARIANT, YEAR_START, YEAR_END, YEAR_OUT)

# LINES ----------------------------------------------------
# prep_data = create_prep_data(results_dict_lines, conversion_table, unit_mapping)

# # Define the complete timeline, including historic tail
time_start = prep_data["stocks"].coords["Time"].min().values
complete_timeline = prism.Timeline(time_start, YEAR_END, 1)
simulation_timeline = prism.Timeline(YEAR_START, YEAR_END, 1) #1970


sec_electr_grid_lines = Sector("electr_grid_lines", prep_data_lines)

main_model_factory_lines = ModelFactory(
    sec_electr_grid_lines, complete_timeline
    ).add(GenericStocks
    ).add(MaterialIntensities
    ).finish()

main_model_factory_lines.simulate(simulation_timeline)
list(main_model_factory_lines.electr_grid_lines)

# ADDITIONS -------------------------------------------------------------------------------------
# prep_data = create_prep_data(results_dict_add, conversion_table, unit_mapping)

# # Define the complete timeline, including historic tail
time_start = prep_data["stocks"].coords["Time"].min().values
complete_timeline = prism.Timeline(time_start, YEAR_END, 1)
simulation_timeline = prism.Timeline(YEAR_START, YEAR_END, 1) #1970


sec_electr_grid_add = Sector("electr_grid_add", prep_data_add)

main_model_factory_add = ModelFactory(
    sec_electr_grid_add, complete_timeline
    ).add(GenericStocks
    ).add(MaterialIntensities
    ).finish()

main_model_factory_add.simulate(simulation_timeline)
list(main_model_factory_add.electr_grid_add)



# Define the coordinates of all dimensions.
# Region = list(prep_data["stocks"].coords["Region"].values)
# Time = [t for t in complete_timeline]
# Cohort = Time
# Type = list(prep_data["stocks"].coords["Type"].values)
# material = list(prep_data["material_intensities"].coords["material"].values)

# Create
# main_model_normal = GenericMainModel(
#     complete_timeline, Region=Region, Time=Time, Cohort=Cohort, Type=Type, prep_data=prep_data,
#     compute_materials=True, compute_battery_materials=False, compute_maintenance_materials=False, 
#     material=material)

path_test_plots = Path(path_base, "imagematerials", "electricity", "out_test", scen_folder, "Figures")

###########################################################################################################
#%%% Stocks 
###########################################################################################################



#%%%% one model ---------------------------------------------------

data_all = main_model_factory.stocks.copy()


data_plot = data_all.sum(dim="Region")
types_top = ['HV - Lines - Overhead', 'HV - Lines - Underground', 'MV - Lines - Overhead', 'MV - Lines - Underground', 
             'LV - Lines - Overhead', 'LV - Lines - Underground', 'LV - Substations', 'LV - Transformers'] 
types_bottom = ['HV - Substations', 'HV - Transformers', 'MV - Substations', 'MV - Transformers']

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 8))
linewidth = 2
s_legend = 12
s_label = 14


i=0
for t in types_top:
    axes[0].plot(data_plot.Time, data_plot.sel(Type=t), label=t, color=DICT_GRID_STYLES_2[t][0], linestyle=DICT_GRID_STYLES_2[t][1], linewidth=linewidth)
# axes.set_title(f"{region} (Types 1–10)")
# axes[0].set_xlabel(" ")
axes[0].set_ylabel("Stocks (unit/km)", fontsize=s_label)
axes[0].grid(alpha=0.3, linestyle='--')
axes[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
axes[0].tick_params(axis='both', which='major', labelsize=s_legend) # set font size of axis ticks
axes[0].legend(loc='upper left', fontsize=s_legend)

# Bottom row: Types 11–20
for t in types_bottom:
    axes[1].plot(data_plot.Time, data_plot.sel(Type=t), label=t, color=DICT_GRID_STYLES_2[t][0], linestyle=DICT_GRID_STYLES_2[t][1], linewidth=linewidth)
# axes.set_title(f"{region} (Types 11–20)")
axes[1].set_xlabel("Time", fontsize=s_label)
axes[1].set_ylabel("Stocks (unit)", fontsize=s_label)
axes[1].grid(alpha=0.3, linestyle='--')
axes[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
axes[1].tick_params(axis='both', which='major', labelsize=s_legend) # set font size of axis ticks
axes[1].legend(loc='upper left', fontsize=s_legend)

plt.suptitle("Electricity Grid - Stocks", fontsize=16)

plt.tight_layout()
# fig.savefig(path_test_plots / "Grid_stocks_world.png", dpi=300)
plt.show()


#%%%% 2 models ---------------------------------------------------


data_lines  = main_model_factory_lines.stocks.copy().sum(dim="Region")
data_add    = main_model_factory_add.stocks.copy().sum(dim="Region")

# data        = xr.concat([data_lines, data_add], dim='Type')
# data_plot   = data.sum(dim="Region")

types_top    = ['HV - Lines - Overhead', 'HV - Lines - Underground', 'MV - Lines - Overhead', 'MV - Lines - Underground', 
                'LV - Lines - Overhead', 'LV - Lines - Underground'] #, 'LV - Transformers', 'LV - Substations',
types_bottom = ['HV - Substations', 'HV - Transformers', 'MV - Substations', 'MV - Transformers']



fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 8))
linewidth = 2
s_legend = 12
s_label = 14

# Top row:
for t in types_top:
    line, = axes[0].plot(data_lines.Time, data_lines.sel(Type=t), label=t, color=dict_grid_styles2[t][0], linestyle=dict_grid_styles2[t][1], linewidth=linewidth)

axes[0].set_ylabel("Stocks (# counts/km)", fontsize=s_label)
axes[0].grid(alpha=0.3, linestyle='--')
axes[0].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
axes[0].tick_params(axis='both', which='major', labelsize=s_legend) # set font size of axis ticks
axes[0].legend(loc='upper left', fontsize=s_legend) #handles=handles, labels=labels, 

# Bottom row:
for t in types_bottom:
    axes[1].plot(data_add.Time, data_add.sel(Type=t), label=t, color=dict_grid_styles2[t][0], linestyle=dict_grid_styles2[t][1], linewidth=linewidth)

axes[1].set_xlabel("Time", fontsize=s_label)
axes[1].set_ylabel("Stocks (# counts)", fontsize=s_label)
axes[1].grid(alpha=0.3, linestyle='--')
axes[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
axes[1].tick_params(axis='both', which='major', labelsize=s_legend) # set font size of axis ticks
axes[1].legend(loc='upper left', fontsize=s_legend)

plt.suptitle(f"{scen_folder}: Electricity Grid - Stocks", fontsize=16)

plt.tight_layout()
# fig.savefig(path_test_plots / "Grid_stocks_world.png", dpi=300)
plt.show()


###########################################################################################################
#%%% Stocks Materials
###########################################################################################################

materials = ["Steel", "Concrete", "Aluminium", "Cu"]


# #%%%% 1 model ---------------------------------------------------

# data_all = main_model_factory.stock_by_cohort_materials.copy().sum('Region')

# # data_all = main_model_factory.inflow_materials.to_array().sum('Region')
# data_all = data_all.pint.to("t")  # Convert kg -> tonnes
# data_all = data_all.sel(Time=slice(1971, None))
# data_plot = data_all.sum(dim='Type')

# lines_sum = data_all.sel(Type=[t for t in data_all.Type.values if 'Lines' in t]).sum(dim='Type') # Get group sums by keyword and sum over types (sum over HV, MV and LV (and overground/underground for lines))
# transformers_sum = data_all.sel(Type=[t for t in data_all.Type.values if 'Transformers' in t]).sum(dim='Type')
# substations_sum = data_all.sel(Type=[t for t in data_all.Type.values if 'Substations' in t]).sum(dim='Type')


# fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 10))
# linewidth = 2
# s_legend = 12
# s_label = 14

# for i, mat in enumerate(materials):
#     lines_sum.sel(material=mat).plot(ax=axes[i], color = DICT_GRID_COLORS['Lines'], label="Lines")
#     transformers_sum.sel(material=mat).plot(ax=axes[i], color = DICT_GRID_COLORS['Transformers'], label="Transformers")
#     substations_sum.sel(material=mat).plot(ax=axes[i], color = DICT_GRID_COLORS['Substations'], label="Substations")
#     data_plot.sel(material=mat).plot(ax=axes[i], label="Total", color='red', alpha=0.8, linestyle='--', linewidth=3)

#     # if mat == "Cu":
#     #     # Add IEA data for copper
#     #     axes[i].plot(df_iea_lt_cu.index, df_iea_lt_cu['Cu'], label="IEA L&T", color='#00a5cf', linestyle=':', linewidth=4)

#     # if mat == "Aluminium":
#     #     # Add IEA data for aluminium
#     #     axes[i].plot(df_iea_lt_alu.index, df_iea_lt_alu['Aluminium'], label="IEA L&T", color='#00a5cf', linestyle=':', linewidth=4)

#     # if mat == "Steel":
#     #     # Add IEA data for steel
#     #     axes[i].plot(df_iea_t_steel.index, df_iea_t_steel['Steel'], label="IEA T", color='#00a5cf', linestyle=':', linewidth=4)
    
#     axes[i].grid(alpha=0.3, linestyle='--')
#     axes[i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
#     axes[i].tick_params(axis='both', which='major', labelsize=s_legend) # set font size of axis ticks
#     axes[i].set_title(f"{mat}")
#     axes[i].set_xlabel(" ")
#     axes[i].set_ylabel("Inflow [t]", fontsize=s_label)
#     axes[i].legend()

# axes[-1].set_xlabel("Time", fontsize=s_label)

# plt.suptitle("Electricity Grid Stocks Materials", fontsize=16)
# plt.tight_layout()
# # fig.savefig(path_test_plots / "Grid_inflow-materials_world.svg")
# # fig.savefig(path_test_plots / "Grid_inflow-materials_world_1971.pdf")
# # fig.savefig(path_test_plots / "Grid_inflow-materials_world_1971.png")
# # fig.savefig(path_test_plots / "Grid_inflow-materials_world_1971.svg")
# plt.show()

#%%%% 2 model ---------------------------------------------------

data_lines  = main_model_factory_lines.stock_by_cohort_materials.copy().sum(dim="Region").pint.to("t")  # Convert kg -> tonnes
data_add    = main_model_factory_add.stock_by_cohort_materials.copy().sum(dim="Region").pint.to("t")  # Convert kg -> tonnes

data        = xr.concat([data_lines, data_add], dim='Type')
data        = data.sel(Time=slice(1971, None))

data_sum    = data.sum(dim='Type')

lines_sum           = data.sel(Type=[t for t in data.Type.values if 'Lines' in t]).sum(dim='Type') # Get group sums by keyword and sum over types (sum over HV, MV and LV (and overground/underground for lines))
transformers_sum    = data.sel(Type=[t for t in data.Type.values if 'Transformers' in t]).sum(dim='Type')
substations_sum     = data.sel(Type=[t for t in data.Type.values if 'Substations' in t]).sum(dim='Type')


fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 10))
linewidth = 2
s_legend = 12
s_label = 14

for i, mat in enumerate(materials):
    lines_sum.sel(material=mat).plot(ax=axes[i], color = DICT_GRID_COLORS['Lines'], label="Lines")
    transformers_sum.sel(material=mat).plot(ax=axes[i], color = DICT_GRID_COLORS['Transformers'], label="Transformers")
    substations_sum.sel(material=mat).plot(ax=axes[i], color = DICT_GRID_COLORS['Substations'], label="Substations")
    data_sum.sel(material=mat).plot(ax=axes[i], label="Total", color='red', alpha=0.8, linestyle='--', linewidth=3)

    # if mat == "Cu":
    #     # Add IEA data for copper
    #     axes[i].plot(df_iea_lt_cu.index, df_iea_lt_cu['Cu'], label="IEA L&T", color='#00a5cf', linestyle=':', linewidth=4)

    # if mat == "Aluminium":
    #     # Add IEA data for aluminium
    #     axes[i].plot(df_iea_lt_alu.index, df_iea_lt_alu['Aluminium'], label="IEA L&T", color='#00a5cf', linestyle=':', linewidth=4)

    # if mat == "Steel":
    #     # Add IEA data for steel
    #     axes[i].plot(df_iea_t_steel.index, df_iea_t_steel['Steel'], label="IEA T", color='#00a5cf', linestyle=':', linewidth=4)
    
    axes[i].grid(alpha=0.3, linestyle='--')
    axes[i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
    axes[i].tick_params(axis='both', which='major', labelsize=s_legend) # set font size of axis ticks
    axes[i].set_title(f"{mat}")
    axes[i].set_xlabel(" ")
    axes[i].set_ylabel("Inflow [t]", fontsize=s_label)
    axes[i].legend()

axes[-1].set_xlabel("Time", fontsize=s_label)

plt.suptitle(f"{scen_folder}: Electricity Grid Stocks Materials", fontsize=16)
plt.tight_layout()
# fig.savefig(path_test_plots / "Grid_stocks-materials_world.svg")
# fig.savefig(path_test_plots / "Grid_stocks-materials_world_1971.pdf")
fig.savefig(path_test_plots / "Grid_stocks-materials_world_1971.png")
# fig.savefig(path_test_plots / "Grid_stocks-materials_world_1971.svg")
plt.show()





###########################################################################################################
#%%% INFLOW Grid
###########################################################################################################


#================================================================================
#%%%% Per TECH - per Region

regions = ['Brazil', 'C.Europe', 'China'] 
threshold = 100_000

data_lines = main_model_factory_lines.inflow.to_array().sel(
    time=slice(1971, None), 
    Type=main_model_factory_lines.inflow.to_array().Type != '<EMPTY>', 
    Region=regions
)

data_add = main_model_factory_add.inflow.to_array().sel(
    time=slice(1971, None), 
    Type=main_model_factory_add.inflow.to_array().Type != '<EMPTY>', 
    Region=regions
)
# data        = xr.concat([data_lines, data_add], dim='Type')
# data        = data.sel(time=slice(1971, None), Type=data.Type != '<EMPTY>', Region=regions)
# techs_upper = [coord_name.item() for coord_name in data.coords['Type']  
#                if data.sel(Type = coord_name).values.max() > threshold]
# techs_lower = [coord_name.item() for coord_name in data.coords['Type']  
#                if data.sel(Type = coord_name).values.max() <= threshold]

techs_upper = data_lines.Type.values
techs_lower = data_add.Type.values


fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 8))  # Now 3 columns for 3 regions
linewidth = 2
s_legend = 12
s_label = 14

for i, region in enumerate(regions):  # regions now has length 3
    col = i  # Column index: 0, 1, 2

    # Top row: Types 1–15
    for t in techs_upper:
        data_plot = data_lines.sel(Type=t, Region=region)
        color, ls = DICT_GENTECH_STYLES.get(t, ('black', '-'))
        axes[0, col].plot(data_plot.time, data_plot.values, label=t, color=DICT_GRID_STYLES_2[t][0], linestyle=DICT_GRID_STYLES_2[t][1], linewidth=linewidth)
    axes[0, col].set_title(f"{region}", fontsize=15)
    axes[0, col].grid(alpha=0.3, linestyle='--')
    axes[0, col].tick_params(axis='both', which='major', labelsize=s_legend)
    axes[0, 0].set_ylabel("Inflow (km)", fontsize=s_label)

    # Bottom row: Types 16–30
    for t in techs_lower:
        data_plot = data_add.sel(Type=t, Region=region)
        color, ls = DICT_GENTECH_STYLES.get(t, ('black', '-'))
        axes[1, col].plot(data_plot.time, data_plot.values, label=t, color=DICT_GRID_STYLES_2[t][0], linestyle=DICT_GRID_STYLES_2[t][1], linewidth=linewidth)
    axes[1, col].grid(alpha=0.3, linestyle='--')
    axes[1, col].tick_params(axis='both', which='major', labelsize=s_legend)
    axes[1, 0].set_ylabel("Inflow (# counts)", fontsize=s_label)
    axes[1, col].set_xlabel("Time", fontsize=s_label)

# Y-axis number formatting and hiding right y-axis ticks
for ax in axes.flat:
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

for row in range(2):
    for col in [1, 2]:  # Hide y-tick labels for middle and right columns
        axes[row, col].tick_params(labelleft=False)
axes[0, 2].legend(fontsize=s_legend, ncol=2, loc='upper center', bbox_to_anchor=(-1.4, -1.41))
axes[1, 2].legend(fontsize=s_legend, ncol=2, loc='upper center', bbox_to_anchor=(-0.1, -0.21))

plt.suptitle(f"{scen_folder}: Grid - Inflow", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to make room for the suptitle
region_str = "_".join(regions)
# fig.savefig(path_test_plots / f"Gen_inflow_{region_str}.png", dpi=300, bbox_inches='tight')
fig.savefig(path_test_plots / f"Grid_inflow_{region_str}_1971.png", dpi=300, bbox_inches='tight')
plt.show()



#================================================================================
#%%%% Sum & Per TECH - World

data_lines = main_model_factory_lines.inflow.to_array().sel(
    time=slice(1971, None), 
    Type=main_model_factory_lines.inflow.to_array().Type != '<EMPTY>'
).sum('Region')

data_add = main_model_factory_add.inflow.to_array().sel(
    time=slice(1971, None), 
    Type=main_model_factory_add.inflow.to_array().Type != '<EMPTY>'
).sum('Region')

# data        = xr.concat([data_lines, data_add], dim='Type')
# data        = data.sel(time=slice(1971, None), Type=data.Type != '<EMPTY>').sum('Region')

types_top    = ['HV - Lines - Overhead', 'HV - Lines - Underground', 'MV - Lines - Overhead', 'MV - Lines - Underground', 
                'LV - Lines - Overhead', 'LV - Lines - Underground']
types_bottom = ['HV - Substations', 'HV - Transformers', 'MV - Substations', 'MV - Transformers', 'LV - Transformers', 'LV - Substations']

fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 9))
linewidth = 2
s_legend = 12
s_label = 14

# Second subplot: data_all_2 (summed over Region and Type, likely only time and material left)
for t in types_top:
    data_plot = data_lines.sel(Type=t)
    data_plot.plot(ax=axes[0], label=t, color=DICT_GRID_STYLES_2[t][0], linestyle=DICT_GRID_STYLES_2[t][1], linewidth=linewidth)
    axes[0].grid(alpha=0.3, linestyle='--')
    axes[0].tick_params(axis='both', which='major', labelsize=s_legend)
    axes[0].set_xlabel(" ", fontsize=s_label)
    axes[0].set_ylabel("Inflow (km)", fontsize=s_label)
    axes[0].set_title('Grid Lines')
    # axes[1].legend(ncol=2, loc='upper center', bbox_to_anchor=(0.35, -0.2), fontsize=s_legend)

# First subplot: data_all_1 (summed over Region, still over Type and time likely)
for t in types_bottom:
    data_plot = data_add.sel(Type=t)
    axes[1].plot(data_plot.time, data_plot.values, label=t, color=DICT_GRID_STYLES_2[t][0], linestyle=DICT_GRID_STYLES_2[t][1], linewidth=linewidth)
    axes[1].grid(alpha=0.3, linestyle='--')
    axes[1].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
    axes[1].tick_params(axis='both', which='major', labelsize=s_legend)
    axes[1].set_xlabel("Time", fontsize=s_label)
    axes[1].set_ylabel("Inflow (# counts)", fontsize=s_label)
    axes[1].set_title('Grid Additions')
    # axes[1].legend(ncol=2, loc='upper center', bbox_to_anchor=(0.9, -0.2), fontsize=s_legend)

# Collect handles and labels from both subplots
handles0, labels0 = axes[0].get_legend_handles_labels()
handles1, labels1 = axes[1].get_legend_handles_labels()
handles = handles0 + handles1
labels = labels0 + labels1
# Add a single legend below the plots
fig.legend(handles, labels, ncol=4, loc='lower center', bbox_to_anchor=(0.5, -0.1), fontsize=s_legend)

plt.suptitle(f"{scen_folder}: Grid - Inflow", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.98])  # Adjust layout to make room for the suptitle
# fig.savefig(path_test_plots / f"Gen_inflow_world.png", dpi=300, bbox_inches='tight')
fig.savefig(path_test_plots / f"Grid_inflow_world_1971.png", dpi=300, bbox_inches='tight')
plt.show()






###########################################################################################################
#%% INFLOW Materials
###########################################################################################################




#================================================================================
#%%%% Per TECH - World


regions = ['Brazil', 'C.Europe', 'China'] 
materials = ["Steel", "Concrete", "Aluminium", "Cu"]


# data IEA ---------------------------------------------------------------------
# values for APS  scenario (NZE scenario)

# Cu -------------
years = np.arange(2012, 2051)
values = np.concatenate([
    np.full(10, 5e9),       # 2012–2021 # 5 Mt = 5*10e9 kg
    np.full(9, 5.5e9),       # 2022–2030
    np.full(10, np.nan),    # 2031–2040 (gap)
    np.full(10, 9e9),       # 2041–2050 (12)
])
df_iea_lt_cu = pd.DataFrame({ # lines & transformers, copper
    'Year': years,
    'Cu': values
})
df_iea_lt_cu.set_index('Year', inplace=True)
df_iea_lt_cu = df_iea_lt_cu.pint.to("t")  # Convert kg -> t

# Alu -------------
years = np.arange(2012, 2051)
values = np.concatenate([
    np.full(10, 12e9),       # 2012–2021 # 12 Mt = 12*10e9 kg
    np.full(9, 13e9),        # 2022–2030
    np.full(10, np.nan),    # 2031–2040 (gap)
    np.full(10, 21e9),       # 2041–2050 (27)
])
df_iea_lt_alu = pd.DataFrame({ # lines & transformers, aluminium
    'Year': years,
    'Aluminium': values
})
df_iea_lt_alu.set_index('Year', inplace=True)
df_iea_lt_alu = df_iea_lt_alu.pint.to("t")  # Convert kg -> t

# Steel -------------
years = np.arange(2012, 2051)
values = np.concatenate([
    np.full(10, 5e9),       # 2012–2021
    np.full(9, np.nan),     # 2022–2030
    np.full(10, 9e9),       # 2031–2040 (13)
    np.full(10, np.nan),    # 2041–2050
])
df_iea_t_steel = pd.DataFrame({ # transformers, steel
    'Year': years,
    'Steel': values
})
df_iea_t_steel.set_index('Year', inplace=True)
df_iea_t_steel = df_iea_t_steel.pint.to("t")  # Convert kg -> t
# ----------------------------------------------------------------------------------


#%%%% 1 model ---------------------------------------------------

# data_all = main_model_factory.inflow_materials.to_array().sum('Region')
# data_all = data_all.pint.to("t")  # Convert kg -> tonnes
# data_all = data_all.sel(time=slice(1971, None))
# data_plot = data_all.sum(dim='Type')

# lines_sum = data_all.sel(Type=[t for t in data_all.Type.values if 'Lines' in t]).sum(dim='Type') # Get group sums by keyword and sum over types (sum over HV, MV and LV (and overground/underground for lines))
# transformers_sum = data_all.sel(Type=[t for t in data_all.Type.values if 'Transformers' in t]).sum(dim='Type')
# substations_sum = data_all.sel(Type=[t for t in data_all.Type.values if 'Substations' in t]).sum(dim='Type')


# fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 10))
# linewidth = 2
# s_legend = 12
# s_label = 14

# for i, mat in enumerate(materials):
#     lines_sum.sel(material=mat).plot(ax=axes[i], color = DICT_GRID_COLORS['Lines'], label="Lines")
#     transformers_sum.sel(material=mat).plot(ax=axes[i], color = DICT_GRID_COLORS['Transformers'], label="Transformers")
#     substations_sum.sel(material=mat).plot(ax=axes[i], color = DICT_GRID_COLORS['Substations'], label="Substations")
#     data_plot.sel(material=mat).plot(ax=axes[i], label="Total", color='red', alpha=0.8, linestyle='--', linewidth=3)

#     if mat == "Cu":
#         # Add IEA data for copper
#         axes[i].plot(df_iea_lt_cu.index, df_iea_lt_cu['Cu'], label="IEA L&T", color='#00a5cf', linestyle=':', linewidth=4)

#     if mat == "Aluminium":
#         # Add IEA data for aluminium
#         axes[i].plot(df_iea_lt_alu.index, df_iea_lt_alu['Aluminium'], label="IEA L&T", color='#00a5cf', linestyle=':', linewidth=4)

#     if mat == "Steel":
#         # Add IEA data for steel
#         axes[i].plot(df_iea_t_steel.index, df_iea_t_steel['Steel'], label="IEA T", color='#00a5cf', linestyle=':', linewidth=4)
    
#     axes[i].grid(alpha=0.3, linestyle='--')
#     axes[i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
#     axes[i].tick_params(axis='both', which='major', labelsize=s_legend) # set font size of axis ticks
#     axes[i].set_title(f"{mat}")
#     axes[i].set_xlabel(" ")
#     axes[i].set_ylabel("Inflow [t]", fontsize=s_label)
#     axes[i].legend()

# axes[-1].set_xlabel("Time", fontsize=s_label)

# plt.suptitle("Electricity Grid Inflow Materials", fontsize=16)
# plt.tight_layout()
# # fig.savefig(path_test_plots / "Grid_inflow-materials_world.svg")
# # fig.savefig(path_test_plots / "Grid_inflow-materials_world_1971.pdf")
# # fig.savefig(path_test_plots / "Grid_inflow-materials_world_1971.png")
# # fig.savefig(path_test_plots / "Grid_inflow-materials_world_1971.svg")
# plt.show()

#%%%% 2 model ---------------------------------------------------

data_lines  = main_model_factory_lines.inflow_materials.to_array().sum(dim="Region").pint.to("t")  # Convert kg -> tonnes
data_add    = main_model_factory_add.inflow_materials.to_array().sum(dim="Region").pint.to("t")  # Convert kg -> tonnes

data_lines  = data_lines.sel(time=slice(1971, None))
data_add    = data_add.sel(time=slice(1971, None))
data        = xr.concat([data_lines, data_add], dim='Type')
data_sum    = data.sum(dim='Type')

lines_sum           = data_lines.sel(Type=[t for t in data_lines.Type.values if 'Lines' in t]).sum(dim='Type') # Get group sums by keyword and sum over types (sum over HV, MV and LV (and overground/underground for lines))
transformers_sum    = data_add.sel(Type=[t for t in data_add.Type.values if 'Transformers' in t]).sum(dim='Type')
substations_sum     = data_add.sel(Type=[t for t in data_add.Type.values if 'Substations' in t]).sum(dim='Type')


fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 10))
linewidth = 2
s_legend = 12
s_label = 14

for i, mat in enumerate(materials):
    lines_sum.sel(material=mat).plot(ax=axes[i], color = DICT_GRID_COLORS['Lines'], label="Lines")
    transformers_sum.sel(material=mat).plot(ax=axes[i], color = DICT_GRID_COLORS['Transformers'], label="Transformers")
    substations_sum.sel(material=mat).plot(ax=axes[i], color = DICT_GRID_COLORS['Substations'], label="Substations")
    data_sum.sel(material=mat).plot(ax=axes[i], label="Total", color='red', alpha=0.8, linestyle='--', linewidth=3)

    if mat == "Cu":
        # Add IEA data for copper
        axes[i].plot(df_iea_lt_cu.index, df_iea_lt_cu['Cu'], label="IEA L&T", color='#00a5cf', linestyle=':', linewidth=4)

    if mat == "Aluminium":
        # Add IEA data for aluminium
        axes[i].plot(df_iea_lt_alu.index, df_iea_lt_alu['Aluminium'], label="IEA L&T", color='#00a5cf', linestyle=':', linewidth=4)

    if mat == "Steel":
        # Add IEA data for steel
        axes[i].plot(df_iea_t_steel.index, df_iea_t_steel['Steel'], label="IEA T", color='#00a5cf', linestyle=':', linewidth=4)
    
    axes[i].grid(alpha=0.3, linestyle='--')
    axes[i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
    axes[i].tick_params(axis='both', which='major', labelsize=s_legend) # set font size of axis ticks
    axes[i].set_title(f"{mat}")
    axes[i].set_xlabel(" ")
    axes[i].set_ylabel("Inflow [t]", fontsize=s_label)
    axes[i].legend()

axes[-1].set_xlabel("Time", fontsize=s_label)

plt.suptitle(f"{scen_folder}: Electricity Grid Inflow Materials", fontsize=16)
plt.tight_layout()
# fig.savefig(path_test_plots / "Grid_inflow-materials_world.svg")
# fig.savefig(path_test_plots / "Grid_inflow-materials_world_1971.pdf")
fig.savefig(path_test_plots / "Grid_inflow-materials_world_1971.png")
# fig.savefig(path_test_plots / "Grid_inflow-materials_world_1971.svg")
plt.show()


###########################################################################################################
#%% Outflow Materials
###########################################################################################################

#%%%% 2 model ---------------------------------------------------

data_lines  = main_model_factory_lines.outflow_by_cohort_materials.to_array().sum(dim="Region").pint.to("t")  # Convert kg -> tonnes
data_add    = main_model_factory_add.outflow_by_cohort_materials.to_array().sum(dim="Region").pint.to("t")  # Convert kg -> tonnes

data_lines  = data_lines.sel(time=slice(1971, None))
data_add    = data_add.sel(time=slice(1971, None))
data        = xr.concat([data_lines, data_add], dim='Type')
data_sum    = data.sum(dim='Type')

lines_sum           = data_lines.sel(Type=[t for t in data_lines.Type.values if 'Lines' in t]).sum(dim='Type') # Get group sums by keyword and sum over types (sum over HV, MV and LV (and overground/underground for lines))
transformers_sum    = data_add.sel(Type=[t for t in data_add.Type.values if 'Transformers' in t]).sum(dim='Type')
substations_sum     = data_add.sel(Type=[t for t in data_add.Type.values if 'Substations' in t]).sum(dim='Type')


fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12, 10))
linewidth = 2
s_legend = 12
s_label = 14

for i, mat in enumerate(materials):
    lines_sum.sel(material=mat).plot(ax=axes[i], color = DICT_GRID_COLORS['Lines'], label="Lines")
    transformers_sum.sel(material=mat).plot(ax=axes[i], color = DICT_GRID_COLORS['Transformers'], label="Transformers")
    substations_sum.sel(material=mat).plot(ax=axes[i], color = DICT_GRID_COLORS['Substations'], label="Substations")
    data_sum.sel(material=mat).plot(ax=axes[i], label="Total", color='red', alpha=0.8, linestyle='--', linewidth=3)
    
    axes[i].grid(alpha=0.3, linestyle='--')
    axes[i].ticklabel_format(style='sci', axis='y', scilimits=(0, 0)) # Scientific notation for y-axis
    axes[i].tick_params(axis='both', which='major', labelsize=s_legend) # set font size of axis ticks
    axes[i].set_title(f"{mat}")
    axes[i].set_xlabel(" ")
    axes[i].set_ylabel("Outflow [t]", fontsize=s_label)
    axes[i].legend()

axes[-1].set_xlabel("Time", fontsize=s_label)

plt.suptitle(f"{scen_folder}: Electricity Grid Outflow Materials", fontsize=16)
plt.tight_layout()
# fig.savefig(path_test_plots / "Grid_inflow-materials_world.svg")
# fig.savefig(path_test_plots / "Grid_inflow-materials_world_1971.pdf")
fig.savefig(path_test_plots / "Grid_outflow-materials_world_1971.png")
# fig.savefig(path_test_plots / "Grid_inflow-materials_world_1971.svg")
plt.show()



###########################################################################################################
#%% Visualize STOCK Materials - World per type Lines, Trans., Subst.

materials = ["Steel", "Concrete", "Aluminium", "Cu"]

da_x = main_model_factory.stock_by_cohort_materials.sum('Region')
da_x_sum = main_model_factory.stock_by_cohort_materials.sum('Region').sum(dim='Type')

lines_sum = da_x.sel(Type=[t for t in da_x.Type.values if 'Lines' in t]).sum(dim='Type') # Get group sums by keyword and sum over types (sum over HV, MV and LV (and overground/underground for lines))
transformers_sum = da_x.sel(Type=[t for t in da_x.Type.values if 'Transformers' in t]).sum(dim='Type')
substations_sum = da_x.sel(Type=[t for t in da_x.Type.values if 'Substations' in t]).sum(dim='Type')


fig, axes = plt.subplots(nrows=len(materials), ncols=1, figsize=(12, 10), sharex=True)

for i, mat in enumerate(materials):
    lines_sum.sel(material=mat).plot(ax=axes[i], label="Lines")
    transformers_sum.sel(material=mat).plot(ax=axes[i], label="Transformers")
    substations_sum.sel(material=mat).plot(ax=axes[i], label="Substations")
    da_x_sum.sel(material=mat).plot(ax=axes[i], label="Total", color='red', alpha=0.8, linestyle='--')
    
    axes[i].set_title(f"{mat}")
    axes[i].set_xlabel(" ")
    axes[i].set_ylabel("Stock [kg]")
    axes[i].legend()
    axes[i].grid(alpha=0.3, linestyle='--')

axes[-1].set_xlabel("Time")

plt.suptitle("Electricity Grid Stocks", fontsize=16)
plt.tight_layout()
fig.savefig(path_test_plots / "Grid_stocks-bulkmaterials_per-type_world.pdf", dpi=300)
plt.show()

#-------------

materials = ["Aluminium"]

da_x = main_model_factory.stock_by_cohort_materials.sum('Region')
da_x_sum = main_model_factory.stock_by_cohort_materials.sum('Region').sum(dim='Type')

lines_sum = da_x.sel(Type=[t for t in da_x.Type.values if 'Lines' in t]).sum(dim='Type') # Get group sums by keyword and sum over types (sum over HV, MV and LV (and overground/underground for lines))
transformers_sum = da_x.sel(Type=[t for t in da_x.Type.values if 'Transformers' in t]).sum(dim='Type')
substations_sum = da_x.sel(Type=[t for t in da_x.Type.values if 'Substations' in t]).sum(dim='Type')


fig, axes = plt.subplots(nrows=len(materials), ncols=1, figsize=(8, 6), sharex=True)

for i, mat in enumerate(materials):
    lines_sum.sel(material=mat).plot(ax=axes, label="Lines")
    transformers_sum.sel(material=mat).plot(ax=axes, label="Transformers")
    substations_sum.sel(material=mat).plot(ax=axes, label="Substations")
    da_x_sum.sel(material=mat).plot(ax=axes, label="Total", color='red', alpha=0.8, linestyle='--')
    
    axes.set_title(f"{mat}")
    axes.set_xlabel(" ")
    axes.set_ylabel("Stock [kg]")
    axes.legend()
    axes.grid(alpha=0.3, linestyle='--')

axes.set_xlabel("Time")
axes.axvline(x=2050, color='grey', linestyle='--', alpha=0.5)

plt.suptitle("Electricity Grid Stocks", fontsize=16)
plt.tight_layout()
fig.savefig(path_test_plots / "Grid_stocks-Alu_per-type_world.pdf", dpi=300)
fig.savefig(path_test_plots / "Grid_stocks-Alu_per-type_world.png", dpi=300)
plt.show()


#------------

materials = ["Aluminium"]

fig, axes = plt.subplots(nrows=len(materials), ncols=1, figsize=(8, 4), sharex=True)

for i, mat in enumerate(materials):
    lines_sum.sel(material=mat).plot(ax=axes, label="Lines")
    transformers_sum.sel(material=mat).plot(ax=axes, label="Transformers")
    substations_sum.sel(material=mat).plot(ax=axes, label="Substations")
    da_x_sum.sel(material=mat).plot(ax=axes, label="Total", color='red', alpha=0.8, linestyle='--')
    
    axes.set_title(f"{mat}")
    axes.set_xlabel(" ")
    axes.set_ylabel("Stock [kg]")
    axes.legend()
    axes.grid(alpha=0.3, linestyle='--')

axes.set_xlabel("Time")
axes.set_xlim(1990, 2050)

plt.suptitle("Electricity Grid Stocks", fontsize=16)
plt.tight_layout()
fig.savefig(path_test_plots / "Grid_stocks-Alu_per-type_world_1990-2050.png",dpi=300)
plt.show()

###########################################################################################################
#%%%%

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
    for mat in types_level1:
        da_x.sel(material=mat, Region=region).plot(ax=axes[0, i], label=mat)
    axes[0, i].set_title(f"{region}")
    axes[0, i].set_xlabel("Time")
    axes[0, i].legend()

    # Middle row: 
    for mat in types_level2:
        da_x.sel(material=mat, Region=region).plot(ax=axes[1, i], label=mat)
    axes[1, i].set_title(f"{region}")
    axes[1, i].set_xlabel("Time")
    axes[1, i].legend(loc ='upper left')

    # Bottom row: 
    for mat in types_level3:
        da_x.sel(material=mat, Region=region).plot(ax=axes[2, i], label=mat)

    axes[2, i].set_title(f"{region}")
    axes[2, i].set_xlabel("Time")
    axes[2, i].legend(loc ='upper left')

# Label only left side with y-axis label
axes[0, 0].set_ylabel("Value")
axes[1, 0].set_ylabel("Value")
axes[2, 0].set_ylabel("Value")

plt.suptitle("Generation - Inflow Materials", fontsize=16)
plt.tight_layout()
fig.savefig(path_test_plots / "Gen_inflow-materials_Brazil-CEurope.png", dpi=300)
plt.show()




# # HV inflow/outflow (in kgs)
# Hv_lines_above_in, Hv_lines_above_out, Hv_lines_above_stock = inflow_outflow(grid_length_Hv_above_new, lifetime_grid_elements['lines'],        materials_grid_interpol.loc[idx[:,'HV overhead'],:].droplevel(1))    
# Hv_lines_under_in, Hv_lines_under_out, Hv_lines_under_stock = inflow_outflow(grid_length_Hv_under_new, lifetime_grid_elements['lines'],        materials_grid_interpol.loc[idx[:,'HV underground'],:].droplevel(1)) 
# Hv_subst_in, Hv_subst_out, Hv_subst_stock                   = inflow_outflow(grid_subst_Hv_new,        lifetime_grid_elements['substations'],  materials_grid_additions_interpol.loc[idx[:,'Hv Substation'],:].droplevel(1))              
# Hv_trans_in, Hv_trans_out, Hv_trans_stock                   = inflow_outflow(grid_trans_Hv_new,        lifetime_grid_elements['transformers'], materials_grid_additions_interpol.loc[idx[:,'Hv Transformer'],:].droplevel(1))            

# #MV inflow/outflow (in kgs)
# Mv_lines_above_in, Mv_lines_above_out, Mv_lines_above_stock = inflow_outflow(grid_length_Mv_above_new, lifetime_grid_elements['lines'],        materials_grid_interpol.loc[idx[:,'MV overhead'],:].droplevel(1))  # For now, the same lifetime assumptions for HV elements are applied to MV & LV
# Mv_lines_under_in, Mv_lines_under_out, Mv_lines_under_stock = inflow_outflow(grid_length_Mv_under_new, lifetime_grid_elements['lines'],        materials_grid_interpol.loc[idx[:,'MV underground'],:].droplevel(1))
# Mv_subst_in, Mv_subst_out, Mv_subst_stock                   = inflow_outflow(grid_subst_Mv_new,        lifetime_grid_elements['substations'],  materials_grid_additions_interpol.loc[idx[:,'Mv Substation'],:].droplevel(1))
# Mv_trans_in, Mv_trans_out, Mv_trans_stock                   = inflow_outflow(grid_trans_Mv_new,        lifetime_grid_elements['transformers'], materials_grid_additions_interpol.loc[idx[:,'Mv Transformer'],:].droplevel(1))

# #LV inflow/outflow (in kgs)
# Lv_lines_above_in, Lv_lines_above_out, Lv_lines_above_stock = inflow_outflow(grid_length_Lv_above_new, lifetime_grid_elements['lines'],        materials_grid_interpol.loc[idx[:,'LV overhead'],:].droplevel(1))
# Lv_lines_under_in, Lv_lines_under_out, Lv_lines_under_stock = inflow_outflow(grid_length_Lv_under_new, lifetime_grid_elements['lines'],        materials_grid_interpol.loc[idx[:,'LV underground'],:].droplevel(1))
# Lv_subst_in, Lv_subst_out, Lv_subst_stock                   = inflow_outflow(grid_subst_Lv_new,        lifetime_grid_elements['substations'],  materials_grid_additions_interpol.loc[idx[:,'Lv Substation'],:].droplevel(1))
# Lv_trans_in, Lv_trans_out, Lv_trans_stock                   = inflow_outflow(grid_trans_Lv_new,        lifetime_grid_elements['transformers'], materials_grid_additions_interpol.loc[idx[:,'Lv Transformer'],:].droplevel(1))










# %%
